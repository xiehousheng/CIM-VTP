import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import logging
from datetime import datetime
import shutil
import argparse
import random
from cimvtp import CIMVTP
from multi_task_dataset import MultiTaskRegistrationDataset
from dataset import (
    AbdominalDataset,
    BrainDataset,
    KneeDataset,
    CardiacDataset,
    HippocampusDataset,
    HipDataset,
)

def jacobian_det(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (dy[:, :, :, :, 1] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 1])
    Jdet1 = dx[:, :, :, :, 1] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 2] - dy[:, :, :, :, 2] * dz[:, :, :, :, 0])
    Jdet2 = dx[:, :, :, :, 2] * (dy[:, :, :, :, 0] * dz[:, :, :, :, 1] - dy[:, :, :, :, 1] * dz[:, :, :, :, 0])

    Jdet = Jdet0 - Jdet1 + Jdet2
    return Jdet


def neg_Jdet_loss_sigmoid(y_pred, sample_grid):
    y_pred = y_pred.permute(0, 2, 3, 4, 1)
    Jdet = jacobian_det(y_pred, sample_grid)
    selected_pos_Jdet = F.relu(Jdet) * 1000
    selected_pos_Jdet_num = (torch.sigmoid(selected_pos_Jdet) - 0.5) * 2
    return 1 - torch.mean(selected_pos_Jdet_num)

def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Grad3d(torch.nn.Module):
    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)

def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None):
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False
        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(target).float()
    intersect = (input * target).sum(-1)
    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)

class DiceLong(nn.Module):
    def __init__(self, epsilon=1e-5, skip_bg=True, num_clus=36):
        super(DiceLong, self).__init__()
        self.epsilon = epsilon
        self.num_clus = num_clus
        self.skip_bg = skip_bg

    def forward(self, input, target):
        input = nn.functional.one_hot(input, num_classes=self.num_clus)
        input = torch.squeeze(input, 1)
        input = input.permute(0, 4, 1, 2, 3).contiguous()
        target = nn.functional.one_hot(target, num_classes=self.num_clus)
        target = torch.squeeze(target, 1)
        target = target.permute(0, 4, 1, 2, 3).contiguous()

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon)
        if self.skip_bg:
            return per_channel_dice[1:]
        else:
            return torch.mean(per_channel_dice)

class MIND_loss(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, device, win=None):
        super(MIND_loss, self).__init__()
        self.win = win
        self.device = device

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
        kernel_size = radius * 2 + 1

        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                        [1, 1, 0],
                                        [1, 0, 1],
                                        [1, 1, 2],
                                        [2, 1, 1],
                                        [1, 2, 1]]).long()

        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(self.device)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(self.device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = torch.nn.ReplicationPad3d(dilation)
        rpad2 = torch.nn.ReplicationPad3d(radius)

        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
            kernel_size, stride=1)

        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind = mind / mind_var
        mind = torch.exp(-mind)
        
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
        
        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)

def setup_logger(log_dir):
    """Set up logging to both file and console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def check_storage(directory, min_gb=1):
    total, used, free = shutil.disk_usage(directory)
    free_gb = free // (2**30)
    return free_gb >= min_gb


def save_checkpoint(model, optimizer, step, save_dir):
    state = {
        'model': model.state_dict(),
        'step': step,
        'opt': optimizer.state_dict()
    }
    save_path = os.path.join(save_dir, f'iter_{step}.pth')
    if check_storage(os.path.dirname(os.path.abspath(save_path)), 1):
        torch.save(state, save_path)
        print(f"Checkpoint saved: {save_path}")
    else:
        print("*** Not enough disk space to save the model ***")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for multi-task registration')
    parser.add_argument('--exclude-tasks', nargs='+', default=[''],
                      help='Tasks to exclude from training, e.g., --exclude-tasks Cardiac Brain')
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU device ID to use')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for training')
    parser.add_argument('--max-iterations', type=int, default=100000,
                      help='Maximum number of training iterations')
    parser.add_argument('--save-interval', type=int, default=2000,
                      help='Save checkpoint every N iterations')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--data-root', type=str, default='/data2/data/xr_project/XHS_MAE/MICCAI/dataset',
                      help='Root directory of the dataset')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                      help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                      help='Directory to save training logs')
    parser.add_argument('--pretrained-path', type=str, default='',
                      help='Path to pretrained checkpoint to load (if empty, train from scratch)')
    return parser.parse_args()
def main():
    args = parse_args()
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")

    data_root = args.data_root
    target_size = (160, 160, 160)
    batch_size = args.batch_size
    initial_lr = args.lr
    log_dir = args.log_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    logger = setup_logger(log_dir)
    logger.info(f"Excluding tasks: {args.exclude_tasks}")
    if args.pretrained_path:
        logger.info(f"Loading pretrained weights from: {args.pretrained_path}")

    train_dataset = MultiTaskRegistrationDataset(
        data_root=data_root,
        target_size=target_size,
        split='train',
        exclude_tasks=args.exclude_tasks
    )
    train_loader = train_dataset.get_dataloader(batch_size=batch_size)

    model = CIMVTP(exclude_name='')
    model.to(device)

    if args.pretrained_path:
        print(f"Loading pretrained weights from: {args.pretrained_path}")
        model_data = torch.load(args.pretrained_path, map_location=device, weights_only=False)
        model.load_state_dict(model_data['model'], strict=False)
        print("Pretrained weights loaded successfully.")

    mind_loss = MIND_loss(device)
    grad_loss = Grad3d(penalty='l2')
    grid = generate_grid([160, 160, 160])
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0, amsgrad=True)

    max_iterations = args.max_iterations
    save_interval = args.save_interval

    total_iterations = 0
    start_time = time.time()
    iterator = iter(train_loader)
    pbar = tqdm(total=max_iterations, desc='Training')

    while total_iterations < max_iterations:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        moving = batch['moving'].to(device)
        fixed = batch['fixed'].to(device)
        task_name = batch['task_name']

        warped, flow, text_loss = model(moving, fixed, task_name)

        smooth_loss = grad_loss(flow)
        jetloss = neg_Jdet_loss_sigmoid(flow, grid)
        sim_loss = mind_loss(warped, fixed)
        loss = sim_loss + 0.5 * smooth_loss + 0.01 * text_loss + jetloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_iterations += 1
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'iter': total_iterations})

        if total_iterations % save_interval == 0:
            save_checkpoint(model, optimizer, total_iterations, save_dir)
            elapsed = time.time() - start_time
            logger.info(f"Iter {total_iterations}/{max_iterations} - Loss: {loss.item():.4f} - Time: {elapsed:.2f}s")

    pbar.close()
    elapsed = time.time() - start_time
    logger.info(f"Training completed: {max_iterations} iterations in {elapsed:.2f}s")
    save_checkpoint(model, optimizer, total_iterations, save_dir)


if __name__ == '__main__':
    main()