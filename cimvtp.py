import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as nnf
import math
import os
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from vtp import VTP, Text_Prompt

class RegHead(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.reg_head = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
    def forward(self, x):
        x_out = self.reg_head(x)
        return x_out       

class ResizeTransformer_block(nn.Module):

    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x

class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

        
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x_out = self.act(x)
        return x_out


class DualRegionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        
    def forward(self, x, mask=None):
        if mask is not None:
            inside_region = x * mask
            outside_region = x * (1 - mask)
            inside_output = self.conv1(inside_region)
            outside_output = self.conv1(outside_region)
            x = inside_output * mask + outside_output * (1 - mask)
        else:
            x = self.conv1(x)
            
        x = self.norm1(x, mask)
        x = self.act1(x)
        
        if mask is not None:
            inside_region = x * mask
            outside_region = x * (1 - mask)
            inside_output = self.conv2(inside_region)
            outside_output = self.conv2(outside_region)
            x = inside_output * mask + outside_output * (1 - mask)
        else:
            x = self.conv2(x)

        x = self.norm2(x, mask)
        x_out = self.act2(x)
        return x_out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

      
        
    def forward(self, x, mask=None):
        if mask is not None:
            inside_region = x * mask
            outside_region = x * (1 - mask)
            inside_output = self.conv1(inside_region)
            outside_output = self.conv1(outside_region)
            x = inside_output * mask + outside_output * (1 - mask)
        else:
            x = self.conv1(x)
            
        x = self.norm1(x, mask)
        x = self.act1(x)
        if mask is not None:
            inside_region = x * mask
            outside_region = x * (1 - mask)
            inside_output = self.conv2(inside_region)
            outside_output = self.conv2(outside_region)
            x = inside_output * mask + outside_output * (1 - mask)
        else:
            x = self.conv2(x)
            
        x = self.norm2(x, mask)
        x = self.act2(x)
        
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=1, channel_num=8):
        super().__init__()
        self.conv_1 = EncoderBlock(in_channels, channel_num)
        self.conv_2 = EncoderBlock(channel_num, channel_num * 2)
        self.conv_3 = EncoderBlock(channel_num * 2, channel_num * 4)
        self.conv_4 = EncoderBlock(channel_num * 4, channel_num * 8)
        self.conv_5 = EncoderBlock(channel_num * 8, channel_num * 16)
        self.downsample = nn.AvgPool3d(2, stride=2)

    
    def forward(self, x_in):
        x_1 = self.conv_1(x_in,None)
        x = self.downsample(x_1)
        x_2 = self.conv_2(x,None)
        x = self.downsample(x_2)
        x_3 = self.conv_3(x,None)
        x = self.downsample(x_3)
        x_4 = self.conv_4(x,None)
        x = self.downsample(x_4)
        x_5 = self.conv_5(x,None)

       
        
        
        return  [x_1, x_2, x_3, x_4, x_5]

class Channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Channel_attention, self).__init__()

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(channel, channel // ratio, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel // ratio, channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.gap(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out * x

class Spacial_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Spacial_attention, self).__init__()

        self.conv1 = nn.Conv3d(channel, channel // ratio, kernel_size=1, stride=1, padding=0)
        self.norm = nn.InstanceNorm3d(channel // ratio)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel // ratio, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out * x

def get_winsize(x_size, window_size):
    use_window_size = list(window_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
    return tuple(use_window_size)

def window_partition(x_in, window_size):

    b, d, h, w, c = x_in.shape
    x = x_in.view(b,
                  d // window_size[0],
                  window_size[0],
                  h // window_size[1],
                  window_size[1],
                  w // window_size[2],
                  window_size[2],
                  c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)

    return windows
def window_reverse(windows, window_size, dims):

    b, d, h, w = dims
    x = windows.view(b,
                     d // window_size[0],
                     h // window_size[1],
                     w // window_size[2],
                     window_size[0],
                     window_size[1],
                     window_size[2],
                     -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x
class LocalCorrModule(nn.Module):
    def __init__(self, embed_dim, num_heads=8, window_size=[2, 2, 2]):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.normx = nn.LayerNorm(embed_dim)
        self.normy = nn.LayerNorm(embed_dim)
    
    def forward(self, x_in, y_in):
        b, c, d, h, w = x_in.shape
        d, h, w = x_in.size(2), x_in.size(3), x_in.size(4)
        x_in = x_in.permute(0, 2, 3, 4, 1)
        y_in = y_in.permute(0, 2, 3, 4, 1)
        x = self.normx(x_in)
        y = self.normy(y_in)
        window_size = get_winsize((d, h, w), self.window_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = nnf.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        y = nnf.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]
        x_windows = window_partition(x, window_size)
        y_windows = window_partition(y, window_size)
        b_, n_, c_ = x_windows.shape
        x = x_windows.reshape(b_, n_, self.num_heads, c_ // self.num_heads).permute(0, 2, 1, 3)
        y = y_windows.reshape(b_, n_, self.num_heads, c_ // self.num_heads).permute(0, 2, 1, 3)
        attn = x @ y.transpose(-2, -1)
        attn_windows = attn.reshape(b_, c_, n_)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        corr = window_reverse(attn_windows, window_size, dims)
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            corr = corr[:, :d, :h, :w, :].contiguous()
        corr = corr.view(-1, d, h, w, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()
       
        return corr

class GlobalCorrModule(nn.Module):
    def __init__(self, embed_dim, window_size=[8, 8, 8]):
        super().__init__()
        self.embed_dim = embed_dim
        self.n = window_size[0]*window_size[1]*window_size[2]
        self.window_size = window_size
        self.normx = nn.LayerNorm(embed_dim)
        self.normy = nn.LayerNorm(embed_dim)
        self.channel_attention = Channel_attention(self.n)
        self.spacial_attention = Spacial_attention(self.n)
    def forward(self, x_in, y_in):
        b, c, d, h, w = x_in.shape
        x_in = x_in.permute(0, 2, 3, 4, 1)
        y_in = y_in.permute(0, 2, 3, 4, 1)
        x = self.normx(x_in)
        y = self.normy(y_in)
        x = x.reshape(b, self.n, c)
        y = y.reshape(b, self.n, c)
        corr = x @ y.transpose(-2, -1)
        corr = corr.view(b, self.n, d, h, w).contiguous()
        corr = self.channel_attention(corr)
        corr = self.spacial_attention(corr)
        return corr


class DomainClassifier(nn.Module):
    def __init__(
        self,
        channel_num=16,
        num_classes=5,
    ):
        super().__init__()
        
        self.feature_dims = [channel_num * 1, channel_num * 2, channel_num * 4, channel_num * 8, channel_num * 16]
        
        self.fusion_layers = nn.ModuleList()
        for feature_dim in self.feature_dims:
            fusion_layer = nn.Sequential(
                nn.Conv3d(feature_dim * 2, feature_dim, 3, 1, 1),
                nn.InstanceNorm3d(feature_dim),
                nn.ReLU(inplace=True)
            )
            self.fusion_layers.append(fusion_layer)
        
        self.bottleneck_layers = nn.ModuleList()
        for feature_dim in self.feature_dims:
            bottleneck_layer = nn.Sequential(
                nn.Conv3d(feature_dim, feature_dim, 3, 1, 1),
                nn.InstanceNorm3d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv3d(feature_dim, feature_dim, 3, 1, 1),
                nn.InstanceNorm3d(feature_dim),
                nn.ReLU(inplace=True)
            )
            self.bottleneck_layers.append(bottleneck_layer)
        
        self.downsample_layers = nn.ModuleList()
        for l, feature_dim in enumerate(self.feature_dims):
            if l < len(self.feature_dims) - 1:
                out_feature_dim = self.feature_dims[l + 1]
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.Conv3d(feature_dim, out_feature_dim, 1, bias=False),
                        nn.AvgPool3d(2, 2),
                        nn.ReLU(inplace=True)
                    )
                )
        
        self.mixing_weights = nn.Parameter(
            torch.ones(len(self.feature_dims)), requires_grad=True
        )
        
      
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dims[-1], 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  
            nn.Linear(16, num_classes)
        )
        
    def forward(self, mov_features, fix_features):

        mixing_weights = torch.nn.functional.softmax(self.mixing_weights, dim=0)
        fused_features = []
        for i, (mov_feat, fix_feat) in enumerate(zip(mov_features, fix_features)):
            concat_feat = torch.cat([mov_feat, fix_feat], dim=1)
            fused_feat = self.fusion_layers[i](concat_feat)
            fused_features.append(fused_feat)
        
        x = fused_features[0]
        x = self.bottleneck_layers[0](x)
        
        for i in range(len(self.feature_dims) - 1):
            x = self.downsample_layers[i](x)
            
            if i + 1 < len(fused_features):
                processed_feature = self.bottleneck_layers[i + 1](fused_features[i + 1])
                x = x + mixing_weights[i + 1] * processed_feature
        
        x = x.mean(dim=[-3, -2, -1])  
        out = self.fc(x)
        
        return out

class CIMVTP(nn.Module):
    def __init__(self, channel_num=16,exclude_name=''):
        super().__init__()

        self.encoder = Encoder(channel_num=16)

        self.conv_1 = DualRegionConvBlock(channel_num * 1 * 4, channel_num * 1)
        self.conv_2 = DualRegionConvBlock(channel_num * 2 * 4, channel_num * 2)
        self.conv_3 = DualRegionConvBlock(channel_num * 4 * 4, channel_num * 4)
        self.conv_4 = DualRegionConvBlock(channel_num * 8 * 4, channel_num * 8)
        self.conv_5 = DualRegionConvBlock(channel_num * 16 * 3 , channel_num * 16)
     
        

        self.corr_1 = LocalCorrModule(channel_num * 1, 2)
        self.corr_2 = LocalCorrModule(channel_num * 2, 4)
        self.corr_3 = LocalCorrModule(channel_num * 4, 8)
        self.corr_4 = LocalCorrModule(channel_num * 8, 16)
        self.corr_5 = LocalCorrModule(channel_num * 16,32)
     

        self.upsample_1 = DeconvBlock(channel_num * 2, channel_num * 1)
        self.upsample_2 = DeconvBlock(channel_num * 4, channel_num * 2)
        self.upsample_3 = DeconvBlock(channel_num * 8, channel_num * 4)
        self.upsample_4 = DeconvBlock(channel_num * 16, channel_num * 8)
        self.upsample_5 = DeconvBlock(channel_num * 32, channel_num * 16)

        self.reghead_1 = RegHead(channel_num * 1)
        self.reghead_2 = RegHead(channel_num * 4)
        self.reghead_3 = RegHead(channel_num * 8)
        self.reghead_4 = RegHead(channel_num * 16)
        self.reghead_5 = RegHead(channel_num * 32)
        
        if exclude_name=='':
            class_num=6
        else:
            class_num=5
        self.tp=Text_Prompt(exclude_name)
        
        

        self.vtp_2=VTP(class_num,80,channel_num * 2)
        self.vtp_3=VTP(class_num,40,channel_num * 4)
        self.vtp_4=VTP(class_num,20,channel_num * 8)
        self.vtp_5=VTP(class_num,10,channel_num * 16)

       
        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')

        self.exclude_name=exclude_name
        if exclude_name=='':
            self.classifier=DomainClassifier(num_classes=6)
        else:
            self.classifier=DomainClassifier()

        all_task_names=['Brain','Hip','Heart','ABDO','Haima','OAI']
        self.task_names = [task for task in all_task_names if task != self.exclude_name]
   

 

    def forward(self, moving, fixed, task_name, is_train=True):
        x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5 = self.encoder(moving)
        x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5 = self.encoder(fixed)

        weights = self.classifier([x_mov_1, x_mov_2, x_mov_3, x_mov_4, x_mov_5], [x_fix_1, x_fix_2, x_fix_3, x_fix_4, x_fix_5])
      
        if is_train:
            task_id = torch.tensor([self.task_names.index(task_name)], device=weights.device)
            cls_loss = self.cls_criterion(weights, task_id)
        else:
            cls_loss = torch.tensor(0.0, device=weights.device)

        weights=F.softmax(weights,dim=1)
        organ_prompt = self.tp()
        
        corr_5 = self.corr_5(x_mov_5, x_fix_5) 
        cat = torch.cat([x_mov_5, corr_5, x_fix_5], dim=1)
        conv_corr_5 = self.conv_5(cat,None)
        d_5=conv_corr_5
        prompt_5,stloss5=self.vtp_5(conv_corr_5,organ_prompt,weights)
        flow_5 = self.reghead_5(torch.cat([conv_corr_5,prompt_5],dim=1))
        flow_5 = self.vecint(flow_5) 
       
   
        d_5=self.upsample_4(d_5)
        flow_5_up = self.ResizeTransformer(flow_5)
        x_mov_4 = self.SpatialTransformer(x_mov_4, flow_5_up)
        corr_4 = self.corr_4(x_mov_4, x_fix_4)
        cat = torch.cat([x_mov_4, corr_4, d_5, x_fix_4], dim=1)
        conv_corr_4 = self.conv_4(cat,None)
        d_4=conv_corr_4
        prompt_4,stloss4=self.vtp_4(conv_corr_4,organ_prompt,weights)
        flow_4 = self.reghead_4(torch.cat([conv_corr_4,prompt_4],dim=1))
        flow_4 = self.SpatialTransformer(flow_5_up, flow_4) + flow_4
    
       
        d_4=self.upsample_3(d_4)
        flow_4_up = self.ResizeTransformer(flow_4)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow_4_up)
        corr_3 = self.corr_3(x_mov_3, x_fix_3)
        cat = torch.cat([x_mov_3, corr_3, d_4, x_fix_3], dim=1)
        conv_corr_3 = self.conv_3(cat,None)
        d_3=conv_corr_3
        prompt_3,stloss3=self.vtp_3(conv_corr_3,organ_prompt,weights)
        flow_3 = self.reghead_3(torch.cat([conv_corr_3,prompt_3],dim=1))
        flow_3 = self.SpatialTransformer(flow_4_up, flow_3) + flow_3
   
    
      
        d_3=self.upsample_2(d_3)
        flow_3_up = self.ResizeTransformer(flow_3)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow_3_up)
        corr_2 = self.corr_2(x_mov_2, x_fix_2)
        cat = torch.cat([x_mov_2, corr_2, d_3, x_fix_2], dim=1)
        conv_corr_2 = self.conv_2(cat,None)
        d_2=conv_corr_2
        prompt_2,stloss2=self.vtp_2(conv_corr_2,organ_prompt,weights)
        flow_2 = self.reghead_2(torch.cat([conv_corr_2,prompt_2],dim=1))
        flow_2 = self.SpatialTransformer(flow_3_up, flow_2) + flow_2
       
      
        d_2=self.upsample_1(d_2)
        flow_2_up = self.ResizeTransformer(flow_2)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow_2_up)
        corr_1 = self.corr_1(x_mov_1, x_fix_1)
        cat = torch.cat([x_mov_1, corr_1, d_2, x_fix_1], dim=1)
        conv_corr_1 = self.conv_1(cat,None)
        flow_1 = self.reghead_1(conv_corr_1)
        flow_1 = self.SpatialTransformer(flow_2_up, flow_1) + flow_1
  
        moved = self.SpatialTransformer(moving, flow_1)

      
     
        return moved, flow_1, weights