import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from torch.distributions.normal import Normal
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from UniMedCLIP.src.open_clip import create_model_and_transforms, get_mean_std
from UniMedCLIP.src.open_clip import HFTokenizer
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c d h w -> b (d h w) c')

def to_5d(x, d, h, w):
    return rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm3D(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3D, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, C, D, H, W]
        """
        d, h, w = x.shape[-3:]
        return to_5d(self.body(to_3d(x)), d, h, w)

class CrossAttention3D(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CrossAttention3D, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x_q, x_kv):
        b, c, d, h, w = x_q.shape
        
        x_q_flat = to_3d(x_q)  
        x_kv_flat = to_3d(x_kv)  
        
        B, N1, C = x_q_flat.size()
        N2 = x_kv_flat.size(1)

      
        q = self.q(x_q_flat).view(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_kv_flat).view(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(x_kv_flat).view(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

    
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
     
        try:
            self.last_attn = attn.detach().cpu()
        except Exception:
            self.last_attn = None
        
      
        out = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        out = self.proj(out)
    
        out = to_5d(out, d, h, w)  
        return out

class FFN3D(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dropout_rate=0.1):
        super(FFN3D, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.dropout(x) 
        x = self.project_out(x)
        return x

class CrossTransformer3D(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, cross_residual=True, dropout_rate=0.1):
        super(CrossTransformer3D, self).__init__()

        self.norm11 = LayerNorm3D(dim, LayerNorm_type)
        self.norm12 = LayerNorm3D(dim, LayerNorm_type)
        self.attn = CrossAttention3D(dim, num_heads, bias)
        self.norm2 = LayerNorm3D(dim, LayerNorm_type)
        self.ffn = FFN3D(dim, ffn_expansion_factor, bias)
    
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)

        self.cross_residual = cross_residual

    def forward(self, x_q, x_kv):
        if self.cross_residual:
            x_attn = x_q + self.dropout_attn(self.attn(self.norm11(x_q), self.norm12(x_kv)))
        else:
            x_attn = self.dropout_attn(self.attn(self.norm11(x_q), self.norm12(x_kv)))
            
        y = x_attn + self.dropout_ffn(self.ffn(self.norm2(x_attn)))
        return y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
       
        return x
    
class Text_Prompt(nn.Module):
    def __init__(self, exclude_task):
        super(Text_Prompt,self).__init__()

        model_name = 'ViT-B-16-quickgelu' 
        pretrained_weights = "./unimed_clip_vit_b16.pt" 
        text_encoder_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" 
        mean, std = get_mean_std()
        device='cuda'

   
        self.clip, _, _ = create_model_and_transforms(
            model_name,
            pretrained_weights,
            precision='amp',
            device=device,
            force_quick_gelu=True,
            mean=mean, std=std,
            inmem=True,
            text_encoder_name=text_encoder_name,)

        self.tokenizer = HFTokenizer(
            text_encoder_name,
            context_length=256,
            **{},)
        
        self.task_text_prompts = {
            'Brain': ['this is a MRI photo of a brain.'],
            'Hip': ['this is a CT photo of a hip.'],
            'Heart': ['this is a MRI photo of a cardiac.'],
            'ABDO': ['this is a CT photo of an abdominal.'],
            'Haima': ['this is a MRI photo of a hippocampus.'],
            'OAI': ['this is a MRI photo of a knee.']
        }

       
        self.organ_text_features = {}
        
        for organ_name, organ_prompts in self.task_text_prompts.items():
            organ_text_inputs = torch.cat([self.tokenizer(prompt) for prompt in organ_prompts])
            with torch.no_grad():
                organ_text_features = self.clip.encode_text(organ_text_inputs.to(device)).float().detach().requires_grad_(False)
                self.organ_text_features[organ_name] = organ_text_features

        self.mapped_clip_prompt_names = []
        self.mapped_clip_prompts = []
        for task_name in self.organ_text_features.keys():
            if task_name != exclude_task:
                clip_prompt = self.organ_text_features[task_name]  
                self.mapped_clip_prompts.append(clip_prompt)
                self.mapped_clip_prompt_names.append(task_name)
        self.mapped_clip_prompts_tensor = torch.cat(self.mapped_clip_prompts, dim=0) 

    def forward(self):
        return self.mapped_clip_prompts_tensor

class VTP(nn.Module):
    def __init__(self, 
                task_classes = 5, 
                prompt_size = 64, 
                prompt_dim = 96,
                out_dim = 96,
                dropout_rate = 0.
                ):
        super(VTP,self).__init__() 
      
        self.task_classes = task_classes
        self.prompt_size = prompt_size
        self.prompt_dim = prompt_dim
        self.dropout_rate = dropout_rate

        self.text_linear = nn.Linear(512, prompt_dim)
        self.visual_prompt = nn.Parameter(torch.randn(task_classes, prompt_dim, prompt_size, prompt_size, prompt_size))

        nn.init.kaiming_normal_(self.visual_prompt, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        
        self.clip_linear = nn.Linear(512, prompt_dim)

        self.visual_proj = ConvBlock(prompt_dim*2, prompt_dim, dropout_rate=dropout_rate)
        self.text_proj = nn.Sequential(
            nn.Linear(prompt_dim*2, prompt_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.cross_transformer = CrossTransformer3D(dim = prompt_dim, num_heads = 2, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias', dropout_rate=dropout_rate)
        
        self.conv_last = nn.Conv3d(prompt_dim, 
                                 prompt_dim, 
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out = nn.Conv3d(prompt_dim*2, 
                                 out_dim, 
                                 kernel_size=3, stride=1, padding=1, bias=False)
        
  
        
        self.text_fuse = ConvBlock(prompt_dim*2, prompt_dim, dropout_rate=dropout_rate)
        self.visual_fuse = ConvBlock(prompt_dim*2, prompt_dim, dropout_rate=dropout_rate)
        self.final_fuse = ConvBlock(prompt_dim*2, prompt_dim, dropout_rate=dropout_rate)

        self.fuse=ConvBlock(prompt_dim*2, prompt_dim, dropout_rate=dropout_rate)

        self.all_task_names=['Brain','Hip','Heart','ABDO','Haima','OAI']

    

    def forward(self, x, clip_prompts, weights, task_name=None, is_train=True, task_names=None, return_intermediate=False):
        B,C,H,W,D = x.shape
        
        visual_weights = weights.view(B, self.task_classes, 1, 1, 1, 1)
        visual_prompt_weighted = (visual_weights * self.visual_prompt.unsqueeze(0)).sum(dim=1) 
        text_weights = weights.view(B, self.task_classes)
        text_prompt_weighted = torch.matmul(text_weights, clip_prompts)  
        text_prompt_weighted = self.clip_linear(text_prompt_weighted)
        text_prompt_weighted = text_prompt_weighted.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  
        text_prompt_weighted = self.cross_transformer(text_prompt_weighted, visual_prompt_weighted)
        text_prompt_weighted = F.interpolate(text_prompt_weighted, size=(self.prompt_size, self.prompt_size, self.prompt_size))
        prompt_output = self.final_fuse(torch.cat([text_prompt_weighted, visual_prompt_weighted], dim=1))
        final_output=self.fuse(torch.cat([prompt_output,x],dim=1))
    
        return final_output