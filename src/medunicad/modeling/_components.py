"""Low-level Med-UniCAD model components.

This module preserves the checkpoint-compatible implementation used for the
paper. Public construction goes through :mod:`medunicad.modeling.freqmoe_dense`,
which fixes the architecture to the finalized dense FreqXpert configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import math
import os
from pathlib import Path
from torchvision.ops import batched_nms
from medunicad.losses.box_ops import *
from torchvision.ops import roi_align
from einops import rearrange
from transformers import AutoTokenizer, AutoModelForCausalLM, Blip2Config, Blip2QFormerModel
from peft import get_peft_model, LoraConfig, TaskType

# -----------------------------------------------------------------------------
# BUS-BRA morphology descriptor schema (shared with the training data layer)
# -----------------------------------------------------------------------------
MEASUREMENT_KEYS_BUSBRA = (
    "max_diameter_px",
    "perpendicular_diameter_px",
    "bbox_width_px",
    "bbox_height_px",
    "aspect_ratio_h_over_w",
    "area_px",
    "perimeter_px",
    "circularity",
    "solidity",
    "extent",
    "eccentricity",
)
N_MEAS_BUSBRA: int = len(MEASUREMENT_KEYS_BUSBRA)
assert N_MEAS_BUSBRA == 11

# Full source-mask-space morphology supplied to BUS-BRA report generation.
# The first 11 values are the trained measurement schema. The final three
# are fitted-ellipse descriptors that occur in the GT reports but are not
# consumed by the learned MeasurementEncoder.
REPORT_MORPHOLOGY_KEYS_BUSBRA = MEASUREMENT_KEYS_BUSBRA + (
    "major_axis_px",
    "minor_axis_px",
    "major_minor_ratio",
)

# --------------------------------------------------------
# 1. Helper Functions
# --------------------------------------------------------
def to_2tuple(x):
    if isinstance(x, (list, tuple)): return x
    return tuple([x] * 2)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x): return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# --------------------------------------------------------
# 2. Swin Transformer Core Components
# --------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.register_buffer("attn_mask", None)
        self._update_mask(self.input_resolution)

    def _update_mask(self, resolution):
        H, W = resolution
        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.attn_mask = attn_mask

    def forward(self, x, resolution):
        H, W = resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            if self.attn_mask is None or self.attn_mask.shape[1] != (H // self.window_size * W // self.window_size):
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                attn_mask = None 
            else:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                attn_mask = self.attn_mask.to(x.device)
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x, resolution):
        H, W = resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    
    def forward(self, x, resolution):
        H, W = resolution
        for blk in self.blocks:
            if self.use_checkpoint: x = checkpoint.checkpoint(blk, x, (H, W))
            else: x = blk(x, (H, W))
        x_out = x 
        if self.downsample is not None: 
            x = self.downsample(x, (H, W))
            return x, x_out, (H//2, W//2)
        else:
            return x, x_out, (H, W)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None: self.norm = norm_layer(embed_dim)
        else: self.norm = None
    def forward(self, x):
        B, C, H, W = x.shape
        H_out, W_out = H // 4, W // 4
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None: x = self.norm(x)
        return x, (H_out, W_out)

# --------------------------------------------------------
# 3. Swin-Unet Decoder Components
# --------------------------------------------------------

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x, resolution):
        H, W = resolution
        x = self.expand(x)
        B, L, C = x.shape
        if L != H * W: H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x, (H*2, W*2)

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
    def forward(self, x, resolution):
        H, W = resolution
        x = self.expand(x)
        B, L, C = x.shape
        if L != H * W: H = W = int(math.sqrt(L))
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x, (H*4, W*4)

class BasicLayer_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        if upsample is not None:
            self.upsample = PatchExpand(dim=dim, dim_scale=2, norm_layer=norm_layer)
        else: self.upsample = None

    def forward(self, x, resolution):
        H, W = resolution
        for blk in self.blocks:
            if self.use_checkpoint: x = checkpoint.checkpoint(blk, x, (H, W))
            else: x = blk(x, (H, W))
        if self.upsample is not None:
            x, res_out = self.upsample(x, (H, W))
            return x, res_out
        return x, (H, W)

class SwinUnetDecoderSys(nn.Module):
    def __init__(self, embed_dim=96, num_classes=1, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], window_size=7, img_size=224):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        norm_layer = nn.LayerNorm
        mlp_ratio = 4.
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        decoder_dims = [384, 192, 96]
        decoder_heads = [12, 6, 3]
        decoder_depths = [2, 2, 2] 
        
        start_res = img_size // 32
        curr_res = (start_res, start_res)
        # curr_res = (7, 7) 
        
        # --- FIX: Initialize Bottleneck here (Stop Lazy Init Resume Error) ---
        self.bottleneck_expand = PatchExpand(dim=768, dim_scale=2, norm_layer=nn.LayerNorm)

        for i in range(3):
            concat_linear = nn.Linear(2 * decoder_dims[i], decoder_dims[i])
            self.concat_back_dim.append(concat_linear)
            layer_up = BasicLayer_up(
                dim=decoder_dims[i], input_resolution=curr_res, depth=decoder_depths[i], num_heads=decoder_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., norm_layer=norm_layer, upsample=PatchExpand if i < 2 else None, use_checkpoint=False
            )
            self.layers_up.append(layer_up)
            curr_res = (curr_res[0]*2, curr_res[1]*2)
        self.norm_up = norm_layer(96)
        self.up = FinalPatchExpand_X4(dim=96, dim_scale=4, norm_layer=norm_layer)
        self.output = nn.Conv2d(in_channels=96, out_channels=num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features):
        (c1, res1), (c2, res2), (c3, res3), (c4, res4) = features
        x = c4
        skips = [c3, c2, c1]
        
        # 1. Bottleneck Expand (Now initialized in __init__)
        x, curr_res = self.bottleneck_expand(x, res4)

        # 2. Decoder Layers
        for i, layer_up in enumerate(self.layers_up):
            skip = skips[i]
            x = torch.cat([x, skip], dim=-1)
            x = self.concat_back_dim[i](x) 
            x, curr_res = layer_up(x, curr_res)

        x = self.norm_up(x)
        x, curr_res = self.up(x, curr_res)
        
        B, L, C = x.shape
        H, W = curr_res
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        logits = self.output(x)
        return logits

# --------------------------------------------------------
# 4. Swin Transformer Encoder
# --------------------------------------------------------

class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, 
                 out_indices=(0, 1, 2, 3), **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices 
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(self.patch_embed.patches_resolution[0] // (2 ** i_layer),
                                                 self.patch_embed.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                               mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint, fused_window_process=fused_window_process)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def load_medclip_weights(self, weights_path=None):
        print(f"--- Loading MedCLIP (HuggingFace Format) -> Custom Swin ---")
        if weights_path is None or not os.path.exists(weights_path):
            print(f"[ERROR] File not found: {weights_path}")
            return

        # 1. Load the MedCLIP Checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle if it's nested under 'model' or 'vision_model'
        source_state_dict = checkpoint
        if 'model' in checkpoint: source_state_dict = checkpoint['model']
        
        target_state_dict = self.state_dict()
        new_state_dict = {}
        
        # Helper to find the MedCLIP key that corresponds to our Custom key
        def get_hf_val(hf_key_suffix):
            # MedCLIP weights usually start with "vision_model.model." or just "model."
            prefixes = ["vision_model.model.", "model.", ""]
            for p in prefixes:
                full_key = p + hf_key_suffix
                if full_key in source_state_dict:
                    return source_state_dict[full_key]
            return None

        # --- 1. Map Patch Embeddings ---
        new_state_dict['patch_embed.proj.weight'] = get_hf_val('embeddings.patch_embeddings.projection.weight')
        new_state_dict['patch_embed.proj.bias'] = get_hf_val('embeddings.patch_embeddings.projection.bias')
        new_state_dict['patch_embed.norm.weight'] = get_hf_val('embeddings.norm.weight')
        new_state_dict['patch_embed.norm.bias'] = get_hf_val('embeddings.norm.bias')

        # --- 2. Map Layers ---
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].blocks)):
                # Keys for your model (Target)
                tgt_prefix = f"layers.{i}.blocks.{j}."
                # Keys for MedCLIP (Source)
                src_prefix = f"encoder.layers.{i}.blocks.{j}."

                # Norms
                new_state_dict[tgt_prefix + "norm1.weight"] = get_hf_val(src_prefix + "layernorm_before.weight")
                new_state_dict[tgt_prefix + "norm1.bias"]   = get_hf_val(src_prefix + "layernorm_before.bias")
                new_state_dict[tgt_prefix + "norm2.weight"] = get_hf_val(src_prefix + "layernorm_after.weight")
                new_state_dict[tgt_prefix + "norm2.bias"]   = get_hf_val(src_prefix + "layernorm_after.bias")

                # MLP
                new_state_dict[tgt_prefix + "mlp.fc1.weight"] = get_hf_val(src_prefix + "intermediate.dense.weight")
                new_state_dict[tgt_prefix + "mlp.fc1.bias"]   = get_hf_val(src_prefix + "intermediate.dense.bias")
                new_state_dict[tgt_prefix + "mlp.fc2.weight"] = get_hf_val(src_prefix + "output.dense.weight")
                new_state_dict[tgt_prefix + "mlp.fc2.bias"]   = get_hf_val(src_prefix + "output.dense.bias")

                # Attention: QKV Fusion (CRITICAL STEP)
                # HF stores Q, K, V separately. Custom Swin stores them as one 'qkv' tensor.
                q_w = get_hf_val(src_prefix + "attention.self.query.weight")
                k_w = get_hf_val(src_prefix + "attention.self.key.weight")
                v_w = get_hf_val(src_prefix + "attention.self.value.weight")
                
                if q_w is not None and k_w is not None and v_w is not None:
                    # Concatenate them to create qkv.weight
                    new_state_dict[tgt_prefix + "attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)

                q_b = get_hf_val(src_prefix + "attention.self.query.bias")
                k_b = get_hf_val(src_prefix + "attention.self.key.bias")
                v_b = get_hf_val(src_prefix + "attention.self.value.bias")
                
                if q_b is not None and k_b is not None and v_b is not None:
                    new_state_dict[tgt_prefix + "attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)

                # Attention: Proj & Pos Bias
                new_state_dict[tgt_prefix + "attn.proj.weight"] = get_hf_val(src_prefix + "attention.output.dense.weight")
                new_state_dict[tgt_prefix + "attn.proj.bias"]   = get_hf_val(src_prefix + "attention.output.dense.bias")
                new_state_dict[tgt_prefix + "attn.relative_position_bias_table"] = get_hf_val(src_prefix + "attention.self.relative_position_bias_table")
                new_state_dict[tgt_prefix + "attn.relative_position_index"] = get_hf_val(src_prefix + "attention.self.relative_position_index")

            # --- 3. Map Downsamples ---
            if self.layers[i].downsample is not None:
                tgt_ds = f"layers.{i}.downsample."
                src_ds = f"encoder.layers.{i}.downsample."
                
                new_state_dict[tgt_ds + "reduction.weight"] = get_hf_val(src_ds + "reduction.weight")
                new_state_dict[tgt_ds + "norm.weight"]      = get_hf_val(src_ds + "norm.weight")
                new_state_dict[tgt_ds + "norm.bias"]        = get_hf_val(src_ds + "norm.bias")

        # --- 4. Final Norm ---
        new_state_dict['norm.weight'] = get_hf_val('layernorm.weight')
        new_state_dict['norm.bias']   = get_hf_val('layernorm.bias')

        # Filter out None values (layers that failed to map)
        filtered_dict = {k: v for k, v in new_state_dict.items() if v is not None}
        
        # Load
        msg = self.load_state_dict(filtered_dict, strict=False)
        print(f"[SUCCESS] Loaded {len(filtered_dict)} layers. Missing: {len(msg.missing_keys)}")
        
        # Validation
        if len(msg.missing_keys) > 20:
            print(f"[WARNING] High missing keys: {len(msg.missing_keys)}. Check prefixes.")
            # print(f"Sample missing: {msg.missing_keys[:5]}")
        
        # Verify first layer specifically
        if 'patch_embed.proj.weight' in filtered_dict:
            print("Verified: Patch Embedding Loaded.")
        else:
            print("CRITICAL: Patch Embedding FAILED to load.")

        torch.cuda.empty_cache()
        
    def forward_features(self, x):
        x, res = self.patch_embed(x)
        if self.ape: x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        outs = []
        for i, layer in enumerate(self.layers):
            # FIX: CAPTURE CURRENT RESOLUTION
            current_res = res
            x, x_out, res = layer(x, res)
            if i in self.out_indices: outs.append((x_out, current_res))
        return x, outs, res

    def forward(self, x):
        return self.forward_features(x)

    def flops(self): return 0

# --------------------------------------------------------
# 5. Specialized Heads & MultiTask Wrapper
# --------------------------------------------------------

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(1, in_dim, 1, 1)) # Learnable query vector
        self.conv = nn.Conv2d(in_dim, 1, 1) # Spatial attention map calculator
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        # 1. Calculate Attention Map
        attn_logits = self.conv(x).view(B, 1, -1) # [B, 1, H*W]
        weights = self.softmax(attn_logits)       # [B, 1, H*W] normalized
        
        # 2. Apply Weighted Average
        x_flat = x.view(B, C, -1)                 # [B, C, H*W]
        weighted_feat = torch.bmm(x_flat, weights.permute(0, 2, 1)) # [B, C, 1]
        
        return weighted_feat.view(B, C)
    
class QFormerProjector(nn.Module):
    # num_query_tokens: 24 -> 8  (3x fewer; c4 is 7x7=49 keys, 8 queries is enough)
    # num_hidden_layers: 4  -> 2  (2x cheaper cross-attn stack)
    def __init__(self, visual_input_dim, llm_hidden_dim, num_query_tokens=8):
        super().__init__()
        config = Blip2Config()
        config.qformer_config.hidden_size = 768
        config.qformer_config.num_hidden_layers = 2      # was 4
        config.qformer_config.encoder_hidden_size = visual_input_dim
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, config.qformer_config.hidden_size))
        self.query_tokens.data.normal_(mean=0.0, std=config.qformer_config.initializer_range)
        self.proj = nn.Linear(config.qformer_config.hidden_size, llm_hidden_dim)

    def forward(self, visual_features):
        B = visual_features.shape[0]
        query_tokens = self.query_tokens.expand(B, -1, -1)
        outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=visual_features)
        return self.proj(outputs.last_hidden_state)

def _get_lora_target_modules(model_name: str):
    """Return LoRA target module names appropriate for each LLM family."""
    n = model_name.lower()
    if "qwen" in n:
        # Qwen2.5 attention projections
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "smollm" in n or "smol" in n:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "llama" in n or "mistral" in n:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if "gpt2" in n or "gpt-2" in n:
        return ["c_attn", "c_proj"]
    # Fallback: standard attn projections
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


# Registry of known LLM hidden sizes — avoids loading the full model just to read config.
# Falls back to model.config.hidden_size when not found.
_LLM_HIDDEN_SIZES = {
    "Qwen/Qwen2.5-0.5B-Instruct":               896,
    "Qwen/Qwen2.5-1.5B-Instruct":               1536,
    "Qwen/Qwen2.5-3B-Instruct":                 2048,
    "HuggingFaceTB/SmolLM2-135M-Instruct":      576,
    "HuggingFaceTB/SmolLM2-360M-Instruct":      960,
    "HuggingFaceTB/SmolLM2-1.7B-Instruct":      2048,
    "meta-llama/Llama-3.2-1B-Instruct":         2048,
    "meta-llama/Llama-3.2-3B-Instruct":         3072,
    "mistralai/Mistral-7B-Instruct-v0.3":        4096,
    "gpt2":                                       768,
    "gpt2-medium":                               1024,
}


class PretrainedReportDecoder(nn.Module):
    """LLM-agnostic report decoder.

    The visual bridge (visual_proj + ln_visual) adapts its output dimension to
    the frozen Llama backbone used by the paper release.

    Key speed improvements vs. the original:
      * max_len reduced to 128 (reports are typically < 80 tokens)
      * LoRA target modules are chosen per LLM family automatically
      * torch.compile can be applied externally after instantiation
    """

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        visual_dim: int = 2048,          # dimension of incoming visual tokens
        max_len: int = 256,              # raised: BUS-BRA GT reports run 60-160 tokens
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.llm_model_name = llm_model_name
        self.max_len = max_len

        # ---- Tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token     = self.tokenizer.eos_token
            self.tokenizer.pad_token_id  = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # ---- Base LLM in FP16 ----
        self.decoder = AutoModelForCausalLM.from_pretrained(
            llm_model_name, dtype=torch.float16
        )
        self.decoder.to("cuda")

        # ---- LoRA — family-aware target modules ----
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=_get_lora_target_modules(llm_model_name),
        )
        self.decoder = get_peft_model(self.decoder, peft_config)

        # Keep LoRA deltas in FP32 for stable mixed-precision training
        for name, param in self.decoder.named_parameters():
            if "lora" in name:
                param.data = param.data.to(torch.float32)

        # ---- Detect LLM hidden size ----
        embed_dim = _LLM_HIDDEN_SIZES.get(llm_model_name, None)
        if embed_dim is None:
            embed_dim = self.decoder.config.hidden_size
        self.embed_dim = embed_dim

        # ---- Visual bridge (FP32 for numerical stability) ----
        # Maps incoming visual_dim → LLM hidden size
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.embed_dim),
        ).float()
        with torch.no_grad():
            self.visual_proj[-1].weight.data.normal_(mean=0.0, std=0.02)
            self.visual_proj[-1].bias.data.zero_()
        self.ln_visual = nn.LayerNorm(self.embed_dim).float()

    # ------------------------------------------------------------------
    def forward(self, visual_features, input_ids=None, labels=None):
        visual_features = visual_features.to(torch.float32)
        visual_embeds   = self.ln_visual(self.visual_proj(visual_features))
        visual_embeds   = torch.clamp(visual_embeds, min=-10.0, max=10.0)
        visual_embeds   = visual_embeds.to(self.decoder.dtype)

        if input_ids is not None:
            text_embeds   = self.decoder.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

            device = visual_features.device
            B, V_Len, _ = visual_embeds.shape
            visual_mask  = torch.ones((B, V_Len), device=device)
            text_mask    = (input_ids != self.tokenizer.pad_token_id).long()
            attention_mask = torch.cat([visual_mask, text_mask], dim=1)

            visual_labels = torch.full((B, V_Len), -100, dtype=torch.long, device=device)
            final_labels  = torch.cat([visual_labels, labels], dim=1)

            return self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=final_labels,
            )
        return None

    # ------------------------------------------------------------------
    def generate(self, visual_features, prompt_ids, attention_mask=None,
                 do_sample=True, num_beams=1):
        visual_features = visual_features.to(torch.float32)
        visual_embeds   = self.ln_visual(self.visual_proj(visual_features))
        visual_embeds   = visual_embeds.to(torch.float16)

        prompt_embeds = self.decoder.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([visual_embeds, prompt_embeds], dim=1).to(torch.float16)

        if attention_mask is not None:
            visual_mask  = torch.ones(
                visual_embeds.shape[:2],
                device=visual_features.device,
                dtype=attention_mask.dtype,
            )
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        generation_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=self.max_len,
            num_beams=num_beams,
            do_sample=do_sample,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            generation_kwargs.update(temperature=0.4, top_p=0.95)
        return self.decoder.generate(**generation_kwargs)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


class SinePositionEmbedding2D(nn.Module):
    """
    Standard 2D sine-cosine positional encoding for feature maps.
    Input : [B, C, H, W]
    Output: [B, hidden_dim, H, W]
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device

        y_embed = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).repeat(H, 1)

        y_embed = y_embed.unsqueeze(0).repeat(B, 1, 1)
        x_embed = x_embed.unsqueeze(0).repeat(B, 1, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (H - 1 + eps) * self.scale
            x_embed = x_embed / (W - 1 + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)   # [B, H, W, 2*num_pos_feats]
        pos = pos.permute(0, 3, 1, 2).contiguous()
        return pos


class MultiScaleDeformableCrossAttention(nn.Module):
    """
    Pure PyTorch deformable-attention-style approximation using grid_sample.
    Query attends to K sampling points on each feature level.

    Now uses 4D reference boxes:
        reference_boxes: [B, Q, 4] in normalized cxcywh
    so sampling is box-aware, not center-only.
    """
    def __init__(self, hidden_dim=256, num_heads=8, num_levels=3, num_points=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = hidden_dim // num_heads

        self.sampling_offsets = nn.Linear(hidden_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(hidden_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)

        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]

        bias = []
        for h in range(self.num_heads):
            for l in range(self.num_levels):
                for p in range(self.num_points):
                    bias.extend((grid_init[h] * (p + 1)).tolist())
        bias = torch.tensor(bias, dtype=torch.float32)

        with torch.no_grad():
            self.sampling_offsets.bias.copy_(bias)

        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, reference_boxes, multi_level_feats):
        """
        query:            [B, Q, D]
        reference_boxes:  [B, Q, 4] normalized cxcywh
        multi_level_feats: list of [B, D, H, W]
        """
        B, Q, D = query.shape
        Hh = self.num_heads
        Dh = self.head_dim

        offsets = self.sampling_offsets(query).view(
            B, Q, Hh, self.num_levels, self.num_points, 2
        )
        attn_weights = self.attention_weights(query).view(
            B, Q, Hh, self.num_levels * self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1).view(
            B, Q, Hh, self.num_levels, self.num_points
        )

        output = torch.zeros(B, Q, Hh, Dh, device=query.device, dtype=query.dtype)

        # reference_boxes = [cx, cy, w, h]
        ref_xy = reference_boxes[..., :2]                      # [B,Q,2]
        ref_wh = reference_boxes[..., 2:].clamp(min=1e-4)     # [B,Q,2]

        for lvl, feat in enumerate(multi_level_feats):
            Bf, Df, H, W = feat.shape
            assert Bf == B and Df == D

            feat_flat = feat.flatten(2).transpose(1, 2)   # [B, HW, D]
            feat_flat = self.value_proj(feat_flat)
            feat_proj = feat_flat.transpose(1, 2).reshape(B, D, H, W)
            feat_proj = feat_proj.view(B, Hh, Dh, H, W)

            # [B,Q,1,1,2]
            ref_xy_lvl = ref_xy[:, :, None, None, :]
            ref_wh_lvl = ref_wh[:, :, None, None, :]

            # [B,Q,Hh,P,2]
            off = offsets[:, :, :, lvl, :, :]

            # box-aware sampling:
            # offsets are scaled by box width/height so queries attend inside/around the box
            sampling_locations = ref_xy_lvl + (off / float(self.num_points)) * (0.5 * ref_wh_lvl)

            grid_x = sampling_locations[..., 0] * 2.0 - 1.0
            grid_y = sampling_locations[..., 1] * 2.0 - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)

            feat_gs = feat_proj.reshape(B * Hh, Dh, H, W)
            grid_gs = grid.permute(0, 2, 1, 3, 4).reshape(B * Hh, Q * self.num_points, 1, 2)

            sampled = F.grid_sample(
                feat_gs,
                grid_gs,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # [B*Hh, Dh, Q*P, 1]

            sampled = sampled.squeeze(-1).transpose(1, 2)
            sampled = sampled.view(B, Hh, Q, self.num_points, Dh).permute(0, 2, 1, 3, 4)
            # [B,Q,Hh,P,Dh]

            w = attn_weights[:, :, :, lvl, :].unsqueeze(-1)  # [B,Q,Hh,P,1]
            output = output + (sampled * w).sum(dim=3)

        output = output.reshape(B, Q, D)
        output = self.output_proj(output)
        return output


class DINODecoderLayer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_levels=3, num_points=4, ffn_dim=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MultiScaleDeformableCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )

    def forward(self, tgt, reference_points, multi_level_feats, attn_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.cross_attn(tgt, reference_points, multi_level_feats)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class DINOHead(nn.Module):
    def __init__(
        self,
        in_channels=(192, 384, 768),
        hidden_dim=256,
        num_queries=10,
        num_classes=5,
        num_decoder_layers=3,
        num_heads=8,
        num_points=8,
        dropout=0.1,
        dn_number=5,
        dn_box_noise_scale=0.2,
        dn_label_noise_ratio=0.1,
        dn_labelbook_size=32,
        return_query_feats=True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_levels = len(in_channels)
        self.num_decoder_layers = num_decoder_layers
        self.return_query_feats = return_query_feats

        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            ) for c in in_channels
        ])

        self.position_embedding = SinePositionEmbedding2D(num_pos_feats=hidden_dim // 2)
        self.level_embed = nn.Parameter(torch.Tensor(self.num_levels, hidden_dim))

        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.refpoint_embed = nn.Embedding(num_queries, 4)

        self.decoder_layers = nn.ModuleList([
            DINODecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_levels=self.num_levels,
                num_points=num_points,
                ffn_dim=hidden_dim * 4,
                dropout=dropout
            ) for _ in range(num_decoder_layers)
        ])

        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_decoder_layers)
        ])

        self.bbox_heads = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3)
            for _ in range(num_decoder_layers)
        ])

        self._reset_parameters()

        # apply anatomical priors AFTER reset, so they are not overwritten
        with torch.no_grad():
            priors = torch.tensor([
                [0.50, 0.20, 0.25, 0.10],
                [0.50, 0.32, 0.25, 0.10],
                [0.50, 0.44, 0.25, 0.10],
                [0.50, 0.56, 0.25, 0.10],
                [0.50, 0.68, 0.25, 0.10],
                [0.50, 0.80, 0.25, 0.10],
                [0.35, 0.50, 0.20, 0.12],
                [0.65, 0.50, 0.20, 0.12],
                [0.50, 0.10, 0.30, 0.15],
                [0.50, 0.90, 0.30, 0.15],
            ], dtype=torch.float32, device=self.refpoint_embed.weight.device)

            priors = inverse_sigmoid(priors.clamp(1e-4, 1.0 - 1e-4))
            n_prior = min(self.num_queries, priors.shape[0])

            # keep random init for all queries, only overwrite first few with anatomical priors
            self.refpoint_embed.weight[:n_prior].copy_(priors[:n_prior])
        
    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.constant_(proj[0].bias, 0.)

        nn.init.normal_(self.level_embed)
        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.normal_(self.refpoint_embed.weight, std=0.02)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        for cls_head in self.class_heads:
            # cls_head = [LayerNorm, Linear, GELU, Linear]
            nn.init.xavier_uniform_(cls_head[1].weight)
            nn.init.constant_(cls_head[1].bias, 0.)
            nn.init.xavier_uniform_(cls_head[3].weight)
            nn.init.constant_(cls_head[3].bias, bias_value)

        for bbox_head in self.bbox_heads:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.)

    def forward(self, feats, targets=None):
        assert len(feats) == self.num_levels
        B = feats[0].shape[0]
        device = feats[0].device

        proj_feats = []
        for lvl, feat in enumerate(feats):
            x = self.input_proj[lvl](feat)
            pos = self.position_embedding(x) + self.level_embed[lvl].view(1, -1, 1, 1)
            proj_feats.append(x + pos)
        # -------------------------------------------------
        # main matching queries
        # -------------------------------------------------
        query_content = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        ref_boxes = self.refpoint_embed.weight.unsqueeze(0).expand(B, -1, -1).sigmoid()

        # -------------------------------------------------
        # DN queries (training only)
        # -------------------------------------------------
        attn_mask = None
        dn_meta = None
        dn_query_content, dn_ref_boxes, dn_attn_mask, dn_meta = self.prepare_for_dn(
            targets=targets,
            batch_size=B,
            device=device
        )

        num_dn_queries = 0
        if dn_query_content is not None and dn_ref_boxes is not None:
            num_dn_queries = dn_query_content.shape[1]
            query_content = torch.cat([dn_query_content, query_content], dim=1)
            ref_boxes = torch.cat([dn_ref_boxes, ref_boxes], dim=1)
            attn_mask = dn_attn_mask

        tgt = torch.zeros_like(query_content)

        outputs_classes = []
        outputs_boxes = []

        # keep the final decoder hidden state
        hs_last = None

        for lid, layer in enumerate(self.decoder_layers):
            tgt = layer(
                tgt=tgt + query_content,
                reference_points=ref_boxes,
                multi_level_feats=proj_feats,
                attn_mask=attn_mask
            )

            logits = self.class_heads[lid](tgt)
            delta = self.bbox_heads[lid](tgt)
            pred_boxes = (inverse_sigmoid(ref_boxes) + delta).sigmoid()

            outputs_classes.append(logits)
            outputs_boxes.append(pred_boxes)

            hs_last = tgt
            ref_boxes = pred_boxes.detach()

        out = {
            "pred_logits": outputs_classes[-1],
            "pred_boxes": outputs_boxes[-1],
            "aux_outputs": [
                {"pred_logits": c, "pred_boxes": b}
                for c, b in zip(outputs_classes[:-1], outputs_boxes[:-1])
            ],
            "dn_meta": dn_meta
        }

        # -------------------------------------------------
        # NEW: return decoder query features
        # query_feats      : all queries (DN + matching, if DN exists)
        # main_query_feats : only the actual matching queries
        # -------------------------------------------------
        if self.return_query_feats and hs_last is not None:
            out["query_feats"] = hs_last
            out["main_query_feats"] = hs_last[:, num_dn_queries:, :]
            out["num_dn_queries"] = num_dn_queries

        return out
    
    def _apply_label_noise(self, labels):
        """
        labels: [N]
        randomly corrupt some labels for DN training
        """
        if labels.numel() == 0:
            return labels

        noisy = labels.clone()
        if self.dn_label_noise_ratio <= 0:
            return noisy

        noise_mask = torch.rand_like(noisy.float()) < self.dn_label_noise_ratio
        if noise_mask.any():
            random_labels = torch.randint(
                low=0,
                high=self.num_classes,
                size=(noise_mask.sum().item(),),
                device=labels.device,
                dtype=labels.dtype
            )
            noisy[noise_mask] = random_labels
        return noisy
    
    def prepare_for_dn(self, targets, batch_size, device):
        """
        Build denoising queries from GT labels/boxes.
        Returns:
            dn_query_content: [B, Q_dn, D]
            dn_ref_boxes:     [B, Q_dn, 4]
            attn_mask:        [Q_total, Q_total] or None
            dn_meta: dict
        """
        if targets is None or self.dn_number <= 0:
            return None, None, None, None

        max_gt = max((len(t["labels"]) for t in targets), default=0)
        if max_gt == 0:
            return None, None, None, None

        # number of DN groups
        num_groups = self.dn_number
        dn_queries_per_group = max_gt
        total_dn_queries = num_groups * dn_queries_per_group

        dn_query_content = torch.zeros(
            batch_size, total_dn_queries, self.hidden_dim,
            device=device
        )
        dn_ref_boxes = torch.zeros(
            batch_size, total_dn_queries, 4,
            device=device
        )

        dn_pos_idx = []
        dn_gt_idx = []

        for b in range(batch_size):
            labels = targets[b]["labels"]
            boxes = targets[b]["boxes"]

            num_gt = len(labels)
            if num_gt == 0:
                dn_pos_idx.append(torch.empty(0, dtype=torch.long, device=device))
                dn_gt_idx.append(torch.empty(0, dtype=torch.long, device=device))
                continue

            pos_idx_b = []
            gt_idx_b = []

            for g in range(num_groups):
                start = g * dn_queries_per_group
                end = start + num_gt

                noisy_labels = self._apply_label_noise(labels)
                noisy_boxes = boxes.clone()

                if self.dn_box_noise_scale > 0:
                    box_noise = (torch.rand_like(noisy_boxes) * 2.0 - 1.0) * self.dn_box_noise_scale
                    noisy_boxes = (noisy_boxes + box_noise).clamp(0.0, 1.0)

                dn_query_content[b, start:end] = self.label_enc(noisy_labels)
                dn_ref_boxes[b, start:end] = noisy_boxes

                pos_idx_b.append(torch.arange(start, end, device=device))
                gt_idx_b.append(torch.arange(num_gt, device=device))

            dn_pos_idx.append(torch.cat(pos_idx_b) if len(pos_idx_b) > 0 else torch.empty(0, dtype=torch.long, device=device))
            dn_gt_idx.append(torch.cat(gt_idx_b) if len(gt_idx_b) > 0 else torch.empty(0, dtype=torch.long, device=device))

        # attention mask:
        # DN groups should not attend across groups or to matching queries in unsafe ways
        total_queries = total_dn_queries + self.num_queries
        attn_mask = torch.zeros((total_queries, total_queries), dtype=torch.bool, device=device)

        # block attention across DN groups
        for g1 in range(num_groups):
            s1 = g1 * dn_queries_per_group
            e1 = s1 + dn_queries_per_group
            for g2 in range(num_groups):
                if g1 == g2:
                    continue
                s2 = g2 * dn_queries_per_group
                e2 = s2 + dn_queries_per_group
                attn_mask[s1:e1, s2:e2] = True

        dn_meta = {
            "num_dn_queries": total_dn_queries,
            "num_dn_groups": num_groups,
            "dn_queries_per_group": dn_queries_per_group,
            "dn_pos_idx": dn_pos_idx,
            "dn_gt_idx": dn_gt_idx,
        }

        return dn_query_content, dn_ref_boxes, attn_mask, dn_meta
    
    
class FixedGaussianMoFXDecomposition(nn.Module):
    """Article-faithful fixed Gaussian low/mid/high decomposition.

    The raw bands satisfy low + mid + high == x exactly (up to floating-point
    round-off). Expert inputs are independently RMS-normalized in FP32, while
    detached raw band energies are retained for routing.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

        def _gaussian_kernel(size, sigma):
            coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
            g = torch.exp(-(coords.square()) / (2.0 * float(sigma) ** 2))
            g = g / g.sum()
            return torch.outer(g, g).view(1, 1, size, size)

        self.register_buffer(
            "kernel_small", _gaussian_kernel(3, 0.8), persistent=False
        )
        self.register_buffer(
            "kernel_large", _gaussian_kernel(5, 1.4), persistent=False
        )
        self.decomposition_name = "fixed_gaussian_3x3_5x5"

    @staticmethod
    def _depthwise_blur(x, kernel):
        channels = x.shape[1]
        k = kernel.to(device=x.device, dtype=x.dtype).expand(
            channels, 1, -1, -1
        )
        pad = int(kernel.shape[-1] // 2)
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x_pad, k, groups=channels)

    def forward(self, x):
        # Keep the large depthwise Gaussian convolutions in the active AMP
        # dtype. Only RMS/energy statistics use FP32 for numerical stability.
        smooth_small = self._depthwise_blur(x, self.kernel_small)
        low_raw = self._depthwise_blur(x, self.kernel_large)
        mid_raw = smooth_small - low_raw
        high_raw = x - smooth_small
        raw = {"low": low_raw, "mid": mid_raw, "high": high_raw}

        normalized, scales, energies = {}, {}, []
        for name in ("low", "mid", "high"):
            band = raw[name]
            rms_f32 = band.detach().float().square().mean(
                dim=(1, 2, 3), keepdim=True
            ).add(self.eps ** 2).sqrt()
            scales[name] = rms_f32
            normalized[name] = band / rms_f32.to(
                dtype=band.dtype
            ).clamp_min(self.eps)
            energies.append(rms_f32.flatten())

        band_energies = torch.stack(energies, dim=-1).detach()
        return normalized, band_energies, scales

    def summary(self):
        return {
            "name": self.decomposition_name,
            "fixed": True,
            "learnable": False,
            "small_filter": "3x3 Gaussian (sigma=0.8)",
            "large_filter": "5x5 Gaussian (sigma=1.4)",
            "exact_reconstruction": True,
            "rms_normalized_expert_inputs": True,
        }


class ArticleDirectedFreqXpert(nn.Module):
    """Directed source -> target frequency interaction from the manuscript."""

    def __init__(
        self,
        channels,
        attn_dim=96,
        num_heads=4,
        kv_pool_stride=1,
        dropout=0.0,
    ):
        super().__init__()
        if int(attn_dim) % int(num_heads) != 0:
            raise ValueError("attn_dim must be divisible by num_heads")
        self.channels = int(channels)
        self.attn_dim = int(attn_dim)
        self.kv_pool_stride = max(1, int(kv_pool_stride))

        groups = max(1, min(16, self.attn_dim // 8))
        self.query_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.attn_dim, 1, bias=False),
            nn.GroupNorm(groups, self.attn_dim),
        )
        self.key_proj = nn.Conv2d(
            self.channels, self.attn_dim, 1, bias=False
        )
        self.value_proj = nn.Conv2d(
            self.channels, self.attn_dim, 1, bias=False
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.attn_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.attn_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.attn_dim),
            nn.Linear(self.attn_dim, 2 * self.attn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.attn_dim, self.attn_dim),
        )
        self.out_proj = nn.Conv2d(
            self.attn_dim, self.channels, 1, bias=True
        )
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, source_band, target_band):
        batch, _, height, width = target_band.shape
        source_for_kv = source_band
        if self.kv_pool_stride > 1:
            source_for_kv = F.avg_pool2d(
                source_for_kv,
                kernel_size=self.kv_pool_stride,
                stride=self.kv_pool_stride,
            )

        q = self.query_proj(target_band).flatten(2).transpose(1, 2)
        k = self.key_proj(source_for_kv).flatten(2).transpose(1, 2)
        v = self.value_proj(source_for_kv).flatten(2).transpose(1, 2)
        attended, _ = self.cross_attn(q, k, v, need_weights=False)
        hidden = self.norm1(q + attended)
        hidden = hidden + self.ffn(hidden)
        hidden = hidden.transpose(1, 2).reshape(
            batch, self.attn_dim, height, width
        )
        return self.out_proj(hidden)


class FactorizedRouterGate(nn.Module):
    """Small image-conditioned gate used by one task or one modality.

    The final logits are centered across experts so a branch cannot win merely
    by learning a common scalar offset. The three branches therefore remain
    directly comparable before their learned convex combination.
    """

    def __init__(self, input_dim, hidden_dim, num_experts, dropout=0.05):
        super().__init__()
        hidden_dim = max(32, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(num_experts)),
        )
        nn.init.trunc_normal_(self.net[1].weight, std=0.02)
        nn.init.zeros_(self.net[1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        logits = self.net(x)
        return logits - logits.mean(dim=-1, keepdim=True)


class RouterContentAdapter(nn.Module):
    """Lightweight residual adapter for modality-dependent router statistics."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = max(32, int(hidden_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        nn.init.trunc_normal_(self.net[1].weight, std=0.02)
        nn.init.zeros_(self.net[1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.net[-1].bias)
        self.raw_residual_scale = nn.Parameter(
            torch.logit(torch.tensor(0.20, dtype=torch.float32))
        )

    def forward(self, x):
        scale = torch.sigmoid(self.raw_residual_scale)
        return x + scale * self.net(x)


class MoFXDenseBlock(nn.Module):
    """Paper-aligned, genuinely input-conditioned MoFX bank.

    Experts are L->H, M->H, H->M, and H->L, plus the identity/null route.
    Expert specialization is learned only from the downstream task losses and
    the image/content, modality, and task context.  No load-balancing target,
    task-separation objective, entropy floor, or expert-choice penalty is used.

    Collapse resistance is provided by neutral optimization mechanisms rather
    than prescribed routing: equal initialization, input-conditioned dense warm-up,
    normalized cosine routing, symmetric expert dropout, decaying exploration,
    bounded logits, and a true identity/null route.
    """

    DIRECTION_SPECS = (
        ("low", "high", "low_to_high"),
        ("mid", "high", "mid_to_high"),
        ("high", "mid", "high_to_mid"),
        ("high", "low", "high_to_low"),
    )

    def __init__(
        self,
        channels,
        route_names,
        stage_name,
        num_modalities=5,
        attn_dim=96,
        num_heads=4,
        router_dim=128,
        top_k=2,
        routing_mode="dense_soft",
        expert_bank_type="freqxpert",
        temperature=1.5,
        modality_scale=1.0,
        router_strategy="content_dominant_factorized_cosine",
        router_task_residual_scale=0.10,
        router_modality_residual_scale=0.10,
        router_attention_pool=True,
        router_alpha_content_init=0.50,
        router_alpha_task_init=0.25,
        router_alpha_modality_init=0.25,
        router_task_specific_gates=True,
        router_modality_specific_gates=True,
        router_modality_content_adapters=True,
        router_branch_hidden=96,
        router_configuration="full",
        router_mi_min_group_size=4,
        router_mi_queue_size=32,
        router_mi_current_fraction=0.50,
        use_granularity_prior=False,
        gamma_init_by_route=None,
        gamma_max_by_route=None,
        residual_cap_by_route=None,
        expert_scale_init=0.35,
        expert_scale_max=1.0,
        generalist_bottleneck=128,
        include_generalist=False,
        include_null=True,
        decomposition="gaussian_fixed",
        fft_initial_boundaries=(0.25, 0.55),
        fft_mask_temperature=0.08,
        fft_min_bandwidth=0.05,
        kv_pool_stride=1,
        ultrasound_modality_id=None,
        router_noise_std=0.08,
        exploration_epsilon=0.06,
        entropy_floor=0.0,
        router_variation_min=0.0,
        expert_dropout=0.10,
        expert_input_grad_scale=0.15,
    ):
        super().__init__()
        del use_granularity_prior, entropy_floor, router_variation_min
        self.channels = int(channels)
        self.stage_name = str(stage_name)
        self.route_names = tuple(str(name) for name in route_names)
        self.route_to_id = {
            name: index for index, name in enumerate(self.route_names)
        }
        self.active_task_ids = list(self.route_names)
        self.num_modalities = max(1, int(num_modalities))
        requested_top_k = max(1, int(top_k))
        _routing_mode = str(routing_mode).strip().lower().replace("-", "_")
        _routing_aliases = {
            "dense": "dense_soft",
            "soft": "dense_soft",
            "soft_dense": "dense_soft",
        }
        self.routing_mode = _routing_aliases.get(_routing_mode, _routing_mode)
        if self.routing_mode != "dense_soft":
            raise ValueError("MoFX-Dense requires dense_soft routing.")
        _bank_type = str(expert_bank_type).strip().lower().replace("-", "_")
        _bank_aliases = {
            "freq": "freqxpert", "frequency": "freqxpert",
            "mofx": "freqxpert", "freqxperts": "freqxpert",
        }
        self.expert_bank_type = _bank_aliases.get(_bank_type, _bank_type)
        if self.expert_bank_type != "freqxpert":
            raise ValueError("MoFX-Dense requires the directed FreqXpert bank.")
        if bool(include_generalist):
            raise ValueError("MoFX-Dense does not include a generalist expert.")
        if str(decomposition).strip().lower() not in {
            "gaussian_fixed", "gaussian", "fixed_gaussian", "paper_gaussian"
        }:
            raise ValueError("MoFX-Dense requires fixed Gaussian decomposition.")
        self.temperature = max(float(temperature), 1e-3)
        self.modality_scale = float(modality_scale)
        _router_strategy = str(router_strategy).strip().lower().replace("-", "_")
        _router_aliases = {
            "content": "content_dominant_factorized_cosine",
            "content_dominant": "content_dominant_factorized_cosine",
            "cosine": "content_dominant_factorized_cosine",
            "multigate": "factorized_multigate",
            "multi_gate": "factorized_multigate",
            "factorized_multi_gate": "factorized_multigate",
        }
        self.router_strategy = _router_aliases.get(
            _router_strategy, _router_strategy
        )
        if self.router_strategy != "factorized_multigate":
            raise ValueError("MoFX-Dense requires factorized_multigate routing.")
        self.router_task_specific_gates = bool(router_task_specific_gates)
        self.router_modality_specific_gates = bool(
            router_modality_specific_gates
        )
        self.router_modality_content_adapters_enabled = bool(
            router_modality_content_adapters
        )
        self.router_branch_hidden = max(32, int(router_branch_hidden))
        self.router_configuration = "full"
        # Content-dominant factorized routing: image evidence generates the
        # main cosine logits. Task and modality can only add small, bounded
        # residual biases and therefore cannot become lookup-table shortcuts.
        self.router_task_residual_scale = max(
            0.0, min(float(router_task_residual_scale), 0.50)
        )
        self.router_modality_residual_scale = max(
            0.0, min(float(router_modality_residual_scale), 0.50)
        )
        self.router_attention_pool_enabled = bool(router_attention_pool)
        # Context-conditioned MI is computed on clean (pre-noise) router
        # probabilities. A small cross-batch queue is required because the
        # fixed eight-slot medical sampler often contains only 1--3 examples
        # for one stage x task x modality context in a single batch.
        self.router_mi_min_group_size = max(2, int(router_mi_min_group_size))
        self.router_mi_queue_size = max(4, int(router_mi_queue_size))
        self.router_mi_current_fraction = min(
            max(float(router_mi_current_fraction), 0.25), 1.0
        )
        self.base_router_noise_std = max(0.0, float(router_noise_std))
        self.current_router_noise_std = self.base_router_noise_std
        self.base_exploration_epsilon = max(
            0.0, min(float(exploration_epsilon), 0.20)
        )
        self.exploration_epsilon = self.base_exploration_epsilon
        self.base_expert_dropout = max(0.0, min(float(expert_dropout), 0.35))
        self.current_expert_dropout = self.base_expert_dropout
        self.expert_input_grad_scale = max(
            0.0, min(float(expert_input_grad_scale), 1.0)
        )
        self.current_epoch = -1
        self.expert_warmup = False
        self.warmup_end_epoch = None
        # 0=dense soft routing, 1=fully sparse top-k routing.
        # Dense-soft mode keeps alpha at zero for the complete run.
        self.sparse_mix_alpha = (
            0.0 if self.routing_mode == "dense_soft" else 1.0
        )
        self.transition_epochs = 6
        self.prior_scale = 0.0
        self.ultrasound_modality_id = (
            None if ultrasound_modality_id is None
            else int(ultrasound_modality_id)
        )

        self.include_generalist = bool(include_generalist)
        self.include_null = bool(include_null)
        self.include_null_expert = self.include_null
        if self.expert_bank_type == "freqxpert":
            self.direction_specs = tuple(self.DIRECTION_SPECS)
            self.expert_names = [spec[2] for spec in self.direction_specs]
        else:
            # Four unconstrained full-feature experts: same expert count as the
            # four directed FreqXperts, but without frequency interaction.
            self.direction_specs = tuple()
            self.expert_names = [f"standard_{i + 1}" for i in range(4)]

        self.base_expert_count = len(self.expert_names)
        self.generalist_expert_index = None
        self.general_expert_index = None
        if self.include_generalist:
            self.generalist_expert_index = len(self.expert_names)
            self.general_expert_index = self.generalist_expert_index
            self.expert_names.append("generalist")
        self.null_expert_index = None
        if self.include_null:
            self.null_expert_index = len(self.expert_names)
            self.expert_names.append("null")
        self.num_experts = len(self.expert_names)
        if self.num_experts < 1:
            raise ValueError("Dense MoE requires at least one expert")
        self.top_k = min(requested_top_k, self.num_experts)
        self.directed_expert_count = len(self.direction_specs)
        self.generic_expert_start = (
            self.directed_expert_count
            if self.expert_bank_type == "freqxpert" else 0
        )
        self.generic_expert_end = self.base_expert_count + (
            1 if self.include_generalist else 0
        )
        self.require_directional = False
        self.min_directional_weight = 0.0
        self.frequency_mass_target = 0.0
        self.frequency_mass_min = 0.0
        self.sparse_dispatch = False
        self.allows_identity_bypass = True

        alpha_init = torch.tensor([
            max(float(router_alpha_content_init), 1e-6),
            max(float(router_alpha_task_init), 1e-6),
            max(float(router_alpha_modality_init), 1e-6),
        ], dtype=torch.float32)
        alpha_init = alpha_init / alpha_init.sum().clamp_min(1e-6)
        self.raw_router_branch_alpha = nn.Parameter(alpha_init.log())
        self.router_branch_names = ("content", "task", "modality")
        if str(router_configuration).strip().lower() != "full":
            raise ValueError("The final MoFX-Dense router requires all branches.")

        # Diagnostic-only neutral reference. It is never used in a loss.
        self.register_buffer(
            "routing_target",
            torch.full((self.num_experts,), 1.0 / self.num_experts),
            persistent=True,
        )
        # Per-bank queue indexed by route and modality. Queue entries are
        # detached clean routing distributions; they approximate the
        # context marginal across recent batches without retaining graphs.
        self.register_buffer(
            "router_mi_queue",
            torch.zeros(
                len(self.route_names), self.num_modalities,
                self.router_mi_queue_size, self.num_experts,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "router_mi_queue_count",
            torch.zeros(
                len(self.route_names), self.num_modalities, dtype=torch.long
            ),
            persistent=True,
        )
        self.register_buffer(
            "router_mi_queue_ptr",
            torch.zeros(
                len(self.route_names), self.num_modalities, dtype=torch.long
            ),
            persistent=True,
        )

        self.requested_decomposition = "gaussian_fixed"
        self.band_decomposition = FixedGaussianMoFXDecomposition()
        self.decomposition_name = self.band_decomposition.decomposition_name
        self.directed_experts = nn.ModuleList([
            ArticleDirectedFreqXpert(
                channels=self.channels,
                attn_dim=int(attn_dim),
                num_heads=int(num_heads),
                kv_pool_stride=int(kv_pool_stride),
            )
            for _ in self.direction_specs
        ])
        self.standard_experts = nn.ModuleList()
        self.general_expert = None

        content_dim = int(router_dim)
        energy_dim = 32
        spatial_dim = 32
        route_dim = 32
        modality_dim = 16

        # Moment statistics are retained, but a learned spatial attention pool
        # is added so two images from the same task/modality can produce
        # genuinely different router queries.
        self.content_proj = nn.Sequential(
            nn.LayerNorm(self.channels * 4),
            nn.Linear(self.channels * 4, content_dim),
            nn.GELU(),
        )
        attn_hidden = max(32, min(content_dim, 128))
        if self.router_attention_pool_enabled:
            self.router_attention_features = nn.Sequential(
                nn.Conv2d(self.channels, attn_hidden, kernel_size=1, bias=False),
                nn.GELU(),
            )
            self.router_attention_scores = nn.Conv2d(
                attn_hidden, 1, kernel_size=1, bias=True
            )
            nn.init.normal_(
                self.router_attention_scores.weight, mean=0.0, std=0.02
            )
            nn.init.zeros_(self.router_attention_scores.bias)
            self.attention_content_proj = nn.Sequential(
                nn.LayerNorm(self.channels),
                nn.Linear(self.channels, content_dim),
                nn.GELU(),
            )
            self.content_fuse = nn.Sequential(
                nn.LayerNorm(content_dim * 2),
                nn.Linear(content_dim * 2, content_dim),
                nn.GELU(),
            )
        else:
            self.router_attention_features = None
            self.router_attention_scores = None
            self.attention_content_proj = None
            self.content_fuse = None

        # Relative log energy, normalized fractions, spectral entropy and
        # low/high ratio describe the image's frequency composition.
        self.energy_proj = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, energy_dim),
            nn.GELU(),
        )
        self.spatial_proj = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, spatial_dim),
            nn.GELU(),
        )

        # Task/modality information is deliberately factorized away from the
        # image query. It enters only as a small bounded logit residual.
        self.route_embedding = nn.Embedding(len(self.route_names), route_dim)
        self.modality_embedding = nn.Embedding(
            self.num_modalities, modality_dim
        )
        nn.init.normal_(self.route_embedding.weight, mean=0.0, std=0.04)
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.04)
        self.task_expert_bias = nn.Embedding(
            len(self.route_names), self.num_experts
        )
        self.modality_expert_bias = nn.Embedding(
            self.num_modalities, self.num_experts
        )
        nn.init.zeros_(self.task_expert_bias.weight)
        nn.init.zeros_(self.modality_expert_bias.weight)

        image_router_in = content_dim + energy_dim + spatial_dim
        aux_router_in = image_router_in + route_dim + modality_dim
        router_hidden = max(128, int(router_dim))
        self.image_router_encoder = nn.Sequential(
            nn.LayerNorm(image_router_in),
            nn.Linear(image_router_in, router_hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(router_hidden, router_hidden),
            nn.LayerNorm(router_hidden),
        )
        self.expert_keys = nn.Parameter(
            torch.empty(self.num_experts, router_hidden)
        )
        nn.init.orthogonal_(self.expert_keys)
        # softplus(raw)=3 initially. Image logits have a much larger available
        # range than the bounded task/modality residuals (default +/-0.1).
        self.raw_router_logit_scale = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(3.0)))
        )

        # MMoE/TaskExpert-inspired task gates and modality-aware gates.  Every
        # gate is still conditioned on the current image descriptor; task and
        # modality do not reduce to fixed lookup vectors.
        self.modality_content_adapters = nn.ModuleList()
        self.task_gates = nn.ModuleDict()
        self.modality_gates = nn.ModuleList()
        if self.router_strategy == "factorized_multigate":
            if self.router_modality_content_adapters_enabled:
                self.modality_content_adapters = nn.ModuleList([
                    RouterContentAdapter(
                        image_router_in, self.router_branch_hidden
                    ) for _ in range(self.num_modalities)
                ])
            if self.router_task_specific_gates:
                self.task_gates = nn.ModuleDict({
                    self._route_module_key(name): FactorizedRouterGate(
                        image_router_in, self.router_branch_hidden,
                        self.num_experts,
                    ) for name in self.route_names
                })
            if self.router_modality_specific_gates:
                self.modality_gates = nn.ModuleList([
                    FactorizedRouterGate(
                        image_router_in, self.router_branch_hidden,
                        self.num_experts,
                    ) for _ in range(self.num_modalities)
                ])

        # Reliability and residual strength may remain task/modality aware;
        # only expert selection is protected from shortcut conditioning.
        router_in = aux_router_in
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(router_in),
            nn.Linear(router_in, max(64, router_hidden // 2)),
            nn.GELU(),
            nn.Linear(max(64, router_hidden // 2), self.num_experts),
        )
        nn.init.zeros_(self.reliability_head[-1].weight)
        # All functional experts start equally reliable.  The null route is an
        # exact identity and therefore has reliability one by definition.
        initial_reliability_values = [0.72] * self.base_expert_count
        if self.include_generalist:
            initial_reliability_values.append(0.72)
        if self.include_null:
            initial_reliability_values.append(0.999)
        initial_reliability = torch.tensor(
            initial_reliability_values, dtype=torch.float32
        )
        with torch.no_grad():
            self.reliability_head[-1].bias.copy_(
                torch.logit(initial_reliability.clamp(1e-4, 1 - 1e-4))
            )

        self.gamma_head = nn.Sequential(
            nn.LayerNorm(router_in),
            nn.Linear(router_in, max(48, router_hidden // 2)),
            nn.GELU(),
            nn.Linear(max(48, router_hidden // 2), 1),
        )
        nn.init.zeros_(self.gamma_head[-1].weight)
        nn.init.zeros_(self.gamma_head[-1].bias)

        route_count = len(self.route_names)
        gamma_max = self._expand_route_values(
            gamma_max_by_route, route_count, default=0.08
        )
        gamma_init = self._expand_route_values(
            gamma_init_by_route, route_count, default=0.012
        )
        residual_cap = self._expand_route_values(
            residual_cap_by_route, route_count, default=0.05
        )
        gamma_init = torch.minimum(
            gamma_init.clamp_min(1e-5),
            (gamma_max - 1e-5).clamp_min(1e-5),
        )
        self.register_buffer(
            "gamma_max_matrix",
            gamma_max.unsqueeze(0).repeat(self.num_modalities, 1),
            persistent=True,
        )
        self.register_buffer(
            "residual_cap_matrix",
            residual_cap.unsqueeze(0).repeat(self.num_modalities, 1),
            persistent=True,
        )
        if (
            self.ultrasound_modality_id is not None
            and 0 <= self.ultrasound_modality_id < self.num_modalities
            and self.stage_name == "c4_semantic"
        ):
            us = self.ultrasound_modality_id
            self.gamma_max_matrix[us].mul_(0.90)
            self.residual_cap_matrix[us].mul_(0.90)

        init_fraction = gamma_init / gamma_max.clamp_min(1e-5)
        self.route_gamma_bias = nn.Parameter(
            torch.logit(init_fraction.clamp(1e-4, 1 - 1e-4))
        )

        self.expert_scale_max = max(0.05, float(expert_scale_max))
        init_scale = min(
            max(float(expert_scale_init), 1e-4),
            self.expert_scale_max - 1e-4,
        )
        scale_fraction = init_scale / self.expert_scale_max
        self.raw_expert_scale = nn.Parameter(torch.full(
            (self.num_experts,),
            float(torch.logit(torch.tensor(scale_fraction))),
        ))
        if self.null_expert_index is not None:
            with torch.no_grad():
                self.raw_expert_scale[self.null_expert_index] = -12.0

        self.gamma_min = 0.0
        self.gamma_max = float(gamma_max.max())
        self.min_delta_ratio = 0.0
        self.max_delta_ratio = float(residual_cap.max())

        self.latest_weights = {}
        self.latest_dense_probs = {}
        self.latest_topk_indices = {}
        self.latest_gates = {}
        self.latest_modalities = {}
        self.latest_execution_counts = {}
        self.latest_contribution_mass = {}
        self.latest_active = {}
        self.latest_delta = {}
        self.latest_alpha = {}
        self.latest_gamma = {}
        self.latest_reliability = {}
        self.latest_image_logits = {}
        self.latest_task_logit_residual = {}
        self.latest_modality_logit_residual = {}
        self.latest_content_branch_logits = {}
        self.latest_task_branch_logits = {}
        self.latest_modality_branch_logits = {}
        self.latest_branch_alphas = {}
        # Detailed expert diagnostics are optional during ordinary training,
        # but reset_runtime_state() clears them unconditionally on every
        # forward pass.  Initialize each store here so the first batch can
        # enter the router even when detailed collection is disabled.
        self.latest_expert_band_rms = {}
        self.latest_expert_band_fractions = {}
        self.latest_expert_target_fraction = {}
        self.latest_expert_target_dominant = {}
        self.latest_expert_source_cosine = {}
        self.latest_expert_target_cosine = {}
        self.latest_expert_applied_scale = {}
        self.latest_expert_visuals = {}
        self.latest_base_rms = None
        self.latest_band_energies = None
        self._latest_logit_magnitude_terms = []
        self._latest_context_mi_loss_terms = []
        self._latest_context_mi_value_terms = []
        self._latest_context_local_entropy_terms = []
        self._latest_context_global_entropy_terms = []
        # Expensive full-resolution contribution/delta diagnostics are disabled
        # during ordinary training and enabled periodically or for validation.
        self.collect_detailed_runtime_stats = False

    @staticmethod
    def _expand_route_values(values, count, default):
        if values is None:
            return torch.full((count,), float(default), dtype=torch.float32)
        tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
        if tensor.numel() == 1:
            tensor = tensor.repeat(count)
        if tensor.numel() != count:
            raise ValueError(
                f"Expected {count} route values, received {tensor.numel()}"
            )
        return tensor

    @staticmethod
    def _route_module_key(route_name):
        return "route_" + str(route_name).replace(".", "_")

    def effective_router_branch_alphas(self, mode=None):
        base = F.softmax(self.raw_router_branch_alpha.float(), dim=0)
        return base / base.sum().clamp_min(1e-8)

    def _apply_modality_content_adapters(self, image_context, modality_ids):
        if (
            self.router_strategy != "factorized_multigate"
            or not self.router_modality_content_adapters_enabled
            or len(self.modality_content_adapters) == 0
        ):
            return image_context
        adapted = torch.empty_like(image_context)
        for mid in torch.unique(modality_ids).tolist():
            mid = int(mid)
            mask = modality_ids == mid
            adapted[mask] = self.modality_content_adapters[mid](
                image_context[mask]
            )
        return adapted

    def reset_runtime_state(self):
        for store in (
            self.latest_weights,
            self.latest_dense_probs,
            self.latest_topk_indices,
            self.latest_gates,
            self.latest_modalities,
            self.latest_execution_counts,
            self.latest_contribution_mass,
            self.latest_active,
            self.latest_delta,
            self.latest_alpha,
            self.latest_gamma,
            self.latest_reliability,
            self.latest_image_logits,
            self.latest_task_logit_residual,
            self.latest_modality_logit_residual,
            self.latest_content_branch_logits,
            self.latest_task_branch_logits,
            self.latest_modality_branch_logits,
            self.latest_branch_alphas,
            self.latest_expert_band_rms,
            self.latest_expert_band_fractions,
            self.latest_expert_target_fraction,
            self.latest_expert_target_dominant,
            self.latest_expert_source_cosine,
            self.latest_expert_target_cosine,
            self.latest_expert_applied_scale,
            self.latest_expert_visuals,
        ):
            store.clear()
        self.latest_base_rms = None
        self.latest_band_energies = None
        self._latest_logit_magnitude_terms = []
        self._latest_context_mi_loss_terms = []
        self._latest_context_mi_value_terms = []
        self._latest_context_local_entropy_terms = []
        self._latest_context_global_entropy_terms = []

    def set_temperature(self, value):
        self.temperature = max(float(value), 1e-3)

    def set_expert_warmup(self, enabled, epoch=-1):
        enabled = bool(enabled)
        self.current_epoch = int(epoch)
        if not enabled and self.warmup_end_epoch is None:
            self.warmup_end_epoch = int(epoch)
        self.expert_warmup = enabled

        self.sparse_mix_alpha = 0.0
        post_warmup_epoch = max(
            0, int(epoch) - max(int(self.warmup_end_epoch or epoch), 0)
        ) if not enabled else 0

        progress = max(0.0, min(post_warmup_epoch / 12.0, 1.0))
        self.current_router_noise_std = (
            self.base_router_noise_std * (1.0 - progress)
        )
        self.exploration_epsilon = (
            self.base_exploration_epsilon * (1.0 - progress)
        )
        self.current_expert_dropout = (
            self.base_expert_dropout * (1.0 - progress)
        )

    def set_detailed_runtime_stats(self, enabled):
        self.collect_detailed_runtime_stats = bool(enabled)

    def effective_expert_scale(self):
        scale = self.expert_scale_max * torch.sigmoid(self.raw_expert_scale)
        scale = scale.clone()
        if self.null_expert_index is not None:
            scale[self.null_expert_index] = 0.0
        return scale

    def effective_gamma(self):
        return self.gamma_max_matrix.mean(dim=0)

    def _base_router_features(self, x, band_energies):
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x_obs = x.detach().float()
            mean = x_obs.mean(dim=(2, 3))
            std = x_obs.std(dim=(2, 3), unbiased=False)
            vmax = x_obs.amax(dim=(2, 3))
            vmin = x_obs.amin(dim=(2, 3))
            moment_content = self.content_proj(
                torch.cat([mean, std, vmax, vmin], dim=1)
            )
            if self.router_attention_pool_enabled:
                attn_features = self.router_attention_features(x_obs)
                attn_scores = self.router_attention_scores(
                    attn_features
                ).flatten(1)
                attn_weights = F.softmax(attn_scores, dim=1)
                spatial_tokens = x_obs.flatten(2).transpose(1, 2)
                pooled = torch.bmm(
                    attn_weights.unsqueeze(1), spatial_tokens
                ).squeeze(1)
                attentive_content = self.attention_content_proj(pooled)
                content = self.content_fuse(torch.cat([
                    moment_content, attentive_content
                ], dim=1))
            else:
                content = moment_content

            energies = band_energies.detach().float().clamp_min(1e-6)
            log_energy = torch.log(energies)
            relative_log = log_energy - log_energy.mean(dim=1, keepdim=True)
            fractions = energies / energies.sum(dim=1, keepdim=True)
            spectral_entropy = -(
                fractions * fractions.clamp_min(1e-8).log()
            ).sum(dim=1, keepdim=True) / math.log(3.0)
            low_high = torch.log(
                energies[:, 0:1] / energies[:, 2:3]
            )
            energy = self.energy_proj(torch.cat([
                relative_log, fractions, spectral_entropy, low_high
            ], dim=1))

            spatial_map = x_obs.mean(dim=1)
            dx = (spatial_map[:, :, 1:] - spatial_map[:, :, :-1]).abs()
            dy = (spatial_map[:, 1:, :] - spatial_map[:, :-1, :]).abs()
            h, w = spatial_map.shape[-2:]
            h0, h1 = h // 4, max(h // 4 + 1, 3 * h // 4)
            w0, w1 = w // 4, max(w // 4 + 1, 3 * w // 4)
            center = spatial_map[:, h0:h1, w0:w1]
            spatial_raw = torch.stack([
                spatial_map.mean(dim=(1, 2)),
                spatial_map.std(dim=(1, 2), unbiased=False),
                spatial_map.amax(dim=(1, 2)),
                spatial_map.amin(dim=(1, 2)),
                dx.mean(dim=(1, 2)),
                dy.mean(dim=(1, 2)),
                spatial_map.square().mean(dim=(1, 2)).sqrt(),
                center.mean(dim=(1, 2)) - spatial_map.mean(dim=(1, 2)),
            ], dim=1)
            spatial = self.spatial_proj(spatial_raw)
        return content.float(), energy.float(), spatial.float()

    def _router_context(
        self, content, energy, spatial, route_name, modality_ids
    ):
        route_id = self.route_to_id[str(route_name)]
        route_ids = torch.full(
            (content.shape[0],), route_id,
            dtype=torch.long, device=content.device,
        )
        mids = modality_ids.to(
            device=content.device, dtype=torch.long
        ).clamp_(0, self.num_modalities - 1)
        route_embed = self.route_embedding(route_ids).float()
        modality_embed = self.modality_embedding(mids).float()

        # Main router query: image evidence only.
        image_context = torch.cat([content, energy, spatial], dim=1).float()
        # Auxiliary heads may still know task/modality.
        aux_context = torch.cat([
            image_context, route_embed,
            self.modality_scale * modality_embed,
        ], dim=1).float()
        return image_context, aux_context, route_ids, mids

    def _accumulate_context_mi(
        self, clean_probs, route_ids, modality_ids
    ):
        """Accumulate -I(image; expert | stage, route, modality).

        When a context has enough samples in the current batch, the exact
        context marginal is used. Otherwise, a detached rolling queue supplies
        the recent context marginal. The current distribution always retains a
        fixed share of the marginal so gradients do not vanish for singleton
        medical-modality slots. Router noise/dropout cannot satisfy this loss
        because ``clean_probs`` are computed before stochastic routing.
        """
        if clean_probs.numel() == 0:
            return
        log_norm = math.log(max(self.num_experts, 2))
        route_id = int(route_ids[0].detach().item())
        unique_modalities = torch.unique(modality_ids.detach())

        for modality_value in unique_modalities:
            modality_id = int(modality_value.item())
            mask = modality_ids == modality_value
            current = clean_probs[mask].float().clamp_min(1e-8)
            n_current = int(current.shape[0])
            if n_current < 1:
                continue

            local_entropy = -(current * current.log()).sum(dim=-1).mean()
            local_entropy = local_entropy / log_norm
            current_mean = current.mean(dim=0)

            history_count = int(
                self.router_mi_queue_count[route_id, modality_id].item()
            )
            enough_current = n_current >= self.router_mi_min_group_size
            enough_total = (
                n_current + history_count >= self.router_mi_min_group_size
            )
            if enough_current:
                marginal = current_mean
            elif enough_total and history_count > 0:
                history = self.router_mi_queue[
                    route_id, modality_id, :history_count
                ].detach().to(device=current.device, dtype=current.dtype)
                history_mean = history.mean(dim=0)
                current_fraction = self.router_mi_current_fraction
                marginal = (
                    current_fraction * current_mean
                    + (1.0 - current_fraction) * history_mean
                )
            else:
                marginal = None

            if marginal is not None:
                marginal = marginal.clamp_min(1e-8)
                global_entropy = -(marginal * marginal.log()).sum()
                global_entropy = global_entropy / log_norm
                # Minimizing local-global maximizes conditional MI. The loss
                # can be negative; its magnitude is bounded by one.
                mi_loss = local_entropy - global_entropy
                mi_value = global_entropy - local_entropy
                self._latest_context_mi_loss_terms.append(mi_loss)
                self._latest_context_mi_value_terms.append(mi_value.detach())
                self._latest_context_local_entropy_terms.append(
                    local_entropy.detach()
                )
                self._latest_context_global_entropy_terms.append(
                    global_entropy.detach()
                )

            if self.training:
                with torch.no_grad():
                    detached = current.detach()
                    for row in detached:
                        ptr = int(
                            self.router_mi_queue_ptr[
                                route_id, modality_id
                            ].item()
                        )
                        self.router_mi_queue[
                            route_id, modality_id, ptr
                        ].copy_(row)
                        ptr = (ptr + 1) % self.router_mi_queue_size
                        self.router_mi_queue_ptr[
                            route_id, modality_id
                        ] = ptr
                        count = int(
                            self.router_mi_queue_count[
                                route_id, modality_id
                            ].item()
                        )
                        self.router_mi_queue_count[
                            route_id, modality_id
                        ] = min(count + 1, self.router_mi_queue_size)

    def _route(self, image_context, aux_context, route_ids, modality_ids):
        device_type = image_context.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            shared_context = image_context.float()
            modality_context = self._apply_modality_content_adapters(
                shared_context, modality_ids
            )
            query = self.image_router_encoder(shared_context).float()
            query = F.normalize(query, dim=-1, eps=1e-6)
            keys = F.normalize(self.expert_keys.float(), dim=-1, eps=1e-6)
            logit_scale = F.softplus(self.raw_router_logit_scale.float()).clamp(
                min=1.0, max=8.0
            )
            content_logits = logit_scale * torch.matmul(query, keys.t())
            content_logits = content_logits - content_logits.mean(
                dim=-1, keepdim=True
            )

            task_logits = torch.zeros_like(content_logits)
            if self.router_task_specific_gates and len(self.task_gates) > 0:
                for rid in torch.unique(route_ids).tolist():
                    rid = int(rid)
                    mask = route_ids == rid
                    route_name = self.route_names[rid]
                    task_logits[mask] = self.task_gates[
                        self._route_module_key(route_name)
                    ](shared_context[mask])

            modality_logits = torch.zeros_like(content_logits)
            if (
                self.router_modality_specific_gates
                and len(self.modality_gates) > 0
            ):
                for mid in torch.unique(modality_ids).tolist():
                    mid = int(mid)
                    mask = modality_ids == mid
                    modality_logits[mask] = self.modality_gates[mid](
                        modality_context[mask]
                    )

            branch_alphas = self.effective_router_branch_alphas()
            content_component = branch_alphas[0] * content_logits
            task_component = branch_alphas[1] * task_logits
            modality_component = branch_alphas[2] * modality_logits
            raw_logits = content_component + task_component + modality_component

            self._latest_logit_magnitude_terms.append(
                torch.logsumexp(raw_logits, dim=-1).square().mean()
            )
            temperature = max(self.temperature, 1e-3)
            clean_dense_probs = F.softmax(raw_logits / temperature, dim=-1)
            self._accumulate_context_mi(
                clean_dense_probs, route_ids, modality_ids
            )
            logits = raw_logits

            if self.training:
                noise_scale = float(self.current_router_noise_std)
                if self.expert_warmup:
                    noise_scale *= 1.25
                if noise_scale > 0.0:
                    logits = logits + noise_scale * torch.randn_like(logits)

                p_drop = float(self.current_expert_dropout)
                if self.expert_warmup:
                    p_drop = min(0.20, 1.25 * p_drop)
                if p_drop > 0.0:
                    drop = torch.rand_like(logits).lt(p_drop)
                    enough = (~drop).sum(dim=1) >= self.top_k
                    drop = drop & enough.unsqueeze(1)
                    logits = logits.masked_fill(drop, -1e4)

            dense_probs = F.softmax(logits / temperature, dim=-1)
            _, top_indices = torch.topk(
                logits, k=self.top_k, dim=-1
            )
            forward_weights = dense_probs
            weights = dense_probs

            reliability = torch.sigmoid(
                self.reliability_head(aux_context.float()).float()
            )
            reliability = reliability.clone()
            if self.null_expert_index is not None:
                reliability[:, self.null_expert_index] = 1.0

            gamma_logits = self.gamma_head(aux_context.float()).flatten()
            gamma_logits = gamma_logits + self.route_gamma_bias.index_select(
                0, route_ids
            )
            gamma_max = self.gamma_max_matrix[
                modality_ids, route_ids
            ].float()
            gamma = gamma_max * torch.sigmoid(gamma_logits)
            if self.expert_warmup and self.training:
                gamma = 0.5 * gamma

        return (
            weights,
            forward_weights.detach(),
            dense_probs,
            top_indices,
            reliability,
            gamma,
            content_component.detach(),
            task_component.detach(),
            modality_component.detach(),
            content_logits.detach(),
            task_logits.detach(),
            modality_logits.detach(),
            branch_alphas.detach(),
        )

    @torch.no_grad()
    def _expert_outputs(self, x, bands):
        # Experts execute in the current AMP dtype. FP32 is used only for the
        # detached RMS calculations that bound each residual safely.
        base_rms = x.detach().float().square().mean(
            dim=(1, 2, 3), keepdim=True
        ).add(1e-12).sqrt()
        raw_outputs = []
        if self.expert_bank_type == "freqxpert":
            for expert, (source, target, _) in zip(
                self.directed_experts, self.direction_specs
            ):
                raw_outputs.append(expert(bands[source], bands[target]))
        else:
            normalized_x = x / base_rms.to(dtype=x.dtype).clamp_min(1e-6)
            for expert in self.standard_experts:
                raw_outputs.append(expert(normalized_x))

        if self.include_generalist:
            normalized_x = x / base_rms.to(dtype=x.dtype).clamp_min(1e-6)
            raw_outputs.append(self.general_expert(normalized_x))
        if self.include_null:
            raw_outputs.append(torch.zeros_like(x))

        if len(raw_outputs) != self.num_experts:
            raise RuntimeError(
                f"Expert inventory mismatch: built {len(raw_outputs)}, "
                f"expected {self.num_experts}"
            )
        expert_scales = self.effective_expert_scale().float()
        outputs = []
        for expert_id, output in enumerate(raw_outputs):
            if (
                self.null_expert_index is not None
                and expert_id == self.null_expert_index
            ):
                outputs.append(output)
                continue
            output_rms = output.detach().float().square().mean(
                dim=(1, 2, 3), keepdim=True
            ).add(1e-12).sqrt()
            max_rms = base_rms * expert_scales[expert_id]
            gain_cap = 4.0 if self.expert_warmup else 8.0
            safe_scale = torch.clamp(
                max_rms / output_rms.clamp_min(1e-8), max=gain_cap
            )
            outputs.append(output * safe_scale.to(dtype=output.dtype))
        return torch.stack(outputs, dim=1)

    def forward_routes(self, feature_map, route_masks, modality_ids=None):
        batch_size = int(feature_map.shape[0])
        device = feature_map.device
        if modality_ids is None:
            modality_ids = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
        else:
            modality_ids = modality_ids.to(device=device, dtype=torch.long)

        masks = {
            name: route_masks.get(
                name,
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            ).to(device=device, dtype=torch.bool)
            for name in self.route_names
        }
        outputs = {name: feature_map for name in self.route_names}
        union = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for mask in masks.values():
            union |= mask

        self.reset_runtime_state()
        self.latest_base_rms = feature_map.detach().float().square().mean(
            dim=(1, 2, 3)
        ).add(1e-12).sqrt()
        for name, mask in masks.items():
            self.latest_active[name] = mask.detach()
            self.latest_delta[name] = torch.zeros_like(feature_map).detach()

        if not bool(union.any()):
            return outputs

        union_idx = union.nonzero(as_tuple=True)[0]
        x_union = feature_map.index_select(0, union_idx)
        # The identity path trains the shared encoder normally.  Expert-branch
        # gradients are deliberately attenuated so MoFX cannot damage the shared
        # representation used by unrelated datasets and semantic heads.
        x_expert_union = (
            x_union.detach()
            + self.expert_input_grad_scale * (x_union - x_union.detach())
        )
        bands, band_energies, _ = self.band_decomposition(x_expert_union)
        self.latest_band_energies = band_energies.detach()
        expert_union = self._expert_outputs(x_expert_union, bands)

        content_union, energy_union, spatial_union = self._base_router_features(
            x_union, band_energies
        )
        modality_union = modality_ids.index_select(0, union_idx)
        union_position = torch.full(
            (batch_size,), -1, dtype=torch.long, device=device
        )
        union_position[union_idx] = torch.arange(
            union_idx.numel(), device=device
        )

        for route_name in self.route_names:
            active = masks[route_name]
            if not bool(active.any()):
                continue
            idx = active.nonzero(as_tuple=True)[0]
            pos = union_position.index_select(0, idx)
            x = x_union.index_select(0, pos)
            route_experts = expert_union.index_select(0, pos)
            image_context, aux_context, route_ids, mids = self._router_context(
                content_union.index_select(0, pos),
                energy_union.index_select(0, pos),
                spatial_union.index_select(0, pos),
                route_name,
                modality_union.index_select(0, pos),
            )
            (
                weights_for_grad,
                forward_weights,
                dense_probs,
                topk,
                reliability,
                gamma,
                image_logits,
                task_logit_residual,
                modality_logit_residual,
                content_branch_logits,
                task_branch_logits,
                modality_branch_logits,
                branch_alphas,
            ) = self._route(
                image_context, aux_context, route_ids, mids
            )

            # Large expert tensors and mixture aggregation remain in the
            # current AMP dtype. Router probabilities and safety statistics are
            # still computed in FP32.
            mix_weights = (
                reliability * weights_for_grad
            ).to(dtype=route_experts.dtype)
            weighted = (
                route_experts
                * mix_weights.view(-1, self.num_experts, 1, 1, 1)
            ).sum(dim=1)
            delta = gamma.to(dtype=weighted.dtype).view(-1, 1, 1, 1) * weighted

            base_rms = x.detach().float().square().mean(
                dim=(1, 2, 3), keepdim=True
            ).add(1e-12).sqrt()
            delta_rms = delta.detach().float().square().mean(
                dim=(1, 2, 3), keepdim=True
            ).add(1e-12).sqrt()
            cap = self.residual_cap_matrix[
                mids, route_ids
            ].float().view(-1, 1, 1, 1)
            cap_scale = torch.clamp(
                cap * base_rms / delta_rms.clamp_min(1e-8),
                max=1.0,
            ).detach()

            delta = delta * cap_scale.to(dtype=delta.dtype)
            adapted_values = x + delta

            adapted = feature_map.clone()
            adapted.index_copy_(0, idx, adapted_values.to(adapted.dtype))
            outputs[route_name] = adapted

            actual_ratio = (
                delta.detach().float().square().mean(
                    dim=(1, 2, 3)
                ).add(1e-12).sqrt()
                / base_rms.flatten().clamp_min(1e-8)
            )
            if self.routing_mode == "dense_soft":
                execution_counts = torch.full(
                    (self.num_experts,), float(x.shape[0]),
                    device=x.device, dtype=torch.float32,
                )
            else:
                execution_counts = torch.bincount(
                    topk.flatten(), minlength=self.num_experts
                ).float()

            contribution = None
            full_delta = None
            if self.collect_detailed_runtime_stats:
                full_delta = torch.zeros_like(feature_map)
                full_delta.index_copy_(0, idx, delta.to(full_delta.dtype))
                contribution_values = []
                for expert_id in range(self.num_experts):
                    applied = (
                        route_experts[:, expert_id].detach().float()
                        * reliability[:, expert_id].detach().float().view(
                            -1, 1, 1, 1
                        )
                        * forward_weights[:, expert_id].detach().float().view(
                            -1, 1, 1, 1
                        )
                        * gamma.detach().float().view(-1, 1, 1, 1)
                        * cap_scale
                    )
                    contribution_values.append(
                        applied.square().mean(
                            dim=(1, 2, 3)
                        ).add(1e-12).sqrt().mean()
                    )
                contribution = torch.stack(contribution_values) / (
                    base_rms.mean().clamp_min(1e-8)
                )

            self.latest_weights[route_name] = forward_weights.detach()
            self.latest_dense_probs[route_name] = dense_probs.detach()
            self.latest_topk_indices[route_name] = topk.detach()
            self.latest_gates[route_name] = (
                forward_weights.max(dim=1).values.detach()
            )
            self.latest_modalities[route_name] = mids.detach()
            self.latest_execution_counts[route_name] = execution_counts.detach()
            if contribution is not None:
                self.latest_contribution_mass[route_name] = contribution.detach()
            if full_delta is not None:
                self.latest_delta[route_name] = full_delta.detach()
            self.latest_alpha[route_name] = actual_ratio.detach()
            self.latest_gamma[route_name] = gamma.detach()
            self.latest_reliability[route_name] = reliability.detach()
            self.latest_image_logits[route_name] = image_logits.detach()
            self.latest_task_logit_residual[route_name] = (
                task_logit_residual.detach()
            )
            self.latest_modality_logit_residual[route_name] = (
                modality_logit_residual.detach()
            )
            self.latest_content_branch_logits[route_name] = (
                content_branch_logits.detach()
            )
            self.latest_task_branch_logits[route_name] = (
                task_branch_logits.detach()
            )
            self.latest_modality_branch_logits[route_name] = (
                modality_branch_logits.detach()
            )
            self.latest_branch_alphas[route_name] = branch_alphas.detach()

        return outputs

    def regularization_losses(self):
        # No uniform-usage or task-separation target is imposed. The optional
        # context-MI term only asks clean image-conditioned probabilities to
        # carry information beyond their stage x task x modality context.
        zero = self.raw_router_logit_scale.sum() * 0.0
        z_loss = (
            torch.stack(self._latest_logit_magnitude_terms).mean()
            if self._latest_logit_magnitude_terms else zero
        )
        context_mi_loss = (
            torch.stack(self._latest_context_mi_loss_terms).mean()
            if self._latest_context_mi_loss_terms else zero
        )
        context_mi_value = (
            torch.stack(self._latest_context_mi_value_terms).mean()
            if self._latest_context_mi_value_terms else zero.detach()
        )
        context_local_entropy = (
            torch.stack(self._latest_context_local_entropy_terms).mean()
            if self._latest_context_local_entropy_terms else zero.detach()
        )
        context_global_entropy = (
            torch.stack(self._latest_context_global_entropy_terms).mean()
            if self._latest_context_global_entropy_terms else zero.detach()
        )
        return {
            "sparsity": zero,
            "load_balance": zero,
            "task_separation": zero,
            "expert_diversity": zero,
            "gate_usage": zero,
            "router_conditioning": zero,
            "context_mi_loss": context_mi_loss,
            "context_mi": context_mi_value,
            "context_local_entropy": context_local_entropy,
            "context_global_entropy": context_global_entropy,
            "logit_magnitude": z_loss,
        }

    @staticmethod
    def _pair_peak(topk):
        if topk is None or topk.numel() == 0:
            return 0.0, []
        pairs = torch.sort(topk.detach().long(), dim=1).values
        unique, counts = torch.unique(pairs, dim=0, return_counts=True)
        index = int(counts.argmax().item())
        return (
            float(counts[index].float() / pairs.shape[0]),
            unique[index].tolist(),
        )

    def _context_mi_summary(self, probabilities, modality_ids):
        p_all = probabilities.float().clamp_min(1e-8)
        mids = modality_ids.long()
        log_norm = math.log(max(self.num_experts, 2))
        total = 0
        local_sum = 0.0
        global_sum = 0.0
        mi_sum = 0.0
        groups = 0
        for modality_value in torch.unique(mids):
            mask = mids == modality_value
            n = int(mask.sum().item())
            if n < self.router_mi_min_group_size:
                continue
            p = p_all[mask]
            local = -(p * p.log()).sum(dim=-1).mean() / log_norm
            mean_p = p.mean(dim=0).clamp_min(1e-8)
            global_entropy = -(mean_p * mean_p.log()).sum() / log_norm
            mi = global_entropy - local
            total += n
            groups += 1
            local_sum += n * float(local.detach().cpu())
            global_sum += n * float(global_entropy.detach().cpu())
            mi_sum += n * float(mi.detach().cpu())
        if total < 1:
            return {
                "within_context_mi": 0.0,
                "within_context_local_entropy": 0.0,
                "within_context_global_entropy": 0.0,
                "within_context_group_count": 0,
            }
        return {
            "within_context_mi": mi_sum / total,
            "within_context_local_entropy": local_sum / total,
            "within_context_global_entropy": global_sum / total,
            "within_context_group_count": groups,
        }

    def routing_summary(self):
        routes = {}
        for route_name, weights in self.latest_weights.items():
            p = weights.float().clamp_min(1e-8)
            dense = self.latest_dense_probs[route_name].float().clamp_min(1e-8)
            topk = self.latest_topk_indices.get(route_name)
            entropy = -(p * p.log()).sum(dim=1) / math.log(self.num_experts)
            usage = torch.bincount(
                p.argmax(dim=1), minlength=self.num_experts
            ).float()
            usage = usage / usage.sum().clamp_min(1.0)
            execution = self.latest_execution_counts[route_name].float()
            pair_peak, pair = self._pair_peak(topk)
            reliability = self.latest_reliability.get(route_name)
            gamma = self.latest_gamma.get(route_name)
            image_logits = self.latest_image_logits.get(route_name)
            task_residual = self.latest_task_logit_residual.get(route_name)
            modality_residual = self.latest_modality_logit_residual.get(route_name)
            context_mi_stats = self._context_mi_summary(
                dense, self.latest_modalities[route_name]
            )
            routes[str(route_name)] = {
                "n": int(p.shape[0]),
                "mean_weights": p.mean(dim=0).tolist(),
                "mean_dense_probs": dense.mean(dim=0).tolist(),
                "normalized_entropy": float(entropy.mean()),
                "top1_usage": usage.tolist(),
                "routing_std": float(p.std(dim=0, unbiased=False).mean()),
                "mean_top1_weight": float(
                    self.latest_gates[route_name].float().mean()
                ),
                "mean_residual_ratio": float(
                    self.latest_alpha[route_name].float().mean()
                ),
                "mean_gamma": (
                    float(gamma.float().mean()) if gamma is not None else 0.0
                ),
                "mean_reliability": (
                    reliability.float().mean(dim=0).tolist()
                    if reliability is not None else []
                ),
                "expert_execution_fraction": (
                    execution / execution.sum().clamp_min(1.0)
                ).tolist(),
                "topk_pair_peak_usage": pair_peak,
                "dominant_topk_pair": pair,
                **context_mi_stats,
            }
        return {
            "stage": self.stage_name,
            "router_count": 1,
            "router_strategy": self.router_strategy,
            "router_configuration": self.router_configuration,
            "router_branch_alphas": {
                name: float(value)
                for name, value in zip(
                    self.router_branch_names,
                    self.effective_router_branch_alphas().detach().cpu().tolist(),
                )
            },
            "router_conditioning": [
                "image_channel_mean_std_max_min",
                "learned_spatial_attention_pool",
                "relative_frequency_energy",
                "spatial_gradient_and_center_statistics",
                "task_specific_image_conditioned_gate",
                "modality_specific_image_conditioned_gate",
                "modality_specific_content_adapter",
            ] if self.router_strategy == "factorized_multigate" else [
                "image_channel_mean_std_max_min",
                "learned_spatial_attention_pool",
                "relative_frequency_energy",
                "spatial_gradient_and_center_statistics",
                "bounded_task_logit_residual",
                "bounded_modality_logit_residual",
            ],
            "router_task_residual_scale": float(
                self.router_task_residual_scale
            ),
            "router_modality_residual_scale": float(
                self.router_modality_residual_scale
            ),
            "router_attention_pool": bool(
                self.router_attention_pool_enabled
            ),
            "routing_objective": "downstream_task_loss_with_optional_context_mi",
            "routing_penalties_enabled": bool(
                len(self._latest_context_mi_loss_terms) > 0
            ),
            "context_mi_conditioning": "stage_x_task_x_modality",
            "context_mi_min_group_size": int(
                self.router_mi_min_group_size
            ),
            "context_mi_queue_size": int(self.router_mi_queue_size),
            "context_mi_current_fraction": float(
                self.router_mi_current_fraction
            ),
            "input_conditioned": True,
            "decomposition": self.band_decomposition.summary(),
            "decomposition_usage": (
                "expert_inputs_and_router_descriptors"
                if self.expert_bank_type == "freqxpert"
                else "router_descriptors_only"
            ),
            "expert_bank_type": self.expert_bank_type,
            "experts": list(self.expert_names),
            "directions": [spec[2] for spec in self.direction_specs],
            "routing_mode": self.routing_mode,
            "selection_rule": (
                "all_experts_softmax"
                if self.routing_mode == "dense_soft"
                else "top_k_softmax"
            ),
            "active_experts_per_sample": (
                self.num_experts
                if self.routing_mode == "dense_soft"
                else self.top_k
            ),
            "top_k": self.top_k,
            "temperature": self.temperature,
            "active_task_ids": list(self.active_task_ids),
            "gamma": self.effective_gamma().detach().cpu().tolist(),
            "expert_scale": self.effective_expert_scale().detach().cpu().tolist(),
            "tasks": routes,
            "all_functional_experts_are_freqxperts": bool(
                self.expert_bank_type == "freqxpert"
                and not self.include_generalist
            ),
            "generalist_expert_enabled": bool(self.include_generalist),
            "null_expert_enabled": bool(self.include_null),
            "identity_bypass_available": True,
            "warmup_active": bool(self.expert_warmup),
            "warmup_mode": (
                "dense_soft_full_training"
                if self.routing_mode == "dense_soft"
                else "input_conditioned_dense_to_sparse"
            ),
            "sparse_mix_alpha": float(self.sparse_mix_alpha),
            "transition_epochs": int(self.transition_epochs),
            "exploration_epsilon": float(self.exploration_epsilon),
            "expert_dropout": float(self.current_expert_dropout),
            "router_noise_std": float(self.current_router_noise_std),
            "expert_input_grad_scale": float(self.expert_input_grad_scale),
            "article_faithful": True,
        }


ArticleMoFXBlock = MoFXDenseBlock


class MoFXDenseHierarchy(nn.Module):
    """Hierarchical MoFX-Dense implementation used by the paper model.

    C2/C3 specialize spatial segmentation and detection. C4 specializes
    classification, BI-RADS, pathology, and report generation. Each bank can
    use either directed FreqXperts or parameter-matched standard experts while
    retaining the same router and bounded residual formulation.
    """

    def __init__(
        self,
        num_tasks=4,
        num_modalities=5,
        attn_dim=96,
        num_heads=4,
        router_dim=128,
        top_k=2,
        routing_mode="dense_soft",
        expert_bank_type="freqxpert",
        temperature=1.5,
        modality_scale=1.0,
        router_strategy="content_dominant_factorized_cosine",
        router_task_residual_scale=0.10,
        router_modality_residual_scale=0.10,
        router_attention_pool=True,
        router_alpha_content_init=0.50,
        router_alpha_task_init=0.25,
        router_alpha_modality_init=0.25,
        router_task_specific_gates=True,
        router_modality_specific_gates=True,
        router_modality_content_adapters=True,
        router_branch_hidden=96,
        router_configuration="full",
        router_mi_min_group_size=4,
        router_mi_queue_size=32,
        router_mi_current_fraction=0.50,
        use_granularity_prior=True,
        gamma_init=0.015,
        gamma_min=0.0,
        gamma_max=0.10,
        expert_scale_init=0.35,
        expert_scale_max=1.0,
        generic_mlp_bottleneck=128,
        include_generalist=False,
        include_null=True,
        decomposition="gaussian_fixed",
        active_stages=("c2", "c3", "c4_semantic"),
        fft_initial_boundaries=(0.25, 0.55),
        fft_mask_temperature=0.08,
        fft_min_bandwidth=0.05,
        ultrasound_modality_id=None,
        router_noise_std=0.08,
        exploration_epsilon=0.06,
    ):
        super().__init__()
        del gamma_min
        self.requested_decomposition = str(decomposition)
        _routing_mode = str(routing_mode).strip().lower().replace("-", "_")
        _routing_aliases = {
            "dense": "dense_soft", "soft": "dense_soft",
        }
        self.routing_mode = _routing_aliases.get(_routing_mode, _routing_mode)
        if self.routing_mode != "dense_soft":
            raise ValueError("MoFX-Dense requires dense_soft routing.")
        _bank_type = str(expert_bank_type).strip().lower().replace("-", "_")
        _bank_aliases = {
            "freq": "freqxpert", "frequency": "freqxpert",
            "mofx": "freqxpert",
        }
        self.expert_bank_type = _bank_aliases.get(_bank_type, _bank_type)
        if self.expert_bank_type != "freqxpert":
            raise ValueError("MoFX-Dense requires the directed FreqXpert bank.")
        if bool(include_generalist):
            raise ValueError("MoFX-Dense does not include a generalist expert.")
        common = dict(
            num_modalities=num_modalities,
            attn_dim=attn_dim,
            num_heads=num_heads,
            router_dim=router_dim,
            top_k=top_k,
            routing_mode=self.routing_mode,
            expert_bank_type=self.expert_bank_type,
            temperature=temperature,
            modality_scale=modality_scale,
            router_strategy=router_strategy,
            router_task_residual_scale=router_task_residual_scale,
            router_modality_residual_scale=router_modality_residual_scale,
            router_attention_pool=router_attention_pool,
            router_alpha_content_init=router_alpha_content_init,
            router_alpha_task_init=router_alpha_task_init,
            router_alpha_modality_init=router_alpha_modality_init,
            router_task_specific_gates=router_task_specific_gates,
            router_modality_specific_gates=router_modality_specific_gates,
            router_modality_content_adapters=router_modality_content_adapters,
            router_branch_hidden=router_branch_hidden,
            router_configuration=router_configuration,
            router_mi_min_group_size=router_mi_min_group_size,
            router_mi_queue_size=router_mi_queue_size,
            router_mi_current_fraction=router_mi_current_fraction,
            use_granularity_prior=False,  # specialization emerges from input/task/modality context
            expert_scale_init=min(float(expert_scale_init), 0.50),
            expert_scale_max=min(max(float(expert_scale_max), 0.60), 1.00),
            generalist_bottleneck=max(
                96, min(int(generic_mlp_bottleneck), 192)
            ),
            include_generalist=bool(include_generalist),
            include_null=bool(include_null),
            decomposition=decomposition,
            fft_initial_boundaries=fft_initial_boundaries,
            fft_mask_temperature=fft_mask_temperature,
            fft_min_bandwidth=fft_min_bandwidth,
            ultrasound_modality_id=ultrasound_modality_id,
            router_noise_std=router_noise_std,
            exploration_epsilon=exploration_epsilon,
        )
        base_init = min(max(float(gamma_init), 0.005), 0.03)
        base_max = min(max(float(gamma_max), 0.08), 0.15)
        self.banks = nn.ModuleDict({
            "c2": MoFXDenseBlock(
                channels=192,
                route_names=("0", "3"),
                stage_name="c2",
                kv_pool_stride=4,
                gamma_init_by_route=(base_init, base_init),
                gamma_max_by_route=(min(base_max, 0.12), min(base_max, 0.10)),
                residual_cap_by_route=(0.08, 0.08),
                expert_input_grad_scale=0.25,
                **common,
            ),
            "c3": MoFXDenseBlock(
                channels=384,
                route_names=("0", "3"),
                stage_name="c3",
                kv_pool_stride=2,
                gamma_init_by_route=(base_init, base_init),
                gamma_max_by_route=(min(base_max, 0.14), min(base_max, 0.12)),
                residual_cap_by_route=(0.10, 0.10),
                expert_input_grad_scale=0.20,
                **common,
            ),
            "c4_semantic": MoFXDenseBlock(
                channels=768,
                route_names=(
                    "birads", "pathology", "classification", "report"
                ),
                stage_name="c4_semantic",
                kv_pool_stride=1,
                gamma_init_by_route=(
                    min(base_init, 0.015), min(base_init, 0.015),
                    min(base_init, 0.012), min(base_init, 0.010),
                ),
                gamma_max_by_route=(0.08, 0.08, 0.07, 0.06),
                residual_cap_by_route=(0.050, 0.050, 0.040, 0.035),
                expert_input_grad_scale=0.10,
                **common,
            ),
        })
        _stage_aliases = {
            "c4": "c4_semantic", "semantic": "c4_semantic",
            "spatial_c2": "c2", "spatial_c3": "c3",
        }
        requested_active_stages = (
            ("c2", "c3", "c4_semantic")
            if active_stages is None else tuple(active_stages)
        )
        normalized_active_stages = []
        for stage in requested_active_stages:
            key = _stage_aliases.get(str(stage).strip().lower(), str(stage).strip().lower())
            if key not in self.banks:
                raise ValueError(
                    f"Unknown active MoFX stage {stage!r}; available stages are "
                    f"{list(self.banks.keys())}"
                )
            if key not in normalized_active_stages:
                normalized_active_stages.append(key)
        if not normalized_active_stages:
            raise ValueError("At least one active MoFX stage is required")
        self.active_stages = tuple(normalized_active_stages)
        if self.active_stages != ("c2", "c3", "c4_semantic"):
            raise ValueError("MoFX-Dense requires active stages C2, C3, and C4.")

        self.decomposition_name = self.banks["c3"].decomposition_name
        self.router_strategy = self.banks["c3"].router_strategy
        self.router_configuration = self.banks["c3"].router_configuration
        self.latest_weights_by_stage = {}
        self.latest_band_energies_by_stage = {}

    def set_expert_warmup(self, enabled, epoch=-1):
        for bank in self.banks.values():
            bank.set_expert_warmup(enabled, epoch=epoch)

    def set_temperature(self, value):
        for bank in self.banks.values():
            bank.set_temperature(value)

    def set_detailed_runtime_stats(self, enabled):
        for bank in self.banks.values():
            bank.set_detailed_runtime_stats(enabled)

    def forward_stage(
        self,
        stage_name,
        feature_map,
        task_ids,
        modality_ids=None,
        task_active_masks=None,
    ):
        stage_name = str(stage_name)
        canonical_stage = (
            "c4_semantic" if stage_name in {"c4", "c4_semantic"}
            else stage_name
        )
        outputs = {}
        if canonical_stage not in self.active_stages:
            # Defensive identity path for an unknown or unavailable stage.
            outputs = {task_id: feature_map for task_id in task_ids}
        elif stage_name in {"c2", "c3"}:
            bank = self.banks[stage_name]
            route_masks = {}
            for task_id in task_ids:
                key = str(int(task_id))
                mask = (
                    task_active_masks.get(int(task_id))
                    if task_active_masks else None
                )
                if mask is None:
                    mask = torch.ones(
                        feature_map.shape[0],
                        dtype=torch.bool,
                        device=feature_map.device,
                    )
                route_masks[key] = mask
            routed = bank.forward_routes(
                feature_map, route_masks, modality_ids
            )
            outputs = {
                int(key): value for key, value in routed.items()
            }
        elif stage_name == "c4":
            bank = self.banks["c4_semantic"]
            route_masks = {}
            for route_name in task_ids:
                key = str(route_name)
                mask = (
                    task_active_masks.get(key)
                    if task_active_masks else None
                )
                if mask is None:
                    mask = torch.ones(
                        feature_map.shape[0],
                        dtype=torch.bool,
                        device=feature_map.device,
                    )
                route_masks[key] = mask
            outputs = bank.forward_routes(
                feature_map, route_masks, modality_ids
            )
        else:
            outputs = {task_id: feature_map for task_id in task_ids}

        self.latest_weights_by_stage = {
            name: dict(bank.latest_weights)
            for name, bank in self.banks.items()
        }
        self.latest_band_energies_by_stage = {
            name: bank.latest_band_energies
            for name, bank in self.banks.items()
        }
        return outputs

    def regularization_losses(self):
        bank_regs = [
            self.banks[name].regularization_losses()
            for name in self.active_stages
        ]
        if not bank_regs:
            raise RuntimeError("No MoFX banks available")
        average_keys = {
            "context_mi_loss", "context_mi",
            "context_local_entropy", "context_global_entropy",
        }
        out = {}
        for key in bank_regs[0]:
            values = [current[key] for current in bank_regs]
            stacked = torch.stack(values)
            out[key] = (
                stacked.mean() if key in average_keys else stacked.sum()
            )
        return out

    def routing_summary(self):
        return {
            "hierarchical": True,
            "design": (
                f"FreqXpert bank ({self.decomposition_name}) with four directed "
                "frequency-interaction experts; backbone residual is the "
                "general path; null expert enabled; input-conditioned "
                "reliability-scaled dense soft bounded residual routing"
            ),
            "routing_mode": self.routing_mode,
            "expert_bank_type": self.expert_bank_type,
            "router_strategy": self.router_strategy,
            "active_stages": list(self.active_stages),
            "router_count": 3,
            "all_experts_are_freqxperts": bool(
                self.expert_bank_type == "freqxpert"
                and not self.banks["c3"].include_generalist
            ),
            "identity_bypass_available": True,
            "article_faithful": True,
            "decomposition": {
                name: bank.band_decomposition.summary()
                for name, bank in self.banks.items()
            },
            "stages": {
                name: {
                    **bank.routing_summary(),
                    "active_in_stage_placement": bool(name in self.active_stages),
                }
                for name, bank in self.banks.items()
            },
        }



class SegmentationReportTokenExtractor(nn.Module):
    def __init__(self, feat_dim=768, llm_dim=2048, mask_size=14):
        super().__init__()
        self.feat_dim = feat_dim
        self.mask_size = mask_size

        # region / mask token from masked image features
        self.mask_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, llm_dim)
        )

        # spatial token from mask geometry
        self.spatial_proj = nn.Sequential(
            nn.Linear(mask_size * mask_size, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, llm_dim)
        )

        self.out_norm = nn.LayerNorm(llm_dim)

    def forward(self, feat_map, seg_attn):
        """
        feat_map: [B, C, H, W]   e.g. c4_rep
        seg_attn: [B, 1, H_img, W_img] or already resized
        returns: [B, 2, llm_dim]
        """
        B, C, H, W = feat_map.shape

        # resize mask to feature map size
        seg_small = F.interpolate(seg_attn, size=(H, W), mode='bilinear', align_corners=False)
        seg_small = seg_small.clamp(0.0, 1.0)

        # 1) mask / region token: masked pooling over report feature map
        denom = seg_small.sum(dim=(2, 3), keepdim=True) + 1e-6
        pooled = (feat_map * seg_small).sum(dim=(2, 3), keepdim=True) / denom
        pooled = pooled.squeeze(-1).squeeze(-1)              # [B, C]
        mask_token = self.mask_proj(pooled)                  # [B, llm_dim]

        # 2) spatial token: flattened mask shape/location
        mask_flat = F.interpolate(seg_small, size=(self.mask_size, self.mask_size),
                                  mode='bilinear', align_corners=False)
        mask_flat = mask_flat.flatten(1)                     # [B, mask_size*mask_size]
        spatial_token = self.spatial_proj(mask_flat)         # [B, llm_dim]

        tokens = torch.stack([mask_token, spatial_token], dim=1)  # [B, 2, llm_dim]
        return self.out_norm(tokens)
 
class LightFPNNeck(nn.Module):
    def __init__(self, in_channels=(192, 384, 768), out_channels=256):
        super().__init__()
        self.lateral_c2 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)

        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )

    def forward(self, c2, c3, c4):
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lateral_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)

        p2 = self.smooth_p2(p2)
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        return p2, p3, p4


import sys

class DummyBackboneSpec(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = list(num_channels)


def prepare_detection_targets(det_targets, active_mask, img_h, img_w, num_classes, device):
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    out = []

    for b in active_idx.tolist():
        t = det_targets[b]

        if t is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            if "boxes_norm" in t and t["boxes_norm"] is not None:
                boxes = t["boxes_norm"].to(device=device, dtype=torch.float32)
            else:
                boxes = normalize_boxes(
                    t["boxes"].to(device=device, dtype=torch.float32),
                    img_w,
                    img_h,
                )

            labels = t["labels"].to(device=device, dtype=torch.long)

        assert boxes.ndim == 2 and boxes.shape[-1] == 4, f"bad boxes shape: {boxes.shape}"
        assert labels.ndim == 1, f"bad labels shape: {labels.shape}"
        assert labels.shape[0] == boxes.shape[0], f"label/box count mismatch: {labels.shape[0]} vs {boxes.shape[0]}"

        if boxes.numel() > 0:
            assert torch.isfinite(boxes).all(), "NaN/Inf in boxes"
            assert boxes.min() >= -1e-6 and boxes.max() <= 1.0 + 1e-6, \
                f"boxes not normalized cxcywh in [0,1], min={boxes.min().item()}, max={boxes.max().item()}"

        if labels.numel() > 0:
            assert ((labels >= 0) & (labels < num_classes)).all(), \
                f"labels out of range: {labels.tolist()} num_classes={num_classes}"

        out.append({"boxes": boxes, "labels": labels})

    return out, active_idx


def compute_mask_measurements(mask):
    """Morphology vector from a binary mask (GT or PREDICTED); matches the BUS-BRA
    data layer so eval predicted-mask measurements match train GT.
    Returns (vec[float32, N_MEAS_BUSBRA], present_bool)."""
    import math
    zeros = np.zeros(N_MEAS_BUSBRA, dtype=np.float32)
    try:
        import cv2
    except Exception:
        return zeros, False
    if mask is None:
        return zeros, False
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim == 3:
        m = m.reshape(m.shape[-2], m.shape[-1]) if m.shape[0] == 1 else m.max(0)
    if m.sum() == 0:
        return zeros, False
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return zeros, False
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)
    max_d = perp_d = 0.0
    if len(hull) >= 2:
        diff = hull[:, None, :] - hull[None, :, :]
        d2 = np.sum(diff ** 2, -1)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        vec = hull[j] - hull[i]
        max_d = float(np.linalg.norm(vec))
        if max_d > 1e-6:
            u = vec / max_d
            v = np.array([-u[1], u[0]], np.float32)
            proj = hull @ v
            perp_d = float(proj.max() - proj.min())
    area = float(m.sum())
    c_area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x0, y0, w0, h0 = cv2.boundingRect(cnt)
    bw, bh = float(w0), float(h0)
    aspect = bh / max(bw, 1.0)
    extent = area / max(float(w0 * h0), 1.0)
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity = c_area / max(hull_area, 1.0)
    circ = (4.0 * math.pi * c_area) / max(perim * perim, 1.0)
    ecc = 0.0
    if len(cnt) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
        maj, mnr = max(a1, a2), min(a1, a2)
        ecc = math.sqrt(max(0.0, 1.0 - (mnr ** 2) / max(maj ** 2, 1.0)))
    vals = {"max_diameter_px": max_d, "perpendicular_diameter_px": perp_d,
            "bbox_width_px": bw, "bbox_height_px": bh, "aspect_ratio_h_over_w": aspect,
            "area_px": area, "perimeter_px": perim, "circularity": circ,
            "solidity": solidity, "extent": extent, "eccentricity": ecc}
    order = ["max_diameter_px", "perpendicular_diameter_px", "bbox_width_px", "bbox_height_px",
            "aspect_ratio_h_over_w", "area_px", "perimeter_px", "circularity",
            "solidity", "extent", "eccentricity"]
    return np.array([vals[k] for k in order], np.float32), True


def compute_mask_measurements_gt_aligned(mask):
    """BUS-BRA morphology matching the GT-report preprocessing definitions.

    This helper is intentionally separate from ``compute_mask_measurements``.
    The legacy helper remains unchanged for the already-trained pathology head,
    while this GT-aligned version is used for source-space report measurements.
    """
    zeros = np.zeros(N_MEAS_BUSBRA, dtype=np.float32)
    try:
        import cv2
    except Exception:
        return zeros, False
    if mask is None:
        return zeros, False
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim == 3:
        m = m.reshape(m.shape[-2], m.shape[-1]) if m.shape[0] == 1 else m.max(0)
    if m.sum() == 0:
        return zeros, False

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return zeros, False
    cnt = max(cnts, key=cv2.contourArea)

    # Feret maximum diameter and perpendicular hull span.
    hull_pts = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)
    max_d = perp_d = 0.0
    if len(hull_pts) >= 2:
        diff = hull_pts[:, None, :] - hull_pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        vec = hull_pts[j] - hull_pts[i]
        max_d = float(np.linalg.norm(vec))
        if max_d > 1e-6:
            u = vec / max_d
            v = np.array([-u[1], u[0]], dtype=np.float32)
            proj = hull_pts @ v
            perp_d = float(proj.max() - proj.min())

    # Match GT preprocessing: area/solidity are pixel counts from filled masks
    # of the largest lesion and its convex hull, respectively.
    lesion_mask = np.zeros_like(m)
    cv2.drawContours(lesion_mask, [cnt], -1, 1, thickness=cv2.FILLED)
    area = float(lesion_mask.sum())
    c_area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    _, _, w0, h0 = cv2.boundingRect(cnt)
    bw, bh = float(w0), float(h0)
    aspect = bh / max(bw, 1.0)
    extent = area / max(float(w0 * h0), 1.0)

    hull_contour = cv2.convexHull(cnt)
    hull_mask = np.zeros_like(m)
    cv2.drawContours(hull_mask, [hull_contour], -1, 1, thickness=cv2.FILLED)
    hull_area = float(hull_mask.sum())
    solidity = area / max(hull_area, 1.0)
    circ = (4.0 * math.pi * c_area) / max(perim * perim, 1.0)

    ecc = 0.0
    if len(cnt) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
        maj, mnr = max(a1, a2), min(a1, a2)
        ecc = math.sqrt(max(0.0, 1.0 - (mnr ** 2) / max(maj ** 2, 1e-8)))

    vals = {
        "max_diameter_px": max_d,
        "perpendicular_diameter_px": perp_d,
        "bbox_width_px": bw,
        "bbox_height_px": bh,
        "aspect_ratio_h_over_w": aspect,
        "area_px": area,
        "perimeter_px": perim,
        "circularity": circ,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": ecc,
    }
    return np.array([vals[k] for k in MEASUREMENT_KEYS_BUSBRA], np.float32), True


def compute_report_morphology_gt_aligned(mask):
    """Return all BUS-BRA quantitative values used by the GT report.

    Measurements are computed from the largest connected lesion in the mask.
    Area and solidity follow the GT preprocessing definitions based on filled
    lesion / convex-hull masks. Fitted-ellipse major/minor axes and their ratio
    are returned in addition to the 11-D learned measurement schema.
    """
    empty = {key: 0.0 for key in REPORT_MORPHOLOGY_KEYS_BUSBRA}
    try:
        import cv2
    except Exception:
        return empty, False
    if mask is None:
        return empty, False
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim == 3:
        m = m.reshape(m.shape[-2], m.shape[-1]) if m.shape[0] == 1 else m.max(0)
    if m.sum() == 0:
        return empty, False

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return empty, False
    cnt = max(cnts, key=cv2.contourArea)

    hull_pts = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)
    max_d = perp_d = 0.0
    if len(hull_pts) >= 2:
        diff = hull_pts[:, None, :] - hull_pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        vec = hull_pts[j] - hull_pts[i]
        max_d = float(np.linalg.norm(vec))
        if max_d > 1e-6:
            u = vec / max_d
            v = np.array([-u[1], u[0]], dtype=np.float32)
            proj = hull_pts @ v
            perp_d = float(proj.max() - proj.min())

    lesion_mask = np.zeros_like(m)
    cv2.drawContours(lesion_mask, [cnt], -1, 1, thickness=cv2.FILLED)
    area = float(lesion_mask.sum())
    c_area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    _, _, w0, h0 = cv2.boundingRect(cnt)
    bw, bh = float(w0), float(h0)
    aspect = bh / max(bw, 1.0)
    extent = area / max(float(w0 * h0), 1.0)

    hull_contour = cv2.convexHull(cnt)
    hull_mask = np.zeros_like(m)
    cv2.drawContours(hull_mask, [hull_contour], -1, 1, thickness=cv2.FILLED)
    hull_area = float(hull_mask.sum())
    solidity = area / max(hull_area, 1.0)
    circularity = (4.0 * math.pi * c_area) / max(perim * perim, 1.0)

    major_axis = 0.0
    minor_axis = 0.0
    major_minor_ratio = 0.0
    eccentricity = 0.0
    if len(cnt) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
        major_axis = float(max(a1, a2))
        minor_axis = float(min(a1, a2))
        if minor_axis > 1e-8:
            major_minor_ratio = major_axis / minor_axis
        eccentricity = math.sqrt(
            max(0.0, 1.0 - (minor_axis ** 2) / max(major_axis ** 2, 1e-8))
        )

    values = {
        "max_diameter_px": max_d,
        "perpendicular_diameter_px": perp_d,
        "bbox_width_px": bw,
        "bbox_height_px": bh,
        "aspect_ratio_h_over_w": aspect,
        "area_px": area,
        "perimeter_px": perim,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": eccentricity,
        "major_axis_px": major_axis,
        "minor_axis_px": minor_axis,
        "major_minor_ratio": major_minor_ratio,
    }
    return values, True


class SharedBackboneOfficialDINO(nn.Module):
    def __init__(
        self,
        repo_path,
        cfg_path,
        num_classes,
        feature_channels=(192, 384, 768),
        pretrained_path=None,
        device="cuda",
        num_queries=None,
    ):
        super().__init__()

        # The official repository is a directory named ``DINO`` containing
        # ``util/`` and ``models/``. Imports use the ``DINO.*`` namespace, so
        # Python needs both the repository parent (for DINO.*) and the
        # repository itself (for absolute util/models imports inside DINO).
        # Resolve these paths explicitly instead of relying on the process CWD.
        dino_repo = Path(repo_path).expanduser().resolve()
        if not (dino_repo / "util" / "slconfig.py").is_file():
            nested_repo = dino_repo / "DINO"
            if (nested_repo / "util" / "slconfig.py").is_file():
                dino_repo = nested_repo
            else:
                raise FileNotFoundError(
                    "Invalid DINO repository path. Expected util/slconfig.py "
                    f"under: {dino_repo}"
                )
        if dino_repo.name.lower() != "dino":
            raise ValueError(
                "The official DINO checkout directory must be named 'DINO' "
                "because the model imports the DINO package namespace; got "
                f"{dino_repo.name!r}."
            )
        for import_path in (dino_repo, dino_repo.parent):
            import_path_str = str(import_path)
            if import_path_str not in sys.path:
                sys.path.insert(0, import_path_str)
        print(
            f"[DINO] repository={dino_repo} | "
            f"package_root={dino_repo.parent}"
        )

        from DINO.util.slconfig import SLConfig
        from DINO.util.misc import NestedTensor, inverse_sigmoid
        from DINO.models.dino.deformable_transformer import build_deformable_transformer
        from DINO.models.dino.matcher import build_matcher
        from DINO.models.dino.dino import DINO, SetCriterion, PostProcess
        from DINO.models.dino.dn_components import prepare_for_cdn, dn_post_process

        # If this import path differs in your local clone, adjust only this one line.
        from DINO.models.dino.position_encoding import build_position_encoding

        self.NestedTensor = NestedTensor
        self.inverse_sigmoid = inverse_sigmoid
        self.prepare_for_cdn = prepare_for_cdn
        self.dn_post_process = dn_post_process

        cfg = SLConfig.fromfile(cfg_path)
        cfg.device = device
        cfg.num_classes = num_classes
        if num_queries is not None:
            cfg.num_queries = int(num_queries)
        cfg.dn_labelbook_size = max(getattr(cfg, "dn_labelbook_size", 91), num_classes + 1)

        self.cfg = cfg
        self.device = device
        self.position_embedding = build_position_encoding(cfg)

        dummy_backbone = DummyBackboneSpec(feature_channels)
        transformer = build_deformable_transformer(cfg)

        self.model = DINO(
            dummy_backbone,
            transformer,
            num_classes=cfg.num_classes,
            num_queries=cfg.num_queries,
            aux_loss=True,
            iter_update=True,
            query_dim=4,
            random_refpoints_xy=cfg.random_refpoints_xy,
            fix_refpoints_hw=cfg.fix_refpoints_hw,
            num_feature_levels=cfg.num_feature_levels,
            nheads=cfg.nheads,
            dec_pred_class_embed_share=cfg.dec_pred_class_embed_share,
            dec_pred_bbox_embed_share=cfg.dec_pred_bbox_embed_share,
            two_stage_type=cfg.two_stage_type,
            two_stage_add_query_num=cfg.two_stage_add_query_num,
            two_stage_bbox_embed_share=cfg.two_stage_bbox_embed_share,
            two_stage_class_embed_share=cfg.two_stage_class_embed_share,
            decoder_sa_type=cfg.decoder_sa_type,
            num_patterns=cfg.num_patterns,
            dn_number=cfg.dn_number,
            dn_box_noise_scale=cfg.dn_box_noise_scale,
            dn_label_noise_ratio=cfg.dn_label_noise_ratio,
            dn_labelbook_size=cfg.dn_labelbook_size,
        ).to(device)

        matcher = build_matcher(cfg)
        self.criterion = SetCriterion(
            num_classes=cfg.num_classes,
            matcher=matcher,
            weight_dict=self._build_weight_dict(cfg),
            focal_alpha=cfg.focal_alpha,
            losses=["labels", "boxes", "cardinality"],
        ).to(device)

        self.postprocessor = PostProcess(
            num_select=getattr(cfg, "num_select", 300),
            nms_iou_threshold=getattr(cfg, "nms_iou_threshold", -1),
        )

        if pretrained_path is not None and os.path.isfile(pretrained_path):
            self.load_pretrained(pretrained_path)

    def _build_weight_dict(self, cfg):
        cls_coef = getattr(cfg, "cls_loss_coef", 1.0)
        bbox_coef = getattr(cfg, "bbox_loss_coef", 5.0)
        giou_coef = getattr(cfg, "giou_loss_coef", 2.0)

        weight_dict = {
            "loss_ce": cls_coef,
            "loss_bbox": bbox_coef,
            "loss_giou": giou_coef,
        }

        # DN losses
        if getattr(cfg, "use_dn", True):
            weight_dict.update({
                "loss_ce_dn": cls_coef,
                "loss_bbox_dn": bbox_coef,
                "loss_giou_dn": giou_coef,
            })

        # Aux decoder losses
        dec_layers = getattr(cfg, "dec_layers", getattr(cfg, "num_decoder_layers", 6))
        for i in range(dec_layers - 1):
            weight_dict.update({
                f"loss_ce_{i}": cls_coef,
                f"loss_bbox_{i}": bbox_coef,
                f"loss_giou_{i}": giou_coef,
            })
            if getattr(cfg, "use_dn", True):
                weight_dict.update({
                    f"loss_ce_dn_{i}": cls_coef,
                    f"loss_bbox_dn_{i}": bbox_coef,
                    f"loss_giou_dn_{i}": giou_coef,
                })

        # Two-stage intermediate loss
        if getattr(cfg, "two_stage_type", "no") != "no":
            weight_dict.update({
                "loss_ce_interm": cls_coef,
                "loss_bbox_interm": bbox_coef,
                "loss_giou_interm": giou_coef,
            })

        return weight_dict

    def load_pretrained(self, ckpt_path):
        import argparse
        import torch

        # PyTorch 2.6+ defaults torch.load(..., weights_only=True),
        # which rejects argparse.Namespace in older checkpoints.
        # This official DINO checkpoint is trusted, so fall back safely.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            try:
                torch.serialization.add_safe_globals([argparse.Namespace])
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            except Exception:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state = ckpt.get("model", ckpt.get("state_dict", ckpt))

        current = self.model.state_dict()
        filtered = {}

        for k, v in state.items():
            # skip checkpoint parts that should not be loaded for shared-backbone Option B
            if k.startswith("backbone."):
                continue
            if k.startswith("input_proj."):
                continue
            if k.startswith("class_embed"):
                continue
            if k.startswith("label_enc"):
                continue
            if k.startswith("transformer.decoder.class_embed"):
                continue
            if k.startswith("transformer.enc_out_class_embed"):
                continue

            if k in current and current[k].shape == v.shape:
                filtered[k] = v

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        print(f"[Shared Official DINO] pretrained load | missing={len(missing)} unexpected={len(unexpected)}")

    def compute_loss(self, outputs, targets, return_indices=False):
        if return_indices:
            return self.criterion(outputs, targets, return_indices=True)

        loss_dict = self.criterion(outputs, targets)
        total = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for k, v in loss_dict.items():
            if k in self.criterion.weight_dict:
                total = total + self.criterion.weight_dict[k] * v
        return total, loss_dict

    def _make_base_mask(self, images):
        if isinstance(images, list):
            B = len(images)
            H, W = images[0].shape[-2:]
            device = images[0].device
        else:
            B, _, H, W = images.shape
            device = images.device

        return torch.zeros((B, H, W), dtype=torch.bool, device=device)

    def forward_from_shared_feats(self, images, feat_maps, targets=None):
        """
        feat_maps: list of shared-backbone feature maps, e.g. [c2_map, c3_map, c4_map]
        Returns:
            pred_logits      : [B, Q_main, C]
            pred_boxes       : [B, Q_main, 4]
            query_feats      : [B, Q_total, D]   (raw decoder queries, may include DN)
            main_query_feats : [B, Q_main, D]    (aligned with pred_logits/pred_boxes)
        """
        base_mask = self._make_base_mask(images)

        srcs = []
        masks = []
        poss = []

        # Build official srcs / masks / poss from YOUR shared Swin features
        for l, feat in enumerate(feat_maps):
            mask = F.interpolate(
                base_mask[:, None].float(),
                size=feat.shape[-2:],
                mode="nearest"
            ).to(torch.bool)[:, 0]

            nested = self.NestedTensor(feat, mask)
            pos = self.position_embedding(nested).to(feat.dtype)

            src = self.model.input_proj[l](feat)

            srcs.append(src)
            masks.append(mask)
            poss.append(pos)

        # Let official DINO create extra levels internally, same pattern as official forward
        if self.model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.model.num_feature_levels):
                if l == _len_srcs:
                    src = self.model.input_proj[l](feat_maps[-1])
                else:
                    src = self.model.input_proj[l](srcs[-1])

                mask = F.interpolate(
                    base_mask[:, None].float(),
                    size=src.shape[-2:],
                    mode="nearest"
                ).to(torch.bool)[:, 0]

                pos_l = self.position_embedding(self.NestedTensor(src, mask)).to(src.dtype)

                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.model.dn_number > 0 or targets is not None:
            if targets is not None:
                for ti, t in enumerate(targets):
                    if "boxes" in t and t["boxes"].numel() > 0:
                        boxes = t["boxes"]
                        if boxes.shape[-1] != 4:
                            raise ValueError(f"[DINO targets] target {ti} boxes must have shape [N,4], got {boxes.shape}")
                        if boxes.max() > 1.001 or boxes.min() < -1e-6:
                            raise ValueError(
                                f"[DINO targets] target {ti} boxes passed to prepare_for_cdn are not normalized cxcywh in [0,1]. "
                                f"min={boxes.min().item():.4f}, max={boxes.max().item():.4f}"
                            )
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                dn_args=(
                    targets,
                    self.model.dn_number,
                    self.model.dn_label_noise_ratio,
                    self.model.dn_box_noise_scale,
                ),
                training=self.training,
                num_queries=self.model.num_queries,
                num_classes=self.model.num_classes,
                hidden_dim=self.model.hidden_dim,
                label_enc=self.model.label_enc,
            )
        else:
            input_query_label = None
            input_query_bbox = None
            attn_mask = None
            dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.model.transformer(
            srcs,
            masks,
            input_query_bbox,
            poss,
            input_query_label,
            attn_mask,
        )

        # official stability trick
        hs[0] += self.model.label_enc.weight[0, 0] * 0.0

        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.model.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + self.inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)

        outputs_coord_list = torch.stack(outputs_coord_list)
        outputs_class = torch.stack([
            layer_cls_embed(layer_hs)
            for layer_cls_embed, layer_hs in zip(self.model.class_embed, hs)
        ])

        if self.model.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = self.dn_post_process(
                outputs_class,
                outputs_coord_list,
                dn_meta,
                self.model.aux_loss,
                self.model._set_aux_loss,
            )

        pred_logits = outputs_class[-1]
        pred_boxes = outputs_coord_list[-1]

        # ----------------------------------------------------------
        # Normalize decoder query features to [B, Q_total, D]
        # Then trim main_query_feats to align with pred_logits/pred_boxes
        # ----------------------------------------------------------
        raw_query_feats = hs[-1]

        if raw_query_feats.dim() == 2:
            # [Q, D] -> [1, Q, D]
            raw_query_feats = raw_query_feats.unsqueeze(0)

        elif raw_query_feats.dim() == 3:
            # Expected either [B, Q, D] or [Q, B, D]
            if raw_query_feats.shape[0] == pred_logits.shape[0]:
                # already [B, Q, D]
                pass
            elif raw_query_feats.shape[1] == pred_logits.shape[0]:
                # [Q, B, D] -> [B, Q, D]
                raw_query_feats = raw_query_feats.transpose(0, 1).contiguous()
            else:
                raise RuntimeError(
                    f"Unexpected raw_query_feats shape {tuple(raw_query_feats.shape)} "
                    f"for pred_logits shape {tuple(pred_logits.shape)}"
                )
        else:
            raise RuntimeError(
                f"Unexpected raw_query_feats ndim={raw_query_feats.dim()} "
                f"shape={tuple(raw_query_feats.shape)}"
            )

        # pred_logits is already post-DN, so its query dimension is the real matching query count
        Q_main = pred_logits.shape[1]
        Q_total = raw_query_feats.shape[1]

        if Q_total < Q_main:
            raise RuntimeError(
                f"raw_query_feats has fewer queries ({Q_total}) than pred_logits ({Q_main})."
            )

        # Keep only the actual matching queries.
        # DN queries come before matching queries, so keep the last Q_main queries.
        main_query_feats = raw_query_feats[:, -Q_main:, :].contiguous()

        out = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "dn_meta": dn_meta,
            "query_feats": raw_query_feats,         # full queries (may include DN)
            "main_query_feats": main_query_feats,   # aligned with pred_logits/pred_boxes
        }

        if self.model.aux_loss:
            out["aux_outputs"] = self.model._set_aux_loss(outputs_class, outputs_coord_list)

        if hs_enc is not None:
            interm_coord = ref_enc[-1]
            interm_class = self.model.transformer.enc_out_class_embed(hs_enc[-1])
            out["interm_outputs"] = {
                "pred_logits": interm_class,
                "pred_boxes": interm_coord,
            }
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": init_box_proposal,
            }

        return out

    @torch.no_grad()
    def postprocess(self, outputs, target_sizes):
        return self.postprocessor(outputs, target_sizes)
    
    

class MeasurementEncoder(nn.Module):
    """Normalize (z-score via registered buffers) + embed the morphology vector.
    Emits a learned NULL embedding where measurements are absent, so it is safe on
    every sample of a mixed unified batch. Call set_stats(mean,std) from the train
    set; defaults to identity normalization until then."""
    def __init__(self, out_dim=128, hidden=256, n_feat=N_MEAS_BUSBRA):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(n_feat))
        self.register_buffer("feat_std", torch.ones(n_feat))
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, out_dim),
        )
        self.null = nn.Parameter(torch.zeros(out_dim))
        self.out_dim = out_dim

    @torch.no_grad()
    def set_stats(self, mean, std):
        self.feat_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.feat_std.copy_(torch.clamp(torch.as_tensor(std, dtype=torch.float32), min=1e-6))

    def forward(self, x, present):
        z = (x - self.feat_mean) / self.feat_std
        emb = self.net(z)
        p = present.float().unsqueeze(-1) if present.dim() == 1 else present.float()
        return emb * p + self.null.unsqueeze(0) * (1.0 - p)


class MaskROIPool(nn.Module):
    """Masked average pool of a feature map over a (resized) binary mask ROI;
    falls back to global average pool where the mask is empty."""
    def forward(self, feat, mask):
        if mask is None:
            return feat.mean(dim=(2, 3))
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        m = (mask > 0.5).float()
        m = F.interpolate(m, size=feat.shape[-2:], mode="nearest")
        s = m.sum(dim=(2, 3))
        roi = (feat * m).sum(dim=(2, 3)) / s.clamp_min(1.0)
        glob = feat.mean(dim=(2, 3))
        use_roi = (s > 0).float()
        return roi * use_roi + glob * (1.0 - use_roi)


class BiRadsOrdinalHead(nn.Module):
    """CORAL-style ordinal BI-RADS head using two visual views only.

    ``global_feat`` is extracted from the complete ultrasound image and
    ``roi_feat`` from a lesion-centred crop passed through the same Swin
    backbone. Explicit pixel morphology measurements are intentionally excluded
    from BI-RADS classification to avoid crop-scale and segmentation-error
    leakage.
    """
    def __init__(self, global_dim, roi_dim, num_classes=4, hidden=384, bias_init=None):
        super().__init__()
        in_dim = global_dim + roi_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 768),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(768, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.20),
        )
        self.shared = nn.Linear(hidden, 1, bias=False)
        if bias_init is not None:
            _b = bias_init.detach().clone().float()
            assert _b.numel() == num_classes - 1,                 f"bias_init has {_b.numel()} entries, expected {num_classes - 1}"
            self.biases = nn.Parameter(_b)
        else:
            self.biases = nn.Parameter(torch.zeros(num_classes - 1))
        self.num_classes = num_classes

    def forward(self, global_feat, roi_feat):
        h = self.trunk(torch.cat([global_feat, roi_feat], dim=-1))
        return self.shared(h) + self.biases.unsqueeze(0)


def birads_ordinal_loss(logits, ordinal_target, mask=None):
    """BCE over cumulative thresholds; ordinal_target = [1]*c + [0]*(K-1-c)."""
    loss = F.binary_cross_entropy_with_logits(
        logits, ordinal_target.float(), reduction="none").sum(-1)
    if mask is not None:
        m = mask.float()
        return (loss * m).sum() / m.sum().clamp_min(1.0)
    return loss.mean()


def birads_decode(logits):
    """Category index = number of cumulative thresholds passed (CORAL decode)."""
    return (torch.sigmoid(logits) > 0.5).long().sum(-1)

class SwinMultiTask(nn.Module):
    def __init__(
        self,
        swin_model,
        use_moe=True,
        moe_type="gradif",
        moe_num_experts=5,
        moe_top_k=2,
        gradif_attn_dim=96,
        gradif_num_heads=4,
        gradif_router_dim=128,
        gradif_top_k=2,
        gradif_routing_mode="dense_soft",
        gradif_expert_bank_type="freqxpert",
        gradif_temperature=1.5,
        gradif_modality_scale=1.0,
        gradif_router_strategy="factorized_multigate",
        gradif_router_task_residual_scale=0.10,
        gradif_router_modality_residual_scale=0.10,
        gradif_router_attention_pool=True,
        gradif_router_alpha_content_init=0.50,
        gradif_router_alpha_task_init=0.25,
        gradif_router_alpha_modality_init=0.25,
        gradif_router_task_specific_gates=True,
        gradif_router_modality_specific_gates=True,
        gradif_router_modality_content_adapters=True,
        gradif_router_branch_hidden=96,
        mofx_router_configuration="full",
        gradif_router_mi_min_group_size=4,
        gradif_router_mi_queue_size=32,
        gradif_router_mi_current_fraction=0.50,
        gradif_gamma_init=0.015,
        gradif_gamma_min=0.0,
        gradif_gamma_max=0.12,
        gradif_expert_scale_init=0.35,
        gradif_expert_scale_max=1.0,
        gradif_generic_mlp_bottleneck=96,
        gradif_include_generalist=False,
        gradif_include_null_expert=True,
        gradif_use_granularity_prior=False,
        gradif_decomposition="gaussian_fixed",
        gradif_active_stages=("c2", "c3", "c4_semantic"),
        gradif_fft_initial_boundaries=(0.25, 0.55),
        gradif_fft_mask_temperature=0.08,
        gradif_fft_min_bandwidth=0.05,
        num_modalities=5,
        ultrasound_modality_id=None,
        ultrasound_adapter_alpha_init=0.10,
        medclip_path=None,
        class_names=None,
        seg_num_classes=1,
        seg_img_size=224,
        det_num_classes=1,
        dino_repo_path=None,
        dino_cfg_path=None,
        dino_pretrained_path=None,
        use_official_dino=True,
        use_cross_roi_reasoning=True,
        llm_model_name="meta-llama/Llama-3.2-1B-Instruct",
        use_gt_descriptors=False,   # NEW: specialist mode -> use GT mask/measurements/birads
        birads_bias_init=None,      # CORAL thresholds aligned to sampled/weighted train distribution
        pathology_bias_init=None,  # two-class softmax bias aligned to sampled/weighted train distribution
        shared_head_seed=1337,
        gradif_seg_safe_branch=True,
        segmentation_gradif_fusion_init=0.85,
        segmentation_gradif_max_delta_ratios=(0.05, 0.04, 0.05),
        classification_gradif_fusion_init=0.25,
        classification_gradif_max_delta_ratio=0.06,
    ):
        super().__init__()
        self.swin = swin_model
        self.use_cross_roi_reasoning = bool(use_cross_roi_reasoning)
        # When True (specialist mode), the BUS-BRA descriptor block uses GT
        # measurements / GT-mask ROI / GT BI-RADS at eval too, because the seg
        # head is NOT trained in a classification specialist and predicted
        # masks would corrupt measurements -> meaningless cls metrics.
        self.use_gt_descriptors = bool(use_gt_descriptors)
        # Unified training gradually transitions from GT morphology/ROI masks to
        # deployment-aligned predicted descriptors.  A permanent GT path lets
        # descriptor heads bypass the visual GraDiF representation and creates
        # a large train/eval mismatch.
        self.descriptor_gt_probability = 0.0

        if medclip_path and os.path.exists(medclip_path):
            self.swin.load_medclip_weights(medclip_path)

        self.vision_proj = None

        # 1) Segmentation decoder
        self.seg_decoder = SwinUnetDecoderSys(
            embed_dim=96,
            num_classes=seg_num_classes,
            img_size=int(seg_img_size),
        )

        self.use_moe = use_moe
        self.moe_type = moe_type.lower() if use_moe else None
        self.num_modalities = num_modalities
        self.ultrasound_modality_id = (
            None if ultrasound_modality_id is None else int(ultrasound_modality_id)
        )
        # Derive hidden size from registry; falls back to 2048 if unknown
        llama_hidden_size = _LLM_HIDDEN_SIZES.get(llm_model_name, 2048)
        self.llm_model_name = llm_model_name

        # 2) Final paper task adaptation: MoFX-Dense only.
        if not self.use_moe or self.moe_type != "gradif":
            raise ValueError("The public model supports MoFX-Dense only.")
        self.task_moe = MoFXDenseHierarchy(
            num_tasks=4,
            num_modalities=num_modalities,
            attn_dim=int(gradif_attn_dim),
            num_heads=int(gradif_num_heads),
            router_dim=int(gradif_router_dim),
            top_k=int(gradif_top_k),
            routing_mode=str(gradif_routing_mode),
            expert_bank_type=str(gradif_expert_bank_type),
            temperature=float(gradif_temperature),
            modality_scale=float(gradif_modality_scale),
            router_strategy=str(gradif_router_strategy),
            router_task_residual_scale=float(gradif_router_task_residual_scale),
            router_modality_residual_scale=float(gradif_router_modality_residual_scale),
            router_attention_pool=bool(gradif_router_attention_pool),
            router_alpha_content_init=float(gradif_router_alpha_content_init),
            router_alpha_task_init=float(gradif_router_alpha_task_init),
            router_alpha_modality_init=float(gradif_router_alpha_modality_init),
            router_task_specific_gates=bool(gradif_router_task_specific_gates),
            router_modality_specific_gates=bool(gradif_router_modality_specific_gates),
            router_modality_content_adapters=bool(gradif_router_modality_content_adapters),
            router_branch_hidden=int(gradif_router_branch_hidden),
            router_configuration=str(mofx_router_configuration),
            router_mi_min_group_size=int(gradif_router_mi_min_group_size),
            router_mi_queue_size=int(gradif_router_mi_queue_size),
            router_mi_current_fraction=float(gradif_router_mi_current_fraction),
            use_granularity_prior=bool(gradif_use_granularity_prior),
            gamma_init=float(gradif_gamma_init),
            gamma_min=float(gradif_gamma_min),
            gamma_max=float(gradif_gamma_max),
            expert_scale_init=float(gradif_expert_scale_init),
            expert_scale_max=float(gradif_expert_scale_max),
            generic_mlp_bottleneck=int(gradif_generic_mlp_bottleneck),
            include_generalist=bool(gradif_include_generalist),
            include_null=bool(gradif_include_null_expert),
            decomposition=gradif_decomposition,
            active_stages=tuple(gradif_active_stages),
            fft_initial_boundaries=gradif_fft_initial_boundaries,
            fft_mask_temperature=float(gradif_fft_mask_temperature),
            fft_min_bandwidth=float(gradif_fft_min_bandwidth),
            ultrasound_modality_id=self.ultrasound_modality_id,
        )
        print(
            f"[MoFX-Dense] experts={self.task_moe.banks['c3'].expert_names}; "
            f"routing={self.task_moe.routing_mode}; "
            f"stages={list(self.task_moe.active_stages)}"
        )

        # Keep shared downstream-head initialization reproducible across seeds.
        if shared_head_seed is not None:
            torch.manual_seed(int(shared_head_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(shared_head_seed))

        # 3) Report modules
        self.visual_adapter = QFormerProjector(visual_input_dim=768, llm_hidden_dim=llama_hidden_size)
        self.report_decoder = PretrainedReportDecoder(
            llm_model_name=llm_model_name,
            visual_dim=llama_hidden_size,
        )
        # Compile the LLM decoder for faster forward+backward (dynamic=True handles
        # variable sequence lengths; mode=reduce-overhead is safe with LoRA)
        # try:
        #     self.report_decoder.decoder = torch.compile(
        #         self.report_decoder.decoder,
        #         mode="reduce-overhead",
        #         dynamic=True,
        #     )
        #     print(f"[ReportDecoder] torch.compile applied to {llm_model_name}")
        # except Exception as _ce:
        #     print(f"[ReportDecoder] torch.compile skipped ({_ce})")

        self.seg_guided_report = self.use_cross_roi_reasoning
        self.seg_report_detach = True
        self.seg_report_use_pred = True
        self.seg_report_token_extractor = SegmentationReportTokenExtractor(
            feat_dim=768,
            llm_dim=llama_hidden_size,
            mask_size=14,
        )
        self.null_report_tokens = nn.Parameter(torch.zeros(1, 2, llama_hidden_size))

        # 4) Generic classification modules (non-MRI datasets)
        self.attn_pool = AttentionPooling(in_dim=768)
        self.roi_pool = AttentionPooling(in_dim=768)
        self.global_pool = AttentionPooling(in_dim=768)

        self.seg_guided_cls = self.use_cross_roi_reasoning
        self.seg_guidance_detach = True
        self.seg_fusion_alpha = 1.0

        self.cls_fuse = nn.Sequential(
            nn.LayerNorm(768 * 2),
            nn.Linear(768 * 2, 768),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.cls_neck = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.35),   # was 0.2 — increased to reduce train/val AUC gap
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.35),   # was 0.2
        )
        self.cls_head = nn.Linear(512, len(class_names))

        # Ultrasound-only semantic residual adapter.  It is applied after the
        # shared GraDiF path and before BI-RADS/pathology heads, leaving X-ray
        # and other modalities unchanged.
        # MoFX-Dense uses the shared routed features without a separate
        # ultrasound residual adapter.
        # The C4 semantic frequency/generic block is now the only ultrasound
        # feature adapter, eliminating raw_ultrasound_adapter_alpha and its
        # recurrent non-finite scalar gradients.
        self.ultrasound_adapter = nn.Identity()
        self.register_buffer(
            "fixed_ultrasound_adapter_alpha",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.ultrasound_adapter_alpha_max = 0.0

        # ---- BUS-BRA descriptor heads (added; used only when wired in forward) ----
        self.birads_num_classes = 4
        self.measurement_encoder = MeasurementEncoder(out_dim=128, n_feat=N_MEAS_BUSBRA)
        self.mask_roi_pool = MaskROIPool()  # retained for checkpoint/API compatibility

        # Dedicated ultrasound pooling modules keep BUS-BRA gradients separate
        # from the generic multi-label classifier's pooling query.
        self.ultrasound_full_pool = AttentionPooling(in_dim=768)
        self.ultrasound_crop_pool = AttentionPooling(in_dim=768)
        self.ultrasound_crop_context = 0.40

        # The block-level residual gate is the primary identity path. A second,
        # lightweight spatial safety gate is retained for segmentation so the
        # decoder can fall back to untouched C2/C3/C4 features when needed.
        self.classification_gradif_fusion_gate = nn.Identity()
        self.classification_gradif_max_delta_ratio = min(
            float(classification_gradif_max_delta_ratio), 0.08
        )
        self.gradif_seg_safe_branch = bool(gradif_seg_safe_branch)
        _seg_init = min(max(float(segmentation_gradif_fusion_init), 1e-4), 1.0 - 1e-4)
        self.segmentation_gradif_fusion_logits = nn.Parameter(
            torch.full(
                (3,), math.log(_seg_init / (1.0 - _seg_init)),
                dtype=torch.float32,
            )
        )
        self.segmentation_gradif_max_delta_ratios = tuple(
            min(max(float(value), 0.0), 0.05)
            for value in segmentation_gradif_max_delta_ratios
        )

        # Validation-calibrated cumulative CORAL thresholds.  They are buffers
        # so the selected best_model.pth carries the decoding calibration.
        self.register_buffer(
            "birads_decode_thresholds",
            torch.full((self.birads_num_classes - 1,), 0.5, dtype=torch.float32),
        )

        # BI-RADS uses full-field and lesion-crop visual evidence only.
        self.birads_head = BiRadsOrdinalHead(
            global_dim=768,
            roi_dim=768,
            num_classes=self.birads_num_classes,
            bias_init=birads_bias_init,
        )

        # Pathology uses the same two visual views and may additionally use
        # full-field morphology computed from the selected GT/predicted mask.
        self.pathology_visual_neck = nn.Sequential(
            nn.LayerNorm(768 * 2),
            nn.Linear(768 * 2, 768),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(p=0.15),
        )
        self.pathology_meas_head = nn.Sequential(
            nn.LayerNorm(512 + 128),
            nn.Dropout(p=0.35),
            nn.Linear(512 + 128, 2),
        )   # dedicated BUS-BRA benign/malignant head
        if pathology_bias_init is not None:
            _pb = pathology_bias_init.detach().clone().float().view(-1)
            if _pb.numel() != 2:
                raise ValueError(
                    f"pathology_bias_init has {_pb.numel()} entries; expected 2"
                )
            with torch.no_grad():
                self.pathology_meas_head[-1].bias.copy_(_pb)

        # Projects BUS-BRA descriptor signals into one LLM-dim visual token.
        # Input = meas_emb(128) + birads cumulative-logits(K-1=3) + pathology_logits(2) = 133.
        # For non-BUS-BRA samples the inputs are zeroed via has_birads gating → near-null token.
        _busbra_desc_in = 128 + (self.birads_num_classes - 1) + 2   # 133
        self.busbra_desc_proj = nn.Sequential(
            nn.LayerNorm(_busbra_desc_in),
            nn.Linear(_busbra_desc_in, llama_hidden_size),
            nn.GELU(),
            nn.LayerNorm(llama_hidden_size),
        )  # → [B, llama_hidden_size]; unsqueeze(1) = [B, 1, llama_hidden_size]

        # 5) Detection modules
        self.det_num_classes = det_num_classes
        self.det_num_decoder_layers = 6
        self.det_num_feature_levels = 4
        self.use_official_dino = use_official_dino
        self.dino_repo_path = dino_repo_path
        self.dino_cfg_path = dino_cfg_path
        self.dino_pretrained_path = dino_pretrained_path

        self.seg_guided_det = False
        self.seg_det_detach = True

        # NOTE: det_proj layers are kept for checkpoint compatibility but
        # are NOT used in forward() — official DINO has its own input_proj.
        self.det_proj_c2 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.GroupNorm(32, 256),
        )
        self.det_proj_c3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1),
            nn.GroupNorm(32, 256),
        )
        self.det_proj_c4 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.GroupNorm(32, 256),
        )
        self.det_proj_p5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
        )

        if self.use_official_dino:
            self.detector = SharedBackboneOfficialDINO(
                repo_path=dino_repo_path,
                cfg_path=dino_cfg_path,
                num_classes=det_num_classes,
                feature_channels=(192, 384, 768),
                pretrained_path=dino_pretrained_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.detector = DINOHead(
                in_channels=(256, 256, 256, 256),
                hidden_dim=256,
                num_classes=det_num_classes,
                num_decoder_layers=6,
                num_heads=8,
                num_points=4,
                dropout=0.0,
                dn_number=100,
                dn_box_noise_scale=0.4,
                dn_label_noise_ratio=0.5,
                dn_labelbook_size=91,
                return_query_feats=True,
            )

        # 7) MRI report-token modules
        self.mri_level_embed = nn.Embedding(det_num_classes, llama_hidden_size)
        self.mri_severity_proj = nn.Sequential(
            nn.Linear(4, 256),
            nn.GELU(),
            nn.Linear(256, llama_hidden_size),
        )
        self.mri_roi_to_report = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, llama_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.mri_global_to_report = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, llama_hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.mri_null_level_token = nn.Parameter(torch.zeros(1, 1, llama_hidden_size))
        nn.init.normal_(self.mri_null_level_token, std=0.02)

        # GraDiF-only pathology no-harm gate.  It is deliberately created after
        # all original v16 modules so every pre-existing parameter keeps the
        # exact same seeded initialization.  The gate cannot change routing or
        # shared features: it only makes a small, sample-dependent late blend
        # from the raw GraDiF pathology logits toward the identity-path logits.
        self.pathology_identity_mix_max = 0.35
        if self.moe_type == "gradif":
            self.pathology_identity_gate = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, 8),
                nn.GELU(),
                nn.Linear(8, 1),
            )
            nn.init.normal_(
                self.pathology_identity_gate[-1].weight,
                mean=0.0, std=1e-3,
            )
            # Initial identity mixture ~= 5%, so the run starts very close to
            # the successful v16 GraDiF pathway rather than replacing it.
            _initial_fraction = 0.05 / self.pathology_identity_mix_max
            _initial_fraction = min(max(_initial_fraction, 1e-4), 1.0 - 1e-4)
            nn.init.constant_(
                self.pathology_identity_gate[-1].bias,
                math.log(_initial_fraction / (1.0 - _initial_fraction)),
            )

    def set_descriptor_gt_probability(self, value):
        self.descriptor_gt_probability = float(min(max(value, 0.0), 1.0))

    def effective_ultrasound_adapter_alpha(self):
        return self.fixed_ultrasound_adapter_alpha

    @staticmethod
    def _limit_residual_ratio(base_feature, adapted_feature, max_ratio):
        """Keep an adaptation inside a feature-relative RMS trust region."""
        if max_ratio is None or float(max_ratio) <= 0:
            return base_feature
        delta = adapted_feature - base_feature
        base_rms = base_feature.float().square().mean(
            dim=(1, 2, 3), keepdim=True
        ).add(1e-12).sqrt()
        delta_rms = delta.float().square().mean(
            dim=(1, 2, 3), keepdim=True
        ).add(1e-12).sqrt()
        allowed = float(max_ratio) * base_rms
        scale = torch.clamp(allowed / delta_rms, max=1.0).to(delta.dtype)
        return base_feature + delta * scale

    def _fuse_classification_features(self, base_feat, gradif_feat):
        """Use the already bounded semantic GraDiF feature directly."""
        gate = torch.ones(
            (base_feat.shape[0], 1),
            device=base_feat.device,
            dtype=base_feat.dtype,
        )
        return gradif_feat, gate

    def _pathology_logits_without_dropout(self, global_feat, roi_feat, meas_feat):
        """Evaluate the existing pathology head while skipping dropout only.

        No new classifier is introduced.  This deterministic view is used only
        to estimate whether the identity or GraDiF feature path is safer for a
        sample.  The returned logits are detached before calibration.
        """
        feat = torch.cat([global_feat, roi_feat], dim=-1)
        feat = self.pathology_visual_neck[0](feat)  # LayerNorm
        feat = self.pathology_visual_neck[1](feat)  # Linear
        feat = self.pathology_visual_neck[2](feat)  # GELU
        # [3] is Dropout and is intentionally skipped.
        feat = self.pathology_visual_neck[4](feat)  # Linear
        feat = self.pathology_visual_neck[5](feat)  # GELU
        # [6] is Dropout and is intentionally skipped.
        feat = torch.cat([feat, meas_feat], dim=-1)
        feat = self.pathology_meas_head[0](feat)    # LayerNorm
        # [1] is Dropout and is intentionally skipped.
        return self.pathology_meas_head[2](feat)   # Linear -> two logits

    def _fuse_spatial_gradif(self, base_feat, gradif_feat, stage_index):
        """Safely blend a bounded spatial residual with an exact identity path."""
        if not self.gradif_seg_safe_branch:
            gate = torch.ones(
                (base_feat.shape[0], 1, 1, 1),
                device=base_feat.device, dtype=base_feat.dtype,
            )
            return gradif_feat, gate
        index = min(max(int(stage_index), 0), 2)
        bounded = self._limit_residual_ratio(
            base_feat, gradif_feat,
            self.segmentation_gradif_max_delta_ratios[index],
        )
        gate_scalar = torch.sigmoid(
            self.segmentation_gradif_fusion_logits[index]
        ).to(dtype=base_feat.dtype, device=base_feat.device)
        gate = gate_scalar.view(1, 1, 1, 1).expand(
            base_feat.shape[0], 1, 1, 1
        )
        fused = base_feat + gate * (bounded - base_feat)
        return fused, gate

    def _mask_context_rois(self, mask, active_idx, image_h, image_w, context_ratio=0.40):
        """Build ROIAlign boxes from a full-resolution binary lesion mask.

        Empty masks fall back to the complete image. Box construction is
        deliberately non-differentiable; classification gradients flow through
        the cropped pixels and the shared encoder, not through thresholded mask
        coordinates.
        """
        rois = []
        valid = []
        mask_f = mask.detach().float() if mask is not None else None
        for batch_index in active_idx.tolist():
            has_roi = False
            if mask_f is not None and batch_index < mask_f.shape[0]:
                m = mask_f[batch_index]
                if m.dim() == 3:
                    m = m[0]
                m = m > 0.5
                coords = torch.nonzero(m, as_tuple=False)
            else:
                coords = torch.empty((0, 2), device=active_idx.device, dtype=torch.long)

            if coords.numel() > 0:
                y1 = float(coords[:, 0].min().item())
                y2 = float(coords[:, 0].max().item() + 1)
                x1 = float(coords[:, 1].min().item())
                x2 = float(coords[:, 1].max().item() + 1)
                bw = max(x2 - x1, 1.0)
                bh = max(y2 - y1, 1.0)
                x1 = max(0.0, x1 - context_ratio * bw)
                x2 = min(float(image_w), x2 + context_ratio * bw)
                y1 = max(0.0, y1 - context_ratio * bh)
                y2 = min(float(image_h), y2 + context_ratio * bh)
                has_roi = (x2 - x1 >= 2.0) and (y2 - y1 >= 2.0)
            else:
                x1, y1, x2, y2 = 0.0, 0.0, float(image_w), float(image_h)

            if not has_roi:
                x1, y1, x2, y2 = 0.0, 0.0, float(image_w), float(image_h)

            rois.append([float(batch_index), x1, y1, x2, y2])
            valid.append(has_roi)

        if not rois:
            return None, torch.empty(0, dtype=torch.bool, device=active_idx.device)
        return (
            torch.tensor(rois, dtype=torch.float32, device=active_idx.device),
            torch.tensor(valid, dtype=torch.bool, device=active_idx.device),
        )

    def _ultrasound_crop_features(self, x, roi_mask, active_idx):
        """Encode lesion crops with the same Swin backbone used for full images."""
        if active_idx.numel() == 0:
            return None, None, None

        rois, crop_valid = self._mask_context_rois(
            roi_mask,
            active_idx,
            image_h=x.shape[-2],
            image_w=x.shape[-1],
            context_ratio=self.ultrasound_crop_context,
        )
        if rois is None:
            return None, None, None

        # ROIAlign on normalized full-field inputs preserves the exact scanner
        # intensity transform seen by the primary stream. FP32 is used because
        # some torchvision builds do not support BF16/FP16 ROIAlign on CUDA.
        crop_images = roi_align(
            x.float(),
            rois,
            output_size=(x.shape[-2], x.shape[-1]),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        ).to(dtype=x.dtype)

        _, crop_hierarchy, _ = self.swin.forward_features(crop_images)
        crop_c4_tokens, crop_c4_res = crop_hierarchy[-1]
        hc, wc = crop_c4_res
        crop_c4 = crop_c4_tokens.transpose(1, 2).reshape(
            crop_c4_tokens.shape[0], crop_c4_tokens.shape[-1], hc, wc
        )

        # The crop is a BUS-BRA semantic view, so apply the same ultrasound
        # residual adapter. GraDiF routing is not called a second time, avoiding
        # corruption of the main-pass routing diagnostics.
        # No separate ultrasound adapter is used in MoFX-Dense.
        crop_feat = self.ultrasound_crop_pool(crop_c4)
        return crop_feat, crop_valid, crop_c4

    # ==================================================================
    # FORWARD
    # ==================================================================
    def forward(
        self,
        x,
        modality_ids=None,
        task_mask=None,
        boxes=None,
        det_targets=None,
        captions=None,
        report_labels=None,
        teacher_forcing_ratio=0,
        use_pred_boxes_for_mri_cls=False,
        det_score_thresh=0.3,
        return_feats=False,
        is_lss_mri=None,
        measurements=None,          # [B, N_MEAS] GT (train) -- predicted-mask filled in forward (eval)
        measurements_present=None,  # [B] bool
        measurement_sizes=None,     # list[(W,H)] source BUS-BRA mask sizes for report measurements
        busbra_seg_channel_indices=None,  # list[tuple[int]] global BUS-BRA lesion channels
        has_birads=None,            # [B] bool (BUS-BRA BI-RADS supervision)
        has_pathology=None,         # [B] bool (BUS-BRA pathology supervision)
        seg_mask_gt=None,           # [B,*,H,W] GT seg mask for train-time ROI pooling
        gt_birads_idx=None,         # [B] long GT BI-RADS class idx (specialist report mode)
        gt_pathology_idx=None,      # [B] long GT pathology class idx (specialist report mode)
        compute_gradif_reference=False,  # expensive identity-reference heads/decoder
    ):
        if task_mask is None:
            task_mask = torch.ones((x.shape[0], 4), device=x.device)

        cross_roi = bool(getattr(self, "use_cross_roi_reasoning", True))
        if not cross_roi:
            use_pred_boxes_for_mri_cls = False

        # ==============================================================
        # 1. Backbone (autocast enabled — runs in FP16 for speed)
        # ==============================================================
        final_x, features, final_res = self.swin.forward_features(x)

        (c1_feat, c1_res), (c2_feat, c2_res), (c3_feat, c3_res), (c4_feat, c4_res) = features

        B, L, C = c4_feat.shape
        H, W = c4_res
        c2_spatial = self.tokens_to_map(c2_feat, c2_res)
        c3_spatial = self.tokens_to_map(c3_feat, c3_res)
        c4_spatial = c4_feat.transpose(1, 2).reshape(B, C, H, W)

        if is_lss_mri is None:
            is_lss_mri = torch.zeros(B, dtype=torch.bool, device=x.device)
        else:
            is_lss_mri = is_lss_mri.to(x.device, non_blocking=True)

        # ==============================================================
        # 2. Task masks / indices (moved BEFORE MoE so we can skip unused tasks)
        # ==============================================================
        seg_mask = (task_mask[:, 0] > 0.5)
        raw_cls_mask = (task_mask[:, 1] > 0.5) & (~is_lss_mri)
        rep_idx  = (task_mask[:, 2] > 0.5)
        det_mask = (task_mask[:, 3] > 0.5) & is_lss_mri

        if has_birads is None:
            birads_route_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        else:
            birads_route_mask = has_birads.to(x.device, dtype=torch.bool) & (~is_lss_mri)
        if has_pathology is None:
            if gt_pathology_idx is not None:
                pathology_route_mask = (
                    gt_pathology_idx.to(x.device).long() >= 0
                ) & (~is_lss_mri)
            else:
                pathology_route_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        else:
            pathology_route_mask = has_pathology.to(
                x.device, dtype=torch.bool
            ) & (~is_lss_mri)

        # BUS-BRA uses dedicated BI-RADS/pathology heads only.  Its generic
        # benign/malignant duplicate target is explicitly excluded here.
        dedicated_us_mask = birads_route_mask | pathology_route_mask
        cls_mask = raw_cls_mask & (~dedicated_us_mask)
        cls_route_mask = cls_mask | dedicated_us_mask

        has_seg_task = bool(seg_mask.any().item())
        has_cls_task = bool(cls_mask.any().item())
        has_cls_route = bool(cls_route_mask.any().item())
        has_rep_task = bool(rep_idx.any().item())
        has_det_task = bool(det_mask.any().item())

        rep_batch_idx = rep_idx.nonzero(as_tuple=True)[0]
        det_batch_idx = det_mask.nonzero(as_tuple=True)[0]

        if cross_roi:
            use_seg_guidance_cls    = seg_mask & cls_mask
            use_seg_guidance_report = seg_mask & rep_idx & (~is_lss_mri)
        else:
            use_seg_guidance_cls    = torch.zeros_like(seg_mask, dtype=torch.bool)
            use_seg_guidance_report = torch.zeros_like(seg_mask, dtype=torch.bool)

        # Seg features are needed for: seg task itself + cls guidance + report guidance
        seg_route_mask = seg_mask | use_seg_guidance_cls | use_seg_guidance_report
        need_seg_feats = bool(seg_route_mask.any().item())

        store_vis = (not self.training) or return_feats

        # ==============================================================
        # 3. Task-specific feature reconciliation
        # ==============================================================
        # Default identity paths. MoFX-Dense adapts c2/c3 for spatial tasks and
        # c4 for every task.
        c2_seg = c2_spatial
        c2_det = c2_spatial
        c3_seg = c3_spatial
        c3_det = c3_spatial
        c4_seg = c4_cls = c4_rep = c4_det = c4_spatial
        c4_birads = c4_pathology = c4_spatial
        # Preserve identity paths for no-harm semantic and spatial fusion.
        c2_seg_base = c2_spatial
        c3_seg_base = c3_spatial
        c4_seg_base = c4_spatial
        c4_cls_base = c4_spatial
        c4_birads_base = c4_spatial
        c4_pathology_base = c4_spatial
        modality_logits = None
        gradif_summary = None
        gradif_task_weights = {}
        gradif_band_energy = None

        if self.use_moe and self.task_moe is not None:
            if modality_ids is None:
                _mod_ids = torch.zeros(B, device=c4_spatial.device, dtype=torch.long)
            elif isinstance(modality_ids, torch.Tensor):
                _mod_ids = modality_ids.to(c4_spatial.device, dtype=torch.long)
            else:
                _mod_ids = torch.as_tensor(
                    modality_ids,
                    device=c4_spatial.device,
                    dtype=torch.long,
                )

            task_active_masks = {
                0: seg_route_mask,
                3: det_mask,
                "classification": cls_mask,
                "birads": birads_route_mask,
                "pathology": pathology_route_mask,
                "report": rep_idx,
            }

            if self.moe_type == "gradif":
                # Direct stage-wise GraDiF. c2/c3 are used only by spatial
                # heads, while c4 is routed for every active task. This avoids
                # both unused semantic c3 routes and approximate c3->c4 lifts.
                spatial_tasks = []
                if need_seg_feats:
                    spatial_tasks.append(0)
                if has_det_task:
                    spatial_tasks.append(3)

                c4_tasks = []
                if has_cls_task:
                    c4_tasks.append("classification")
                if bool(birads_route_mask.any().item()):
                    c4_tasks.append("birads")
                if bool(pathology_route_mask.any().item()):
                    c4_tasks.append("pathology")
                if has_rep_task:
                    c4_tasks.append("report")

                c2_out = self.task_moe.forward_stage(
                    "c2",
                    c2_spatial,
                    task_ids=spatial_tasks,
                    modality_ids=_mod_ids,
                    task_active_masks=task_active_masks,
                )
                c3_out = self.task_moe.forward_stage(
                    "c3",
                    c3_spatial,
                    task_ids=spatial_tasks,
                    modality_ids=_mod_ids,
                    task_active_masks=task_active_masks,
                )
                c4_out = self.task_moe.forward_stage(
                    "c4",
                    c4_spatial,
                    task_ids=c4_tasks,
                    modality_ids=_mod_ids,
                    task_active_masks=task_active_masks,
                )

                if need_seg_feats:
                    _c2_seg_raw = c2_out.get(0, c2_spatial)
                    _c3_seg_raw = c3_out.get(0, c3_spatial)
                    _c4_seg_raw = c4_out.get(0, c4_spatial)
                    c2_seg, _seg_gate_c2 = self._fuse_spatial_gradif(
                        c2_seg_base, _c2_seg_raw, 0
                    )
                    c3_seg, _seg_gate_c3 = self._fuse_spatial_gradif(
                        c3_seg_base, _c3_seg_raw, 1
                    )
                    c4_seg, _seg_gate_c4 = self._fuse_spatial_gradif(
                        c4_seg_base, _c4_seg_raw, 2
                    )
                if has_cls_task:
                    c4_cls = self._limit_residual_ratio(
                        c4_cls_base,
                        c4_out.get("classification", c4_spatial),
                        self.classification_gradif_max_delta_ratio,
                    )
                if bool(birads_route_mask.any().item()):
                    c4_birads = self._limit_residual_ratio(
                        c4_birads_base,
                        c4_out.get("birads", c4_spatial),
                        self.classification_gradif_max_delta_ratio,
                    )
                if bool(pathology_route_mask.any().item()):
                    c4_pathology = self._limit_residual_ratio(
                        c4_pathology_base,
                        c4_out.get("pathology", c4_spatial),
                        self.classification_gradif_max_delta_ratio,
                    )
                if has_rep_task:
                    c4_rep = c4_out.get("report", c4_spatial)
                if has_det_task:
                    c2_det = c2_out.get(3, c2_spatial)
                    c3_det = c3_out.get(3, c3_spatial)
                    c4_det = c4_out.get(3, c4_spatial)

                gradif_summary = self.task_moe.routing_summary()
                gradif_task_weights = self.task_moe.latest_weights_by_stage
                gradif_band_energy = self.task_moe.latest_band_energies_by_stage

        # ==============================================================
        # Output dict
        # ==============================================================
        outputs = {
            # Segmentation
            "pred_masks":           None,
            "pred_masks_small":     None,
            "seg_pred_batch_idx":   None,
            "use_seg_guidance":     use_seg_guidance_cls,
            # Classification
            "class_logits":         None,
            "birads_logits":        None,
            "pathology_meas_logits": None,
            "measurements_used":    None,
            "measurements_present_used": None,
            # Source-mask-space measurements used only for BUS-BRA free-text reports.
            "report_measurements_used": None,
            "report_measurements_present_used": None,
            # Exact source-space BUS-BRA morphology supplied to the report prompt.
            "report_morphology_used": None,
            "report_source_masks": None,
            "report_measurement_sizes_used": None,
            # Report
            "rep_loss":             None,
            "report_logits":        None,
            "report_tokens":        None,
            "visual_tokens":        None,
            "seg_report_tokens":    None,
            "use_seg_report":       use_seg_guidance_report,
            "report_batch_idx":     rep_batch_idx if rep_batch_idx.numel() > 0 else None,
            "mri_report_struct":    [None for _ in range(B)] if store_vis else None,
            # Seg attention maps
            "seg_attn_full":        None,
            "seg_attn_c4cls":       None,
            "seg_attn_c2det":       None,
            "seg_attn_c3det":       None,
            "seg_attn_c4det":       None,
            # Detection
            "detection":            None,
            "detection_active":     None,
            "pred_logits":          None,
            "pred_boxes":           None,
            "det_query_feats":      None,
            "det_query_feats_active": None,
            "det_batch_idx":        det_batch_idx if det_batch_idx.numel() > 0 else None,
            # MRI ROI
            "mri_roi_logits":       None,
            "mri_roi_batch_idx":    None,
            "mri_roi_box_idx":      None,
            "mri_roi_boxes":        None,
            "mri_roi_level_logits": None,
            "mri_pred_boxes":       None,
            "mri_pred_labels":      None,
            "mri_pred_scores":      None,
            "mri_pred_query_ids":   None,
            "mri_roi_base_feats":   None,
            "modality_logits":      modality_logits,
            # GraDiF-MoE diagnostics
            "gradif_summary": gradif_summary,
            "gradif_task_weights": gradif_task_weights,
            "gradif_band_energy": gradif_band_energy,
        }
        if self.moe_type == "gradif" and need_seg_feats:
            outputs["segmentation_gradif_stage_gates"] = torch.stack([
                _seg_gate_c2, _seg_gate_c3, _seg_gate_c4
            ])

        # ==============================================================
        # TASK 0: SEGMENTATION
        # ==============================================================
        seg_need_mask = seg_mask.clone()
        if cross_roi and self.seg_guided_cls:
            seg_need_mask = seg_need_mask | use_seg_guidance_cls
        if cross_roi and self.seg_guided_report:
            seg_need_mask = seg_need_mask | use_seg_guidance_report

        seg_needed_idx = seg_need_mask.nonzero(as_tuple=True)[0]
        need_seg_preds = seg_needed_idx.numel() > 0

        seg_pred_batch_idx = None
        pred_masks_small = None

        if need_seg_preds:
            seg_pred_batch_idx = seg_needed_idx

            c4_seg_flat = c4_seg[seg_needed_idx].flatten(2).transpose(1, 2)
            seg_input = [
                (c1_feat[seg_needed_idx], c1_res),
                (c2_seg[seg_needed_idx].flatten(2).transpose(1, 2), c2_res),
                (c3_seg[seg_needed_idx].flatten(2).transpose(1, 2), c3_res),
                (c4_seg_flat, c4_res),
            ]
            pred_masks_small = self.seg_decoder(seg_input)

            outputs["pred_masks_small"]   = pred_masks_small
            outputs["seg_pred_batch_idx"] = seg_pred_batch_idx

            # Same decoder, untouched C1-C4 features.  This branch is a
            # counterfactual reference for spatial GraDiF utility and the
            # training no-harm objective; it does not replace the main output.
            if (
                self.moe_type == "gradif"
                and self.gradif_seg_safe_branch
                and bool(compute_gradif_reference)
            ):
                seg_input_base = [
                    (c1_feat[seg_needed_idx], c1_res),
                    (c2_seg_base[seg_needed_idx].flatten(2).transpose(1, 2), c2_res),
                    (c3_seg_base[seg_needed_idx].flatten(2).transpose(1, 2), c3_res),
                    (c4_seg_base[seg_needed_idx].flatten(2).transpose(1, 2), c4_res),
                ]
                outputs["pred_masks_base_small"] = self.seg_decoder(seg_input_base)

            # BUS-BRA classification requires the predicted mask during
            # training whenever scheduled teacher forcing selects deployment-
            # aligned evidence. Previously this tensor was omitted for GraDiF
            # training, silently forcing a global-ROI/null-measurement fallback.
            need_full_pred_masks = (
                store_vis
                or bool(dedicated_us_mask.any().item())
            )
            if need_full_pred_masks:
                full_pred_masks = pred_masks_small.new_zeros(
                    (B, pred_masks_small.shape[1], pred_masks_small.shape[2], pred_masks_small.shape[3])
                )
                full_pred_masks[seg_needed_idx] = pred_masks_small
                outputs["pred_masks"] = full_pred_masks
                if outputs.get("pred_masks_base_small", None) is not None:
                    full_pred_masks_base = outputs["pred_masks_base_small"].new_zeros(
                        (B, outputs["pred_masks_base_small"].shape[1],
                         outputs["pred_masks_base_small"].shape[2],
                         outputs["pred_masks_base_small"].shape[3])
                    )
                    full_pred_masks_base[seg_needed_idx] = outputs["pred_masks_base_small"]
                    outputs["pred_masks_base"] = full_pred_masks_base

        # ----------------------------------------------------------
        # Shared segmentation attention map (non-MRI cls/report only)
        # ----------------------------------------------------------
        seg_attn_full = None
        need_seg_attn = (
            cross_roi
            and pred_masks_small is not None
            and (
                (self.seg_guided_cls    and bool(use_seg_guidance_cls.any().item()))
                or (self.seg_guided_report and bool(use_seg_guidance_report.any().item()))
            )
        )

        if need_seg_attn:
            seg_attn_small = self.build_seg_attention_map(pred_masks_small)
            seg_attn_full = seg_attn_small.new_zeros(
                (B, seg_attn_small.shape[1], seg_attn_small.shape[2], seg_attn_small.shape[3])
            )
            seg_attn_full[seg_pred_batch_idx] = seg_attn_small
            if store_vis:
                outputs["seg_attn_full"] = seg_attn_full

        # ==============================================================
        # BUS-BRA DUAL-VIEW CLASSIFICATION
        #   view 1: complete ultrasound image (c4_cls)
        #   view 2: lesion-centred crop encoded by the same Swin backbone
        # BI-RADS is visual-only. Pathology additionally receives full-field
        # morphology from the selected GT/predicted lesion mask.
        # ==============================================================
        ultrasound_semantic_mask = birads_route_mask | pathology_route_mask

        # Apply the same ultrasound-only semantic adapter to both identity and
        # GraDiF full-field streams.  This prevents a full/crop domain mismatch
        # and isolates the effect of GraDiF itself.
        if ultrasound_semantic_mask.any():
            outputs["ultrasound_adapter_alpha"] = (
                self.fixed_ultrasound_adapter_alpha.detach()
            )

        # Build a full-batch predicted lesion mask. For BUS-BRA samples whose
        # segmentation path was executed, outputs['pred_masks'] now exists even
        # during GraDiF training (see the segmentation block above).
        pm_full = outputs.get("pred_masks", None)
        pred_mask_full = None
        if pm_full is not None:
            pred_mask_full = (torch.sigmoid(pm_full[:, :1].float()) > 0.5).float()

        gt_mask_full = None
        if seg_mask_gt is not None:
            gt_mask_full = (seg_mask_gt[:, :1].to(x.device).float() > 0.5).float()

        # Per-sample teacher forcing avoids switching the entire heterogeneous
        # batch between GT and predicted evidence at once.
        gt_choice = torch.zeros(B, dtype=torch.bool, device=x.device)
        if ultrasound_semantic_mask.any() and gt_mask_full is not None:
            if self.use_gt_descriptors:
                gt_choice = ultrasound_semantic_mask.clone()
            elif self.training and float(self.descriptor_gt_probability) > 0.0:
                gt_choice = ultrasound_semantic_mask & (
                    torch.rand(B, device=x.device) < float(self.descriptor_gt_probability)
                )

        # Predicted evidence is the deployment path. If a mask is unavailable,
        # an empty mask makes the crop helper fall back to the full image.
        roi_mask_src = torch.zeros(
            (B, 1, x.shape[-2], x.shape[-1]),
            dtype=torch.float32,
            device=x.device,
        )
        if pred_mask_full is not None:
            roi_mask_src.copy_(pred_mask_full)
        if gt_mask_full is not None and gt_choice.any():
            roi_mask_src[gt_choice] = gt_mask_full[gt_choice]

        # Full-field measurements remain available only for pathology/reporting;
        # they are never passed to the BI-RADS head.
        meas_vec = torch.zeros(B, N_MEAS_BUSBRA, device=x.device)
        meas_pres = torch.zeros(B, dtype=torch.bool, device=x.device)
        for _b in ultrasound_semantic_mask.nonzero(as_tuple=True)[0].tolist():
            _v, _p = compute_mask_measurements(roi_mask_src[_b, 0])
            meas_vec[_b] = torch.from_numpy(_v).to(x.device)
            meas_pres[_b] = bool(_p)
        # Keep the learned classification/pathology descriptor path unchanged:
        # ``meas_vec`` remains in model-input space for checkpoint compatibility.
        # Report generation is separate: select the true BUS-BRA lesion channel(s),
        # resize the *probability map* to the original source-mask resolution, then
        # threshold and recompute every quantitative value used by the GT report.
        report_meas_vec = meas_vec.clone()
        report_meas_pres = meas_pres.clone()
        report_morphology_used = [None for _ in range(B)]
        report_source_masks = [None for _ in range(B)]
        report_measurement_sizes_used = [None for _ in range(B)]

        report_measurement_mask = ultrasound_semantic_mask.clone()
        if measurements_present is not None:
            report_measurement_mask = report_measurement_mask | (
                rep_idx
                & measurements_present.to(x.device, dtype=torch.bool)
                & (~is_lss_mri)
            )

        if (not self.training) and measurement_sizes is not None and pm_full is not None:
            for _b in report_measurement_mask.nonzero(as_tuple=True)[0].tolist():
                try:
                    _size = measurement_sizes[_b]
                    if torch.is_tensor(_size):
                        _size = _size.detach().cpu().tolist()
                    _mw, _mh = int(_size[0]), int(_size[1])
                    if _mw <= 0 or _mh <= 0:
                        continue

                    # Resolve BUS-BRA foreground channel(s) supplied by the data layer.
                    _channels = []
                    if busbra_seg_channel_indices is not None and _b < len(busbra_seg_channel_indices):
                        _raw_channels = busbra_seg_channel_indices[_b]
                        if torch.is_tensor(_raw_channels):
                            _raw_channels = _raw_channels.detach().cpu().reshape(-1).tolist()
                        elif isinstance(_raw_channels, (int, np.integer)):
                            _raw_channels = [int(_raw_channels)]
                        for _ch in (_raw_channels or []):
                            _ch = int(_ch)
                            if 0 <= _ch < int(pm_full.shape[1]):
                                _channels.append(_ch)
                    _channels = sorted(set(_channels))
                    if not _channels:
                        if int(pm_full.shape[1]) == 1:
                            _channels = [0]
                        else:
                            # Do not silently assume channel 0 in a multi-class decoder.
                            continue

                    # IMPORTANT: resize probability first, threshold second. This
                    # avoids geometry distortions from resizing an already-binarized
                    # 224x224 mask and puts prediction/GT in the same pixel space.
                    _prob = torch.sigmoid(pm_full[_b:_b + 1, _channels].float()).amax(
                        dim=1, keepdim=True
                    )
                    if tuple(_prob.shape[-2:]) != (_mh, _mw):
                        _prob = F.interpolate(
                            _prob, size=(_mh, _mw), mode="bilinear", align_corners=False
                        )
                    _report_mask = (_prob > 0.5).to(torch.uint8)

                    _morph, _rp = compute_report_morphology_gt_aligned(
                        _report_mask[0, 0]
                    )
                    if not _rp:
                        continue

                    _rv = np.array(
                        [_morph[k] for k in MEASUREMENT_KEYS_BUSBRA],
                        dtype=np.float32,
                    )
                    report_meas_vec[_b] = torch.from_numpy(_rv).to(x.device)
                    report_meas_pres[_b] = True
                    report_morphology_used[_b] = {
                        key: float(_morph[key]) for key in REPORT_MORPHOLOGY_KEYS_BUSBRA
                    }
                    report_source_masks[_b] = _report_mask[0, 0].detach().cpu()
                    report_measurement_sizes_used[_b] = (int(_mw), int(_mh))
                except Exception:
                    # Leave report morphology unavailable rather than silently mixing
                    # 224-space values with source-space GT measurements.
                    continue

        meas_emb = self.measurement_encoder(meas_vec, meas_pres)

        # Dedicated full-field streams use one semantic router conditioned by BI-RADS/pathology output identity.
        global_feat_birads_desc = c4_birads.new_zeros((B, 768))
        global_feat_pathology_desc = c4_pathology.new_zeros((B, 768))
        global_feat_base_desc = c4_spatial.new_zeros((B, 768))
        roi_feat_desc = c4_spatial.new_zeros((B, 768))
        crop_valid_full = torch.zeros(B, dtype=torch.bool, device=x.device)
        birads_gradif_gate_full = torch.zeros(
            B, 1, dtype=torch.float32, device=x.device
        )
        pathology_gradif_gate_full = torch.zeros(
            B, 1, dtype=torch.float32, device=x.device
        )
        crop_c4_active = None

        if ultrasound_semantic_mask.any():
            _us_idx = ultrasound_semantic_mask.nonzero(as_tuple=True)[0]
            _full_base_feat = self.ultrasound_full_pool(c4_spatial[_us_idx])
            global_feat_base_desc[_us_idx] = _full_base_feat
            _crop_feat, _crop_valid, crop_c4_active = self._ultrasound_crop_features(
                x,
                roi_mask_src,
                _us_idx,
            )
            if _crop_feat is None:
                _crop_feat = _full_base_feat
                _crop_valid = torch.zeros_like(_us_idx, dtype=torch.bool)
            roi_feat_desc[_us_idx] = _crop_feat
            crop_valid_full[_us_idx] = _crop_valid

        if birads_route_mask.any():
            _bi = birads_route_mask.nonzero(as_tuple=True)[0]
            _bi_base = self.ultrasound_full_pool(c4_birads_base[_bi])
            _bi_gradif = self.ultrasound_full_pool(c4_birads[_bi])
            _bi_feat, _bi_gate = self._fuse_classification_features(
                _bi_base, _bi_gradif
            )
            global_feat_birads_desc[_bi] = _bi_feat
            birads_gradif_gate_full[_bi] = _bi_gate.float()

        if pathology_route_mask.any():
            _pi = pathology_route_mask.nonzero(as_tuple=True)[0]
            _pi_base = self.ultrasound_full_pool(c4_pathology_base[_pi])
            _pi_gradif = self.ultrasound_full_pool(c4_pathology[_pi])
            _pi_feat, _pi_gate = self._fuse_classification_features(
                _pi_base, _pi_gradif
            )
            global_feat_pathology_desc[_pi] = _pi_feat
            pathology_gradif_gate_full[_pi] = _pi_gate.float()

        outputs["birads_gradif_gate"] = birads_gradif_gate_full
        outputs["pathology_gradif_gate"] = pathology_gradif_gate_full
        # BI-RADS: active BUS-BRA samples only; visual full + visual crop.
        # Keep ordinal logits in FP32.  Under outer AMP, c4_cls may be FP16/BF16
        # while the CORAL head can return FP32 (e.g. because its numerically
        # sensitive operations are evaluated outside autocast).  Allocating the
        # full-batch placeholder from c4_cls therefore causes an index-put dtype
        # mismatch.  A fixed FP32 destination also keeps BCE-with-logits stable.
        full_birads_logits = torch.zeros(
            (B, self.birads_num_classes - 1),
            device=x.device,
            dtype=torch.float32,
        )
        if birads_route_mask.any():
            _bi = birads_route_mask.nonzero(as_tuple=True)[0]
            _bi_logits = self.birads_head(
                global_feat_birads_desc[_bi],
                roi_feat_desc[_bi],
            ).float()
            full_birads_logits.index_copy_(0, _bi, _bi_logits)
        outputs["birads_logits"] = full_birads_logits
        if bool(compute_gradif_reference):
            full_birads_base_logits = torch.zeros_like(full_birads_logits)
            if birads_route_mask.any():
                _bi = birads_route_mask.nonzero(as_tuple=True)[0]
                _bi_base_logits = self.birads_head(
                    global_feat_base_desc[_bi],
                    roi_feat_desc[_bi],
                ).float()
                full_birads_base_logits.index_copy_(0, _bi, _bi_base_logits)
            outputs["birads_base_logits"] = full_birads_base_logits
        outputs["feat_c4_birads"] = c4_birads
        outputs["feat_c4_birads_base"] = c4_birads_base
        outputs["ultrasound_crop_valid"] = crop_valid_full
        if store_vis and crop_c4_active is not None:
            outputs["feat_c4_ultrasound_crop_active"] = crop_c4_active
        outputs["measurements_used"] = meas_vec
        outputs["measurements_present_used"] = meas_pres
        outputs["report_measurements_used"] = report_meas_vec
        outputs["report_measurements_present_used"] = report_meas_pres
        outputs["report_morphology_used"] = report_morphology_used
        outputs["report_source_masks"] = report_source_masks
        outputs["report_measurement_sizes_used"] = report_measurement_sizes_used

        # ==============================================================
        # TASK 1: GENERIC CLASSIFICATION (non-MRI only)
        # ==============================================================
        if has_cls_task:
            cls_idx = cls_mask.nonzero(as_tuple=True)[0]

            if cls_idx.numel() > 0:
                c4_cls_sel = c4_cls[cls_idx]
                c4_cls_base_sel = c4_cls_base[cls_idx]
                cls_feat_base_sel = self.attn_pool(c4_cls_base_sel)
                cls_feat_gradif_sel = self.attn_pool(c4_cls_sel)
                cls_feat_sel, cls_semantic_gate = self._fuse_classification_features(
                    cls_feat_base_sel, cls_feat_gradif_sel
                )
                outputs["generic_classification_gradif_gate"] = cls_semantic_gate

                use_seg_guidance_sel = use_seg_guidance_cls[cls_idx]

                if (cross_roi and self.seg_guided_cls
                        and use_seg_guidance_sel.any()
                        and seg_attn_full is not None):

                    seg_attn_cls = seg_attn_full.detach() if self.seg_guidance_detach else seg_attn_full
                    seg_attn_small_cls = F.interpolate(
                        seg_attn_cls[cls_idx],
                        size=c4_cls_sel.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).clamp(0.0, 1.0)

                    guided_local_idx = use_seg_guidance_sel.nonzero(as_tuple=True)[0]
                    c4_cls_guided = c4_cls_sel[guided_local_idx]
                    c4_cls_base_guided = c4_cls_base_sel[guided_local_idx]
                    seg_attn_guided = seg_attn_small_cls[guided_local_idx]

                    roi_map_base = c4_cls_base_guided * (
                        1.0 + self.seg_fusion_alpha * seg_attn_guided
                    )
                    roi_map_gradif = c4_cls_guided * (
                        1.0 + self.seg_fusion_alpha * seg_attn_guided
                    )
                    guided_base = self.cls_fuse(torch.cat([
                        self.global_pool(c4_cls_base_guided),
                        self.roi_pool(roi_map_base),
                    ], dim=1))
                    guided_gradif = self.cls_fuse(torch.cat([
                        self.global_pool(c4_cls_guided),
                        self.roi_pool(roi_map_gradif),
                    ], dim=1))
                    fused_feat, guided_gate = self._fuse_classification_features(
                        guided_base, guided_gradif
                    )

                    cls_feat_sel = cls_feat_sel.clone()
                    cls_feat_sel.index_copy_(0, guided_local_idx, fused_feat)
                    outputs["generic_guided_gradif_gate"] = guided_gate

                    if store_vis:
                        full_seg_attn_c4cls = seg_attn_small_cls.new_zeros(
                            (B, seg_attn_small_cls.shape[1],
                             seg_attn_small_cls.shape[2], seg_attn_small_cls.shape[3])
                        )
                        full_seg_attn_c4cls[cls_idx] = seg_attn_small_cls
                        outputs["seg_attn_c4cls"] = full_seg_attn_c4cls

                neck_feat_sel = self.cls_neck(cls_feat_sel)
                class_logits_sel = self.cls_head(neck_feat_sel)
                full_class_logits = class_logits_sel.new_zeros((B, class_logits_sel.shape[1]))
                full_class_logits[cls_idx] = class_logits_sel
                outputs["class_logits"] = full_class_logits

        # Dedicated pathology path, independent of the generic classifier and
        # independent of the BI-RADS supervision mask.
        if pathology_route_mask.any():
            path_idx = pathology_route_mask.nonzero(as_tuple=True)[0]
            path_visual = self.pathology_visual_neck(torch.cat([
                global_feat_pathology_desc[path_idx], roi_feat_desc[path_idx]
            ], dim=-1))
            path_logits_gradif = self.pathology_meas_head(torch.cat([
                path_visual, meas_emb[path_idx]
            ], dim=-1)).float()

            # Optional base-feature comparison for calibration: compare the routed
            # pathology pathway with its identity-path counterpart using the
            # SAME pathology head weights and no dropout.  All calibration
            # evidence and the correction vector are detached, so the ordinary
            # pathology gradient into the v16 GraDiF representation is exactly
            # preserved.  Only the tiny gate learns the late mixture.
            path_logits_sel = path_logits_gradif
            path_base_det = None
            identity_mix = torch.zeros(
                (path_idx.numel(), 1),
                device=x.device, dtype=torch.float32,
            )
            if self.moe_type == "gradif" and hasattr(self, "pathology_identity_gate"):
                with torch.no_grad():
                    path_gradif_det = self._pathology_logits_without_dropout(
                        global_feat_pathology_desc[path_idx].detach(),
                        roi_feat_desc[path_idx].detach(),
                        meas_emb[path_idx].detach(),
                    ).float()
                    path_base_det = self._pathology_logits_without_dropout(
                        global_feat_base_desc[path_idx].detach(),
                        roi_feat_desc[path_idx].detach(),
                        meas_emb[path_idx].detach(),
                    ).float()
                gate_input = torch.cat([
                    path_gradif_det,
                    path_base_det,
                    (path_gradif_det - path_base_det).abs(),
                ], dim=-1)
                identity_mix = self.pathology_identity_mix_max * torch.sigmoid(
                    self.pathology_identity_gate(gate_input)
                )
                path_logits_sel = path_logits_gradif + identity_mix * (
                    path_base_det - path_gradif_det
                )

            full_path_raw = torch.zeros(
                (B, 2), device=x.device, dtype=path_logits_gradif.dtype
            )
            full_path_raw.index_copy_(0, path_idx, path_logits_gradif)
            outputs["pathology_raw_gradif_logits"] = full_path_raw

            full_identity_mix = torch.zeros(
                (B, 1), device=x.device, dtype=torch.float32
            )
            full_identity_mix.index_copy_(0, path_idx, identity_mix.float())
            outputs["pathology_identity_mix_gate"] = full_identity_mix

            full_path_logits = torch.zeros(
                (B, 2), device=x.device, dtype=path_logits_sel.dtype
            )
            full_path_logits.index_copy_(0, path_idx, path_logits_sel)
            outputs["pathology_meas_logits"] = full_path_logits

            if bool(compute_gradif_reference):
                if path_base_det is None:
                    with torch.no_grad():
                        path_base_det = self._pathology_logits_without_dropout(
                            global_feat_base_desc[path_idx].detach(),
                            roi_feat_desc[path_idx].detach(),
                            meas_emb[path_idx].detach(),
                        ).float()
                full_path_base_logits = torch.zeros_like(full_path_logits)
                full_path_base_logits.index_copy_(0, path_idx, path_base_det)
                outputs["pathology_base_logits"] = full_path_base_logits
            outputs["pathology_batch_idx"] = path_idx

        # ==============================================================
        # TASK 2: REPORTING — non-MRI tokens first; MRI tokens later
        # ==============================================================
        non_mri_report_token_map = {}

        if has_rep_task:
            is_non_mri_rep = rep_idx & (~is_lss_mri)

            if is_non_mri_rep.any():
                non_mri_idx = is_non_mri_rep.nonzero(as_tuple=True)[0]
                c4_rep_sel  = c4_rep[non_mri_idx]
                use_seg_report_local = use_seg_guidance_report[non_mri_idx]

                # Base report tokens from full-image c4_rep
                c4_rep_flat = c4_rep_sel.flatten(2).transpose(1, 2)
                base_report_tokens = self.visual_adapter(
                    F.layer_norm(c4_rep_flat, c4_rep_flat.shape[-1:])
                )

                seg_report_tokens = None

                # Seg-guided report tokens (non-MRI only)
                if (cross_roi and self.seg_guided_report
                        and use_seg_report_local.any()
                        and seg_attn_full is not None):

                    seg_attn_rep = seg_attn_full[non_mri_idx]
                    if self.seg_report_detach:
                        seg_attn_rep = seg_attn_rep.detach()

                    seg_report_tokens = self.seg_report_token_extractor(c4_rep_sel, seg_attn_rep)
                    if store_vis:
                        outputs["seg_report_tokens"] = seg_report_tokens

                # Pre-compute BUS-BRA descriptor tokens for the whole non-MRI sub-batch.
                # Concatenate meas_emb + birads_logits + pathology_logits and project
                # to one LLM-dim token per sample.  For non-BUS-BRA samples the inputs
                # are zero-filled via has_birads gating, so the token is near-null (harmless).
                _birads_raw   = outputs.get('birads_logits', None)        # [B, K-1]
                _path_raw     = outputs.get('pathology_meas_logits', None) # [B, 2]
                _hb_mask = birads_route_mask | pathology_route_mask
                _meas_emb_sel = meas_emb[non_mri_idx]                     # [n_sel, 128]

                # Specialist report mode: build the descriptor token from GT
                # BI-RADS / pathology (predicted heads are untrained / absent).
                _use_gt_desc = bool(getattr(self, "use_gt_descriptors", False))
                if _use_gt_desc and (gt_birads_idx is not None):
                    # CORAL-style ordinal target: K-1 thresholds, 1.0 for k < idx
                    _bidx_sel = gt_birads_idx[non_mri_idx].to(x.device).long().clamp(min=-1)
                    _Km1 = self.birads_num_classes - 1
                    _ar = torch.arange(_Km1, device=x.device).unsqueeze(0)  # [1, K-1]
                    _birads_sel = (_ar < _bidx_sel.unsqueeze(1)).float()    # [n_sel, K-1]
                    _birads_sel = _birads_sel * (_bidx_sel.unsqueeze(1) >= 0).float()
                else:
                    _birads_sel = (
                        _birads_raw[non_mri_idx].float()
                        if _birads_raw is not None
                        else torch.zeros(non_mri_idx.shape[0], self.birads_num_classes - 1,
                                         device=x.device)
                    )
                if _use_gt_desc and (gt_pathology_idx is not None):
                    _pidx_sel = gt_pathology_idx[non_mri_idx].to(x.device).long()
                    _path_sel = torch.zeros(non_mri_idx.shape[0], 2, device=x.device)
                    _valid_p = (_pidx_sel >= 0) & (_pidx_sel < 2)
                    if _valid_p.any():
                        _vi = _valid_p.nonzero(as_tuple=True)[0]
                        _path_sel[_vi, _pidx_sel[_vi]] = 1.0
                else:
                    _path_sel = (
                        _path_raw[non_mri_idx].float()
                        if _path_raw is not None
                        else torch.zeros(non_mri_idx.shape[0], 2, device=x.device)
                    )
                # Gate non-BUS-BRA samples so their descriptor token is near-zero
                _hb_sel = _hb_mask[non_mri_idx].float().unsqueeze(-1)    # [n_sel, 1]
                _desc_in = torch.cat([
                    _meas_emb_sel * _hb_sel,
                    _birads_sel   * _hb_sel,
                    _path_sel     * _hb_sel,
                ], dim=-1)                                                  # [n_sel, 133]
                busbra_desc_tokens = self.busbra_desc_proj(_desc_in).unsqueeze(1)  # [n_sel, 1, llm_dim]

                # Assemble per-sample non-MRI report tokens
                for local_i, batch_i in enumerate(non_mri_idx.tolist()):
                    tok_i = base_report_tokens[local_i]
                    if (seg_report_tokens is not None
                            and bool(use_seg_report_local[local_i].item())):
                        tok_i = torch.cat([tok_i, seg_report_tokens[local_i]], dim=0)
                    # Append descriptor token (near-null for non-BUS-BRA)
                    tok_i = torch.cat([tok_i, busbra_desc_tokens[local_i]], dim=0)
                    non_mri_report_token_map[batch_i] = tok_i

        # ==============================================================
        # TASK 3: DETECTION (MRI only, official DINO, no seg guidance)
        # ==============================================================
        roi_boxes = boxes
        do_det = det_batch_idx.numel() > 0

        if do_det:
            x_det       = x[det_batch_idx]
            c2_map_det  = c2_det[det_batch_idx].float()
            c3_map_det  = c3_det[det_batch_idx].float()
            c4_map_det  = c4_det[det_batch_idx].float()

            det_targets_det = None
            if det_targets is not None and self.training:
                det_active_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
                det_active_mask[det_batch_idx] = True
                det_targets_det, _ = prepare_detection_targets(
                    det_targets=det_targets,
                    active_mask=det_active_mask,
                    img_h=x.shape[-2],
                    img_w=x.shape[-1],
                    num_classes=self.det_num_classes,
                    device=x.device,
                )

            with torch.amp.autocast(device_type="cuda", enabled=False):
                det_out_det = self.detector.forward_from_shared_feats(
                    images=x_det.float(),
                    feat_maps=[c2_map_det, c3_map_det, c4_map_det],
                    targets=det_targets_det,
                )

            outputs["detection_active"] = det_out_det

            # Extract query feats for compatibility
            active_query_feats = self._extract_query_feats(det_out_det, det_batch_idx)
            outputs["det_query_feats_active"] = active_query_feats

            if self.training:
                outputs["detection"]      = det_out_det
                outputs["pred_logits"]    = det_out_det["pred_logits"]
                outputs["pred_boxes"]     = det_out_det["pred_boxes"]
                outputs["det_query_feats"] = active_query_feats
            else:
                det_out_full = self._pad_det_outputs_to_batch(det_out_det, det_batch_idx, B, active_query_feats)
                outputs["detection"]       = det_out_full
                outputs["pred_logits"]     = det_out_full["pred_logits"]
                outputs["pred_boxes"]      = det_out_full["pred_boxes"]
                outputs["det_query_feats"] = det_out_full.get("main_query_feats", None)

            # Decode predictions for ROI guidance
            if cross_roi and use_pred_boxes_for_mri_cls:
                with torch.no_grad():
                    pb, pl, ps, pq = self.decode_mri_det_predictions(
                        det_out_det, image_size=x.shape[-2:],
                        score_thresh=det_score_thresh, topk=8, nms_iou=0.4,
                    )
                empty_b = torch.zeros((0, 4), device=x.device, dtype=torch.float32)
                empty_l = torch.zeros((0,),   device=x.device, dtype=torch.long)
                empty_s = torch.zeros((0,),   device=x.device, dtype=torch.float32)
                empty_q = torch.zeros((0,),   device=x.device, dtype=torch.long)

                pred_boxes_list     = [empty_b.clone() for _ in range(B)]
                pred_labels_list    = [empty_l.clone() for _ in range(B)]
                pred_scores_list    = [empty_s.clone() for _ in range(B)]
                pred_query_ids_list = [empty_q.clone() for _ in range(B)]

                for local_i, batch_i in enumerate(det_batch_idx.tolist()):
                    pred_boxes_list[batch_i]     = pb[local_i]
                    pred_labels_list[batch_i]    = pl[local_i]
                    pred_scores_list[batch_i]    = ps[local_i]
                    pred_query_ids_list[batch_i] = pq[local_i]

                roi_boxes = pred_boxes_list
                outputs["mri_pred_boxes"]     = pred_boxes_list
                outputs["mri_pred_labels"]    = pred_labels_list
                outputs["mri_pred_scores"]    = pred_scores_list
                outputs["mri_pred_query_ids"] = pred_query_ids_list

        # ==============================================================
        # REPORT TOKEN ASSEMBLY (non-MRI + MRI)
        # ==============================================================
        if has_rep_task and rep_batch_idx.numel() > 0:
            report_tokens_list = []
            build_mri_struct = store_vis

            for b in rep_batch_idx.tolist():
                if bool(is_lss_mri[b]):
                    # MRI: full-image c4_rep tokens only
                    c4_rep_b = c4_rep[b:b+1].flatten(2).transpose(1, 2)
                    tok_b = self.visual_adapter(
                        F.layer_norm(c4_rep_b, c4_rep_b.shape[-1:])
                    ).squeeze(0)

                    if build_mri_struct:
                        outputs["mri_report_struct"][b] = {
                            "mode": "global_full_image_only",
                            "roi_count": 0,
                            "roi_pred_classes": [],
                            "roi_confidence": [],
                            "used_pred_boxes": False,
                        }
                else:
                    if b in non_mri_report_token_map:
                        tok_b = non_mri_report_token_map[b]
                    else:
                        c4_rep_b = c4_rep[b:b+1].flatten(2).transpose(1, 2)
                        tok_b = self.visual_adapter(
                            F.layer_norm(c4_rep_b, c4_rep_b.shape[-1:])
                        ).squeeze(0)

                report_tokens_list.append(tok_b)

            # Pad to uniform length and stack
            maxT = max(t.shape[0] for t in report_tokens_list)
            padded = []
            for t in report_tokens_list:
                if t.shape[0] < maxT:
                    pad = t.new_zeros((maxT - t.shape[0], t.shape[1]))
                    t = torch.cat([t, pad], dim=0)
                padded.append(t)

            report_tokens = torch.stack(padded, dim=0)
            outputs["report_tokens"] = report_tokens
            outputs["visual_tokens"] = report_tokens

            # Compute report loss
            if captions is not None:
                captions_rep = captions[rep_idx]
                report_labels_rep = report_labels[rep_idx] if report_labels is not None else None

                if report_labels_rep is not None:
                    valid_rep = (report_labels_rep != -100).any(dim=1)
                else:
                    valid_rep = torch.ones(captions_rep.shape[0], dtype=torch.bool, device=x.device)

                if valid_rep.any():
                    report_tokens_valid     = report_tokens[valid_rep]
                    captions_rep_valid      = captions_rep[valid_rep]
                    report_labels_rep_valid = report_labels_rep[valid_rep] if report_labels_rep is not None else None

                    if not torch.isfinite(report_tokens_valid).all():
                        outputs["rep_loss"]      = None
                        outputs["report_logits"] = None
                    else:
                        out = self.report_decoder(
                            report_tokens_valid,
                            input_ids=captions_rep_valid,
                            labels=report_labels_rep_valid,
                        )
                        if not torch.isfinite(out.loss):
                            outputs["rep_loss"]      = None
                            outputs["report_logits"] = None
                        else:
                            outputs["rep_loss"]      = out.loss
                            outputs["report_logits"] = out.logits[:, report_tokens_valid.shape[1]:, :]
                else:
                    outputs["rep_loss"]      = None
                    outputs["report_logits"] = None

        # ==============================================================
        # Feature exports
        # ==============================================================
        outputs["feat_c2_shared"] = c2_spatial
        outputs["feat_c3_shared"] = c3_spatial
        outputs["feat_c4_shared"] = c4_spatial
        outputs["feat_c2_seg"] = c2_seg
        outputs["feat_c2_det"] = c2_det
        outputs["feat_c3_seg"] = c3_seg
        outputs["feat_c3_det"] = c3_det
        outputs["feat_c4_seg"] = c4_seg
        outputs["feat_c4_cls"] = c4_cls
        # Explicit aliases consumed by inference artifact exporters.
        outputs["ct_gradcam_feature"] = c4_cls
        outputs["morphology_measurement_names"] = tuple(MEASUREMENT_KEYS_BUSBRA)
        outputs["feat_c4_rep"] = c4_rep
        outputs["feat_c4_det"] = c4_det

        if self.use_moe and self.task_moe is not None:
            outputs["moe_gamma"] = {
                stage: bank.effective_gamma().detach().cpu().tolist()
                for stage, bank in self.task_moe.banks.items()
            }
            outputs["moe_expert_bias"] = None
            outputs["gradif_routing_summary"] = self.task_moe.routing_summary()
            outputs["ultrasound_adapter_alpha_value"] = float(
                self.effective_ultrasound_adapter_alpha().detach().cpu()
            )
            outputs["descriptor_gt_probability"] = float(
                self.descriptor_gt_probability
            )


        return outputs

    # ==================================================================
    # Helpers (factored out of forward for readability)
    # ==================================================================
    def _extract_query_feats(self, det_out, det_batch_idx):
        """Pull main_query_feats or query_feats from det_out, fixing shape."""
        for key in ("main_query_feats", "query_feats"):
            qf = det_out.get(key)
            if qf is None:
                continue

            if qf.dim() == 2:
                qf = qf.unsqueeze(0)
            elif (qf.dim() == 3
                  and qf.shape[0] != det_batch_idx.numel()
                  and qf.shape[1] == det_batch_idx.numel()):
                qf = qf.transpose(0, 1).contiguous()

            if key == "main_query_feats":
                return qf

            # For raw query_feats, trim to match pred_logits query count
            q_expected = det_out["pred_logits"].shape[1]
            if qf.shape[1] != q_expected:
                qf = qf[:, -q_expected:, :].contiguous()
            return qf

        return None

    def _pad_det_outputs_to_batch(self, det_out_det, det_batch_idx, B, active_query_feats):
        """Expand detection-active outputs to full batch size for eval."""
        Q    = det_out_det["pred_logits"].shape[1]
        Cdet = det_out_det["pred_logits"].shape[2]
        neg  = -1e4 if det_out_det["pred_logits"].dtype == torch.float16 else -1e9

        full_logits = det_out_det["pred_logits"].new_full((B, Q, Cdet), neg)
        full_boxes  = det_out_det["pred_boxes"].new_zeros((B, Q, 4))
        full_logits[det_batch_idx] = det_out_det["pred_logits"]
        full_boxes[det_batch_idx]  = det_out_det["pred_boxes"]

        det_out_full = {
            "pred_logits": full_logits,
            "pred_boxes":  full_boxes,
        }

        if "aux_outputs" in det_out_det and det_out_det["aux_outputs"] is not None:
            aux_full = []
            for aux in det_out_det["aux_outputs"]:
                aneg = -1e4 if aux["pred_logits"].dtype == torch.float16 else -1e9
                al = aux["pred_logits"].new_full((B, Q, Cdet), aneg)
                ab = aux["pred_boxes"].new_zeros((B, Q, 4))
                al[det_batch_idx] = aux["pred_logits"]
                ab[det_batch_idx] = aux["pred_boxes"]
                aux_full.append({"pred_logits": al, "pred_boxes": ab})
            det_out_full["aux_outputs"] = aux_full

        if "dn_meta" in det_out_det:
            det_out_full["dn_meta"] = det_out_det["dn_meta"]

        if active_query_feats is not None:
            full_qf = active_query_feats.new_zeros(
                (B, active_query_feats.shape[1], active_query_feats.shape[2])
            )
            full_qf[det_batch_idx] = active_query_feats
            det_out_full["main_query_feats"] = full_qf

        return det_out_full

    # def fuse_roi_query_features(
    #     self,
    #     roi_base_feats,
    #     roi_batch_idx,
    #     roi_box_idx,
    #     main_query_feats,
    #     roi_query_ids,
    # ):
    #     """
    #     roi_base_feats:   [Nroi, 512] from self.mri_cls_neck(.)
    #     roi_batch_idx:    [Nroi]
    #     roi_box_idx:      [Nroi]  (index into the per-image ROI source list)
    #     main_query_feats: [B, Q, 256] from det_out["main_query_feats"]
    #     roi_query_ids:    list of length B
    #                     each item is Tensor [Ni] mapping ROI index -> matched query index
    #                     unmatched entries should be -1

    #     returns:
    #         fused_logits:      [Nvalid, 4] or None
    #         roi_base_valid:    [Nvalid, 512] or None
    #         valid_batch_idx:   [Nvalid] or None
    #         valid_box_idx:     [Nvalid] or None
    #     """
    #     if (
    #         roi_base_feats is None or
    #         roi_batch_idx is None or
    #         roi_box_idx is None or
    #         main_query_feats is None or
    #         roi_query_ids is None
    #     ):
    #         return None, None, None, None

    #     Nroi = roi_batch_idx.shape[0]
    #     device = roi_batch_idx.device

    #     roi_query_idx = torch.full((Nroi,), -1, device=device, dtype=torch.long)
    #     valid_mask = torch.zeros((Nroi,), device=device, dtype=torch.bool)

    #     for k in range(Nroi):
    #         b = int(roi_batch_idx[k].item())
    #         j = int(roi_box_idx[k].item())

    #         if b < len(roi_query_ids) and roi_query_ids[b] is not None and j < roi_query_ids[b].numel():
    #             qid = int(roi_query_ids[b][j].item())
    #             if qid >= 0:
    #                 roi_query_idx[k] = qid
    #                 valid_mask[k] = True

    #     if not valid_mask.any():
    #         return None, None, None, None

    #     valid_batch_idx = roi_batch_idx[valid_mask]
    #     valid_box_idx = roi_box_idx[valid_mask]
    #     valid_query_idx = roi_query_idx[valid_mask]

    #     roi_base_valid = roi_base_feats[valid_mask]                          # [Nvalid, 512]
    #     roi_query_feats = main_query_feats[valid_batch_idx, valid_query_idx] # [Nvalid, 256]
    #     roi_query_feats = self.query_to_roi(roi_query_feats)                 # [Nvalid, 512]

    #     roi_feat_cat = torch.cat([roi_base_valid, roi_query_feats], dim=-1)  # [Nvalid, 1024]
    #     fused_logits = self.mri_cls_head_fused(roi_feat_cat)                 # [Nvalid, 4]

    #     return fused_logits, roi_base_valid, valid_batch_idx, valid_box_idx
        
    def get_mri_level_labels_for_rois(
        self,
        batch_idx,
        roi_box_idx_b,
        use_pred_boxes_for_mri_cls,
        det_targets,
        pred_labels_list,
        device,
    ):
        if use_pred_boxes_for_mri_cls:
            if pred_labels_list is None or pred_labels_list[batch_idx] is None:
                return None
            src = pred_labels_list[batch_idx]
        else:
            if det_targets is None or det_targets[batch_idx] is None:
                return None
            src = det_targets[batch_idx].get("labels", None)
            if src is None:
                return None

        if src.numel() == 0:
            return None

        valid = (roi_box_idx_b >= 0) & (roi_box_idx_b < src.numel())
        if not valid.any():
            return None

        out = torch.full((roi_box_idx_b.shape[0],), -1, dtype=torch.long, device=device)
        out[valid] = src[roi_box_idx_b[valid]].long()
        return out

    def build_mri_report_tokens_for_image(
        self,
        batch_idx,
        c4_rep,
        roi_logits,
        roi_base_feats,
        roi_batch_idx,
        roi_box_idx,
        use_pred_boxes_for_mri_cls,
        det_targets,
        pred_labels_list,
        build_struct=False,   # NEW
    ):
        device = c4_rep.device

        global_feat = self.global_pool(c4_rep[batch_idx:batch_idx + 1])
        global_tok = self.mri_global_to_report(global_feat)  # [1, 2048]

        mask_b = (roi_batch_idx == batch_idx)
        if not mask_b.any():
            tokens = torch.cat([global_tok, self.mri_null_level_token.squeeze(0)], dim=0)
            return tokens, ([] if build_struct else None)

        roi_base_b = roi_base_feats[mask_b]
        roi_logits_b = roi_logits[mask_b]
        roi_box_idx_b = roi_box_idx[mask_b]

        level_labels_b = self.get_mri_level_labels_for_rois(
            batch_idx=batch_idx,
            roi_box_idx_b=roi_box_idx_b,
            use_pred_boxes_for_mri_cls=use_pred_boxes_for_mri_cls,
            det_targets=det_targets,
            pred_labels_list=pred_labels_list,
            device=device,
        )
        if level_labels_b is None:
            tokens = torch.cat([global_tok, self.mri_null_level_token.squeeze(0)], dim=0)
            return tokens, ([] if build_struct else None)

        valid = (level_labels_b >= 0)
        if not valid.any():
            tokens = torch.cat([global_tok, self.mri_null_level_token.squeeze(0)], dim=0)
            return tokens, ([] if build_struct else None)

        roi_base_b = roi_base_b[valid]
        roi_logits_b = roi_logits_b[valid]
        level_labels_b = level_labels_b[valid]

        sev_probs_b = torch.softmax(roi_logits_b.float(), dim=-1)
        roi_tok = self.mri_roi_to_report(roi_base_b)
        roi_tok = roi_tok + self.mri_level_embed(level_labels_b) + self.mri_severity_proj(sev_probs_b)

        order = torch.argsort(level_labels_b)
        roi_tok = roi_tok[order]
        level_labels_b = level_labels_b[order]
        sev_probs_b = sev_probs_b[order]

        tokens = torch.cat([global_tok, roi_tok], dim=0)

        if not build_struct:
            return tokens, None

        report_struct = []
        for j in range(level_labels_b.numel()):
            report_struct.append({
                "level_label": int(level_labels_b[j].item()),
                "severity_probs": sev_probs_b[j].detach().cpu(),
            })

        return tokens, report_struct

    def build_seg_attention_map(self, seg_logits):
            """
            seg_logits: [B, Cseg, H, W]
            return:     [B, 1, H, W]
            """
            seg_prob = torch.sigmoid(seg_logits)
            seg_attn = seg_prob.max(dim=1, keepdim=True)[0]
            return seg_attn

    def tokens_to_map(self, feat, res):
            B, L, C = feat.shape
            H, W = res
            return feat.transpose(1, 2).reshape(B, C, H, W).contiguous()
        
    def extract_box_roi_features(self, feat_map, boxes, image_size, pool_size=1, box_expand_ratio=0.15):
            """
            feat_map:   [B, C, Hf, Wf]
            boxes:      list of length B, each Tensor [Ni, 4] in xyxy image coordinates
            image_size: (H_img, W_img)

            returns:
                roi_feats:     [Nroi, C * pool_size * pool_size]
                roi_batch_idx: [Nroi]
                roi_box_idx:   [Nroi]
            """
            B, C, Hf, Wf = feat_map.shape
            H_img, W_img = image_size
            device = feat_map.device

            rois = []
            roi_batch_idx = []
            roi_box_idx = []

            sx = Wf / float(W_img)
            sy = Hf / float(H_img)

            for b in range(B):
                if boxes[b] is None or boxes[b].numel() == 0:
                    continue

                cur_boxes = boxes[b].float()
                if cur_boxes.ndim == 1:
                    cur_boxes = cur_boxes.unsqueeze(0)

                # expand boxes to keep local context
                cx = 0.5 * (cur_boxes[:, 0] + cur_boxes[:, 2])
                cy = 0.5 * (cur_boxes[:, 1] + cur_boxes[:, 3])
                bw = (cur_boxes[:, 2] - cur_boxes[:, 0]).clamp(min=1.0)
                bh = (cur_boxes[:, 3] - cur_boxes[:, 1]).clamp(min=1.0)

                bw = bw * (1.0 + box_expand_ratio)
                bh = bh * (1.0 + box_expand_ratio)

                cur_boxes[:, 0] = cx - 0.5 * bw
                cur_boxes[:, 2] = cx + 0.5 * bw
                cur_boxes[:, 1] = cy - 0.5 * bh
                cur_boxes[:, 3] = cy + 0.5 * bh

                # sanitize
                cur_boxes = torch.nan_to_num(cur_boxes, nan=0.0, posinf=max(W_img, H_img), neginf=0.0)
                cur_boxes[:, 0::2] = cur_boxes[:, 0::2].clamp(0, W_img - 1)
                cur_boxes[:, 1::2] = cur_boxes[:, 1::2].clamp(0, H_img - 1)

                # scale from image coordinates -> feature-map coordinates
                scaled = cur_boxes.clone()
                scaled[:, 0] *= sx
                scaled[:, 2] *= sx
                scaled[:, 1] *= sy
                scaled[:, 3] *= sy

                x1 = scaled[:, 0].clamp(0, Wf - 1)
                y1 = scaled[:, 1].clamp(0, Hf - 1)
                x2 = scaled[:, 2].clamp(0, Wf - 1)
                y2 = scaled[:, 3].clamp(0, Hf - 1)

                # make sure boxes have non-zero size
                x2 = torch.maximum(x2, x1 + 1.0)
                y2 = torch.maximum(y2, y1 + 1.0)

                valid = torch.isfinite(torch.stack([x1, y1, x2, y2], dim=1)).all(dim=1)
                valid = valid & (x2 > x1) & (y2 > y1)

                if not valid.any():
                    continue

                valid_boxes = torch.stack([x1, y1, x2, y2], dim=1)[valid]
                valid_orig_idx = torch.arange(cur_boxes.shape[0], device=device, dtype=torch.long)[valid]

                batch_col = torch.full(
                    (valid_boxes.shape[0], 1),
                    float(b),
                    device=device,
                    dtype=torch.float32
                )

                rois.append(torch.cat([batch_col, valid_boxes], dim=1))  # [K, 5]
                roi_batch_idx.append(
                    torch.full((valid_boxes.shape[0],), b, device=device, dtype=torch.long)
                )
                roi_box_idx.append(valid_orig_idx)

            if len(rois) == 0:
                return None, None, None

            rois = torch.cat(rois, dim=0)                  # [Nroi, 5]
            roi_batch_idx = torch.cat(roi_batch_idx, dim=0)
            roi_box_idx = torch.cat(roi_box_idx, dim=0)

            pooled = roi_align(
                feat_map,
                rois,
                output_size=(pool_size, pool_size),
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=True
            )  # [Nroi, C, pool_size, pool_size]

            pooled = pooled.flatten(1)  # [Nroi, C * pool_size * pool_size]

            finite = torch.isfinite(pooled).all(dim=1)
            if not finite.any():
                return None, None, None

            pooled = pooled[finite]
            roi_batch_idx = roi_batch_idx[finite]
            roi_box_idx = roi_box_idx[finite]

            return pooled, roi_batch_idx, roi_box_idx
        
    def postprocess_dino_predictions(
            self,
            pred_logits,
            pred_boxes,
            img_w,
            img_h,
            score_thresh=0.30,
            topk=8,
            nms_iou=0.4
        ):
            """
            pred_logits: [Q, C]      DINO-style, no background class
            pred_boxes : [Q, 4]      normalized cxcywh

            returns:
                boxes_xyxy_abs: [N,4]
                labels:         [N]
                scores:         [N]
                query_ids:      [N]   # original kept query indices
            """
            pred_logits = pred_logits.float()
            pred_boxes = pred_boxes.float()

            Q = pred_logits.shape[0]
            query_ids = torch.arange(Q, device=pred_logits.device, dtype=torch.long)

            # one best class per query
            probs = pred_logits.sigmoid()              # [Q, C]
            scores, labels = probs.max(dim=-1)         # [Q]

            boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes).clone()
            boxes_xyxy[:, 0::2] *= float(img_w)
            boxes_xyxy[:, 1::2] *= float(img_h)
            boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0, img_w - 1)
            boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0, img_h - 1)

            keep = scores > score_thresh
            boxes_xyxy = boxes_xyxy[keep]
            labels = labels[keep]
            scores = scores[keep]
            query_ids = query_ids[keep]

            if scores.numel() == 0:
                return (
                    torch.zeros((0, 4), device=pred_boxes.device, dtype=pred_boxes.dtype),
                    torch.zeros((0,), device=pred_boxes.device, dtype=torch.long),
                    torch.zeros((0,), device=pred_boxes.device, dtype=pred_boxes.dtype),
                    torch.zeros((0,), device=pred_boxes.device, dtype=torch.long),
                )

            keep_idx = batched_nms(boxes_xyxy, scores, labels, nms_iou)
            keep_idx = keep_idx[:topk]

            return (
                boxes_xyxy[keep_idx],
                labels[keep_idx],
                scores[keep_idx],
                query_ids[keep_idx],
            )

    def decode_mri_det_predictions(self, det_out, image_size, score_thresh=0.30, topk=8, nms_iou=0.4):
            """
            det_out:
                pred_logits: [B, Q, C]
                pred_boxes : [B, Q, 4]

            returns:
                pred_boxes_list
                pred_labels_list
                pred_scores_list
                pred_query_ids_list
            """
            pred_logits = det_out["pred_logits"].float()
            pred_boxes = det_out["pred_boxes"].float()
            H_img, W_img = image_size

            pred_boxes_list = []
            pred_labels_list = []
            pred_scores_list = []
            pred_query_ids_list = []

            for b in range(pred_logits.shape[0]):
                b_boxes, b_labels, b_scores, b_query_ids = self.postprocess_dino_predictions(
                    pred_logits[b],
                    pred_boxes[b],
                    W_img,
                    H_img,
                    score_thresh=score_thresh,
                    topk=topk,
                    nms_iou=nms_iou,
                )
                pred_boxes_list.append(b_boxes)
                pred_labels_list.append(b_labels)
                pred_scores_list.append(b_scores)
                pred_query_ids_list.append(b_query_ids)

            return pred_boxes_list, pred_labels_list, pred_scores_list, pred_query_ids_list
        
    def load_detector_pretrained(self, ckpt_path):
            import os
            if ckpt_path is None or (not os.path.isfile(ckpt_path)):
                print(f"[Detector] pretrained checkpoint not found: {ckpt_path}")
                return

            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))

            # official fine-tuning recommendation:
            # ignore class-specific keys for custom dataset
            ignore_prefixes = (
                "class_embed",
                "label_enc",
                "transformer.decoder.class_embed",
                "transformer.enc_out_class_embed",
            )

            filtered = {}
            for k, v in state.items():
                if any(k.startswith(p) for p in ignore_prefixes):
                    continue
                filtered[k] = v

            missing, unexpected = self.detector.load_state_dict(filtered, strict=False)
            print(f"[Detector] pretrained load done | missing={len(missing)} unexpected={len(unexpected)}")
