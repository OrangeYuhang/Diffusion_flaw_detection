# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch import einsum
from einops import rearrange, repeat
from autoencoder import *

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                        时间步和类别标签的嵌入层                               #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    将标量时间步嵌入为向量表示。
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建正弦时间步嵌入。
        :param t: 一个包含 N 个索引的一维张量，每个批次元素对应一个索引。
                          这些索引可以是小数值。
        :param dim: 输出的维度。
        :param max_period: 控制嵌入的最小频率。
        :return: 一个 (N, D) 的位置嵌入张量。
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                              DiT 核心模型                                     #
#################################################################################


class CrossAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = query_dim / heads

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=True)
        self.to_k = nn.Linear(query_dim, query_dim, bias=True)
        self.to_v = nn.Linear(query_dim, query_dim, bias=True)

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context).unsqueeze(1)
        v = self.to_v(context).unsqueeze(1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 注意力机制，我们永远不嫌多
        attn = sim.softmax(dim=-2)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        attn_out = rearrange(attn, '(b h) n d -> b n (h d)', h=h)
        return out, attn_out


class Cross_Norm(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attention = CrossAttention(hidden_size, heads=num_heads)

    def forward(self, x, c):
        x = self.norm(x)
        x = self.cross_attention(x, c)
        
        return x



class DiTBlock(nn.Module):
    """
    使用自适应零初始化层归一化 (adaLN-Zero) 条件注入的 DiT 块。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    DiT 的最后一层。
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class temp_Adaptive_Mask(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
            nn.Linear(2 * hidden_size, hidden_size, bias=True)
        )


    def forward(self, x):
        x = self.norm(x)
        x = self.mlp(x)
        x = self.linear(x)
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        c = self.out_channels
        p = self.patch_size

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return torch.tanh(x)  # 约束到 [-1, 1]，匹配 VAE 潜空间掩码范围


class DiT(nn.Module):
    """
    基于 Transformer 骨干网络的扩散模型。
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedders = nn.Linear(1024, 1152)
        num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.cross_defect = nn.ModuleList([
            Cross_Norm(hidden_size, num_heads) for _ in range(10)
                                     ])
        self.adapt_mask = temp_Adaptive_Mask(num_heads*10, patch_size, in_channels)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # 初始化 Transformer 层：
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 通过正余弦嵌入初始化（并冻结）pos_embed：
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 像 nn.Linear 一样初始化 patch_embed（而非 nn.Conv2d）：
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # 初始化时间步嵌入 MLP：
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # 将 DiT 块中的 adaLN 调制层置零：
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # 将输出层置零：
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        将 patch 序列还原为图像。
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        DiT 的前向传播。
        x: (N, C, H, W) 空间输入张量（图像或图像的潜在表示）
        t: (N,) 扩散时间步张量
        y: (N,) 类别标签张量
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)，其中 T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y_defect = self.y_embedders(y[0])
        y_class = self.y_embedders(y[1])       # (N, D)
        y_all = self.y_embedders(y[2])
        att_map = []
        loss_att = 0
        for i in range(28):
            block = self.blocks[i]
            if i < 10:
                c = t + y_class
                x = block(x, c)                      # (N, T, D)
            elif i < 20:
                cross = self.cross_defect[i - 10]
                c = t + y_defect

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(c).chunk(6,
                                                                                                                 dim=1)
                x = x + gate_msa.unsqueeze(1) * block.attn(modulate(block.norm1(x), shift_msa, scale_msa))

                cross_att, att_weight = cross(x, c)
                loss_att += att_weight
                att_map.append(att_weight)
                x = x + cross_att

                x = x + gate_mlp.unsqueeze(1) * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))

            elif i < 28:
                c = t + y_all
                x = block(x, c)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        att_map = torch.cat(att_map, dim=-1)
        att_mask = self.adapt_mask(att_map)
        
        return x, att_mask, loss_att.reshape(x.shape[0], x.shape[2]//2, x.shape[3]//2, 16).mean(dim=-1)
    
    def forward_free_2(self, x, t, y, mask_temp=None):
        """
        DiT 的前向传播（双分支无条件）。
        x: (N, C, H, W) 空间输入张量（图像或图像的潜在表示）
        t: (N,) 扩散时间步张量
        y: (N,) 类别标签张量
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)，其中 T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y_defect = self.y_embedders(torch.cat([y[0][0], y[1][0]], dim=0))
        y_class = self.y_embedders(torch.cat([y[0][1], y[1][1]], dim=0))       # (N, D)
        y_all = self.y_embedders(torch.cat([y[0][2], y[1][2]], dim=0))
        att_map = []
        loss_att = 0
        for i in range(28):
            block = self.blocks[i]
            if i < 10:
                c = t + y_class
                x = block(x, c)                      # (N, T, D)
            elif i < 20:
                cross = self.cross_defect[i - 10]
                c = t + y_defect
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(c).chunk(6,
                                                                                                dim=1)
                x = x + gate_msa.unsqueeze(1) * block.attn(modulate(block.norm1(x), shift_msa, scale_msa))
                cross_att, att_weight = cross(x, c)
                att_map.append(att_weight)
                loss_att+=att_weight
                x = x + cross_att
                x = x + gate_mlp.unsqueeze(1) * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))

            elif i < 28:
                c = t + y_all
                x = block(x, c)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        att_map = torch.cat(att_map, dim=-1)
        att_mask = self.adapt_mask(att_map)

        return x, att_mask, loss_att
    
    def forward_with_cfg_2(self, x, t, y, cfg_scale):
        """
        DiT 的前向传播，同时批量执行无条件前向传播以实现无分类器引导（双分支）。
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out, mask, _ = self.forward_free_2(combined, t, y)
        # 对所有 in_channels 应用 CFG 以实现完整的潜在空间引导
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1), mask, _
    
    def forward_free_3(self, x, t, y, mask_temp=None):
        """
        DiT 的前向传播（三分支无条件）。
        x: (N, C, H, W) 空间输入张量（图像或图像的潜在表示）
        t: (N,) 扩散时间步张量
        y: (N,) 类别标签张量
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)，其中 T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y_defect = self.y_embedders(torch.cat([y[0][0], y[1][0], y[2][0]], dim=0))
        y_class = self.y_embedders(torch.cat([y[0][1], y[1][1], y[2][1]], dim=0))       # (N, D)
        y_all = self.y_embedders(torch.cat([y[0][2], y[1][2], y[2][2]], dim=0))
        att_map = []
        att_loss = 0
        for i in range(28):
            block = self.blocks[i]
            if i < 10:
                c = t + y_class
                x = block(x, c)                      # (N, T, D)
            elif i < 20:
                cross = self.cross_defect[i - 10]
                c = t + y_defect
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaLN_modulation(c).chunk(6,
                                                                                                dim=1)
                x = x + gate_msa.unsqueeze(1) * block.attn(modulate(block.norm1(x), shift_msa, scale_msa))
                cross_att, att_weight = cross(x, c)
                att_map.append(att_weight)
                att_loss += att_weight
                x = x + cross_att
                x = x + gate_mlp.unsqueeze(1) * block.mlp(modulate(block.norm2(x), shift_mlp, scale_mlp))

            elif i < 28:
                c = t + y_all
                x = block(x, c)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        att_map = torch.cat(att_map, dim=-1)
        att_mask = self.adapt_mask(att_map)

        return x, att_mask, att_loss
    
    def forward_with_cfg_3(self, x, t, y, cfg_scale):
        """
        DiT 的前向传播，同时批量执行无条件前向传播以实现无分类器引导（三分支）。
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 3]
        combined = torch.cat([half, half, half], dim=0)
        model_out, mask, _ = self.forward_free_3(combined, t, y)
        # 对所有 in_channels 应用 CFG 以实现完整的潜在空间引导
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps_defect, uncond_eps = torch.split(eps, len(eps) // 3, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps_defect) + cfg_scale * (uncond_eps_defect - uncond_eps)
        eps = torch.cat([half_eps, half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1), mask, _

#################################################################################
#                        正余弦位置嵌入函数                                     #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    生成二维正余弦位置嵌入。
    grid_size: 网格高度和宽度的整数值
    return:
    pos_embed: [grid_size*grid_size, embed_dim] 或 [1+grid_size*grid_size, embed_dim]（带/不带 cls_token）
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 此处 w 在前
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # 使用一半维度来编码 grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    从网格生成一维正余弦位置嵌入。
    embed_dim: 每个位置的输出维度
    pos: 待编码的位置列表，大小为 (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)，外积

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
