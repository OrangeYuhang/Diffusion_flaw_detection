"""
缺陷位置解耦生成。
扩展 DiT 以支持显式掩码条件，从而在推理时控制缺陷的空间位置。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from models_add_cross_concate import (
    DiT, DiTBlock, Cross_Norm, FinalLayer, TimestepEmbedder,
    temp_Adaptive_Mask, modulate, get_2d_sincos_pos_embed
)


class MaskEncoder(nn.Module):
    """
    将空间掩码 (B, 1, H, W) 编码为条件向量 (B, D)
    和空间特征图，用于细粒度位置控制。
    """
    def __init__(self, hidden_size=1152, input_size=32):
        super().__init__()
        self.input_size = input_size

        # 轻量级卷积编码器（用于掩码）
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),   # -> 16
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> 8
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # -> 4
            nn.SiLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # -> 2
            nn.SiLU(),
        )
        # 全局投影
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # 空间投影（用于细粒度注入）
        self.spatial_proj = nn.Conv2d(512, hidden_size, 1)

    def forward(self, mask):
        """
        参数:
            mask: (B, 1, H, W) 二值掩码，值域 [0, 1] 或 [-1, 1]
        返回:
            global_cond: (B, D) 全局掩码嵌入
            spatial_cond: (B, D, H', W') 空间掩码特征
        """
        feat = self.conv(mask)  # (B, 512, 2, 2)
        global_cond = self.global_proj(feat)  # (B, D)
        spatial_cond = self.spatial_proj(feat)  # (B, D, 2, 2)
        return global_cond, spatial_cond


class DiTMaskConditioned(DiT):
    """
    扩展 DiT，添加掩码条件以实现位置感知生成。
    掩码条件通过 adaLN 与时间 + 文本条件进行融合。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hidden_size = kwargs.get('hidden_size', 1152)
        latent_size = kwargs.get('input_size', 32)

        self.mask_encoder = MaskEncoder(hidden_size, latent_size)

        # 融合层：将掩码条件与现有 adaLN 输入结合
        self.mask_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size),
            ) for _ in range(kwargs.get('depth', 28))
        ])
        # 零初始化 mask_fusion 最后一层 — 训练初期等价恒等映射，防止 NaN
        for fusion in self.mask_fusion:
            nn.init.constant_(fusion[-1].weight, 0)
            nn.init.constant_(fusion[-1].bias, 0)

        # 空间掩码投影到 token 级
        num_patches = latent_size * latent_size // (kwargs.get('patch_size', 2) ** 2)
        self.mask_to_tokens = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.Flatten(2),
        )
        nn.init.constant_(self.mask_to_tokens[0].weight, 0)
        nn.init.constant_(self.mask_to_tokens[0].bias, 0)

    def forward(self, x, t, y, mask_cond=None):
        """
        重载 DiT.forward() — 与 diffusion.training_losses 签名兼容。
        当提供 mask_cond 时路由到 forward_with_mask()，否则回退到 DiT.forward()。
        """
        if mask_cond is not None:
            return self.forward_with_mask(x, t, y, mask_cond)
        return super().forward(x, t, y)

    def _apply_mask_condition(self, c_base, mask, block_idx):
        """
        将掩码条件融合到基础条件向量中。
        参数:
            c_base: (B, D) 来自时间 + 文本的基础条件
            mask: (B, 1, H, W) 空间掩码
            block_idx: 当前 DiT 块的索引
        返回:
            c_fused: (B, 6*D) 融合后的 adaLN 参数
        """
        mask_global, mask_spatial = self.mask_encoder(mask)
        # 拼接并融合
        c_combined = torch.cat([c_base, mask_global], dim=-1)
        adaLN_params = self.mask_fusion[block_idx](c_combined)
        return adaLN_params, mask_spatial

    def forward_with_mask(self, x, t, y, mask_cond=None):
        """
        带显式掩码条件的前向传播。
        x: (N, C, H, W) 潜变量输入
        t: (N,) 时间步
        y: 文本嵌入列表 [defect, class, all]
        mask_cond: (N, 1, H, W) 可选的空间掩码条件
        """
        if mask_cond is None:
            mask_cond = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3],
                                    device=x.device)

        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_defect = self.y_embedders(y[0])
        y_class = self.y_embedders(y[1])
        y_all = self.y_embedders(y[2])
        att_map = []
        loss_att = 0

        # 将掩码空间特征上采样到 token 网格尺寸，注入序列作为位置偏置
        mask_global, mask_spatial = self.mask_encoder(mask_cond)  # (B,D), (B,D,2,2)
        # 上采样到 16×16 匹配 DiT 的 patch 网格 (256 tokens)
        mask_spatial_up = F.interpolate(mask_spatial, size=(16, 16), mode='bilinear',
                                        align_corners=False)  # (B, D, 16, 16)
        mask_tokens = mask_spatial_up.flatten(2).permute(0, 2, 1)  # (B, 256, D)
        x = x + 0.1 * mask_tokens  # 微小的空间偏置

        for i in range(28):
            block = self.blocks[i]
            if i < 10:
                c_base = t_emb + y_class
                adaLN, _ = self._apply_mask_condition(c_base, mask_cond, i)
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN.chunk(6, dim=1)
                x = x + gate_msa.unsqueeze(1) * block.attn(
                    modulate(block.norm1(x), shift_msa, scale_msa))
                x = x + gate_mlp.unsqueeze(1) * block.mlp(
                    modulate(block.norm2(x), shift_mlp, scale_mlp))
            elif i < 20:
                cross = self.cross_defect[i - 10]
                c_base = t_emb + y_defect
                adaLN, _ = self._apply_mask_condition(c_base, mask_cond, i)
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN.chunk(6, dim=1)
                x = x + gate_msa.unsqueeze(1) * block.attn(
                    modulate(block.norm1(x), shift_msa, scale_msa))
                cross_att, att_weight = cross(x, c_base)
                loss_att += att_weight
                att_map.append(att_weight)
                x = x + cross_att
                x = x + gate_mlp.unsqueeze(1) * block.mlp(
                    modulate(block.norm2(x), shift_mlp, scale_mlp))
            else:
                c_base = t_emb + y_all
                adaLN, _ = self._apply_mask_condition(c_base, mask_cond, i)
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN.chunk(6, dim=1)
                x = x + gate_msa.unsqueeze(1) * block.attn(
                    modulate(block.norm1(x), shift_msa, scale_msa))
                x = x + gate_mlp.unsqueeze(1) * block.mlp(
                    modulate(block.norm2(x), shift_mlp, scale_mlp))

        x = self.final_layer(x, t_emb + y_all)
        x = self.unpatchify(x)
        att_map = torch.cat(att_map, dim=-1)
        att_mask = self.adapt_mask(att_map)
        loss_att = loss_att.resize(x.shape[0], x.shape[2]//2, x.shape[3]//2, 16).mean(dim=-1)
        return x, att_mask, loss_att

    def forward_with_cfg_mask(self, x, t, y, cfg_scale, mask_cond=None):
        """推理用 CFG + 掩码条件前向传播。"""
        half = x[:len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if mask_cond is not None:
            mask_combined = torch.cat([mask_cond, mask_cond], dim=0)
        else:
            mask_combined = None
        # y = [[cond_defect, cond_class, cond_all], [uncond, uncond_class, uncond_all]]
        # forward_with_mask 期望 [defect, class, all] — 拼接 CFG 对
        y_combined = [
            torch.cat([y[0][0], y[1][0]], dim=0),
            torch.cat([y[0][1], y[1][1]], dim=0),
            torch.cat([y[0][2], y[1][2]], dim=0),
        ]
        model_out, mask, _ = self.forward_with_mask(combined, t, y_combined, mask_combined)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1), mask, _


# ============================================================
#  Mask synthesis utilities for inference
# ============================================================

def generate_random_mask(latent_size, device, mode='blob'):
    """
    为推理生成随机的合成缺陷掩码。
    模式:
      - 'blob': 随机位置的高斯斑点
      - 'line': 线条/划痕形状
      - 'noise': 类柏林噪声阈值
    """
    h = w = latent_size

    if mode == 'blob':
        # 随机位置的高斯斑点
        cx, cy = np.random.uniform(0.2, 0.8, 2)
        sigma_x = np.random.uniform(0.05, 0.2)
        sigma_y = np.random.uniform(0.05, 0.2)
        ys = torch.linspace(0, 1, h, device=device)
        xs = torch.linspace(0, 1, w, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        mask = torch.exp(-((xx - cx)**2 / (2 * sigma_x**2)
                          + (yy - cy)**2 / (2 * sigma_y**2)))
        # 阈值化使其接近二值
        mask = (mask > 0.3).float()

    elif mode == 'line':
        # 随机划痕线条
        mask = torch.zeros(h, w, device=device)
        x0, y0 = np.random.uniform(0.1, 0.3, 2)
        x1, y1 = np.random.uniform(0.7, 0.9, 2)
        t = torch.linspace(0, 1, max(h, w) * 2, device=device)
        xs = x0 + t * (x1 - x0)
        ys = y0 + t * (y1 - y0)
        for i in range(len(t)):
            xi, yi = int(xs[i] * w), int(ys[i] * h)
            if 0 <= xi < w and 0 <= yi < h:
                mask[max(0, yi-1):min(h, yi+2), max(0, xi-1):min(w, xi+2)] = 1.0

    elif mode == 'noise':
        # 随机噪声阈值
        mask = torch.rand(h, w, device=device)
        mask = (mask > 0.85).float()

    else:
        mask = torch.zeros(h, w, device=device)

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
