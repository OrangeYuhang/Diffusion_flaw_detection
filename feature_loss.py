"""
PatchCore 对齐的特征空间一致性损失。
使用 WideResNet-50 作为特征提取器（与 PatchCore 默认骨干网络一致），
惩罚生成图像中非缺陷区域与真实正常样本特征分布之间的偏差。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from collections import deque
import numpy as np


class PatchCoreFeatureExtractor(nn.Module):
    """
    从 WideResNet-50 提取中间层特征，模拟 PatchCore
    的特征提取流程（第 2 层和第 3 层）。
    """
    def __init__(self, device="cuda"):
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        backbone = wide_resnet50_2(weights=weights)

        # 提取中间层（PatchCore 使用 layer2 和 layer3）
        self.layer1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu,
            backbone.maxpool, backbone.layer1
        )
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for p in self.parameters():
            p.requires_grad = False
        self.to(device)
        self.eval()

        self._device = device

    @torch.no_grad()
    def forward(self, x):
        """
        参数:
            x: (B, 3, H, W) 归一化图像，值域 [-1, 1]（VAE 解码输出）
        返回:
            features: (B, C, H', W') layer2 和 layer3 特征的拼接
        """
        # VAE 输出在 [-1, 1] 范围；WideResNet 需要 ImageNet 归一化
        x = (x + 1.0) / 2.0  # -> [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        f1 = self.layer1(x)        # (B, 256, H/4, W/4)
        f2 = self.layer2(f1)       # (B, 512, H/8, W/8)
        f3 = self.layer3(f2)       # (B, 1024, H/16, W/16)

        # 将 f3 上采样到 f2 的空间尺寸并拼接
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        features = torch.cat([f2, f3_up], dim=1)  # (B, 1536, H/8, W/8)
        return features


def compute_normal_statistics(extractor, normal_loader, device, max_samples=200):
    """
    为正常（无缺陷）样本预计算特征均值和精度矩阵。
    这是训练开始前的一次性开销。
    """
    features_list = []
    count = 0
    for batch in normal_loader:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)
        feat = extractor(x)
        # 平均池化到 (B, C, 1, 1) -> (B, C)
        feat_pooled = feat.mean(dim=[2, 3])
        features_list.append(feat_pooled.cpu().numpy())
        count += x.shape[0]
        if count >= max_samples:
            break

    features_all = np.concatenate(features_list, axis=0)  # (N, C)
    mean = torch.from_numpy(features_all.mean(axis=0)).float().to(device)
    # 协方差矩阵（用于马氏距离）
    cov = np.cov(features_all.T)
    # 正则化：特征维度高（1536），样本量少（≤200），协方差严重秩亏
    # 使用更大的正则化保证数值稳定性和正定性
    cov += np.eye(cov.shape[0]) * 1e-2
    # 使用稳定 Cholesky 分解求逆，避免 np.linalg.inv 的数值不稳定性
    L = np.linalg.cholesky(cov)
    inv_cov = np.linalg.inv(L.T) @ np.linalg.inv(L)
    precision = torch.from_numpy(inv_cov).float().to(device)
    return mean, precision


class FeatureConsistencyLoss(nn.Module):
    """
    惩罚生成图像中非缺陷区域与预计算正常样本统计量
    在特征空间中的距离。
    """
    def __init__(self, extractor, normal_mean, normal_precision):
        super().__init__()
        self.extractor = extractor
        self.register_buffer('normal_mean', normal_mean)
        self.register_buffer('normal_precision', normal_precision)

    def forward(self, generated_img, mask_resize):
        """
        参数:
            generated_img: (B, 3, H, W) 解码后的图像，值域 [-1, 1]
            mask_resize: (B, C, H', W') 潜空间中的掩码，将被调整尺寸
        返回:
            标量损失值
        """
        B = generated_img.shape[0]
        features = self.extractor(generated_img)  # (B, C_feat, H_f, W_f)
        C_f, H_f, W_f = features.shape[1], features.shape[2], features.shape[3]

        # 将掩码调整到特征图的空间尺寸
        # mask_resize 各通道有独立信息；在通道维度取平均
        mask_spatial = mask_resize.mean(dim=1, keepdim=True)  # (B, 1, H_m, W_m)
        mask_f = F.interpolate(mask_spatial, size=(H_f, W_f),
                               mode='bilinear', align_corners=False)  # (B, 1, H_f, W_f)

        # 特征加权：背景（非缺陷）区域权重高，缺陷区域权重低
        bg_weight = 1.0 - mask_f.clamp(0.0, 1.0)  # (B, 1, H_f, W_f)

        # 每个样本的加权平均特征
        weighted_feat = (features * bg_weight).sum(dim=[2, 3]) / (bg_weight.sum(dim=[2, 3]) + 1e-8)  # (B, C_f)

        # 到正常样本统计量的马氏距离
        diff = weighted_feat - self.normal_mean.unsqueeze(0)  # (B, C_f)
        # 二次型: diffᵀ × precision × diff，逐个样本计算
        # (B, 1, C_f) @ (C_f, C_f) × (B, C_f, 1) → (B,)
        quad = (diff.unsqueeze(1) @ self.normal_precision @ diff.unsqueeze(2)).squeeze(-1).squeeze(-1)
        # clamp: 精度矩阵数值误差可能导致二次型为微小负数
        mahalanobis = torch.sqrt(quad.clamp(min=0)).mean()

        return mahalanobis


def build_feature_loss(data_path, image_size, device):
    """
    通过从 MVTec train/good 目录预计算正常样本统计量来构建 FeatureConsistencyLoss。
    如果正常样本数量不足，返回 None。
    """
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image
    import os

    class NormalOnlyDataset(Dataset):
        def __init__(self, root, img_size):
            self.paths = []
            for cls in os.listdir(root):
                good_dir = os.path.join(root, cls, 'train', 'good')
                if os.path.isdir(good_dir):
                    for f in os.listdir(good_dir):
                        if f.endswith(('.png', '.jpg')):
                            self.paths.append(os.path.join(good_dir, f))
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ])

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img)

    try:
        dataset = NormalOnlyDataset(data_path, image_size)
        if len(dataset) < 10:
            print("[FeatureLoss] Warning: < 10 normal samples found, skipping feature loss.")
            return None
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

        extractor = PatchCoreFeatureExtractor(device)
        mean, precision = compute_normal_statistics(extractor, loader, device)
        print(f"[FeatureLoss] Normal statistics computed from {len(dataset)} samples.")
        return FeatureConsistencyLoss(extractor, mean, precision)
    except Exception as e:
        print(f"[FeatureLoss] Failed to build: {e}")
        return None
