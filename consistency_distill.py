"""
DefectDiffu 的一致性蒸馏。
使用一致性模型风格的训练，将 50 步 DDPM 采样器蒸馏为 1-4 步生成。

参考文献: Song et al., "Consistency Models" (ICML 2023)
"""
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from models_add_cross_concate import DiT
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import clip.clip as clip


# ============================================================
#  Consistency training utilities
# ============================================================

def get_skip_schedule(num_train_timesteps, num_student_timesteps):
    """
    将 1000 个教师时间步映射到 N 个学生时间步。
    使用指数调度，重点关注一致性难度较高的高噪声水平。
    """
    # 指数调度：低 t（干净）时更密集，高 t（噪声）时更稀疏
    t = np.linspace(0, num_train_timesteps - 1, num_student_timesteps + 1)
    return (t[:-1] + t[1:]) / 2  # 中点


def consistency_loss(student_model, teacher_model, x, t_student, t_teacher,
                     model_kwargs, diffusion):
    """
    一致性蒸馏损失。
    对于相邻时间步对 (t_s, t_t)，其中 t_s < t_t：
      L = MSE(student(x_{t_s}, t_s), teacher(x_{t_t}, t_t))
    教师从较高噪声去噪；学生从较低噪声去噪。
    两者应预测相同的干净 x_0。
    """
    B = x.shape[0]
    device = x.device

    # 在相邻噪声级别采样噪声
    noise = torch.randn_like(x)

    # 较高噪声级别（教师）
    x_t_teacher = diffusion.q_sample(x, t_teacher, noise=noise)
    # 较低噪声级别（学生）— 使用相同噪声方向以保持一致性
    x_t_student = diffusion.q_sample(x, t_student, noise=noise)

    # 教师预测（无梯度）
    with torch.no_grad():
        teacher_out, _, _ = teacher_model(x_t_teacher, t_teacher, **model_kwargs)
        # 提取 epsilon 预测
        teacher_eps = teacher_out[:, :x.shape[1]]

    # 学生预测
    student_out, _, _ = student_model(x_t_student, t_student, **model_kwargs)
    student_eps = student_out[:, :x.shape[1]]

    # 两者都预测 epsilon；一致性意味着它们应该匹配
    loss = nn.functional.mse_loss(student_eps, teacher_eps)
    return loss


def boundary_loss_fn(student_model, x, model_kwargs):
    """边界条件：在 t=0 时，学生应输出干净的 x。"""
    t_zero = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
    out, _, _ = student_model(x, t_zero, **model_kwargs)
    # 对于 epsilon 预测，t=0 时预测噪声应为 0
    eps_pred = out[:, :x.shape[1]]
    return nn.functional.mse_loss(eps_pred, torch.zeros_like(eps_pred))


# ============================================================
#  Fast sampler (post-distillation)
# ============================================================

@torch.no_grad()
def fast_sample(student_model, diffusion, shape, model_kwargs,
                num_steps=4, device="cuda"):
    """
    使用蒸馏后的一致性模型进行快速采样。
    使用简单的 Euler 类方案进行 N 步采样。
    """
    b, c, h, w = shape
    x = torch.randn(b, c, h, w, device=device)

    # 将 N 步映射到教师时间步
    timesteps = torch.linspace(diffusion.num_timesteps - 1, 0, num_steps + 1).long()

    for i in range(num_steps):
        t = torch.full((b,), timesteps[i], device=device, dtype=torch.long)
        t_next = torch.full((b,), timesteps[i + 1], device=device, dtype=torch.long)

        out, mask, att = student_model(x, t, **model_kwargs)
        eps = out[:, :c]

        # DDIM 风格的步进
        alpha_bar_t = diffusion._extract(diffusion.alphas_cumprod, t, x.shape)
        alpha_bar_t_next = diffusion._extract(diffusion.alphas_cumprod, t_next, x.shape)

        # 从 epsilon 预测 x0
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        # 步进到下一个时间步
        x = torch.sqrt(alpha_bar_t_next) * x0_pred + torch.sqrt(1 - alpha_bar_t_next) * eps

    return x, mask


def _extract(arr, timesteps, broadcast_shape):
    """辅助函数：从数组中提取指定时间步的值。"""
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res


# Monkey-patch GaussianDiffusion 添加 _extract 方法供采样使用
from diffusion.gaussian_diffusion import GaussianDiffusion
GaussianDiffusion._extract = staticmethod(_extract)


# ============================================================
#  Distillation training loop
# ============================================================

class MVTecDistillDataset(Dataset):
    """用于蒸馏的轻量级数据集 — 仅使用 train/good 正常图像。"""
    def __init__(self, data_path, image_size=256):
        self.img_paths = []
        for cls in os.listdir(data_path):
            good_dir = os.path.join(data_path, cls, 'train', 'good')
            if not os.path.isdir(good_dir):
                continue
            for f in os.listdir(good_dir):
                if f.endswith(('.png', '.jpg')):
                    self.img_paths.append(os.path.join(good_dir, f))
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="Path to trained DefectDiffu checkpoint")
    parser.add_argument("--vae", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out_ckpt", type=str, default="./checkpoint/distilled.pth")
    parser.add_argument("--num_student_steps", type=int, default=4,
                        help="Target number of sampling steps")
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 进程数（云端建议 0）")
    parser.add_argument("--lambda_boundary", type=float, default=0.1,
                        help="Weight of boundary condition loss")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    latent_size = args.imagesize // 8

    # 加载教师模型
    print("Loading teacher model...")
    teacher = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                  input_size=latent_size, num_classes=1000).to(device)
    state_dict = torch.load(args.teacher_ckpt, map_location=device)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    teacher.load_state_dict(state_dict, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Teacher frozen.")

    # 创建学生模型（相同架构，可训练）
    print("Creating student model...")
    student = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                  input_size=latent_size, num_classes=1000).to(device)
    # 用教师权重初始化学生以获得更快的收敛
    student_state = torch.load(args.teacher_ckpt, map_location=device)
    if isinstance(student_state, dict) and 'model_state_dict' in student_state:
        student_state = student_state['model_state_dict']
    student.load_state_dict(student_state, strict=False)
    student.train()

    # VAE 和 CLIP
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    vae.eval()
    model_clip, _ = clip.load('RN50', device)

    # 教师使用全时间步扩散
    diffusion_full = create_diffusion(timestep_respacing="")

    # 学生时间步调度
    student_ts = get_skip_schedule(
        diffusion_full.num_timesteps, args.num_student_steps)
    student_ts = torch.from_numpy(student_ts).long().to(device)
    print(f"Student timesteps: {student_ts.tolist()}")

    # 数据集
    dataset = MVTecDistillDataset(args.data, args.imagesize)
    loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-8)

    print(f"Starting distillation for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            x = batch.to(device)

            # VAE 编码到潜空间
            with torch.no_grad():
                x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
                # 用于蒸馏的虚拟文本嵌入（我们蒸馏的是无/类别条件行为）
                dummy_text = clip.tokenize(["a photo of industry"] * x.shape[0]).to(device)
                text_emb = model_clip.encode_text(dummy_text).float()
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            B = x_latent.shape[0]
            model_kwargs = {'y': [text_emb, text_emb, text_emb]}

            # 采样一对相邻时间步
            idx = torch.randint(0, len(student_ts) - 1, (1,)).item()
            t_student = student_ts[idx]  # 较低噪声
            t_teacher = student_ts[idx + 1]  # 较高噪声
            t_s = torch.full((B,), t_student, device=device)
            t_t = torch.full((B,), t_teacher, device=device)

            # 一致性损失
            loss = consistency_loss(
                student, teacher, x_latent, t_s, t_t,
                model_kwargs, diffusion_full)

            # 边界条件（每 10 步计算一次以节省计算量）
            if torch.rand(1).item() < 0.1:
                loss_boundary = boundary_loss_fn(student, x_latent, model_kwargs)
                loss = loss + args.lambda_boundary * loss_boundary

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.6f}")

        # 保存检查点
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': avg_loss,
                'student_timesteps': student_ts.cpu().tolist(),
            }, args.out_ckpt.replace('.pth', f'_e{epoch}.pth'))
            print(f"  Saved checkpoint.")

    # 最终保存
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': student.state_dict(),
        'student_timesteps': student_ts.cpu().tolist(),
    }, args.out_ckpt)
    print(f"Distillation complete. Model saved to {args.out_ckpt}")


if __name__ == "__main__":
    main()
