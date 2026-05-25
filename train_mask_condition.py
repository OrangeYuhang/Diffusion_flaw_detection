"""
DiTMaskConditioned 训练脚本。
通过 ground truth 掩码作为 mask_cond 训练，使模型学会在指定位置生成缺陷。
与 train.py 复用相同的 fg/bg 分离损失、Dice mask loss、正常样本训练、特征空间一致性损失。
"""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models_mask_condition import DiTMaskConditioned
from diffusion import create_diffusion
from autoencoder import *
import clip.clip as clip
from diffusers.models import AutoencoderKL
from torchvision.transforms import Lambda
from torch.utils.data import Dataset
import argparse
import csv
import matplotlib.pyplot as plt
from feature_loss import build_feature_loss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip, _ = clip.load('RN50', device)

    data_path = args.data
    image_size = args.imagesize
    batch_size = args.batchsize
    latent_size = image_size // 8

    # 模型：DiTMaskConditioned（继承自 DiT + MaskEncoder + mask_fusion）
    model = DiTMaskConditioned(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16,
        input_size=latent_size, num_classes=1000).to(device)

    # 从预训练的 DiT 检查点加载权重（strict=False：
    #   标准 DiT 的权重加载到基类部分，
    #   MaskEncoder / mask_fusion / mask_to_tokens 保留随机初始化）
    state_dict = torch.load(args.ckpt, map_location=device)
    # 兼容旧式 dict 格式 checkpoint 和纯 state_dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Init] Loaded {len(state_dict)} keys from pretrained DiT")
    print(f"[Init] Missing keys (random init): {len(missing)}  — MaskEncoder/mask_fusion")
    print(f"[Init] Unexpected keys (ignored): {len(unexpected)}")

    # 两阶段训练：控制基类 DiT 和新模块的冻结/解冻
    BASE_PARAM_NAMES = [
        'blocks', 'x_embedder', 't_embedder', 'y_embedders',
        'cross_defect', 'adapt_mask', 'final_layer',
    ]
    if args.stage == 1:
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in BASE_PARAM_NAMES):
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[Stage 1] Frozen base DiT ({total - trainable:,}/{total:,} params)")
        print(f"[Stage 1] Trainable: MaskEncoder + mask_fusion + mask_to_tokens "
              f"({trainable:,} params)")
    elif args.stage == 2:
        # 确保所有参数可训练
        for p in model.parameters():
            p.requires_grad = True
        print("[Stage 2] All 710M params unfrozen for fine-tuning")

    lr = 1e-4 if args.stage == 1 else 2e-5
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=1e-8)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir) if args.tensorboard else None
    global_step = 0

    # 可选：特征空间一致性损失
    feature_loss_fn = None
    if getattr(args, 'feature_loss', False):
        feature_loss_fn = build_feature_loss(data_path, image_size, device)
        if feature_loss_fn is not None:
            print(f"[FeatureLoss] 启用 — 每 {getattr(args, 'feature_loss_freq', 50)} epoch 计算一次")
        else:
            print("[FeatureLoss] 禁用 — 正常样本不足")

    # 图像变换
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        Lambda(lambda t: (t * 2) - 1),
    ])
    transform_resize_mask = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(latent_size),
        transforms.CenterCrop(latent_size),
    ])
    transform_mask_loss = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(latent_size // 2),
        transforms.CenterCrop(latent_size // 2),
    ])
    # 掩码条件：单通道，潜空间分辨率，值域 [0, 1]
    transform_mask_cond = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(latent_size),
        transforms.CenterCrop(latent_size),
    ])

    class Dataset_MaskCond(Dataset):
        """数据集，额外返回 mask_cond 用于掩码条件训练。"""

        def __init__(self, img_root, preprocess_list, good_ratio=0.3):
            self.img_root = img_root
            self.img_process = preprocess_list  # [img_tr, mask_tr, mask_rsz, mask_ls, mask_cond]
            self.img = []
            self.label_word = []
            self.label_mask = []
            self.is_good = []

            for name_class in os.listdir(self.img_root):
                class_path = os.path.join(self.img_root, name_class)
                if not os.path.isdir(class_path):
                    continue

                # 缺陷样本 — test 目录
                test_path = os.path.join(class_path, 'test')
                if not os.path.isdir(test_path):
                    continue

                for defect_type in os.listdir(test_path):
                    if defect_type == 'good':
                        continue
                    defect_dir = os.path.join(test_path, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue

                    mask_dir = os.path.join(class_path, 'ground_truth', defect_type)
                    if not os.path.isdir(mask_dir):
                        print(f"Warning: {name_class}/{defect_type} 缺少掩码目录")
                        continue

                    for img_name in os.listdir(defect_dir):
                        if not (img_name.endswith('.png') or img_name.endswith('.jpg')):
                            continue
                        img_path = os.path.join(defect_dir, img_name)
                        mask_name = img_name.replace('.png', '_mask.png')
                        mask_path = os.path.join(mask_dir, mask_name)
                        if not os.path.exists(mask_path):
                            mask_path = os.path.join(mask_dir, img_name)
                        if not os.path.exists(mask_path):
                            continue
                        self.img.append(img_path)
                        self.label_word.append(f"{defect_type} {name_class}")
                        self.label_mask.append(mask_path)
                        self.is_good.append(False)

                # 正常样本 — train/good 目录
                train_good = os.path.join(class_path, 'train', 'good')
                if os.path.isdir(train_good):
                    good_imgs = [f for f in os.listdir(train_good)
                                 if f.endswith('.png') or f.endswith('.jpg')]
                    n_defect_class = sum(
                        1 for i, g in enumerate(self.is_good)
                        if not g and self.label_word[i].endswith(name_class))
                    n_good_max = max(len(good_imgs), int(n_defect_class * good_ratio)) if self.img else len(good_imgs)
                    n_good_max = max(1, min(n_good_max, len(good_imgs)))
                    for i, img_name in enumerate(good_imgs[:n_good_max]):
                        self.img.append(os.path.join(train_good, img_name))
                        self.label_word.append(f"good {name_class}")
                        self.label_mask.append(None)
                        self.is_good.append(True)

            n_defect = sum(1 for g in self.is_good if not g)
            n_good = sum(1 for g in self.is_good if g)
            print(f"Loaded {n_defect} defect + {n_good} normal samples")

        def __len__(self):
            return len(self.img)

        def __getitem__(self, idx):
            img_path = self.img[idx]
            image = Image.open(img_path).convert('RGB')
            image = self.img_process[0](image)

            if self.is_good[idx]:
                # 正常样本：零掩码（无缺陷），mask_cond 也为零
                dummy = Image.new('L', (256, 256), 0)
                label_mask = self.img_process[1](dummy)
                mask_resize = self.img_process[2](dummy)
                mask_loss_raw = self.img_process[3](dummy)
                mask_loss = mask_loss_raw[0, :, :]
                mask_loss.zero_()
                mask_resize_res = torch.cat(
                    [mask_resize, mask_resize[0, :, :].unsqueeze(0)], dim=0)
                # mask_cond: (1, 32, 32) 全零
                mask_cond = self.img_process[4](dummy)
                return image, self.label_word[idx], label_mask, mask_resize_res, mask_loss, mask_cond
            else:
                label_mask_path = self.label_mask[idx]
                label_mask_img = Image.open(label_mask_path)
                label_mask = self.img_process[1](label_mask_img)
                mask_resize = self.img_process[2](label_mask_img)
                mask_loss = self.img_process[3](label_mask_img)
                mask_loss = mask_loss[0, :, :]
                mask_loss[mask_loss != 0] = 1
                mask_resize_res = torch.cat(
                    [mask_resize, mask_resize[0, :, :].unsqueeze(0)], dim=0)
                # mask_cond: (1, 32, 32) ground truth 掩码缩放到潜空间
                mask_cond = self.img_process[4](label_mask_img)
                # 二值化：值 > 0.3 设为 1，其余为 0
                mask_cond = (mask_cond > 0.3).float()
                return image, self.label_word[idx], label_mask, mask_resize_res, mask_loss, mask_cond

    dataset = Dataset_MaskCond(
        img_root=data_path,
        preprocess_list=[transform, transform_mask, transform_resize_mask,
                         transform_mask_loss, transform_mask_cond],
        good_ratio=args.good_ratio)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model.train()

    if args.bf16:
        class _BF16Wrapper(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base_model = base
            def forward(self, *a, **kw):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = self.base_model(*a, **kw)
                if isinstance(out, tuple):
                    return tuple(o.float() if torch.is_tensor(o) else o for o in out)
                return out.float()
        train_model = _BF16Wrapper(model)
    else:
        train_model = model

    EPOCH = 1501
    amp_enabled = 'BF16 AMP' if args.bf16 else 'FP32'
    lr_str = f'lr={lr:.0e}'
    print(f'===== 掩码条件 DiT Stage {args.stage} ({amp_enabled}, {lr_str}) =====')
    losses = []

    csv_file = open(f'loss_log_mask_cond_s{args.stage}.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = ['epoch', 'loss', 'vb', 'mse', 'mask_mse', 'mask_dice', 'att_align']
    if feature_loss_fn is not None:
        csv_header.append('feat')
    csv_writer.writerow(csv_header)
    csv_file.flush()

    for epoch in range(EPOCH):
        epoch_losses = []
        epoch_components = {}
        for x, y, mask, mask_resize, mask_loss, mask_cond in loader:
            x = x.to(device)
            mask = mask.to(device)
            mask_resize = mask_resize.to(device)
            mask_loss = mask_loss.to(device)
            mask_cond = mask_cond.to(device)

            # —— CFG 文本 dropout（与 train.py 相同） ——
            drop_rat = 0.2
            if args.free == 2:
                y = list(y)
                for i in range(len(y)):
                    c = y[i]
                    if c.split()[0] == 'good':
                        if torch.rand(1) < drop_rat:
                            y[i] = 'good industry'
                    else:
                        if torch.rand(1) < drop_rat:
                            y[i] = 'good ' + c.split()[1]
            else:
                y = list(y)
                for i in range(len(y)):
                    c = y[i]
                    if c.split()[0] != 'good':
                        if torch.rand(1) < drop_rat:
                            y[i] = 'good ' + c.split()[1]

            defect = torch.cat(
                [clip.tokenize(f"a photo of {c.split()[0]}") for c in y]).to(device)
            classes = torch.cat(
                [clip.tokenize(f"a photo of {c.split()[1]}") for c in y]).to(device)
            y_all = torch.cat(
                [clip.tokenize(f"a photo of {c}") for c in y]).to(device)

            with torch.no_grad():
                defect = model_clip.encode_text(defect)
                classes = model_clip.encode_text(classes)
                y_all = model_clip.encode_text(y_all)

            defect /= defect.norm(dim=-1, keepdim=True)
            defect = defect.float().to(device)
            classes /= classes.norm(dim=-1, keepdim=True)
            classes = classes.float().to(device)
            y_all /= y_all.norm(dim=-1, keepdim=True)
            y_all = y_all.float().to(device)

            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                if mask.shape[1] == 1:
                    mask = mask.repeat(1, 3, 1, 1)
                mask_gt = vae.encode(mask).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            # mask_cond 传入 model_kwargs，DiTMaskConditioned.forward() 自动识别
            model_kwargs = dict(
                y=[defect, classes, y_all],
                mask_cond=mask_cond)

            if mask_resize.shape[1] != x.shape[1]:
                mask_resize = mask_resize.repeat(
                    1, x.shape[1] // mask_resize.shape[1], 1, 1)

            loss_dict = diffusion.training_losses(
                train_model, x, t, model_kwargs,
                mask_resize=mask_resize,
                mask_att=mask_loss,
                label_mask=mask_gt)
            loss = loss_dict["loss"].mean()

            # 特征空间一致性损失
            if (feature_loss_fn is not None and epoch > 0
                    and epoch % getattr(args, 'feature_loss_freq', 50) == 0):
                with torch.no_grad():
                    img_decoded = vae.decode(x.detach() / 0.18215).sample
                loss_feat = feature_loss_fn(img_decoded, mask_resize.detach())
                loss = loss + 0.05 * loss_feat
                loss_dict["feat"] = loss_feat.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            if epoch % 10 == 0:
                loss_parts = {
                    k: v.mean().item() if isinstance(v, torch.Tensor) else v
                    for k, v in loss_dict.items()}
                print(f"epoch {epoch}: loss={loss.item():.4f}  {loss_parts}")

            # TensorBoard 记录（异步写入，影响可忽略）
            if writer is not None:
                writer.add_scalar("Loss/total", loss.item(), global_step)
                for k, v in loss_dict.items():
                    val = v.mean().item() if isinstance(v, torch.Tensor) else v
                    if val == val:  # 跳过 NaN
                        writer.add_scalar(f"Loss/{k}", val, global_step)
                global_step += 1

            losses.append(loss.item())
            epoch_losses.append(loss.item())
            for k, v in loss_dict.items():
                val = v.mean().item() if isinstance(v, torch.Tensor) else v
                epoch_components.setdefault(k, []).append(val)

        # 每个 epoch 结束：计算均值并写入 CSV
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        row = [epoch, avg_loss]
        for key in csv_header[2:]:
            vals = epoch_components.get(key)
            row.append(sum(vals) / len(vals) if vals else 0.0)
        csv_writer.writerow(row)
        csv_file.flush()
        print(f"epoch {epoch}: avg_loss={avg_loss:.4f}")

        if epoch % 100 == 0 and 2000 >= epoch >= 100:
            torch.save(model.state_dict(),
                       f'checkpoint/model_mask_cond_s{args.stage}_{epoch}.pth')
            print(f"  -> saved checkpoint/model_mask_cond_s{args.stage}_{epoch}.pth")

    # 最终保存
    final_name = f'checkpoint/model_mask_cond_s{args.stage}_final.pth'
    torch.save(model.state_dict(), final_name)
    print(f"===== 训练完成 -> {final_name} =====")

    if writer is not None:
        writer.close()

    csv_file.close()
    print(f"损失日志已保存至 loss_log_mask_cond_s{args.stage}.csv")

    # 损失曲线
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DiTMaskConditioned Training Loss')
    plt.grid()
    plt.savefig('./img/loss_curve_mask_cond.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--free", type=int, default=1,
                        help="0/1/2 — CFG 分支策略")
    parser.add_argument("--data", type=str, required=True,
                        help="MVTec AD 数据集根路径")
    parser.add_argument("--imagesize", type=int, choices=[256, 512], default=256)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="预训练 DiT 检查点（基类权重初始化）")
    parser.add_argument("--vae", type=str, required=True,
                        help="VAE 检查点路径")
    parser.add_argument("--good_ratio", type=float, default=0.3,
                        help="正常样本相对缺陷样本的比例")
    parser.add_argument("--feature_loss", action="store_true", default=False,
                        help="启用 PatchCore 对齐的特征空间一致性损失")
    parser.add_argument("--feature_loss_freq", type=int, default=50,
                        help="特征空间损失的 epoch 间隔")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 工作进程数。云端/共享存储建议 0（避免 NFS 死锁），本地可设 2-4。")
    parser.add_argument("--tensorboard", action="store_true", default=False,
                        help="启用 TensorBoard 实时监控损失曲线。")
    parser.add_argument("--log_dir", type=str, default="./runs/train_mask",
                        help="TensorBoard 日志目录。")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="启用 BF16 自动混合精度（减少约 30%% 显存，速度提升 15-25%%，不影响精度）。")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1,
                        help="两阶段训练: 1=冻结基类DiT仅训练新模块(lr=5e-4), "
                             "2=全参数微调(lr=2e-5, 需阶段一的checkpoint)")
    args = parser.parse_args()
    main(args)
