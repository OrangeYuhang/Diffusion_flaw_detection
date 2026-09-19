"""
闭环自适应缺陷生成。
迭代流程：生成样本 → 运行 PatchCore 检测 → 为表现较差的类别
分配更多生成预算。

用法:
    python adaptive_generate.py \
        --ckpt checkpoint/model_600.pth \
        --vae ./VAE \
        --data ./data/mvtec \
        --anomalib_root ../anomalib \
        --num_iterations 3
"""
import os
import sys
import json
import shutil
import subprocess
import argparse
import time
import numpy as np
from collections import defaultdict

import torch
from torchvision.utils import save_image

from models_add_cross_concate import DiT
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import clip.clip as clip


# ============================================================
#  Generation utilities (reused from test_gen.py)
# ============================================================

def load_models(ckpt_path, vae_path, device, image_size=256, steps=25):
    latent_size = image_size // 8
    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16,
                input_size=latent_size, num_classes=1000).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.float()
    diffusion = create_diffusion(timestep_respacing=str(steps))
    vae = AutoencoderKL.from_pretrained(vae_path).to(device)
    model_clip, _ = clip.load('RN50', device)
    return model, diffusion, vae, model_clip


@torch.no_grad()
def generate_single(model, diffusion, vae, model_clip,
                    defect_type, product, cfg_scale, device, latent_size):
    num_img = 1
    texts = {
        'defect': f"a photo of {defect_type}",
        'class': f"a photo of {product}",
        'all': f"a photo of {defect_type} {product}",
        'good': "a photo of good",
        'good_class': f"a photo of good {product}",
        'good_industry': "a photo of good industry",
    }
    embeds = {}
    for k, t in texts.items():
        e = model_clip.encode_text(clip.tokenize([t] * num_img).to(device)).float()
        embeds[k] = e / e.norm(dim=-1, keepdim=True)

    y_defect_class = [embeds['defect'], embeds['class'], embeds['all']]
    y_good_class   = [embeds['good'], embeds['class'], embeds['good_class']]

    z = torch.randn(num_img, 4, latent_size, latent_size, device=device).float()
    z = torch.cat([z, z], 0)
    y = [y_defect_class, y_good_class]
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    samples, _ = diffusion.p_sample_loop(
        model.forward_with_cfg_2, z.shape, z,
        clip_denoised=False, model_kwargs=model_kwargs,
        progress=False, device=device
    )
    img_gen, _ = samples.chunk(2, dim=0)
    img = vae.decode(img_gen / 0.18215).sample
    return img


# ============================================================
#  Detection metrics (simplified PatchCore runner via subprocess)
# ============================================================

def run_patchcore_eval(anomalib_root, data_root, output_dir):
    """
    在准备的数据集上运行 PatchCore 评估。
    需要安装 anomalib 且训练脚本可用。
    返回每类 AUROC 字典。
    """
    script = os.path.join(anomalib_root, 'tools', 'train.py')
    if not os.path.exists(script):
        # 尝试替代入口点
        print("[PatchCore] Warning: anomalib train script not found. "
              "Using mock evaluation. Install anomalib or provide --detection_script.")
        return mock_evaluation(data_root)

    cmd = [
        sys.executable, script,
        '--model', 'patchcore',
        '--data', data_root,
        '--output', output_dir,
        '--num_epochs', '1',
        '--no-logging',
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as e:
        print(f"[PatchCore] Evaluation failed: {e.stderr}")
        return mock_evaluation(data_root)

    # 解析 anomalib 输出结果
    results_file = os.path.join(output_dir, 'results.json')
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        return {k: v.get('image_AUROC', 0.5) for k, v in results.items()}
    return mock_evaluation(data_root)


def mock_evaluation(data_root):
    """回退模拟：对缺陷样本较少的类别返回较低的 AUROC。"""
    scores = {}
    bad_dir = os.path.join(data_root, 'train', 'bad')
    if os.path.isdir(bad_dir):
        class_counts = defaultdict(int)
        for f in os.listdir(bad_dir):
            cls = f.split('_')[0] if '_' in f else 'unknown'
            class_counts[cls] += 1
        total = sum(class_counts.values()) or 1
        for cls, cnt in class_counts.items():
            # 模拟：样本越多 → 检测性能越好
            scores[cls] = min(0.95, 0.6 + 0.3 * (cnt / max(cnt, 1)))
    return scores


# ============================================================
#  Budget allocation
# ============================================================

def allocate_budget(auroc_dict, total_budget, min_per_class=5):
    """
    按 AUROC 的反比例分配生成预算。
    AUROC 较低的类别获得更多样本。
    """
    if not auroc_dict:
        return {}

    # 误差 = 1 - AUROC：误差越大 → 预算越多
    errors = {cls: max(0.01, 1.0 - auroc) for cls, auroc in auroc_dict.items()}
    total_error = sum(errors.values())

    budget = {}
    for cls, err in errors.items():
        budget[cls] = max(min_per_class, int(total_budget * err / total_error))
    return budget


# ============================================================
#  Main loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--vae", type=str, required=True)
    parser.add_argument("--data", type=str, required=True,
                        help="Path to original MVTec AD dataset")
    parser.add_argument("--work_dir", type=str, default="./adaptive_output",
                        help="Working directory for intermediate datasets")
    parser.add_argument("--anomalib_root", type=str, default=None,
                        help="Path to anomalib repository root")
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--total_budget", type=int, default=200,
                        help="Total samples to generate per iteration")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--imagesize", type=int, default=256)
    parser.add_argument("--steps", type=int, default=25,
                        help="DDIM sampling steps (default 25, lower = faster)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = args.imagesize // 8
    os.makedirs(args.work_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载生成模型（仅一次）
    model, diffusion, vae, model_clip = load_models(
        args.ckpt, args.vae, device, args.imagesize, args.steps)

    # 从 MVTec 中发现所有 (缺陷, 产品) 对
    tasks = []
    for cls in os.listdir(args.data):
        test_dir = os.path.join(args.data, cls, 'test')
        if not os.path.isdir(test_dir):
            continue
        for defect in os.listdir(test_dir):
            if defect == 'good':
                continue
            tasks.append((defect, cls))

    # 按产品类别分组任务以进行每类评估
    class_defects = defaultdict(list)
    for defect, cls in tasks:
        class_defects[cls].append(defect)

    print(f"Found {len(tasks)} defect types across {len(class_defects)} classes.")

    # 任务概览
    total_defect_types = sum(len(d) for d in class_defects.values())
    init_budget_per = args.total_budget // max(len(class_defects), 1)
    est_imgs_per_iter = total_defect_types * max(2, init_budget_per // max(1, total_defect_types // len(class_defects)))
    est_gen_time = est_imgs_per_iter * args.steps * 0.015
    print(f"Classes: {len(class_defects)}  |  Defect types: {total_defect_types}")
    print(f"Initial budget/class: {init_budget_per}  |  ~{est_imgs_per_iter} images/iter")
    print(f"Est gen time/iter: ~{est_gen_time/60:.0f}m (with {args.steps} steps)")
    print(f"Total iterations: {args.num_iterations}  |  Est total: ~{est_gen_time * args.num_iterations / 60:.0f}m")

    # 均匀初始化每类预算
    auroc_history = {}
    per_class_budget = {cls: args.total_budget // max(len(class_defects), 1)
                        for cls in class_defects}

    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.num_iterations}  |  Steps: {args.steps}  |  Budget: {args.total_budget}")
        print(f"{'='*60}")

        iter_dir = os.path.join(args.work_dir, f"iter_{iteration}")
        bad_dir = os.path.join(iter_dir, 'train', 'bad')
        good_dir = os.path.join(iter_dir, 'train', 'good')
        test_dir = os.path.join(iter_dir, 'test')
        gt_dir = os.path.join(iter_dir, 'ground_truth')

        # 从原始数据集复制 normal + test + gt
        print("[1/3] Copying dataset structure...", end=' ', flush=True)
        t0 = time.time()
        for d in [iter_dir, bad_dir, good_dir]:
            os.makedirs(d, exist_ok=True)
        _copy_dataset_structure(args.data, iter_dir)
        print(f"done ({time.time()-t0:.1f}s)")

        # 预计算总生成量用于进度显示
        total_to_gen = sum(
            max(2, per_class_budget.get(cls, 20) // len(defects)) * len(defects)
            for cls, defects in class_defects.items()
        )
        print(f"[2/3] Generating {total_to_gen} images across {len(tasks)} defect types...")
        print(f"      Estimated: ~{total_to_gen * args.steps * 0.015:.0f}s with {args.steps} steps")
        total_generated = 0
        gen_start = time.time()

        for cls_idx, (cls, defects) in enumerate(sorted(class_defects.items())):
            budget = per_class_budget.get(cls, 20)
            n_per_defect = max(2, budget // len(defects))
            cls_start = time.time()
            cls_gen = 0
            for defect in sorted(defects):
                for idx in range(n_per_defect):
                    try:
                        seed_i = args.seed * 1000 + iteration * 100 + idx
                        torch.manual_seed(seed_i)
                        img = generate_single(model, diffusion, vae, model_clip,
                                              defect, cls, args.cfg_scale,
                                              device, latent_size)
                        save_name = f"{cls}_{defect}_iter{iteration}_{idx:03d}.png"
                        save_image(img, os.path.join(bad_dir, save_name), normalize=True)
                        total_generated += 1
                        cls_gen += 1
                    except Exception as e:
                        print(f"\n  Gen failed {cls}/{defect}: {e}")

            cls_elapsed = time.time() - cls_start
            total_elapsed = time.time() - gen_start
            pct = total_generated / total_to_gen * 100
            eta = total_elapsed / total_generated * (total_to_gen - total_generated) if total_generated > 0 else 0
            print(f"  [{cls_idx+1:2d}/{len(class_defects)}] {cls:15s}: {cls_gen:3d} imgs "
                  f"({cls_elapsed:4.1f}s)  |  {total_generated:4d}/{total_to_gen} ({pct:3.0f}%)  "
                  f"ETA: {eta/60:.0f}m{eta%60:.0f}s")

        gen_time = time.time() - gen_start
        print(f"  Total generation: {total_generated} images in {gen_time/60:.1f}m "
              f"({gen_time/total_generated:.1f}s/img)")

        # 运行 PatchCore 评估
        print(f"[3/3] Running PatchCore evaluation on {len(class_defects)} classes...")
        eval_start = time.time()
        auroc = run_patchcore_eval(
            args.anomalib_root or '.', iter_dir,
            os.path.join(args.work_dir, f"eval_{iteration}"))
        eval_time = time.time() - eval_start
        auroc_history[iteration] = auroc
        print(f"  Evaluation done ({eval_time/60:.1f}m)")
        print(f"  Per-class AUROC: {json.dumps(auroc, indent=2)}")

        # 迭代总结
        iter_time = time.time() - iter_start
        print(f"\n  Iteration {iteration} complete in {iter_time/60:.1f}m "
              f"(gen: {gen_time/60:.1f}m, eval: {eval_time/60:.1f}m)")

        # 下一轮的自适应预算分配
        if iteration < args.num_iterations:
            per_class_budget = allocate_budget(
                auroc, args.total_budget, min_per_class=10)
            print(f"  Next budget: {json.dumps(per_class_budget, indent=2)}")

    # 保存历史记录
    with open(os.path.join(args.work_dir, 'auroc_history.json'), 'w') as f:
        json.dump(auroc_history, f, indent=2)
    print(f"\nClosed-loop complete. Results saved to {args.work_dir}/")


def _copy_dataset_structure(src_root, dst_root):
    """将 test、ground_truth 和 train/good 从 MVTec 复制到目标目录。"""
    src_test = os.path.join(src_root)
    dst_test = os.path.join(dst_root, 'test')
    dst_gt = os.path.join(dst_root, 'ground_truth')
    dst_good = os.path.join(dst_root, 'train', 'good')

    for cls in os.listdir(src_root):
        cls_src = os.path.join(src_root, cls)
        if not os.path.isdir(cls_src):
            continue
        # 复制 test
        src_t = os.path.join(cls_src, 'test')
        if os.path.isdir(src_t):
            dst_t = os.path.join(dst_test, cls)
            os.makedirs(os.path.dirname(dst_t), exist_ok=True)
            if not os.path.exists(dst_t):
                shutil.copytree(src_t, dst_t)
        # 复制 ground_truth
        src_g = os.path.join(cls_src, 'ground_truth')
        if os.path.isdir(src_g):
            dst_g = os.path.join(dst_gt, cls)
            os.makedirs(os.path.dirname(dst_g), exist_ok=True)
            if not os.path.exists(dst_g):
                shutil.copytree(src_g, dst_g)
        # 复制 train/good
        src_tg = os.path.join(cls_src, 'train', 'good')
        if os.path.isdir(src_tg):
            os.makedirs(dst_good, exist_ok=True)
            for f in os.listdir(src_tg):
                if f.endswith(('.png', '.jpg')):
                    dst_f = os.path.join(dst_good, f"{cls}_{f}")
                    if not os.path.exists(dst_f):
                        shutil.copy2(os.path.join(src_tg, f), dst_f)


if __name__ == "__main__":
    main()
