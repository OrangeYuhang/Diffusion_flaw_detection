"""
DefectDiffu 实验框架 v2 — 基于 anomalib 2.x Python API。

用法（在 anodetection conda 环境下运行）：
  conda activate anodetection

  # 准备所有实验数据目录
  python experiments/run_experiments.py prepare \
    --data ./anodetection/data/mvtec --workdir ./experiments/output

  # 运行单个实验配置
  python experiments/run_experiments.py eval \
    --exp-dir ./experiments/output/exp1_fewshot/N5_M50

  # 批量运行所有已准备好的实验（依次执行，完成后汇总）
  python experiments/run_experiments.py run-all --workdir ./experiments/output

模型选择：
  --model patchcore      无监督，仅用 train/good（默认）
  --model efficient_ad   利用 train/bad 中的合成缺陷进行监督训练
  --model draem          自身也生成合成异常，叠加 DefectDiffu 的缺陷
"""
import os
import sys
import json
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

try:
    import torch
    _has_torch = True
except ImportError:
    torch = None  # type: ignore
    _has_torch = False

# ============================================================
#  常量
# ============================================================

MVTEC_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper',
]

# 长尾缺陷 — MVTec 中样本稀缺的缺陷类型
LONGTAIL_DEFECTS = {
    'bottle':      ['broken_large', 'contamination'],
    'cable':       ['bent_wire', 'cut_outer_insulation', 'missing_wire', 'poke_insulation'],
    'capsule':     ['crack', 'faulty_imprint', 'poke', 'scratch', 'squeeze'],
    'metal_nut':   ['bent', 'flip', 'scratch'],
    'pill':        ['combined', 'faulty_imprint', 'pill_type'],
    'screw':       ['manipulated_front', 'scratch_head'],
    'toothbrush':  ['defective'],
    'wood':        ['color', 'combined', 'hole', 'liquid'],
    'zipper':      ['combined', 'fabric_border', 'rough', 'split_teeth', 'squeezed_teeth'],
}


def seed_everything(seed=42):
    random.seed(seed)
    if _has_torch:
        torch.manual_seed(seed)


def get_class_from_filename(fname):
    parts = fname.replace('.png', '').replace('.jpg', '').split('_')
    return parts[0] if parts else None


# ============================================================
#  Phase 1: 数据准备
# ============================================================

def build_file_index(data_root):
    """扫描 train/good 和 train/bad，按类别建立文件索引。"""
    index = defaultdict(lambda: {'good': [], 'bad': {'real': [], 'synthetic': []}})
    good_dir = os.path.join(data_root, 'train', 'good')
    bad_dir = os.path.join(data_root, 'train', 'bad')

    if os.path.isdir(good_dir):
        for fname in sorted(os.listdir(good_dir)):
            cls = get_class_from_filename(fname)
            if cls in MVTEC_CLASSES:
                index[cls]['good'].append(fname)

    if os.path.isdir(bad_dir):
        for fname in sorted(os.listdir(bad_dir)):
            cls = get_class_from_filename(fname)
            if cls not in MVTEC_CLASSES:
                continue
            cat = 'synthetic' if 'cfg' in fname else 'real'
            index[cls]['bad'][cat].append(fname)

    return index


def _copy_dir(src, dst):
    """复制目录内容（处理已存在的情况）。"""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        for item in Path(src).iterdir():
            dst_item = dst / item.name
            if item.is_dir():
                if not dst_item.exists():
                    shutil.copytree(item, dst_item)
            else:
                if not dst_item.exists():
                    shutil.copy2(item, dst_item)
    else:
        # Linux/Mac: 符号链接
        if not any(dst.iterdir()):
            for item in Path(src).iterdir():
                os.symlink(item, dst / item.name)


def create_mvtec_structure(exp_dir, file_index, n_shot, include_synthetic=False,
                           synthetic_limit=None):
    """
    创建 MVTec-format 实验目录。

    目录结构：
      {exp_dir}/{class}/train/good/        — N 张正常样本
      {exp_dir}/{class}/train/bad/         — 合成缺陷（仅 include_synthetic=True）
      {exp_dir}/{class}/test/{defect}/     — 链接自原始 test
      {exp_dir}/{class}/ground_truth/{defect}/ — 链接自原始 gt
    """
    exp_dir = Path(exp_dir)
    src_good = Path(file_index['_src_good'])
    src_bad = Path(file_index['_src_bad'])
    src_test = Path(file_index['_src_test'])
    src_gt = Path(file_index['_src_ground_truth'])

    if n_shot == 'full':
        pass  # handled per-class below

    for cls in MVTEC_CLASSES:
        # --- train/good ---
        good_dst = exp_dir / cls / 'train' / 'good'
        good_dst.mkdir(parents=True, exist_ok=True)

        available = file_index[cls]['good']
        if n_shot == 'full':
            n = len(available)
        elif isinstance(n_shot, dict):
            n = n_shot.get(cls, 5)
        else:
            n = int(n_shot)

        sampled = random.sample(available, min(n, len(available)))
        for fname in sampled:
            shutil.copy2(src_good / fname, good_dst / fname)

        # --- train/bad (合成缺陷) ---
        if include_synthetic:
            synthetic_files = file_index[cls]['bad']['synthetic']
            if synthetic_limit and len(synthetic_files) > synthetic_limit:
                synthetic_files = random.sample(synthetic_files, synthetic_limit)
            if synthetic_files:
                bad_dst = exp_dir / cls / 'train' / 'bad'
                bad_dst.mkdir(parents=True, exist_ok=True)
                for fname in synthetic_files:
                    shutil.copy2(src_bad / fname, bad_dst / fname)

        # --- test & ground_truth (链接) ---
        if (src_test / cls).is_dir():
            _copy_dir(src_test / cls, exp_dir / cls / 'test')
        if (src_gt / cls).is_dir():
            _copy_dir(src_gt / cls, exp_dir / cls / 'ground_truth')


# ============================================================
#  各实验数据准备
# ============================================================

def _init_file_index(data_root):
    idx = build_file_index(data_root)
    idx['_src_good'] = os.path.join(data_root, 'train', 'good')
    idx['_src_bad']  = os.path.join(data_root, 'train', 'bad')
    idx['_src_test'] = os.path.join(data_root, 'test')
    idx['_src_ground_truth'] = os.path.join(data_root, 'ground_truth')
    return idx


def prepare_exp1_fewshot(data_root, workdir, n_shots=[1, 2, 5, 10], m_synthetic=50):
    """实验1: 不同 few-shot 规模下的增益。"""
    idx = _init_file_index(data_root)
    seed_everything(42)
    configs = []

    for n in n_shots:
        # B: few-shot baseline
        name = f'exp1_fewshot/N{n}_real_only'
        create_mvtec_structure(os.path.join(workdir, name), idx, n,
                               include_synthetic=False)
        configs.append((name, f'N={n} real only'))

        # C: Ours
        name = f'exp1_fewshot/N{n}_M{m_synthetic}'
        create_mvtec_structure(os.path.join(workdir, name), idx, n,
                               include_synthetic=True,
                               synthetic_limit=m_synthetic)
        configs.append((name, f'N={n} real + ≤{m_synthetic} synthetic'))

    # A: full baseline
    name = 'exp1_fewshot/full_real_baseline'
    create_mvtec_structure(os.path.join(workdir, name), idx, 'full',
                           include_synthetic=False)
    configs.append((name, 'Full real baseline'))

    return configs


def prepare_exp2_synthetic_count(data_root, workdir, n_shot=5,
                                  m_counts=[10, 25, 50, 100]):
    """实验2: 合成样本数量的影响。"""
    idx = _init_file_index(data_root)
    seed_everything(42)
    configs = []

    for m in m_counts:
        name = f'exp2_syn_count/N{n_shot}_M{m}'
        create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                               include_synthetic=True, synthetic_limit=m)
        configs.append((name, f'N={n_shot} real + ≤{m} synthetic'))

    # Baseline
    name = 'exp2_syn_count/N5_real_only'
    create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                           include_synthetic=False)
    configs.append((name, f'N={n_shot} real only'))

    return configs


def prepare_exp3_augmentation(data_root, workdir, n_shot=5):
    """实验3: 对比传统数据增强。"""
    try:
        from PIL import Image, ImageEnhance
    except ImportError:
        print("  [WARN] PIL 未安装，跳过传统数据增强 (exp3 仅生成 real_only 和 ours)")
        # 回退：仅生成 real_only 和 ours，不生成 traditional_aug
        idx = _init_file_index(data_root)
        seed_everything(42)
        configs = []
        name = 'exp3_aug/real_only'
        create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                               include_synthetic=False)
        configs.append((name, 'Pure real N=5'))
        name = 'exp3_aug/ours'
        create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                               include_synthetic=True)
        configs.append((name, 'Ours N=5 + synthetic'))
        return configs

    idx = _init_file_index(data_root)
    seed_everything(42)
    configs = []

    # 纯真实
    name = 'exp3_aug/real_only'
    create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                           include_synthetic=False)
    configs.append((name, 'Pure real N=5'))

    # 传统增强
    name = 'exp3_aug/traditional_aug'
    create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                           include_synthetic=False)
    # 在已创建的 good/ 目录基础上做增强
    for cls in MVTEC_CLASSES:
        good_dir = Path(workdir) / name / cls / 'train' / 'good'
        if not good_dir.is_dir():
            continue
        for fname in list(good_dir.iterdir()):
            if not fname.suffix.lower() in ('.png', '.jpg'):
                continue
            img = Image.open(fname)
            base = fname.stem
            ext = fname.suffix
            # Rotation
            for deg in [90, 180, 270]:
                img.rotate(deg, expand=True).save(good_dir / f'{base}_rot{deg}{ext}')
            # Flip
            img.transpose(Image.FLIP_LEFT_RIGHT).save(good_dir / f'{base}_flipH{ext}')
            img.transpose(Image.FLIP_TOP_BOTTOM).save(good_dir / f'{base}_flipV{ext}')
            # Color jitter
            for factor in [0.8, 1.2]:
                for enh_name, EnhCls in [('brightness', ImageEnhance.Brightness),
                                          ('contrast', ImageEnhance.Contrast)]:
                    EnhCls(img).enhance(factor).save(
                        good_dir / f'{base}_{enh_name}{factor}{ext}')
    configs.append((name, 'Traditional aug N=5'))

    # Ours
    name = 'exp3_aug/ours'
    create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                           include_synthetic=True)
    configs.append((name, 'Ours N=5 + synthetic'))

    return configs


def prepare_exp4_longtail(data_root, workdir, n_shot=5):
    """实验4: 长尾缺陷覆盖率。"""
    idx = _init_file_index(data_root)
    seed_everything(42)
    configs = []

    # 仅长尾类别
    lt_classes = list(LONGTAIL_DEFECTS.keys())

    # Baseline
    name = 'exp4_longtail/real_only'
    for cls in lt_classes:
        create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                               include_synthetic=False)
    configs.append((name, f'Long-tail N={n_shot} real only'))

    # Ours
    name = 'exp4_longtail/ours'
    for cls in lt_classes:
        create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                               include_synthetic=True)
    configs.append((name, f'Long-tail N={n_shot} + synthetic'))

    return configs


def _cfg_in_range(fname, lo, hi):
    import re
    m = re.search(r'cfg(\d+\.?\d*)', fname)
    if m:
        val = float(m.group(1))
        return lo <= val <= hi
    return False


def prepare_exp5_quality(data_root, workdir, n_shot=5):
    """实验5: 合成质量消融 — 按 CFG scale 分组。"""
    idx = _init_file_index(data_root)
    seed_everything(42)
    configs = []

    cfg_groups = {
        'low':    (1.0, 1.5),
        'medium': (1.6, 2.5),
        'high':   (2.6, 4.0),
    }

    for group_name, (lo, hi) in cfg_groups.items():
        idx_filtered = deepcopy(idx)
        for cls in MVTEC_CLASSES:
            filtered = [f for f in idx[cls]['bad']['synthetic']
                        if _cfg_in_range(f, lo, hi)]
            idx_filtered[cls]['bad']['synthetic'] = filtered
        name = f'exp5_quality/cfg_{group_name}'
        create_mvtec_structure(os.path.join(workdir, name), idx_filtered, n_shot,
                               include_synthetic=True)
        configs.append((name, f'CFG {lo}-{hi} ({group_name})'))

    # All
    name = 'exp5_quality/cfg_all'
    create_mvtec_structure(os.path.join(workdir, name), idx, n_shot,
                           include_synthetic=True)
    configs.append((name, 'All CFG scales'))

    return configs


# ============================================================
#  Phase 2: 评估（anomalib 2.x Python API）
# ============================================================

def load_anomalib():
    """延迟导入 anomalib — 允许 prepare 子命令在无 anomalib 环境下运行。"""
    from anomalib.engine import Engine
    from anomalib.data import MVTecAD
    from anomalib.models import Patchcore
    return Engine, MVTecAD, Patchcore


def run_single_eval(exp_dir, model_name='patchcore', seed=42, device='auto'):
    """
    在给定的 MVTec-format 目录上运行训练+评估。
    对每个类别分别运行，返回合并的指标字典。

    device: 'auto'（优先 GPU，不兼容时回退 CPU）、'cuda'、'cpu'
    """
    if not _has_torch:
        raise RuntimeError("eval 需要 PyTorch（anomalib 依赖），请在 anodetection 环境下运行")
    Engine, MVTecAD, Patchcore = load_anomalib()

    torch.manual_seed(seed)
    all_metrics = {}

    # 决定 accelerator
    if device == 'cpu':
        accelerator = 'cpu'
        dev_count = 1
    elif device == 'cuda':
        accelerator = 'cuda'
        dev_count = 1
    else:  # auto
        accelerator = 'auto'
        dev_count = 1

    for cls in MVTEC_CLASSES:
        cls_dir = os.path.join(exp_dir, cls, 'train', 'good')
        if not os.path.isdir(cls_dir) or not os.listdir(cls_dir):
            print(f"  [{cls}] SKIP — no training data")
            continue

        print(f"  [{cls}] ", end='', flush=True)

        model = _create_model(model_name)

        engine = Engine(
            default_root_dir=os.path.join(exp_dir, '.anomalib', cls),
            max_epochs=1,
            accelerator=accelerator,
            devices=dev_count,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        try:
            datamodule = MVTecAD(
                root=exp_dir,
                category=cls,
                train_batch_size=32,
                eval_batch_size=32,
                num_workers=0,
                seed=seed,
            )
            engine.train(model=model, datamodule=datamodule)
            test_results = engine.test(model=model, datamodule=datamodule)
        except RuntimeError as e:
            err_msg = str(e)
            if 'no kernel image' in err_msg or 'sm_' in err_msg:
                print(f"GPU 不兼容 — 请用 --device cpu 或升级 PyTorch nightly")
            else:
                print(f"FAILED: {err_msg[-120:]}")
            all_metrics[cls] = {'image_AUROC': None, 'error': err_msg[:200]}
            continue
        except Exception as e:
            print(f"FAILED: {e}")
            all_metrics[cls] = {'image_AUROC': None, 'error': str(e)}
            continue

        if test_results and len(test_results) > 0:
            metrics = {}
            for d in test_results:
                metrics.update({k: v for k, v in d.items()
                                if isinstance(v, (int, float))})
            all_metrics[cls] = metrics
            img_auroc = metrics.get('image_AUROC', float('nan'))
            if img_auroc == img_auroc:  # not NaN
                print(f"image_AUROC={img_auroc:.4f}")
            else:
                print(f"keys: {list(metrics.keys())[:5]}")
        else:
            all_metrics[cls] = {'image_AUROC': None}
            print("no results")

    return all_metrics


def _load_wide_resnet_backbone():
    """
    加载 WideResNet50_2 骨干网络。
    优先使用本地缓存的 torchvision 权重（避免 HuggingFace 网络问题）。
    """
    import torch
    from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

    try:
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
        model = wide_resnet50_2(weights=weights)
        print("        [backbone] loaded from local torchvision cache")
        return model
    except Exception:
        pass

    # 回退：尝试 timm 默认下载（需设置 HF_ENDPOINT=https://hf-mirror.com）
    import timm
    try:
        model = timm.create_model('wide_resnet50_2', pretrained=True)
        print("        [backbone] loaded via timm (may use HF mirror)")
        return model
    except Exception:
        print("        [backbone] FAILED — no pretrained weights available, using random init")
        return timm.create_model('wide_resnet50_2', pretrained=False)


def _create_model(model_name):
    """根据名称创建 anomalib 模型。"""
    if model_name == 'patchcore':
        from anomalib.models import Patchcore
        backbone = _load_wide_resnet_backbone()
        # Patchcore(backbone=...)  — check if signature accepts Module
        import inspect
        sig = inspect.signature(Patchcore.__init__)
        if 'backbone' in sig.parameters:
            return Patchcore(backbone=backbone)
        return Patchcore()
    elif model_name == 'efficient_ad':
        from anomalib.models import EfficientAd
        return EfficientAd()
    elif model_name == 'draem':
        from anomalib.models import Draem
        return Draem()
    elif model_name == 'cflow':
        from anomalib.models import Cflow
        return Cflow()
    elif model_name == 'fastflow':
        from anomalib.models import Fastflow
        return Fastflow()
    elif model_name == 'reverse_distillation':
        from anomalib.models import ReverseDistillation
        return ReverseDistillation()
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         "Available: patchcore, efficient_ad, draem, cflow, fastflow, "
                         "reverse_distillation")


def compute_summary(metrics_dict):
    """从 per-class 指标计算均值，跳过 None/NaN。"""
    img_aurocs = []
    pix_aurocs = []
    pix_aupros = []
    img_f1s = []
    pix_f1s = []

    for cls, m in metrics_dict.items():
        if m is None:
            continue
        for key, val in m.items():
            if val is None:
                continue
            if isinstance(val, str):
                continue
            if not isinstance(val, (int, float)):
                continue
            if val != val:  # NaN
                continue
            kl = key.lower()
            if 'image_auroc' in kl:
                img_aurocs.append(val)
            elif 'pixel_auroc' in kl:
                pix_aurocs.append(val)
            elif 'pixel_aupro' in kl:
                pix_aupros.append(val)
            elif 'image_f1' in kl:
                img_f1s.append(val)
            elif 'pixel_f1' in kl:
                pix_f1s.append(val)

    def _mean(vals):
        return sum(vals) / len(vals) if vals else None

    return {
        'mean_image_AUROC': _mean(img_aurocs),
        'mean_pixel_AUROC': _mean(pix_aurocs),
        'mean_pixel_AUPRO': _mean(pix_aupros),
        'mean_image_F1': _mean(img_f1s),
        'mean_pixel_F1': _mean(pix_f1s),
        'per_class': metrics_dict,
    }


# ============================================================
#  CLI
# ============================================================

def cmd_prepare(args):
    workdir = os.path.abspath(args.workdir)
    data_root = os.path.abspath(args.data)
    os.makedirs(workdir, exist_ok=True)

    all_configs = {}

    exps = args.experiments
    if 'all' in exps:
        exps = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']

    for exp in exps:
        print(f"\n{'='*60}")
        if exp == 'exp1':
            print("实验1: 不同 few-shot 规模下的增益")
            configs = prepare_exp1_fewshot(data_root, workdir)
        elif exp == 'exp2':
            print("实验2: 合成样本数量的影响")
            configs = prepare_exp2_synthetic_count(data_root, workdir)
        elif exp == 'exp3':
            print("实验3: 对比传统数据增强")
            configs = prepare_exp3_augmentation(data_root, workdir)
        elif exp == 'exp4':
            print("实验4: 长尾缺陷覆盖率")
            configs = prepare_exp4_longtail(data_root, workdir)
        elif exp == 'exp5':
            print("实验5: 合成质量消融")
            configs = prepare_exp5_quality(data_root, workdir)
        else:
            print(f"未知实验: {exp}")
            continue

        for name, desc in configs:
            print(f"  OK  {name}")
        all_configs[exp] = [(name, desc) for name, desc in configs]

    # 保存清单
    manifest_path = os.path.join(workdir, 'experiment_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(all_configs, f, indent=2, ensure_ascii=False)
    total = sum(len(v) for v in all_configs.values())
    print(f"\n清单: {manifest_path}  ({total} 个配置)")


def cmd_eval(args):
    exp_dir = os.path.abspath(args.exp_dir)
    print(f"实验: {exp_dir}")
    print(f"模型: {args.model}  |  设备: {args.device}")
    metrics = run_single_eval(exp_dir, model_name=args.model, seed=args.seed,
                              device=args.device)
    summary = compute_summary(metrics)

    out_path = os.path.join(exp_dir, 'metrics.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"指标已保存: {out_path}")

    if summary['mean_image_AUROC'] is not None:
        print(f"  Mean Image AUROC: {summary['mean_image_AUROC']:.4f}")
    if summary['mean_pixel_AUROC'] is not None:
        print(f"  Mean Pixel AUROC: {summary['mean_pixel_AUROC']:.4f}")
    if summary['mean_pixel_AUPRO'] is not None:
        print(f"  Mean Pixel AUPRO: {summary['mean_pixel_AUPRO']:.4f}")


def cmd_run_all(args):
    workdir = os.path.abspath(args.workdir)
    manifest_path = os.path.join(workdir, 'experiment_manifest.json')

    if not os.path.exists(manifest_path):
        print(f"错误: 未找到 {manifest_path}，请先运行 prepare")
        sys.exit(1)

    with open(manifest_path) as f:
        all_configs = json.load(f)

    all_summaries = {}
    total = sum(len(v) for v in all_configs.values())
    current = 0

    for exp_group, configs in all_configs.items():
        print(f"\n{'='*60}")
        print(f"{exp_group} ({len(configs)} 个配置)")
        print('='*60)

        for exp_name, desc in configs:
            current += 1
            exp_dir = os.path.join(workdir, exp_name)

            print(f"\n[{current}/{total}] {exp_name}")
            print(f"        {desc}")

            if not os.path.isdir(exp_dir):
                print(f"  SKIP — 目录不存在")
                continue

            metrics = run_single_eval(exp_dir, model_name=args.model, seed=args.seed,
                                      device=args.device)
            summary = compute_summary(metrics)

            out_path = os.path.join(exp_dir, 'metrics.json')
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            all_summaries[exp_name] = summary

            if summary['mean_image_AUROC'] is not None:
                print(f"  => Mean Image AUROC: {summary['mean_image_AUROC']:.4f}")

    # 汇总
    summary_path = os.path.join(workdir, 'all_metrics.json')
    with open(summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\n汇总: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='DefectDiffu 实验框架 v2 (anomalib 2.x)')
    sub = parser.add_subparsers(dest='command')

    # prepare
    p_prep = sub.add_parser('prepare', help='准备实验数据目录（无需 anomalib）')
    p_prep.add_argument('--data', required=True,
                        help='数据集根路径 (anodetection/data/mvtec)')
    p_prep.add_argument('--workdir', default='./experiments/output',
                        help='实验输出根目录')
    p_prep.add_argument('--experiments', nargs='+', default=['all'],
                        choices=['all', 'exp1', 'exp2', 'exp3', 'exp4', 'exp5'])

    # eval
    p_eval = sub.add_parser('eval', help='对单个实验配置运行 PatchCore 评估')
    p_eval.add_argument('--exp-dir', required=True, help='实验数据目录')
    p_eval.add_argument('--model', default='patchcore',
                        choices=['patchcore', 'efficient_ad', 'draem', 'cflow',
                                 'fastflow', 'reverse_distillation'],
                        help='anomalib 模型 (默认: patchcore)')
    p_eval.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='计算设备 (默认: auto)')
    p_eval.add_argument('--seed', type=int, default=42)

    # run-all
    p_all = sub.add_parser('run-all', help='批量运行所有已准备的实验')
    p_all.add_argument('--workdir', default='./experiments/output',
                       help='实验输出根目录')
    p_all.add_argument('--model', default='patchcore',
                       choices=['patchcore', 'efficient_ad', 'draem', 'cflow',
                                'fastflow', 'reverse_distillation'],
                       help='anomalib 模型 (默认: patchcore)')
    p_all.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='计算设备 (默认: auto)')
    p_all.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'run-all':
        cmd_run_all(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
