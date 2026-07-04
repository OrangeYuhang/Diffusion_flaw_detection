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
    for cls in MVTEC_CLASSES:
        if fname.startswith(cls + '_'):
            return cls
    return None


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


def create_mvtec_structure(exp_dir, file_index, n_shot):
    """
    创建原生 MVTec-format 实验目录（仅 train/good + test + ground_truth）。

    合成缺陷通过 copy_synthetic_dir() 存入平行目录 synthetic/，
    不会放入 MVTec 结构内，确保与 anomalib 完全兼容。
    """
    exp_dir = Path(exp_dir)
    src_good = Path(file_index['_src_good'])
    src_test = Path(file_index['_src_test'])
    src_gt = Path(file_index['_src_ground_truth'])

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

        # --- test & ground_truth ---
        if (src_test / cls).is_dir():
            _copy_dir(src_test / cls, exp_dir / cls / 'test')
        if (src_gt / cls).is_dir():
            _copy_dir(src_gt / cls, exp_dir / cls / 'ground_truth')


def copy_synthetic_dir(exp_dir, file_index, synthetic_limit=None,
                       cfg_filter=None):
    """
    将合成缺陷复制到 exp_dir/synthetic/{class}/，与 MVTec 目录平行。

    cfg_filter: 可选 (lo, hi) 元组，按 CFG scale 筛选。
    """
    exp_dir = Path(exp_dir)
    src_bad = Path(file_index['_src_bad'])

    for cls in MVTEC_CLASSES:
        synthetic_files = list(file_index[cls]['bad']['synthetic'])

        if cfg_filter is not None:
            lo, hi = cfg_filter
            synthetic_files = [f for f in synthetic_files if _cfg_in_range(f, lo, hi)]

        if synthetic_limit and len(synthetic_files) > synthetic_limit:
            synthetic_files = random.sample(synthetic_files, synthetic_limit)

        if not synthetic_files:
            continue

        dst = exp_dir / 'synthetic' / cls
        dst.mkdir(parents=True, exist_ok=True)
        for fname in synthetic_files:
            shutil.copy2(src_bad / fname, dst / fname)


def merge_synthetic_to_train_bad(exp_dir):
    """将 synthetic/ 合并到各类的 train/bad/，供监督模型使用。
    同时为合成图像创建零值 mask 文件，以避免 anomalib 读取 mask_path=None 崩溃。"""
    from PIL import Image
    exp_dir = Path(exp_dir)
    syn_root = exp_dir / 'synthetic'
    if not syn_root.is_dir():
        return
    for cls_dir in syn_root.iterdir():
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        bad_dst = exp_dir / cls / 'train' / 'bad'
        bad_dst.mkdir(parents=True, exist_ok=True)
        # 确保 ground_truth/bad/ 目录也存在
        mask_dst = exp_dir / cls / 'ground_truth' / 'bad'
        mask_dst.mkdir(parents=True, exist_ok=True)
        for f in cls_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, bad_dst / f.name)
                # 为合成缺陷创建零值 mask（与图像同尺寸）
                try:
                    img = Image.open(f)
                    w, h = img.size
                    # 零值 mask = 无缺陷区域标注（仅用于避免崩溃，
                    # 模型仍通过 label_index 学习这些是异常样本）
                    zero_mask = Image.new('L', (w, h), 0)
                    mask_name = f.stem + '_mask' + f.suffix
                    zero_mask.save(mask_dst / mask_name)
                except Exception:
                    pass


def cleanup_train_bad(exp_dir):
    """移除各类的 train/bad/ 目录，保留 synthetic/。"""
    for cls in MVTEC_CLASSES:
        bad_dir = os.path.join(exp_dir, cls, 'train', 'bad')
        if os.path.isdir(bad_dir):
            shutil.rmtree(bad_dir)


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
        create_mvtec_structure(os.path.join(workdir, name), idx, n)
        configs.append((name, f'N={n} real only'))

        # C: Ours
        name = f'exp1_fewshot/N{n}_M{m_synthetic}'
        exp_dir = os.path.join(workdir, name)
        create_mvtec_structure(exp_dir, idx, n)
        copy_synthetic_dir(exp_dir, idx, synthetic_limit=m_synthetic)
        configs.append((name, f'N={n} real + ≤{m_synthetic} synthetic'))

    # A: full baseline
    name = 'exp1_fewshot/full_real_baseline'
    create_mvtec_structure(os.path.join(workdir, name), idx, 'full')
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
        exp_dir = os.path.join(workdir, name)
        create_mvtec_structure(exp_dir, idx, n_shot)
        copy_synthetic_dir(exp_dir, idx, synthetic_limit=m)
        configs.append((name, f'N={n_shot} real + ≤{m} synthetic'))

    # Baseline
    name = 'exp2_syn_count/N5_real_only'
    create_mvtec_structure(os.path.join(workdir, name), idx, n_shot)
    configs.append((name, f'N={n_shot} real only'))

    return configs


def prepare_exp3_augmentation(data_root, workdir, n_shot=5):
    """实验3: 对比传统数据增强。"""
    try:
        from PIL import Image, ImageEnhance
    except ImportError:
        print("  [WARN] PIL 未安装，跳过传统数据增强 (exp3 仅生成 real_only 和 ours)")
        idx = _init_file_index(data_root)
        seed_everything(42)
        configs = []
        name = 'exp3_aug/real_only'
        create_mvtec_structure(os.path.join(workdir, name), idx, n_shot)
        configs.append((name, 'Pure real N=5'))
        exp_dir = os.path.join(workdir, 'exp3_aug/ours')
        create_mvtec_structure(exp_dir, idx, n_shot)
        copy_synthetic_dir(exp_dir, idx)
        configs.append(('exp3_aug/ours', 'Ours N=5 + synthetic'))
        return configs

    idx = _init_file_index(data_root)
    seed_everything(42)
    configs = []

    # 纯真实
    name = 'exp3_aug/real_only'
    create_mvtec_structure(os.path.join(workdir, name), idx, n_shot)
    configs.append((name, 'Pure real N=5'))

    # 传统增强
    name = 'exp3_aug/traditional_aug'
    create_mvtec_structure(os.path.join(workdir, name), idx, n_shot)
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
    exp_dir = os.path.join(workdir, name)
    create_mvtec_structure(exp_dir, idx, n_shot)
    copy_synthetic_dir(exp_dir, idx)
    configs.append((name, 'Ours N=5 + synthetic'))

    return configs


def prepare_exp4_longtail(data_root, workdir, n_shot=5):
    """实验4: 长尾缺陷覆盖率。"""
    idx = _init_file_index(data_root)
    seed_everything(42)
    configs = []

    # Baseline — 仅创建长尾类别的 MVTec 结构
    name = 'exp4_longtail/real_only'
    exp_dir = os.path.join(workdir, name)
    for cls in LONGTAIL_DEFECTS:
        create_mvtec_structure(exp_dir, idx, n_shot)
    configs.append((name, f'Long-tail N={n_shot} real only'))

    # Ours
    name = 'exp4_longtail/ours'
    exp_dir = os.path.join(workdir, name)
    for cls in LONGTAIL_DEFECTS:
        create_mvtec_structure(exp_dir, idx, n_shot)
    copy_synthetic_dir(exp_dir, idx)
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
        name = f'exp5_quality/cfg_{group_name}'
        exp_dir = os.path.join(workdir, name)
        create_mvtec_structure(exp_dir, idx, n_shot)
        copy_synthetic_dir(exp_dir, idx, cfg_filter=(lo, hi))
        configs.append((name, f'CFG {lo}-{hi} ({group_name})'))

    # All
    name = 'exp5_quality/cfg_all'
    exp_dir = os.path.join(workdir, name)
    create_mvtec_structure(exp_dir, idx, n_shot)
    copy_synthetic_dir(exp_dir, idx)
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

    # 监督模型：将 synthetic/ 合并为 train/bad/
    SUPERVISED_MODELS = {'efficient_ad', 'draem'}
    merged_for_supervised = model_name in SUPERVISED_MODELS
    if merged_for_supervised:
        merge_synthetic_to_train_bad(exp_dir)

    # efficient_ad 要求 batch_size=1
    batch_size = 1 if model_name == 'efficient_ad' else 32

    torch.manual_seed(seed)
    torch.cuda.is_available()  # 触发 CUDA 初始化，提前捕获兼容性问题
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
                train_batch_size=batch_size,
                eval_batch_size=batch_size,
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

    if merged_for_supervised:
        cleanup_train_bad(exp_dir)

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
    model = None
    if model_name == 'patchcore':
        from anomalib.models import Patchcore
        backbone = _load_wide_resnet_backbone()
        import inspect
        sig = inspect.signature(Patchcore.__init__)
        if 'backbone' in sig.parameters:
            model = Patchcore(backbone=backbone)
        else:
            model = Patchcore()
    elif model_name == 'efficient_ad':
        from anomalib.models import EfficientAd
        model = EfficientAd()
    elif model_name == 'draem':
        from anomalib.models import Draem
        model = Draem()
    elif model_name == 'cflow':
        from anomalib.models import Cflow
        model = Cflow()
    elif model_name == 'fastflow':
        from anomalib.models import Fastflow
        model = Fastflow()
    elif model_name == 'reverse_distillation':
        from anomalib.models import ReverseDistillation
        model = ReverseDistillation()
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         "Available: patchcore, efficient_ad, draem, cflow, fastflow, "
                         "reverse_distillation")

    # Add pixel_AUPRO to evaluator (not included by default in anomalib)
    from anomalib.metrics import AUPRO
    model.evaluator.test_metrics.append(
        AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
    )

    return model


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
