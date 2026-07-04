"""Re-run all evaluations with AUPRO metric enabled."""
import os, sys, json, shutil, traceback, random

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['WANDB_CONSOLE'] = 'off'

sys.path.insert(0, r'D:\desktop\DefectDiffu-main\anodetection\experiments')
random.seed(42)

from run_experiments import (
    build_file_index, create_mvtec_structure, copy_synthetic_dir,
    merge_synthetic_to_train_bad, run_single_eval, compute_summary, MVTEC_CLASSES,
)

DATA = r'D:\desktop\DefectDiffu-main\anodetection\data\mvtec'
OUTPUT = r'D:\desktop\DefectDiffu-main\anodetection\experiments\output'

# ============================================================
# Phase 1: PatchCore (23 configs)
# ============================================================

PATCHCORE_CONFIGS = [
    # exp1_fewshot
    ("exp1_fewshot", "N1_real_only", 1, None, None),
    ("exp1_fewshot", "N1_M50", 1, 50, None),
    ("exp1_fewshot", "N2_real_only", 2, None, None),
    ("exp1_fewshot", "N2_M50", 2, 50, None),
    ("exp1_fewshot", "N5_real_only", 5, None, None),
    ("exp1_fewshot", "N5_M50", 5, 50, None),
    ("exp1_fewshot", "N10_real_only", 10, None, None),
    ("exp1_fewshot", "N10_M50", 10, 50, None),
    ("exp1_fewshot", "full_real_baseline", "full", None, None),
    # exp2_syn_count
    ("exp2_syn_count", "N5_real_only", 5, None, None),
    ("exp2_syn_count", "N5_M10", 5, 10, None),
    ("exp2_syn_count", "N5_M25", 5, 25, None),
    ("exp2_syn_count", "N5_M50", 5, 50, None),
    ("exp2_syn_count", "N5_M100", 5, 100, None),
    # exp3_aug
    ("exp3_aug", "real_only", 5, None, None),
    ("exp3_aug", "traditional_aug", 5, None, True),
    ("exp3_aug", "ours", 5, 50, None),
    # exp4_longtail
    ("exp4_longtail", "real_only", 5, None, None),
    ("exp4_longtail", "ours", 5, 50, None),
    # exp5_quality
    ("exp5_quality", "cfg_low", 5, 50, (1.0, 1.5)),
    ("exp5_quality", "cfg_medium", 5, 50, (1.6, 2.5)),
    ("exp5_quality", "cfg_high", 5, 50, (2.6, 4.0)),
    ("exp5_quality", "cfg_all", 5, 50, None),
]

EFFICIENTAD_CONFIGS = [
    ("exp1_fewshot", "N5_real_only", 5, None, None),
    ("exp1_fewshot", "N5_M50", 5, 50, None),
    ("exp3_aug", "real_only", 5, None, None),
    ("exp3_aug", "traditional_aug", 5, None, True),
    ("exp3_aug", "ours", 5, 50, None),
]


def apply_traditional_aug(exp_dir):
    """Apply traditional augmentation to train/good/ in-place."""
    from PIL import Image, ImageEnhance
    for cls in MVTEC_CLASSES:
        good_dir = os.path.join(exp_dir, cls, 'train', 'good')
        if not os.path.isdir(good_dir):
            continue
        for fname in list(os.listdir(good_dir)):
            fpath = os.path.join(good_dir, fname)
            if not fname.lower().endswith(('.png', '.jpg')):
                continue
            img = Image.open(fpath)
            base, ext = os.path.splitext(fname)
            for deg in [90, 180, 270]:
                img.rotate(deg, expand=True).save(
                    os.path.join(good_dir, f'{base}_rot{deg}{ext}'))
            img.transpose(Image.FLIP_LEFT_RIGHT).save(
                os.path.join(good_dir, f'{base}_flipH{ext}'))
            img.transpose(Image.FLIP_TOP_BOTTOM).save(
                os.path.join(good_dir, f'{base}_flipV{ext}'))
            for factor in [0.8, 1.2]:
                for enh_name, EnhCls in [
                    ('brightness', ImageEnhance.Brightness),
                    ('contrast', ImageEnhance.Contrast)]:
                    EnhCls(img).enhance(factor).save(
                        os.path.join(good_dir, f'{base}_{enh_name}{factor}{ext}'))


def run_phase(configs, model_name, phase_label, results_key):
    """Run all configs for one model phase."""
    file_index = build_file_index(DATA)
    file_index['_src_good'] = os.path.join(DATA, 'train', 'good')
    file_index['_src_bad'] = os.path.join(DATA, 'train', 'bad')
    file_index['_src_test'] = os.path.join(DATA, 'test')
    file_index['_src_ground_truth'] = os.path.join(DATA, 'ground_truth')

    results_path = os.path.join(OUTPUT, f'{results_key}.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    total = len(configs)
    print(f"\n{'#'*60}")
    print(f"# {phase_label}: {total} configs, model={model_name}")
    print(f"{'#'*60}")

    for idx, (exp_name, config_name, n_shot, m_syn, aug_or_cfg) in enumerate(configs):
        current = idx + 1
        exp_dir = os.path.join(OUTPUT, exp_name, config_name)
        config_key = f"{exp_name}/{config_name}"

        # Clean previous class data
        if os.path.isdir(exp_dir):
            for item in os.listdir(exp_dir):
                item_path = os.path.join(exp_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
        os.makedirs(exp_dir, exist_ok=True)

        ts_start = __import__('time').time()
        print(f"\n[{current}/{total}] {config_key}")

        try:
            create_mvtec_structure(exp_dir, file_index, n_shot)

            # Handle cfg_filter (tuple) vs m_syn (int)
            if isinstance(aug_or_cfg, tuple):
                copy_synthetic_dir(exp_dir, file_index, synthetic_limit=m_syn,
                                   cfg_filter=aug_or_cfg)
            elif m_syn is not None:
                copy_synthetic_dir(exp_dir, file_index, synthetic_limit=m_syn)

            if aug_or_cfg is True:
                apply_traditional_aug(exp_dir)

            metrics = run_single_eval(exp_dir, model_name=model_name, seed=42, device='cuda')
            summary = compute_summary(metrics)

            # Save per-config metrics
            out_path = os.path.join(exp_dir, f'metrics_{model_name}.json')
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            img = summary.get('mean_image_AUROC')
            pix = summary.get('mean_pixel_AUROC')
            aupro = summary.get('mean_pixel_AUPRO')
            elapsed = __import__('time').time() - ts_start
            parts = []
            if img: parts.append(f'img_AUROC={img:.4f}')
            if pix: parts.append(f'pix_AUROC={pix:.4f}')
            if aupro: parts.append(f'pix_AUPRO={aupro:.4f}')
            print(f"  {'  '.join(parts)}  ({elapsed:.0f}s)")

            all_results[config_key] = summary

        except Exception as e:
            print(f"  [ERROR]: {e}")
            traceback.print_exc()
            all_results[config_key] = {"error": str(e)}

        # Clean class dirs
        for item in os.listdir(exp_dir):
            item_path = os.path.join(exp_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)

        # Save incremental
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        usage = shutil.disk_usage(OUTPUT)
        print(f"  Disk: {usage.free / (1024**3):.1f} GB free")

    print(f"\n{phase_label} complete. Results: {results_path}")
    return all_results


if __name__ == '__main__':
    # Phase 1: PatchCore
    run_phase(PATCHCORE_CONFIGS, 'patchcore', 'Phase 1: PatchCore', 'all_results')

    # Phase 2: efficient_ad
    run_phase(EFFICIENTAD_CONFIGS, 'efficient_ad', 'Phase 2: efficient_ad', 'results_efficient_ad')

    print(f"\n{'#'*60}")
    print("# All evaluations complete!")
    print(f"{'#'*60}")
