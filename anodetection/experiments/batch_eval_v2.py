"""逐个配置评估脚本：prepare → eval → save → cleanup → next。
每次只有一个配置的数据在磁盘上，避免磁盘满。"""
import os, sys, json, shutil, traceback, random

# Fix UnicodeEncodeError with wandb console capture on Windows GBK
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['WANDB_CONSOLE'] = 'off'

sys.path.insert(0, r'D:\desktop\DefectDiffu-main\anodetection\experiments')
random.seed(42)

from run_experiments import (
    build_file_index,
    create_mvtec_structure,
    copy_synthetic_dir,
    run_single_eval,
    compute_summary,
    MVTEC_CLASSES,
    get_class_from_filename,
    _copy_dir,
)

DATA = r'D:\desktop\DefectDiffu-main\anodetection\data\mvtec'
OUTPUT = r'D:\desktop\DefectDiffu-main\anodetection\experiments\output'
os.makedirs(OUTPUT, exist_ok=True)

# 所有实验配置定义
ALL_CONFIGS = [
    # Experiment 1
    ("exp1_fewshot", "N1_real_only", "N=1 real only", 1, None, None, None),
    ("exp1_fewshot", "N1_M50", "N=1 + M=50 syn", 1, 50, None, None),
    ("exp1_fewshot", "N2_real_only", "N=2 real only", 2, None, None, None),
    ("exp1_fewshot", "N2_M50", "N=2 + M=50 syn", 2, 50, None, None),
    ("exp1_fewshot", "N5_real_only", "N=5 real only", 5, None, None, None),
    ("exp1_fewshot", "N5_M50", "N=5 + M=50 syn", 5, 50, None, None),
    ("exp1_fewshot", "N10_real_only", "N=10 real only", 10, None, None, None),
    ("exp1_fewshot", "N10_M50", "N=10 + M=50 syn", 10, 50, None, None),
    ("exp1_fewshot", "full_real_baseline", "Full real baseline", "full", None, None, None),
    # Experiment 2
    ("exp2_syn_count", "N5_M10", "N=5 + M=10 syn", 5, 10, None, None),
    ("exp2_syn_count", "N5_M25", "N=5 + M=25 syn", 5, 25, None, None),
    ("exp2_syn_count", "N5_M50", "N=5 + M=50 syn", 5, 50, None, None),
    ("exp2_syn_count", "N5_M100", "N=5 + M=100 syn", 5, 100, None, None),
    ("exp2_syn_count", "N5_real_only", "N=5 real only (exp2)", 5, None, None, None),
    # Experiment 3
    ("exp3_aug", "real_only", "N=5 real only", 5, None, None, None),
    ("exp3_aug", "traditional_aug", "N=5 + traditional aug", 5, None, True, None),
    ("exp3_aug", "ours", "N=5 + ours (synthetic)", 5, 50, None, None),
    # Experiment 4
    ("exp4_longtail", "real_only", "N=5 real only", 5, None, None, None),
    ("exp4_longtail", "ours", "N=5 + ours (synthetic)", 5, 50, None, None),
    # Experiment 5
    ("exp5_quality", "cfg_low", "CFG 1.0-1.5", 5, 50, None, (1.0, 1.5)),
    ("exp5_quality", "cfg_medium", "CFG 1.6-2.5", 5, 50, None, (1.6, 2.5)),
    ("exp5_quality", "cfg_high", "CFG 2.6-4.0", 5, 50, None, (2.6, 4.0)),
    ("exp5_quality", "cfg_all", "All CFG", 5, 50, None, None),
]

def clean_class_dirs(config_dir):
    """删除所有子目录（类数据和合成数据），仅保留文件（如 metrics.json）。"""
    for item in os.listdir(config_dir):
        item_path = os.path.join(config_dir, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path, ignore_errors=True)

def main():
    file_index = build_file_index(DATA)
    file_index['_src_good'] = os.path.join(DATA, 'train', 'good')
    file_index['_src_bad'] = os.path.join(DATA, 'train', 'bad')
    file_index['_src_test'] = os.path.join(DATA, 'test')
    file_index['_src_ground_truth'] = os.path.join(DATA, 'ground_truth')

    total = len(ALL_CONFIGS)
    all_results = {}

    for idx, (exp_name, config_name, desc, n_shot, m_syn, use_aug, cfg_filter) in enumerate(ALL_CONFIGS):
        current = idx + 1
        exp_dir = os.path.join(OUTPUT, exp_name, config_name)
        # Clean any leftover data from previous interrupted run
        if os.path.isdir(exp_dir):
            for item in os.listdir(exp_dir):
                item_path = os.path.join(exp_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                elif item != 'metrics.json':
                    os.remove(item_path)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{current}/{total}] {exp_name}/{config_name}: {desc}")
        print(f"{'='*60}")

        try:
            # Step 1: create MVTec structure
            create_mvtec_structure(exp_dir, file_index, n_shot)

            # Step 2: copy synthetic if needed
            if m_syn is not None:
                copy_synthetic_dir(exp_dir, file_index, synthetic_limit=m_syn, cfg_filter=cfg_filter)

            # Step 3: traditional augmentation (exp3 only)
            if use_aug:
                print(f"  应用传统数据增强...")
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
                            img.rotate(deg, expand=True).save(os.path.join(good_dir, f'{base}_rot{deg}{ext}'))
                        img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(good_dir, f'{base}_flipH{ext}'))
                        img.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(good_dir, f'{base}_flipV{ext}'))
                        for factor in [0.8, 1.2]:
                            for enh_name, EnhCls in [('brightness', ImageEnhance.Brightness),
                                                      ('contrast', ImageEnhance.Contrast)]:
                                EnhCls(img).enhance(factor).save(
                                    os.path.join(good_dir, f'{base}_{enh_name}{factor}{ext}'))

            # Step 4: run eval
            metrics = run_single_eval(exp_dir, model_name='patchcore', seed=42, device='cuda')
            summary = compute_summary(metrics)

            # Step 5: save metrics
            out_path = os.path.join(exp_dir, 'metrics.json')
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            if summary['mean_image_AUROC'] is not None:
                print(f"  image_AUROC: {summary['mean_image_AUROC']:.4f}")
            if summary['mean_pixel_AUROC'] is not None:
                print(f"  pixel_AUROC: {summary['mean_pixel_AUROC']:.4f}")

            all_results[f"{exp_name}/{config_name}"] = summary
            print(f"  [OK] Done")

        except Exception as e:
            print(f"  [ERROR]: {e}")
            traceback.print_exc()
            all_results[f"{exp_name}/{config_name}"] = {"error": str(e)}

        # Step 6: clean class dirs (keep metrics.json)
        clean_class_dirs(exp_dir)

        # Save incremental results
        summary_path = os.path.join(OUTPUT, 'all_results.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Report disk space
        try:
            import shutil as sh
            usage = sh.disk_usage(OUTPUT)
            print(f"  Disk: {usage.free / (1024**3):.1f} GB free")
        except:
            pass

    print(f"\n{'='*60}")
    print(f"全部完成！共 {total} 个配置")
    print(f"结果: {os.path.join(OUTPUT, 'all_results.json')}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
