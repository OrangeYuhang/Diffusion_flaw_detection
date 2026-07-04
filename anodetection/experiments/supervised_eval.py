"""监督模型评估：efficient_ad 逐个配置 prepare -> eval -> save -> cleanup"""
import os, sys, json, shutil, traceback, random

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['WANDB_CONSOLE'] = 'off'

sys.path.insert(0, r'D:\desktop\DefectDiffu-main\anodetection\experiments')
random.seed(42)

from run_experiments import (
    build_file_index, create_mvtec_structure, copy_synthetic_dir,
    run_single_eval, compute_summary, MVTEC_CLASSES,
)

DATA = r'D:\desktop\DefectDiffu-main\anodetection\data\mvtec'
OUTPUT = r'D:\desktop\DefectDiffu-main\anodetection\experiments\output'
MODEL = 'efficient_ad'

CONFIGS = [
    ("exp1_fewshot", "N5_real_only", 5, None, None),       # baseline
    ("exp1_fewshot", "N5_M50", 5, 50, None),               # core comparison
    ("exp3_aug", "real_only", 5, None, None),              # exp3 baseline
    ("exp3_aug", "traditional_aug", 5, None, True),        # best competing
    ("exp3_aug", "ours", 5, 50, None),                     # ours vs trad
]

def main():
    file_index = build_file_index(DATA)
    file_index['_src_good'] = os.path.join(DATA, 'train', 'good')
    file_index['_src_bad'] = os.path.join(DATA, 'train', 'bad')
    file_index['_src_test'] = os.path.join(DATA, 'test')
    file_index['_src_ground_truth'] = os.path.join(DATA, 'ground_truth')

    total = len(CONFIGS)
    all_results = {}

    for idx, (exp_name, config_name, n_shot, m_syn, use_aug) in enumerate(CONFIGS):
        current = idx + 1
        exp_dir = os.path.join(OUTPUT, exp_name, config_name)
        out_path = os.path.join(exp_dir, f'metrics_{MODEL}.json')

        # Clean previous class data
        if os.path.isdir(exp_dir):
            for item in os.listdir(exp_dir):
                item_path = os.path.join(exp_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{current}/{total}] {exp_name}/{config_name} ({MODEL})")
        print(f"{'='*60}")

        try:
            create_mvtec_structure(exp_dir, file_index, n_shot)
            if m_syn is not None:
                copy_synthetic_dir(exp_dir, file_index, synthetic_limit=m_syn)

            if use_aug:
                print("  Applying traditional augmentation...")
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

            metrics = run_single_eval(exp_dir, model_name=MODEL, seed=42, device='cuda')
            summary = compute_summary(metrics)

            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            if summary.get('mean_image_AUROC'):
                print(f"  image_AUROC: {summary['mean_image_AUROC']:.4f}")
            if summary.get('mean_pixel_AUROC'):
                print(f"  pixel_AUROC: {summary['mean_pixel_AUROC']:.4f}")

            all_results[f"{exp_name}/{config_name}"] = summary
            print(f"  [OK] Done")

        except Exception as e:
            print(f"  [ERROR]: {e}")
            traceback.print_exc()
            all_results[f"{exp_name}/{config_name}"] = {"error": str(e)}

        # Clean class dirs
        for item in os.listdir(exp_dir):
            item_path = os.path.join(exp_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)

        # Save incremental
        summary_path = os.path.join(OUTPUT, f'results_{MODEL}.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        import shutil as sh
        usage = sh.disk_usage(OUTPUT)
        print(f"  Disk: {usage.free / (1024**3):.1f} GB free")

    print(f"\n{'='*60}")
    print(f"All {total} configs done. Results: {os.path.join(OUTPUT, f'results_{MODEL}.json')}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
