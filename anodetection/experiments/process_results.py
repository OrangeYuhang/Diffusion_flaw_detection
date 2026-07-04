"""从 all_results.json 提取结果，生成实验文档所需的数据表格。"""
import json, os, sys

RESULTS_PATH = 'D:/desktop/DefectDiffu-main/anodetection/experiments/output/all_results.json'
OUT_PATH = 'D:/desktop/DefectDiffu-main/anodetection/experiments/output/result_tables.md'

MVTEC_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
    'transistor', 'wood', 'zipper',
]

def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

def safe_auroc(r, key='image_AUROC'):
    """Safe extraction with 4 decimal rounding."""
    v = r.get(key)
    if v is None:
        return '—'
    return f'{v:.4f}'

def compute_mean(aurocs):
    """Compute mean of list, handling '—' values."""
    vals = []
    for a in aurocs:
        if a != '—':
            try:
                vals.append(float(a))
            except:
                pass
    if not vals:
        return '—'
    return f'{sum(vals)/len(vals):.4f}'

def get_class_results(results, exp_name, config_name):
    """Get per-class results for a specific config."""
    key = f'{exp_name}/{config_name}'
    data = results.get(key, {})
    classes = data.get('per_class', {})
    out = {}
    for cls in MVTEC_CLASSES:
        if cls in classes:
            if 'error' in classes[cls]:
                out[cls] = '—'
            else:
                out[cls] = safe_auroc(classes[cls])
        else:
            out[cls] = '—'
    return out

def main():
    results = load_results()

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        # ========== Experiment 1 ==========
        f.write('# Experiment 1: Image-level AUROC\n\n')
        f.write('| Class | N=1 real | N=1 + syn | N=2 real | N=2 + syn | N=5 real | N=5 + syn | N=10 real | N=10 + syn | full real |\n')
        f.write('|------|:--------:|:---------:|:--------:|:---------:|:--------:|:---------:|:---------:|:----------:|:---------:|\n')

        configs = ['N1_real_only', 'N1_M50', 'N2_real_only', 'N2_M50', 'N5_real_only', 'N5_M50', 'N10_real_only', 'N10_M50', 'full_real_baseline']
        all_class_results = {cfg: get_class_results(results, 'exp1_fewshot', cfg) for cfg in configs}

        means = {cfg: compute_mean(list(all_class_results[cfg].values())) for cfg in configs}

        for cls in MVTEC_CLASSES:
            vals = ' | '.join(all_class_results[cfg][cls] for cfg in configs)
            f.write(f'| {cls} | {vals} |\n')
        mean_vals = ' | '.join(means[cfg] for cfg in configs)
        f.write(f'| **Mean** | {mean_vals} |\n')

        # ========== Experiment 2 ==========
        f.write('\n# Experiment 2: Image-level AUROC\n\n')
        f.write('| Class | M=0 | M=10 | M=25 | M=50 | M=100 |\n')
        f.write('|------|:---:|:----:|:----:|:----:|:-----:|\n')

        configs2 = ['N5_real_only', 'N5_M10', 'N5_M25', 'N5_M50', 'N5_M100']
        cls2 = {cfg: get_class_results(results, 'exp2_syn_count', cfg) for cfg in configs2}
        means2 = {cfg: compute_mean(list(cls2[cfg].values())) for cfg in configs2}

        for cls in MVTEC_CLASSES:
            vals = ' | '.join(cls2[cfg][cls] for cfg in configs2)
            f.write(f'| {cls} | {vals} |\n')
        mean_vals2 = ' | '.join(means2[cfg] for cfg in configs2)
        f.write(f'| **Mean** | {mean_vals2} |\n')

        # ========== Experiment 3 ==========
        f.write('\n# Experiment 3: Image-level AUROC\n\n')
        f.write('| Class | 5 real only | + Traditional Aug | + Ours (synthetic) |\n')
        f.write('|------|:-----------:|:-----------------:|:-------------------:|\n')

        configs3 = ['real_only', 'traditional_aug', 'ours']
        cls3 = {cfg: get_class_results(results, 'exp3_aug', cfg) for cfg in configs3}
        means3 = {cfg: compute_mean(list(cls3[cfg].values())) for cfg in configs3}

        for cls in MVTEC_CLASSES:
            vals = ' | '.join(cls3[cfg][cls] for cfg in configs3)
            f.write(f'| {cls} | {vals} |\n')
        mean_vals3 = ' | '.join(means3[cfg] for cfg in configs3)
        f.write(f'| **Mean** | {mean_vals3} |\n')

        # ========== Experiment 4 ==========
        f.write('\n# Experiment 4: Image-level AUROC\n\n')
        f.write('| Class | 5 real only | + Ours | Delta |\n')
        f.write('|------|:-----------:|:------:|:-----:|\n')

        configs4 = ['real_only', 'ours']
        cls4 = {cfg: get_class_results(results, 'exp4_longtail', cfg) for cfg in configs4}
        means4 = {cfg: compute_mean(list(cls4[cfg].values())) for cfg in configs4}

        for cls in MVTEC_CLASSES:
            r = cls4['real_only'][cls]
            o = cls4['ours'][cls]
            if r != '—' and o != '—':
                delta = f'+{float(o)-float(r):.4f}' if float(o) >= float(r) else f'{float(o)-float(r):.4f}'
            else:
                delta = '—'
            f.write(f'| {cls} | {r} | {o} | {delta} |\n')

        # ========== Experiment 5 ==========
        f.write('\n# Experiment 5: Image-level AUROC by CFG group\n\n')
        f.write('| Class | CFG low (1.0-1.5) | CFG medium (1.6-2.5) | CFG high (2.6-4.0) | CFG all |\n')
        f.write('|------|:-----------------:|:--------------------:|:------------------:|:-------:|\n')

        configs5 = ['cfg_low', 'cfg_medium', 'cfg_high', 'cfg_all']
        cls5 = {cfg: get_class_results(results, 'exp5_quality', cfg) for cfg in configs5}
        means5 = {cfg: compute_mean(list(cls5[cfg].values())) for cfg in configs5}

        for cls in MVTEC_CLASSES:
            vals = ' | '.join(cls5[cfg][cls] for cfg in configs5)
            f.write(f'| {cls} | {vals} |\n')
        mean_vals5 = ' | '.join(means5[cfg] for cfg in configs5)
        f.write(f'| **Mean** | {mean_vals5} |\n')

        # ========== Summary ==========
        f.write('\n# Summary Stats\n\n')
        for exp, cfg_list in [('exp1_fewshot', configs), ('exp2_syn_count', configs2),
                               ('exp3_aug', configs3), ('exp4_longtail', configs4),
                               ('exp5_quality', configs5)]:
            for cfg in cfg_list:
                key = f'{exp}/{cfg}'
                data = results.get(key, {})
                m_img = data.get('mean_image_AUROC')
                m_pix = data.get('mean_pixel_AUROC')
                err = data.get('error', '')
                status = f'ERROR: {err}' if err else f'img={m_img:.4f}' if m_img else 'INCOMPLETE'
                f.write(f'{key}: {status}\n')

    print(f'Results written to {OUT_PATH}')

if __name__ == '__main__':
    main()
