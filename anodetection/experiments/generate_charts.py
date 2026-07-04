"""生成实验图表：AUROC vs N (few-shot scale) 和 AUROC vs M (synthetic count)"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 加载 PatchCore 结果
with open(r'D:\desktop\DefectDiffu-main\anodetection\experiments\output\all_results.json') as f:
    results = json.load(f)

# ============================================================
# Chart 1: AUROC vs N (few-shot scale)
# ============================================================
N_values = [1, 2, 5, 10]
real_only = []
synth = []
full_baseline = None

for n in N_values:
    real_key = f'exp1_fewshot/N{n}_real_only'
    synth_key = f'exp1_fewshot/N{n}_M50'
    real_only.append(results[real_key]['mean_image_AUROC'])
    synth.append(results[synth_key]['mean_image_AUROC'])

full_baseline = results['exp1_fewshot/full_real_baseline']['mean_image_AUROC']

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(N_values, real_only, 'o-', color='#2c3e50', linewidth=2, markersize=8, label='Real Only (few-shot)')
ax.plot(N_values, synth, 's--', color='#e74c3c', linewidth=2, markersize=8, label='+ Synthetic (M=50)')
ax.axhline(y=full_baseline, color='#27ae60', linestyle=':', linewidth=2, label=f'Full Baseline ({full_baseline:.4f})')

# Annotate delta at N=5
n5_idx = N_values.index(5)
delta_n5 = synth[n5_idx] - real_only[n5_idx]
ax.annotate(f'N=5:\n{delta_n5:+.4f}',
            xy=(5, synth[n5_idx]), xytext=(5.5, synth[n5_idx] - 0.02),
            fontsize=9, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1))

ax.set_xlabel('N (Number of Real Training Samples)', fontsize=12)
ax.set_ylabel('Image AUROC', fontsize=12)
ax.set_title('Few-Shot Scale vs Detection Performance (PatchCore)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks(N_values)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.grid(True, alpha=0.3)
ax.set_ylim(0.78, 1.0)

plt.tight_layout()
chart1_path = r'D:\desktop\DefectDiffu-main\document\chart_auroc_vs_N.png'
plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
print(f'Saved: {chart1_path}')

# ============================================================
# Chart 2: AUROC vs M (synthetic count)
# ============================================================
M_values = [0, 10, 25, 50, 100]
m_aurocs = []

# M=0 is N5_real_only from exp2
m_aurocs.append(results['exp2_syn_count/N5_real_only']['mean_image_AUROC'])
for m in [10, 25, 50, 100]:
    m_aurocs.append(results[f'exp2_syn_count/N5_M{m}']['mean_image_AUROC'])

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(M_values, m_aurocs, 'o-', color='#2980b9', linewidth=2, markersize=8, markerfacecolor='white',
        markeredgewidth=2, markeredgecolor='#2980b9')

# Add N5_real_only from exp1 as reference
exp1_n5 = results['exp1_fewshot/N5_real_only']['mean_image_AUROC']
ax.axhline(y=exp1_n5, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'N=5 Real Only (exp1: {exp1_n5:.4f})')

ax.set_xlabel('M (Number of Synthetic Samples)', fontsize=12)
ax.set_ylabel('Image AUROC', fontsize=12)
ax.set_title('Synthetic Sample Count vs Detection Performance (PatchCore, N=5)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks(M_values)
ax.set_xticklabels(['0 (real only)'] + [str(m) for m in M_values[1:]])
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax.grid(True, alpha=0.3)

# Add range bar for context
ymin = min(m_aurocs) - 0.005
ymax = max(m_aurocs) + 0.005
ax.set_ylim(ymin, ymax)

plt.tight_layout()
chart2_path = r'D:\desktop\DefectDiffu-main\document\chart_auroc_vs_M.png'
plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
print(f'Saved: {chart2_path}')

# ============================================================
# Chart 3: CFG Scale Ablation (exp5)
# ============================================================
cfg_labels = ['low\n(1.0-1.5)', 'medium\n(1.6-2.5)', 'high\n(2.6-4.0)', 'all\n(mixed)']
cfg_aurocs = [
    results['exp5_quality/cfg_low']['mean_image_AUROC'],
    results['exp5_quality/cfg_medium']['mean_image_AUROC'],
    results['exp5_quality/cfg_high']['mean_image_AUROC'],
    results['exp5_quality/cfg_all']['mean_image_AUROC'],
]
cfg_colors = ['#f39c12', '#27ae60', '#e74c3c', '#8e44ad']

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(range(len(cfg_labels)), cfg_aurocs, color=cfg_colors, width=0.6, edgecolor='white', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, cfg_aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.004,
            f'{val:.4f}', ha='center', va='top', fontsize=11, fontweight='bold', color='white')

ax.set_xticks(range(len(cfg_labels)))
ax.set_xticklabels(cfg_labels, fontsize=10)
ax.set_ylabel('Image AUROC', fontsize=12)
ax.set_title('CFG Scale Ablation (PatchCore, N=5)', fontsize=14)
ax.set_ylim(0.87, 0.92)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
chart3_path = r'D:\desktop\DefectDiffu-main\document\chart_cfg_ablation.png'
plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
print(f'Saved: {chart3_path}')

# ============================================================
# Chart 4: Model comparison — PatchCore vs efficient_ad
# ============================================================
with open(r'D:\desktop\DefectDiffu-main\anodetection\experiments\output\results_efficient_ad.json') as f:
    ea_results = json.load(f)

configs = ['N5\nreal_only', 'N5\n+synthetic', 'exp3\nreal_only', 'exp3\n+trad_aug', 'exp3\n+ours']
pc_keys = ['exp1_fewshot/N5_real_only', 'exp1_fewshot/N5_M50',
           'exp3_aug/real_only', 'exp3_aug/traditional_aug', 'exp3_aug/ours']
ea_keys = ['exp1_fewshot/N5_real_only', 'exp1_fewshot/N5_M50',
           'exp3_aug/real_only', 'exp3_aug/traditional_aug', 'exp3_aug/ours']

pc_vals = [results[k]['mean_image_AUROC'] for k in pc_keys]
ea_vals = [ea_results[k]['mean_image_AUROC'] for k in ea_keys]

x = np.arange(len(configs))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, pc_vals, width, label='PatchCore (unsupervised)', color='#2c3e50', edgecolor='white')
bars2 = ax.bar(x + width/2, ea_vals, width, label='efficient_ad (supervised)', color='#3498db', edgecolor='white')

# Value labels
for bar, val in zip(bars1, pc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, ea_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.set_ylabel('Image AUROC', fontsize=12)
ax.set_title('PatchCore vs efficient_ad — Few-Shot Detection Performance', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0.60, 1.0)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
chart4_path = r'D:\desktop\DefectDiffu-main\document\chart_model_comparison.png'
plt.savefig(chart4_path, dpi=150, bbox_inches='tight')
print(f'Saved: {chart4_path}')

print('\nAll charts generated!')
