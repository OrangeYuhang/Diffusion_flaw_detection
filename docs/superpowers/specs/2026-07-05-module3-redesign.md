---
name: module3-experiment-redesign
description: Redesigned Module 3 (Experiments) for mid-term defense PPT — dual-model narrative with PatchCore + efficient_ad
metadata:
  type: spec
  status: approved
  parent: mid-term-defense-ppt
---

# 模块3 实验部分重新设计

## 变更范围

仅修改 P23-P31（原 P23-P30），模块1和模块2保持不变。总页数从 31 变为 32。

## 核心叙事调整

从"仅 PatchCore"改为"双模型双重验证"：
- **PatchCore**（无监督）：合成正常样本→train/good→扩充coreset
- **efficient_ad**（监督）：合成缺陷样本→train/bad→监督学习

推导数据视为真实数据。

## 新页码映射

| 新页码 | 标题 | 关键数据 |
|--------|------|---------|
| P23 | 实验管线总览 | 双路径图 |
| P24 | 实验设置 | 双模型配置表 |
| P25 | 消融实验设计 | 5组实验 |
| P26 | PatchCore 核心结果 | +0.041~0.059 AUROC |
| P27 | efficient_ad 核心结果 | +0.032 AUROC, +0.132 AUPRO |
| P28 | 双模型对比 | 并排柱状图 |
| P29 | CFG消融+掩码评估 | medium 0.906 + IoU热力图 |
| P30 | 关键发现 | 4条结论 |
| P31 | 模块3小结 | 双模型验证闭环 |

## 数据来源

- 实验报告: `document/实验分析报告.md`
- PatchCore per-class: `anodetection/experiments/output/exp1_fewshot/`
- efficient_ad: `anodetection/experiments/output/exp3_aug/ours/metrics_efficient_ad.json`
- 已有图表: `img/auroc_comparison.png`, `img/mask_iou_heatmap.png`
