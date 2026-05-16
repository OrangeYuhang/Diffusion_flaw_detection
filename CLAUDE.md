# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

DefectDiffu — official PyTorch implementation of "Few-shot Defect Image Generation based on Consistency Modeling" (ECCV 2024). Finetunes a pretrained DiT-XL/2 diffusion transformer with CLIP text conditioning and cross-attention to generate industrial defect images **and** their segmentation masks, trained on MVTec AD.

This repo is the **generation half** of a closed-loop defect detection pipeline. The detection half uses **PatchCore** from the [anomalib](https://github.com/openvinotoolkit/anomalib) library. The flow: DefectDiffu generates synthetic defect samples → `prepare_dataset.py` formats them for anomalib → PatchCore trains on the augmented dataset → detection results feed back to improve generation.

## Environment

Python 3.10, PyTorch 2.12 nightly (CUDA 12.8). See `environment.yml` for the full Conda environment. Key dependencies: `diffusers`, `timm`, `einops`, `clip` (vendored), `gradio`.

Pretrained models required (not in repo):
- [DiT-XL-2-256x256.pt](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) → `./DiT-256/`
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) → `./VAE/`
- MVTec AD dataset → `./data/mvtec/`

## Commands

```bash
# Train with all optimizations (fg/bg loss, normal samples, Dice mask loss)
python train.py --ckpt ./DiT-256/DiT-XL-2-256x256.pt --batchsize 2 --vae ./VAE --data ./data/mvtec --good_ratio 0.3

# Train with PatchCore feature-space consistency loss
python train.py --ckpt ./DiT-256/DiT-XL-2-256x256.pt --batchsize 2 --vae ./VAE --data ./data/mvtec --feature_loss

# Generate defect images for all classes/defects in MVTec (saves to ./img/)
python test.py --ckpt ./checkpoint/model_1500.pth --vae ./VAE --data ./data/mvtec

# Diverse generation (CFG/noise/seed variation for same prompt)
python generate_diverse.py --ckpt checkpoint/model_600.pth --vae ./VAE --defect scratch --product bottle

# Closed-loop adaptive generation (generation → PatchCore feedback loop)
python adaptive_generate.py --ckpt checkpoint/model_600.pth --vae ./VAE --data ./data/mvtec --num_iterations 3

# Mask-conditioned training (位置可控缺陷生成)
python train_mask_condition.py --ckpt ./DiT-256/DiT-XL-2-256x256.pt --batchsize 2 --vae ./VAE --data ./data/mvtec

# Consistency distillation (50-step → 1-4 step)
python consistency_distill.py --teacher_ckpt checkpoint/model_600.pth --vae ./VAE --data ./data/mvtec --num_student_steps 4

# Launch Gradio web UI
python app.py
```

## Architecture

**DiT backbone** (`models_add_cross_concate.py`): 28 transformer blocks operating on VAE latents (32×32 patches). Three conditioning stages:
- Blocks 0–9: conditioned on **product class** (e.g., "bottle")
- Blocks 10–19: conditioned on **defect type** (e.g., "scratch") via `CrossAttention` — these cross-attention maps are accumulated and decoded into the output mask via `temp_Adaptive_Mask`
- Blocks 20–27: conditioned on **combined** text (defect + product)

**Dual output**: The model outputs both the denoised latent (8 channels: mean + variance) and a defect mask latent (4 channels), both decoded through the same SD VAE.

**Classifier-free guidance** (`forward_with_cfg_2` / `forward_with_cfg_3`): Batches conditional + unconditional forward passes, applies CFG to the first 3 channels. The `cfg_scale` parameter controls defect strength (higher = stronger defects).

**Diffusion** (`diffusion/`): Modified from OpenAI's GLIDE/ADM. Supports respaced timesteps for faster sampling (typically `"50"` steps at inference, full 1000 during training).

**Text encoding**: CLIP RN50 (`clip/` — vendored copy). Text prompts follow the format `"a photo of {defect} {product}"`. Training uses three text embeddings per sample: defect-only, class-only, and combined.

## Data layout (MVTec AD)

```
data/mvtec/
  <class>/train/good/          # normal (defect-free) training images
  <class>/test/<defect_type>/  # defect images
  <class>/ground_truth/<defect_type>/  # segmentation masks
```

## Key details

- The VAE latent scaling factor is **0.18215** (Stable Diffusion convention)
- Training uses `torch.backends.cuda.matmul.allow_tf32 = True`
- Checkpoints saved every 100 epochs to `checkpoint/model_{epoch}.pth`
- The `--free` flag (0/1/2) in train/test controls the CFG branch strategy (2-branch vs 3-branch)
- `--good_ratio` (default 0.3) adds normal samples to training with zero masks, improving background realism
- `--feature_loss` enables PatchCore-aligned WideResNet-50 feature-space consistency loss
- CFG now applies to all 4 VAE latent channels (not just 3)
- `train.bak.py` is an earlier version of the training script — the active one is `train.py`

## New modules

- `feature_loss.py` — PatchCore-aligned WideResNet-50 feature extractor + Mahalanobis consistency loss for non-defect regions
- `models_mask_condition.py` — DiT variant with explicit mask conditioning for location-controllable generation
- `train_mask_condition.py` — Training script for DiTMaskConditioned (uses ground truth masks from MVTec to learn mask→position mapping)
- `generate_diverse.py` — Multi-axis diversity generation (CFG sweep, noise scale sweep, seed variation, grid)
- `adaptive_generate.py` — Closed-loop: generates → runs PatchCore → allocates budget to weak classes → iterates
- `consistency_distill.py` — LCM-style consistency distillation from 50-step to 1-4 step sampling

## Code style (behavioral guidelines)

**Think before coding.** Surface assumptions and tradeoffs before implementing. If multiple interpretations exist, present them — don't pick silently. If something is unclear, ask.

**Simplicity first.** Minimum code that solves the problem. No abstractions for single-use code, no configurability that wasn't requested, no error handling for impossible scenarios. Match existing code style — don't refactor adjacent code that isn't broken.

**Surgical changes.** Touch only what you must. Don't "improve" unrelated code, comments, or formatting. Remove only imports/variables that *your* changes made unused. Every changed line should trace directly to the request.

**Goal-driven.** State verifiable success criteria before implementing. For multi-step tasks, lay out the plan concisely. Define what "done" looks like so you can loop independently.
