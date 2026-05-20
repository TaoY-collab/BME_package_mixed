# Coarse-to-Fine SwinUNETR/Swin-Unet Workflow

This branch keeps the earlier late-fusion model and adds a residual coarse-to-fine
variant for the proposed nodule workflow.

## Model Modes

- `BME_MODEL=swinunetr`: original 3D SwinUNETR.
- `BME_MODEL=dual_swin_fusion`: previous 3D SwinUNETR plus slice-wise 2D Swin-Unet late fusion.
- `BME_MODEL=coarse_to_fine`: new 3D coarse branch plus 2.5D residual refiner.

The new coarse-to-fine model feeds the 2.5D refiner with:

- neighboring CT slices;
- the 3D coarse foreground probability;
- the 3D coarse uncertainty map.

The final logits are:

```text
fused_logits = coarse_3d_logits + tanh(residual_scale) * refine_2d_logits
```

This keeps the 3D SwinUNETR as the high-recall locator and makes Swin-Unet learn
residual boundary and missed-region recovery.

## Loss

Use `BME_LOSS_MODE=coarse_to_fine` with `BME_MODEL=coarse_to_fine`.

The loss combines:

- final fused Dice/Tversky/boundary loss;
- auxiliary coarse 3D loss;
- auxiliary 2.5D refine loss;
- residual recall loss for coarse-missed target voxels;
- false-positive suppression on low-coarse-confidence background;
- background consistency between fused and coarse probabilities.

## RTX 4060 Test Profile

Use this profile to verify code paths and run small experiments:

```powershell
$env:BME_MODEL="coarse_to_fine"
$env:BME_LOSS_MODE="coarse_to_fine"
$env:BME_STAGE2_REFINEMENT="0"
$env:BME_ROI_SIZE="64"
$env:BME_FEATURE_SIZE="12"
$env:BME_DUAL_2D_FEATURE_SIZE="12"
$env:BME_CTF_CONTEXT_SLICES="3"
$env:BME_TRAIN_BATCH_SIZE="1"
$env:BME_ACCUMULATION_STEPS="4"
$env:BME_AMP_MODE="fp16"
$env:BME_MAX_DATASET_SIZE="32"
$env:BME_MAX_EPOCHS="5"
python train.py --train-mode 3d
```

If 8GB VRAM is still tight, reduce `BME_MAX_DATASET_SIZE`, set
`BME_CTF_CONTEXT_SLICES=1`, or test `BME_MODEL=swinunetr` first.

## DGX Spark Formal Profile

Use this once the 4060 path is stable:

```powershell
$env:BME_MODEL="coarse_to_fine"
$env:BME_LOSS_MODE="coarse_to_fine"
$env:BME_STAGE2_REFINEMENT="0"
$env:BME_ROI_SIZE="96"
$env:BME_FEATURE_SIZE="24"
$env:BME_DUAL_2D_FEATURE_SIZE="24"
$env:BME_CTF_CONTEXT_SLICES="3"
$env:BME_TRAIN_BATCH_SIZE="1"
$env:BME_ACCUMULATION_STEPS="2"
$env:BME_AMP_MODE="bf16"
$env:BME_MAX_DATASET_SIZE="400"
$env:BME_MAX_EPOCHS="200"
python train.py --train-mode 3d
```

After a strong SwinUNETR checkpoint exists, set `BME_STAGE2_REFINEMENT=1` and
point `BME_STAGE2_SOURCE_CKPT` at that checkpoint to run residual refinement
from a pretrained coarse branch.
