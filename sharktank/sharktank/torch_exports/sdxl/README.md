# SDXL torch exports

This directory contains export tooling for various functionalities extending from the SDXL1.0 inference pipeline.

## Instant ID

From within this directory,

```
    git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    git clone https://https://huggingface.co/InstantX/InstantID
```

To acquire MLIR export of a denoising module that combines controlnet (InstantID) with unet (IP-Adapter):
```
python export.py --component=controlled_unet
```
To acquire VAE encode/decode module MLIR:
```
python export.py --component=vae
```
