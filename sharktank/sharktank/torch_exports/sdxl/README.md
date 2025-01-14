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


Run compiled unet module with generated sample inputs:
```
iree-run-module --module=stable_diffusion_xl_base_1_0_controlled_unet_bs1_64_1024x960_fp16_amdgpu_gfx942.vmfb --device=hip://0 --input=@controlled_unet_0_input_0.npy --input=@controlled_unet_0_input_1.npy --input=@controlled_unet_0_input_2.npy --input=@controlled_unet_0_input_3.npy --input=@controlled_unet_0_input_4.npy --input=@controlled_unet_0_input_5.npy --input=@controlled_unet_0_input_6.npy --input=@controlled_unet_0_input_7.npy --input=@controlled_unet_0_input_8.npy --device_allocator=caching --parameters=model=stable_diffusion_xl_base_1_0_controlled_unet_fp16.irpa --function=run_forward
```