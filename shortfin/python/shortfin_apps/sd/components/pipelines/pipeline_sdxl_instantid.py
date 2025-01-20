# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import numpy as np
import os
import PIL
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import requests
import base64
import cv2
from io import BytesIO
from diffusers.utils.torch_utils import randn_tensor
import torch

import shortfin as sf
import shortfin.array as sfnp

from ..messages import InferenceExecRequest, InferencePhase, StrobeMessage
from ..tokenizer import Tokenizer
from ..metrics import measure
from ..face_analysis import draw_kps

logger = logging.getLogger("shortfin-sd.instantid")

prog_isolations = {
    "none": sf.ProgramIsolation.NONE,
    "per_fiber": sf.ProgramIsolation.PER_FIBER,
    "per_call": sf.ProgramIsolation.PER_CALL,
}

def prepare_image_controlnet(image, width, height, dtype):
    image = image.convert("RGB")
    image = image.resize((width, height), resample=Image.LANCZOS)
    image = np.asarray(image)
    image = np.array([image.transpose((2, 0, 1))]*2)
    return image.astype(dtype.name)


########################################################################################
# Inference Executors
########################################################################################

class InferenceExecutorProcess(sf.Process):
    """Executes a stable diffusion inference batch"""

    def __init__(
        self,
        service,
        fiber,
    ):
        super().__init__(fiber=fiber)
        self.service = service
        self.cfg_mult = 2 if self.service.model_params.cfg_mode else 1
        self.worker_index = self.service.get_worker_index(fiber)
        self.exec_requests: list[InferenceExecRequest] = []

    @measure(type="exec", task="inference process")
    async def run(self):
        try:
            phase = None
            for req in self.exec_requests:
                if phase:
                    if phase != req.phase:
                        logger.error("Executor process recieved disjoint batch.")
                phase = req.phase
            phases = self.exec_requests[0].phases
            req_count = len(self.exec_requests)
            device0 = self.fiber.device(0)
            if phases[InferencePhase.PREPARE]["required"]:
                procs = [
                    self._face_analysis(device=device0, reqs=self.exec_requests),
                    self._prepare(device=device0, requests=self.exec_requests),
                ]
                await asyncio.gather(*procs)
            if phases[InferencePhase.ENCODE]["required"]:
                await self._encode(device=device0, requests=self.exec_requests)
            if phases[InferencePhase.DENOISE]["required"]:
                await self._denoise(device=device0, requests=self.exec_requests)
            if phases[InferencePhase.DECODE]["required"]:
                await self._decode(device=device0, requests=self.exec_requests)
            if phases[InferencePhase.POSTPROCESS]["required"]:
                await self._postprocess(device=device0, requests=self.exec_requests)
            await device0
            for i in range(req_count):
                req = self.exec_requests[i]
                req.done.set_success()
            if self.service.prog_isolation == sf.ProgramIsolation.PER_FIBER:
                self.service.idle_fibers.add(self.fiber)

        except Exception:
            logger.exception("Fatal error in image generation")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.done.set_success()

    @measure(type="exec", task="face analysis")
    async def _face_analysis(self, device, reqs):
        for request in reqs:
            url = request.image
            response = requests.get(url, stream=True).raw
            face_image = Image.open(response)
            face_info = self.service.face_analyzers[self.worker_index].get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
            face_emb = face_info['embedding'].astype("float32")
            face_kps = draw_kps(face_image, face_info['kps'])
            face_kps.save("face_kps_sfsdxl.png")
            face_kps_prep = prepare_image_controlnet(face_kps, request.width, request.height, self.service.model_params.unet_dtype)
            image_in_np = np.asarray(face_kps_prep).astype("float32")
            request.image_emb = sfnp.device_array.for_device(
                device, (512,), self.service.model_params.unet_dtype
            )
            # shape[0] of image_in is program batch size * classifier free guidance multiplier (1 or 2)
            request.image_in = sfnp.device_array.for_device(
                device, (self.cfg_mult, 3, request.height, request.width), self.service.model_params.unet_dtype
            )
            emb_host = request.image_emb.for_host(
                device, (512,), sfnp.float32
            )
            with emb_host.map(discard=True) as m:
                m.fill(face_emb)
            kps_host = sfnp.device_array.for_host(
                device, (self.cfg_mult, 3, request.height, request.width), sfnp.float32
            )
            with kps_host.map(discard=True) as m:
                m.fill(image_in_np)

            if self.service.model_params.unet_dtype != sfnp.float32:
                kps_cast = request.image_in.for_transfer()
                emb_cast = request.image_emb.for_transfer()
                with kps_cast.map(discard=True) as m:
                    m.fill(0)
                with emb_cast.map(discard=True) as m:
                    m.fill(0)
                sfnp.convert(kps_host, dtype=self.service.model_params.unet_dtype, out=kps_cast)
                sfnp.convert(emb_host, dtype=self.service.model_params.unet_dtype, out=emb_cast)
                request.image_in.copy_from(kps_cast)
                request.image_emb.copy_from(emb_cast)
            else:
                request.image_in.copy_from(kps_host)
                request.image_emb.copy_from(emb_host)

            await device

    @measure(type="exec", task="prepare")
    async def _prepare(self, device, requests):
        for request in requests:
            # Tokenize prompts and negative prompts. We tokenize in bs1 for now and join later.
            input_ids_list = []
            neg_ids_list = []
            for tokenizer in self.service.tokenizers:
                input_ids = tokenizer.encode(request.prompt)
                input_ids_list.append(input_ids)
                neg_ids = tokenizer.encode(request.neg_prompt)
                neg_ids_list.append(neg_ids)
            ids_list = [*input_ids_list, *neg_ids_list]

            request.input_ids = ids_list

            # Generate random sample latents.
            seed = request.seed
            channels = self.service.model_params.num_latents_channels
            unet_dtype = self.service.model_params.unet_dtype
            latents_shape = (
                1,
                channels,
                request.height // 8,
                request.width // 8,
            )
            generator = torch.Generator()
            generator = generator.manual_seed(seed)
            latents = randn_tensor(latents_shape, generator=generator, dtype=torch.float16).numpy()
            # Create and populate sample device array.
            # generator = sfnp.RandomGenerator(seed)
            request.sample = sfnp.device_array.for_device(
                device, latents_shape, unet_dtype
            )

            sample_host = request.sample.for_transfer()
            with sample_host.map(discard=True) as m:
                m.fill(latents)

            # sfnp.fill_randn(sample_host, generator=generator)

            request.sample.copy_from(sample_host)
            await device
        return

    @measure(type="exec", task="clip encode")
    async def _encode(self, device, requests):
        req_bs = len(requests)
        entrypoints = self.service.inference_functions[self.worker_index]["encode"]
        if req_bs not in list(entrypoints.keys()):
            for request in requests:
                await self._encode(device, [request])
            return
        for bs, fn in entrypoints.items():
            if bs == req_bs:
                break

        resampler_fn = self.service.inference_functions[self.worker_index]["resample"]
        prompt_img_emb_shape = [
            req_bs * self.cfg_mult,
            16,
            2048
        ]
        for i in range(req_bs):
            image_emb = requests[i].image_emb
            (requests[i].prompt_image_emb,) = await resampler_fn(image_emb, fiber=self.fiber)

        # Prepare tokenized input ids for CLIP inference

        clip_inputs = [
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
            sfnp.device_array.for_device(
                device, [req_bs, self.service.model_params.max_seq_len], sfnp.sint64
            ),
        ]
        host_arrs = [None] * 4
        for idx, arr in enumerate(clip_inputs):
            host_arrs[idx] = arr.for_transfer()
            for i in range(req_bs):
                with host_arrs[idx].view(i).map(write=True, discard=True) as m:

                    # TODO: fix this attr redundancy
                    np_arr = requests[i].input_ids[idx].input_ids

                    m.fill(np_arr)
            clip_inputs[idx].copy_from(host_arrs[idx])

        # Encode tokenized inputs.
        logger.debug(
            "INVOKE %r: %s",
            fn,
            "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(clip_inputs)]),
        )
        await device
        pe, te = await fn(*clip_inputs, fiber=self.fiber)

        for i in range(req_bs):
            cfg_mult = 2 if self.service.model_params.cfg_mode else 1
            requests[i].prompt_embeds = pe.view(slice(i * cfg_mult, (i + 1) * cfg_mult))
            requests[i].text_embeds = te.view(slice(i * cfg_mult, (i + 1) * cfg_mult))
        return

    @measure(type="exec", task="denoise")
    async def _denoise(self, device, requests):
        req_bs = len(requests)
        step_count = requests[0].steps
        cfg_mult = self.cfg_mult
        # Produce denoised latents
        
        await device
        entrypoints = self.service.inference_functions[self.worker_index]["denoise"]
        if req_bs not in list(entrypoints.keys()):
            for request in requests:
                await self._denoise(device, [request])
            return
        for bs, fns in entrypoints.items():
            if bs == req_bs:
                break

        # Get shape of batched latents.
        # This assumes all requests are dense at this point.
        latents_shape = [
            req_bs,
            self.service.model_params.num_latents_channels,
            requests[0].height // 8,
            requests[0].width // 8,
        ]
        # Prepared controlnet input image.
        image_shape = [
            req_bs * cfg_mult,
            3,
            requests[0].height,
            requests[0].width,
        ]
        prompt_img_emb_shape = [
            req_bs * cfg_mult,
            16,
            2048
        ]
        # Assume we are doing classifier-free guidance
        hidden_states_shape = [
            req_bs * cfg_mult,
            self.service.model_params.max_seq_len,
            2048,
        ]
        text_embeds_shape = [
            req_bs * cfg_mult,
            1280,
        ]
        denoise_inputs = {
            "sample": sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.unet_dtype
            ),
            "image": sfnp.device_array.for_device(
                device, image_shape, self.service.model_params.unet_dtype
            ),
            "prompt_embeds": sfnp.device_array.for_device(
                device, hidden_states_shape, self.service.model_params.unet_dtype
            ),
            "text_embeds": sfnp.device_array.for_device(
                device, text_embeds_shape, self.service.model_params.unet_dtype
            ),
            "cond_scale": sfnp.device_array.for_device(
                device, [1], self.service.model_params.unet_dtype
            ),
            "guidance_scale": sfnp.device_array.for_device(
                device, [1], self.service.model_params.unet_dtype
            ),
        }
        await device
        # Send guidance scale to device.
        gs_host = denoise_inputs["guidance_scale"].for_transfer()
        cs_host = denoise_inputs["cond_scale"].for_transfer()
        for i in range(req_bs):
            cfg_dim = i * cfg_mult
            with gs_host.view(i).map(write=True, discard=True) as m:
                # TODO: do this without numpy
                np_arr = np.asarray(requests[i].guidance_scale, dtype="float16")

                m.fill(np_arr)
            with cs_host.view(i).map(write=True, discard=True) as m:
                # TODO: do this without numpy
                np_arr = np.asarray(requests[i].cond_scale, dtype="float16")

                m.fill(np_arr)
            # Batch sample latent inputs on device.
            req_samp = requests[i].sample
            denoise_inputs["sample"].view(i).copy_from(req_samp)

            req_img = requests[i].image_in
            req_img_embeds = requests[i].prompt_image_emb
            enc = requests[i].prompt_embeds
            temb = requests[i].text_embeds
            denoise_inputs["prompt_image_emb"] = requests[i].prompt_image_emb
            if self.service.model_params.cfg_mode:
                denoise_inputs["image"].view(
                    slice(cfg_dim, cfg_dim + cfg_mult)
                ).copy_from(req_img)
                denoise_inputs["prompt_embeds"].view(
                    slice(cfg_dim, cfg_dim + cfg_mult)
                ).copy_from(enc)
                denoise_inputs["text_embeds"].view(
                    slice(cfg_dim, cfg_dim + cfg_mult)
                ).copy_from(temb)
            else:
                denoise_inputs["image"].view(i).copy_from(req_img)
                
                denoise_inputs["prompt_embeds"].view(i).copy_from(enc)

                denoise_inputs["text_embeds"].view(i).copy_from(temb)
        denoise_inputs["guidance_scale"].copy_from(gs_host)
        denoise_inputs["cond_scale"].copy_from(cs_host)
        await device

        img_host = denoise_inputs["image"].for_transfer()
        img_host.copy_from(denoise_inputs["image"])
        await device

        num_steps = sfnp.device_array.for_device(device, [1], sfnp.sint64)
        ns_host = num_steps.for_transfer()
        with ns_host.map(write=True) as m:
            ns_host.items = [step_count]
        num_steps.copy_from(ns_host)

        init_inputs = [
            denoise_inputs["sample"],
            num_steps,
        ]
        
        await device
        # Initialize scheduler.
        logger.debug(
            "INVOKE %r",
            fns["init"],
        )
        (latents, time_ids, timesteps, sigmas) = await fns["init"](
            *init_inputs, fiber=self.fiber
        )
        for i, t in tqdm(
            enumerate(range(step_count)),
            disable=(not self.service.show_progress),
            desc=f"DENOISE (bs{req_bs})",
        ):
            step = sfnp.device_array.for_device(device, [1], sfnp.sint64)
            s_host = step.for_transfer()
            with s_host.map(write=True) as m:
                s_host.items = [i]
            step.copy_from(s_host)
            scale_inputs = [latents, step, timesteps, sigmas]
            logger.debug(
                "INVOKE %r",
                fns["scale"],
            )
            latent_model_input, t, sigma, next_sigma = await fns["scale"](
                *scale_inputs, fiber=self.fiber
            )
            await device

            unet_inputs = [
                latent_model_input,
                t,
                denoise_inputs["image"],
                denoise_inputs["prompt_embeds"],
                denoise_inputs["text_embeds"],
                time_ids,
                denoise_inputs["prompt_image_emb"],
                denoise_inputs["cond_scale"],
                denoise_inputs["guidance_scale"],
            ]
            # for idx, i in enumerate(unet_inputs):
            #     printable = i.for_transfer()
            #     printable.copy_from(i)
            #     await device
            #     print("UNET INPUT ", idx)
            #     print(printable)
            logger.debug(
                "INVOKE %r",
                fns["unet"],
            )
            (noise_pred,) = await fns["unet"](*unet_inputs, fiber=self.fiber)

            step_inputs = [noise_pred, latents, sigma, next_sigma]
            logger.debug(
                "INVOKE %r",
                fns["step"],
            )
            (latent_model_output,) = await fns["step"](*step_inputs, fiber=self.fiber)
            latents.copy_from(latent_model_output)
            await device

        for idx, req in enumerate(requests):
            req.denoised_latents = sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.vae_dtype
            )
            req.denoised_latents.copy_from(latents.view(idx))
        return

    @measure(type="exec", task="vae decode")
    async def _decode(self, device, requests):
        req_bs = len(requests)
        # Decode latents to images
        entrypoints = self.service.inference_functions[self.worker_index]["decode"]
        if req_bs not in list(entrypoints.keys()):
            for request in requests:
                await self._decode(device, [request])
            return
        for bs, fn in entrypoints.items():
            if bs == req_bs:
                break

        latents_shape = [
            req_bs,
            self.service.model_params.num_latents_channels,
            requests[0].height // 8,
            requests[0].width // 8,
        ]
        latents = sfnp.device_array.for_device(
            device, latents_shape, self.service.model_params.vae_dtype
        )
        for i in range(req_bs):
            latents.view(i).copy_from(requests[i].denoised_latents)

        await device
        # Decode the denoised latents.
        logger.debug(
            "INVOKE %r: %s",
            fn,
            "".join([f"\n  0: {latents.shape}"]),
        )
        (image,) = await fn(latents, fiber=self.fiber)

        await device
        images_shape = [
            req_bs,
            3,
            requests[0].height,
            requests[0].width,
        ]
        image_shape = [
            req_bs,
            3,
            requests[0].height,
            requests[0].width,
        ]
        images_host = sfnp.device_array.for_host(device, images_shape, sfnp.float16)
        images_host.copy_from(image)
        await device
        for idx, req in enumerate(requests):
            image_array = images_host.view(idx).items
            dtype = image_array.typecode
            if images_host.dtype == sfnp.float16:
                dtype = np.float16
            req.image_array = np.frombuffer(image_array, dtype=dtype).reshape(
                *image_shape
            )
        return

    @measure(type="exec", task="postprocess")
    async def _postprocess(self, device, requests):
        # Process output images
        for req in requests:
            # TODO: reimpl with sfnp
            permuted = np.transpose(req.image_array, (0, 2, 3, 1))[0]
            cast_image = (permuted * 255).round().astype("uint8")
            image_bytes = Image.fromarray(cast_image).tobytes()

            image = base64.b64encode(image_bytes).decode("utf-8")
            req.result_image = image
        return
