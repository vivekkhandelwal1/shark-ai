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
import base64
import cv2

import shortfin as sf
import shortfin.array as sfnp

from ..messages import InferenceExecRequest, InferencePhase, StrobeMessage
from ..tokenizer import Tokenizer
from ..metrics import measure
from ..face_analysis import IREEFaceAnalysis, draw_kps

logger = logging.getLogger("shortfin-sd.instantid")

prog_isolations = {
    "none": sf.ProgramIsolation.NONE,
    "per_fiber": sf.ProgramIsolation.PER_FIBER,
    "per_call": sf.ProgramIsolation.PER_CALL,
}


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
        self.worker_index = self.service.get_worker_index(fiber)
        self.exec_requests: list[InferenceExecRequest] = []
        self.face_analyzer = IREEFaceAnalysis(name='antelopev2', root="./", dim_param_dict = {'None' : 1, '?' : 640}, extra_compile_args=["--iree-llvmcpu-target-cpu=host"])
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

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
                    self._face_analysis(device=device0, requests=self.exec_requests),
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

    async def _face_analysis(self, device, requests):
        for request in requests:
            image = request.image
            # Copied from diffusers.load_image
            if isinstance(image, str):
                if image.startswith("http://") or image.startswith("https://"):
                    image = PIL.Image.open(requests.get(image, stream=True).raw)
                elif os.path.isfile(image):
                    image = PIL.Image.open(image)
                else:
                    raise ValueError(
                        f"Incorrect path or URL. URLs must start with `http://` or `https://`, and {image} is not a valid path."
                    )
            elif isinstance(image, PIL.Image.Image):
                image = image
            else:
                raise ValueError(
                    "Incorrect format used for the image. Should be a URL linking to an image, a local path, or a PIL image."
                )
            face_image = PIL.ImageOps.exif_transpose(image)
            face_info = self.face_analyzer.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
            face_emb = face_info['embedding']
            face_kps = draw_kps(face_image, face_info['kps'])
            request.prompt_image_emb = face_emb
            request.image_in = face_kps


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

            # Create and populate sample device array.
            generator = sfnp.RandomGenerator(seed)
            request.sample = sfnp.device_array.for_device(
                device, latents_shape, unet_dtype
            )

            sample_host = request.sample.for_transfer()
            with sample_host.map(discard=True) as m:
                m.fill(bytes(1))

            sfnp.fill_randn(sample_host, generator=generator)

            request.sample.copy_from(sample_host)
        return

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
            cfg_mult = 2
            requests[i].prompt_embeds = pe.view(slice(i * cfg_mult, (i + 1) * cfg_mult))
            requests[i].text_embeds = te.view(slice(i * cfg_mult, (i + 1) * cfg_mult))

        return

    async def _denoise(self, device, requests):
        req_bs = len(requests)
        step_count = requests[0].steps
        cfg_mult = 2 if self.service.model_params.cfg_mode else 1
        # Produce denoised latents
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
            "encoder_hidden_states": sfnp.device_array.for_device(
                device, hidden_states_shape, self.service.model_params.unet_dtype
            ),
            "text_embeds": sfnp.device_array.for_device(
                device, text_embeds_shape, self.service.model_params.unet_dtype
            ),
            "guidance_scale": sfnp.device_array.for_device(
                device, [req_bs], self.service.model_params.unet_dtype
            ),
        }

        # Send guidance scale to device.
        gs_host = denoise_inputs["guidance_scale"].for_transfer()
        for i in range(req_bs):
            cfg_dim = i * cfg_mult
            with gs_host.view(i).map(write=True, discard=True) as m:
                # TODO: do this without numpy
                np_arr = np.asarray(requests[i].guidance_scale, dtype="float16")

                m.fill(np_arr)
            # Batch sample latent inputs on device.
            req_samp = requests[i].sample
            denoise_inputs["sample"].view(i).copy_from(req_samp)

            # Batch CLIP hidden states.
            enc = requests[i].prompt_embeds
            denoise_inputs["encoder_hidden_states"].view(
                slice(cfg_dim, cfg_dim + cfg_mult)
            ).copy_from(enc)

            # Batch CLIP text embeds.
            temb = requests[i].text_embeds
            denoise_inputs["text_embeds"].view(
                slice(cfg_dim, cfg_dim + cfg_mult)
            ).copy_from(temb)

        denoise_inputs["guidance_scale"].copy_from(gs_host)

        num_steps = sfnp.device_array.for_device(device, [1], sfnp.sint64)
        ns_host = num_steps.for_transfer()
        with ns_host.map(write=True) as m:
            ns_host.items = [step_count]
        num_steps.copy_from(ns_host)

        init_inputs = [
            denoise_inputs["sample"],
            num_steps,
        ]

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
                denoise_inputs["encoder_hidden_states"],
                denoise_inputs["text_embeds"],
                time_ids,
                denoise_inputs["guidance_scale"],
            ]
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

        for idx, req in enumerate(requests):
            req.denoised_latents = sfnp.device_array.for_device(
                device, latents_shape, self.service.model_params.vae_dtype
            )
            req.denoised_latents.copy_from(latents.view(idx))
        return

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
