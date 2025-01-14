# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
import base64

import shortfin as sf
import shortfin.array as sfnp

from .config_struct import ModelParams
from .manager import SystemManager
from .messages import InferenceExecRequest, InferencePhase, StrobeMessage
from .tokenizer import Tokenizer
from .metrics import measure
from .pipelines.pipeline_sdxl_txt2img import InferenceExecutorProcess as Txt2ImgProc
from .pipelines.pipeline_sdxl_instantid import InferenceExecutorProcess as InstantIDProc

logger = logging.getLogger("shortfin-sd.service")

prog_isolations = {
    "none": sf.ProgramIsolation.NONE,
    "per_fiber": sf.ProgramIsolation.PER_FIBER,
    "per_call": sf.ProgramIsolation.PER_CALL,
}


class GenerateService:
    """Top level service interface for image generation."""

    inference_programs: dict[str, sf.Program]

    inference_functions: dict[str, dict[str, sf.ProgramFunction]]

    def __init__(
        self,
        *,
        name: str,
        sysman: SystemManager,
        tokenizers: list[Tokenizer],
        model_params: ModelParams,
        fibers_per_device: int,
        workers_per_device: int = 1,
        prog_isolation: str = "per_fiber",
        show_progress: bool = False,
        trace_execution: bool = False,
    ):
        self.name = name

        # Application objects.
        self.sysman = sysman
        self.tokenizers = tokenizers
        self.model_params = model_params
        self.inference_parameters: dict[str, list[sf.BaseProgramParameters]] = {}
        self.inference_modules: dict[str, sf.ProgramModule] = {}
        self.inference_functions: dict[str, dict[str, sf.ProgramFunction]] = {}
        self.inference_programs: dict[int, dict[str, sf.Program]] = {}
        self.trace_execution = trace_execution
        self.show_progress = show_progress

        self.prog_isolation = prog_isolations[prog_isolation]

        self.workers_per_device = workers_per_device
        self.fibers_per_device = fibers_per_device
        if fibers_per_device % workers_per_device != 0:
            raise ValueError(
                "Currently, fibers_per_device must be divisible by workers_per_device"
            )
        self.fibers_per_worker = int(fibers_per_device / workers_per_device)

        self.workers = []
        self.fibers = []
        self.idle_fibers = set()
        # For each worker index we create one on each device, and add their fibers to the idle set.
        # This roughly ensures that the first picked fibers are distributed across available devices.
        for i in range(self.workers_per_device):
            for idx, device in enumerate(self.sysman.ls.devices):
                worker = sysman.ls.create_worker(f"{name}-inference-{device.name}-{i}")
                self.workers.append(worker)
        for idx, device in enumerate(self.sysman.ls.devices):
            for i in range(self.fibers_per_device):
                tgt_worker = self.workers[i % len(self.workers)]
                fiber = sysman.ls.create_fiber(tgt_worker, devices=[device])
                self.fibers.append(fiber)
                self.idle_fibers.add(fiber)
        for idx in range(len(self.workers)):
            self.inference_programs[idx] = {}
            self.inference_functions[idx] = {}
        # Scope dependent objects.
        self.batcher = BatcherProcess(self)

        self.pipeline = model_params.pipeline

    def get_worker_index(self, fiber):
        if fiber not in self.fibers:
            raise ValueError("A worker was requested from a rogue fiber.")
        fiber_idx = self.fibers.index(fiber)
        worker_idx = int(
            (fiber_idx - fiber_idx % self.fibers_per_worker) / self.fibers_per_worker
        )
        return worker_idx

    def load_inference_module(self, vmfb_path: Path, component: str = None):
        if not self.inference_modules.get(component):
            self.inference_modules[component] = []
        self.inference_modules[component].append(
            sf.ProgramModule.load(self.sysman.ls, vmfb_path)
        )

    def load_inference_parameters(
        self,
        *paths: Path,
        parameter_scope: str,
        format: str = "",
        component: str = None,
    ):
        p = sf.StaticProgramParameters(self.sysman.ls, parameter_scope=parameter_scope)
        for path in paths:
            logger.info("Loading parameter fiber '%s' from: %s", parameter_scope, path)
            p.load(path, format=format)
        if not self.inference_parameters.get(component):
            self.inference_parameters[component] = []
        self.inference_parameters[component].append(p)

    def start(self):
        # Initialize programs.
        for component in self.inference_modules:
            logger.info(f"Loading component: {component}")
            component_modules = [
                sf.ProgramModule.parameter_provider(
                    self.sysman.ls, *self.inference_parameters.get(component, [])
                ),
                *self.inference_modules[component],
            ]

            for worker_idx, worker in enumerate(self.workers):
                worker_devices = self.fibers[
                    worker_idx * (self.fibers_per_worker)
                ].raw_devices
                logger.info(
                    f"Loading inference program: {component}, worker index: {worker_idx}, device: {worker_devices}"
                )
                self.inference_programs[worker_idx][component] = sf.Program(
                    modules=component_modules,
                    devices=worker_devices,
                    isolation=self.prog_isolation,
                    trace_execution=self.trace_execution,
                )

        for worker_idx, worker in enumerate(self.workers):
            self.inference_functions[worker_idx]["encode"] = {}
            for bs in self.model_params.clip_batch_sizes:
                self.inference_functions[worker_idx]["encode"][
                    bs
                ] = {
                    "encode": self.inference_programs[worker_idx]["clip"][
                        f"{self.model_params.clip_module_name}.encode_prompts"
                    ]
                }
                if self.model_params.pipeline == "instantid":
                    self.inference_functions[worker_idx]["encode"][
                        bs
                    ]["image_proj"] = self.inference_programs[worker_idx]["resampler"][
                        f"compiled_resampler.run_image_proj"
                    ]
            
            self.inference_functions[worker_idx]["denoise"] = {}
            for bs in self.model_params.unet_batch_sizes:
                self.inference_functions[worker_idx]["denoise"][bs] = {
                    "unet": self.inference_programs[worker_idx]["unet"][
                        f"{self.model_params.unet_module_name}.{self.model_params.unet_fn_name}"
                    ],
                    "init": self.inference_programs[worker_idx]["scheduler"][
                        f"{self.model_params.scheduler_module_name}.run_initialize"
                    ],
                    "scale": self.inference_programs[worker_idx]["scheduler"][
                        f"{self.model_params.scheduler_module_name}.run_scale"
                    ],
                    "step": self.inference_programs[worker_idx]["scheduler"][
                        f"{self.model_params.scheduler_module_name}.run_step"
                    ],
                }
            self.inference_functions[worker_idx]["decode"] = {}
            for bs in self.model_params.vae_batch_sizes:
                self.inference_functions[worker_idx]["decode"][
                    bs
                ] = self.inference_programs[worker_idx]["vae"][
                    f"{self.model_params.vae_module_name}.decode"
                ]
        self.batcher.launch()

    def shutdown(self):
        self.batcher.shutdown()

    def __repr__(self):
        modules = [
            f"     {key} : {value}" for key, value in self.inference_modules.items()
        ]
        params = [
            f"     {key} : {value}" for key, value in self.inference_parameters.items()
        ]
        # For python 3.11 since we can't have \ in the f"" expression.
        new_line = "\n"
        return (
            f"ServiceManager("
            f"\n  INFERENCE DEVICES : \n"
            f"     {self.sysman.ls.devices}\n"
            f"\n  MODEL PARAMS : \n"
            f"{self.model_params}"
            f"\n  SERVICE PARAMS : \n"
            f"     fibers per device : {self.fibers_per_device}\n"
            f"     program isolation mode : {self.prog_isolation}\n"
            f"\n  INFERENCE MODULES : \n"
            f"{new_line.join(modules)}\n"
            f"\n  INFERENCE PARAMETERS : \n"
            f"{new_line.join(params)}\n"
            f")"
        )


########################################################################################
# Batcher
########################################################################################


class BatcherProcess(sf.Process):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches.
    """

    STROBE_SHORT_DELAY = 0.5
    STROBE_LONG_DELAY = 1

    def __init__(self, service: GenerateService):
        super().__init__(fiber=service.fibers[0])
        self.service = service
        self.batcher_infeed = self.system.create_queue()
        self.pending_requests: set[InferenceExecRequest] = set()
        self.strobe_enabled = True
        self.strobes: int = 0
        self.ideal_batch_size: int = max(service.model_params.max_batch_size)
        self.num_fibers = len(service.fibers)

    def shutdown(self):
        self.batcher_infeed.close()

    def submit(self, request: StrobeMessage | InferenceExecRequest):
        self.batcher_infeed.write_nodelay(request)

    async def _background_strober(self):
        while not self.batcher_infeed.closed:
            await asyncio.sleep(
                BatcherProcess.STROBE_SHORT_DELAY
                if len(self.pending_requests) > 0
                else BatcherProcess.STROBE_LONG_DELAY
            )
            if self.strobe_enabled:
                self.submit(StrobeMessage())

    async def run(self):
        strober_task = asyncio.create_task(self._background_strober())
        reader = self.batcher_infeed.reader()
        while item := await reader():
            self.strobe_enabled = False
            if isinstance(item, InferenceExecRequest):
                self.pending_requests.add(item)
            elif isinstance(item, StrobeMessage):
                self.strobes += 1
            else:
                logger.error("Illegal message received by batcher: %r", item)

            self.board_flights()

            self.strobe_enabled = True
        await strober_task

    def board_flights(self):
        waiting_count = len(self.pending_requests)
        if waiting_count == 0:
            return
        if waiting_count < self.ideal_batch_size and self.strobes < 2:
            logger.info("Waiting a bit longer to fill flight")
            return
        self.strobes = 0
        batches = self.sort_batches()
        for batch in batches.values():
            # Assign the batch to the next idle fiber.
            if len(self.service.idle_fibers) == 0:
                return
            fiber = self.service.idle_fibers.pop()
            fiber_idx = self.service.fibers.index(fiber)
            worker_idx = self.service.get_worker_index(fiber)
            logger.debug(f"Sending batch to fiber {fiber_idx} (worker {worker_idx})")
            self.board(batch["reqs"], fiber=fiber)
            if self.service.prog_isolation != sf.ProgramIsolation.PER_FIBER:
                self.service.idle_fibers.add(fiber)

    def sort_batches(self):
        """Files pending requests into sorted batches suitable for program invocations."""
        reqs = self.pending_requests
        next_key = 0
        batches = {}
        for req in reqs:
            is_sorted = False
            req_metas = [req.phases[phase]["metadata"] for phase in req.phases.keys()]

            for idx_key, data in batches.items():
                if not isinstance(data, dict):
                    logger.error(
                        "Expected to find a dictionary containing a list of requests and their shared metadatas."
                    )
                if len(batches[idx_key]["reqs"]) >= self.ideal_batch_size:
                    # Batch is full
                    next_key = idx_key + 1
                    continue
                elif data["meta"] == req_metas:
                    batches[idx_key]["reqs"].extend([req])
                    is_sorted = True
                    break
                else:
                    next_key = idx_key + 1
            if not is_sorted:
                batches[next_key] = {
                    "reqs": [req],
                    "meta": req_metas,
                }
        return batches

    def board(self, request_bundle, fiber):
        pending = request_bundle
        if len(pending) == 0:
            return
        if self.pipeline == "instantid":
            exec_process = InstantIDProc(self.service, fiber)
        else:
            exec_process = Txt2ImgProc(self.service, fiber)
        for req in pending:
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break
            exec_process.exec_requests.append(req)
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.pending_requests.remove(flighted_request)
            exec_process.launch()