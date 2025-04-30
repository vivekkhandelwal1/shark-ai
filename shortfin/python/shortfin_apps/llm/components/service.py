# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from dataclasses import dataclass
from typing import List

import shortfin as sf
import shortfin.array as sfnp


from .batcher import (
    PrefillBatcherProcess,
    DecodeBatcherProcess,
    FiberPool,
    MetaFiber,
    initialize_buffer_object,
    LlmBufferObject,
)
from .config_struct import ModelParams, ServerParams
from .generate import ClientGenerateBatchProcess
from .kvcache.base_attention_cache import (
    BasePagedAttentionCache,
)
from .kvcache.trie_attention_cache import TriePagedAttentionCache
from .kvcache.page_pool import PagePoolConfig, PagePool
from .io_struct import GenerateReqInput
from .manager import LlmSystemManager
from .service_debug_dumper import SERVICE_DEBUG_DUMPER
from .tokenizer import Tokenizer
from .token_selection_strategy import get_strategy_from_str, is_ref_counted
from .token_selection_strategy import DecodeConfig

from ...utils import GenerateService
from multiprocessing import Queue

import multiprocessing
from multiprocessing import JoinableQueue

logger = logging.getLogger(__name__)


class LlmGenerateService(GenerateService):
    """Top level service interface for generating text against a model."""

    inference_program: sf.Program
    prefill_functions: dict[int, sf.ProgramFunction]
    decode_functions: dict[int, sf.ProgramFunction]

    def __init__(
        self,
        *,
        name: str,
        instance_num: int,  # Instance number within the service manager
        sysman: LlmSystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
        server_params: "ServerParams",
        program_isolation: str = "per_call",
        max_queue_size: int = 3,  # Maximum number of requests in queue
    ):
        super().__init__(sysman)
        self.name = name
        self.instance_num = instance_num
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.server_params = server_params
        self.max_queue_size = max_queue_size
        self.current_queue_size = 0

        self.set_isolation(program_isolation)
        self.initialize_worker_and_fiber()
        self.initialize_queues()
        self.initialize_page_cache()
    
    def initialize_queues(self):
        """Initialize request and response queues"""
        self.request_queue = self.sysman.ls.create_queue(f"{self.name}-request-queue")
        self.response_queue = self.sysman.ls.create_queue(f"{self.name}-response-queue")
        if self.model_params.decode_batch_sizes:
            self.max_queue_size = max(self.model_params.decode_batch_sizes) + 9
            print(f"Max queue size: {self.max_queue_size}")

    def add_to_queue(self) -> bool:
        """Try to add a request to the queue. Returns True if successful, False if queue is full."""
        if self.current_queue_size >= self.max_queue_size:
            return False
        self.current_queue_size += 1
        return True

    def remove_from_queue(self):
        """Remove a request from the queue."""
        if self.current_queue_size > 0:
            self.current_queue_size -= 1

    def initialize_queues(self):
        """Initialize request and response queues"""
        if self.model_params.decode_batch_sizes:
            self.max_queue_size = max(self.model_params.decode_batch_sizes) + 2
            print(f"Max queue size: {self.max_queue_size}")

    def add_to_queue(self) -> bool:
        """Try to add a request to the queue. Returns True if successful, False if queue is full."""
        # if self.current_queue_size >= self.max_queue_size:
        #     return False
        self.current_queue_size += 1
        return True

    def remove_from_queue(self):
        """Remove a request from the queue."""
        if self.current_queue_size > 0:
            self.current_queue_size -= 1

    def initialize_worker_and_fiber(self):
        num_workers = self.server_params.workers
        fibers_per_worker = self.server_params.fibers_per_worker

        logger.info(
            f"Creating {num_workers} workers, with {fibers_per_worker} fibers per worker..."
        )
        fibers = []
        for i in range(num_workers):
            worker = self.sysman.ls.create_worker(f"{self.name}-inference-{i}")
            for _ in range(fibers_per_worker):
                fiber = self.sysman.ls.create_fiber(worker)
                fibers.append(fiber)

        meta_fibers = []
        for fiber in fibers:
            device_buffer = LlmBufferObject(
                initialize_buffer_object(fibers[0].device(0), self.model_params)
            )
            meta_fiber = MetaFiber(device_buffer, fiber)
            meta_fibers.append(meta_fiber)

        self.fiber_pool = FiberPool(meta_fibers)

        self.devices = fibers[0].devices_dict.values()
        self.main_worker = self.sysman.ls.create_worker(f"{self.name}-inference")
        self.main_fiber = self.sysman.ls.create_fiber(self.main_worker)
        self.prefill_fiber = self.sysman.ls.create_fiber(self.main_worker)
        self.decode_fiber = self.sysman.ls.create_fiber(self.main_worker)

    def initialize_page_cache(self):
        """Initialize page pool and attention cache."""
        page_pool_config = PagePoolConfig(
            dtype=self.model_params.paged_kv_cache.kv_cache_dtype,
            alloc_page_count=self.model_params.paged_kv_cache.device_block_count,
            paged_kv_block_size_elements=self.model_params.paged_kv_block_size_elements,
        )
        page_pool = PagePool(devices=self.devices, config=page_pool_config)

        if self.server_params.prefix_sharing_algorithm == "trie":
            self.page_cache = TriePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=self.model_params.paged_kv_cache.block_seq_stride,
            )
        elif self.server_params.prefix_sharing_algorithm == "none":
            self.page_cache = BasePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=self.model_params.paged_kv_cache.block_seq_stride,
                use_ref_counts=is_ref_counted(
                    self.server_params.decode_config.token_selection_strategy
                ),
            )
        else:
            raise ValueError(
                f"Unknown prefix_sharing_algorithm {self.server_params.prefix_sharing_algorithm}. Currently only supporting 'trie' and 'none'."
            )

    def start(self):
        component_modules = self.initialize_program_modules("main")
        self.inference_program = self.create_program(
            modules=component_modules, devices=self.sysman.ls.devices
        )
        self.initialize_function_references()

        self.prefill_batcher = PrefillBatcherProcess(
            self.fiber_pool,
            self.page_cache,
            self.model_params,
            self.prefill_functions,
            self.prog_isolation,
        )

        self.decode_batcher = DecodeBatcherProcess(
            self.fiber_pool,
            self.page_cache,
            self.model_params,
            self.decode_functions,
            self.prog_isolation,
        )

        self.prefill_batcher.launch()
        self.decode_batcher.launch()

    def initialize_function_references(self):
        self.prefill_functions = {}
        for bs in self.model_params.prefill_batch_sizes:
            self.prefill_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.prefill_bs{bs}"
            ]
        # Resolve decode entrypoints.
        self.decode_functions = {}
        for bs in self.model_params.decode_batch_sizes:
            self.decode_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.decode_bs{bs}"
            ]

    def __repr__(self):
        return (
            f"ServiceManager(\n"
            f"  model_params={self.model_params}\n"
            f"  server_params={self.server_params}\n"
            f"  inference_modules={self.inference_modules}\n"
            f"  page_cache={self.page_cache}\n"
            f")"
        )


class LlmServiceEnvironment:
    """
    Environment  for LLM services, responsible for creating and managing
    instances of LlmGenerateService.

    This class, which also holds the shortfin system, is a singleton per
    process. For single-process usage, create one of these at the server
    (lifecycle) level. For multi-process usage, create one of these in each
    subprocess.
    """

    @staticmethod
    def get_server_params(args) -> ServerParams:
        server_params = ServerParams.load(
            args.server_config if hasattr(args, "server_config") else None
        )
        server_params.update_from_args(args)
        return server_params
    
    def __init__(self, args, num_services: int):
        # Load server configuration with priority:
        # command line > config file > defaults
        model_params = ModelParams.load_json(args.model_config)
        server_params = LlmServiceEnvironment.get_server_params(args)
        decode_bs = model_params.decode_batch_sizes[-1]
        if server_params.decode_config is None:
            decode_config = DecodeConfig(
                args.num_beams,
                args.token_selection_strategy,
                logits_normalization=model_params.logits_normalization,
                max_decode_batch_size=decode_bs,
            )
            server_params.decode_config = decode_config

        # Setup system (configure devices, etc).
        sysman = LlmSystemManager(
            device=args.device,
            device_ids=server_params.device_ids,
            async_allocs=server_params.amdgpu_async_allocations,
            amdgpu_allocators=server_params.amdgpu_allocators,
            amdgpu_allow_device_reuse=server_params.amdgpu_allow_device_reuse,
        )

        # Setup each service we are hosting.
        eos_token = LlmServiceProcess.get_eos_from_tokenizer_config(
            args.tokenizer_config_json)
        tokenizer = Tokenizer.from_tokenizer_json_file(
            args.tokenizer_json, eos_token=eos_token
        )
        print(f"Creating {server_params.instances} instances")
        # Create a list of LLM instances
        services: List[LlmGenerateService] = {}
        for i in range(server_params.instances):
            service = LlmGenerateService(
                name=f"instance{i}",
                instance_num=i,
                sysman=sysman,
                tokenizer=tokenizer,
                model_params=model_params,
                server_params=server_params,
                program_isolation=server_params.program_isolation,
            )
            service.load_inference_module(args.vmfb)
            service.load_inference_parameters(*args.parameters,
                                              parameter_scope="model")
            services[i] = service

        self.sysman = sysman
        self.services = services
        self.request_counter = 0
        self.current_instance = 0

    def shutdown(self):
        for service in self.services:
            service.shutdown()
        self.sysman.shutdown()

class LlmServiceProcess(multiprocessing.Process):
    @staticmethod
    def get_eos_from_tokenizer_config(json_path):
        import json

        with open(json_path, "rt") as f:
            json_text = f.read()
        config = json.loads(json_text)
        return config["eos_token"]

    def __init__(
        self, args,
        request_queue: JoinableQueue,
        response_queue: JoinableQueue
    ):
        super().__init__()
        self.request_queue: JoinableQueue = request_queue
        self.response_queue: JoinableQueue = response_queue
        self.service_environment = LlmServiceEnvironment(args)

    def run(self):
        logger.info(f"Starting LLM service process {self.name}")
        while True:
            try:
                request = self.request_queue.get(timeout=1)
            except multiprocessing.queues.Empty:
                continue

            if request is None:
                # Shutdown signal
                break

            try:
                if not self.service.add_to_queue():
                    logger.warning(
                        f"Queue full, dropping request: {request}"
                    )
                    continue

                response = self.handle_request(request)
                self.response_queue.put(response)
            except Exception as e:
                logger.exception(f"Error handling request: {e}")
            finally:
                self.service.remove_from_queue()
        self.shutdown()

    def handle_request(self, request):
        """Handle a single request."""
        if SERVICE_DEBUG_DUMPER.is_enabled():
            SERVICE_DEBUG_DUMPER.dump_request(request)

        # Process the request using the service's inference program.
        response = self.service.process_request(request)

        if SERVICE_DEBUG_DUMPER.is_enabled():
            SERVICE_DEBUG_DUMPER.dump_response(response)

        return response

    def shutdown(self):
        logger.info(f"Shutting down LLM service process {self.name}")
        self.service_environment.shutdown()


class LlmServiceManager:
    """
    Base class for managing LLM service instances.
    """

    def __init__(self, args):
        self.args = args
        self.cur_instance_num = -1

    async def send_request(self, gen_req: GenerateReqInput,
                           response_handler: callable):
        """
        """
        pass

    def shutdown(self):
        pass

    def get_next_instance_num(self) -> int:
        self.cur_instance_num += 1
        if self.cur_instance_num >= self.server_params.instances:
            self.cur_instance_num = 0
        return self.cur_instance_num


class LlmSingleProcesssServiceManager:
    """
    Manages all service instances in a single process.
    """

    def __init__(self, args):
        self.service_environment = LlmServiceEnvironment(args)

    async def send_request(self, gen_req: GenerateReqInput,
                           response_handler: callable):
        instance_num = self.get_next_instance_num()
        while not self.service_environment.services[instance_num].add_to_queue():
            instance_num = self.get_next_instance_num()
            await asyncio.sleep(0.001)
        service = self.service_environment.services[instance_num]

        batch_proc = ClientGenerateBatchProcess(service, gen_req,
                                                response_handler)
        batch_proc.launch()
        await batch_proc

    def shutdown(self):
        self.service_environment.shutdown()

class LlmMultiProcessServiceManager:
    """
    Manages each service instance in its own process.
    """

    def __init__(self, args):
        self.args = args
        self.server_params = LlmServiceEnvironment.get_server_params(args)
        max_queue_size = self.server_params.instances * 2

        self.request_queues: List[JoinableQueue] = [
            JoinableQueue(max_queue_size) for _ in
            range(self.server_params.instances)
        ]
        self.response_queues: List[JoinableQueue] = [
            JoinableQueue(max_queue_size) for _ in
            range(self.server_params.instances)
        ]

        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.server_params.instances
        )

        self.service_processes: List[LlmServiceProcess] = [
            LlmServiceProcess(
                args,
                request_queue=self.request_queues[i],
                response_queue=self.response_queues[i]
            )
            for i in range(self.server_params.instances)
        ]

        for process in self.service_processes:
            process.start()

    async def send_request(self, gen_req: GenerateReqInput,
                           response_handler: callable):
        instance_num = self.get_next_instance_num()
        while self.request_queues[instance_num].full():
            instance_num = self.get_next_instance_num()
            await asyncio.sleep(0.001)

        request_queue = self.request_queues[instance_num]
        response_queue = self.response_queues[instance_num]

        def enqueue_request():
            try:
                request_queue.put(gen_req)
                return response_queue.get()
            except Exception as e:
                logger.exception(f"Error enqueueing request: {e}")
                raise e

        response = await self.executor.submit(enqueue_request)
        response_handler(response)

    def shutdown(self):
        self.executor.shutdown(wait=True)
        for respnse_queue in self.response_queues:
            respnse_queue.join()
        for request_queue in self.request_queues:
            request_queue.put(None)
        for process in self.service_processes:
            process.join()
