# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import multiprocessing

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import JoinableQueue
import os
from typing import List, Tuple

from .config_struct import ModelParams, ServerParams
from .token_selection_strategy import DecodeConfig
from .generate import ClientGenerateBatchProcess
from .io_struct import GenerateReqInput
from .manager import LlmSystemManager
from .service import LlmGenerateService
from .service_debug_dumper import SERVICE_DEBUG_DUMPER
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


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

    def __init__(self, args, create_multiple_services: bool):
        # Load server configuration with priority:
        # command line > config file > defaults
        model_params = ModelParams.load_json(args.model_config)
        server_params = LlmServiceEnvironment.get_server_params(args)
        self.num_instances = server_params.instances if create_multiple_services else 1
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
            args.tokenizer_config_json
        )
        tokenizer = Tokenizer.from_tokenizer_json_file(
            args.tokenizer_json, eos_token=eos_token
        )
        print(f"Creating {self.num_instances} instances")
        # Create a list of LLM instances
        services: List[LlmGenerateService] = {}
        for i in range(self.num_instances):
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
            service.load_inference_parameters(*args.parameters, parameter_scope="model")
            services[i] = service

        self.sysman = sysman
        self.services = services

    def start(self):
        self.sysman.start()
        for service_name, service in self.services.items():
            print(f"Initializing service '{service_name}'")
            service.start()

    def shutdown(self):
        for service_name, service in self.services.items():
            print(f"Shutting down service '{service_name}'")
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
        self, args, request_queue: JoinableQueue, response_queue: JoinableQueue
    ):
        super().__init__()
        self.args = args
        self.request_queue: JoinableQueue = request_queue
        self.response_queue: JoinableQueue = response_queue
        self.service_environment = None
        self.service = None  # Single instance in this process

    def run(self):
        print(f"Starting LLM service process {self.name}")
        self.service_environment = LlmServiceEnvironment(
            self.args, create_multiple_services=False)
        self.service = self.service_environment.services[0]  # Single instance in this process
        self.service_environment.start()

        while True:
            print(f"LlmServiceProcess {self.name} waiting for request")
            request_packet: Tuple[int, GenerateReqInput] = self.request_queue.get()
            request_counter = request_packet[0]
            request = request_packet[1]
            print(f"LlmServiceProcess {self.name} received request packet: {request_counter}")

            if request is None:
                # Shutdown signal
                break

            if not self.service.add_to_queue():
                logger.warning(f"Queue full, dropping request {request_counter}: {request}")
                continue

            def response_handler(response):
                self.response_queue.put((request_counter, response))
                print(f"LlmServiceProcess {self.name} response for request {request_counter} enqueued")

            async def generate_wrapper():
                print(f"LlmServiceProcess {self.name} creating batch process for request {request_counter}")
                batch_proc = ClientGenerateBatchProcess(
                    self.service, request, response_handler)
                print(f"LlmServiceProcess {self.name} launching batch process for request {request_counter}")
                await asyncio.gather(batch_proc.launch())
                print(f"LlmServiceProcess {self.name} batch process completed for request {request_counter}")
                self.service.remove_from_queue()

            print(f"LlmServiceProcess {self.name} launching generate_wrapper for request {request_counter}")
            asyncio.run_coroutine_threadsafe(
                generate_wrapper(), loop=self.service.main_worker.loop)
            # Don't wait for it to finish! response_handler should take care of
            # notifying when the response is ready. Waiting here increases latency

        print(f"Shutting down LLM service process {self.name}")
        self.service_environment.shutdown()


class LlmServiceManager:
    """
    Base class for managing LLM service instances.
    """

    def __init__(self, args):
        self.args = args
        self.cur_instance_num = -1
        self.request_counter = 0
        self.num_instances = 0

    def start(self):
        """
        Start all resources contained in this service manager.
        """
        pass

    async def send_request(self, gen_req: GenerateReqInput, response_handler: callable):
        """
        Send a request to an LLM service instance. The response is sent
        to the given callable and takes the form of a single byte array or a
        list of byte arrays.
        """
        pass

    def shutdown(self):
        """
        Shutdown all resources contained in this service manager.
        """
        pass

    def get_next_instance_num(self) -> int:
        """
        Get the next service instance number in a round-robin fashion.
        """
        self.cur_instance_num += 1
        if self.cur_instance_num >= self.num_instances:
            self.cur_instance_num = 0
        return self.cur_instance_num


class LlmSingleProcessServiceManager(LlmServiceManager):
    """
    Manages all service instances in a single process.
    """

    def __init__(self, args):
        super().__init__(args)
        self.service_environment = LlmServiceEnvironment(args, create_multiple_services=True)
        self.num_instances = self.service_environment.num_instances

    def start(self):
        self.service_environment.start()

    async def send_request(self, gen_req: GenerateReqInput, response_handler: callable):
        self.request_counter += 1
        instance_num = self.get_next_instance_num()
        while not self.service_environment.services[instance_num].add_to_queue():
            instance_num = self.get_next_instance_num()
            await asyncio.sleep(0.001)

        service = self.service_environment.services[instance_num]
        print(
            f"Generating in-process with service: {instance_num}"
            f" current queue size: {service.current_queue_size}"
            f" max queue size: {service.max_queue_size}"
        )
        service = self.service_environment.services[instance_num]

        # Bridge from the asyncio event loop to the service's worker

        async def generate_wrapper():
            batch_proc = ClientGenerateBatchProcess(
                service, gen_req, response_handler)
            await asyncio.gather(batch_proc.launch())
            service.remove_from_queue()

        asyncio.run_coroutine_threadsafe(
            generate_wrapper(), loop=service.main_worker.loop)
        # Don't wait for it to finish! response_handler should take care of
        # notifying when the response is ready. Waiting here increases latency

    def shutdown(self):
        self.service_environment.shutdown()


class LlmMultiProcessServiceManager(LlmServiceManager):
    """
    Manages each service instance in its own process.
    """

    def __init__(self, args):
        super().__init__(args)
        self.server_params = LlmServiceEnvironment.get_server_params(args)
        self.num_instances = self.server_params.instances
        max_queue_size = self.server_params.instances * 2

        self.request_queues: List[JoinableQueue] = [
            JoinableQueue(max_queue_size) for _ in range(self.num_instances)
        ]
        self.response_queues: List[JoinableQueue] = [
            JoinableQueue(max_queue_size) for _ in range(self.num_instances)
        ]

        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.num_instances
        )

        self.service_processes: List[LlmServiceProcess] = [
            LlmServiceProcess(
                args,
                request_queue=self.request_queues[i],
                response_queue=self.response_queues[i],
            )
            for i in range(self.num_instances)
        ]

    def start(self):
        for process in self.service_processes:
            process.start()

    async def send_request(self, gen_req: GenerateReqInput, response_handler: callable):
        self.request_counter += 1
        instance_num = self.get_next_instance_num()
        while self.request_queues[instance_num].full():
            instance_num = self.get_next_instance_num()
            await asyncio.sleep(0.001)

        print(f"Generating multi-process with service: {instance_num}")
        request_queue = self.request_queues[instance_num]
        response_queue = self.response_queues[instance_num]

        def enqueue_request():
            request_counter = self.request_counter
            print(f"send_request: enqueing {request_counter} to instance {instance_num}")
            request_queue.put((request_counter, gen_req))
            print(f"send_request: waiting for response {request_counter} from instance {instance_num}")
            response_packet: Tuple[int, bytes] = response_queue.get()
            response_counter = response_packet[0]
            if response_counter != request_counter:
                logger.error(
                    f"Response counter mismatch: expected {request_counter}, "
                    f"got {response_packet[0]}"
                )
            print(f"send_request: received response {response_counter} from instance {instance_num}")
            return response_packet[1]  # response payload

        loop = asyncio.get_running_loop()
        print(f"send_request: running in executor for instance {instance_num}")
        future = loop.run_in_executor(self.executor, enqueue_request)
        print(f"send_request: waiting for future response from instance {instance_num}")
        response = await future
        print(f"send_request: received future response from instance {instance_num}")
        response_handler(response)
        print(f"send_request: response handler called for instance {instance_num}")

    def shutdown(self):
        self.executor.shutdown(wait=True)
        for respnse_queue in self.response_queues:
            respnse_queue.join()
        for request_queue in self.request_queues:
            request_queue.put((0, None))
        for process in self.service_processes:
            process.join()
