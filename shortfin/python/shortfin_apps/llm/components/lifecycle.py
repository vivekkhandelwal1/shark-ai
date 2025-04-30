# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Implements a context manager that configures a shortfin llm server from a namespace mirroring server.py's commandline args, and exposes a context manager interface such that we can do:

```python
def lifecycle(app: FastApi):
    with lifecycle_manager(args) as man:
        yield
```
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from .config_struct import ModelParams, ServerParams
from .io_struct import GenerateReqInput
from .token_selection_strategy import DecodeConfig
from .manager import LlmSystemManager
from .service import LlmGenerateService, LlmServiceProcess, LlmServiceManager
from .tokenizer import Tokenizer
from multiprocessing import JoinableQueue
from typing import TYPE_CHECKING, List
from fastapi import FastAPI


from contextlib import asynccontextmanager
import logging


def get_eos_from_tokenizer_config(json_path):
    import json

    with open(json_path, "rt") as f:
        json_text = f.read()
    config = json.loads(json_text)
    return config["eos_token"]


class ShortfinLlmLifecycleManager:
    """
    Manages the lifecycle of a shortfin llm server, including config loading and parameter setup.

    There are generally two ways to use this.

    To start a full shortfin server, use the context manager or the fastapi_lifespan method.

    To initialize a shortfin server but not start it, use the constructor, then manipulate the services and sysman attributes directly.
    """

    def __init__(self, args):
        self.service_manager = LlmServiceManager(args)

        self.sysman = sysman
        self.services = services
        self.processes = processes
        self.request_counter = 0
        self.current_instance = 0

        self.requent_queues: List[JoinableQueue] = [
            JoinableQueue() for _ in range(args.instances)
        ]
        self.response_queues: List[JoinableQueue] = [
            JoinableQueue() for _ in range(args.instances)
        ]

        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=args.instances
        )

    def __enter__(self):
        self.sysman.start()
        for service_name, service in self.service_manager.services.items():
            logging.info("Initializing service '%s': %r", service_name, service)
            service.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for service_name, service in self.service_manager.services.items():
            logging.info("Shutting down service '%s'", service_name)
            service.shutdown()
        self.service_manager.sysman.shutdown()
        return False

    @asynccontextmanager
    async def fastapi_lifespan(self, app: FastAPI):
        """
        Context manager for FastAPI lifespan events.

        Initializes the system manager and services when the app starts, and shuts them down when the app stops.
        Also provides the services via app.state, which can be accessed from route handlers via
        request.app.state.services.

        Implements API described in https://fastapi.tiangolo.com/advanced/events/#lifespan

        See `server.py` for a usage example.
        """
        with self:
            app.state.services = self.service_manager.services
            app.state.request_counter = self.service_manager.request_counter
            app.state.current_instance = self.service_manager.current_instance
            app.state.send_request = self.send_request
            app.state.get_response = self.get_response
            yield

    def get_current_instance(self) -> int:
        return self.service_manager.current_instance
    
    def increment_instance(self):
        self.service_manager.current_instance += 1

    def get_current_service(self) -> LlmGenerateService:
        return self.service_manager.services[self.service_manager.current_instance]
    
    def send_request(self, service: LlmGenerateService, gen_req: GenerateReqInput):
        instance_num: int = service.instance_num
        self.request_queues[instance_num].put(gen_req)
        print(f"Request queued: {gen_req.text}")

    async def get_response(self, service: LlmGenerateService):
        instance_num: int = service.instance_num
        response = await asyncio.get_running_loop().run_in_executor(
            self.executor, self.request_queues[instance_num].get()
        )
        self.response_queues[instance_num].task_done()
        return response
