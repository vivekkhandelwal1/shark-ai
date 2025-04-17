import argparse
import logging
import multiprocessing
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import Response
import httpx

from .server import (
    parse_args,
    get_app,
    run_server,
    ShortfinLlmLifecycleManager,
    UVICORN_LOG_CONFIG,
)

logger = logging.getLogger(__name__)


class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0

    def get_next_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server


def create_load_balancer_app(servers):
    app = FastAPI()

    @app.middleware("http")
    async def load_balance(request: Request, call_next):
        server = servers.get_next_server()
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=request.method,
                url=f"{server}{request.url.path}",
                headers=dict(request.headers),
                content=await request.body(),
                params=request.query_params,
                timeout=1000
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type"),
            )

    return app


def main(argv):
    args = parse_args(argv)

    # Define the ports for each server instance
    ports = [8000, 8001]
    servers = LoadBalancer([f"http://{args.host}:{port}" for port in ports])

    # Create processes for each server
    processes = []
    for port in ports:
        process = multiprocessing.Process(
            target=run_server, args=(argv, UVICORN_LOG_CONFIG, port)
        )
        processes.append(process)
        process.start()

    # Create and run the load balancer
    load_balancer_app = create_load_balancer_app(servers)
    uvicorn.run(
        load_balancer_app,
        host=args.host,
        port=8080,  # Load balancer port
        log_config=UVICORN_LOG_CONFIG,
    )


if __name__ == "__main__":
    from shortfin.support.logging_setup import configure_main_logger

    logger = configure_main_logger("loadbalanced_server")
    main(sys.argv[1:])
