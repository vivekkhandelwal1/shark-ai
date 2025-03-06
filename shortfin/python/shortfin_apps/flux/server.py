# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any
import argparse
import logging
from pathlib import Path
import sys
import os
import copy
import subprocess
from contextlib import asynccontextmanager
import uvicorn

# Import first as it does dep checking and reporting.
from shortfin.interop.fastapi import FastAPIResponder
from shortfin.support.logging_setup import native_handler

from fastapi import FastAPI, Request, Response

from .components.generate import ClientGenerateBatchProcess
from .components.config_struct import ModelParams
from .components.io_struct import GenerateReqInput
from .components.manager import SystemManager
from .components.service import GenerateService, SequentialGenerateService
from .components.tokenizer import Tokenizer


logger = logging.getLogger("shortfin-flux")
logger.addHandler(native_handler)
logger.propagate = False

THIS_DIR = Path(__file__).resolve().parent

UVICORN_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "format": "[{asctime}] {message}",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "style": "{",
            "use_colors": True,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    sysman.start()
    try:
        for service_name, service in services.items():
            logger.info("Initializing service '%s':", service_name)
            logger.info(str(service))
            service.start()
    except:
        sysman.shutdown()
        raise
    yield
    try:
        for service_name, service in services.items():
            logger.info("Shutting down service '%s'", service_name)
            service.shutdown()
    finally:
        sysman.shutdown()


sysman: SystemManager
services: dict[str, Any] = {}
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


async def generate_request(gen_req: GenerateReqInput, request: Request):
    service = services["sd"]
    gen_req.post_init()
    responder = FastAPIResponder(request)
    ClientGenerateBatchProcess(service, gen_req, responder).launch()
    return await responder.response


app.post("/generate")(generate_request)
app.put("/generate")(generate_request)


def configure_sys(args) -> SystemManager:
    # Setup system (configure devices, etc).
    model_config, topology_config, flagfile, tuning_spec, args = get_configs(args)
    sysman = SystemManager(args.device, args.device_ids, args.amdgpu_async_allocations)
    return sysman, model_config, flagfile, tuning_spec


def configure_service(args, sysman, model_config, flagfile, tuning_spec):
    # Setup each service we are hosting.
    clip_tokenizers = [
        Tokenizer.from_pretrained(args.tokenizer_source, subfolder="tokenizer")
    ]
    t5xxl_tokenizers = [
        Tokenizer.from_pretrained(args.tokenizer_source, subfolder="tokenizer_2")
    ]

    model_params = ModelParams.load_json(model_config)
    try:
        vmfbs, params = get_modules(args, model_config, flagfile, tuning_spec)
    except Exception as e:
        logger.error(f"Failed to retrieve required modules: {e}")
        sys.exit(1)

    if args.sequential_mode:
        logger.info("Using sequential model loading for lower memory usage")
        sm = SequentialGenerateService(
            name="sd",
            sysman=sysman,
            clip_tokenizers=clip_tokenizers,
            t5xxl_tokenizers=t5xxl_tokenizers,
            model_params=model_params,
            fibers_per_device=args.fibers_per_device,
            workers_per_device=args.workers_per_device,
            prog_isolation=args.isolation,
            show_progress=args.show_progress,
            trace_execution=args.trace_execution,
            split_denoise=args.split_denoise,
        )
    else:
        logger.info("Using standard model loading")
        sm = GenerateService(
            name="sd",
            sysman=sysman,
            clip_tokenizers=clip_tokenizers,
            t5xxl_tokenizers=t5xxl_tokenizers,
            model_params=model_params,
            fibers_per_device=args.fibers_per_device,
            workers_per_device=args.workers_per_device,
            prog_isolation=args.isolation,
            show_progress=args.show_progress,
            trace_execution=args.trace_execution,
        )
    
    for key, vmfblist in vmfbs.items():
        for vmfb in vmfblist:
            sm.load_inference_module(vmfb, component=key)
    for key, datasets in params.items():
        sm.load_inference_parameters(*datasets, parameter_scope="model", component=key)
    services[sm.name] = sm
    return sysman


def get_configs(args):
    # Returns one set of config artifacts.
    modelname = "flux"
    model_config = args.model_config if args.model_config else None
    topology_config = None
    tuning_spec = None
    flagfile = args.flagfile if args.flagfile else None
    cfg_builder_args = [
        sys.executable,
        "-m",
        "iree.build",
        os.path.join(THIS_DIR, "components", "config_artifacts.py"),
        f"--target={args.target}",
        f"--output-dir={args.artifacts_dir}",
        f"--model={modelname}",
    ]
    if args.topology:
        cfg_builder_args.extend(
            [
                f"--topology={args.topology}",
            ]
        )
    outs = subprocess.check_output(cfg_builder_args).decode()
    outs_paths = outs.splitlines()
    for i in outs_paths:
        if "flux_config" in i and not args.model_config:
            model_config = i
        elif "topology" in i and args.topology:
            topology_config = i
        elif "flagfile" in i and not args.flagfile:
            flagfile = i
        elif "attention_and_matmul_spec" in i and args.use_tuned:
            tuning_spec = i

    if args.use_tuned and args.tuning_spec:
        tuning_spec = os.path.abspath(args.tuning_spec)

    if topology_config:
        with open(topology_config, "r") as f:
            contents = [line.rstrip() for line in f]
        for spec in contents:
            if "--" in spec:
                arglist = spec.strip("--").split("=")
                arg = arglist[0]
                if len(arglist) > 2:
                    value = arglist[1:]
                    for val in value:
                        try:
                            val = int(val)
                        except ValueError:
                            val = val
                elif len(arglist) == 2:
                    value = arglist[-1]
                    try:
                        value = int(value)
                    except ValueError:
                        value = value
                else:
                    # It's a boolean arg.
                    value = True
                setattr(args, arg, value)
            else:
                # It's an env var.
                arglist = spec.split("=")
                os.environ[arglist[0]] = arglist[1]
    return model_config, topology_config, flagfile, tuning_spec, args


def get_modules(args, model_config, flagfile, td_spec):
    # TODO: Move this out of server entrypoint
    vmfbs = {"clip": [], "t5xxl": [], "vae": []}
    params = {"clip": [], "t5xxl": [], "vae": []}
    
    # Add support for split sampler model if needed
    if args.split_denoise:
        vmfbs["sampler_front"] = []
        vmfbs["sampler_back"] = []
        params["sampler_front"] = []
        params["sampler_back"] = []
    else:
        vmfbs["sampler"] = []
        params["sampler"] = []
    
    model_flags = copy.deepcopy(vmfbs)
    # Add sampler to model_flags if we're in split_denoise mode to collect flags for it
    if args.split_denoise and "sampler" not in model_flags:
        model_flags["sampler"] = []
    
    model_flags["all"] = args.compile_flags

    if flagfile:
        with open(flagfile, "r") as f:
            contents = [line.rstrip() for line in f]
        flagged_model = "all"
        for elem in contents:
            match = [keyw in elem for keyw in model_flags.keys()]
            if any(match):
                flagged_model = elem
            else:
                model_flags[flagged_model].extend([elem])
    if td_spec:
        # Only add TD spec to sampler if we have it in model_flags
        if "sampler" in model_flags:
            model_flags["sampler"].extend(
                [f"--iree-codegen-transform-dialect-library={td_spec}"]
            )
        # In split mode, copy sampler flags to front/back models
        if args.split_denoise:
            model_flags["sampler_front"] = model_flags["sampler"].copy()
            model_flags["sampler_back"] = model_flags["sampler"].copy()

    filenames = []
    logger.info(f"Processing models: {list(vmfbs.keys())}")
    for modelname in vmfbs.keys():
        ireec_args = model_flags["all"] + model_flags[modelname]
        ireec_extra_args = " ".join(ireec_args)
        builder_args = [
            sys.executable,
            "-m",
            "iree.build",
            os.path.join(THIS_DIR, "components", "builders.py"),
            f"--model-json={model_config}",
            f"--target={args.target}",
            f"--splat={args.splat}",
            f"--build-preference={args.build_preference}",
            f"--output-dir={args.artifacts_dir}",
            f"--model={modelname}",
            f"--iree-hal-target-device={args.device}",
            f"--iree-hip-target={args.target}",
            f"--iree-compile-extra-args={ireec_extra_args}",
        ]
        logger.info(f"Preparing runtime artifacts for {modelname}...")
        if not args.build_preference == "precompiled":
            logger.info(
                "COMMAND LINE EQUIVALENT: " + " ".join([str(argn) for argn in builder_args])
            )
        
        try:
            # Run the subprocess and capture both stdout and stderr
            process = subprocess.Popen(
                builder_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            # Collect all stdout and stderr output
            all_stdout = []
            all_stderr = []
            
            # Read output line by line as it becomes available
            for line in process.stdout:
                line = line.rstrip()
                all_stdout.append(line)
                # Echo to our own stdout for real-time feedback
                print(f"BUILDER [{modelname}]: {line}")
                sys.stdout.flush()
            
            # Get the return code
            process.wait()
            
            # Read any stderr after process completes
            for line in process.stderr:
                line = line.rstrip()
                all_stderr.append(line)
                print(f"BUILDER [{modelname}] - ERROR: {line}")
                sys.stderr.flush()
            
            # Check return code
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, builder_args, 
                                                   output="\n".join(all_stdout), 
                                                   stderr="\n".join(all_stderr))
            
            # Process the output to get file paths
            output_paths = [line for line in all_stdout if os.path.exists(line)]
            filenames.extend(output_paths)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to fetch artifacts for {modelname}"
            
            # Include both stdout and stderr in the error message
            if hasattr(e, 'output') and e.output:
                error_msg += f"\nProcess output: {e.output}"
            
            if hasattr(e, 'stderr') and e.stderr:
                error_msg += f"\nProcess stderr: {e.stderr}"
            
            # For split model, add more helpful message
            if hasattr(args, 'split_denoise') and args.split_denoise and modelname in ["sampler_front", "sampler_back"]:
                error_msg += "\nSplit model files are required but couldn't be found."
                error_msg += "\nMake sure to export the split model using export_components.py with --split_model flag."
                
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    print(filenames)
    for name in filenames:
        # Standard file handling
        for key in vmfbs.keys():
            if key == "t5xxl" and all(x in name.lower() for x in ["xxl", "irpa"]):
                params[key].extend([name])
            if key in name.lower():
                if any(x in name for x in [".irpa", ".safetensors", ".gguf"]):
                    params[key].extend([name])
                elif "vmfb" in name:
                    vmfbs[key].extend([name])
    
    # When using split_denoise, verify that we have the required files
    if hasattr(args, 'split_denoise') and args.split_denoise:
        if not vmfbs["sampler_front"] or not vmfbs["sampler_back"]:
            error_msg = "Split denoise mode requires both sampler_front and sampler_back model files."
            error_msg += "\nMake sure to export the split model using export_components.py with --split_model flag."
            raise RuntimeError(error_msg)
            
    return vmfbs, params


def main(argv, log_config=UVICORN_LOG_CONFIG):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--timeout-keep-alive", type=int, default=5, help="Keep alive timeout"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["local-task", "hip", "amdgpu"],
        help="Primary inferencing device",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=False,
        default="gfx942",
        choices=["gfx942", "gfx1100", "gfx90a", "gfx1201"],
        help="Primary inferencing device LLVM target arch.",
    )
    parser.add_argument(
        "--device_ids",
        type=str,
        nargs="*",
        default=None,
        help="Device IDs visible to the system builder. Defaults to None (full visibility). Can be an index or a sf device id like amdgpu:0:0@0",
    )
    parser.add_argument(
        "--tokenizer_source",
        type=Path,
        default="black-forest-labs/FLUX.1-dev",
        help="HF repo from which to load tokenizer(s).",
    )
    parser.add_argument(
        "--model_config", type=Path, help="Path to the model config file."
    )
    parser.add_argument(
        "--workers_per_device",
        type=int,
        default=1,
        help="Concurrency control -- how many fibers are created per device to run inference.",
    )
    parser.add_argument(
        "--fibers_per_device",
        type=int,
        default=1,
        help="Concurrency control -- how many fibers are created per device to run inference.",
    )
    parser.add_argument(
        "--isolation",
        type=str,
        default="per_call",
        choices=["per_fiber", "per_call", "none"],
        help="Concurrency control -- How to isolate programs.",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="enable tqdm progress for sampler iterations.",
    )
    parser.add_argument(
        "--trace_execution",
        action="store_true",
        help="Enable tracing of program modules.",
    )
    parser.add_argument(
        "--amdgpu_async_allocations",
        action="store_true",
        help="Enable asynchronous allocations for amdgpu device contexts.",
    )
    parser.add_argument(
        "--splat",
        action="store_true",
        help="Use splat (empty) parameter files, usually for testing.",
    )
    parser.add_argument(
        "--build_preference",
        type=str,
        choices=["compile", "precompiled"],
        default="precompiled",
        help="Specify preference for builder artifact generation.",
    )
    parser.add_argument(
        "--compile_flags",
        type=str,
        nargs="*",
        default=[],
        help="extra compile flags for all compile actions. For fine-grained control, use flagfiles.",
    )
    parser.add_argument(
        "--flagfile",
        type=Path,
        help="Path to a flagfile to use for SDXL. If not specified, will use latest flagfile from azure.",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=Path,
        default=None,
        help="Path to local artifacts cache.",
    )
    parser.add_argument(
        "--tuning_spec",
        type=str,
        default=None,
        help="Path to transform dialect spec if compiling an executable with tunings.",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        choices=["spx_single", "cpx_single", "spx_multi", "cpx_multi"],
        help="Use one of four known performant preconfigured device/fiber topologies.",
    )
    parser.add_argument(
        "--use_tuned",
        type=int,
        default=1,
        help="Use tunings for attention and matmul ops. 0 to disable.",
    )
    parser.add_argument(
        "--sequential_mode",
        action="store_true",
        help="Use sequential loading of models for lower memory usage.",
    )
    parser.add_argument(
        "--split_denoise",
        action="store_true",
        help="Use split front/back model for denoising with sequential loading.",
    )
    args = parser.parse_args(argv)
    if not args.artifacts_dir:
        home = Path.home()
        artdir = home / ".cache" / "shark"
        args.artifacts_dir = str(artdir)
    else:
        args.artifacts_dir = Path(args.artifacts_dir).resolve()

    global sysman
    sysman, model_config, flagfile, tuning_spec = configure_sys(args)
    configure_service(args, sysman, model_config, flagfile, tuning_spec)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=log_config,
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)
    main(
        sys.argv[1:],
        # Make logging defer to the default shortfin logging config.
        log_config=UVICORN_LOG_CONFIG,
    )
