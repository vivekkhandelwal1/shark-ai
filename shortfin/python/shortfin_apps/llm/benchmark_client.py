import asyncio
import time
import requests
import uuid
from typing import Dict, Any, List, Tuple
import numpy as np
import csv
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import argparse



class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8080", max_workers: int = 128):
        self.base_url = base_url.rstrip("/")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def generate(
        self,
        text: str,
        sampling_params: Dict[str, Any] = None,
        save_output: bool = False,
    ) -> Dict[str, Any]:
        """Send a generation request to the LLM server, and return the results."""
        data = {
            "text": text,
            "sampling_params": sampling_params or {},
            # "rid": uuid.uuid4().hex,
            "return_logprob": False,
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}

        results = {}

        def process_stream():
            start_time = time.perf_counter()
            token_times = []
            generated_text = []
            try:
                print(f"Sending request to {self.base_url}/generate")
                with requests.post(
                    f"{self.base_url}/generate",
                    headers=headers,
                    json=data,
                    stream=False,
                    timeout=1000
                    # timeout=60,  # Add timeout to prevent hanging
                ) as response:
                    response.raise_for_status()

                    # Process the response as it arrives
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode("utf-8")
                            if line_text.startswith("data"):
                                token_time = time.perf_counter()
                                token_times.append(token_time)
                                if save_output:
                                    generated_text.append(line_text.split(": ", 1)[1])
                print(f"Received response from {self.base_url}/generate")
            except Exception as e:
                print(f"Error in process_stream: {e}")
                # Return empty results in case of error
                return start_time, [], []

            return start_time, token_times, generated_text

        # Run the processing function in the thread pool with our dedicated executor
        start_time, token_times, generated_text = (
            await asyncio.get_event_loop().run_in_executor(
                self.executor, process_stream
            )
        )

        # Handle case where no tokens were generated
        if not token_times:
            results["metrics"] = {
                "start_time": start_time,
                "end_time": start_time,
                "time_to_first_token": 0,
                "token_generation_times": [],
                "num_tokens": 0,
                "generated_text": "",
                "error": "No tokens generated",
            }
            return results

        time_to_first_token = token_times[0] - start_time if token_times else 0
        num_tokens = len(token_times)

        results["metrics"] = {
            "start_time": start_time,
            "end_time": end_time,
            "num_tokens": num_tokens,
            "generated_text": "".join(generated_text) if save_output else "",
            "is_streaming": self.stream,
        }

        return results

    def __del__(self):
        # Ensure executor is shut down when client is destroyed
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)


def calculate_metrics(
    results: List[Dict[str, Any]], start_time: float
) -> Dict[str, Any]:
    """Calculate benchmark metrics from results."""
    num_concurrent_requests = len(results)

    # Extract raw metrics
    token_generation_times = [
        result["metrics"]["token_generation_times"] for result in results
    ]
    time_to_first_token = [
        result["metrics"]["time_to_first_token"] for result in results
    ]
    start_times = [result["metrics"]["start_time"] - start_time for result in results]
    end_times = [result["metrics"]["end_time"] - start_time for result in results]
    time_per_request = [
        end_times[i] - start_times[i] for i in range(num_concurrent_requests)
    ]
    num_generated_tokens = [result["metrics"]["num_tokens"] for result in results]

    # Calculate token-level metrics
    flattened_token_times = sorted(
        [item for sublist in token_generation_times for item in sublist]
    )
    TPS_times = [
        flattened_token_times[i] - flattened_token_times[i - 1]
        for i in range(1, len(flattened_token_times))
    ]
    TPOT_times = [
        [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
        for token_times in token_generation_times
    ]
    TPOT_times = [item for sublist in TPOT_times for item in sublist]

    return {
        "time_to_first_token": time_to_first_token,
        "TPOT_times": TPOT_times,
        "TPS_times": TPS_times,
        "start_times": start_times,
        "end_times": end_times,
        "time_per_request": time_per_request,
        "num_generated_tokens": num_generated_tokens,
    }


def print_benchmark_results(
    metrics: Dict[str, Any], config: Dict[str, Any], total_time: float
):
    """Print benchmark results in a formatted way."""
    num_concurrent_requests = len(metrics["time_to_first_token"])

    print("\nBenchmark Configuration:")
    print(f"Input token length: {config['input_token_length']}")
    print(f"Output token length: {config['output_token_length']}")
    print(f"Number of concurrent requests: {num_concurrent_requests}")
    print(f"Token selection strategy: {config['token_selection_strategy']}")

    print("\nPerformance Metrics:")
    print(f"E2E latency: {total_time:.2f} seconds")
    print(f"Requests per second: {num_concurrent_requests/total_time:.2f}")
    print(f"Average latency: {total_time/num_concurrent_requests:.2f} seconds")

    print("\nDetailed Metrics:")
    metric_names = {
        "time_to_first_token": "Time to first token",
        "TPOT_times": "Time per output token",
        "TPS_times": "Tokens per second",
        "start_times": "Request processing start time",
        "end_times": "Request processing end time",
        "time_per_request": "Time per request",
        "num_generated_tokens": "Number of generated tokens",
    }

    for metric_key, metric_name in metric_names.items():
        values = metrics[metric_key]
        if values:
            print(
                f"{metric_name}: Mean: {np.mean(values):.4f}s, SD: {np.std(values):.4f}s, "
                f"Median: {np.median(values):.4f}s, Min: {np.min(values):.4f}s, "
                f"Max: {np.max(values):.4f}s"
            )


def get_csv_results(
    metrics: Dict[str, Any], config: Dict[str, Any], total_time: float
) -> Dict[str, Any]:
    """Get results in a format suitable for CSV export."""
    num_concurrent_requests = len(metrics["time_to_first_token"])

    return {
        "token_selection_strategy": config["token_selection_strategy"],
        "input_token_length": config["input_token_length"],
        "output_token_length": config["output_token_length"],
        "num_concurrent_requests": num_concurrent_requests,
        "total_time": total_time,
        "requests_per_second": num_concurrent_requests / total_time,
        "avg_latency": total_time / num_concurrent_requests,
        "TTFT_median": np.median(metrics["time_to_first_token"]),
        "TPOT_median": np.median(metrics["TPOT_times"]),
        "TPS_median": np.median(metrics["TPS_times"]),
        "Time_per_request_median": np.median(metrics["time_per_request"]),
    }


async def run_benchmark(
    input_token_length: int = 100,
    output_token_length: int = 50,
    num_concurrent_requests: int = 64,
    token_selection_strategy: str = "multi_greedy",
):
    client = LLMClient()

    prompt = " ".join(["one" for _ in range(input_token_length)])
    config = {
        "input_token_length": input_token_length,
        "output_token_length": output_token_length,
        "token_selection_strategy": token_selection_strategy,
    }

    # Create tasks
    tasks = []
    # Run benchmark
    start_time = time.perf_counter()
    for _ in range(num_concurrent_requests):
        tasks.append(
            client.generate(
                text=prompt,
                sampling_params={
                    "max_completion_tokens": output_token_length,
                    "token_selection_strategy": token_selection_strategy,
                    "num_beams": 8,
                },
                save_output=False,
            )
        )

    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Calculate and print metrics
    metrics = calculate_metrics(results, start_time)
    print_benchmark_results(metrics, config, total_time)

    # Print sample generated text
    if (
        results
        and "metrics" in results[0]
        and "generated_text" in results[0]["metrics"]
    ):
        print(f"\nSample Generated Text:\n{results[0]['metrics']['generated_text']}")

    # Return results for CSV
    return get_csv_results(metrics, config, total_time)


async def run_all_benchmarks(
    input_token_lengths: List[int] = [1024],
    output_token_lengths: List[int] = [64],
    num_concurrent_requests: List[int] = [100],
    token_selection_strategy: str = "greedy",
):
    all_results = []

    for request_count in num_concurrent_requests:
        for input_token_length in input_token_lengths:
            for output_token_length in output_token_lengths:
                print(
                    f"\n\nRunning benchmark with max_completion_tokens = {output_token_length}"
                )
                result = await run_benchmark(
                    input_token_length=input_token_length,
                    output_token_length=output_token_length,
                    num_concurrent_requests=request_count,
                    token_selection_strategy=token_selection_strategy,
                )
                all_results.append(result)

    # Create CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"/home/zeeshan/projects/shark-ai/results/benchmark_results_{token_selection_strategy}.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"\nResults saved to {csv_filename}")


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
