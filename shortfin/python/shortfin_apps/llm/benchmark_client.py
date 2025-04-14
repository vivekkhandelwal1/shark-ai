import asyncio
import time
import requests
import uuid
from typing import Dict, Any
import numpy as np
import csv
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8000", max_workers: int = 128):
        self.base_url = base_url.rstrip('/')
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
            "rid": uuid.uuid4().hex,
            "return_logprob": False,
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
            "stream": True,
        }
        
        headers = {"Content-Type": "application/json"}
        
        results = {}
        
        def process_stream():
            start_time = time.perf_counter()
            token_times = []
            generated_text = []
            try:
                with requests.post(
                    f"{self.base_url}/generate", 
                    headers=headers, 
                    json=data,
                    stream=True,
                    timeout=60  # Add timeout to prevent hanging
                ) as response:
                    response.raise_for_status()
                    
                    # Process the response as it arrives
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data'):
                                token_time = time.perf_counter()
                                token_times.append(token_time)
                                if save_output:
                                    generated_text.append(line_text.split(": ", 1)[1])
            except Exception as e:
                print(f"Error in process_stream: {e}")
                # Return empty results in case of error
                return start_time, [], []
                
            return start_time, token_times, generated_text
        print("Processing stream")
        # Run the processing function in the thread pool with our dedicated executor
        start_time, token_times, generated_text = await asyncio.get_event_loop().run_in_executor(
            self.executor, process_stream
        )
        print("Stream processed")
        # Handle case where no tokens were generated
        if not token_times:
            results["metrics"] = {
                "start_time": start_time,
                "end_time": start_time,
                "time_to_first_token": 0,
                "token_generation_times": [],
                "num_tokens": 0,
                "generated_text": "",
                "error": "No tokens generated"
            }
            return results
            
        time_to_first_token = token_times[0] - start_time if token_times else 0
        
        num_tokens = len(token_times)
        
        results["metrics"] = {
            "start_time": start_time,
            "end_time": token_times[-1],
            "time_to_first_token": time_to_first_token,
            "token_generation_times": token_times,
            "num_tokens": num_tokens,
            "generated_text": "".join(generated_text) if save_output else "",
        }
        
        return results
        
    def __del__(self):
        # Ensure executor is shut down when client is destroyed
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

async def run_benchmark(input_token_length:int = 100, output_token_length: int = 50,
                        num_concurrent_requests: int = 64, token_selection_strategy: str = "multi_greedy"):
    client = LLMClient()

    prompt = [ "one" for _ in range(input_token_length)]
    prompt = " ".join(prompt)

    # Create tasks
    tasks = []
    for _ in range(num_concurrent_requests):
        tasks.append(client.generate(
            text=prompt,
            sampling_params={
                "max_completion_tokens": output_token_length,
                "token_selection_strategy": token_selection_strategy,
                "num_beams": 8
            },
            save_output=True
        ))
    
    # Run benchmark
    start_time = time.perf_counter()
    results = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Collect metrics
    token_generation_times = [result["metrics"]["token_generation_times"] for result in results]
    time_to_first_token = [result["metrics"]["time_to_first_token"] for result in results]
    start_times = [result["metrics"]["start_time"] - start_time for result in results]
    end_times = [result["metrics"]["end_time"] - start_time for result in results]
    time_per_request = [end_times[i] - start_times[i] for i in range(num_concurrent_requests)]
    num_generated_tokens = [result["metrics"]["num_tokens"] for result in results]
    # print(f"Generated text: {results[0]['metrics']['generated_text']}")
    flattened_token_times = sorted([item for sublist in token_generation_times for item in sublist])
    TPS_times = [flattened_token_times[i] - flattened_token_times[i-1] for i in range(1, len(flattened_token_times))]
    TPOT_times = [[token_times[i] - token_times[i-1] for i in range(1, len(token_times))] for token_times in token_generation_times]
    TPOT_times = [item for sublist in TPOT_times for item in sublist]
    
    # Print results
    print(f"Input token length: {input_token_length}")
    print(f"Output token length: {output_token_length}")
    print(f"Number of concurrent requests: {num_concurrent_requests}")
    print(f"Token selection strategy: {token_selection_strategy}")
    print(f"E2E latency: {total_time:.2f} seconds")
    print(f"Requests per second: {num_concurrent_requests/total_time:.2f}")
    print(f"Average latency: {total_time/num_concurrent_requests:.2f} seconds")
    print(f"Time to first token: Mean: {np.mean(time_to_first_token):.4f}s, SD: {np.std(time_to_first_token):.4f}s, Median: {np.median(time_to_first_token):.4f}s, Min: {np.min(time_to_first_token):.4f}s, Max: {np.max(time_to_first_token):.4f}s")
    print(f"Time per output token: Mean: {np.mean(TPOT_times):.4f}s, SD: {np.std(TPOT_times):.4f}s, Median: {np.median(TPOT_times):.4f}s, Min: {np.min(TPOT_times):.4f}s, Max: {np.max(TPOT_times):.4f}s")
    print(f"Tokens per second: Mean: {np.mean(TPS_times):.4f}s, SD: {np.std(TPS_times):.4f}s, Median: {np.median(TPS_times):.4f}s, Min: {np.min(TPS_times):.4f}s, Max: {np.max(TPS_times):.4f}s")
    print(f"Request processing start time: Mean: {np.mean(start_times):.4f}s, SD: {np.std(start_times):.4f}s, Median: {np.median(start_times):.4f}s, Min: {np.min(start_times):.4f}s, Max: {np.max(start_times):.4f}s")
    print(f"Request processing end time: Mean: {np.mean(end_times):.4f}s, SD: {np.std(end_times):.4f}s, Median: {np.median(end_times):.4f}s, Min: {np.min(end_times):.4f}s, Max: {np.max(end_times):.4f}s")
    print(f"Time per request: Mean: {np.mean(time_per_request):.4f}s, SD: {np.std(time_per_request):.4f}s, Median: {np.median(time_per_request):.4f}s, Min: {np.min(time_per_request):.4f}s, Max: {np.max(time_per_request):.4f}s")
    print(f"Number of generated tokens: Mean: {np.mean(num_generated_tokens):.4f}, SD: {np.std(num_generated_tokens):.4f}, Median: {np.median(num_generated_tokens):.4f}, Min: {np.min(num_generated_tokens):.4f}, Max: {np.max(num_generated_tokens):.4f}")
    print(f"Generated text:\n {results[0]['metrics']['generated_text']}")
    
    # Return results for CSV
    return {
        "token_selection_strategy": token_selection_strategy,
        "input_token_length": input_token_length,
        "output_token_length": output_token_length,
        "num_concurrent_requests": num_concurrent_requests,
        "total_time": total_time,
        "requests_per_second": num_concurrent_requests/total_time,
        "avg_latency": total_time/num_concurrent_requests,
        "TTFT": f"Mean: {np.mean(time_to_first_token):.4f}s, SD: {np.std(time_to_first_token):.4f}s, Median: {np.median(time_to_first_token):.4f}s, Min: {np.min(time_to_first_token):.4f}s, Max: {np.max(time_to_first_token):.4f}s",
        "TPOT": f"Mean: {np.mean(TPOT_times):.4f}s, SD: {np.std(TPOT_times):.4f}s, Median: {np.median(TPOT_times):.4f}s, Min: {np.min(TPOT_times):.4f}s, Max: {np.max(TPOT_times):.4f}s",
        "TPS": f"Mean: {np.mean(TPS_times):.4f}s, SD: {np.std(TPS_times):.4f}s, Median: {np.median(TPS_times):.4f}s, Min: {np.min(TPS_times):.4f}s, Max: {np.max(TPS_times):.4f}s",
        "start_time": f"Mean: {np.mean(start_times):.4f}s, SD: {np.std(start_times):.4f}s, Median: {np.median(start_times):.4f}s, Min: {np.min(start_times):.4f}s, Max: {np.max(start_times):.4f}s",
        "end_time": f"Mean: {np.mean(end_times):.4f}s, SD: {np.std(end_times):.4f}s, Median: {np.median(end_times):.4f}s, Min: {np.min(end_times):.4f}s, Max: {np.max(end_times):.4f}s",
        "time_per_request": f"Mean: {np.mean(time_per_request):.4f}s, SD: {np.std(time_per_request):.4f}s, Median: {np.median(time_per_request):.4f}s, Min: {np.min(time_per_request):.4f}s, Max: {np.max(time_per_request):.4f}s",
        "num_actual_tokens_generated": f"Mean: {np.mean(num_generated_tokens):.4f}, SD: {np.std(num_generated_tokens):.4f}, Median: {np.median(num_generated_tokens):.4f}, Min: {np.min(num_generated_tokens):.4f}, Max: {np.max(num_generated_tokens):.4f}",
    }

async def run_all_benchmarks():
    input_token_lengths = [1024]
    output_token_lengths = [1, 16, 32, 64]
    num_concurrent_requests = [1, 2, 4, 8]
    token_selection_strategies = ["greedy", "multi_greedy", "beam_search"]
    token_selection_strategy = token_selection_strategies[0]
    all_results = []

    for num_concurrent_requests in num_concurrent_requests:
        for input_token_length in input_token_lengths:
            for output_token_length in output_token_lengths:
                print(f"\n\nRunning benchmark with max_completion_tokens = {output_token_length}")
                result = await run_benchmark(input_token_length=input_token_length, output_token_length=output_token_length, num_concurrent_requests=num_concurrent_requests, token_selection_strategy=token_selection_strategy)
                all_results.append(result)
    
    # Create CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"/home/zeeshan/projects/shark-ai/results/benchmark_results_{token_selection_strategy}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"\nResults saved to {csv_filename}")
    
if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())