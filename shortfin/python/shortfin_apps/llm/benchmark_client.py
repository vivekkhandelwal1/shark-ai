import asyncio
import time
import requests
import uuid
from typing import Dict, Any
import numpy as np
import csv
import os
from datetime import datetime

class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
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
        loop = asyncio.get_event_loop()
        
        def process_stream():
            start_time = time.perf_counter()
            token_times = []
            generated_text = []
            with requests.post(
                f"{self.base_url}/generate", 
                headers=headers, 
                json=data,
                stream=True
            ) as response:
                response.raise_for_status()
                
                # Process the response as it arrives
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data'):
                            # print(line_text)
                            token_time = time.perf_counter()
                            token_times.append(token_time)
                            if save_output:
                                generated_text.append(line_text.split(": ", 1)[1])
                return start_time, token_times, generated_text
        
        # Run the processing function in the thread pool
        start_time, token_times, generated_text = await loop.run_in_executor(None, process_stream)
        
        time_to_first_token = token_times[0] - start_time
        token_generation_times = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
        
        num_tokens = len(token_times)
        
        results["metrics"] = {
            "start_time": start_time,
            "end_time": token_times[-1],
            "time_to_first_token": time_to_first_token,
            "token_generation_times": token_generation_times,
            "num_tokens": num_tokens,
            "generated_text": "".join(generated_text) if save_output else "",
        }
        
        return results

async def run_benchmark(input_token_length:int = 100, output_token_length: int = 50, num_concurrent_requests: int = 64, token_selection_strategy: str = "multi_greedy"):
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
                "token_selection_strategy": "multi_greedy",
                "num_beams": 8
            }
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
    
    # Print results
    print(f"Max Completion Tokens: {output_token_length}")
    print(f"Total number of requests: {num_concurrent_requests}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Requests per second: {num_concurrent_requests/total_time:.2f}")
    print(f"Average latency: {total_time/num_concurrent_requests:.2f} seconds")
    print(f"Time to first token: Mean: {np.mean(time_to_first_token):.4f}s, SD: {np.std(time_to_first_token):.4f}s, Median: {np.median(time_to_first_token):.4f}s, Min: {np.min(time_to_first_token):.4f}s, Max: {np.max(time_to_first_token):.4f}s")
    print(f"Time per token: Mean: {np.mean(token_generation_times):.4f}s, SD: {np.std(token_generation_times):.4f}s, Median: {np.median(token_generation_times):.4f}s, Min: {np.min(token_generation_times):.4f}s, Max: {np.max(token_generation_times):.4f}s")
    print(f"Request processing start time: Mean: {np.mean(start_times):.4f}s, SD: {np.std(start_times):.4f}s, Median: {np.median(start_times):.4f}s, Min: {np.min(start_times):.4f}s, Max: {np.max(start_times):.4f}s")
    print(f"Request processing end time: Mean: {np.mean(end_times):.4f}s, SD: {np.std(end_times):.4f}s, Median: {np.median(end_times):.4f}s, Min: {np.min(end_times):.4f}s, Max: {np.max(end_times):.4f}s")
    print(f"Time per request: Mean: {np.mean(time_per_request):.4f}s, SD: {np.std(time_per_request):.4f}s, Median: {np.median(time_per_request):.4f}s, Min: {np.min(time_per_request):.4f}s, Max: {np.max(time_per_request):.4f}s")
    print(f"Number of generated tokens: Mean: {np.mean(num_generated_tokens):.4f}, SD: {np.std(num_generated_tokens):.4f}, Median: {np.median(num_generated_tokens):.4f}, Min: {np.min(num_generated_tokens):.4f}, Max: {np.max(num_generated_tokens):.4f}")
    print(f"Generated text:\n {results[0]['metrics']['generated_text']}")
    
    # Return results for CSV
    return {
        "max_completion_tokens": output_token_length,
        "num_requests": num_concurrent_requests,
        "total_time": total_time,
        "requests_per_second": num_concurrent_requests/total_time,
        "avg_latency": total_time/num_concurrent_requests,
        "ttft_mean": np.mean(time_to_first_token),
        "ttft_std": np.std(time_to_first_token),
        "ttft_median": np.median(time_to_first_token),
        "ttft_min": np.min(time_to_first_token),
        "ttft_max": np.max(time_to_first_token),
        "time_per_token_mean": np.mean(token_generation_times),
        "time_per_token_std": np.std(token_generation_times),
        "time_per_token_median": np.median(token_generation_times),
        "time_per_token_min": np.min(token_generation_times),
        "time_per_token_max": np.max(token_generation_times),
        "start_time_mean": np.mean(start_times),
        "start_time_std": np.std(start_times),
        "start_time_median": np.median(start_times),
        "start_time_min": np.min(start_times),
        "start_time_max": np.max(start_times),
        "end_time_mean": np.mean(end_times),
        "end_time_std": np.std(end_times),
        "end_time_median": np.median(end_times),
        "end_time_min": np.min(end_times),
        "end_time_max": np.max(end_times),
        "time_per_request_mean": np.mean(time_per_request),
        "time_per_request_std": np.std(time_per_request),
        "time_per_request_median": np.median(time_per_request),
        "time_per_request_min": np.min(time_per_request),
        "time_per_request_max": np.max(time_per_request),
        "num_tokens_mean": np.mean(num_generated_tokens),
        "num_tokens_std": np.std(num_generated_tokens),
        "num_tokens_median": np.median(num_generated_tokens),
        "num_tokens_min": np.min(num_generated_tokens),
        "num_tokens_max": np.max(num_generated_tokens),
    }

async def run_all_benchmarks():
    token_values = [1, 16, 32, 64, 256]
    all_results = []
    
    for tokens in token_values:
        print(f"\n\nRunning benchmark with max_completion_tokens = {tokens}")
        result = await run_benchmark(output_token_length=tokens, num_concurrent_requests=100)
        all_results.append(result)
    
    # Create CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_results_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"\nResults saved to {csv_filename}")
    
if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())