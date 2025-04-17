import asyncio
import time
from typing import Dict, Any, List, Optional
import numpy as np
from collections import deque
from datetime import datetime
import csv
import argparse
from .benchmark_client import LLMClient

class MedianThroughputBenchmark:
    def __init__(
        self,
        endpoint: str = "http://localhost:8001",
        target_latency: float = 4.2,
        warmup_time: float = 10.0,
        measurement_time: float = 60.0,
        max_concurrent_requests: int = 100
    ):
        self.client = LLMClient(base_url=endpoint)
        self.target_latency = target_latency
        self.warmup_time = warmup_time
        self.measurement_time = measurement_time
        self.max_concurrent_requests = max_concurrent_requests
        
        # State tracking
        self.active_requests = 0
        self.completed_requests = 0
        self.request_latencies = []
        self.request_start_times = {}
        self.throughput_window = deque(maxlen=50)  # Rolling window of throughput measurements
        self.concurrent_requests = 1
        self.benchmark_start_time = None
        self.median_latency = float('inf')
        self.delay_between_requests = 0.001

    async def _request_worker(self, text: str, sampling_params: Dict[str, Any]) -> float:
        """Handle a single request and return its latency."""
        request_id = id(asyncio.current_task())
        self.request_start_times[request_id] = time.perf_counter()
        
        try:
            print(f"Sending request {request_id}")
            result = await self.client.generate(text=text, sampling_params=sampling_params)
            end_time = time.perf_counter()
            latency = end_time - self.request_start_times[request_id]
            
            print(f"Request {request_id} completed with latency {latency:.2f}s")
            self.request_latencies.append(latency)
            self.completed_requests += 1
            recent_latencies = self.request_latencies[-1:]  # Look at last 3 requests
            self.median_latency = float('inf') if len(recent_latencies) == 0 else np.median(recent_latencies)
            if self.median_latency < self.target_latency:
                self.delay_between_requests *= 10
                self.concurrent_requests += 1
            if self.median_latency > self.target_latency + 0.5:
                self.delay_between_requests /= 10
            print(f"Current median latency: {self.median_latency:.2f}s, Concurrent requests: {self.concurrent_requests}")
            
            # Update throughput window
            elapsed = end_time - self.benchmark_start_time
            if elapsed >= self.warmup_time:
                current_throughput = self.completed_requests / elapsed
                self.throughput_window.append(current_throughput)
            
            return latency
            
        except Exception as e:
            print(f"Error in request {request_id}: {e}")
            raise
        finally:
            self.active_requests -= 1
            del self.request_start_times[request_id]

    async def _request_manager(
        self,
        text: str,
        sampling_params: Dict[str, Any]
    ):
        """Manage adding new requests based on latency conditions."""
        # Start with one request immediately
        print("Starting initial request...")
        self.active_requests += 1
        first_task = asyncio.create_task(self._request_worker(text, sampling_params))
        self.concurrent_requests = 1
        while True:
            
            current_time = time.perf_counter()
            elapsed = current_time - self.benchmark_start_time
            
           
            # Check if benchmark should continue
            if elapsed >= (self.warmup_time + self.measurement_time):
                print("Benchmark duration reached, stopping...")
                break
            # Print current state
              
            # Check if we should add a new request
            if self.active_requests < self.concurrent_requests:
                self.active_requests += 1
                print(f"Adding new request (total active: {self.active_requests})")
                print(f"Active requests: {self.active_requests}, Concurrent requests: {self.concurrent_requests}, Completed: {self.completed_requests}, Elapsed: {elapsed:.2f}s, Median latency: {self.median_latency:.2f}s")
            
                asyncio.create_task(self._request_worker(text, sampling_params))
               
                # Wait a small amount of time before potentially adding another request
                await asyncio.sleep(self.delay_between_requests)
            else:
                await asyncio.sleep(self.delay_between_requests)
                
            # Update should_add_request based on recent latencies
            # if len(self.request_latencies) > 0:
                

    async def run_benchmark(
        self,
        input_token_length: int = 1024,
        output_token_length: int = 64,
        token_selection_strategy: str = "greedy"
    ) -> Dict[str, Any]:
        """Run the benchmark and return results."""
        # Reset state
        self.active_requests = 0
        self.completed_requests = 0
        self.request_latencies = []
        self.request_start_times = {}
        self.throughput_window.clear()
        
        # Prepare request parameters
        text = " ".join(["one" for _ in range(input_token_length)])
        sampling_params = {
            "max_completion_tokens": output_token_length,
            "token_selection_strategy": token_selection_strategy,
        }
        
        # Start benchmark
        print(f"\nStarting benchmark with:")
        print(f"- Input tokens: {input_token_length}")
        print(f"- Output tokens: {output_token_length}")
        print(f"- Token selection: {token_selection_strategy}")
        print(f"- Target latency: {self.target_latency}s")
        
        self.benchmark_start_time = time.perf_counter()
        
        try:
            # Start request manager
            await self._request_manager(text, sampling_params)
            
            # Calculate results
            end_time = time.perf_counter()
            total_time = end_time - self.benchmark_start_time
            measurement_time = total_time - self.warmup_time
            
            if not self.request_latencies:
                print("No requests completed successfully!")
                return {
                    "error": "No requests completed",
                    "input_token_length": input_token_length,
                    "output_token_length": output_token_length,
                    "token_selection_strategy": token_selection_strategy,
                }
            
            # Filter latencies from measurement period
            measurement_latencies = [
                lat for lat in self.request_latencies
                if lat <= measurement_time
            ]
            
            results = {
                "input_token_length": input_token_length,
                "output_token_length": output_token_length,
                "token_selection_strategy": token_selection_strategy,
                "total_requests": self.completed_requests,
                "measurement_requests": len(measurement_latencies),
                "median_latency": np.median(measurement_latencies),
                "p95_latency": np.percentile(measurement_latencies, 95),
                "p99_latency": np.percentile(measurement_latencies, 99),
                "median_throughput": np.median(list(self.throughput_window)) if self.throughput_window else 0,
                "max_concurrent_requests": max(self.active_requests, 1),
                "measurement_time": measurement_time,
            }
            
            return results
            
        except Exception as e:
            print(f"Benchmark failed with error: {e}")
            raise

def save_results(results: Dict[str, Any], token_selection_strategy: str):
    """Save benchmark results to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"median_throughput_results_{token_selection_strategy}_{timestamp}.csv"
    
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    print(f"\nResults saved to {filename}")

async def main():
    parser = argparse.ArgumentParser(description="Run median throughput benchmark")
    parser.add_argument("--input-tokens", type=int, default=1024,
                      help="Number of input tokens")
    parser.add_argument("--output-tokens", type=int, default=64,
                      help="Number of output tokens")
    parser.add_argument("--token-selection", type=str, default="greedy",
                      choices=["greedy", "multi_greedy", "beam_search"],
                      help="Token selection strategy")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080",
                      help="LLM server endpoint")
    parser.add_argument("--target-latency", type=float, default=4.2,
                      help="Target E2E latency in seconds")
    parser.add_argument("--warmup-time", type=float, default=10.0,
                      help="Warmup time in seconds")
    parser.add_argument("--measurement-time", type=float, default=60.0,
                      help="Measurement time in seconds")
    
    args = parser.parse_args()
    
    benchmark = MedianThroughputBenchmark(
        endpoint=args.endpoint,
        target_latency=args.target_latency,
        warmup_time=args.warmup_time,
        measurement_time=args.measurement_time
    )
    
    results = await benchmark.run_benchmark(
        input_token_length=args.input_tokens,
        output_token_length=args.output_tokens,
        token_selection_strategy=args.token_selection
    )
    
    save_results(results, args.token_selection)

if __name__ == "__main__":
    asyncio.run(main()) 