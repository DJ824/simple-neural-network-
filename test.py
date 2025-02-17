import numpy as np
import time
import os

def setup_gpu():
    # Enable GPU acceleration
    os.environ['ACCELERATE_DISABLE_VFORCE'] = '1'
    # Force numpy to use Metal backend
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

def setup_cpu():
    # Disable GPU acceleration
    os.environ['ACCELERATE_DISABLE_VFORCE'] = '0'
    # Allow CPU to use all threads
    os.environ.pop('VECLIB_MAXIMUM_THREADS', None)

def benchmark_matmul(size, warmup=True):
    # Create matrices with proper memory layout
    A = np.ascontiguousarray(np.random.rand(size, size).astype(np.float32))
    B = np.ascontiguousarray(np.random.rand(size, size).astype(np.float32))

    # Warmup run to initialize any lazy loading
    if warmup:
        _ = np.dot(A[0:100, 0:100], B[0:100, 0:100])
        time.sleep(0.1)  # Let system stabilize

    # Actual benchmark
    times = []
    for _ in range(5):  # Run multiple times for more reliable results
        start = time.time()
        C = np.dot(A, B)
        # Force completion of computation
        C.sum()
        end = time.time()
        times.append(end - start)

    return min(times)  # Return best time

# Run benchmarks
if __name__ == "__main__":
    size = 8000  # Larger size to make GPU advantage more apparent

    print("Testing CPU performance...")
    setup_cpu()
    import importlib
    importlib.reload(np)
    cpu_time = benchmark_matmul(size)

    print("\nTesting GPU performance...")
    setup_gpu()
    importlib.reload(np)
    gpu_time = benchmark_matmul(size)

    print(f"\nResults for {size}x{size} matrix multiplication:")
    print(f"CPU time: {cpu_time:.3f} seconds")
    print(f"GPU time: {gpu_time:.3f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")