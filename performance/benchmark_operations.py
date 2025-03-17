import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import coordsys
sys.path.insert(0, str(Path(__file__).parent.parent))

from coordsys.transformations import Transformation
import quantities as q

def run_operation(operations, points, num_runs=5, device=None):
    """Run a series of operations on a coordinate system."""

    # Initialize results dictionary
    times = []

    for op in operations:        
        # Setup fresh coordinate system
        ts = Transformation(device=device)
        
        # Measure operation time
        start_time = time.time()
        for _ in range(num_runs):
            if op == 'Translation':
                ts.translation(1, 5, .5)
            elif op == 'Rotation':
                ts.rotation(45 * q.deg, [1, 0, 0])
            elif op == 'Scaling':
                ts.scaling(1.5, 1.5, 1.5)
            else:
                raise ValueError(f"Invalid operation: {op}")
            
            # Apply transformation to points
            ts.apply(points)
            ts.reset()
        
        elapsed = (time.time() - start_time) / num_runs
        times.append(elapsed)
        print(f" Done. Avg time: {elapsed:.6f}s")
    
    return times

def benchmark_operations (nmin=100, nmax=1e6, num_runs=5):
    """Benchmark different coordinate system operations."""
    
    # Initialize results dictionary
    operations = [
        'Translation', 
        'Rotation', 
        'Scaling'
    ]
    results = {
        'operations': operations,
        'cpu_times': [],
        'gpu_times': [],
        'npoints': []
    }
    

    npoints = np.logspace(np.log10(nmin), np.log10(nmax), num=10, dtype=int)
    for num_points in npoints:
        results['npoints'].append(num_points)

        print(f"\nBenchmarking operations with {num_points} points...")
        points = np.random.rand(num_points, 3).astype(np.float32)
        
        # Run operations on CPU
        print ("Benchmarking CPU operations...")
        cpu_times = run_operation(operations, points, device="cpu", num_runs=num_runs)
        results['cpu_times'].append(cpu_times)

        # Run operations on GPU if available
        print ("Benchmarking GPU operations...")
        gpu_times = run_operation(operations, points, device="gpu", num_runs=num_runs)
        results['gpu_times'].append(gpu_times)

    return results

def main():
    print("Starting coordinate system operations benchmark...\n")
    
    # Run benchmarks
    results = benchmark_operations(nmax=1e7, num_runs=1)
    
    # Bar plot of results
    fig, ax = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(results['operations']))
    opacity = 0.8

    for i, device in enumerate(['cpu', 'gpu']):
        ax.bar(index + i * bar_width, results[f'{device}_times'][-1], bar_width,
               alpha=opacity, label=device)
        
    ax.set_xlabel('Operations')
    ax.set_ylabel('Time (s)')
    ax.set_title('Coordinate System Operations Benchmark')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(results['operations'])
    ax.legend()

    # Total time plot
    fig, ax = plt.subplots()
    twinx = ax.twinx()
    npoints = results['npoints']
    times_cpu = np.sum(np.array(results['cpu_times']), axis=1)
    times_gpu = np.sum(np.array(results['gpu_times']), axis=1)
    speedup = times_cpu / times_gpu

    ax.plot(npoints, times_cpu, label='CPU')
    ax.plot(npoints, times_gpu, label='GPU')
    twinx.plot(npoints, speedup, label='Speedup', color='red')
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Total Time (s)')
    ax.set_title('Total Time vs Number of Points')
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
