"""
============================================================================
 Module 9 — Benchmarking (src/05_benchmark.py)
============================================================================
 What this script does:
   1. Runs local CPU inference and measures latency
   2. Compiles the model for multiple backends via AI Hub
   3. Profiles on different compute units (CPU/GPU/NPU)
   4. Compares performance across all backends
   5. Generates a benchmark report with a comparison table

 Run:
   python src/05_benchmark.py

 This module teaches you to MEASURE before you OPTIMIZE.
============================================================================
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import torch
    import numpy as np
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
except ImportError as e:
    print(f"\n❌ Missing required package: {e.name}")
    print("   Please install workshop dependencies:")
    print("     pip install -r requirements.txt")
    print("   Or install directly:")
    print(f"     pip install {e.name or 'torch torchvision'}\n")
    sys.exit(1)

try:
    import qai_hub as hub
    HAS_HUB = True
except ImportError:
    HAS_HUB = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_SHAPE = (1, 3, 224, 224)
NUM_WARMUP = 5
NUM_RUNS = 50
RESULTS_DIR = Path("benchmarks/results")
TARGET_DEVICE = "Samsung Galaxy S24 (Family)"


def benchmark_local_cpu() -> dict:
    """
    Benchmark MobileNetV2 on local CPU.

    This is our baseline — a standard PyTorch model running on your
    development machine's CPU. We measure:
      - Warmup time (first few runs are slower due to JIT, cache, etc.)
      - Average latency over many runs
      - Standard deviation (consistency)
      - Min/Max latency
      - Throughput (inferences per second)
    """
    print("  Loading model...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    dummy_input = torch.randn(INPUT_SHAPE)

    # Warmup
    print(f"  Warming up ({NUM_WARMUP} runs)...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(dummy_input)

    # Benchmark
    print(f"  Benchmarking ({NUM_RUNS} runs)...")
    latencies = []
    with torch.no_grad():
        for i in range(NUM_RUNS):
            start = time.perf_counter()
            _ = model(dummy_input)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

    results = {
        "backend": "Local CPU (PyTorch)",
        "device": "Development Machine",
        "avg_latency_ms": statistics.mean(latencies),
        "std_latency_ms": statistics.stdev(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))],
        "throughput_fps": 1000 / statistics.mean(latencies),
        "num_runs": NUM_RUNS,
    }
    return results


def benchmark_hub_profile(runtime: str = "tflite") -> dict | None:
    """
    Benchmark via AI Hub profiling on real Snapdragon hardware.

    This submits a profiling job that runs on actual hardware and returns
    precise measurements from the device.
    """
    if not HAS_HUB:
        return None

    compiled_path = Path(f"models/compiled/mobilenet_v2_compiled.{runtime}")
    if runtime == "tflite":
        compiled_path = Path("models/compiled/mobilenet_v2_compiled.tflite")

    if not compiled_path.exists():
        print(f"  ⚠ Compiled model not found: {compiled_path}")
        print(f"    Run Module 5 first: python src/03_compile_model.py")
        return None

    try:
        device = hub.Device(TARGET_DEVICE)
        print(f"  Submitting profile job for {runtime}...")

        profile_job = hub.submit_profile_job(
            model=str(compiled_path),
            device=device,
            name=f"workshop-benchmark-{runtime}",
        )

        status = profile_job.get_status()
        if status.success:
            profile_data = profile_job.download_profile()
            return {
                "backend": f"Snapdragon ({runtime})",
                "device": TARGET_DEVICE,
                "profile_data": str(profile_data),
                "job_id": profile_job.job_id,
            }
    except Exception as e:
        print(f"  ⚠ Profile failed: {e}")

    return None


def display_results(all_results: list[dict]) -> None:
    """Display benchmark results as a formatted table."""
    if not all_results:
        return

    if HAS_TABULATE:
        headers = ["Backend", "Avg Latency (ms)", "P95 (ms)", "Throughput (FPS)", "Std Dev"]
        rows = []
        for r in all_results:
            rows.append([
                r.get("backend", "N/A"),
                f"{r.get('avg_latency_ms', 0):.1f}",
                f"{r.get('p95_latency_ms', 0):.1f}",
                f"{r.get('throughput_fps', 0):.1f}",
                f"±{r.get('std_latency_ms', 0):.1f}",
            ])
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    else:
        print(f"  {'Backend':<30} {'Avg (ms)':<12} {'P95 (ms)':<12} {'FPS':<10}")
        print(f"  {'—'*28:<30} {'—'*10:<12} {'—'*10:<12} {'—'*8:<10}")
        for r in all_results:
            print(
                f"  {r.get('backend', 'N/A'):<30} "
                f"{r.get('avg_latency_ms', 0):>10.1f}  "
                f"{r.get('p95_latency_ms', 0):>10.1f}  "
                f"{r.get('throughput_fps', 0):>8.1f}"
            )


def explain_results() -> None:
    """Explain why performance differs across backends."""
    print()
    print("📊 Why do the numbers differ?")
    print()
    print("  CPU (your development machine):")
    print("    • General-purpose processor")
    print("    • Runs ML ops using floating-point math libraries")
    print("    • No hardware acceleration for neural networks")
    print()
    print("  Snapdragon CPU (on-device):")
    print("    • ARM-based mobile CPU (Kryo/Oryon)")
    print("    • Lower clock speed than desktop, but more power efficient")
    print()
    print("  Snapdragon GPU (Adreno):")
    print("    • Designed for parallel computation")
    print("    • FP16 support for faster math")
    print("    • Good for models with many parallel operations")
    print()
    print("  Snapdragon NPU (Hexagon):")
    print("    • Purpose-built for neural network inference")
    print("    • INT8/INT16 quantized operations at very high throughput")
    print("    • Lowest power consumption for ML workloads")
    print("    • Can be 5–50× faster than CPU for supported operations")
    print()


def save_results(all_results: list[dict]) -> None:
    """Save benchmark results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "benchmark_comparison.json"

    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "MobileNetV2",
        "input_shape": list(INPUT_SHAPE),
        "num_runs": NUM_RUNS,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  ✓ Results saved to: {output_path}")


def main():
    print("=" * 70)
    print(" Module 9 — Benchmarking: Measure Before You Optimize")
    print("=" * 70)
    print()

    all_results = []

    # Step 1: Local CPU benchmark
    print("Step 1: Benchmarking on local CPU")
    cpu_results = benchmark_local_cpu()
    all_results.append(cpu_results)
    print(f"  ✓ Avg latency: {cpu_results['avg_latency_ms']:.1f} ms")
    print(f"  ✓ Throughput:  {cpu_results['throughput_fps']:.1f} FPS")
    print()

    # Step 2: AI Hub profiling (if available)
    print("Step 2: Profiling on Snapdragon (via AI Hub)")
    if HAS_HUB:
        hub_results = benchmark_hub_profile("tflite")
        if hub_results:
            all_results.append(hub_results)
            print(f"  ✓ Profile data collected")
    else:
        print("  ⚠ qai-hub not installed — skipping cloud profiling")
        print("    Install: pip install qai-hub")
    print()

    # Step 3: Display comparison
    print("Step 3: Benchmark Comparison")
    print("-" * 70)
    display_results(all_results)
    print("-" * 70)

    # Step 4: Explain
    explain_results()

    # Step 5: Save
    print("Step 5: Saving results")
    save_results(all_results)
    print()

    print("⏭  Next: Module 10 — Optimize the model")
    print("   Run: python src/06_optimize.py")
    print()


if __name__ == "__main__":
    main()
