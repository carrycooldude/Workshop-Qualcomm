"""
============================================================================
 Module 10 — Optimization (src/06_optimize.py)
============================================================================
 What this script does:
   1. Establishes a baseline with FP32 model
   2. Compiles with INT8 quantization via AI Hub
   3. Tries different model variants (MobileNetV3, EfficientNet-Lite)
   4. Compares: Baseline → Quantized → Different Architecture
   5. Shows the optimization-accuracy tradeoff

 Run:
   python src/06_optimize.py

 Optimization strategies covered:
   • Quantization (FP32 → INT8) — reduces model size 4× and speeds up NPU
   • Model architecture choice — smaller models run faster
   • Runtime selection — choosing the right compute backend
============================================================================
"""

import os
import sys
import time
import json
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
    from torchvision.models import (
        mobilenet_v2, MobileNet_V2_Weights,
        mobilenet_v3_small, MobileNet_V3_Small_Weights,
        efficientnet_b0, EfficientNet_B0_Weights,
    )
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_SHAPE = (1, 3, 224, 224)
TARGET_DEVICE = "Samsung Galaxy S24 (Family)"
COMPILED_DIR = Path("models/compiled")
RESULTS_DIR = Path("benchmarks/results")


def compare_model_sizes():
    """
    Compare different model architectures.

    When optimizing for edge, your first lever is MODEL CHOICE.
    A smaller model that meets your accuracy needs will always be
    faster than a large model that you try to compress.
    """
    models_info = []

    configs = [
        ("MobileNetV2", mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1, 71.9),
        ("MobileNetV3-Small", mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1, 67.7),
        ("EfficientNet-B0", efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 77.7),
    ]

    for name, model_fn, weights, top1_acc in configs:
        model = model_fn(weights=weights)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

        # Quick local benchmark
        dummy = torch.randn(INPUT_SHAPE)
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = model(dummy)
            # Measure
            start = time.perf_counter()
            for _ in range(20):
                _ = model(dummy)
            avg_ms = (time.perf_counter() - start) * 1000 / 20

        info = {
            "name": name,
            "params": num_params,
            "size_mb": size_mb,
            "top1_accuracy": top1_acc,
            "local_cpu_ms": avg_ms,
        }
        models_info.append(info)
        print(f"  {name:>20}: {num_params/1e6:.1f}M params, {size_mb:.1f} MB, "
              f"Top-1={top1_acc}%, CPU={avg_ms:.1f}ms")

    return models_info


def demonstrate_quantization_concepts():
    """
    Explain quantization without requiring AI Hub.

    Quantization converts a model's weights and activations from
    floating-point (FP32) to lower-precision integers (INT8, INT4).

    This is the SINGLE MOST IMPORTANT optimization for Snapdragon NPUs,
    because the Hexagon NPU is specifically designed for integer math.
    """
    print()
    print("  What is Quantization?")
    print("  " + "─" * 50)
    print()
    print("  FP32 (32 bits per value):")
    print("    • Highest precision — what PyTorch uses by default")
    print("    • Model size: ~14 MB for MobileNetV2")
    print("    • Runs on CPU and GPU")
    print()
    print("  FP16 (16 bits per value):")
    print("    • Half precision — 2× smaller than FP32")
    print("    • Model size: ~7 MB")
    print("    • Minimal accuracy loss (<0.1%)")
    print("    • Runs well on GPU and modern NPUs")
    print()
    print("  INT8 (8 bits per value):")
    print("    • 4× smaller than FP32")
    print("    • Model size: ~3.5 MB")
    print("    • Small accuracy loss (~0.5-1%)")
    print("    • ⚡ Runs FASTEST on Hexagon NPU")
    print("    • Requires calibration data for best results")
    print()
    print("  INT4 (4 bits per value):")
    print("    • 8× smaller than FP32")
    print("    • Primarily used for LLMs (Genie/GenieX)")
    print("    • Larger accuracy loss — acceptable for LLMs")
    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │          Precision vs Performance Tradeoff       │")
    print("  │                                                  │")
    print("  │  Accuracy  ████████████████████  FP32            │")
    print("  │            ███████████████████   FP16            │")
    print("  │            █████████████████     INT8            │")
    print("  │            ████████████          INT4            │")
    print("  │                                                  │")
    print("  │  Speed     ████                  FP32 (CPU)      │")
    print("  │            ████████              FP16 (GPU)      │")
    print("  │            ████████████████████  INT8 (NPU) ⚡   │")
    print("  │            ████████████████████  INT4 (NPU) ⚡   │")
    print("  └──────────────────────────────────────────────────┘")


def compile_with_quantization():
    """
    Compile MobileNetV2 with INT8 quantization via AI Hub.

    When you submit a compilation job with quantization, AI Hub will:
      1. Analyze your model's operations
      2. Determine which layers can be quantized
      3. Use calibration data (or default calibration) to find optimal
         quantization parameters (scale and zero-point for each tensor)
      4. Convert FP32 → INT8 for those layers
      5. Compile for the NPU
    """
    if not HAS_HUB:
        print("  ⚠ qai-hub not installed — showing concepts only")
        print("    To compile with quantization, install qai-hub and run:")
        print()
        print("    import qai_hub as hub")
        print("    compile_job = hub.submit_compile_job(")
        print("        model=exported_model,")
        print("        device=hub.Device('Samsung Galaxy S24 (Family)'),")
        print("        input_specs=dict(image=(1, 3, 224, 224)),")
        print("        options='--quantize_full_type int8 --quantize_io',")
        print("    )")
        return

    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    example_input = torch.randn(INPUT_SHAPE)

    with torch.no_grad():
        try:
            exported_model = torch.export.export(model, (example_input,))
        except Exception:
            exported_model = torch.jit.trace(model, example_input)

    device = hub.Device(TARGET_DEVICE)

    # Generate simple calibration data (random for demo; use real data in production)
    calibration_data = hub.upload_dataset(
        {"image": [np.random.randn(*INPUT_SHAPE).astype(np.float32) for _ in range(10)]}
    )

    print("  Submitting INT8 quantized compilation...")
    compile_job = hub.submit_compile_job(
        model=exported_model,
        device=device,
        input_specs=dict(image=INPUT_SHAPE),
        calibration_data=calibration_data,
        name="workshop-mobilenet-v2-int8",
    )

    print(f"  ℹ Job ID: {compile_job.job_id}")
    status = compile_job.get_status()

    if status.success:
        COMPILED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = COMPILED_DIR / "mobilenet_v2_int8.tflite"
        target_model = compile_job.get_target_model()
        target_model.download(str(output_path))
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ INT8 model saved: {output_path} ({file_size:.1f} MB)")
    else:
        print(f"  ❌ Quantized compilation failed. Check the dashboard.")


def main():
    print("=" * 70)
    print(" Module 10 — Optimization: Making It Faster")
    print("=" * 70)
    print()

    # Step 1: Compare architectures
    print("Step 1: Comparing model architectures")
    print("-" * 60)
    models_info = compare_model_sizes()
    print()

    # Step 2: Understand quantization
    print("Step 2: Understanding quantization")
    print("-" * 60)
    demonstrate_quantization_concepts()
    print()

    # Step 3: Compile with quantization
    print("Step 3: Compiling with INT8 quantization")
    print("-" * 60)
    compile_with_quantization()
    print()

    # Summary
    print("=" * 70)
    print("🎯 Optimization Strategy Summary:")
    print()
    print("  1. CHOOSE the right model architecture first")
    print("     → Smaller model = faster inference, if accuracy is sufficient")
    print()
    print("  2. QUANTIZE (FP32 → INT8) to unlock the NPU")
    print("     → 4× smaller, often 5–10× faster on Hexagon NPU")
    print()
    print("  3. SELECT the right compute backend")
    print("     → CPU: always works, slowest")
    print("     → GPU: good for FP16, parallel workloads")
    print("     → NPU: fastest for INT8 quantized models")
    print()
    print("  4. PROFILE to verify your optimization helped")
    print("     → Never assume — always measure")
    print()
    print("⏭  Next: Module 7 — Try Generative AI with GenieX")
    print("   Run: python src/07_geniex_demo.py")
    print()


if __name__ == "__main__":
    main()
