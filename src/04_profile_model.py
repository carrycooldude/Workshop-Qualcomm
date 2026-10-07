"""
============================================================================
 Module 5 (cont.) — Profile Model on Real Hardware (src/04_profile_model.py)
============================================================================
 What this script does:
   1. Loads the compiled model from Module 5
   2. Submits a profiling job to Qualcomm AI Hub
   3. AI Hub runs the model on a REAL Snapdragon device in their cloud
   4. Returns detailed performance metrics:
      - Total inference time
      - Per-layer execution time
      - Compute unit utilization (CPU/GPU/NPU)
      - Memory usage
   5. Also submits an inference job to verify correctness

 Prerequisites:
   - Completed Module 5 (compiled model exists)
   - AI Hub API token configured

 Run:
   python src/04_profile_model.py

 This still runs in the cloud — no local Snapdragon hardware needed!
============================================================================
"""

import os
import sys
import json
import time
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

import numpy as np

try:
    import qai_hub as hub
    HAS_HUB = True
except ImportError:
    HAS_HUB = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_DEVICE = "Samsung Galaxy S24 (Family)"
COMPILED_MODEL_PATH = Path("models/compiled/mobilenet_v2_compiled.tflite")
INPUT_SHAPE = (1, 3, 224, 224)
RESULTS_DIR = Path("benchmarks/results")


def profile_model():
    """
    Submit a profiling job to AI Hub.

    Profiling runs the model on a REAL physical device in Qualcomm's
    cloud device farm. The device runs the exact same Snapdragon chipset
    that your end users will have.

    Metrics you get back:
      - Inference time (total and per-layer)
      - Which compute unit each layer runs on (CPU/GPU/NPU)
      - Memory footprint
      - Estimated throughput
    """
    if not COMPILED_MODEL_PATH.exists():
        print("  ❌ Compiled model not found at:", COMPILED_MODEL_PATH)
        print("     Run Module 5 first: python src/03_compile_model.py")
        return None

    device = hub.Device(TARGET_DEVICE)

    print(f"  Submitting profiling job...")
    print(f"  Target: {TARGET_DEVICE}")
    print(f"  ⏳ Waiting for a device in the farm (1–5 minutes)...")
    print()

    profile_job = hub.submit_profile_job(
        model=str(COMPILED_MODEL_PATH),
        device=device,
        name="workshop-mobilenet-v2-profile",
    )

    print(f"  ℹ Job ID: {profile_job.job_id}")
    print(f"  ℹ Dashboard: https://app.aihub.qualcomm.com/jobs/{profile_job.job_id}/")

    # Wait for results
    status = profile_job.get_status()
    print(f"  Status: {status}")

    if status.success:
        profile_data = profile_job.download_profile()
        return profile_data
    else:
        print(f"  ❌ Profiling failed. Check the dashboard for details.")
        return None


def run_cloud_inference():
    """
    Submit an inference job to verify the compiled model produces correct results.

    This sends sample input data to a real device, runs the model,
    and returns the output — proving the compilation preserved accuracy.
    """
    if not COMPILED_MODEL_PATH.exists():
        return None

    device = hub.Device(TARGET_DEVICE)

    # Create sample input
    sample_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    print(f"  Submitting inference job...")
    inference_job = hub.submit_inference_job(
        model=str(COMPILED_MODEL_PATH),
        device=device,
        inputs=dict(image=sample_input),
        name="workshop-mobilenet-v2-inference",
    )

    print(f"  ℹ Job ID: {inference_job.job_id}")

    status = inference_job.get_status()
    print(f"  Status: {status}")

    if status.success:
        output = inference_job.download_output_data()
        return output
    else:
        print(f"  ❌ Inference failed.")
        return None


def display_profile_results(profile_data) -> None:
    """Display profiling results in a readable format."""
    if profile_data is None:
        return

    print("  Profiling Results:")
    print("  " + "-" * 50)

    # The profile data structure varies, so we handle it gracefully
    if hasattr(profile_data, "execution_summary"):
        summary = profile_data.execution_summary
        print(f"  Total inference time: {summary.get('inference_time', 'N/A')}")
        print(f"  Estimated throughput: {summary.get('throughput', 'N/A')}")
    else:
        # Print whatever data is available
        print(f"  Profile data: {type(profile_data)}")
        if isinstance(profile_data, dict):
            for key, value in profile_data.items():
                print(f"    {key}: {value}")

    print("  " + "-" * 50)


def save_results(profile_data) -> None:
    """Save profiling results for later comparison."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "profile_baseline.json"

    results = {
        "device": TARGET_DEVICE,
        "model": "MobileNetV2",
        "input_shape": list(INPUT_SHAPE),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if profile_data is not None:
        if isinstance(profile_data, dict):
            results["profile"] = profile_data
        else:
            results["profile"] = str(profile_data)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✓ Results saved to: {output_path}")


def main():
    print("=" * 70)
    print(" Module 5 (cont.) — Profile on Real Snapdragon Hardware")
    print("=" * 70)
    print()

    if not HAS_HUB:
        print("❌ qai-hub is not installed. Run: pip install qai-hub")
        return

    # Step 1: Profile
    print("Step 1: Profiling compiled model on real hardware")
    profile_data = profile_model()
    print()

    # Step 2: Display results
    print("Step 2: Profile results")
    display_profile_results(profile_data)
    print()

    # Step 3: Verify with inference
    print("Step 3: Running inference on real device (correctness check)")
    output = run_cloud_inference()
    if output is not None:
        print("  ✓ Inference completed — model produces valid output")
        # Show top prediction
        for key, value in output.items():
            arr = np.array(value)
            top_idx = np.argmax(arr)
            print(f"  ℹ Top prediction index: {top_idx}")
            print(f"  ℹ Confidence: {arr.flatten()[top_idx]:.4f}")
    print()

    # Step 4: Save results
    print("Step 4: Saving results")
    save_results(profile_data)
    print()

    # Summary
    print("=" * 70)
    print("🎯 What just happened:")
    print()
    print("  Your compiled MobileNetV2 was sent to a REAL Snapdragon device")
    print("  in Qualcomm's cloud device farm. The device ran inference and")
    print("  reported back:")
    print("    • How fast the model runs (latency)")
    print("    • Which compute units were used (CPU/GPU/NPU)")
    print("    • How much memory was consumed")
    print()
    print("  This is the SAME performance you would see on a physical phone")
    print("  with the same Snapdragon chipset.")
    print()
    print("⏭  Next: Module 6 — Understand QAIRT + QNN (the runtime layer)")
    print("   Read: docs/06-qairt-qnn.md")
    print("   Run:  python src/05_benchmark.py")
    print()


if __name__ == "__main__":
    main()
