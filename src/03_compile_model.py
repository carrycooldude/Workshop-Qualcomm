"""
============================================================================
 Module 5 — Qualcomm AI Hub: Compile & Profile (src/03_compile_model.py)
============================================================================
 What this script does:
   1. Connects to Qualcomm AI Hub using your API token
   2. Lists available target devices
   3. Submits a compilation job for MobileNetV2
   4. Waits for compilation to complete
   5. Downloads the compiled model artifact

 Prerequisites:
   1. Qualcomm AI Hub account: https://aihub.qualcomm.com
   2. API token configured:
        qai-hub configure --api_token YOUR_API_TOKEN

 Run:
   python src/03_compile_model.py

 This script runs ENTIRELY in the cloud — no Snapdragon hardware needed yet.
 AI Hub compiles and tests your model on real devices in their device farm.
============================================================================
"""

import os
import sys
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

try:
    import torch
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_SHAPE = (1, 3, 224, 224)
COMPILED_DIR = Path("models/compiled")

# Target device — change this to match your hardware or preferred target.
# Common options:
#   "Samsung Galaxy S24 (Family)"     — Snapdragon 8 Gen 3
#   "Samsung Galaxy S25 (Family)"     — Snapdragon 8 Elite
#   "QCS6490 (Proxy)"                 — Qualcomm Robotics RB3 Gen 2
#   "Snapdragon X Elite CRD"          — Snapdragon X Elite dev kit
#
# List all: qai-hub list-devices
TARGET_DEVICE = "Samsung Galaxy S24 (Family)"


def check_hub_configured() -> bool:
    """Verify that qai-hub is installed and configured."""
    if not HAS_HUB:
        print("  ❌ qai-hub is not installed.")
        print("     Fix: pip install qai-hub")
        return False

    try:
        # This will fail if no API token is configured
        devices = hub.get_devices()
        print(f"  ✓ Connected to Qualcomm AI Hub")
        print(f"  ℹ {len(devices)} target devices available")
        return True
    except Exception as e:
        print(f"  ❌ AI Hub connection failed: {e}")
        print("     Fix: qai-hub configure --api_token YOUR_API_TOKEN")
        print("     Get your token at: https://aihub.qualcomm.com")
        return False


def list_sample_devices() -> None:
    """Show a few example target devices."""
    devices = hub.get_devices()
    print("  Sample target devices:")
    for d in devices[:10]:
        print(f"    • {d.name}")
    if len(devices) > 10:
        print(f"    ... and {len(devices) - 10} more")
    print(f"  ℹ Full list: qai-hub list-devices")


def compile_model() -> None:
    """
    Submit a compilation job to Qualcomm AI Hub.

    What happens behind the scenes:
      1. Your PyTorch model is uploaded to AI Hub
      2. AI Hub converts it to an optimized format for the target device
      3. The model is compiled for the specific Snapdragon chipset
      4. Optimizations include:
         - Operator fusion
         - Memory layout optimization
         - Quantization (if specified)
         - Backend selection (CPU/GPU/NPU)
      5. The result is a compiled artifact (TFLite or QNN DLC)
    """
    print("  Loading MobileNetV2...")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    # Create example input for tracing
    example_input = torch.randn(INPUT_SHAPE)

    # Export the model using torch.export (preferred by AI Hub)
    print("  Exporting model with torch.export...")
    with torch.no_grad():
        try:
            exported_model = torch.export.export(model, (example_input,))
        except Exception:
            # Fallback: use traced model
            print("  ℹ torch.export unavailable, using TorchScript trace...")
            exported_model = torch.jit.trace(model, example_input)

    # Select the target device
    print(f"  Target device: {TARGET_DEVICE}")
    device = hub.Device(TARGET_DEVICE)

    # Submit the compilation job
    print("  Submitting compilation job to AI Hub...")
    print("  ⏳ This may take 2–5 minutes (the model is being compiled")
    print("     on Qualcomm's cloud infrastructure)...")
    print()

    compile_job = hub.submit_compile_job(
        model=exported_model,
        device=device,
        input_specs=dict(image=INPUT_SHAPE),
        name="workshop-mobilenet-v2",
    )

    print(f"  ℹ Job ID: {compile_job.job_id}")
    print(f"  ℹ Dashboard: https://app.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
    print()

    # Wait for completion
    status = compile_job.get_status()
    print(f"  Status: {status}")

    if status.success:
        # Download the compiled model
        COMPILED_DIR.mkdir(parents=True, exist_ok=True)
        target_model = compile_job.get_target_model()
        output_path = COMPILED_DIR / "mobilenet_v2_compiled.tflite"
        target_model.download(str(output_path))

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Compiled model saved to: {output_path}")
        print(f"  ℹ Compiled size: {file_size:.1f} MB")
    else:
        print(f"  ❌ Compilation failed.")
        print(f"     Check the dashboard for details:")
        print(f"     https://app.aihub.qualcomm.com/jobs/{compile_job.job_id}/")


def main():
    print("=" * 70)
    print(" Module 5 — Qualcomm AI Hub: Compile for Snapdragon")
    print("=" * 70)
    print()

    # Step 1: Verify AI Hub connection
    print("Step 1: Checking Qualcomm AI Hub connection")
    if not check_hub_configured():
        print()
        print("=" * 70)
        print("⚠  AI Hub is not configured. To complete this module:")
        print()
        print("  1. Go to https://aihub.qualcomm.com")
        print("  2. Create a free Qualcomm ID")
        print("  3. Go to Settings → API Token")
        print("  4. Run: qai-hub configure --api_token YOUR_TOKEN")
        print("  5. Re-run this script")
        print()
        print("  You can still read the code and continue to Module 6")
        print("  (QAIRT/QNN concepts) while waiting for access.")
        print("=" * 70)
        return
    print()

    # Step 2: Show available devices
    print("Step 2: Available target devices")
    list_sample_devices()
    print()

    # Step 3: Compile
    print("Step 3: Compiling MobileNetV2 for Snapdragon")
    compile_model()
    print()

    # Summary
    print("=" * 70)
    print("🎯 What just happened:")
    print()
    print("  PyTorch Model")
    print("       │")
    print("       ▼")
    print("  ┌─────────────────────┐")
    print("  │  Qualcomm AI Hub    │  ← Cloud service")
    print("  │  ┌───────────────┐  │")
    print("  │  │ Model Compile │  │  ← Converts to target format")
    print("  │  │   Optimize    │  │  ← Fuses ops, optimizes memory")
    print("  │  │   Quantize    │  │  ← Reduces precision (optional)")
    print("  │  └───────────────┘  │")
    print("  └─────────────────────┘")
    print("       │")
    print("       ▼")
    print("  Compiled Model (.tflite / .dlc)")
    print("       │")
    print("       ▼")
    print("  Ready for Snapdragon NPU")
    print()
    print("⏭  Next: Profile the compiled model on real hardware")
    print("   Run: python src/04_profile_model.py")
    print()


if __name__ == "__main__":
    main()
