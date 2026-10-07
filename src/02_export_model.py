"""
============================================================================
 Module 4 — Model Export (src/02_export_model.py)
============================================================================
 What this script does:
   1. Loads MobileNetV2 from PyTorch
   2. Exports it to ONNX format (the interchange format)
   3. Exports it using torch.export (for ExecuTorch / AI Hub)
   4. Validates the exported ONNX model
   5. Shows the model's input/output specification

 Run:
   python src/02_export_model.py

 Why we export:
   PyTorch models live in Python — but Snapdragon NPUs don't run Python.
   We need an intermediate format (ONNX) that Qualcomm's tools can read,
   optimize, and compile into NPU-native code.

   PyTorch (.pt) → ONNX (.onnx) → Qualcomm AI Hub → QNN/QAIRT → NPU
============================================================================
"""

import os
import sys
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
    import torchvision.models as models
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
except ImportError as e:
    print(f"\n❌ Missing required package: {e.name}")
    print("   Please install workshop dependencies:")
    print("     pip install -r requirements.txt")
    print("   Or install directly:")
    print(f"     pip install {e.name or 'torch torchvision'}\n")
    sys.exit(1)

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPORT_DIR = Path("models/exported")
ONNX_PATH = EXPORT_DIR / "mobilenet_v2.onnx"
PT2_PATH = EXPORT_DIR / "mobilenet_v2_exported.pt2"
INPUT_SHAPE = (1, 3, 224, 224)


def export_to_onnx(model: torch.nn.Module) -> Path:
    """
    Export a PyTorch model to ONNX format.

    ONNX (Open Neural Network Exchange) is an open standard for representing
    machine learning models. It acts as a bridge between training frameworks
    (PyTorch, TensorFlow) and inference runtimes (QAIRT, TensorRT, etc.).
    """
    print("  Exporting to ONNX...")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(INPUT_SHAPE)

    torch.onnx.export(
        model,                          # The PyTorch model
        dummy_input,                    # Example input for tracing
        str(ONNX_PATH),                 # Output file path
        export_params=True,             # Store trained weights inside the ONNX file
        opset_version=13,               # ONNX operator set version (13 is well-supported)
        do_constant_folding=True,       # Optimize: fold constant operations
        input_names=["input"],          # Name the input tensor
        output_names=["output"],        # Name the output tensor
        dynamic_axes={                  # Allow variable batch size
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    file_size = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"  ✓ ONNX model saved to: {ONNX_PATH}")
    print(f"  ℹ File size: {file_size:.1f} MB")
    return ONNX_PATH


def validate_onnx(onnx_path: Path) -> None:
    """Validate the exported ONNX model is well-formed."""
    if not HAS_ONNX:
        print("  ⚠ onnx package not installed — skipping validation")
        return

    print("  Validating ONNX model...")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("  ✓ ONNX model is valid!")

    # Print model info
    print(f"  ℹ IR version: {model.ir_version}")
    print(f"  ℹ Opset version: {model.opset_import[0].version}")
    print(f"  ℹ Number of nodes: {len(model.graph.node)}")

    # Input info
    for inp in model.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"  ℹ Input '{inp.name}': shape={dims}")

    # Output info
    for out in model.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  ℹ Output '{out.name}': shape={dims}")


def export_with_torch_export(model: torch.nn.Module) -> None:
    """
    Export using torch.export (PyTorch 2.x).

    torch.export produces a serialized ExportedProgram that Qualcomm AI Hub
    can accept directly — no ONNX conversion needed. This is the modern
    path that works best with ExecuTorch and the AI Hub Python API.
    """
    print("  Exporting with torch.export (PyTorch 2.x)...")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(INPUT_SHAPE)

    try:
        with torch.no_grad():
            exported = torch.export.export(model, (dummy_input,))

        torch.export.save(exported, str(PT2_PATH))
        file_size = PT2_PATH.stat().st_size / (1024 * 1024)
        print(f"  ✓ torch.export model saved to: {PT2_PATH}")
        print(f"  ℹ File size: {file_size:.1f} MB")
    except Exception as e:
        print(f"  ⚠ torch.export failed (requires PyTorch 2.4+): {e}")
        print(f"  ℹ This is OK — ONNX export is the primary path for this workshop.")


def main():
    print("=" * 70)
    print(" Module 4 — Model Export to ONNX and torch.export")
    print("=" * 70)
    print()

    # Step 1: Load model
    print("Step 1: Loading MobileNetV2")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    print(f"  ✓ Model loaded")
    print()

    # Step 2: Export to ONNX
    print("Step 2: Exporting to ONNX")
    onnx_path = export_to_onnx(model)
    print()

    # Step 3: Validate ONNX
    print("Step 3: Validating ONNX model")
    validate_onnx(onnx_path)
    print()

    # Step 4: Export with torch.export
    print("Step 4: Exporting with torch.export (optional)")
    export_with_torch_export(model)
    print()

    # Summary
    print("=" * 70)
    print("🎯 What just happened:")
    print()
    print("  PyTorch Model (.pt)")
    print("       │")
    print("       ├──→ ONNX (.onnx)          — Universal interchange format")
    print("       │                              Used by: QAIRT, TensorRT, ONNX Runtime")
    print("       │")
    print("       └──→ torch.export (.pt2)    — PyTorch-native export")
    print("                                      Used by: AI Hub, ExecuTorch")
    print()
    print("  Both formats can be submitted to Qualcomm AI Hub for compilation.")
    print("  ONNX is the most widely supported path for Qualcomm tools.")
    print()
    print("⏭  Next: Module 5 — Compile and profile with Qualcomm AI Hub")
    print("   Run: python src/03_compile_model.py")
    print()


if __name__ == "__main__":
    main()
