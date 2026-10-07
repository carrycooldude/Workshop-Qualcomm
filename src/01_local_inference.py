"""
============================================================================
 Module 3 — Run the Model Locally (src/01_local_inference.py)
============================================================================
 What this script does:
   1. Downloads a pre-trained MobileNetV2 model from PyTorch Hub
   2. Loads a sample image (or downloads one)
   3. Preprocesses the image using standard ImageNet transforms
   4. Runs inference on CPU
   5. Prints the top-5 predictions with confidence scores

 Run:
   python src/01_local_inference.py

 Expected output:
   Top-5 Predictions:
     1. golden retriever        — 96.23%
     2. Labrador retriever      — 1.87%
     3. ...

 This is your BASELINE — we will optimize this for Snapdragon later.
============================================================================
"""

import os
import sys
import json
import urllib.request
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
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from PIL import Image
except ImportError as e:
    print(f"\n❌ Missing required package: {e.name}")
    print("   Please install workshop dependencies:")
    print("     pip install -r requirements.txt")
    print("   Or install directly:")
    print(f"     pip install {e.name or 'torch torchvision Pillow'}\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg"
SAMPLE_IMAGE_PATH = Path("models/sample_image.jpg")
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGENET_LABELS_PATH = Path("models/imagenet_classes.txt")


def download_file(url: str, dest: Path, description: str) -> None:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        print(f"  ✓ {description} already exists at {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇ Downloading {description}...")
    urllib.request.urlretrieve(url, str(dest))
    print(f"  ✓ Saved to {dest}")


def load_and_preprocess_image(image_path: Path) -> torch.Tensor:
    """
    Load an image and apply standard ImageNet preprocessing.

    ImageNet models expect:
      - 224×224 pixel images
      - Normalized with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      - Batch dimension: [1, 3, 224, 224]
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    return input_batch


def run_inference(model: torch.nn.Module, input_batch: torch.Tensor) -> torch.Tensor:
    """Run inference and return probabilities."""
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities


def load_labels(labels_path: Path) -> list[str]:
    """Load ImageNet class labels."""
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def main():
    print("=" * 70)
    print(" Module 3 — Local Inference with MobileNetV2")
    print("=" * 70)
    print()

    # Step 1: Download sample image and labels
    print("Step 1: Preparing sample data")
    download_file(SAMPLE_IMAGE_URL, SAMPLE_IMAGE_PATH, "sample image")
    download_file(IMAGENET_LABELS_URL, IMAGENET_LABELS_PATH, "ImageNet labels")
    print()

    # Step 2: Load the pre-trained model
    print("Step 2: Loading MobileNetV2 (pre-trained on ImageNet)")
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    print(f"  ✓ Model loaded successfully")
    print(f"  ℹ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  ℹ Model size: ~{sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6:.1f} MB")
    print()

    # Step 3: Preprocess the image
    print("Step 3: Preprocessing image")
    input_batch = load_and_preprocess_image(SAMPLE_IMAGE_PATH)
    print(f"  ✓ Input shape: {list(input_batch.shape)}")
    print(f"  ℹ Shape meaning: [batch=1, channels=3, height=224, width=224]")
    print()

    # Step 4: Run inference
    print("Step 4: Running inference on CPU")
    import time
    start = time.perf_counter()
    probabilities = run_inference(model, input_batch)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  ✓ Inference completed in {elapsed:.1f} ms")
    print()

    # Step 5: Display results
    labels = load_labels(IMAGENET_LABELS_PATH)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    print("Step 5: Results")
    print("-" * 50)
    print(f"  {'Rank':<6} {'Class':<30} {'Confidence':>10}")
    print(f"  {'—'*4:<6} {'—'*28:<30} {'—'*8:>10}")
    for i in range(5):
        idx = top5_idx[i].item()
        prob = top5_prob[i].item() * 100
        print(f"  {i+1:<6} {labels[idx]:<30} {prob:>9.2f}%")
    print("-" * 50)
    print()

    # Summary
    print("🎯 What just happened:")
    print("  1. We loaded a pre-trained MobileNetV2 model (~3.4M parameters)")
    print("  2. Preprocessed an image to 224×224 with ImageNet normalization")
    print("  3. Ran inference on CPU and got classification probabilities")
    print()
    print("⏭  Next: Module 4 — Understand model formats (ONNX export)")
    print("   Run: python src/02_export_model.py")
    print()


if __name__ == "__main__":
    main()
