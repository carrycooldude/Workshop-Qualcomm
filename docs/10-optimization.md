# Module 10 — Optimization

| | |
|---|---|
| **Duration** | 20 minutes |
| **Goal** | Optimize the model for maximum edge performance |
| **Prerequisites** | Module 9 completed |

---

## 🎯 Goal

Take your working model and make it faster, smaller, and more power-efficient using real optimization techniques.

---

## Concepts

### The Optimization Loop

```mermaid
graph LR
    A["📊 Baseline"] --> B["📏 Benchmark"]
    B --> C["🔧 Optimize"]
    C --> D["📏 Benchmark Again"]
    D --> E{"Better?"}
    E -->|Yes| F["✅ Ship It"]
    E -->|No| C
```

**Golden Rule:** Never optimize without measuring first. Never assume an optimization helped — verify with benchmarks.

### Optimization Levers

| Lever | Impact | Effort | Risk |
|---|---|---|---|
| **Model choice** | 🟢🟢🟢 Huge | Low | Low — pick a smaller model |
| **Quantization** | 🟢🟢🟢 Huge | Low | Medium — some accuracy loss |
| **Backend selection** | 🟢🟢 Large | Low | Low — just configuration |
| **Input resolution** | 🟢🟢 Large | Low | Medium — affects accuracy |
| **Operator fusion** | 🟢 Moderate | None (automatic) | None |
| **Batch size** | 🟡 Variable | Low | Low |

---

## Step 1: Run the Optimization Script

```bash
python src/06_optimize.py
```

**Expected output:**
```
======================================================================
 Module 10 — Optimization: Making It Faster
======================================================================

Step 1: Comparing model architectures
------------------------------------------------------------
          MobileNetV2: 3.5M params, 13.4 MB, Top-1=71.9%, CPU=42.3ms
      MobileNetV3-Small: 2.5M params,  9.9 MB, Top-1=67.7%, CPU=28.1ms
       EfficientNet-B0: 5.3M params, 20.5 MB, Top-1=77.7%, CPU=58.9ms

Step 2: Understanding quantization
──────────────────────────────────────────────────────
  FP32 → FP16 → INT8 → INT4
  (decreasing precision, increasing speed)

Step 3: Compiling with INT8 quantization
  Submitting INT8 quantized compilation...
  ✓ INT8 model saved: models/compiled/mobilenet_v2_int8.tflite (3.5 MB)
```

---

## Step 2: Understand the Architecture Tradeoff

| Model | Params | Size | Accuracy | Speed |
|---|---|---|---|---|
| **EfficientNet-B0** | 5.3M | 20.5 MB | **77.7%** ⬆️ | 58.9 ms (slowest) |
| **MobileNetV2** | 3.5M | 13.4 MB | 71.9% | 42.3 ms |
| **MobileNetV3-Small** | 2.5M | 9.9 MB | 67.7% | **28.1 ms** ⬆️ (fastest) |

**Insight:** If your application can tolerate 4% lower accuracy, MobileNetV3-Small is 33% faster than MobileNetV2 — without any quantization.

---

## Step 3: Understand Quantization

Quantization converts model weights and activations from floating-point to integers:

```
┌─────────────────────────────────────────────────────────┐
│                    Quantization                          │
│                                                          │
│  FP32 weight: 0.4827634                                  │
│  INT8 weight: 123 (with scale=0.003922, zero_point=0)   │
│                                                          │
│  FP32 × FP32 = FP32    (32-bit multiply)                │
│  INT8 × INT8 = INT32   (8-bit multiply, 4× faster!)     │
│                                                          │
│  The NPU has thousands of INT8 multiply-accumulate       │
│  units — quantization unlocks their full power.          │
└─────────────────────────────────────────────────────────┘
```

### Expected Optimization Results

| Configuration | Size | Latency (NPU) | Accuracy |
|---|---|---|---|
| MobileNetV2 FP32 | 13.4 MB | ~5 ms | 71.9% |
| MobileNetV2 FP16 | 6.7 MB | ~3 ms | 71.8% |
| MobileNetV2 INT8 | **3.4 MB** | **~1.5 ms** ⚡ | 71.2% |
| MobileNetV3-Small INT8 | **2.5 MB** | **~1.0 ms** ⚡⚡ | 66.9% |

---

## Step 4: Optimization Strategies Summary

### Strategy 1: Choose the Right Model

```python
# Instead of MobileNetV2, try MobileNetV3-Small
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
```

### Strategy 2: Quantize (FP32 → INT8)

```python
# Via AI Hub — just add calibration data
compile_job = hub.submit_compile_job(
    model=exported_model,
    device=hub.Device("Samsung Galaxy S24 (Family)"),
    input_specs=dict(image=(1, 3, 224, 224)),
    calibration_data=calibration_dataset,  # <-- enables INT8 quantization
)
```

### Strategy 3: Select the Right Backend

```python
# Target specific compute unit via compile options
compile_job = hub.submit_compile_job(
    model=exported_model,
    device=device,
    options="--target_runtime qnn_dlc",  # Direct QNN for maximum NPU control
)
```

### Strategy 4: Reduce Input Resolution

```python
# 224×224 → 192×192 (14% fewer pixels, ~20% faster)
INPUT_SHAPE = (1, 3, 192, 192)
```

---

## Common Problems

### Problem: INT8 quantized model has much lower accuracy

**Cause:** Poor calibration data or model is sensitive to quantization.

**Fix:**
1. Use **real representative data** for calibration (not random noise)
2. Try **FP16** instead of INT8 (less aggressive, less accuracy loss)
3. Use **mixed precision** — some layers in FP16, others in INT8

### Problem: Quantized compilation fails

**Cause:** Some operators don't support INT8 execution.

**Fix:** AI Hub will automatically keep those operators in FP16. Check the compilation report for "quantization fallback" warnings.

---

## Checkpoint

- [x] You've compared different model architectures
- [x] You understand quantization (FP32 → INT8)
- [x] You know the optimization loop: baseline → benchmark → optimize → verify
- [x] You understand the accuracy-speed tradeoff

---

## ⏭️ Next Step

Continue to **[Module 11 — Final Project](11-final-project.md)**
