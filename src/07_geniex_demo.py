"""
============================================================================
 Module 7 — GenieX: On-Device LLM Inference (src/07_geniex_demo.py)
============================================================================
 What this script does:
   1. Explains LLM inference concepts (prefill, decode, KV cache)
   2. If GenieX is available: runs a real LLM on Snapdragon NPU
   3. If not: shows the code and explains what would happen

 Requirements:
   - Snapdragon-powered device (Windows ARM64, Android, or Linux ARM64)
   - GenieX installed: pip install geniex
   - OR: GenieX CLI installed

 Run:
   python src/07_geniex_demo.py

 GenieX supports two backends:
   • llama.cpp — runs GGUF models from Hugging Face (easier to start)
   • QAIRT    — runs pre-compiled AI Hub bundles (maximum NPU performance)
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
    from geniex import AutoModelForCausalLM
    HAS_GENIEX = True
except ImportError:
    HAS_GENIEX = False


def explain_llm_inference():
    """
    Explain how LLM inference works — fundamentally different from
    image classification.
    """
    print("  How LLM Inference Works")
    print("  " + "═" * 50)
    print()
    print("  Image Classification (MobileNetV2):")
    print("    Input → [One Forward Pass] → Output")
    print("    • Single pass, fixed compute, fixed time")
    print()
    print("  LLM Text Generation:")
    print("    Prompt → [Prefill] → [Decode] → [Decode] → ... → [Done]")
    print("    • Two phases, variable length, autoregressive")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │ Phase 1: PREFILL                                │")
    print("  │                                                 │")
    print("  │ Process the entire input prompt at once.        │")
    print("  │ All input tokens are processed in parallel.     │")
    print("  │ Builds the KV (Key-Value) cache.                │")
    print("  │                                                 │")
    print("  │ Metric: Time to First Token (TTFT)              │")
    print("  │ Bottleneck: Compute (lots of matrix math)       │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │ Phase 2: DECODE (repeated for each token)       │")
    print("  │                                                 │")
    print("  │ Generate one new token at a time.               │")
    print("  │ Each step reads from the KV cache.              │")
    print("  │ New KV entries are appended to the cache.        │")
    print("  │                                                 │")
    print("  │ Metric: Tokens per second (tok/s)               │")
    print("  │ Bottleneck: Memory bandwidth (loading weights)  │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │ KV CACHE                                        │")
    print("  │                                                 │")
    print("  │ Stores intermediate attention computations.     │")
    print("  │ Grows with each generated token.                │")
    print("  │ Without it, we'd recompute all attention        │")
    print("  │ from scratch for every single token.            │")
    print("  │                                                 │")
    print("  │ Memory: context_length × num_layers × head_dim  │")
    print("  │ For a 2B model with 2K context: ~200–500 MB     │")
    print("  └─────────────────────────────────────────────────┘")
    print()
    print("  Why Edge LLMs need special treatment:")
    print("    • Models are 1–8 GB even when quantized")
    print("    • KV cache consumes hundreds of MB of RAM")
    print("    • Memory bandwidth is the bottleneck, not compute")
    print("    • INT4 quantization is essential to fit in memory")
    print("    • NPU acceleration provides massive power savings")


def run_geniex_demo():
    """
    Run a real LLM using GenieX.

    GenieX abstracts away the complexity of:
      - Model loading and memory management
      - KV cache allocation
      - Token encoding/decoding
      - Hardware backend selection (CPU/GPU/NPU)
      - Quantization-aware execution
    """
    # Using a small, fast model for the demo
    MODEL_ID = "unsloth/Qwen3-0.6B-GGUF"
    PRECISION = "Q4_0"

    print(f"  Loading model: {MODEL_ID}")
    print(f"  Precision: {PRECISION} (4-bit quantized)")
    print(f"  ⏳ First load downloads the model (~400 MB)...")
    print()

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, precision=PRECISION)

    # Test prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is edge AI and why does it matter? Answer in 2-3 sentences."},
    ]

    print("  Generating response...")
    print("  " + "─" * 50)

    start = time.perf_counter()
    response = model.generate(messages)
    elapsed = time.perf_counter() - start

    print(f"  {response}")
    print("  " + "─" * 50)
    print()

    # Estimate metrics
    # Token count is approximate — GenieX may not expose this directly
    approx_tokens = len(response.split()) * 1.3  # rough approximation
    print(f"  ⏱ Total time: {elapsed:.2f}s")
    print(f"  ℹ Approx tokens: ~{int(approx_tokens)}")
    print(f"  ℹ Approx speed: ~{approx_tokens/elapsed:.1f} tok/s")


def show_geniex_code():
    """Show what the GenieX code looks like even if it can't be run."""
    print()
    print("  GenieX Python API Example:")
    print("  " + "─" * 50)
    print()
    print("  from geniex import AutoModelForCausalLM")
    print()
    print("  # Load model — GGUF from Hugging Face")
    print("  model = AutoModelForCausalLM.from_pretrained(")
    print("      'unsloth/Qwen3-0.6B-GGUF',")
    print("      precision='Q4_0',  # 4-bit quantization")
    print("  )")
    print()
    print("  # Generate text")
    print("  messages = [")
    print("      {'role': 'user', 'content': 'What is edge AI?'},")
    print("  ]")
    print("  response = model.generate(messages)")
    print("  print(response)")
    print()
    print("  " + "─" * 50)
    print()
    print("  GenieX CLI Example:")
    print("  " + "─" * 50)
    print()
    print("  # Interactive chat")
    print("  geniex infer unsloth/Qwen3-0.6B-GGUF")
    print()
    print("  # Using AI Hub optimized bundle (best NPU performance)")
    print("  geniex infer ai-hub-models/Qwen3-4B")
    print()
    print("  # Start an OpenAI-compatible local server")
    print("  geniex pull ai-hub-models/Qwen3-4B")
    print("  geniex serve")
    print("  # Server runs at http://127.0.0.1:18181")
    print("  # Use with: curl, LangChain, any OpenAI-compatible client")
    print()
    print("  " + "─" * 50)


def main():
    print("=" * 70)
    print(" Module 7 — GenieX: On-Device LLM Inference")
    print("=" * 70)
    print()

    # Step 1: Explain LLM inference
    print("Step 1: Understanding LLM Inference")
    print("-" * 60)
    explain_llm_inference()
    print()

    # Step 2: Show code / run demo
    if HAS_GENIEX:
        print("Step 2: Running LLM with GenieX")
        print("-" * 60)
        print("  ✓ GenieX is installed!")
        run_geniex_demo()
    else:
        print("Step 2: GenieX Code Reference")
        print("-" * 60)
        print("  ℹ GenieX is not installed on this system.")
        print("  ℹ GenieX requires a Snapdragon-powered device:")
        print("    • Windows on Snapdragon (ARM64)")
        print("    • Android device with Snapdragon")
        print("    • Linux ARM64 on Snapdragon")
        print()
        print("  Install: pip install geniex")
        print("  CLI:     curl -fsSL https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install.sh | sh")
        show_geniex_code()
    print()

    # Architecture diagram
    print("Step 3: GenieX Architecture")
    print("-" * 60)
    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │                Your Application                  │")
    print("  ├──────────────────────────────────────────────────┤")
    print("  │              GenieX Runtime                      │")
    print("  │  ┌────────────────┐  ┌────────────────────┐     │")
    print("  │  │  llama.cpp     │  │  QAIRT Plugin       │     │")
    print("  │  │  (GGUF models) │  │  (AI Hub bundles)   │     │")
    print("  │  └────────┬───────┘  └────────┬───────────┘     │")
    print("  ├───────────┼───────────────────┼─────────────────┤")
    print("  │           ▼                   ▼                  │")
    print("  │     ┌──────────┐        ┌──────────┐            │")
    print("  │     │   CPU    │        │   NPU    │            │")
    print("  │     │  (Kryo)  │        │ (Hexagon)│            │")
    print("  │     └──────────┘        └──────────┘            │")
    print("  │                  Snapdragon SoC                  │")
    print("  └──────────────────────────────────────────────────┘")
    print()

    # Summary
    print("=" * 70)
    print("🎯 Key Takeaways:")
    print()
    print("  1. LLM inference has two phases: Prefill (parallel) + Decode (sequential)")
    print("  2. The KV cache is the key data structure — it grows with context length")
    print("  3. Memory bandwidth is the bottleneck, not compute")
    print("  4. INT4 quantization is essential for on-device LLMs")
    print("  5. GenieX provides a simple API that handles all the complexity")
    print("  6. Two backends: llama.cpp (easy) and QAIRT (fast)")
    print()
    print("⏭  Next: Module 11 — Final Project")
    print("   Read: docs/11-final-project.md")
    print()


if __name__ == "__main__":
    main()
