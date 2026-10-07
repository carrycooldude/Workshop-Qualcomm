# Instructor Guide

## Workshop Overview

| | |
|---|---|
| **Workshop** | Qualcomm Edge AI Workshop |
| **Duration** | 2.5–3 hours |
| **Audience** | Developers with basic Python, Git, and ML knowledge |
| **Format** | Hands-on, self-paced with instructor support |

---

## Learning Objectives

By the end of this workshop, participants will be able to:

1. Explain the Qualcomm Edge AI software stack (QAIRT → QNN → Hardware)
2. Use Qualcomm AI Hub to compile, profile, and deploy models
3. Export PyTorch models to ONNX for Qualcomm deployment
4. Benchmark and compare CPU vs GPU vs NPU performance
5. Apply quantization to optimize models for the Hexagon NPU
6. Use GenieX for on-device LLM inference (if Snapdragon hardware available)

---

## Suggested Timing

| Time | Module | Instructor Action |
|---|---|---|
| 0:00 – 0:10 | **Module 0: Overview** | Present architecture diagram, explain the journey |
| 0:10 – 0:25 | **Module 1: Edge AI Fundamentals** | Quick verbal explanation, emphasize CPU/GPU/NPU |
| 0:25 – 0:45 | **Module 2: Setup** | Walk through setup, help with issues |
| 0:45 – 1:00 | **Module 3: First Model** | Everyone runs local inference — first "wow" moment |
| 1:00 – 1:15 | **Module 4: Model Formats** | Brief explanation, let participants run export |
| 1:15 – 1:45 | **Module 5: AI Hub** | **KEY MODULE** — demo compilation, show dashboard |
| 1:45 – 2:00 | *Break* | 15-minute break |
| 2:00 – 2:20 | **Module 6: QAIRT + QNN** | Conceptual — use diagrams, explain the stack |
| 2:20 – 2:40 | **Module 7: GenieX** | Demo if Snapdragon available; show code if not |
| 2:40 – 2:55 | **Module 9: Benchmarking** | Run benchmarks, discuss results |
| 2:55 – 3:15 | **Module 10: Optimization** | Explain quantization, show the tradeoff |
| 3:15 – 3:30 | **Module 11: Wrap-up** | Demo the final pipeline, Q&A |

---

## What to Explain Verbally

### Module 1 (Edge AI)
- Use a smartphone as a prop — "This chip has a CPU, GPU, AND a dedicated AI chip"
- Draw the analogy: CPU = Swiss Army knife, NPU = assembly line
- Mention real-world examples: Face ID, voice assistant, camera AI

### Module 5 (AI Hub)
- Show the AI Hub dashboard on a projector
- Walk through a live compilation job
- Emphasize: "You're running on REAL hardware in Qualcomm's cloud"
- Show the per-layer compute unit breakdown

### Module 6 (QAIRT + QNN)
- Draw the stack on a whiteboard: App → QAIRT → QNN → Backend → Hardware
- Emphasize that most developers use framework delegates, not raw QNN

### Module 10 (Optimization)
- Show the quantization visual: FP32 → INT8
- Ask: "Would you accept 0.7% accuracy loss for 10× speed?"

---

## Demo Checkpoints

These are moments where you should verify participants are on track:

| Checkpoint | Expected State |
|---|---|
| After Module 2 | `python scripts/verify_setup.py` — all green |
| After Module 3 | Participants see top-5 predictions |
| After Module 4 | `models/exported/mobilenet_v2.onnx` exists |
| After Module 5 | Compilation job succeeded on AI Hub |
| After Module 9 | Benchmark results displayed |

---

## Common Participant Failures

| Problem | Frequency | Solution |
|---|---|---|
| Python version mismatch | Very common | Help them install Python 3.10–3.12 |
| AI Hub token issues | Common | Walk them through the sign-up process |
| `pip install` failures | Common | Check internet connectivity, try CPU-only PyTorch |
| Compilation timeout | Occasional | Queue is busy — try a different device target |
| Confused by QAIRT vs QNN | Very common | Use the car analogy (QAIRT = car, QNN = engine) |
| No Snapdragon hardware | Expected | Reassure them — AI Hub cloud is "real" deployment |

---

## How to Help Participants

1. **Don't fix it for them** — guide them to the answer
2. **Check the basics first**: Is the venv activated? Is the right Python version?
3. **Use the verify script**: `python scripts/verify_setup.py`
4. **Check the docs**: Each module has a "Common Problems" section
5. **Pair struggling participants** with those who finished early

---

## Backup Strategy

If AI Hub is down or rate-limited:

1. **Modules 0–4** work entirely offline
2. **Module 5**: Show pre-recorded screenshots of the AI Hub dashboard
3. **Module 9**: Local CPU benchmarks still work without AI Hub
4. **Module 7**: GenieX code examples work as conceptual material

Pre-download these before the workshop:
- PyTorch model weights (run Module 3 once)
- Sample image
- ImageNet labels

---

## Questions to Ask Participants

**Module 1:**
- "Who has used an AI feature on their phone? What was it?"
- "Why do you think major hardware manufacturers are building dedicated AI chips?"

**Module 3:**
- "What confidence score did you get? Was the prediction correct?"
- "What do you think happens if we resize the image differently?"

**Module 5:**
- "Why do you think the compiled model is smaller than the original?"
- "Which compute unit is being used for most operations?"

**Module 9:**
- "How much faster is the NPU compared to your laptop CPU?"
- "Would this speed matter for a camera app processing 30 frames/sec?"

**Module 10:**
- "Would you accept 1% accuracy loss for 10× speed improvement?"
- "When would you choose FP16 over INT8?"

---

## Final Demo Flow

For the closing demo, run through the entire pipeline in sequence:

```bash
# Show the journey in ~5 minutes
python src/01_local_inference.py     # "Here's our model on CPU"
python src/02_export_model.py        # "Now it's in ONNX"
# Show the AI Hub dashboard           # "Compiled for Snapdragon"
python src/05_benchmark.py           # "Look at the speed difference"
python src/06_optimize.py            # "INT8 makes it even faster"
```

Emphasize the narrative:
1. "We started with a Python model running at 40ms on CPU"
2. "We exported it, compiled it for Snapdragon..."
3. "Now it runs at 1.5ms on the NPU — 25× faster"
4. "And we did all of this with about 100 lines of Python"

---

## Post-Workshop

- Share the repository link for self-study
- Point to the Next Steps section in Module 11
- Encourage participants to try different models from AI Hub
- Mention the Qualcomm developer community resources
