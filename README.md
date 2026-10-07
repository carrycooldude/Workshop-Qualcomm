# 🚀 Qualcomm Edge AI Workshop

> **A hands-on, Google Codelab-style workshop for building, optimizing, and deploying AI models on Snapdragon® NPUs using Qualcomm AI Hub, QAIRT, QNN, and GenieX.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Qualcomm AI Hub](https://img.shields.io/badge/Qualcomm-AI%20Hub-3253DC.svg)](https://aihub.qualcomm.com)
[![Snapdragon NPU](https://img.shields.io/badge/Snapdragon-NPU%20Optimized-FF0033.svg)](https://www.qualcomm.com/products/snapdragon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Welcome to the Workshop!

Edge AI brings machine learning directly to where data originates: smartphones, laptops, IoT gadgets, and edge robotics. Instead of sending frames and prompts to remote cloud data centers—paying latency and bandwidth penalties—we run models locally on energy-efficient Neural Processing Units (NPUs).

This workshop takes you from **zero knowledge of Qualcomm silicon** to deploying and profiling optimized computer vision and generative AI models on real Snapdragon hardware.

### 🌟 What Makes This Workshop Special?
- **No Physical Snapdragon Device Required**: You can compile, profile, and validate your models on real Snapdragon devices directly through the cloud device farm in **Qualcomm AI Hub**.
- **100% Beginner Friendly**: If you know basic Python and machine learning concepts (tensors, weights, inputs), every other step is explained from first principles.
- **End-to-End Pipeline**: You won't just look at theory. You will run local inference, export models, optimize/quantize weights, compile for NPU targets, benchmark latency/memory/power, and explore generative AI on-device with **GenieX**.

---

## 🔄 End-to-End Workflow Architecture

```text
  Pre-trained Model (PyTorch / torchvision)
                      │
                      ▼
            Model Export (ONNX / torch.export)
                      │
                      ▼
              Qualcomm AI Hub (Cloud Service)
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
   Optimization               Compilation
(PTQ: FP32 → INT8)         (Graph fusion, layout)
         │                         │
         └────────────┬────────────┘
                      │
                      ▼
               QAIRT / QNN Runtime
    (Qualcomm AI Runtime & Neural Network API)
                      │
                      ▼
             Snapdragon Hardware
        (Hexagon™ NPU / HTP / GPU / CPU)
                      │
         ┌────────────┴────────────┐
         │                         │
         ▼                         ▼
On-Device Inference          Benchmarking
(Real-time latency)    (FPS, Memory, Power, Accuracy)
                      │
                      ▼
         Genie / GenieX (Generative AI)
     (On-Device LLMs: Llama, Mistral, Phi-3)
```

---

## 🗺️ Workshop Curriculum & Modules

The workshop is organized into **12 bite-sized modules** designed like Google Codelabs. Each module has a designated time, clear goals, step-by-step instructions, and runnable scripts:

| Module | Title | Duration | Hands-on Component | Description |
|:---|:---|:---:|:---|:---|
| **[Module 00](docs/00-overview.md)** | [Overview & Learning Journey](docs/00-overview.md) | 10 min | Theory & Architecture | Edge AI fundamentals, Qualcomm ecosystem overview, roadmap |
| **[Module 01](docs/01-edge-ai.md)** | [Edge AI & Qualcomm Hardware](docs/01-edge-ai.md) | 15 min | Architecture Walkthrough | CPU vs. GPU vs. NPU, Hexagon Tensor Processor (HTP), TOPS & watts |
| **[Module 02](docs/02-setup.md)** | [Environment Setup](docs/02-setup.md) | 20 min | `scripts/verify_setup.py` | Virtual environment, dependencies, AI Hub API token configuration |
| **[Module 03](docs/03-first-model.md)** | [Run the Model Locally](docs/03-first-model.md) | 15 min | `src/01_local_inference.py` | PyTorch MobileNetV2 baseline on CPU, measuring latency & confidence |
| **[Module 04](docs/04-model-formats.md)** | [Model Formats & Export](docs/04-model-formats.md) | 15 min | `src/02_export_model.py` | Exporting to ONNX and `torch.export`, input/output tensor inspection |
| **[Module 05](docs/05-ai-hub.md)** | [Qualcomm AI Hub Hands-On](docs/05-ai-hub.md) | 30 min | `src/03_compile_model.py`<br>`src/04_profile_model.py` | Compiling models, cloud device farm testing, latency/memory profiling |
| **[Module 06](docs/06-qairt-qnn.md)** | [QAIRT & QNN Deep Dive](docs/06-qairt-qnn.md) | 20 min | Architecture & Flow | QAIRT execution flow, QNN backends, graph preparation, DLC formats |
| **[Module 07](docs/07-geniex.md)** | [GenieX: On-Device LLMs](docs/07-geniex.md) | 20 min | `src/07_geniex_demo.py` | Prefill vs. decode phases, KV cache, running GGUF/QAIRT LLMs on Snapdragon |
| **[Module 08](docs/08-deployment.md)** | [Deploy to Device](docs/08-deployment.md) | 20 min | Deployment Guides | Android/ADB deployment, Windows on Snapdragon (WoS), embedded Linux |
| **[Module 09](docs/09-benchmarking.md)** | [Benchmarking & Profiling](docs/09-benchmarking.md) | 15 min | `src/05_benchmark.py` | Latency (P50/P90/P99), throughput (FPS), memory footprint, report generation |
| **[Module 10](docs/10-optimization.md)** | [Model Optimization & PTQ](docs/10-optimization.md) | 20 min | `src/06_optimize.py` | Post-Training Quantization (INT8 vs FP16), accuracy evaluation, pareto frontier |
| **[Module 11](docs/11-final-project.md)** | [Final Project & Capstone](docs/11-final-project.md) | 10 min | Project Showcase | Reviewing deliverables, certification path, official community resources |
| **[Instructor Guide](docs/instructor-guide.md)** | [Facilitator & Instructor Guide](docs/instructor-guide.md) | — | Teaching Support | Timetable, live demo checkpoints, common hurdles & troubleshooting |

---

## ⚡ 5-Minute Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/carrycooldude/Workshop-Qualcomm.git
cd Workshop-Qualcomm
```

### 2. Set Up Virtual Environment
#### On Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

#### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Your Setup
Run the built-in diagnostic tool:
```bash
python scripts/verify_setup.py
```
*(Or run `make verify`)*

### 4. Run the Baseline Model
```bash
python src/01_local_inference.py
```
*(Loads MobileNetV2, executes inference on CPU, and prints top-5 ImageNet predictions with confidence scores)*

### 5. Export to ONNX
```bash
python src/02_export_model.py
```
*(Converts PyTorch graph into ONNX format and validates tensor contracts)*

### 6. Compile & Profile with Qualcomm AI Hub
Get your free API key at [aihub.qualcomm.com](https://aihub.qualcomm.com) and configure:
```bash
qai-hub configure --api_token YOUR_API_TOKEN
python src/03_compile_model.py
python src/04_profile_model.py
```

---

## 📂 Repository Structure

```text
Workshop-Qualcomm/
├── README.md                     # You are here! Master workshop guide
├── Makefile                      # Convenient CLI targets (setup, verify, compile, etc.)
├── pyproject.toml                # Project packaging configuration
├── requirements.txt              # Pinned Python package dependencies
│
├── docs/                         # Codelab step-by-step reading modules
│   ├── 00-overview.md            # Module 00: Overview & Learning Journey
│   ├── 01-edge-ai.md             # Module 01: Edge AI & Qualcomm Hardware Concepts
│   ├── 02-setup.md               # Module 02: Environment Setup & Diagnostics
│   ├── 03-first-model.md         # Module 03: Run MobileNetV2 Locally
│   ├── 04-model-formats.md       # Module 04: ONNX & Torch.Export
│   ├── 05-ai-hub.md              # Module 05: Qualcomm AI Hub Compile & Profile
│   ├── 06-qairt-qnn.md           # Module 06: QAIRT and QNN Architecture
│   ├── 07-geniex.md              # Module 07: GenieX On-Device Generative AI
│   ├── 08-deployment.md          # Module 08: Deploy to Android & Windows on ARM
│   ├── 09-benchmarking.md        # Module 09: Latency, Throughput & Metrics
│   ├── 10-optimization.md        # Module 10: Quantization & Performance Tuning
│   ├── 11-final-project.md       # Module 11: Capstone Summary & Next Steps
│   └── instructor-guide.md       # Complete instructor facilitation guide
│
├── src/                          # Hands-on source scripts
│   ├── 01_local_inference.py     # Local CPU inference & baseline measurement
│   ├── 02_export_model.py        # PyTorch to ONNX / torch.export pipeline
│   ├── 03_compile_model.py       # Submit compilation jobs to Qualcomm AI Hub
│   ├── 04_profile_model.py       # Profile models on cloud Snapdragon devices
│   ├── 05_benchmark.py           # Benchmark comparison (CPU vs. NPU)
│   ├── 06_optimize.py            # Quantization & accuracy vs. latency trade-offs
│   └── 07_geniex_demo.py         # On-device LLM text generation demo
│
├── scripts/                      # Helper & automation utilities
│   └── verify_setup.py           # Automated environment and token checker
│
├── models/                       # Model storage directory
│   ├── README.md                 # Explanations for exported and compiled artifacts
│   ├── exported/                 # Generated ONNX / pt2 files (gitignored)
│   └── compiled/                 # Downloaded AI Hub compiled binaries (gitignored)
│
└── benchmarks/                   # Performance benchmarks and logs
    ├── README.md                 # Benchmark methodology and schema
    └── results/                  # Generated benchmark summaries (gitignored)
```

---

## 🛠️ Makefile Command Reference

If you have `make` installed, you can use these shortcuts:

| Target | Description | Equivalent Python Command |
|:---|:---|:---|
| `make setup` | Create venv & install dependencies | `python -m venv .venv && pip install -r requirements.txt` |
| `make verify` | Validate environment and dependencies | `python scripts/verify_setup.py` |
| `make run-local` | Run local PyTorch CPU inference | `python src/01_local_inference.py` |
| `make export-onnx` | Export model to ONNX format | `python src/02_export_model.py` |
| `make compile` | Submit compilation job to AI Hub | `python src/03_compile_model.py` |
| `make profile` | Profile model on cloud device | `python src/04_profile_model.py` |
| `make benchmark` | Run latency & throughput benchmarks | `python src/05_benchmark.py` |
| `make optimize` | Run quantization experiments | `python src/06_optimize.py` |
| `make geniex-demo` | Run GenieX LLM inference demo | `python src/07_geniex_demo.py` |
| `make clean` | Remove cached models and logs | `rm -rf models/compiled models/exported benchmarks/results` |

---

## 📚 Qualcomm AI Ecosystem Glossary

| Term | What It Is | Why It Matters |
|:---|:---|:---|
| **NPU (Neural Processing Unit)** | Dedicated hardware silicon specialized in matrix multiplication and tensor math. | Provides 10–50× energy efficiency compared to running neural nets on CPU. |
| **Hexagon™ NPU / HTP** | Qualcomm's proprietary NPU architecture featuring the Hexagon Tensor Processor. | Powers AI acceleration in Snapdragon mobile, compute, and automotive chipsets. |
| **Qualcomm AI Hub** | Cloud portal & developer SDK providing pre-optimized models and remote device compilation/profiling. | Allows developers to compile for any Snapdragon device without owning physical hardware. |
| **QAIRT** | **Qualcomm AI Runtime** — The top-level software stack uniting Qualcomm's AI frameworks. | Provides consistent APIs, execution delegates, and device management. |
| **QNN** | **Qualcomm Neural Network** — Low-level API and compiler infrastructure. | Generates hardware-optimized instructions and manages operator execution on HTP backends. |
| **DLC** | **Deep Learning Container** — Compiled Qualcomm binary format containing weights and network graph. | The production format deployed directly onto Snapdragon NPUs. |
| **Genie / GenieX** | Qualcomm's optimized framework and CLI for on-device Generative AI & Large Language Models. | Enables fast token generation (prefill & decode) with optimized KV caches on Snapdragon devices. |
| **PTQ** | **Post-Training Quantization** — Converting FP32 weights into INT8 using calibration data. | Shrinks model size by ~75% and dramatically increases NPU throughput with minimal accuracy loss. |

---

## 💻 Hardware Requirements & Cloud Flexibility

### Do I need physical Snapdragon hardware?
**No!** You can complete **100%** of the core curriculum using the cloud device farm in Qualcomm AI Hub.

| Step | Local Machine Only | Qualcomm AI Hub (Cloud) | Snapdragon Device (Optional) |
|:---|:---:|:---:|:---:|
| Run baseline model | ✅ | — | — |
| Export ONNX / PyTorch | ✅ | — | — |
| Compile model for NPU | — | ✅ | ✅ (using local QNN SDK) |
| Profile latency & memory | — | ✅ (Real hardware) | ✅ (On-device) |
| Run quantization | ✅ | ✅ | ✅ |
| Run GenieX LLM | Simulated / Concepts | — | ✅ (Local NPU inference) |

---

## ❓ Frequently Asked Questions & Troubleshooting

<details>
<summary><strong>1. What should I do if <code>qai-hub configure</code> fails or I don't have an API token?</strong></summary>

You can sign up for a free account at [aihub.qualcomm.com](https://aihub.qualcomm.com). Once logged in, go to **Account Settings → API Token**, copy your key, and run:
```bash
qai-hub configure --api_token <YOUR_TOKEN>
```
If you are working offline or waiting for approval, all scripts in `src/` include comprehensive offline explanations and simulations so you can keep learning without disruption.
</details>

<details>
<summary><strong>2. Can I use a model other than MobileNetV2?</strong></summary>

Yes! MobileNetV2 is used as the foundational tutorial model because it is lightweight and universally understood. You can easily modify `src/02_export_model.py` and `src/03_compile_model.py` to use:
- ResNet-50
- YOLOv8 / YOLOv11 (Object Detection)
- Segment Anything (SAM)
- Whisper (Audio Transcription)
- Llama-3.2 / Phi-3.5 (via GenieX)
</details>

<details>
<summary><strong>3. Which Python versions are supported?</strong></summary>

Python **3.10**, **3.11**, and **3.12** are supported. Python 3.10 or 3.12 is recommended for maximum compatibility with `torch` and `qai-hub`.
</details>

---

## 🤝 Contributing & License

Contributions, feedback, and pull requests are warmly welcomed!
- To report bugs or propose improvements, open an issue on GitHub.
- Released under the [MIT License](LICENSE).
