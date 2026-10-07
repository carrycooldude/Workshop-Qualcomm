# Module 2 — Environment Setup

| | |
|---|---|
| **Duration** | 20 minutes |
| **Goal** | Install all required tools and verify the environment |
| **Prerequisites** | Python 3.10+, Git, internet connection |

---

## 🎯 Goal

Set up a complete development environment for the workshop. After this module, every tool will be installed and verified.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/qualcomm-edge-ai-workshop.git
cd qualcomm-edge-ai-workshop
```

**What this does:** Downloads all workshop code, scripts, and documentation to your machine.

**Expected output:**
```
Cloning into 'qualcomm-edge-ai-workshop'...
remote: Enumerating objects: ...
Receiving objects: 100% ...
```

**Verify:**
```bash
ls src/
```
You should see `01_local_inference.py`, `02_export_model.py`, etc.

---

## Step 2: Create a Python Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> **If you get an execution policy error:**
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Windows (Command Prompt)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**What this does:** Creates an isolated Python environment so workshop dependencies don't interfere with your system Python.

**Why we need it:** Different projects need different package versions. Virtual environments prevent conflicts.

**Expected output:** Your terminal prompt should now show `(.venv)`:
```
(.venv) C:\qualcomm-edge-ai-workshop>
```

**Verify:**
```bash
python --version
```
Should show Python 3.10.x, 3.11.x, or 3.12.x.

---

## Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**What this does:**
- Upgrades pip to the latest version
- Installs all required Python packages:
  - `torch` + `torchvision` — PyTorch ML framework
  - `qai-hub` — Qualcomm AI Hub client
  - `qai-hub-models` — Pre-optimized model library
  - `onnx` — Model interchange format
  - `Pillow` — Image processing
  - `numpy` — Numerical computing
  - `tabulate` — Pretty-printing tables
  - `matplotlib` — Plotting
  - `tqdm` — Progress bars
  - `requests` — HTTP client

**Expected output:**
```
Successfully installed torch-2.x.x torchvision-0.x.x qai-hub-0.x.x ...
```

**How to verify:**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import qai_hub; print('qai-hub OK')"
```

---

## Step 4: Configure Qualcomm AI Hub

### 4a: Create a Qualcomm ID

1. Go to **[https://aihub.qualcomm.com](https://aihub.qualcomm.com)**
2. Click **Sign Up** or **Log In**
3. Create a free **Qualcomm ID** (use your email)
4. Verify your email

### 4b: Get Your API Token

1. Log in to [AI Hub](https://aihub.qualcomm.com)
2. Click your **profile icon** (top-right)
3. Go to **Settings**
4. Find **API Token**
5. Click **Copy**

### 4c: Configure the CLI

```bash
qai-hub configure --api_token YOUR_API_TOKEN_HERE
```

**What this does:** Saves your API token so the `qai-hub` CLI and Python API can authenticate with Qualcomm's cloud.

**Expected output:**
```
Token saved to ~/.qai_hub/config.yaml
```

**Verify:**
```bash
qai-hub list-devices
```

**Expected output:**
```
Samsung Galaxy S24 (Family)
Samsung Galaxy S25 (Family)
QCS6490 (Proxy)
Snapdragon X Elite CRD
...
```

If you see a list of devices, you're authenticated! ✅

---

## Step 5: Install GenieX (Optional)

GenieX is needed only for Module 7 (LLM inference) and requires a **Snapdragon-powered device**.

### If you're on Windows ARM64 (Snapdragon X Elite/Plus laptop):

```bash
pip install geniex
```

### If you're on Linux ARM64:

```bash
curl -fsSL https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install.sh | sh
```

### If you're NOT on Snapdragon hardware:

Skip this step. Module 7 will show you the concepts and code without requiring GenieX.

**Verify (if installed):**
```bash
python -c "from geniex import AutoModelForCausalLM; print('GenieX OK')"
```

---

## Step 6: Verify Everything

Run the verification script:

```bash
python scripts/verify_setup.py
```

**Expected output:**
```
============================================================
 Qualcomm Edge AI Workshop — Environment Verification
============================================================

Python:
  ✓ Python 3.10.x

Required Packages:
  ✓ torch (2.x.x)
  ✓ torchvision (0.x.x)
  ✓ PIL (10.x.x)
  ✓ numpy (1.x.x)
  ✓ onnx (1.x.x)
  ✓ tabulate (0.9.x)
  ✓ matplotlib (3.x.x)
  ✓ tqdm (4.x.x)
  ✓ requests (2.x.x)

Qualcomm Packages:
  ✓ qai_hub (0.x.x)
  ✓ qai_hub_models (0.x.x)

AI Hub Authentication:
  ✓ AI Hub authenticated (XX devices available)

============================================================
✅ All required checks passed!

You're ready to start! Begin with:
  python src/01_local_inference.py
============================================================
```

---

## Common Problems

### Problem: `python` command not found

**Symptom:** `'python' is not recognized as an internal or external command`

**Cause:** Python is not in your PATH.

**Fix (Windows):** Reinstall Python from [python.org](https://python.org) and check **"Add Python to PATH"** during installation.

**Fix (Linux/Mac):** Use `python3` instead of `python`.

---

### Problem: `pip install qai-hub` fails

**Symptom:** `ERROR: No matching distribution found for qai-hub`

**Cause:** Wrong Python version or architecture.

**Fix:** qai-hub requires Python 3.10–3.12. On Windows, use 64-bit (AMD64) Python, not ARM64 Python.

**Verify:** `python -c "import struct; print(struct.calcsize('P') * 8, 'bit')"`

---

### Problem: `qai-hub configure` says "invalid token"

**Symptom:** `Error: Invalid API token`

**Cause:** Token was copied incorrectly (extra spaces, truncated).

**Fix:**
1. Go to [aihub.qualcomm.com](https://aihub.qualcomm.com) → Settings → API Token
2. Click "Regenerate" to get a fresh token
3. Copy the ENTIRE token (it's a long string)
4. Re-run: `qai-hub configure --api_token YOUR_NEW_TOKEN`

---

### Problem: `pip install torch` is very slow or fails

**Symptom:** Download takes forever or times out.

**Cause:** PyTorch packages are large (~800 MB).

**Fix:** Install CPU-only PyTorch (sufficient for this workshop):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Checkpoint

Before continuing, verify:

- [x] Virtual environment is activated (you see `(.venv)` in your prompt)
- [x] `python scripts/verify_setup.py` shows all green checkmarks
- [x] `qai-hub list-devices` returns a list of devices

---

## ⏭️ Next Step

Continue to **[Module 3 — Run the Model Locally](03-first-model.md)**
