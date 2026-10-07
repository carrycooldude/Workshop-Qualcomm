# =============================================================================
# Qualcomm Edge AI Workshop — Makefile
# =============================================================================
# Usage:
#   make setup          — Create virtual environment and install dependencies
#   make verify         — Verify the environment is ready
#   make run-local      — Run local inference with MobileNetV2
#   make export-onnx    — Export model to ONNX format
#   make compile        — Compile model via Qualcomm AI Hub
#   make profile        — Profile compiled model on target hardware
#   make benchmark      — Run benchmark comparison (CPU vs compiled)
#   make optimize       — Run optimization experiments
#   make geniex-demo    — Run GenieX LLM demo (requires Snapdragon hardware)
#   make clean          — Remove generated files
#   make help           — Show this help message
# =============================================================================

.PHONY: help setup verify run-local export-onnx compile profile benchmark optimize geniex-demo clean

PYTHON ?= python
VENV_DIR ?= .venv
PIP = $(VENV_DIR)/Scripts/pip
PYTHON_VENV = $(VENV_DIR)/Scripts/python

# Linux/macOS override
ifeq ($(OS),)
    PIP = $(VENV_DIR)/bin/pip
    PYTHON_VENV = $(VENV_DIR)/bin/python
endif

help: ## Show this help message
	@echo "============================================="
	@echo " Qualcomm Edge AI Workshop"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Create venv and install dependencies"
	@echo "  verify       - Verify environment is ready"
	@echo "  run-local    - Run local inference with MobileNetV2"
	@echo "  export-onnx  - Export model to ONNX format"
	@echo "  compile      - Compile model via Qualcomm AI Hub"
	@echo "  profile      - Profile compiled model on target"
	@echo "  benchmark    - Run benchmark comparison"
	@echo "  optimize     - Run optimization experiments"
	@echo "  geniex-demo  - Run GenieX LLM demo"
	@echo "  clean        - Remove generated files"

setup: ## Create virtual environment and install dependencies
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "✅ Setup complete! Activate your venv:"
	@echo "   Windows:  .venv\\Scripts\\activate"
	@echo "   Linux:    source .venv/bin/activate"

verify: ## Verify the environment is ready
	$(PYTHON_VENV) scripts/verify_setup.py

run-local: ## Run local inference with MobileNetV2
	$(PYTHON_VENV) src/01_local_inference.py

export-onnx: ## Export model to ONNX format
	$(PYTHON_VENV) src/02_export_model.py

compile: ## Compile model via Qualcomm AI Hub
	$(PYTHON_VENV) src/03_compile_model.py

profile: ## Profile compiled model on target hardware
	$(PYTHON_VENV) src/04_profile_model.py

benchmark: ## Run benchmark comparison
	$(PYTHON_VENV) src/05_benchmark.py

optimize: ## Run optimization experiments
	$(PYTHON_VENV) src/06_optimize.py

geniex-demo: ## Run GenieX LLM demo (Snapdragon hardware required)
	$(PYTHON_VENV) src/07_geniex_demo.py

clean: ## Remove generated files
	rm -rf models/compiled models/exported models/quantized
	rm -rf benchmarks/results
	rm -rf __pycache__ src/__pycache__
	rm -rf qnn_output qairt_output
	@echo "✅ Cleaned generated files."
