"""
Unit and integration tests for the Qualcomm Edge AI Workshop starter repo.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class TestWorkshopStructure(unittest.TestCase):
    """Verify that all required files and directories exist."""

    def test_directory_structure(self):
        required_dirs = ["src", "docs", "scripts", "models", "benchmarks"]
        for d in required_dirs:
            path = ROOT_DIR / d
            self.assertTrue(path.exists() and path.is_dir(), f"Missing directory: {d}")

    def test_doc_modules_exist(self):
        expected_docs = [
            "00-overview.md",
            "01-edge-ai.md",
            "02-setup.md",
            "03-first-model.md",
            "04-model-formats.md",
            "05-ai-hub.md",
            "06-qairt-qnn.md",
            "07-geniex.md",
            "08-deployment.md",
            "09-benchmarking.md",
            "10-optimization.md",
            "11-final-project.md",
            "instructor-guide.md",
        ]
        docs_dir = ROOT_DIR / "docs"
        for doc in expected_docs:
            doc_path = docs_dir / doc
            self.assertTrue(doc_path.exists(), f"Missing documentation file: {doc}")

    def test_source_scripts_exist(self):
        expected_scripts = [
            "01_local_inference.py",
            "02_export_model.py",
            "03_compile_model.py",
            "04_profile_model.py",
            "05_benchmark.py",
            "06_optimize.py",
            "07_geniex_demo.py",
        ]
        src_dir = ROOT_DIR / "src"
        for script in expected_scripts:
            script_path = src_dir / script
            self.assertTrue(script_path.exists(), f"Missing source script: {script}")

    def test_setup_script_exists(self):
        verify_script = ROOT_DIR / "scripts" / "verify_setup.py"
        self.assertTrue(verify_script.exists(), "Missing scripts/verify_setup.py")


class TestOfflineInferenceBasics(unittest.TestCase):
    """Basic sanity check on imports and data preprocessing."""

    def test_torch_importable(self):
        try:
            import torch
            import torchvision
            tensor = torch.zeros(1, 3, 224, 224)
            self.assertEqual(tensor.shape, (1, 3, 224, 224))
        except ImportError:
            self.skipTest("PyTorch/torchvision not installed in test runner environment")


if __name__ == "__main__":
    unittest.main()
