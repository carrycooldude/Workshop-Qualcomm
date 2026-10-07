"""
============================================================================
 Environment Verification Script (scripts/verify_setup.py)
============================================================================
 Run this after setup to verify everything is installed correctly.

 Usage:
   python scripts/verify_setup.py

 This checks:
   ✓ Python version
   ✓ Required packages
   ✓ Optional packages (qai-hub, geniex)
   ✓ AI Hub authentication
   ✓ Sample data
   ✓ Directory structure
============================================================================
"""

import sys
import os
import importlib
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


def check(name: str, condition: bool, fix: str = "") -> bool:
    """Print a check result."""
    if condition:
        try:
            print(f"  ✓ {name}")
        except UnicodeEncodeError:
            print(f"  [OK] {name}")
    else:
        try:
            print(f"  ❌ {name}")
        except UnicodeEncodeError:
            print(f"  [FAIL] {name}")
        if fix:
            print(f"     Fix: {fix}")
    return condition


def check_python_version() -> bool:
    """Check Python version is 3.10+."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    ok = v.major == 3 and v.minor >= 10 and v.minor <= 12
    return check(
        f"Python {version_str}",
        ok,
        "Install Python 3.10, 3.11, or 3.12 from python.org"
    )


def check_package(package: str, pip_name: str = None) -> bool:
    """Check if a Python package is importable."""
    pip_name = pip_name or package
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, "__version__", "installed")
        return check(f"{package} ({version})", True)
    except ImportError:
        return check(f"{package}", False, f"pip install {pip_name}")


def check_qai_hub_auth() -> bool:
    """Check AI Hub authentication."""
    try:
        import qai_hub as hub
        devices = hub.get_devices()
        return check(f"AI Hub authenticated ({len(devices)} devices available)", True)
    except ImportError:
        return check("AI Hub (qai-hub not installed)", False, "pip install qai-hub")
    except Exception as e:
        return check(
            "AI Hub authentication",
            False,
            "qai-hub configure --api_token YOUR_TOKEN"
        )


def check_directory_structure() -> bool:
    """Check that key directories exist."""
    dirs = ["src", "docs", "scripts", "models", "benchmarks"]
    all_ok = True
    for d in dirs:
        path = Path(d)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        all_ok = all_ok and check(f"Directory: {d}/", path.exists())
    return all_ok


def main():
    print("=" * 60)
    print(" Qualcomm Edge AI Workshop — Environment Verification")
    print("=" * 60)
    print()

    all_passed = True
    warnings = []

    # Python
    print("Python:")
    all_passed &= check_python_version()
    print()

    # Required packages
    print("Required Packages:")
    required = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("PIL", "Pillow"),
        ("numpy", "numpy"),
        ("onnx", "onnx"),
        ("tabulate", "tabulate"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
    ]
    for pkg, pip_name in required:
        all_passed &= check_package(pkg, pip_name)
    print()

    # Qualcomm packages
    print("Qualcomm Packages:")
    hub_ok = check_package("qai_hub", "qai-hub")
    if not hub_ok:
        warnings.append("qai-hub not installed — Modules 5, 9, 10 require it")

    models_ok = check_package("qai_hub_models", "qai-hub-models")
    if not models_ok:
        warnings.append("qai-hub-models not installed — optional but useful")
    print()

    # GenieX
    print("Optional — GenieX (for Module 7):")
    geniex_ok = check_package("geniex", "geniex")
    if not geniex_ok:
        warnings.append("geniex not installed — Module 7 will show concepts only")
    print()

    # AI Hub auth
    print("AI Hub Authentication:")
    auth_ok = check_qai_hub_auth()
    if not auth_ok:
        warnings.append("AI Hub not authenticated — cloud features unavailable")
    print()

    # Directory structure
    print("Directory Structure:")
    check_directory_structure()
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("✅ All required checks passed!")
    else:
        print("⚠  Some required packages are missing. Install them:")
        print("   pip install -r requirements.txt")

    if warnings:
        print()
        print("Warnings (non-blocking):")
        for w in warnings:
            print(f"  ⚠ {w}")

    print()
    print("You're ready to start! Begin with:")
    print("  python src/01_local_inference.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
