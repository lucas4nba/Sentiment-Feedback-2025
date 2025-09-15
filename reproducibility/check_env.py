#!/usr/bin/env python3
"""
Environment check script for Sentiment Feedback analysis reproduction.
Validates Python version and required package availability.
"""

import sys
import importlib
from typing import List, Tuple, Optional

def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def check_package(package_name: str, import_name: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Check if a package is available and get its version.
    
    Args:
        package_name: Name of the package (for pip/conda)
        import_name: Name to import (if different from package_name)
    
    Returns:
        Tuple of (success, version, error_message)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version, ""
    except ImportError as e:
        return False, "", str(e)
    except Exception as e:
        return False, "", f"Unexpected error: {str(e)}"

def main():
    """Main environment check function."""
    print("=" * 60)
    print("Sentiment Feedback Analysis - Environment Check")
    print("=" * 60)
    
    # Check Python version
    python_version = get_python_version()
    print(f"Python version: {python_version}")
    
    # Check if Python version is adequate
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8+ required")
        sys.exit(1)
    else:
        print("✅ Python version OK")
    
    print("\nChecking required packages:")
    print("-" * 40)
    
    # Required packages for the analysis
    required_packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("statsmodels", "statsmodels"),
        ("linearmodels", "linearmodels"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pyarrow", "pyarrow"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
    ]
    
    # Optional packages (warn if missing but don't fail)
    optional_packages = [
        ("pytables", "tables"),
        ("arch", "arch"),
        ("cvxpy", "cvxpy"),
        ("scikit-learn", "sklearn"),
    ]
    
    failed_packages = []
    
    # Check required packages
    for package_name, import_name in required_packages:
        success, version, error = check_package(package_name, import_name)
        if success:
            print(f"✅ {package_name:<15} {version}")
        else:
            print(f"❌ {package_name:<15} MISSING - {error}")
            failed_packages.append(package_name)
    
    # Check optional packages
    print("\nChecking optional packages:")
    print("-" * 40)
    for package_name, import_name in optional_packages:
        success, version, error = check_package(package_name, import_name)
        if success:
            print(f"✅ {package_name:<15} {version}")
        else:
            print(f"⚠️  {package_name:<15} MISSING - {error}")
    
    # Summary
    print("\n" + "=" * 60)
    if failed_packages:
        print("❌ ENVIRONMENT CHECK FAILED")
        print(f"Missing required packages: {', '.join(failed_packages)}")
        print("\nTo install missing packages:")
        print("  pip install -r reproducibility/requirements.txt")
        print("\nOr create a conda environment:")
        print("  conda env create -f reproducibility/env.yml")
        sys.exit(1)
    else:
        print("✅ ENVIRONMENT CHECK PASSED")
        print("All required packages are available.")
        print("\nYou can now run the analysis:")
        print("  ./reproducibility/run.sh")
        print("  make -f reproducibility/Makefile all")
        sys.exit(0)

if __name__ == "__main__":
    main()
