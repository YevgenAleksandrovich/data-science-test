"""
Setup script to check and install dependencies
Run: python setup.py
"""

import subprocess
import sys
import pkg_resources

required = {
    'numpy': '1.26.4',
    'scikit-learn': '1.3.0',
    'matplotlib': '3.7.2',
    'pillow': '10.0.0'
}

print("\n" + "="*60)
print("🔧 CHECKING DEPENDENCIES")
print("="*60)

installed = {pkg.key for pkg in pkg_resources.working_set}
missing = []

for package, version in required.items():
    if package not in installed:
        missing.append(f"{package}=={version}")
        print(f"❌ {package} not installed")
    else:
        print(f"✅ {package} installed")

if missing:
    print(f"\n📦 Installing missing packages: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    print("✅ All dependencies installed!")
else:
    print("\n✅ All dependencies are already installed!")

print("\nYou can now run:")
print("\nYou can now run:")
print("  python run_all.py")
print("  python task1_mnist_advanced/run_task1.py")
print("  python task2_animal_pipeline/run_task2.py")