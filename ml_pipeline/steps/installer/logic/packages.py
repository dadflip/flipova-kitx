import importlib
import subprocess
import sys

def check_package(pkg):
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False

def install_packages(packages):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        return True, "Success"
    except Exception as e:
        return False, str(e)

def uninstall_packages(packages):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *packages])
        return True, "Success"
    except Exception as e:
        return False, str(e)
