import importlib
import subprocess
import sys

def check_package(pkg):
    try:
        if pkg in ("tensorflow", "keras"):
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import logging
            logging.getLogger('absl').setLevel(logging.ERROR)
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False

def install_packages(packages):
    try:
        args = []
        for p in packages:
            args.extend(p.split())
        subprocess.check_call([sys.executable, "-m", "pip", "install", *args])
        return True, "Success"
    except Exception as e:
        return False, str(e)

def uninstall_packages(packages):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *packages])
        return True, "Success"
    except Exception as e:
        return False, str(e)
