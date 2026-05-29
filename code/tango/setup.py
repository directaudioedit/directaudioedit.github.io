import os
from setuptools import setup, find_packages

def get_requirements():
    req_path = "requirements.txt"
    return [line.strip() for line in open(req_path) if line.strip() and not line.startswith("#")] if os.path.exists(req_path) else []

setup(
    name="tango",
    version="0.1.0",
    packages=find_packages(include=["audioldm", "audioldm_eval", "mustango", "tango2", "tools"]),  # 包含所有核心子包
    install_requires=get_requirements(),
    python_requires=">=3.8",
    description="Audio generation project with Tango/AudioLDM",
)