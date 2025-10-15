"""
安装配置文件
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 读取requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="wheel-ocr",
    version="1.0.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="汽车轮毂字母识别系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wheel-ocr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "wheel-ocr=cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
)
