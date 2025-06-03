"""
Setup script for ESRGAN Super Resolution
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="esrgan-super-resolution",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ESRGAN Super Resolution - 4x Image Enhancement using custom GAN model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ESRGAN-SuperRes",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "esrgan-enhance=inference:main",
            "esrgan-train=train:main",
        ],
    },
)
