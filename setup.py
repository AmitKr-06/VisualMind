"""
VisualMind — setup.py for editable installs.
Run: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="visualmind",
    version="1.0.0",
    description="Multimodal RAG pipeline for NCERT Class 10 Science",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[],   # dependencies are declared in requirements.txt
)