"""
SynthWhisperer setup script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="synthwhisperer",
    version="1.0.0",
    author="SynthWhisperer Team", 
    author_email="",
    description="AI assistant that translates natural language into synthesizer parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/synthwhisperer",  # Update with actual repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "synthwhisperer": [
            "synth-config/*.json",
            "*.json",
            "*.jsonl"
        ],
    },
    entry_points={
        "console_scripts": [
            "synthwhisperer=demo:main",
        ],
    },
    keywords="synthesizer, AI, natural language processing, music, sound design, audio",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/synthwhisperer/issues",  # Update with actual repo URL
        "Source": "https://github.com/yourusername/synthwhisperer",  # Update with actual repo URL
        "Documentation": "https://github.com/yourusername/synthwhisperer#readme",  # Update with actual repo URL
    },
)