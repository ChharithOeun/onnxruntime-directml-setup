# Installation Guide

## Requirements

- Windows 10 (21H2+) or Windows 11
- Python 3.10, 3.11, or 3.12
- AMD GPU with DirectX 12 support
- AMD Adrenalin drivers (recent version)

## Steps

```bat
git clone https://github.com/ChharithOeun/onnxruntime-directml-setup.git
cd onnxruntime-directml-setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts\verify_gpu.py
```

## Important: onnxruntime vs onnxruntime-directml

These two packages are **mutually exclusive**. If you have plain `onnxruntime` installed:

```bat
pip uninstall onnxruntime -y
pip install onnxruntime-directml
```

`onnxruntime-directml` includes full CPU execution provider support as fallback — you don't lose anything by switching.

## Optional: model export tools

To convert HuggingFace models:

```bat
pip install optimum[exporters]
```

## Verify

```bat
python scripts\verify_gpu.py
```

Look for `DirectML EP available: True`.
