# Changelog

All notable changes will be documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Features

- Initial release: ONNX Runtime DirectML execution provider setup for AMD GPUs
- `verify_gpu.py` — DirectML EP detection + inference smoke test
- `benchmark.py` — DML vs CPU throughput comparison with synthetic MatMul model
- `infer_image.py` — image classification with ImageNet top-K, batch mode, HuggingFace auto-export
- `infer_text.py` — sentiment, NER, embeddings, cosine similarity, Whisper ASR
- `convert_model.py` — HuggingFace → ONNX export via optimum with preset models
- fp16 export support for ~2× VRAM savings
- IOBinding documentation and performance tuning guide
- Supported model table: ResNet, EfficientNet, BERT, DistilBERT, Whisper, MiniLM
