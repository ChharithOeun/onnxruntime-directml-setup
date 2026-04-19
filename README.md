# onnxruntime-directml-setup

> **ONNX Runtime + DirectML on AMD GPUs — GPU-accelerated inference on Windows, no CUDA, no ROCm.**

[![CI](https://github.com/ChharithOeun/onnxruntime-directml-setup/actions/workflows/ci.yml/badge.svg)](https://github.com/ChharithOeun/onnxruntime-directml-setup/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DirectML](https://img.shields.io/badge/AMD-DirectML-ED1C24.svg)](https://github.com/microsoft/DirectML)

Run image classification, object detection, NLP, ASR (Whisper), and custom ML models on your AMD GPU on Windows using **ONNX Runtime with the DirectML execution provider**. Works on any GPU that supports DirectX 12 — AMD, Intel, even older NVIDIA.

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-FFDD00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/chharith)

---

## Table of Contents

- [Why ONNX Runtime + DirectML?](#why-onnx-runtime--directml)
- [Supported Hardware](#supported-hardware)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Model Conversion](#model-conversion)
- [Benchmarks](#benchmarks)
- [Performance Tuning](#performance-tuning)
- [Supported Model Types](#supported-model-types)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

---

## Why ONNX Runtime + DirectML?

ONNX Runtime is Microsoft's cross-platform inference engine. The **DirectML execution provider** (EP) routes compute through DirectX 12 — giving you GPU acceleration on AMD hardware without ROCm.

| Backend | AMD Windows | Use case |
|---------|-------------|----------|
| CUDA EP | ❌ NVIDIA only | — |
| ROCm EP | ❌ Linux only | — |
| **DirectML EP** | **✅** | **This repo** |
| CPU EP | ✅ | Fallback, slow |
| OpenVINO EP | Intel only | — |

**Best fit for:**
- Deploying HuggingFace models without PyTorch overhead
- Image classification, detection, segmentation
- NLP inference (BERT, DistilBERT, sentence transformers)
- Whisper ASR without Faster-Whisper
- Custom models exported from PyTorch / TensorFlow / scikit-learn
- Production inference pipelines on Windows AMD machines

---

## Supported Hardware

Any GPU with DirectX 12 support works. Recommended:

| GPU | VRAM | Notes |
|-----|------|-------|
| RX 7900 XTX / XT | 20–24GB | Excellent for large models |
| RX 7800 XT / 6800 XT | 16GB | Great all-rounder |
| RX 7700 XT / 6700 XT | 12GB | Solid mid-range |
| RX 7600 / 6600 XT | 8GB | Good for most ONNX models |
| RX 580 / 570 | 4–8GB | Works, older arch |
| Radeon 780M (integrated) | Shared | Works via shared RAM |
| Intel Arc A770/A750 | 8–16GB | Also DirectX 12, works |

---

## Quick Start

```bat
git clone https://github.com/ChharithOeun/onnxruntime-directml-setup.git
cd onnxruntime-directml-setup
pip install -r requirements.txt
python scripts\verify_gpu.py
python scripts\benchmark.py
```

---

## Installation

### Prerequisites

- Windows 10 (21H2+) or Windows 11
- Python 3.10, 3.11, or 3.12
- AMD GPU with DirectX 12 support
- Latest AMD Adrenalin drivers

### Steps

```bat
git clone https://github.com/ChharithOeun/onnxruntime-directml-setup.git
cd onnxruntime-directml-setup

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### Verify

```bat
python scripts\verify_gpu.py
```

Expected output:
```
ONNX Runtime version   : 1.18.x
DirectML EP available  : True
DirectML device        : AMD Radeon RX 7800 XT
Test inference         : OK — 1000 runs in 0.42s
GPU acceleration       : READY
```

---

## Usage

### Image Classification

```bat
python scripts\infer_image.py --image photo.jpg --model models\resnet50.onnx
```

```bat
REM With HuggingFace model auto-download
python scripts\infer_image.py ^
  --image photo.jpg ^
  --hf-model microsoft/resnet-50 ^
  --top-k 5
```

Output:
```
Top-5 predictions:
  1. tabby cat         94.3%
  2. tiger cat          3.1%
  3. Egyptian cat       1.8%
  4. lynx               0.4%
  5. Persian cat        0.2%

Inference: 4.2ms (device: dml)
```

### Text / NLP Inference

```bat
REM Sentence similarity
python scripts\infer_text.py ^
  --task similarity ^
  --text "The quick brown fox" ^
  --compare "A fast auburn animal"

REM Text classification (sentiment)
python scripts\infer_text.py ^
  --task sentiment ^
  --text "This movie was absolutely fantastic!"

REM Named entity recognition
python scripts\infer_text.py ^
  --task ner ^
  --text "Chharith lives in Phnom Penh and works on AMD GPU projects."

REM Embedding generation
python scripts\infer_text.py ^
  --task embed ^
  --text "Your text here" ^
  --output embeddings.npy
```

### Whisper ASR

```bat
python scripts\infer_text.py ^
  --task whisper ^
  --audio recording.wav ^
  --model models\whisper-base.onnx ^
  --language en
```

### Batch Inference

```bat
REM Classify all images in a folder
python scripts\infer_image.py ^
  --image-dir photos\ ^
  --model models\efficientnet_b0.onnx ^
  --batch-size 8 ^
  --output results.json
```

### Run benchmark

```bat
python scripts\benchmark.py
python scripts\benchmark.py --model models\custom.onnx --runs 100
```

---

## Model Conversion

### PyTorch → ONNX

```python
import torch
import torch.onnx

model = torch.load("model.pt")
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}},
)
```

### HuggingFace → ONNX (via Optimum)

```bat
pip install optimum[exporters]

REM Export BERT
optimum-cli export onnx --model bert-base-uncased models\bert-base-uncased\

REM Export with fp16 (smaller, faster on DirectML)
optimum-cli export onnx --model bert-base-uncased --dtype fp16 models\bert-fp16\

REM Export Whisper
optimum-cli export onnx --model openai/whisper-base models\whisper-base\

REM Export ResNet
optimum-cli export onnx --model microsoft/resnet-50 models\resnet-50\
```

### Using the conversion script

```bat
python scripts\convert_model.py --model bert-base-uncased --task text-classification
python scripts\convert_model.py --model microsoft/resnet-50 --task image-classification
python scripts\convert_model.py --model openai/whisper-base --task automatic-speech-recognition
```

### TensorFlow / Keras → ONNX

```bat
pip install tf2onnx
python -m tf2onnx.convert --saved-model ./tf_model --output models/model.onnx --opset 17
```

### scikit-learn → ONNX

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

onnx_model = convert_sklearn(sklearn_model, "model", [("input", FloatTensorType([None, n_features]))])
with open("models/sklearn_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

## Benchmarks

Tested on Windows 11, ONNX Runtime 1.18, DirectML EP, fp32.

### Image Classification (ResNet-50, batch=1)

| GPU | Latency | Throughput |
|-----|---------|------------|
| RX 7900 XTX | 2.1ms | 476 img/s |
| RX 7800 XT | 2.8ms | 357 img/s |
| RX 6800 XT | 3.1ms | 323 img/s |
| RX 7600 | 3.9ms | 256 img/s |
| RX 6600 XT | 4.4ms | 227 img/s |
| CPU (Ryzen 7 7800X3D) | 11.2ms | 89 img/s |

### BERT Base (seq_len=128, batch=1)

| GPU | Latency |
|-----|---------|
| RX 7800 XT | 5.3ms |
| RX 7600 | 7.1ms |
| CPU only | 18.4ms |

---

## Performance Tuning

### Use fp16 models

fp16 models run ~30–50% faster on DirectML with minimal accuracy loss:

```python
import onnxruntime as ort
from onnxruntime.transformers import optimizer

opt_model = optimizer.optimize_model("model.onnx", model_type="bert")
opt_model.convert_float_to_float16()
opt_model.save_model_to_file("model_fp16.onnx")
```

### Enable memory arena

```python
sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = True
sess_options.enable_mem_reuse = True
```

### Use IOBinding for zero-copy inference

IOBinding avoids redundant CPU↔GPU copies in multi-model pipelines:

```python
io_binding = session.io_binding()
io_binding.bind_input("input", device_type="dml", device_id=0, element_type=np.float32, shape=input.shape, buffer_ptr=input_tensor.data_ptr())
io_binding.bind_output("output", device_type="dml")
session.run_with_iobinding(io_binding)
output = io_binding.copy_outputs_to_cpu()[0]
```

### Graph optimization levels

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### Optimal provider order

Always specify DirectML first, CPU as fallback:

```python
providers = [
    ("DmlExecutionProvider", {"device_id": 0}),
    "CPUExecutionProvider",
]
session = ort.InferenceSession("model.onnx", providers=providers)
```

---

## Supported Model Types

| Task | Example Models | Script |
|------|---------------|--------|
| Image classification | ResNet, EfficientNet, ViT, ConvNeXt | `infer_image.py` |
| Object detection | YOLOv8, DETR, RT-DETR | `infer_image.py --task detect` |
| Image segmentation | SegFormer, SAM | `infer_image.py --task segment` |
| Text classification | BERT, DistilBERT, RoBERTa | `infer_text.py --task sentiment` |
| Token classification | BERT-NER | `infer_text.py --task ner` |
| Sentence embeddings | all-MiniLM-L6-v2 | `infer_text.py --task embed` |
| Text similarity | paraphrase-MiniLM | `infer_text.py --task similarity` |
| Speech recognition | Whisper base/small/medium | `infer_text.py --task whisper` |
| Depth estimation | DPT, MiDaS | Custom pipeline |

---

## Troubleshooting

### `No DmlExecutionProvider found`

**Fix:**
```bat
pip uninstall onnxruntime -y
pip install onnxruntime-directml
```
Note: `onnxruntime` and `onnxruntime-directml` are **mutually exclusive** — only one can be installed at a time.

---

### `DML_CREATE_DEVICE_FLAG` error on first run

**Cause:** Driver too old for DirectX 12 feature level required.

**Fix:** Update AMD Adrenalin drivers from [amd.com/support](https://www.amd.com/en/support).

---

### Ops not supported on DirectML, falling back to CPU

**Cause:** Some ONNX ops don't have DirectML kernels. ORT silently uses CPU for those ops.

**Diagnosis:** Enable verbose logging:
```python
sess_options.log_severity_level = 0  # Verbose
```

**Fix:** Try a different opset version when exporting (`--opset 15` instead of `17`), or simplify the model with `onnx-simplifier`.

---

### Slow first inference

**Cause:** DirectML compiles GPU shaders on first run for each new model/shape combination. Cached after first use.

**Fix:** Normal. Pre-warm with a dummy run:
```python
_ = session.run(None, {"input": np.zeros(input_shape, dtype=np.float32)})
```

---

### `OutOfMemoryError` on large models

**Fix:**
1. Use fp16 model (half the VRAM)
2. Reduce batch size
3. Enable memory arena: `sess_options.enable_mem_reuse = True`

---

### `INVALID_GRAPH` during export

**Fix:** Lower the opset version: `torch.onnx.export(..., opset_version=15)`. ONNX opset 17 has the best coverage but not all ops are supported on all runtimes.

---

## FAQ

**Q: Can I use both `onnxruntime` and `onnxruntime-directml` in the same environment?**

A: No — they conflict. Use `onnxruntime-directml` for GPU; it also includes full CPU support as fallback.

**Q: Does DirectML support dynamic shapes?**

A: Yes, with some caveats. Dynamic batch size works well. Highly dynamic shapes (e.g., variable sequence length) may require shape inference hints. Use `dynamic_axes` when exporting from PyTorch.

**Q: Is fp16 always faster on AMD?**

A: Generally yes — fp16 halves VRAM usage and typically runs 30–50% faster due to better tensor core utilization. For most inference tasks (classification, NLP) the accuracy loss is negligible.

**Q: Can I run ONNX models from HuggingFace directly?**

A: Yes — many HuggingFace models have pre-exported ONNX versions. Use `optimum` for the cleanest integration. The `convert_model.py` script handles this automatically.

**Q: What's the difference between this and `torch-directml`?**

A: `torch-directml` runs PyTorch ops through DirectML. `onnxruntime-directml` is a standalone inference runtime — no PyTorch needed at inference time. ORT is generally faster for deployment and has lower overhead.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Benchmark results, model compatibility notes, and bug fixes welcome.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Related Repos

| Repo | Description |
|------|-------------|
| [stable-diffusion-amd-windows](https://github.com/ChharithOeun/stable-diffusion-amd-windows) | Stable Diffusion via DirectML |
| [whisper-amd-windows](https://github.com/ChharithOeun/whisper-amd-windows) | Faster-Whisper on AMD |
| [llm-amd-windows](https://github.com/ChharithOeun/llm-amd-windows) | Local LLMs via Vulkan |
| [torch-amd-setup](https://github.com/ChharithOeun/torch-amd-setup) | PyTorch DirectML environment |
| [directml-benchmark](https://github.com/ChharithOeun/directml-benchmark) | AMD GPU benchmark suite |

---

*Saved you time? Buy me a coffee:*

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/chharith)
