"""
infer_image.py — GPU-accelerated image inference via ONNX Runtime DirectML.

Supports image classification and object detection.

Usage:
    python scripts/infer_image.py --image photo.jpg --hf-model microsoft/resnet-50
    python scripts/infer_image.py --image photo.jpg --model models/resnet50.onnx
    python scripts/infer_image.py --image-dir photos/ --model models/efficientnet.onnx
    python scripts/infer_image.py --help
"""
import argparse
import json
import sys
import time
from pathlib import Path


IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)


def parse_args():
    p = argparse.ArgumentParser(description="Image inference via ONNX Runtime DirectML")
    p.add_argument("--image", default=None, help="Input image path")
    p.add_argument("--image-dir", default=None, help="Directory of images (batch mode)")
    p.add_argument("--model", default=None, help="Path to .onnx model file")
    p.add_argument("--hf-model", default=None,
                   help="HuggingFace model ID to export & run (e.g. microsoft/resnet-50)")
    p.add_argument("--task", default="classify",
                   choices=["classify", "detect"],
                   help="Task type [default: classify]")
    p.add_argument("--top-k", type=int, default=5, help="Top-K results [default: 5]")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size [default: 1]")
    p.add_argument("--output", default=None, help="Save results to JSON file")
    p.add_argument("--device", default="dml", choices=["dml", "cpu"],
                   help="Execution device [default: dml]")
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing")
    return p.parse_args()


def get_providers(device):
    import onnxruntime as ort
    if device == "dml" and "DmlExecutionProvider" in ort.get_available_providers():
        return [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def export_hf_model(model_id, output_dir):
    """Export a HuggingFace model to ONNX using optimum."""
    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print("ERROR: optimum not installed.")
        print("  Run: pip install optimum[exporters]")
        sys.exit(1)

    out_path = Path(output_dir) / model_id.replace("/", "_")
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {model_id} → {out_path}")
    main_export(model_id, output=str(out_path), task="image-classification")
    onnx_files = list(out_path.glob("*.onnx"))
    return str(onnx_files[0]) if onnx_files else None


def load_imagenet_labels():
    """Load ImageNet class labels."""
    labels_path = Path("models/imagenet_labels.txt")
    if labels_path.exists():
        return labels_path.read_text().strip().splitlines()
    try:
        import urllib.request
        data = urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=5).read().decode()
        labels_path.parent.mkdir(exist_ok=True)
        labels_path.write_text(data)
        return data.strip().splitlines()
    except Exception:
        return [f"class_{i}" for i in range(1000)]


def preprocess_image(image_path, size=(224, 224)):
    """Preprocess image for standard ImageNet models."""
    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # HWC → NCHW
    return arr


def classify_image(session, image_path, labels, top_k, args):
    import numpy as np
    import onnxruntime as ort

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = input_meta.shape

    # Determine input size from model
    h = shape[2] if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0 else 224
    w = shape[3] if len(shape) == 4 and isinstance(shape[3], int) and shape[3] > 0 else 224

    inp = preprocess_image(image_path, (w, h))

    # Warmup
    for _ in range(args.warmup):
        session.run(None, {input_name: inp})

    t0 = time.time()
    outputs = session.run(None, {input_name: inp})
    elapsed = time.time() - t0

    logits = outputs[0][0]
    # Softmax
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    top_indices = probs.argsort()[::-1][:top_k]
    results = [
        {"rank": i + 1, "label": labels[idx] if idx < len(labels) else f"class_{idx}",
         "score": float(probs[idx])}
        for i, idx in enumerate(top_indices)
    ]
    return results, elapsed


def main():
    args = parse_args()

    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("ERROR: onnxruntime-directml not installed.")
        print("  Run: pip install onnxruntime-directml")
        sys.exit(1)

    # Resolve model
    model_path = args.model
    if args.hf_model and not model_path:
        model_path = export_hf_model(args.hf_model, "models")
    if not model_path:
        print("ERROR: Specify --model or --hf-model")
        sys.exit(1)
    if not Path(model_path).exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    providers = get_providers(args.device)
    print(f"Model    : {Path(model_path).name}")
    print(f"Device   : {providers[0] if isinstance(providers[0], str) else providers[0][0]}")

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_opts, providers=providers)

    labels = load_imagenet_labels() if args.task == "classify" else []

    # Collect images
    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [f for f in Path(args.image_dir).iterdir() if f.suffix.lower() in exts]
        print(f"Images   : {len(images)} found in {args.image_dir}\n")
    elif args.image:
        images = [Path(args.image)]
    else:
        print("ERROR: Specify --image or --image-dir")
        sys.exit(1)

    all_results = {}
    for img_path in images:
        print(f"[{img_path.name}]")
        if args.task == "classify":
            results, elapsed = classify_image(session, str(img_path), labels, args.top_k, args)
            for r in results:
                bar = "█" * int(r["score"] * 40)
                print(f"  {r['rank']:>2}. {r['label']:<30} {r['score']*100:5.1f}%  {bar}")
            print(f"     Inference: {elapsed*1000:.1f}ms\n")
            all_results[str(img_path)] = results
        else:
            print("  Object detection output depends on model — run raw inference.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()
