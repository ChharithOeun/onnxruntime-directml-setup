"""
convert_model.py — Convert HuggingFace models to ONNX for DirectML inference.

Usage:
    python scripts/convert_model.py --model microsoft/resnet-50 --task image-classification
    python scripts/convert_model.py --model bert-base-uncased --task text-classification
    python scripts/convert_model.py --model openai/whisper-base --task automatic-speech-recognition
    python scripts/convert_model.py --model sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction
"""
import argparse
import sys
from pathlib import Path


TASK_ALIASES = {
    "classify": "image-classification",
    "sentiment": "text-classification",
    "ner": "token-classification",
    "embed": "feature-extraction",
    "whisper": "automatic-speech-recognition",
    "qa": "question-answering",
    "seq2seq": "text2text-generation",
}

PRESETS = {
    "resnet50": ("microsoft/resnet-50", "image-classification"),
    "efficientnet-b0": ("google/efficientnet-b0", "image-classification"),
    "bert-base": ("bert-base-uncased", "feature-extraction"),
    "distilbert-sst2": ("distilbert-base-uncased-finetuned-sst-2-english", "text-classification"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "feature-extraction"),
    "whisper-base": ("openai/whisper-base", "automatic-speech-recognition"),
    "whisper-small": ("openai/whisper-small", "automatic-speech-recognition"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Convert HuggingFace models to ONNX")
    p.add_argument("--model", default=None, help="HuggingFace model ID or preset name")
    p.add_argument("--task", default=None, help="Task type (see --list-tasks)")
    p.add_argument("--output-dir", default="models", help="Output directory [default: models]")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                   help="Export dtype [default: fp32]")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version [default: 17]")
    p.add_argument("--list", action="store_true", help="List preset models")
    p.add_argument("--list-tasks", action="store_true", help="List supported tasks")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_tasks:
        print("Supported tasks:")
        for alias, full in TASK_ALIASES.items():
            print(f"  {alias:<20} -> {full}")
        print("\nFull task names also accepted directly.")
        return

    if args.list:
        print("Preset models:")
        print(f"  {'Alias':<20} {'Model ID':<45} {'Task'}")
        print(f"  {'-'*20} {'-'*45} {'-'*30}")
        for alias, (model_id, task) in PRESETS.items():
            print(f"  {alias:<20} {model_id:<45} {task}")
        return

    if not args.model:
        print("ERROR: Specify --model or use --list for presets.")
        sys.exit(1)

    # Resolve preset
    if args.model in PRESETS:
        model_id, task = PRESETS[args.model]
        task = args.task or task
    else:
        model_id = args.model
        task = args.task
        if not task:
            print("ERROR: --task is required when not using a preset. Use --list-tasks.")
            sys.exit(1)

    # Resolve task alias
    task = TASK_ALIASES.get(task, task)

    output_path = Path(args.output_dir) / model_id.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Model      : {model_id}")
    print(f"Task       : {task}")
    print(f"Dtype      : {args.dtype}")
    print(f"Opset      : {args.opset}")
    print(f"Output     : {output_path}")
    print()

    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print("ERROR: optimum not installed.")
        print("  Run: pip install optimum[exporters]")
        sys.exit(1)

    kwargs = {
        "model_name_or_path": model_id,
        "output": str(output_path),
        "task": task,
        "opset": args.opset,
    }
    if args.dtype == "fp16":
        kwargs["dtype"] = "fp16"

    print("Exporting to ONNX (first run downloads model weights)...")
    try:
        main_export(**kwargs)
    except Exception as e:
        print(f"ERROR during export: {e}")
        print("\nTry: --opset 15 or a different --task value")
        sys.exit(1)

    onnx_files = list(output_path.glob("*.onnx"))
    print(f"\nExport complete. ONNX files:")
    for f in onnx_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}  ({size_mb:.1f} MB)")

    print(f"\nTo run inference:")
    if "image" in task:
        print(f'  python scripts\\infer_image.py --model "{onnx_files[0]}" --image photo.jpg')
    else:
        print(f'  python scripts\\infer_text.py --task sentiment --model "{output_path}"')


if __name__ == "__main__":
    main()
