"""
infer_text.py — GPU-accelerated NLP inference via ONNX Runtime DirectML.

Supports: sentiment analysis, NER, embeddings, similarity, Whisper ASR.

Usage:
    python scripts/infer_text.py --task sentiment --text "This is great!"
    python scripts/infer_text.py --task embed --text "Hello world" --output emb.npy
    python scripts/infer_text.py --task whisper --audio speech.wav
"""
import argparse
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="NLP inference via ONNX Runtime DirectML")
    p.add_argument("--task", required=True,
                   choices=["sentiment", "ner", "embed", "similarity", "whisper"],
                   help="Inference task")
    p.add_argument("--text", default=None, help="Input text")
    p.add_argument("--compare", default=None, help="Second text for similarity task")
    p.add_argument("--audio", default=None, help="Audio file path (whisper task)")
    p.add_argument("--model", default=None,
                   help="Path to ONNX model directory or file (auto-downloads if omitted)")
    p.add_argument("--device", default="dml", choices=["dml", "cpu"])
    p.add_argument("--output", default=None, help="Save output to file (embeddings: .npy)")
    p.add_argument("--language", default=None, help="Language code for Whisper (e.g. en)")
    return p.parse_args()


def get_providers(device):
    import onnxruntime as ort
    if device == "dml" and "DmlExecutionProvider" in ort.get_available_providers():
        return [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def task_sentiment(args, providers):
    """Run sentiment analysis using optimum pipeline."""
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer, pipeline
    except ImportError:
        print("ERROR: optimum not installed. Run: pip install optimum[exporters] transformers")
        sys.exit(1)

    model_id = args.model or "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Model: {model_id}")
    print("Loading...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForSequenceClassification.from_pretrained(
        model_id, export=True, provider=providers[0] if isinstance(providers[0], str) else providers[0][0]
    )
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

    print(f"\nText: {args.text}")
    t0 = time.time()
    result = clf(args.text)
    elapsed = time.time() - t0

    for r in result:
        print(f"  Label : {r['label']}")
        print(f"  Score : {r['score']:.4f} ({r['score']*100:.1f}%)")
    print(f"  Time  : {elapsed*1000:.1f}ms")


def task_ner(args, providers):
    """Run named entity recognition."""
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification
        from transformers import AutoTokenizer, pipeline
    except ImportError:
        print("ERROR: optimum not installed. Run: pip install optimum[exporters] transformers")
        sys.exit(1)

    model_id = args.model or "elastic/distilbert-base-uncased-finetuned-conll03-english"
    print(f"Model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    print(f"\nText: {args.text}")
    t0 = time.time()
    entities = ner(args.text)
    elapsed = time.time() - t0

    for ent in entities:
        print(f"  [{ent['entity_group']:8}] {ent['word']:<25} score={ent['score']:.3f}")
    print(f"  Time: {elapsed*1000:.1f}ms")


def task_embed(args, providers):
    """Generate sentence embeddings."""
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        import numpy as np
    except ImportError:
        print("ERROR: optimum not installed. Run: pip install optimum[exporters] transformers")
        sys.exit(1)

    model_id = args.model or "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)

    inputs = tokenizer(args.text, return_tensors="pt", padding=True, truncation=True)

    t0 = time.time()
    outputs = model(**inputs)
    elapsed = time.time() - t0

    # Mean pool
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    print(f"\nText      : {args.text}")
    print(f"Shape     : {embedding.shape}")
    print(f"Norm      : {float(np.linalg.norm(embedding)):.4f}")
    print(f"Preview   : [{', '.join(f'{x:.4f}' for x in embedding[:6])}...]")
    print(f"Time      : {elapsed*1000:.1f}ms")

    if args.output:
        np.save(args.output, embedding)
        print(f"Saved     : {args.output}")

    return embedding


def task_similarity(args, providers):
    """Compute cosine similarity between two texts."""
    import numpy as np

    print(f"Text 1: {args.text}")
    print(f"Text 2: {args.compare or '(none)'}\n")

    if not args.compare:
        print("ERROR: Provide --compare text for similarity task")
        sys.exit(1)

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: optimum not installed. Run: pip install optimum[exporters] transformers")
        sys.exit(1)

    model_id = args.model or "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)

    def embed(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        out = model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        return emb / np.linalg.norm(emb)

    e1 = embed(args.text)
    e2 = embed(args.compare)
    sim = float(np.dot(e1, e2))

    print(f"Cosine similarity : {sim:.4f}")
    if sim > 0.8:
        print("Interpretation   : Very similar")
    elif sim > 0.5:
        print("Interpretation   : Moderately similar")
    elif sim > 0.2:
        print("Interpretation   : Somewhat related")
    else:
        print("Interpretation   : Dissimilar")


def task_whisper(args, providers):
    """Run Whisper ASR via ONNX Runtime."""
    if not args.audio:
        print("ERROR: --audio is required for whisper task")
        sys.exit(1)
    if not Path(args.audio).exists():
        print(f"ERROR: Audio file not found: {args.audio}")
        sys.exit(1)

    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import AutoProcessor, pipeline
    except ImportError:
        print("ERROR: optimum not installed. Run: pip install optimum[exporters] transformers")
        sys.exit(1)

    model_id = args.model or "openai/whisper-base"
    print(f"Model: {model_id}")
    print("Loading (first run exports to ONNX, may take a moment)...")

    processor = AutoProcessor.from_pretrained(model_id)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)

    kwargs = {}
    if args.language:
        kwargs["generate_kwargs"] = {"language": args.language}

    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, **kwargs)

    print(f"\nAudio: {args.audio}")
    t0 = time.time()
    result = asr(args.audio)
    elapsed = time.time() - t0

    print(f"Transcript: {result['text']}")
    print(f"Time      : {elapsed:.2f}s")


def main():
    args = parse_args()

    try:
        import onnxruntime as ort  # noqa: F401
    except ImportError:
        print("ERROR: onnxruntime-directml not installed.")
        print("  Run: pip install onnxruntime-directml")
        sys.exit(1)

    providers = get_providers(args.device)
    device_name = providers[0] if isinstance(providers[0], str) else providers[0][0]
    print(f"Device: {device_name}\n")

    tasks = {
        "sentiment": task_sentiment,
        "ner": task_ner,
        "embed": task_embed,
        "similarity": task_similarity,
        "whisper": task_whisper,
    }
    tasks[args.task](args, providers)


if __name__ == "__main__":
    main()
