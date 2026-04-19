"""
benchmark.py — Benchmark ONNX Runtime DirectML vs CPU execution provider.

Generates a synthetic model and measures throughput to confirm GPU is active.
"""
import argparse
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark ONNX Runtime DirectML")
    p.add_argument("--model", default=None, help="Path to .onnx model (uses synthetic if omitted)")
    p.add_argument("--input-name", default=None, help="Input tensor name")
    p.add_argument("--input-shape", default=None, help="Input shape as comma-separated ints (e.g. 1,3,224,224)")
    p.add_argument("--runs", type=int, default=200, help="Number of timed runs [default: 200]")
    p.add_argument("--warmup", type=int, default=20, help="Warmup runs [default: 20]")
    p.add_argument("--device", default="dml", choices=["dml", "cpu", "both"],
                   help="Device to benchmark [default: dml]")
    return p.parse_args()


def build_synthetic_model():
    """Build a small MatMul model for benchmarking."""
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np

        W = np.random.randn(512, 512).astype(np.float32)
        w_tensor = numpy_helper.from_array(W, name="W")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 512])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 512])

        node = helper.make_node("MatMul", ["X", "W"], ["Y"])
        graph = helper.make_graph([node], "bench", [X], [Y], initializer=[w_tensor])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        import io
        buf = io.BytesIO()
        onnx.save(model, buf)
        return buf.getvalue(), "X", (1, 512)
    except ImportError:
        return None, None, None


def run_bench(session, input_name, inp, runs, warmup):
    import numpy as np

    for _ in range(warmup):
        session.run(None, {input_name: inp})

    t0 = time.time()
    for _ in range(runs):
        session.run(None, {input_name: inp})
    elapsed = time.time() - t0
    return elapsed, runs / elapsed


def main():
    args = parse_args()
    print("=== ONNX Runtime DirectML Benchmark ===\n")

    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("ERROR: onnxruntime not installed. Run: pip install onnxruntime-directml")
        sys.exit(1)

    print(f"ONNX Runtime : {ort.__version__}")
    print(f"Providers    : {ort.get_available_providers()}\n")

    # Load or build model
    if args.model:
        model_path = args.model
        if not Path(model_path).exists():
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
        model_bytes = Path(model_path).read_bytes()
        sess_tmp = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        input_name = args.input_name or sess_tmp.get_inputs()[0].name
        if args.input_shape:
            shape = tuple(int(x) for x in args.input_shape.split(","))
        else:
            shape = tuple(d if isinstance(d, int) and d > 0 else 1
                         for d in sess_tmp.get_inputs()[0].shape)
        print(f"Model        : {Path(model_path).name}")
        print(f"Input        : {input_name} {shape}")
    else:
        print("No model specified — using synthetic MatMul benchmark model")
        model_bytes, input_name, shape = build_synthetic_model()
        if model_bytes is None:
            print("ERROR: onnx package needed for synthetic model: pip install onnx")
            sys.exit(1)
        print(f"Input        : {input_name} {shape}")

    print(f"Runs         : {args.runs} (warmup: {args.warmup})\n")

    inp = np.random.randn(*shape).astype(np.float32)
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    results = {}

    if args.device in ("dml", "both"):
        if "DmlExecutionProvider" not in ort.get_available_providers():
            print("WARN: DmlExecutionProvider not available — skipping DML benchmark")
        else:
            print("Benchmarking DirectML (GPU)...")
            dml_sess = ort.InferenceSession(
                model_bytes, sess_opts,
                providers=[("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            )
            elapsed, rps = run_bench(dml_sess, input_name, inp, args.runs, args.warmup)
            results["DirectML (GPU)"] = (elapsed, rps)
            print(f"  {rps:.1f} inf/s  |  {elapsed/args.runs*1000:.2f} ms/inf\n")

    if args.device in ("cpu", "both"):
        print("Benchmarking CPU...")
        cpu_sess = ort.InferenceSession(
            model_bytes, sess_opts, providers=["CPUExecutionProvider"]
        )
        elapsed, rps = run_bench(cpu_sess, input_name, inp, args.runs, args.warmup)
        results["CPU"] = (elapsed, rps)
        print(f"  {rps:.1f} inf/s  |  {elapsed/args.runs*1000:.2f} ms/inf\n")

    print("=== Summary ===")
    print(f"{'Provider':<20} {'inf/s':>10} {'ms/inf':>10}")
    print("-" * 42)
    for name, (elapsed, rps) in results.items():
        print(f"{name:<20} {rps:>10.1f} {elapsed/args.runs*1000:>10.2f}")

    if "DirectML (GPU)" in results and "CPU" in results:
        speedup = results["DirectML (GPU)"][1] / results["CPU"][1]
        print(f"\nGPU speedup: {speedup:.2f}x vs CPU")


if __name__ == "__main__":
    main()
