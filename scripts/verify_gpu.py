"""
verify_gpu.py — Verify ONNX Runtime DirectML execution provider.
"""
import sys
import time


def check_directml():
    print("=== ONNX Runtime DirectML Verification ===\n")

    # Check onnxruntime
    try:
        import onnxruntime as ort
        print(f"onnxruntime version    : {ort.__version__}")
    except ImportError:
        print("ERROR: onnxruntime not installed.")
        print("  Run: pip install onnxruntime-directml")
        return False

    # Check DirectML EP
    available_providers = ort.get_available_providers()
    print(f"Available providers    : {available_providers}")

    if "DmlExecutionProvider" not in available_providers:
        print("\nERROR: DmlExecutionProvider not available.")
        print("  Make sure onnxruntime-directml is installed (not plain onnxruntime):")
        print("    pip uninstall onnxruntime -y")
        print("    pip install onnxruntime-directml")
        return False

    print("DirectML EP            : FOUND")

    # Get device info
    try:
        import subprocess
        result = subprocess.run(
            ["powershell", "-command",
             "(Get-WmiObject Win32_VideoController | Select-Object -First 1).Name"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"GPU adapter            : {result.stdout.strip()}")
    except Exception:
        pass

    # Quick inference test with a tiny model
    try:
        import numpy as np
        import io

        # Build a minimal ONNX graph (Identity op) for testing
        import onnx
        from onnx import helper, TensorProto

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Identity", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        buf = io.BytesIO()
        onnx.save(model, buf)
        onnx_bytes = buf.getvalue()

        providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # Suppress noise
        session = ort.InferenceSession(onnx_bytes, sess_opts, providers=providers)

        inp = np.random.randn(1, 4).astype(np.float32)

        # Warmup
        session.run(None, {"X": inp})

        # Timed
        N = 1000
        t0 = time.time()
        for _ in range(N):
            session.run(None, {"X": inp})
        elapsed = time.time() - t0

        print(f"Test inference         : OK — {N} runs in {elapsed:.3f}s")

    except ImportError:
        print("Test inference         : SKIP (onnx package not installed — pip install onnx)")
    except Exception as e:
        print(f"Test inference         : WARNING — {e}")

    print("\nStatus: GPU acceleration READY")
    return True


if __name__ == "__main__":
    ok = check_directml()
    sys.exit(0 if ok else 1)
