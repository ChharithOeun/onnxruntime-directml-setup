# Contributing

Contributions welcome — benchmark results, model compatibility reports, and bug fixes.

## Most wanted

- **Benchmark numbers** on GPUs not in the table
- **Model compatibility** — which ONNX models work/fail on AMD DirectML
- **Op coverage gaps** — ops that fall back to CPU unexpectedly
- **New inference scripts** — detection, segmentation, depth estimation

## How to contribute

1. Fork, branch, change, test, PR.
2. For benchmarks: run `python scripts/benchmark.py --device both` and open an issue with your GPU + results.

## Bug reports

Include: GPU, driver version, Python version, onnxruntime-directml version, full traceback, exact command.
