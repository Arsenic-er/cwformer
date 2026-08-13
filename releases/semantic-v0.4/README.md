# QsoSemanticFormer v0.4 release

This public release contains the first promoted local semantic-understanding model for Game Morse Adventurer QSO traffic.

## Runtime artifact

Use `qso-semanticformer-v0.4.release.int8.onnx` for the desktop game. Keep the FP32 model as the parity/reference artifact and the `.pt` file for reproducible export only.

The INT8 model is 1,164,897 bytes, runs on CPU through ONNX Runtime and passed the full release gates on three disjoint 15,000-example confirmation seeds.

## Files

- `qso-semanticformer-v0.4.release.int8.onnx`: desktop runtime candidate.
- `qso-semanticformer-v0.4.release.fp32.onnx`: FP32 reference.
- `qso-semanticformer-v0.4.release.pt`: promoted PyTorch checkpoint.
- `qso-semanticformer-v0.4.release.fp32.runtime-contract.json`: input/output and label contract.
- `qso-semanticformer-v0.4.release.fp32.manifest.json`: export size, parity and latency report.
- `release-eval.json` and `confirm-seed-*.json`: promoted PyTorch evaluations.
- `fp32-onnx-release-eval.json`: direct FP32 ONNX evaluation.
- `int8-confirm-*.json`: direct INT8 ONNX evaluations.

The files are published in this dedicated AI repository so the game repository can consume the reviewed INT8 artifact without carrying the training pipeline.
