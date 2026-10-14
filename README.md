# GpuTelem

A small GPU telemetry harness for a research project on the compute cost of
text-to-speech. It instruments the **dots.tts** model and records per-run GPU
statistics (hardware/runtime fingerprint, timing, memory, utilization) so that
runs can be compared and reproduced.

## Layout

- `telemetry/` — the harness package
  - `fingerprint.py` — captures the machine's GPU/runtime environment

## Usage

```
python -m telemetry.fingerprint
```
