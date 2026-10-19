# GpuTelem

A small GPU telemetry harness for a research project on the compute cost of
text-to-speech. It instruments the **dots.tts** model and records per-run GPU
statistics (hardware/runtime fingerprint, timing, memory, utilization) so that
runs can be compared and reproduced.

The harness currently drives a toy flow-matching model (`toy_model.py`) as a
stand-in for dots.tts. The run-record format below is the same either way.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install torch nvidia-ml-py
```

`nvidia-ml-py` provides the `pynvml` module. Both it and a GPU are optional:
on a CPU-only machine every module still runs and fills GPU fields with `"n/a"`.

## Running a sweep

Run from the repo root (the `-m` form is required so package imports resolve):

```
python -m telemetry.run_sweep
```

For each NFE value this times the model, measures peak GPU memory and samples
GPU utilization/power/clock/temperature, then writes one run-record per NFE
plus a CSV summary to `outputs/`.

| Flag | Default | Meaning |
|---|---|---|
| `--nfe N [N ...]` | `4 8 16 32 64` | NFE values to sweep |
| `--warmups` | `3` | untimed runs before measuring each NFE |
| `--repeats` | `10` | minimum timed runs per NFE |
| `--min-duration-s` | `1.0` | keep repeating until the timed section lasts at least this long; `0` disables |
| `--sample-interval-ms` | `50.0` | GPU sampling period |

A timed section ends once **both** `--repeats` runs are done **and**
`--min-duration-s` has passed, so fast configurations run more times than slow
ones. The minimum duration exists so the GPU sampler sees enough of the run to
be meaningful (NVML's own utilization window is up to 1 s).

Each module also has a small demo: `python -m telemetry.<module>`.

## Modules

All live in `telemetry/`.

| Module | What it does |
|---|---|
| `fingerprint.py` | `get_fingerprint()`: Python/torch/CUDA versions, GPU name, total memory, driver version |
| `record.py` | `RunRecord`: one run's config + fingerprint + metrics, saved atomically as JSON |
| `timing.py` | `time_callable()`: warm-ups, then timed repeats (CUDA events on GPU, `perf_counter` on CPU) |
| `memory.py` | `reset()` / `read_peak()`: peak allocated and reserved CUDA memory |
| `nvml_sampler.py` | `GpuSampler`: background thread polling NVML; summarizes mean/peak per metric |
| `run_sweep.py` | the pipeline: runs everything above for each NFE and writes the outputs |
| `toy_model.py` | stand-in model: an Euler flow-matching loop whose cost scales with NFE |

What each measurement covers, per NFE:

- **memory**: warm-ups and timed repeats. One-time allocations are real memory the run needs.
- **timing** and **gpu**: timed repeats only. Warm-ups (clock ramp-up, one-time init) would skew means.

## Output files

Both go to `outputs/` at the repo root, which is git-ignored.

- `run_<run_id>.json`: one run-record per NFE (schema below)
- `sweep_<sweep_id>.csv`: one row per NFE, flattened from the run-records

## Run-record schema (`schema_version: 1`)

Example (from a CPU-only machine, so GPU fields are `"n/a"`):

```json
{
  "run_id": "20261018_220406_3bd9",
  "created_at": "2026-10-18T22:04:06-05:00",
  "schema_version": 1,
  "config": {
    "model": "toy",
    "nfe": 4,
    "sweep_id": "20261018_220406"
  },
  "fingerprint": {
    "python_version": "3.14.3",
    "torch_version": "2.14.1+cpu",
    "torch_cuda_version": "n/a",
    "device": "cpu",
    "gpu_name": "n/a",
    "gpu_total_memory_mb": "n/a",
    "driver_version": "n/a"
  },
  "metrics": {
    "timing": {
      "steps": 4,
      "median_ms": 47.81229980289936,
      "min_ms": 44.33709988370538,
      "max_ms": 50.7171000353992,
      "per_step_ms": 11.95307495072484,
      "device": "cpu",
      "timer": "perf_counter",
      "warmups": 3,
      "repeats": 3,
      "min_repeats": 3,
      "min_duration_s": 0.0,
      "measured_s": 0.14291389985010028
    },
    "memory": {
      "peak_allocated_mb": "n/a",
      "peak_reserved_mb": "n/a"
    },
    "gpu": {
      "util_mean_pct": "n/a",
      "util_peak_pct": "n/a",
      "power_mean_w": "n/a",
      "power_peak_w": "n/a",
      "sm_clock_mean_mhz": "n/a",
      "sm_clock_peak_mhz": "n/a",
      "temp_mean_c": "n/a",
      "temp_peak_c": "n/a",
      "num_samples": 0,
      "interval_ms": 50.0
    }
  }
}
```

### Conventions

- **Missing values are the string `"n/a"`**, never `null` or `0`. This happens
  when there is no GPU, or the GPU doesn't report a metric (e.g. power on some
  laptop GPUs). pandas `read_csv` treats `"n/a"` as NaN by default.
- **Units are in the key name**: `_ms`, `_s`, `_mb` (MiB, 2^20 bytes), `_pct`, `_w`, `_mhz`, `_c`.
- **Times are fixed US Eastern Standard Time (UTC-05:00)**, no daylight
  saving, so timestamps never jump or repeat.
- Stored values are full precision; round only for display.

### Top level

| Field | Type | Meaning |
|---|---|---|
| `run_id` | str | `YYYYMMDD_HHMMSS_xxxx` (creation time + 4 random hex chars); also the file name |
| `created_at` | str | ISO 8601 with offset, e.g. `2026-10-18T22:04:06-05:00` |
| `schema_version` | int | version of this format |
| `config` | object | what was run |
| `fingerprint` | object | where it ran |
| `metrics` | object | what was measured |

### `config`

| Field | Type | Meaning |
|---|---|---|
| `model` | str | model name (`"toy"` for now) |
| `nfe` | int | number of function evaluations (sampler steps) |
| `sweep_id` | str | `YYYYMMDD_HHMMSS` of the sweep this run belongs to; shared by all its runs |

### `fingerprint`

| Field | Type | Meaning |
|---|---|---|
| `python_version` | str | e.g. `3.14.3` |
| `torch_version` | str | e.g. `2.14.1+cu130` |
| `torch_cuda_version` | str | CUDA version torch was *built* against |
| `device` | str | `cuda:<index>` or `cpu` |
| `gpu_name` | str | e.g. `NVIDIA GeForce RTX 4090` |
| `gpu_total_memory_mb` | int | total device memory |
| `driver_version` | str | NVIDIA driver version, read via NVML |

### `metrics.timing`

| Field | Type | Meaning |
|---|---|---|
| `steps` | int | steps per run (equals `config.nfe`) |
| `median_ms` | float | median wall time of **one** run |
| `min_ms`, `max_ms` | float | fastest and slowest single run |
| `per_step_ms` | float | `median_ms / steps` |
| `device` | str | `cuda` or `cpu` |
| `timer` | str | `cuda_event` (GPU) or `perf_counter` (CPU) |
| `warmups` | int | untimed runs before measuring |
| `repeats` | int | timed runs that **actually ran**; the sample size behind the stats above |
| `min_repeats` | int | requested minimum (`--repeats`) |
| `min_duration_s` | float | requested minimum timed duration |
| `measured_s` | float | wall time of the whole timed section, including gaps between runs |

`repeats` varies between NFEs when `min_duration_s` is set. That does not
change what `median_ms` means: every sample is one complete run of the same
work, so the median is time per run regardless of how many samples there are;
more repeats only make it more precise. `measured_s` is not
`median_ms * repeats`. It includes per-run overhead and is the window the GPU
sampler observed.

### `metrics.memory`

| Field | Type | Meaning |
|---|---|---|
| `peak_allocated_mb` | float | high-water mark of memory held by live tensors; what the model needs |
| `peak_reserved_mb` | float | high-water mark of memory PyTorch's caching allocator held from the driver (≥ allocated; closer to `nvidia-smi` and what determines OOM) |

Peak reserved starts from whatever the allocator had cached before the run, so
it can carry over between NFEs in one sweep. Peak allocated does not.

### `metrics.gpu`

Summary of NVML samples taken every `interval_ms` during the timed repeats.

| Field | Type | Meaning |
|---|---|---|
| `util_mean_pct`, `util_peak_pct` | float, int | % of time at least one kernel was running (NVML definition; not SM occupancy) |
| `power_mean_w`, `power_peak_w` | float | board power draw |
| `sm_clock_mean_mhz`, `sm_clock_peak_mhz` | float, int | SM clock |
| `temp_mean_c`, `temp_peak_c` | float, int | GPU core temperature |
| `num_samples` | int | samples taken; `0` with no GPU. Check this before trusting the means |
| `interval_ms` | float | sampling period |

## Sweep CSV columns

One row per NFE, in this order:

`nfe, median_ms, per_step_ms, min_ms, max_ms, repeats, measured_s,
peak_allocated_mb, peak_reserved_mb, util_mean_pct, util_peak_pct,
power_mean_w, power_peak_w, sm_clock_mean_mhz, sm_clock_peak_mhz, temp_mean_c,
temp_peak_c, gpu_num_samples, run_id`

Each column has the same meaning as the run-record field of the same name;
`gpu_num_samples` is `metrics.gpu.num_samples`. Use `run_id` to find the full
JSON record for a row.
