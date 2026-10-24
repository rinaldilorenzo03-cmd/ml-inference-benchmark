# ML Inference Benchmark on an x86 Laptop CPU (Raspberry Pi 4 planned)

### ResNet-50 with PyTorch, ONNX Runtime and OpenVINO: FP32 vs INT8

A small, reproducible benchmark of single-stream inference latency and Top-1 accuracy for **ResNet-50** on an Intel mobile CPU (Core i5-1155G7). It compares PyTorch eager mode with two dedicated runtimes (ONNX Runtime and OpenVINO), each in FP32 and in INT8 (static post-training quantization), always at `batch_size = 1`.

The goal is to learn the inference toolchain (export, quantization, measurement methodology) and to produce numbers that someone else can reproduce. It is not a claim about edge hardware: the Raspberry Pi 4 measurements are still to be done.

---

## 1. Project status

- [x] **Milestone 1: x86-64 laptop CPU (completed)**
  - Data pipeline on `timm/mini-imagenet` (`test` split, 2,000 validation images, 300 calibration images).
  - PyTorch FP32 baseline with explicit `ResNet50_Weights.IMAGENET1K_V2` and ONNX export check.
  - Static PTQ: ONNX Runtime (QDQ, per-channel) and OpenVINO (NNCF, with bias correction).
  - Accuracy on 100 classes (closed-world) and on 1,000 classes (open-world), with WordNet ID mapping.
  - Two-pass latency measurement with a 60-second pause between passes.
- [ ] **Milestone 2: Raspberry Pi 4 (planned)**
  - Run the same ONNX models on a Raspberry Pi 4 Model B (Broadcom BCM2711, 4x Cortex-A72), once a board is available.
  - Compare ARM NEON with x86 AVX-512 VNNI for INT8.

---

## 2. Test Environment

Measurements on the x86 laptop were run in a native Windows 11 console session without virtualization (no Docker, no WSL2), on AC power with the *High Performance* power plan.

Measurements on the Raspberry Pi 4 are executed headless via SSH on a minimal Debian installation, connected to an official USB-C power supply, with the CPU governor set to `performance` to eliminate dynamic frequency scaling latencies.

| Parameter | Intel Laptop (Milestone 1) | Raspberry Pi 4 Model B (Milestone 2) |
| :--- | :--- | :--- |
| **SoC / Processor** | Intel Core i5-1155G7 (2.50 GHz base, up to 4.50 GHz turbo) | Broadcom BCM2711 Rev 1.5 (C0 stepping, fixed @ 1.80 GHz) |
| **Microarchitecture** | Tiger Lake-UP3 (Willow Cove, 10nm SuperFin) | ARM Cortex-A72 (ARMv8.0-A, 28nm) |
| **CPU Topology** | 4 physical cores / 8 logical threads (SMT enabled) | 4 physical cores / 4 threads (no SMT) |
| **Cache Hierarchy** | 8 MB shared L3 cache | 1 MB shared L2 cache (no L3) |
| **Vector Extensions** | AVX-512 (F, BW, CD, DQ, VL) | ARM NEON (`asimd`) |
| **Neural / INT8 Instructions** | **AVX-512 VNNI** (`VPDPBUSD`) | **None** (`asimddp` Dot Product absent in ARMv8.0-A) |
| **Memory (RAM)** | 8.00 GB DDR4-3200 (7.76 GB allocatable) | 2.00 GB LPDDR4-3200 (1.8 GiB allocatable) |
| **Operating System** | Windows 11 Home x64 (build 26200) | Debian GNU/Linux 13 (*trixie*) Lite 64-bit (`aarch64`) |
| **CPU Governor** | OS High Performance Profile | Linux `performance` governor (1800 MHz fixed) |
| **Thermal Telemetry** | Not recorded | Hardware sensors via `vcgencmd` (baseline ~36.5 °C, `throttled=0x0`) |
| **Software Stack** | Python 3.14, ONNX Runtime 1.31.0, OpenVINO 2026.4.1 | Python 3.13, ONNX Runtime (TBD via venv) |

## 3. Methodology

### 3.1 Model and preprocessing

- ResNet-50, 25.55M parameters, weights pinned to `torchvision.models.ResNet50_Weights.IMAGENET1K_V2` (reference ImageNet-1k Top-1: 80.86%).
- Same preprocessing for every framework: `Resize(232)`, `CenterCrop(224)`, `ToTensor()`, `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- ONNX export (opset 18, constant folding). The maximum absolute difference from the PyTorch output on test tensors was 2.15e-6, consistent with float32 rounding.

### 3.2 Data

- **Validation (2,000 images):** from the `test` split of `timm/mini-imagenet`. According to the dataset card, this split comes from the official ImageNet-1k validation set, which the pretrained model did not see during training. The first 20 images per class (in dataset order, not randomly sampled) are used, for 100 classes.
- **Calibration (300 images):** from the `train` split, 3 images per class, disjoint from the validation images.
- **Storage:** original JPEG bytes are written to disk without re-encoding.
- **Label mapping:** `imagenet_class_index.json` (same class order as torchvision) links the 100 local classes to the 1,000 output logits through WordNet IDs.
- **Note:** an earlier version of this project used the `validation` split of the same dataset. That split is taken from the ImageNet-1k *training* set, so accuracy was inflated (91.95% on 1,000 classes). It was found and fixed; all numbers below use the `test` split.

### 3.3 Accuracy

- **Closed-world (100 classes):** argmax over the 100 logits of the subset classes.
- **Open-world (1,000 classes):** argmax over all 1,000 logits. A prediction outside the correct class counts as an error, including predictions in the other 900 classes.

### 3.4 Benchmark configuration

- **Workload:** single-stream, `batch_size = 1`, images processed one after the other.
- **Input:** a random float32 tensor of shape 1x3x224x224. Preprocessing and disk I/O are not included in the timed region.
- **Threads:** 4 intra-op threads (`intra_op_num_threads = 4`, `torch.set_num_threads(4)`), matching the 4 physical cores. Thread affinity was not configured. Inter-op parallelism is 1.
- **Loop:** 15 untimed warm-up iterations, then 100 timed iterations with `time.perf_counter()`.
- **Repeatability:** two passes (Run 1 and Run 2) of the whole suite in the same process, with a 60-second pause between them.
- **Procedure before each session:** laptop on AC power with a charged battery, High Performance power plan, all other applications closed, a few minutes of idle time before starting.

---

## 4. Results

### 4.1 Latency and accuracy (batch size = 1)

Latency values come from a single benchmark session (consolidated results in `data/benchmark_results.json`). Accuracy was measured on the 2,000 validation images. Model sizes are on-disk sizes in MB = 2^20 bytes (weights included).

| Runtime | Precision | Model size | Median latency (p50) | p95 latency | Single-stream rate (1/latency) | Top-1, 100 classes | Top-1, 1,000 classes | Speedup vs PyTorch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PyTorch (eager) | FP32 | 97.7 MB | 65.81 ms | 71.08 ms | 15.37 fps | 95.55% | 82.80% | 1.00x |
| ONNX Runtime | FP32 | 97.7 MB | 26.51 ms | 35.38 ms | 37.36 fps | 95.55% | 82.80% | 2.48x |
| OpenVINO | FP32 (weights stored as FP16) | 48.7 MB | 22.18 ms | 23.77 ms | 44.55 fps | 95.55% | 82.80% | 2.97x |
| ONNX Runtime | INT8 (static QDQ) | 25.0 MB | 8.70 ms | 10.17 ms | 116.90 fps | 95.75% | 82.60% | 7.56x |
| OpenVINO | INT8 (NNCF PTQ) | 24.5 MB | 6.39 ms | 7.21 ms | 153.33 fps | 95.45% | 82.30% | 10.30x |

The single-stream rate is 1000 divided by the mean latency in ms. At batch 1 it carries the same information as the latency and is not a measure of saturated throughput with larger batches or several streams.

### 4.2 Repeatability between the two passes (same session)

Difference in median latency between Run 1 and Run 2, as shown in the console summary of the session (the exact per-pass values were not saved in the results file, so the differences are given approximately):

| Configuration | Difference |
| :--- | :--- |
| ONNX Runtime FP32 | about 5% |
| ONNX Runtime INT8 | about 5% |
| PyTorch FP32 | below 2% |
| OpenVINO FP32 | below 2% |
| OpenVINO INT8 | below 2% |

### 4.3 Variability between sessions

The benchmark was run in several sessions during this work. Absolute latency changed a lot from session to session, while the ranking of the runtimes did not. Approximate range of the median latency for the same configuration:

| Configuration | Median latency range across sessions |
| :--- | :--- |
| PyTorch FP32 | 63 to 85 ms |
| ONNX Runtime FP32 | 27 to 39 ms |
| OpenVINO FP32 | 22 to 33 ms |
| ONNX Runtime INT8 | 8.7 to 12.2 ms |
| OpenVINO INT8 | 6.4 to 9.6 ms |

The slowest session was therefore 35 to 50% slower than the fastest. In some sessions the pass-to-pass difference was much larger than in the table above (13.43% for ONNX Runtime FP32 in one, 22.79% for PyTorch FP32 in another). The cause was not identified: clock frequency and temperature were not logged. Speedups relative to PyTorch ranged from about 6.7x to 7.6x for ONNX Runtime INT8 and from about 8.0x to 10.3x for OpenVINO INT8, with the same ranking of configurations in every session.

---

## 5. Discussion

1. **Runtime overhead.** PyTorch eager mode pays for Python dispatch and dynamic memory handling layer by layer. ONNX Runtime and OpenVINO apply graph optimizations (node fusion, static memory planning) in C++, which is consistent with the lower latency in FP32. This explanation was not tested separately (for example with a profiler).
2. **INT8 speedup.** INT8 is 3.05x faster than FP32 in ONNX Runtime and 3.47x in OpenVINO. This is consistent with the AVX-512 VNNI instructions available on this CPU (such as `VPDPBUSD`, which computes 4-way 8-bit dot products with 32-bit accumulation). The contribution of VNNI was not isolated experimentally.
3. **Memory footprint.** The INT8 model is 24.5 MB, larger than the 8 MB L3 cache, so the whole model cannot stay in cache. A plausible contribution to the speedup is the roughly 75% reduction in weight bytes read from DRAM, but this was not measured.
4. **Threads.** 4 threads were used to match the 4 physical cores. 8 threads (SMT) were not tested, so no claim is made about their effect.
5. **Accuracy under PTQ (1,000 classes).** FP32 82.80%, ONNX Runtime INT8 82.60% (-0.20 points), OpenVINO INT8 82.30% (-0.50 points). With N = 2,000 these differences correspond to 4 and 10 images. The standard error of a single accuracy value at this sample size is about 0.84 points, so the differences are not statistically significant. A paired test on the same images (for example McNemar) would be the appropriate next step.
6. **OpenVINO FP32 model size.** The OpenVINO IR stores weights as FP16 by default, hence 48.7 MB (about half of the 97.7 MB FP32 ONNX file). Compared with true FP32 weights, the OpenVINO INT8 model is about 4x smaller. Accuracy on this sample is identical to FP32.

---

## 6. Limitations

1. **Single-stream only.** Only `batch_size = 1` was tested. Throughput with larger batches or several concurrent streams is not characterized.
2. **Subset evaluation.** 2,000 images over 100 classes, taken in dataset order. This does not replace the 50,000-image ImageNet-1k validation set.
3. **One machine.** All data comes from a single laptop (HP Pavilion, Core i5-1155G7). Other CPUs, cooling and memory configurations may behave differently.
4. **No hardware telemetry.** Clock frequency, power and temperature were not logged. Thermal behavior was judged only from latency differences between passes, and the variability between sessions (section 4.3) is unexplained.
5. **Synthetic input.** Latency was measured with a random tensor, not with real images, and without preprocessing.
6. **Short loop.** With 100 timed iterations, p95 depends on the 5 slowest measurements and is noisy.
7. **No edge hardware yet.** The results do not say anything about ARM or about dedicated accelerators.
8. **ONNX Runtime pre-processing not applied.** ONNX Runtime recommends running its `quant_pre_process` step (shape inference and graph optimization) before static quantization and prints a warning when it is skipped. That step was not run, and its effect on the INT8 results was not measured.

---

## 7. How to reproduce

```bash
# 1. Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Data preparation (streaming download, original JPEG bytes)
python 02_prepare_data.py

# 3. Model export and numerical check
python 03_export_and_verify.py

# 4. INT8 quantization (ONNX Runtime and OpenVINO)
python 06_quantize_models.py

# 5. Accuracy evaluation
python 07_evaluate_all_accuracy.py

# 6. Latency benchmark
python 08_final_benchmarks.py
```

`04_benchmark_runner.py` is a standalone script that measures a single `.onnx` file from the command line (for example `python 04_benchmark_runner.py --model models/<file>.onnx`). It is meant to be copied to the Raspberry Pi together with the ONNX models.

---

## 8. Development notes

The code was written with the help of AI assistants. The author ran all experiments on the hardware described above.