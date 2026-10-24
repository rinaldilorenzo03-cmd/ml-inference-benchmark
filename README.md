# ML Inference Benchmark: x86 Laptop CPU vs. Raspberry Pi 4 Model B
 
### ResNet-50 with PyTorch, ONNX Runtime, and OpenVINO: FP32 vs. INT8
 
A reproducible benchmark of single-stream inference latency, model size, and Top-1 accuracy for **ResNet-50** on two hardware platforms: an Intel mobile CPU (Core i5-1155G7) and an embedded board (Raspberry Pi 4 Model B Rev 1.5).
 
On the laptop the benchmark compares PyTorch eager mode with two deployment runtimes (ONNX Runtime and OpenVINO), each in FP32 and INT8 (static post-training quantization), always at `batch_size = 1`. On the Raspberry Pi only ONNX Runtime was run. The main question is how much static INT8 quantization helps on two different CPU microarchitectures: x86-64 with **AVX-512 VNNI** and a 64-bit ARM Cortex-A72 without the dot-product extension.
 
The project is a learning exercise on the inference toolchain (export, quantization, measurement methodology). Results are given with their limitations (section 6).
 
---
 
## 1. Project Status
 
- [x] **Milestone 1: x86-64 Laptop CPU (completed)**
  - Data pipeline on `timm/mini-imagenet` (`test` split, 2,000 validation images, 300 calibration images).
  - PyTorch FP32 baseline with explicit `ResNet50_Weights.IMAGENET1K_V2` and ONNX export check.
  - Static PTQ: ONNX Runtime (QDQ, per-channel) and OpenVINO (NNCF, with bias correction).
  - Accuracy on 100 classes (closed-world) and on 1,000 classes (open-world) via WordNet ID mapping.
  - Two-pass latency measurement and observation of session-to-session variability.
- [x] **Milestone 2: Raspberry Pi 4 Model B (completed, ONNX Runtime only)**
  - Execution of the exported ONNX models (FP32 and INT8) on Cortex-A72 with `onnxruntime==1.31.0`.
  - Local consistency check of the outputs on the board (FP32 vs. INT8).
  - CPU governor set to `performance`; temperature and throttling flags sampled before and after each run.
  - Comparison of x86 and ARM results and discussion of compute vs. memory hypotheses (not tested).
---
 
## 2. Test Environment
 
Measurements on the laptop were run in a native Windows 11 console session without virtualization (no Docker, no WSL2), on AC power with the *High Performance* power plan.
 
Measurements on the Raspberry Pi 4 were run headless over an SSH session on a minimal Debian installation, powered through USB-C, with the CPU scaling governor set to `performance`.
 
Cooling on the Raspberry Pi: bare board, no heatsink or fan, in open air; ambient temperature not recorded. Runs were short (about 16 to 35 s) and the temperature rose from 47.7 to 69.6 °C (FP32) and from 60.3 to 73.0 °C (INT8). The results apply to short runs from a cool start; sustained load would likely reach the 80 °C throttling threshold.
 
| Parameter | Intel Laptop (Milestone 1) | Raspberry Pi 4 Model B (Milestone 2) |
| :--- | :--- | :--- |
| **SoC / Processor** | Intel Core i5-1155G7 (2.50 GHz base, up to 4.50 GHz turbo) | Broadcom BCM2711 Rev 1.5 (C0 stepping, 1.80 GHz) |
| **Microarchitecture** | Tiger Lake-UP3 (Willow Cove, 10nm SuperFin) | ARM Cortex-A72 (ARMv8.0-A, 28nm) |
| **CPU Topology** | 4 physical cores / 8 logical threads (SMT enabled) | 4 physical cores / 4 threads (no SMT) |
| **Cache Hierarchy** | 8 MB shared L3, 1.25 MB L2 per core | 1 MB shared L2 (no L3) |
| **Vector Extensions** | AVX-512 (F, BW, CD, DQ, VL) | ARM NEON (`asimd`) |
| **INT8 Instructions** | **AVX-512 VNNI** (`VPDPBUSD`, 4-way 8-bit dot product) | Dot-product extension absent (`asimddp` not in `/proc/cpuinfo`) |
| **Memory (RAM)** | 8.00 GB DDR4-3200 (7.76 GB allocatable) | 2.00 GB LPDDR4-3200 (1.8 GiB allocatable) |
| **Storage** | NVMe PCIe SSD | SanDisk MicroSD Class 10 (0 B swap used) |
| **Cooling** | Internal laptop fan | Bare board, no heatsink or fan, in open air |
| **Operating System** | Windows 11 Home x64 (build 26200) | Debian GNU/Linux 13 (trixie), 64-bit (`aarch64`), minimal installation |
| **CPU Governor** | Windows High Performance power plan | Linux `performance` governor (1800 MHz) |
| **Thermal Telemetry** | Not logged | Temperature and throttling flags via `vcgencmd`, sampled before and after each run |
| **Software Stack** | Python 3.14, ONNX Runtime 1.31.0, OpenVINO 2026.4.1 | Python 3.13.5, ONNX Runtime 1.31.0 (`CPUExecutionProvider`) |
 
---
 
## 3. Methodology
 
### 3.1 Model and Preprocessing
 
- **Architecture:** ResNet-50 (25.55M parameters), pinned to `torchvision.models.ResNet50_Weights.IMAGENET1K_V2` (ImageNet-1k Top-1 reference: 80.86%).
- **Preprocessing:** identical across all frameworks: `Resize(232)`, `CenterCrop(224)`, `ToTensor()`, `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- **ONNX export:** opset 18 with constant folding. The maximum absolute difference from the PyTorch output on test tensors was 2.15e-6, consistent with float32 rounding.
- **Local consistency check on ARM (FP32 vs. INT8):** before benchmarking on the Raspberry Pi, the outputs of `resnet50_fp32.onnx` and `resnet50_ort_int8.onnx` were compared on a deterministic synthetic tensor (1x3x224x224, seed 42) directly on the board:
  - Top-1 output class: 21 for both FP32 and INT8.
  - Cosine similarity: 0.98700.
  - Mean absolute error (MAE): 0.1069.
  This check compares FP32 and INT8 on the same board. It is not a comparison of ARM outputs against x86 outputs.
### 3.2 Calibration and Validation Data
 
- **Validation (2,000 images):** from the `test` split of `timm/mini-imagenet`. According to the dataset card, this split comes from the official ImageNet-1k validation set. The first 20 images per class (dataset order, not shuffled) across 100 classes are used.
- **Calibration (300 images):** from the `train` split, 3 images per class, disjoint from the validation images.
- **Label mapping:** `imagenet_class_index.json` links the 100 local classes to the 1,000 output logits through WordNet IDs.
- **Note:** an earlier version of this project used the `validation` split of the same dataset. That split is taken from the ImageNet-1k *training* set, so the accuracy was inflated (91.95% on 1,000 classes). It was found and fixed; all numbers here use the `test` split.
### 3.3 Accuracy Definitions
 
- **Closed-world (100 classes):** argmax over the 100 logits of the subset classes.
- **Open-world (1,000 classes):** argmax over all 1,000 logits. A prediction outside the correct class counts as an error, including predictions in the other 900 classes.
### 3.4 Benchmark Configuration and Repeatability
 
- **Workload:** single-stream, `batch_size = 1`. Preprocessing, disk I/O and image decoding are not part of the timed loop.
- **Input:** a synthetic float32 tensor of shape 1x3x224x224 (fixed seed on the Raspberry Pi, random on the laptop).
- **Threads:** 4 intra-op threads (`intra_op_num_threads = 4`, `torch.set_num_threads(4)`), equal to the number of physical cores on both platforms. Thread affinity was not configured. Inter-op parallelism is 1, sequential execution mode.
- **Loop:** 15 untimed warm-up iterations on the laptop (10 to 15 on the Raspberry Pi, see `benchmark_rpi.py`), then 100 timed iterations with `time.perf_counter()`.
- **Repeatability on x86:** two passes (Run 1 and Run 2) with a 60 s pause. The median latency differed by about 5% for ONNX Runtime (FP32 and INT8) and by less than 2% for PyTorch, OpenVINO FP32 and OpenVINO INT8. These values come from the console summary; the exact per-pass values were not saved.
- **Telemetry on ARM:** before each run the script checks that `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` is `performance` and aborts otherwise. Temperature and the throttling flags (`vcgencmd get_throttled`) are sampled immediately before and after each model. ARM measurements are a single pass.
---
 
## 4. Results
 
### 4.1 x86 Laptop (Intel Core i5-1155G7)
 
Latency values come from a single session (consolidated results in `data/benchmark_results.json`). Accuracy was measured on the 2,000 validation images. Model sizes are on-disk sizes (1 MB = 2^20 bytes).
 
| Runtime | Precision | Model Size | Median Latency (p50) | p95 Latency | Single-Stream Rate | Top-1 (100 cls) | Top-1 (1,000 cls) | Speedup vs. PyTorch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch (eager)** | FP32 | 97.7 MB | 65.81 ms | 71.08 ms | 15.37 fps | 95.55% | 82.80% | 1.00x |
| **ONNX Runtime** | FP32 | 97.7 MB | 26.51 ms | 35.38 ms | 37.36 fps | 95.55% | 82.80% | 2.48x |
| **OpenVINO** | FP32 (weights stored as FP16) | 48.7 MB | 22.18 ms | 23.77 ms | 44.55 fps | 95.55% | 82.80% | 2.97x |
| **ONNX Runtime** | INT8 (static QDQ) | 25.0 MB | 8.70 ms | 10.17 ms | 116.90 fps | 95.75% | 82.60% | 7.56x |
| **OpenVINO** | INT8 (NNCF PTQ) | 24.5 MB | 6.39 ms | 7.21 ms | 153.33 fps | 95.45% | 82.30% | 10.30x |
 
The single-stream rate is 1000 divided by the mean latency in ms. At batch 1 it carries the same information as the latency and is not a throughput measurement with larger batches or several streams.
 
**Session variability on x86.** Across the sessions run during this work, the absolute median latency of the same configuration varied by 35 to 50% (for example ONNX Runtime FP32 from 27 to 39 ms, ONNX Runtime INT8 from 8.7 to 12.2 ms), while the ranking of the configurations stayed the same. This variability was not explained: clock frequency and temperature were not logged on the laptop. Within ONNX Runtime, INT8 was 3.05x faster than FP32 (median latency, consolidated session).
 
### 4.2 Raspberry Pi 4 Model B (BCM2711, 4x Cortex-A72)
 
Single headless session, ONNX Runtime 1.31.0, `CPUExecutionProvider`, 4 threads, governor `performance`.
 
| Model | Precision | Model Size | Mean Latency | Median (p50) | p95 Latency | Single-Stream Rate | Temperature / Throttling flags |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | FP32 | 97.7 MB | 349.79 ms | 349.28 ms | 351.49 ms | 2.86 fps | 47.7 °C -> 69.6 °C \| `0x0` |
| **ResNet-50** | INT8 (static QDQ) | 25.0 MB | 158.39 ms | 158.38 ms | 159.13 ms | 6.31 fps | 60.3 °C -> 73.0 °C \| `0x0` |
 
- **INT8 speedup on ARM:** about 2.2x (median latency 349.28 ms vs. 158.38 ms), a latency reduction of about 55%.
- **Latency spread:** p95 was within 0.7% of the median over the 100 iterations of each run, so the jitter was low in these short runs.
- **Throttling:** `vcgencmd get_throttled` returned `0x0` after both runs. The flags also cover events since boot, so no under-voltage or throttling occurred. The two runs started from different temperatures (47.7 and 60.3 °C), and the INT8 run reached 73.0 °C in about 16 s. Longer or repeated runs would likely reach the 80 °C throttling threshold.
### 4.3 Comparison between Platforms
 
Same ONNX models and ONNX Runtime 1.31.0, 4 threads, `batch_size = 1`:
 
| Metric | Intel Core i5-1155G7 (x86-64) | Raspberry Pi 4 Model B (ARM64) | Approximate Ratio |
| :--- | :--- | :--- | :--- |
| **FP32 median latency** | 26.51 ms | 349.28 ms | about 9x to 13x faster on x86 |
| **INT8 median latency** | 8.70 ms | 158.38 ms | about 13x to 18x faster on x86 |
| **FP32 single-stream rate** | 37.36 fps | 2.86 fps | about 9x to 13x higher on x86 |
| **INT8 single-stream rate** | 116.90 fps | 6.31 fps | about 13x to 18x higher on x86 |
| **INT8 speedup vs. FP32** | 3.05x (about -67% latency) | about 2.2x (about -55% latency) | smaller gain on ARM |
| **Model size** | 97.7 MB -> 25.0 MB | 97.7 MB -> 25.0 MB | 3.91x smaller |
 
The ratio ranges reflect the x86 session variability (4.1): the laptop median varied between sessions by 35 to 50%, so a single multi-digit ratio would be spuriously precise. The ARM side is a single pass.
 
---
 
## 5. Discussion
 
1. **INT8 on VNNI vs. on a core without dot-product instructions.**
   - On the Intel Core i5-1155G7, INT8 gives 3.05x. The CPU implements AVX-512 VNNI (`VPDPBUSD`), which computes 4-way 8-bit dot products with 32-bit accumulation in one instruction.
   - On the Broadcom BCM2711 (Cortex-A72, ARMv8.0-A), INT8 gives about 2.2x. This core lacks the ARMv8.2-A dot-product extension (`asimddp` absent in `/proc/cpuinfo`). How ONNX Runtime executes INT8 on this core was not inspected.
2. **Why INT8 is faster on Cortex-A72 (hypotheses, not tested).** It is tempting to credit the memory traffic saved by the smaller model, but a simple calculation points against a DRAM bandwidth limit:
   - FP32 weights (97.7 MB) are read in about 0.35 s per inference, roughly 0.28 GB/s.
   - INT8 weights (25.0 MB) are read in about 0.158 s, roughly 0.16 GB/s.
   - The sustained LPDDR4 bandwidth usually quoted for this board is about 4 to 5 GB/s (not measured here), so weight traffic alone is a small fraction of it. Activation traffic was not estimated.
   More plausible contributions:
   - Vector register packing: a 128-bit NEON register holds sixteen 8-bit integers against four 32-bit floats, so more multiply-accumulates fit per instruction. The kernels actually used were not inspected.
   - Cache footprint: smaller weights and activations may fit better in the 1 MB L2 cache. Not measured.
   A simple check would be to measure latency with 1, 2 and 4 threads (near-linear scaling suggests a compute-bound workload) or to profile the run with hardware counters.
3. **Runtime overhead on x86.** PyTorch eager mode pays for per-operator dispatch and dynamic memory handling. ONNX Runtime and OpenVINO apply graph optimizations (node fusion, static memory planning, constant folding) in C++, which is consistent with the 2.48x to 2.97x FP32 improvements. This was not tested separately (for example with a profiler).
4. **Accuracy under PTQ.** On 1,000 classes, ONNX Runtime INT8 is 0.20 points below FP32 (82.60% vs. 82.80%) and OpenVINO INT8 is 0.50 points below (82.30%). With N = 2,000 these are 4 and 10 images. The standard error of a single accuracy value is about 0.84 points, so the differences are statistically indistinguishable from zero. They show no measurable degradation, but they do not prove that accuracy is preserved without a larger evaluation set. A paired test on the same images (for example McNemar) would be the next step.
5. **Model size.** The OpenVINO IR stores weights as FP16 by default, hence 48.7 MB for the "FP32" model (about half of the 97.7 MB ONNX file). Compared with true FP32 weights, the OpenVINO INT8 model is about 4x smaller. Accuracy on this sample is identical to FP32.
---
 
## 6. Limitations
 
1. **Single-stream only.** Only `batch_size = 1` was tested; throughput with batches or several concurrent streams was not measured.
2. **Subset evaluation.** 2,000 images over 100 classes of `timm/mini-imagenet`, in dataset order. It does not replace the 50,000-image ImageNet-1k validation set.
3. **Inference only, not the full pipeline.** Latency was measured on a synthetic input tensor and does not include image decoding or preprocessing. For a dense network such as ResNet-50 the amount of computation does not depend on the pixel values, so the effect of a synthetic input is expected to be small (not tested).
4. **No telemetry on the laptop.** Clock frequency and temperature were not logged on x86, and the session-to-session variability (35 to 50%) is unexplained. Cross-platform ratios are therefore approximate.
5. **Single pass on ARM, no cooling.** The Raspberry Pi results come from one pass on a bare board. The two runs started from different temperatures (47.7 and 60.3 °C), and sustained or repeated runs would likely cause thermal throttling at 80 °C.
6. **INT8 model not tuned for ARM.** The INT8 model was quantized with settings used on x86 (`uint8` activations, `int8` weights, QDQ) and used unchanged on the Raspberry Pi. Different settings for ARM might give different results.
7. **Single framework on ARM.** Only ONNX Runtime was run on the Raspberry Pi; OpenVINO and PyTorch were not.
8. **Thread scaling not measured.** Only 4 threads were used, so the compute-bound hypothesis (section 5.2) was not verified.
9. **Numerical check is local.** On the Raspberry Pi, FP32 and INT8 were compared with each other on a synthetic input; the board's logits were not compared with the outputs of the laptop.
10. **Short loops.** With 100 timed iterations, p95 depends on the 5 slowest measurements and is noisy.
11. **ONNX Runtime pre-processing not applied.** The recommended `quant_pre_process` step was not run before static quantization, and its effect on the INT8 results was not measured.
---
 
## 7. How to Reproduce
 
### 7.1 Milestone 1: x86-64 Laptop
 
```powershell
# 1. Environment setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
 
# 2. Download and prepare the data
python 02_prepare_data.py
 
# 3. Export to ONNX (FP32) and check the export
python 03_export_and_verify.py
 
# 4. Quantize to INT8 (ONNX Runtime and OpenVINO)
python 06_quantize_models.py
 
# 5. Evaluate the accuracy of all models
python 07_evaluate_all_accuracy.py
 
# 6. Run the latency benchmark
python 08_final_benchmarks.py
```
 
### 7.2 Milestone 2: Raspberry Pi 4
 
```bash
# 1. System packages (64-bit Debian-based OS)
sudo apt update
sudo apt install -y python3-venv python3-pip git libraspberrypi-bin
 
# 2. Clone the repository and create a virtual environment
git clone https://github.com/rinaldilorenzo03-cmd/ml-inference-benchmark.git
cd ml-inference-benchmark
python3 -m venv .venv
source .venv/bin/activate
 
# 3. Install dependencies (quoted to avoid bash redirection)
pip install --upgrade pip setuptools wheel
pip install "numpy>=1.26.0,<2.3.0" "onnx>=1.16.0" onnxruntime==1.31.0 "psutil>=5.9.0"
 
# 4. Copy the exported models from the PC (run on the PC)
# scp models/* <user>@<host>.local:~/ml-inference-benchmark/models/
 
# 5. Set the CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
 
# 6. Check the models on ARM
python test_models.py
 
# 7. Run the benchmark with telemetry
python benchmark_rpi.py
```
 
---
 
## 8. Development Notes
 
The code and the benchmark scripts were written with the help of AI assistants. The author ran all experiments and measurements on the hardware described above.