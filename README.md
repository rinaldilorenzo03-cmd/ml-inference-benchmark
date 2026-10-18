# Real-Time Edge Inference Benchmark: PyTorch, ONNX Runtime, and OpenVINO (FP32 vs INT8)

An empirical benchmark evaluating single-stream inference latency, tail jitter ($p95$), and Top-1 accuracy for **ResNet-50** on modern x86 mobile hardware. This project investigates the efficiency gains of Post-Training Quantization (PTQ) and hardware acceleration via **Intel AVX-512 VNNI** under strict real-time constraints (batch size = 1).

---

## 1. Project Status & Roadmap

- [x] **Milestone 1 — x86-64 Mobile CPU Platform (Completed):**
  - Bit-exact streaming pipeline and stratified sampling on `timm/mini-imagenet` (unseen `test` split).
  - Explicit `ResNet50_Weights.IMAGENET1K_V2` baseline and validated ONNX export (numerical tolerance $< 2.15 \times 10^{-6}$).
  - Static PTQ quantization: ONNX Runtime (QDQ per-channel) and OpenVINO (NNCF with bias correction).
  - Dual accuracy evaluation: Closed-world (100 classes) vs Open-world (1,000 classes) using canonical WordNet ID mapping.
  - Multi-pass benchmark suite with 60-second cooldown verification ruling out thermal throttling and background noise.
- [ ] **Milestone 2 — ARM Edge Deployment (In Progress):**
  - Deployment and cross-compilation on Raspberry Pi 4 Model B (Broadcom BCM2711, 4x ARM Cortex-A72).
  - ONNX Runtime execution benchmarking ARM NEON vectorization against x86 AVX-512 VNNI.
  - Cross-architecture latency, jitter, and efficiency comparison.

---

## 2. Hardware Specifications & Testbed Environment

All benchmarks were conducted natively in a clean Windows 11 console environment without IDE overhead or virtualization layers (avoiding scheduling jitter from Docker/WSL2). The host was connected to continuous AC power under the **High Performance** power plan.

| Parameter | Specification |
| :--- | :--- |
| **Processor** | Intel Core i5-1155G7 @ 2.50 GHz (Turbo boost up to 4.50 GHz) |
| **Microarchitecture** | Tiger Lake-UP3 (Willow Cove cores, Intel 10nm SuperFin) |
| **CPU Topology** | 4 Physical Cores / 8 Logical Threads (SMT enabled) |
| **L3 Cache Hierarchy** | 8 MB Intel Smart Cache (shared across all cores) |
| **Vector ISA Detected** | AVX-512F, AVX-512BW, AVX-512CD, AVX-512DQ, AVX-512VL |
| **Dedicated Neural ISA** | **AVX-512 VNNI** (*Vector Neural Network Instructions*, Intel DL Boost) |
| **System Memory (RAM)** | 8.00 GB DDR4-3200 MHz (7.76 GB allocatable) |
| **Operating System** | Windows 11 Home x64 (Build 26200) |
| **Software Stack** | Python 3.14, PyTorch 2.14.1 (CPU-only), ONNX 1.23.2, ONNX Runtime 1.31.0, OpenVINO 2026.4.1, NNCF 3.4.0 |

---

## 3. Experimental Methodology

### 3.1 Model & Weight Pinning
* **Model:** ResNet-50 (25.55M parameters, 4.12 GFLOPs at $224 \times 224$).
* **Exact Weights:** Explicitly pinned to `torchvision.models.ResNet50_Weights.IMAGENET1K_V2` (modern training recipe with CutMix, MixUp, and cosine annealing; canonical Top-1 accuracy: 80.86%).
* **Preprocessing Pipeline:** Identical across all runtimes: `Resize(232)` $\rightarrow$ `CenterCrop(224)` $\rightarrow$ `ToTensor()` $\rightarrow$ `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
* **Export Verification:** ONNX Opset 18 with constant folding. Maximum absolute numerical error against PyTorch eager execution was verified at **$2.15 \times 10^{-6}$**, confirming strict equivalence within IEEE 754 float32 precision limits.

### 3.2 Dataset Pipeline & Zero Data-Leakage Protocol
* **Validation Subset (2,000 images):** Sampled strictly from the `test` split of `timm/mini-imagenet` (originating from the unseen ILSVRC-2012 validation set), taking exactly 20 images per class across 100 classes.
* **Calibration Subset (300 images):** Extracted exclusively from the disjoint `train` split (3 images per class), eliminating calibration-to-validation data leakage.
* **Bit-Exact Raw Byte Streams:** Implemented via streaming with `decode=False` to store native JPEG byte buffers directly to disk, avoiding secondary DCT lossy re-compression artifacts.
* **Label Mapping:** Mapped via canonical `imagenet_class_index.json` to link the subset's local indices ($0..99$) with the global network output logits ($0..999$) using unique WordNet IDs (`wnid`).

### 3.3 Real-Time Edge Benchmarking Protocol
* **Workload Scenario:** **Single-Stream (Batch Size = 1)**, reflecting real-time edge processing where sensory inputs arrive sequentially and must be processed with minimal latency.
* **Threading Policy:** Pinned to **4 intra-op threads** (1 thread per physical core); inter-op threads set to 1. Over-subscribing the CPU to 8 logical threads causes L1/L2 cache thrashing and pipeline stalls on single-image workloads.
* **Warm-up:** 15 unrecorded inference passes.
* **Measurement Loop:** 100 iterations timed via high-resolution hardware counters (`time.perf_counter()`).
* **Thermal Repeatability Protocol:** Multi-pass execution separated by a **60-second idle cooldown** to verify that results are not biased by thermal throttling.

---

## 4. Empirical Results

### 4.1 Master Edge Benchmark (Batch Size = 1)

Evaluated across the 2,000 unseen validation images (`test` split) and consolidated across two thermal-verified passes ($N=100$ per pass).

| Runtime Engine | Precision Format | Model Size | Median Latency ($p50$) | 95th Perc ($p95$) | Single-Stream FPS ($1/\text{lat}$) | Top-1 (100 cl) | Top-1 (1,000 cl) | Speedup vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Eager)** | FP32 | 97.8 MB | 65.81 ms | 71.08 ms | 15.37 fps | 95.55% | 82.80% | **1.00x** (baseline) |
| **ONNX Runtime** | FP32 | 97.8 MB | 26.51 ms | 35.38 ms | 37.36 fps | 95.55% | 82.80% | **2.48x** |
| **OpenVINO** | FP32 *(FP16 weights)* | 48.7 MB | 22.18 ms | 23.77 ms | 44.55 fps | 95.55% | 82.80% | **2.97x** |
| **ONNX Runtime** | INT8 (Static QDQ) | 25.0 MB | 8.70 ms | 10.17 ms | 116.90 fps | 95.75% | 82.60% | **7.56x** |
| **OpenVINO** | INT8 (NNCF PTQ) | 24.5 MB | **6.39 ms** | **7.21 ms** | **153.33 fps** | 95.45% | **82.30%** | **10.30x** |

*\*Note: In alignment with MLPerf standards for real-time edge processing, single-image sequential execution reports the single-stream rate ($1/\text{latency}$). Offline batch throughput is intentionally omitted as batching introduces queuing delays incompatible with hard real-time deadlines.*

---

## 5. Architectural & Engineering Insights

1. **Execution Determinism & Tail Jitter ($p95$):**  
   Executing in a clean host environment demonstrates that dedicated inference engines achieve exceptional execution determinism. On OpenVINO INT8, the gap between median latency (6.39 ms) and the 95th percentile (7.21 ms) is bounded to just **0.82 ms**, satisfying the strictest soft real-time edge SLA constraints.
2. **AVX-512 VNNI Vector Acceleration:**  
   INT8 quantization accelerates execution by **$3.05\text{x}$ on ONNX Runtime** and **$3.47\text{x}$ on OpenVINO** over their respective FP32 baselines, culminating in an overall **$10.30\text{x}$ speedup** over PyTorch eager mode. This performance leap is enabled by Intel's `VPDPBUSD` instruction, which fuses four 8-bit integer multiply-accumulate operations into a 32-bit destination register in a single clock cycle.
3. **Cache Line & Working Set Optimization:**  
   The INT8 model (24.5 MB) achieves a **$75\%$ reduction in memory footprint**, significantly lowering bandwidth demands on the DDR4 bus. Active working sets and intermediate activations remain resident within the CPU's L2/L3 cache hierarchy, eliminating memory bus stalls.
4. **Accuracy Retention Under PTQ:**  
   On unseen test data across all 1,000 classes, the baseline achieves **82.80%** Top-1 accuracy. **ONNX Runtime INT8** drops by only **$0.20\%$** (82.60%), while **OpenVINO NNCF INT8** drops by only **$0.50\%$** (82.30%), validating that post-training calibration with bias correction preserves model fidelity for production edge use.

---

## 6. How to Reproduce

```bash
# 1. Environment Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install numpy onnx onnxscript onnxruntime openvino nncf psutil pillow tqdm tabulate datasets

# 2. Data Preparation (Streaming & Bit-Exact JPEGs)
python 02_prepare_data.py

# 3. Model Export & Numerical Verification
python 03_export_and_verify.py

# 4. PTQ INT8 Quantization
python 06_quantize_models.py

# 5. Dual Accuracy Evaluation
python 07_evaluate_all_accuracy.py

# 6. Multi-Pass Benchmark Suite
python 08_final_benchmarks.py