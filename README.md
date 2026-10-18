# Edge ML Inference Benchmark: PyTorch, ONNX Runtime, and OpenVINO (FP32 vs INT8)

## Project Roadmap & Status

- [x] **Milestone 1 — x86 Mobile CPU (Completed):**
  - Data pipeline with streaming & stratified sampling (`timm/mini-imagenet`).
  - PyTorch FP32 baseline & ONNX export validation (error $< 10^{-5}$).
  - Static PTQ quantization: ONNX Runtime (QDQ) & OpenVINO (NNCF).
  - Accuracy evaluation (100 classes closed-world vs 1000 classes open-world).
  - Benchmark suite: latency ($p50$, $p95$), throughput, and repeatability verification.
- [ ] **Milestone 2 — ARM Edge Deployment (In Progress):**
  - Porting and validation on Raspberry Pi 4 (Broadcom BCM2711, 4x Cortex-A72).
  - ONNX Runtime benchmark (FP32 vs INT8 via ARM NEON SIMD).
  - Cross-architecture comparison (x86 AVX-512 VNNI vs ARM NEON).

---

## 1. Hardware Specifications & Test Environment

All benchmarks were executed directly on native Windows 11 (avoiding virtualization overhead and scheduling jitter from Docker/WSL2), connected to AC power under the *High Performance* power profile.

* **Processor:** Intel Core i5-1155G7 @ 2.50GHz (Tiger Lake, 10nm SuperFin)
* **Topology:** 4 Physical Cores / 8 Logical Threads
* **L3 Cache:** 8 MB Intel Smart Cache
* **Hardware Instruction Sets:** AVX-512F, AVX-512BW, AVX-512CD, AVX-512DQ, AVX-512VL, **AVX-512 VNNI (Vector Neural Network Instructions)**
* **System RAM:** 8.00 GB DDR4 (7.76 GB allocatable)
* **Operating System:** Windows 11 Home (x64)
* **Software Stack:** Python 3.14, PyTorch 2.14.1 (CPU-only), ONNX Runtime 1.31.0, OpenVINO 2026.4.1, NNCF 3.4.0

---

## 2. Experimental Methodology

### Dataset Pipeline and Label Mapping
* **Model Architecture:** ResNet-50 pretrained on ImageNet-1k (25.55M parameters).
* **Validation Subset:** Stratified subset of **2,000 images** (20 images per class across 100 classes from `timm/mini-imagenet`).
* **Calibration Subset (PTQ):** Disjoint subset of **300 images** (3 images per class extracted from the training split) to eliminate data leakage.
* **WordNet ID Mapping:** Handled via official `imagenet_class_index.json` to map local subset indices ($0..99$) to global network logits ($0..999$).
* **Accuracy Metrics:**
  * **Top-1 (100 classes - Closed-world):** Argmax computed solely over the 100 target logits.
  * **Top-1 (1000 classes - Open-world):** Argmax evaluated over all 1000 ImageNet-1k logits (testing discrimination against 900 distractor classes).

### Measurement Parameters
* **Batch Size:** $1$ (real-time edge vision scenario).
* **CPU Threading:** $4$ `intra-op` threads pinned to physical cores; `inter-op` threads fixed to 1 to eliminate thread switching overhead.
* **Warm-up:** 15 unrecorded runs for cache warming and runtime initialization.
* **Benchmarking Loop:** 100 iterations measured using `time.perf_counter()`, executed over two independent runs (*Run 1* and *Run 2*) to verify thermal repeatability.

---

## 3. Experimental Results

| Runtime | Precision | Model Size | Median Latency ($p50$) | 95th Perc Latency ($p95$) | Throughput (FPS) | Top-1 (100 cl) | Top-1 (1000 cl) | Speedup vs PyTorch |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Eager)** | FP32 | 97.8 MB | 62.84 ms | 147.69 ms | 15.91 fps | 98.00% | 91.95% | **1.00x** |
| **ONNX Runtime** | FP32 | 97.8 MB | 33.19 ms | 48.20 ms | 30.13 fps | 98.00% | 91.95% | **1.89x** |
| **OpenVINO** | FP32 | 48.7 MB (FP16)* | 25.94 ms | 36.50 ms | 38.55 fps | 98.00% | 91.95% | **2.42x** |
| **ONNX Runtime** | INT8 (Static QDQ) | 25.0 MB | 9.34 ms | 14.10 ms | 107.07 fps | 98.10% | 91.30% | **6.73x** |
| **OpenVINO** | INT8 (NNCF PTQ) | 24.5 MB | **7.87 ms** | **11.80 ms** | **127.06 fps** | 97.95% | **91.70%** | **7.98x** |

*\*Note: OpenVINO FP32 IR stores weights in FP16 representation by default.*

---

## 4. Key Engineering Takeaways

1. **Tail Latency ($p95$) & Jitter Control:**
   PyTorch exhibits high tail latency (147.69 ms, $+135\%$ over median) due to dynamic buffer allocations and Python runtime overhead. Static memory planning in ONNX Runtime and OpenVINO limits $p95$ jitter to under 15 ms.
2. **AVX-512 VNNI Hardware Acceleration:**
   INT8 quantisation achieves a $\sim 3.5\text{x}$ speedup over FP32 on this CPU through the `VPDPBUSD` instruction, which computes 4-way INT8 dot products with INT32 accumulation in a single clock cycle.
3. **Memory Footprint & Cache Saturation:**
   Compressing weights from 97.8 MB down to 24.5 MB ($-75\%$) alleviates memory bus pressure on DDR4 RAM, keeping active layers resident inside CPU L2/L3 caches.
4. **Accuracy Retention:**
   OpenVINO with NNCF retains 91.70% Top-1 open-world accuracy (a loss of only $0.25\%$), validated against the baseline 91.95%.

---

## 5. How to Reproduce

```bash
# 1. Environment Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install numpy onnx onnxscript onnxruntime openvino nncf psutil pillow tqdm tabulate datasets torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# 2. Data Preparation (Streaming download & stratification)
python 02_prepare_data.py

# 3. Model Export & Numerical Verification
python 03_export_and_verify.py

# 4. PTQ INT8 Quantization
python 06_quantize_models.py

# 5. Accuracy Evaluation
python 07_evaluate_all_accuracy.py

# 6. Final Benchmark Suite
python 08_final_benchmarks.py