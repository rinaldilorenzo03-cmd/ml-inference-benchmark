## 3. Experimental Results

All numbers reflect the consolidated evaluation on $N=2,000$ unseen validation images (split `test` from `mini-imagenet`, bit-exact JPEG bytes) and median latency across two independent 100-run passes separated by a 60-second thermal stabilization cooldown.

| Runtime | Precision | Model Size | Median Latency ($p50$) | Throughput (FPS) | Top-1 (100 cl) | Top-1 (1000 cl) | Speedup vs PyTorch |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Eager)** | FP32 | 97.8 MB | 84.38 ms | 11.85 fps | 95.55% | 82.80% | **1.00x** |
| **ONNX Runtime** | FP32 | 97.8 MB | 38.55 ms | 25.94 fps | 95.55% | 82.80% | **2.19x** |
| **OpenVINO** | FP32 | 48.7 MB (FP16)* | 32.65 ms | 30.63 fps | 95.55% | 82.80% | **2.58x** |
| **ONNX Runtime** | INT8 (Static QDQ) | 25.0 MB | 12.19 ms | 82.03 fps | 95.75% | 82.60% | **6.92x** |
| **OpenVINO** | INT8 (NNCF PTQ) | 24.5 MB | **9.53 ms** | **104.93 fps** | 95.45% | **82.30%** | **8.85x** |

*\*Note: OpenVINO FP32 IR stores weights in FP16 precision by default.*

### Thermal Repeatability Verification (60s Cooldown)
* **PyTorch FP32:** Run 1 = 83.89 ms | Run 2 = 84.86 ms (Delta: 1.16%)
* **ONNX Runtime FP32:** Run 1 = 39.11 ms | Run 2 = 37.99 ms (Delta: 2.86%)
* **ONNX Runtime INT8:** Run 1 = 12.13 ms | Run 2 = 12.24 ms (Delta: 0.92%)
* **OpenVINO FP32:** Run 1 = 33.07 ms | Run 2 = 32.23 ms (Delta: 2.54%)
* **OpenVINO INT8:** Run 1 = 9.61 ms | Run 2 = 9.45 ms (Delta: 1.72%)