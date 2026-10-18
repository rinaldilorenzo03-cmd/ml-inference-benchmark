"""
Passo 8: Suite di Benchmark Completa e Test di Ripetibilita'.
Misura:
  1. PyTorch FP32 (CPU)
  2. ONNX Runtime FP32 (CPU)
  3. ONNX Runtime INT8 (CPU)
  4. OpenVINO FP32 (CPU)
  5. OpenVINO INT8 (CPU)

Condizioni fisse:
  - 4 Thread CPU (core fisici)
  - batch_size = 1
  - 15 cicli di warm-up
  - 100 iterazioni di misura
  - Doppio passaggio (Run 1 vs Run 2) per verificare la stabilita' termica e l'assenza di jitter.
"""
import time
import json
from pathlib import Path
import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort
import openvino as ov


NUM_THREADS = 4
WARMUP_RUNS = 15
BENCH_RUNS = 100

ONNX_FP32_PATH = Path("models/resnet50_fp32.onnx")
ONNX_INT8_PATH = Path("models/resnet50_ort_int8.onnx")
OV_FP32_PATH = Path("models/ov_models/resnet50_fp32.xml")
OV_INT8_PATH = Path("models/ov_models/resnet50_int8.xml")
OUTPUT_BENCHMARK_JSON = Path("data/benchmark_results.json")


def time_execution(callable_fn, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    """Esegue warm-up e campionamento ad alta precisione con time.perf_counter()."""
    # 1. Warm-up
    for _ in range(warmup):
        callable_fn()

    # 2. Misura
    lats = []
    for _ in range(runs):
        t0 = time.perf_counter()
        callable_fn()
        t1 = time.perf_counter()
        lats.append((t1 - t0) * 1000.0)

    lats = np.array(lats)
    mean_lat = float(np.mean(lats))
    return {
        "median_ms": float(np.median(lats)),
        "p95_ms": float(np.percentile(lats, 95)),
        "mean_ms": mean_lat,
        "std_ms": float(np.std(lats)),
        "min_ms": float(np.min(lats)),
        "max_ms": float(np.max(lats)),
        "throughput_fps": float(1000.0 / mean_lat)
    }


def benchmark_single_pass(run_id=1):
    print(f"\n>>> AVVIO MISURAZIONE COMPLETA: PASSAGGIO {run_id} <<<")
    pass_results = {}
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # 1. PyTorch FP32
    print("  [1/5] Benchmark PyTorch FP32...")
    torch.set_num_threads(NUM_THREADS)
    weights = models.ResNet50_Weights.DEFAULT
    pt_model = models.resnet50(weights=weights).eval()
    dummy_torch = torch.from_numpy(dummy_np)
    
    with torch.no_grad():
        pass_results["PyTorch FP32"] = time_execution(lambda: pt_model(dummy_torch))
    del pt_model

    # 2. ONNX Runtime FP32
    print("  [2/5] Benchmark ONNX Runtime FP32...")
    ort_opts = ort.SessionOptions()
    ort_opts.intra_op_num_threads = NUM_THREADS
    ort_opts.inter_op_num_threads = 1
    ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    ort_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    ort_fp32_in = ort_fp32.get_inputs()[0].name
    pass_results["ONNX Runtime FP32"] = time_execution(lambda: ort_fp32.run(None, {ort_fp32_in: dummy_np}))
    del ort_fp32

    # 3. ONNX Runtime INT8
    print("  [3/5] Benchmark ONNX Runtime INT8 (Static QDQ)...")
    ort_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    ort_int8_in = ort_int8.get_inputs()[0].name
    pass_results["ONNX Runtime INT8"] = time_execution(lambda: ort_int8.run(None, {ort_int8_in: dummy_np}))
    del ort_int8

    # 4. OpenVINO FP32
    print("  [4/5] Benchmark OpenVINO FP32...")
    core = ov.Core()
    core.set_property("CPU", {"INFERENCE_NUM_THREADS": NUM_THREADS})
    ov_fp32_model = core.compile_model(core.read_model(str(OV_FP32_PATH)), "CPU")
    ov_fp32_out = ov_fp32_model.output(0)
    pass_results["OpenVINO FP32"] = time_execution(lambda: ov_fp32_model([dummy_np])[ov_fp32_out])
    del ov_fp32_model

    # 5. OpenVINO INT8
    print("  [5/5] Benchmark OpenVINO INT8 (NNCF PTQ)...")
    ov_int8_model = core.compile_model(core.read_model(str(OV_INT8_PATH)), "CPU")
    ov_int8_out = ov_int8_model.output(0)
    pass_results["OpenVINO INT8"] = time_execution(lambda: ov_int8_model([dummy_np])[ov_int8_out])
    del ov_int8_model

    return pass_results


def main():
    print("=" * 72)
    print("PASSO 8: ESECUZIONE BENCHMARK FINALI E VERIFICA DI RIPETIBILITA'")
    print("=" * 72)
    print(f"Configurazione: batch_size=1 | Thread={NUM_THREADS} | Warmup={WARMUP_RUNS} | Iterazioni={BENCH_RUNS}")

    # Esecuzione Passaggio 1
    run_1 = benchmark_single_pass(run_id=1)

    print("\nPausa di 5 secondi per stabilizzazione termica prima del Run 2...")
    time.sleep(5)

    # Esecuzione Passaggio 2 (Ripetibilita')
    run_2 = benchmark_single_pass(run_id=2)

    # Verifica di ripetibilita' (confronto mediana Run 1 vs Run 2)
    print("\n" + "=" * 72)
    print("VERIFICA DI RIPETIBILITA' (Run 1 vs Run 2 - Latenza Mediana)")
    print("=" * 72)
    print(f"{'Configurazione':<24} | {'Run 1 (ms)':<12} | {'Run 2 (ms)':<12} | {'Scostamento':<12}")
    print("-" * 72)
    
    final_averaged_results = {}
    for cfg in run_1:
        m1 = run_1[cfg]["median_ms"]
        m2 = run_2[cfg]["median_ms"]
        diff_pct = abs(m1 - m2) / m1 * 100.0
        print(f"{cfg:<24} | {m1:>10.2f} ms | {m2:>10.2f} ms | {diff_pct:>10.2f}%")

        # Media delle due sessioni per la massima robustezza statistica
        final_averaged_results[cfg] = {
            "median_ms": round((m1 + m2) / 2.0, 2),
            "p95_ms": round((run_1[cfg]["p95_ms"] + run_2[cfg]["p95_ms"]) / 2.0, 2),
            "throughput_fps": round((run_1[cfg]["throughput_fps"] + run_2[cfg]["throughput_fps"]) / 2.0, 2)
        }

    # Salvataggio dati consolidati per la tabella del Passo 9 e il README del Passo 10
    with open(OUTPUT_BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(final_averaged_results, f, indent=2)

    print("\n" + "=" * 72)
    print(f"Misure consolidate e salvate in: '{OUTPUT_BENCHMARK_JSON}'")
    print("=" * 72)


if __name__ == "__main__":
    main()