import os
import sys
import time
import subprocess
import numpy as np
import onnxruntime as ort

MODELS = {
    "ResNet-50 FP32": "models/resnet50_fp32.onnx",
    "ResNet-50 INT8": "models/resnet50_ort_int8.onnx"
}

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
BATCH_SIZE = 1
THREADS = 4

def get_vcgen_data():
    """Interroga il firmware VideoCore per temperatura e registro throttling."""
    try:
        temp_out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("utf-8").strip()
        temp = float(temp_out.replace("temp=", "").replace("'C", ""))
        
        throttled_out = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("utf-8").strip()
        throttled = throttled_out.replace("throttled=", "")
        return temp, throttled
    except Exception:
        # Fallback se vcgencmd non è accessibile
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read().strip()) / 1000.0
        return temp, "N/A"

def verify_governor():
    """Garantisce che tutti i core siano bloccati alla frequenza massima."""
    gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(gov_path):
        with open(gov_path, "r") as f:
            governor = f.read().strip()
        if governor != "performance":
            print(f"[ERRORE METODOLOGICO] CPU governor attuale: '{governor}'.")
            print("Esegui: echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
            sys.exit(1)
    else:
        print("[WARN] Impossibile verificare scaling_governor via sysfs.")

def run_benchmark(model_name, model_path):
    print(f"\n=======================================================")
    print(f" Avvio Benchmark: {model_name}")
    print(f" File: {model_path}")
    print(f"=======================================================")
    
    # 1. Telemetria Iniziale
    t_start, throttled_start = get_vcgen_data()
    print(f"Telemetria pre-test:  Temp={t_start:.1f}°C | Throttling={throttled_start}")

    # 2. Configurazione Sessione ONNX Runtime
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    np.random.seed(42)
    dummy_input = np.random.randn(BATCH_SIZE, 3, 224, 224).astype(np.float32)

    # 3. Fase di Warm-up
    print(f"Esecuzione warm-up ({WARMUP_RUNS} iterazioni)...")
    for _ in range(WARMUP_RUNS):
        session.run([output_name], {input_name: dummy_input})

    # 4. Fase di Benchmark
    print(f"Campionamento latenza ({BENCHMARK_RUNS} iterazioni)...")
    latencies = []
    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        session.run([output_name], {input_name: dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # conversione in millisecondi

    # 5. Telemetria Finale
    t_end, throttled_end = get_vcgen_data()
    print(f"Telemetria post-test: Temp={t_end:.1f}°C | Throttling={throttled_end}")

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    std_lat = np.std(latencies)
    p90_lat = np.percentile(latencies, 90)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    throughput = 1000.0 / mean_lat

    return {
        "model": model_name,
        "mean": mean_lat,
        "median": median_lat,
        "std": std_lat,
        "p90": p90_lat,
        "p95": p95_lat,
        "p99": p99_lat,
        "throughput": throughput,
        "temp_start": t_start,
        "temp_end": t_end,
        "throttled_end": throttled_end
    }

def main():
    verify_governor()
    results = []

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[ERRORE] File {path} non trovato!")
            sys.exit(1)
        res = run_benchmark(name, path)
        results.append(res)
        time.sleep(3)  # Raffreddamento tra un test e l'altro

    # Stampa Riepilogo Formattato
    print("\n" + "=" * 90)
    print(f"{'Modello':<16} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'p95 (ms)':<9} | {'FPS':<8} | {'Temp (°C)':<11} | {'Throttled':<9}")
    print("-" * 90)
    for r in results:
        temp_str = f"{r['temp_start']:.1f}->{r['temp_end']:.1f}"
        print(f"{r['model']:<16} | {r['mean']:<10.2f} | {r['median']:<11.2f} | {r['p95']:<9.2f} | {r['throughput']:<8.2f} | {temp_str:<11} | {r['throttled_end']:<9}")
    print("=" * 90)

    # Confronto Quantizzazione (Speedup e Delta)
    fp32_mean = results[0]["mean"]
    int8_mean = results[1]["mean"]
    speedup = fp32_mean / int8_mean
    delta_pct = ((fp32_mean - int8_mean) / fp32_mean) * 100.0

    print(f"\nRapporto Prestazionale ARM Cortex-A72 (INT8 vs FP32):")
    print(f"  - Speedup Ratio: {speedup:.2f}x")
    print(f"  - Variazione Latenza: {delta_pct:+.2f}% (rispetto a FP32)")
    if speedup < 1.0:
        print("  - Nota: L'inferenza INT8 risulta PIÙ LENTA di FP32 (assenza di accelerazione asimddp).")
    else:
        print(f"  - Nota: L'inferenza INT8 offre un vantaggio di {speedup:.2f}x.")

if __name__ == "__main__":
    main()
