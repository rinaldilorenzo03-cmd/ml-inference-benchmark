"""
Passo 4: Script di benchmarking universale per ONNX Runtime.
Progettato per girare in modo identico su architettura x86 (portatile) e ARM (Raspberry Pi 4).
Calcola: Mediana (p50), 95° Percentile (p95), Media, Deviazione Standard e Throughput (FPS).
"""
import argparse
import time
import psutil
import numpy as np
import onnxruntime as ort
from pathlib import Path


def run_benchmark(
    model_path: str,
    num_threads: int = 4,
    warmup_runs: int = 15,
    benchmark_runs: int = 100,
    batch_size: int = 1
):
    print("=" * 65)
    print("BENCHMARK INFERENZA: CONFIGURAZIONE ED ESECUZIONE")
    print("=" * 65)
    print(f"Modello target:        {model_path}")
    print(f"Thread intra-op:       {num_threads}")
    print(f"Cicli di warm-up:      {warmup_runs}")
    print(f"Iterazioni di misura:  {benchmark_runs}")
    print(f"Batch size:            {batch_size}")

    # 1. Configurazione delle opzioni di sessione del runtime
    session_options = ort.SessionOptions()
    
    # intra_op: parallelizza il calcolo all'interno del singolo operatore (es. convoluzione)
    session_options.intra_op_num_threads = num_threads
    
    # inter_op: esecuzione parallela di nodi indipendenti nel grafo (1 = esecuzione sequenziale)
    session_options.inter_op_num_threads = 1
    
    # Livello massimo di ottimizzazione del grafo (fusione di nodi, eliminazione costanti)
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # 2. Inizializzazione della sessione di inferenza
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )

    # 3. Rilevamento automatico della forma del tensore di input dal modello ONNX
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = list(input_meta.shape)
    
    # Gestione di dimensioni dinamiche (es. batch_size stringa o None)
    if input_shape[0] is None or isinstance(input_shape[0], str):
        input_shape[0] = batch_size
    else:
        input_shape[0] = batch_size

    print(f"Tensore di input:      nome='{input_name}', shape={input_shape}, dtype=float32")

    # Creazione del tensore di prova in memoria contigua (layout NCHW standard)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    ort_inputs = {input_name: dummy_input}

    # 4. FASE DI WARM-UP (stabilizzazione termica e riempimento cache L1/L2/L3)
    print(f"\nAvvio warm-up ({warmup_runs} iterazioni)...")
    for _ in range(warmup_runs):
        session.run(None, ort_inputs)
    print("Warm-up completato.")

    # 5. FASE DI MISURA EFFETTIVA
    latencies_ms = []
    print(f"Misurazione in corso ({benchmark_runs} iterazioni)...")
    
    for _ in range(benchmark_runs):
        t_start = time.perf_counter()
        session.run(None, ort_inputs)
        t_end = time.perf_counter()
        
        # Conversione da secondi a millisecondi
        latencies_ms.append((t_end - t_start) * 1000.0)

    # 6. ELABORAZIONE STATISTICA DEI RISULTATI
    latencies_ms = np.array(latencies_ms)
    
    median_lat = np.median(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)
    mean_lat = np.mean(latencies_ms)
    std_lat = np.std(latencies_ms)
    min_lat = np.min(latencies_ms)
    max_lat = np.max(latencies_ms)
    
    # Throughput calcolato come: (immagini per batch * 1000) / latenza_media_ms
    throughput = (batch_size * 1000.0) / mean_lat

    print("\n" + "=" * 65)
    print("RISULTATI DEL BENCHMARK")
    print("=" * 65)
    print(f"Latenza Mediana (p50): {median_lat:6.2f} ms")
    print(f"Latenza 95° perc (p95):{p95_lat:6.2f} ms")
    print(f"Latenza Media:         {mean_lat:6.2f} ms (± {std_lat:.2f} ms)")
    print(f"Latenza Min / Max:     {min_lat:6.2f} ms / {max_lat:6.2f} ms")
    print(f"Throughput:            {throughput:6.2f} frame/s (FPS)")
    print("=" * 65)

    return {
        "model": model_path,
        "median_ms": median_lat,
        "p95_ms": p95_lat,
        "mean_ms": mean_lat,
        "std_ms": std_lat,
        "throughput_fps": throughput
    }


def main():
    # Numero di core fisici calcolato automaticamente dal sistema
    default_threads = psutil.cpu_count(logical=False) or 4

    parser = argparse.ArgumentParser(description="Strumento di misura latenza e throughput ONNX.")
    parser.add_argument("--model", type=str, default="models/resnet50_fp32.onnx", help="Percorso del modello .onnx")
    parser.add_argument("--threads", type=int, default=default_threads, help="Numero di thread CPU per il calcolo")
    parser.add_argument("--warmup", type=int, default=15, help="Numero di passate di warm-up")
    parser.add_argument("--runs", type=int, default=100, help="Numero di iterazioni per il test")
    parser.add_argument("--batch-size", type=int, default=1, help="Dimensione del batch")

    args = parser.parse_args()
    run_benchmark(
        model_path=args.model,
        num_threads=args.threads,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()