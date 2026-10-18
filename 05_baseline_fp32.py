"""
Passo 5: Confronto Baseline FP32 tra PyTorch nativo e ONNX Runtime.
- Misura Latenza e Throughput su CPU (4 thread, batch_size=1)
- Misura Accuracy Top-1 sulle 2.000 immagini di validation:
    (a) Modalita' 100 classi (logit mascherati sul sottoinsieme)
    (b) Modalita' 1000 classi (argmax su tutte le classi ImageNet-1k)
"""
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort
from tqdm import tqdm


DATA_MANIFEST = Path("data/dataset_manifest.json")
ONNX_MODEL_PATH = Path("models/resnet50_fp32.onnx")
NUM_THREADS = 4
BATCH_SIZE_EVAL = 16  # Batch size per velocizzare la valutazione dell'accuracy


def benchmark_pytorch_latency(model, num_threads=4, warmup=15, runs=100):
    """Misura latenza e throughput di PyTorch FP32 alle stesse condizioni di ONNX Runtime."""
    torch.set_num_threads(num_threads)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Misure
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies)
    mean_lat = np.mean(latencies)
    return {
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "mean_ms": mean_lat,
        "throughput_fps": 1000.0 / mean_lat
    }


class ValidationDataset(torch.utils.data.Dataset):
    """Dataset leggero che legge le immagini una alla volta dal manifest senza saturare la RAM."""
    def __init__(self, manifest_data, transform):
        self.samples = manifest_data["val_samples"]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["rel_path"]).convert("RGB")
        tensor = self.transform(img)
        return tensor, item["idx_100"], item["idx_1000"]


def evaluate_accuracy(dataloader, run_fn, sub_to_1k_indices):
    """
    Calcola l'Accuracy Top-1 sia sulle 100 classi (mascherate) sia sulle 1000 classi.
    run_fn accetta un tensore NumPy (B, 3, 224, 224) e restituisce i logit (B, 1000).
    """
    correct_100 = 0
    correct_1000 = 0
    total = 0

    for batch_tensors, labels_100, labels_1000 in tqdm(dataloader, desc="Valutazione", leave=False):
        batch_np = batch_tensors.numpy()
        labels_100 = labels_100.numpy()
        labels_1000 = labels_1000.numpy()

        # Inferenza
        logits_1000 = run_fn(batch_np)  # Shape: (B, 1000)

        # Modalita' (b): Argmax su tutte le 1000 classi
        preds_1000 = np.argmax(logits_1000, axis=1)
        correct_1000 += np.sum(preds_1000 == labels_1000)

        # Modalita' (a): Estrazione e argmax solo sui 100 indici del sottoinsieme
        logits_sub = logits_1000[:, sub_to_1k_indices]  # Shape: (B, 100)
        preds_100 = np.argmax(logits_sub, axis=1)
        correct_100 += np.sum(preds_100 == labels_100)

        total += len(labels_100)

    acc_100 = (correct_100 / total) * 100.0
    acc_1000 = (correct_1000 / total) * 100.0
    return acc_100, acc_1000


def main():
    print("=" * 68)
    print("PASSO 5: BASELINE FP32 - PYTORCH vs ONNX RUNTIME")
    print("=" * 68)

    # 1. Carica configurazione e manifest
    with open(DATA_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Vettore di mappatura: data la colonna j (0..99), restituisce l'indice corrispondente in ImageNet-1k (0..999)
    sub_to_1k_indices = np.array([c["idx_1000"] for c in manifest["classes_mapping"]])

    # 2. Carica modello PyTorch e relative trasformazioni ufficiali
    print("Caricamento ResNet-50 PyTorch...")
    weights = models.ResNet50_Weights.DEFAULT
    model_pt = models.resnet50(weights=weights)
    model_pt.eval()
    preprocess = weights.transforms()

    # 3. Inizializza Sessione ONNX Runtime
    print(f"Caricamento modello ONNX: {ONNX_MODEL_PATH}...")
    ort_opts = ort.SessionOptions()
    ort_opts.intra_op_num_threads = NUM_THREADS
    ort_opts.inter_op_num_threads = 1
    ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    ort_input_name = ort_session.get_inputs()[0].name

    # 4. Misura Latenza PyTorch FP32
    print(f"\nMisurazione latenza PyTorch FP32 (CPU, {NUM_THREADS} thread, batch_size=1)...")
    pt_perf = benchmark_pytorch_latency(model_pt, num_threads=NUM_THREADS, warmup=15, runs=100)

    # 5. Misura Latenza ONNX Runtime FP32 (usando il runner analogo)
    print(f"Misurazione latenza ONNX Runtime FP32 (CPU, {NUM_THREADS} thread, batch_size=1)...")
    # Tensore fittizio per il test di velocità ONNX
    dummy_ort = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warm-up ONNX
    for _ in range(15):
        _ = ort_session.run(None, {ort_input_name: dummy_ort})
    
    # Benchmark ONNX
    ort_lats = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = ort_session.run(None, {ort_input_name: dummy_ort})
        t1 = time.perf_counter()
        ort_lats.append((t1 - t0) * 1000.0)
    
    ort_lats = np.array(ort_lats)
    ort_mean = np.mean(ort_lats)
    ort_perf = {
        "median_ms": np.median(ort_lats),
        "p95_ms": np.percentile(ort_lats, 95),
        "mean_ms": ort_mean,
        "throughput_fps": 1000.0 / ort_mean
    }

    # 6. Preparazione DataLoader per il calcolo dell'Accuracy
    dataset_val = ValidationDataset(manifest, transform=preprocess)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=0  # 0 evita problemi di multiprocessing su Windows
    )

    # Funzioni di inferenza per la valutazione
    def run_pt_batch(batch_np):
        with torch.no_grad():
            tensor = torch.from_numpy(batch_np)
            return model_pt(tensor).numpy()

    def run_ort_batch(batch_np):
        return ort_session.run(None, {ort_input_name: batch_np})[0]

    # 7. Valutazione Accuracy
    print(f"\nCalcolo Accuracy su {len(dataset_val)} immagini (PyTorch FP32)...")
    pt_acc_100, pt_acc_1000 = evaluate_accuracy(val_loader, run_pt_batch, sub_to_1k_indices)

    print(f"Calcolo Accuracy su {len(dataset_val)} immagini (ONNX Runtime FP32)...")
    ort_acc_100, ort_acc_1000 = evaluate_accuracy(val_loader, run_ort_batch, sub_to_1k_indices)

    # 8. Stampa Tabella di Confronto
    print("\n" + "=" * 68)
    print("CONFRONTO BASELINE FP32 (CPU - 4 Thread)")
    print("=" * 68)
    print(f"{'Metrica':<28} | {'PyTorch FP32':<16} | {'ONNX Runtime FP32':<16}")
    print("-" * 68)
    print(f"{'Latenza Mediana (p50)':<28} | {pt_perf['median_ms']:>13.2f} ms | {ort_perf['median_ms']:>13.2f} ms")
    print(f"{'Latenza 95° perc (p95)':<28} | {pt_perf['p95_ms']:>13.2f} ms | {ort_perf['p95_ms']:>13.2f} ms")
    print(f"{'Throughput (FPS)':<28} | {pt_perf['throughput_fps']:>13.2f} fps| {ort_perf['throughput_fps']:>13.2f} fps")
    print(f"{'Top-1 Accuracy (100 classi)':<28} | {pt_acc_100:>15.2f}% | {ort_acc_100:>15.2f}%")
    print(f"{'Top-1 Accuracy (1000 classi)':<28} | {pt_acc_1000:>15.2f}% | {ort_acc_1000:>15.2f}%")
    print("=" * 68)


if __name__ == "__main__":
    main()