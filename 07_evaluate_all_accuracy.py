"""
Passo 7: Valutazione comparativa dell'Accuracy Top-1 su 2.000 immagini di validation.
Confronta 5 configurazioni:
  - PyTorch FP32
  - ONNX Runtime FP32
  - ONNX Runtime INT8
  - OpenVINO FP32
  - OpenVINO INT8
Metriche:
  (a) Modalita' 100 classi (sottoinsieme mascherato con mapping WordNet ID)
  (b) Modalita' 1000 classi (argmax su tutte le classi ImageNet-1k)
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort
import openvino as ov
from tqdm import tqdm


MANIFEST_PATH = Path("data/dataset_manifest.json")
ONNX_FP32_PATH = Path("models/resnet50_fp32.onnx")
ONNX_INT8_PATH = Path("models/resnet50_ort_int8.onnx")
OV_FP32_PATH = Path("models/ov_models/resnet50_fp32.xml")
OV_INT8_PATH = Path("models/ov_models/resnet50_int8.xml")

BATCH_SIZE_EVAL = 16
NUM_THREADS = 4


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img = Image.open(item["rel_path"]).convert("RGB")
        tensor = self.transform(img)
        return tensor, item["idx_100"], item["idx_1000"]


def evaluate_runner(dataloader, predict_fn, sub_to_1k_indices, desc="Valutazione"):
    correct_100 = 0
    correct_1000 = 0
    total = 0

    for batch_tensors, labels_100, labels_1000 in tqdm(dataloader, desc=desc, leave=False):
        batch_np = batch_tensors.numpy()
        labels_100 = labels_100.numpy()
        labels_1000 = labels_1000.numpy()

        # Inferenza del modello: restituisce logit con forma (B, 1000)
        logits_1000 = predict_fn(batch_np)

        # Modalita' (b): argmax su 1000 classi
        preds_1000 = np.argmax(logits_1000, axis=1)
        correct_1000 += np.sum(preds_1000 == labels_1000)

        # Modalita' (a): argmax sui 100 logit mascherati
        logits_sub = logits_1000[:, sub_to_1k_indices]
        preds_100 = np.argmax(logits_sub, axis=1)
        correct_100 += np.sum(preds_100 == labels_100)

        total += len(labels_100)

    acc_100 = (correct_100 / total) * 100.0
    acc_1000 = (correct_1000 / total) * 100.0
    return acc_100, acc_1000


def main():
    print("=" * 70)
    print("PASSO 7: VALUTAZIONE ACCURACY SU VALIDATION SET (2.000 IMMAGINI)")
    print("=" * 70)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    sub_to_1k_indices = np.array([c["idx_1000"] for c in manifest["classes_mapping"]])
    weights = models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    dataset_val = ValidationDataset(manifest["val_samples"], transform)
    dataloader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=0
    )

    results = []

    # 1. PyTorch FP32
    print("\n1/5. Caricamento PyTorch FP32...")
    model_pt = models.resnet50(weights=weights).eval()
    torch.set_num_threads(NUM_THREADS)

    def predict_pt(batch_np):
        with torch.no_grad():
            return model_pt(torch.from_numpy(batch_np)).numpy()

    acc_100, acc_1000 = evaluate_runner(dataloader, predict_pt, sub_to_1k_indices, "PyTorch FP32")
    results.append(("PyTorch", "FP32", acc_100, acc_1000))
    del model_pt  # Libera RAM

    # 2. ONNX Runtime FP32
    print("2/5. Caricamento ONNX Runtime FP32...")
    ort_opts = ort.SessionOptions()
    ort_opts.intra_op_num_threads = NUM_THREADS
    sess_ort_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    in_name_ort_fp32 = sess_ort_fp32.get_inputs()[0].name

    def predict_ort_fp32(batch_np):
        return sess_ort_fp32.run(None, {in_name_ort_fp32: batch_np})[0]

    acc_100, acc_1000 = evaluate_runner(dataloader, predict_ort_fp32, sub_to_1k_indices, "ORT FP32")
    results.append(("ONNX Runtime", "FP32", acc_100, acc_1000))
    del sess_ort_fp32

    # 3. ONNX Runtime INT8
    print("3/5. Caricamento ONNX Runtime INT8...")
    sess_ort_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    in_name_ort_int8 = sess_ort_int8.get_inputs()[0].name

    def predict_ort_int8(batch_np):
        return sess_ort_int8.run(None, {in_name_ort_int8: batch_np})[0]

    acc_100, acc_1000 = evaluate_runner(dataloader, predict_ort_int8, sub_to_1k_indices, "ORT INT8")
    results.append(("ONNX Runtime", "INT8 (Static QDQ)", acc_100, acc_1000))
    del sess_ort_int8

    # 4. OpenVINO FP32
    print("4/5. Caricamento OpenVINO FP32 (CPU)...")
    core = ov.Core()
    core.set_property("CPU", {"INFERENCE_NUM_THREADS": NUM_THREADS})
    ov_fp32_compiled = core.compile_model(core.read_model(str(OV_FP32_PATH)), "CPU")
    ov_fp32_output = ov_fp32_compiled.output(0)

    def predict_ov_fp32(batch_np):
        return ov_fp32_compiled([batch_np])[ov_fp32_output]

    acc_100, acc_1000 = evaluate_runner(dataloader, predict_ov_fp32, sub_to_1k_indices, "OpenVINO FP32")
    results.append(("OpenVINO", "FP32", acc_100, acc_1000))
    del ov_fp32_compiled

    # 5. OpenVINO INT8
    print("5/5. Caricamento OpenVINO INT8 (CPU)...")
    ov_int8_compiled = core.compile_model(core.read_model(str(OV_INT8_PATH)), "CPU")
    ov_int8_output = ov_int8_compiled.output(0)

    def predict_ov_int8(batch_np):
        return ov_int8_compiled([batch_np])[ov_int8_output]

    acc_100, acc_1000 = evaluate_runner(dataloader, predict_ov_int8, sub_to_1k_indices, "OpenVINO INT8")
    results.append(("OpenVINO", "INT8 (NNCF PTQ)", acc_100, acc_1000))
    del ov_int8_compiled

    # Stampa Tabella Riassuntiva
    print("\n" + "=" * 70)
    print("RISULTATI FINALI ACCURACY TOP-1 (Validation Set: 2.000 Immagini)")
    print("=" * 70)
    print(f"{'Runtime':<16} | {'Precisione':<18} | {'Top-1 (100 cl)':<14} | {'Top-1 (1000 cl)':<14}")
    print("-" * 70)
    for r in results:
        print(f"{r[0]:<16} | {r[1]:<18} | {r[2]:>13.2f}% | {r[3]:>13.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()