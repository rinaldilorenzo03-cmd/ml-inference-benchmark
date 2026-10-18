"""
Passo 6: Quantizzazione Post-Training (PTQ) INT8.
Genera due modelli INT8 a partire dal modello FP32 e dalle 300 immagini di calibrazione:
  1. ONNX Runtime INT8 (Static QDQ, Per-Channel sui pesi, file .onnx)
  2. OpenVINO / NNCF INT8 (Intermediate Representation, file .xml + .bin)
"""
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.models as models

# Librerie di quantizzazione
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)
import openvino as ov
import nncf


FP32_ONNX_PATH = Path("models/resnet50_fp32.onnx")
ORT_INT8_PATH = Path("models/resnet50_ort_int8.onnx")
OV_DIR = Path("models/ov_models")
OV_FP32_PATH = OV_DIR / "resnet50_fp32.xml"
OV_INT8_PATH = OV_DIR / "resnet50_int8.xml"
MANIFEST_PATH = Path("data/dataset_manifest.json")


# -----------------------------------------------------------------------------
# 1. CALIBRATION DATA READER PER ONNX RUNTIME
# -----------------------------------------------------------------------------
class ResNetCalibrationDataReader(CalibrationDataReader):
    """
    Fornisce un'immagine alla volta al quantizzatore di ONNX Runtime.
    Legge dal manifest, applica il preprocessing ufficiale e restituisce un dizionario {input_name: numpy_array}.
    """
    def __init__(self, samples, transform_fn, input_name):
        self.samples = samples
        self.transform_fn = transform_fn
        self.input_name = input_name
        self.iterator = iter(self.samples)

    def get_next(self):
        try:
            item = next(self.iterator)
            img = Image.open(item["rel_path"]).convert("RGB")
            # Preprocessing PyTorch -> shape (1, 3, 224, 224)
            tensor_np = self.transform_fn(img).unsqueeze(0).numpy()
            return {self.input_name: tensor_np}
        except StopIteration:
            return None

    def rewind(self):
        self.iterator = iter(self.samples)


# -----------------------------------------------------------------------------
# 2. QUANTIZZAZIONE CON ONNX RUNTIME
# -----------------------------------------------------------------------------
def quantize_with_onnxruntime(calib_samples, preprocess_fn):
    print("\n" + "=" * 65)
    print("AVVIO QUANTIZZAZIONE INT8: ONNX RUNTIME (PTQ STATICA)")
    print("=" * 65)

    # Rileva il nome dell'input dal modello FP32
    session = ort.InferenceSession(str(FP32_ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    data_reader = ResNetCalibrationDataReader(calib_samples, preprocess_fn, input_name)

    print("Calibrazione e quantizzazione in corso su 300 campioni...")
    quantize_static(
        model_input=str(FP32_ONNX_PATH),
        model_output=str(ORT_INT8_PATH),
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,          # Formato moderno Quantize-Dequantize
        per_channel=True,                      # Pesi quantizzati per canale (maggiore accuratezza)
        weight_type=QuantType.QInt8,           # Pesi INT8 simmetrici con segno [-127, 127]
        activation_type=QuantType.QUInt8,      # Attivazioni UINT8 asimmetriche [0, 255] (ottimale con ReLU)
        calibrate_method=CalibrationMethod.MinMax
    )
    print(f"Modello ONNX INT8 salvato su: '{ORT_INT8_PATH}'")


# -----------------------------------------------------------------------------
# 3. QUANTIZZAZIONE CON OPENVINO / NNCF
# -----------------------------------------------------------------------------
def quantize_with_openvino_nncf(calib_samples, preprocess_fn):
    print("\n" + "=" * 65)
    print("AVVIO QUANTIZZAZIONE INT8: OPENVINO / NNCF (PTQ)")
    print("=" * 65)
    OV_DIR.mkdir(parents=True, exist_ok=True)

    core = ov.Core()

    # Conversione iniziale da ONNX a OpenVINO IR FP32 (se non gia' presente)
    if not OV_FP32_PATH.exists():
        print("Conversione modello ONNX in OpenVINO IR (FP32)...")
        ov_fp32_model = core.read_model(str(FP32_ONNX_PATH))
        ov.save_model(ov_fp32_model, str(OV_FP32_PATH))
        print(f"Modello OpenVINO FP32 salvato su: '{OV_FP32_PATH}'")
    else:
        ov_fp32_model = core.read_model(str(OV_FP32_PATH))

    # Preparazione dataset per NNCF
    def nncf_transform(item):
        img = Image.open(item["rel_path"]).convert("RGB")
        return preprocess_fn(img).unsqueeze(0).numpy()

    nncf_calibration_dataset = nncf.Dataset(calib_samples, nncf_transform)

    print("Quantizzazione NNCF in corso (calcolo statistiche e correzione bias)...")
    # nncf.quantize inserisce le trasformazioni ottimali e applica bias-correction
    quantized_ov_model = nncf.quantize(
        model=ov_fp32_model,
        calibration_dataset=nncf_calibration_dataset,
        subset_size=len(calib_samples)
    )

    ov.save_model(quantized_ov_model, str(OV_INT8_PATH))
    print(f"Modello OpenVINO INT8 salvato su: '{OV_INT8_PATH}' (.xml + .bin)")


# -----------------------------------------------------------------------------
# 4. CONFRONTO DIMENSIONI FILE SU DISCO
# -----------------------------------------------------------------------------
def print_model_sizes():
    print("\n" + "=" * 65)
    print("CONFRONTO FOOTPRINT SU DISCO (FP32 vs INT8)")
    print("=" * 65)

    size_onnx_fp32 = os.path.getsize(FP32_ONNX_PATH) / (1024 * 1024)
    size_onnx_int8 = os.path.getsize(ORT_INT8_PATH) / (1024 * 1024)

    # Per OpenVINO la dimensione reale dei pesi risiede nel file binario .bin
    bin_fp32 = OV_FP32_PATH.with_suffix(".bin")
    bin_int8 = OV_INT8_PATH.with_suffix(".bin")
    size_ov_fp32 = os.path.getsize(bin_fp32) / (1024 * 1024) if bin_fp32.exists() else 0.0
    size_ov_int8 = os.path.getsize(bin_int8) / (1024 * 1024) if bin_int8.exists() else 0.0

    print(f"{'Formato':<24} | {'FP32 (MB)':<12} | {'INT8 (MB)':<12} | {'Compressione':<12}")
    print("-" * 65)
    print(f"{'ONNX Runtime (.onnx)':<24} | {size_onnx_fp32:>9.2f} MB | {size_onnx_int8:>9.2f} MB | {size_onnx_fp32/size_onnx_int8:>9.2f}x")
    print(f"{'OpenVINO IR (.bin)':<24} | {size_ov_fp32:>9.2f} MB | {size_ov_int8:>9.2f} MB | {size_ov_fp32/size_ov_int8:>9.2f}x")
    print("=" * 65)


def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    calib_samples = manifest["calib_samples"]
    weights = models.ResNet50_Weights.DEFAULT
    preprocess_fn = weights.transforms()

    # 1. Quantizzazione ONNX Runtime
    quantize_with_onnxruntime(calib_samples, preprocess_fn)

    # 2. Quantizzazione OpenVINO / NNCF
    quantize_with_openvino_nncf(calib_samples, preprocess_fn)

    # 3. Riepilogo dimensioni
    print_model_sizes()


if __name__ == "__main__":
    main()