"""
Passo 3: Caricamento ResNet-50 (PyTorch), export in ONNX e test di coerenza numerica.
"""
import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

def main():
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODELS_DIR / "resnet50_fp32.onnx"

    print("=" * 60)
    print("1. CARICAMENTO MODELLO PYTORCH (ResNet-50 FP32)")
    print("=" * 60)
    # Carica i pesi preaddestrati ufficiali di ImageNet-1k
    weights = models.ResNet50_Weights.DEFAULT
    model_pt = models.resnet50(weights=weights)
    model_pt.eval()  # Modalità inferenza (disattiva Dropout / BatchNorm training stats)
    print("Modello PyTorch caricato con successo in modalità eval.")

    print("\n" + "=" * 60)
    print("2. ESPORTAZIONE IN FORMATO ONNX")
    print("=" * 60)
    # Tensor di input fittizio con batch_size=1, 3 canali, risoluzione standard 224x224
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"Esportazione del grafo in corso su '{onnx_path}'...")
    torch.onnx.export(
        model_pt,
        dummy_input,
        str(onnx_path),
        export_params=True,        # Salva i pesi dentro il file ONNX
        opset_version=17,          # Versione dell'operator set ONNX moderna e compatibile
        do_constant_folding=True,  # Ottimizzazione: fonde i pesi costanti a compile-time
        input_names=["input"],     # Nome del nodo di input nel grafo
        output_names=["output"],   # Nome del nodo di output nel grafo
        dynamic_axes={
            "input": {0: "batch_size"},   # Rende dinamica la dimensione del batch (utile per il futuro)
            "output": {0: "batch_size"}
        }
    )
    print("Esportazione ONNX completata.")

    print("\n" + "=" * 60)
    print("3. VALIDAZIONE STRUTTURALE DEL FILE ONNX")
    print("=" * 60)
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("Il file ONNX rispetta formalmente le specifiche dello standard ONNX.")

    print("\n" + "=" * 60)
    print("4. VERIFICA DI COERENZA NUMERICA (PyTorch vs ONNX Runtime)")
    print("=" * 60)
    # Genera un'immagine fittizia di test (valori normalizzati casuali)
    test_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # A) Inferenza con PyTorch
    with torch.no_grad():
        out_pt = model_pt(test_tensor).numpy()

    # B) Inferenza con ONNX Runtime (CPUExecutionProvider)
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: test_tensor.numpy()}
    out_ort = ort_session.run(None, ort_inputs)[0]

    # C) Confronto delle differenze
    max_diff = np.max(np.abs(out_pt - out_ort))
    mean_diff = np.mean(np.abs(out_pt - out_ort))
    
    print(f"Massima differenza assoluta tra PyTorch e ONNX Runtime: {max_diff:.2e}")
    print(f"Differenza media assoluta:                               {mean_diff:.2e}")

    # Soglia di tolleranza standard per i test float32 (1e-5)
    tolerance = 1e-5
    if max_diff < tolerance:
        print("\nESITO VERIFICA: SUCCESSO! I logit coincidono perfettamente.")
    else:
        print("\nATTENZIONE: Differenza superiore alla tolleranza prevista.")

if __name__ == "__main__":
    main()