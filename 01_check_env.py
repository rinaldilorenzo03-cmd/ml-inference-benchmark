"""
Passo 1: Verifica dell'ambiente software e rilevamento hardware (CPU / iGPU).
"""
import platform
import sys
import psutil
import numpy as np
import torch
import torchvision
import onnx
import onnxruntime as ort
import openvino as ov
import nncf

def main():
    print("=" * 60)
    print("1. INFORMAZIONI SISTEMA E HARDWARE (per il README)")
    print("=" * 60)
    print(f"OS:                {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Python Version:    {sys.version.split()[0]}")
    print(f"Processore:        {platform.processor()}")

    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    print(f"Core CPU:          {physical_cores} fisici / {logical_cores} logici")
    print(f"RAM Totale:        {ram_gb} GB")

    print("\n" + "=" * 60)
    print("2. VERSIONI LIBRERIE")
    print("=" * 60)
    print(f"NumPy:             {np.__version__}")
    print(f"PyTorch:           {torch.__version__}")
    print(f"Torchvision:       {torchvision.__version__}")
    print(f"ONNX:              {onnx.__version__}")
    print(f"ONNX Runtime:      {ort.__version__}")
    print(f"OpenVINO:          {ov.__version__}")
    print(f"NNCF:              {nncf.__version__}")

    print("\n" + "=" * 60)
    print("3. DISPOSITIVI E PROVIDER DI ESECUZIONE RILEVATI")
    print("=" * 60)
    ort_providers = ort.get_available_providers()
    print(f"ONNX Runtime Providers: {ort_providers}")

    core = ov.Core()
    ov_devices = core.available_devices
    print(f"OpenVINO Devices:       {ov_devices}")

    for device in ov_devices:
        device_name = core.get_property(device, "FULL_DEVICE_NAME")
        print(f"  -> [{device}]: {device_name}")

    if "CPU" in ov_devices:
        try:
            optimization_caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
            print(f"  -> [CPU Optimization Capabilities]: {optimization_caps}")
        except Exception as e:
            print(f"  -> Impossibile leggere le optimization capabilities: {e}")

    print("\nVerifica completata con successo!")

if __name__ == "__main__":
    main()