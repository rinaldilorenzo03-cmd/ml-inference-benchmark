import numpy as np
import onnxruntime as ort

fp32_path = "models/resnet50_fp32.onnx"
int8_path = "models/resnet50_ort_int8.onnx"

print("1. Inizializzazione sessioni ONNX Runtime...")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4  # saturazione dei 4 core Cortex-A72

sess_fp32 = ort.InferenceSession(fp32_path, sess_options=opts, providers=['CPUExecutionProvider'])
sess_int8 = ort.InferenceSession(int8_path, sess_options=opts, providers=['CPUExecutionProvider'])

input_name = sess_fp32.get_inputs()[0].name
output_name = sess_fp32.get_outputs()[0].name

print(f"   Input node:  {input_name} {sess_fp32.get_inputs()[0].shape}")
print(f"   Output node: {output_name} {sess_fp32.get_outputs()[0].shape}")

# Generazione di un tensore di test riproducibile (seed fissato)
np.random.seed(42)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

print("\n2. Esecuzione inferenza di test...")
out_fp32 = sess_fp32.run([output_name], {input_name: dummy_input})[0]
out_int8 = sess_int8.run([output_name], {input_name: dummy_input})[0]

# Metriche numeriche di confronto
mae = float(np.mean(np.abs(out_fp32 - out_int8)))
cos_sim = float(
    np.dot(out_fp32.flatten(), out_int8.flatten()) 
    / (np.linalg.norm(out_fp32) * np.linalg.norm(out_int8))
)

top1_fp32 = int(np.argmax(out_fp32))
top1_int8 = int(np.argmax(out_int8))

print("\n3. Risultati della validazione numerica:")
print(f"   - Shape output:       {out_fp32.shape}")
print(f"   - Top-1 Class FP32:   {top1_fp32}")
print(f"   - Top-1 Class INT8:   {top1_int8}")
print(f"   - Mean Absolute Error:{mae:.4f}")
print(f"   - Cosine Similarity:  {cos_sim:.5f}")

if cos_sim > 0.95:
    print("\n[OK] Validazione superata: modelli pronti per il benchmark.")
else:
    print("\n[WARN] Disallineamento numerico elevato tra FP32 e INT8.")
