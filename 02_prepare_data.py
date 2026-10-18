"""
Passo 2 (Revisionato): Download in streaming e salvataggio bit-exact di:
  (a) Validation subset: 2.000 immagini dallo split 'test' di mini-imagenet
      (corrispondente al vero validation set non visto di ImageNet-1k)
  (b) Calibration subset: 300 immagini disgiunte dallo split 'train'
Preserva i byte JPEG originali senza ricompressione (zero generation loss)
e mappa le classi 0..99 a 0..999 tramite WordNet ID.
"""
import json
import urllib.request
from pathlib import Path
from datasets import load_dataset, Image as HFImage
from tqdm import tqdm

VAL_IMAGES_PER_CLASS = 20    # 20 * 100 classi = 2.000 immagini di validation
CALIB_IMAGES_PER_CLASS = 3   # 3 * 100 classi  = 300 immagini di calibrazione

DATA_DIR = Path("data")
VAL_DIR = DATA_DIR / "val"
CALIB_DIR = DATA_DIR / "calib"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

# Tabella storica standard (distribuita originariamente nel repository Keras deep-learning-models)
# per mappare l'indice ImageNet-1k (0..999) con il rispettivo WordNet ID (nXXXXXXXX).
IMAGENET_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


def get_imagenet1k_mapping(data_dir: Path) -> dict:
    """
    Scarica imagenet_class_index.json e indicizza per WordNet ID:
    wnid (es. 'n01532829') -> {'idx_1000': 12, 'name': 'house_finch'}
    """
    index_file = data_dir / "imagenet_class_index.json"
    if not index_file.exists():
        print("Scaricamento tabella WordNet ID (imagenet_class_index.json)...")
        urllib.request.urlretrieve(IMAGENET_INDEX_URL, index_file)

    with open(index_file, "r", encoding="utf-8") as f:
        raw_index = json.load(f)

    wnid_to_1k = {}
    for idx_str, (wnid, class_name) in raw_index.items():
        wnid_to_1k[wnid] = {
            "idx_1000": int(idx_str),
            "name": class_name
        }
    return wnid_to_1k


def collect_stratified_subset(
    split_name: str,
    target_per_class: int,
    output_dir: Path,
    wnid_to_1k: dict,
    int2str_fn
) -> list:
    """
    Itera in streaming ed estrae i byte originali delle immagini senza decodifica PIL,
    evitando artefatti di doppia compressione JPEG su disco.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_per_class = {i: 0 for i in range(100)}
    total_target = 100 * target_per_class
    saved_records = []

    print(f"\nScaricamento in streaming per '{split_name}' (target: {total_target} immagini)...")
    
    # Disattivando la decodifica (decode=False), sample["image"] restituisce un dizionario
    # contenente direttamente i byte binari grezzi compressi originali ('bytes')
    dataset_stream = load_dataset("timm/mini-imagenet", split=split_name, streaming=True)
    dataset_stream = dataset_stream.cast_column("image", HFImage(decode=False))

    pbar = tqdm(total=total_target, desc=f"Salvataggio {split_name}")
    for sample in dataset_stream:
        idx_100 = sample["label"]

        if counts_per_class[idx_100] >= target_per_class:
            continue

        wnid = int2str_fn(idx_100)
        if wnid not in wnid_to_1k:
            raise ValueError(f"ERRORE: WordNet ID {wnid} non presente in ImageNet-1k!")

        idx_1000 = wnid_to_1k[wnid]["idx_1000"]
        class_name = wnid_to_1k[wnid]["name"]

        img_num = counts_per_class[idx_100]
        filename = f"{split_name}_c{idx_100:03d}_{img_num:03d}_{wnid}_k{idx_1000:04d}.jpg"
        file_path = output_dir / filename

        # Scrittura bit-exact dei byte originali del file (nessuna ricompressione lossy)
        raw_bytes = sample["image"]["bytes"]
        file_path.write_bytes(raw_bytes)

        saved_records.append({
            "rel_path": str(file_path.as_posix()),
            "wnid": wnid,
            "class_name": class_name,
            "idx_100": idx_100,
            "idx_1000": idx_1000
        })

        counts_per_class[idx_100] += 1
        pbar.update(1)

        if len(saved_records) >= total_target:
            break

    pbar.close()
    return saved_records


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    wnid_to_1k = get_imagenet1k_mapping(DATA_DIR)

    print("Connessione a Hugging Face Hub per metadati di 'timm/mini-imagenet'...")
    ds_info = load_dataset("timm/mini-imagenet", split="test", streaming=True)
    label_feature = ds_info.features["label"]
    int2str_fn = label_feature.int2str

    subset_100_classes = []
    for idx_100 in range(100):
        wnid = int2str_fn(idx_100)
        info_1k = wnid_to_1k[wnid]
        subset_100_classes.append({
            "idx_100": idx_100,
            "wnid": wnid,
            "idx_1000": info_1k["idx_1000"],
            "name": info_1k["name"]
        })

    # 1. Validation Set: 2.000 immagini dallo split 'test'
    #    (corrispondente alle immagini non viste della validation originale ImageNet-1k)
    val_records = collect_stratified_subset(
        split_name="test",
        target_per_class=VAL_IMAGES_PER_CLASS,
        output_dir=VAL_DIR,
        wnid_to_1k=wnid_to_1k,
        int2str_fn=int2str_fn
    )

    # 2. Calibration Set: 300 immagini dallo split 'train' (disgiunte e indipendenti)
    calib_records = collect_stratified_subset(
        split_name="train",
        target_per_class=CALIB_IMAGES_PER_CLASS,
        output_dir=CALIB_DIR,
        wnid_to_1k=wnid_to_1k,
        int2str_fn=int2str_fn
    )

    manifest = {
        "dataset_source": "timm/mini-imagenet (100 ImageNet-1k classes)",
        "num_classes_subset": 100,
        "num_classes_model": 1000,
        "val_count": len(val_records),
        "calib_count": len(calib_records),
        "classes_mapping": subset_100_classes,
        "val_samples": val_records,
        "calib_samples": calib_records
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("PREPARAZIONE DATI COMPLETATA CON SUCCESSO")
    print("=" * 60)
    print(f"Validation:  {len(val_records)} immagini in '{VAL_DIR}' (split 'test', bit-exact)")
    print(f"Calibration: {len(calib_records)} immagini in '{CALIB_DIR}' (split 'train', bit-exact)")
    print(f"Manifest:    '{MANIFEST_PATH}'")


if __name__ == "__main__":
    main()