"""
Passo 2: Download in streaming e preparazione di:
  (a) Validation subset: 2.000 immagini (20 per classe su 100 classi ImageNet)
  (b) Calibration subset: 300 immagini disgiunte (3 per classe dallo split train)
Con mappatura rigorosa tramite WordNet ID (wnid) verso le 1000 classi di ResNet-50.
"""
import json
import urllib.request
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Configurazione quantità immagini (bilanciate sulle 100 classi)
VAL_IMAGES_PER_CLASS = 20    # 20 * 100 classi = 2.000 immagini di validation
CALIB_IMAGES_PER_CLASS = 3   # 3 * 100 classi  = 300 immagini di calibrazione

DATA_DIR = Path("data")
VAL_DIR = DATA_DIR / "val"
CALIB_DIR = DATA_DIR / "calib"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"

# URL ufficiale usato da PyTorch/Torchvision per mappare indice (0..999) <-> WordNet ID
IMAGENET_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


def get_imagenet1k_mapping(data_dir: Path) -> dict:
    """
    Scarica imagenet_class_index.json e costruisce un dizionario:
    wnid (es. 'n01532829') -> {'idx_1000': 12, 'name': 'house_finch'}
    """
    index_file = data_dir / "imagenet_class_index.json"
    if not index_file.exists():
        print("Scaricamento tabella ufficiale WordNet ID (imagenet_class_index.json)...")
        urllib.request.urlretrieve(IMAGENET_INDEX_URL, index_file)

    with open(index_file, "r", encoding="utf-8") as f:
        raw_index = json.load(f)

    # raw_index ha formato: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
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
    Itera in streaming sul dataset senza saturare la RAM, raccogliendo esattamente
    'target_per_class' immagini per ciascuna delle 100 classi.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_per_class = {i: 0 for i in range(100)}
    total_target = 100 * target_per_class
    saved_records = []

    print(f"\nScaricamento in streaming per '{split_name}' (target: {total_target} immagini)...")
    dataset_stream = load_dataset("timm/mini-imagenet", split=split_name, streaming=True)

    pbar = tqdm(total=total_target, desc=f"Salvataggio {split_name}")
    for sample in dataset_stream:
        idx_100 = sample["label"]

        # Se abbiamo già abbastanza immagini per questa classe, saltiamo
        if counts_per_class[idx_100] >= target_per_class:
            continue

        # Ricava il WordNet ID (es. 'n01532829') e verifica che esista nelle 1000 classi
        wnid = int2str_fn(idx_100)
        if wnid not in wnid_to_1k:
            raise ValueError(f"ERRORE CRITICO: WordNet ID {wnid} non trovato in ImageNet-1k!")

        idx_1000 = wnid_to_1k[wnid]["idx_1000"]
        class_name = wnid_to_1k[wnid]["name"]

        # Conversione esplicita in RGB: alcune immagini di ImageNet sono in scala di grigi (1 canale)
        # o RGBA (4 canali) e farebbero crashare il preprocessing a 3 canali di ResNet-50.
        img = sample["image"].convert("RGB")

        # Nome file parlante: es. val_00012_n01532829_k0012.jpg
        img_num = counts_per_class[idx_100]
        filename = f"{split_name}_c{idx_100:03d}_{img_num:03d}_{wnid}_k{idx_1000:04d}.jpg"
        file_path = output_dir / filename
        img.save(file_path, format="JPEG", quality=95)

        saved_records.append({
            "rel_path": str(file_path.as_posix()),
            "wnid": wnid,
            "class_name": class_name,
            "idx_100": idx_100,     # Indice locale nel sottoinsieme (0..99)
            "idx_1000": idx_1000    # Indice globale in ResNet-50 ImageNet-1k (0..999)
        })

        counts_per_class[idx_100] += 1
        pbar.update(1)

        # Appena tutte le 100 classi hanno raggiunto il target, interrompiamo lo stream!
        if len(saved_records) >= total_target:
            break

    pbar.close()
    return saved_records


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carica mappatura ufficiale ImageNet-1k (WordNet ID -> 0..999)
    wnid_to_1k = get_imagenet1k_mapping(DATA_DIR)

    # 2. Ispeziona le feature del dataset per ottenere la funzione int2str (0..99 -> wnid)
    print("Connessione a Hugging Face Hub per leggere i metadati di 'timm/mini-imagenet'...")
    ds_info = load_dataset("timm/mini-imagenet", split="validation", streaming=True)
    label_feature = ds_info.features["label"]
    int2str_fn = label_feature.int2str

    # Costruisci la tabella di corrispondenza delle 100 classi
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

    print("\nVerifica prime 5 classi mappate (Sottoinsieme 0..99 -> ImageNet-1k 0..999):")
    for c in subset_100_classes[:5]:
        print(f"  Subset[{c['idx_100']:2d}] -> {c['wnid']} -> ImageNet1k[{c['idx_1000']:3d}] ({c['name']})")

    # 3. Scarica il Validation Set (2.000 immagini dallo split 'validation')
    val_records = collect_stratified_subset(
        split_name="validation",
        target_per_class=VAL_IMAGES_PER_CLASS,
        output_dir=VAL_DIR,
        wnid_to_1k=wnid_to_1k,
        int2str_fn=int2str_fn
    )

    # 4. Scarica il Calibration Set (300 immagini dallo split 'train', disgiunte dalla validation!)
    calib_records = collect_stratified_subset(
        split_name="train",
        target_per_class=CALIB_IMAGES_PER_CLASS,
        output_dir=CALIB_DIR,
        wnid_to_1k=wnid_to_1k,
        int2str_fn=int2str_fn
    )

    # 5. Salva il manifest JSON che useremo in tutti gli script successivi
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
    print("RIEPILOGO PREPARAZIONE DATI COMPLETATO")
    print("=" * 60)
    print(f"Immagini di Validation salvate:   {len(val_records)} in '{VAL_DIR}'")
    print(f"Immagini di Calibrazione salvate: {len(calib_records)} in '{CALIB_DIR}'")
    print(f"File Manifest con mappature:      '{MANIFEST_PATH}'")


if __name__ == "__main__":
    main()