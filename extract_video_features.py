"""
Ekstrakcija video obelezja pomocu InceptionResnetV1
(pretrained na VGGFace2, iz facenet-pytorch).

Ovo je NAS PREDLOG video ekstraktora obelezja za projekat. Model je
pretreniran za prepoznavanje/verifikaciju identiteta lica (NE za
prepoznavanje emocija), i koristi se kao fiksni ekstraktor obelezja - bez
fine-tuning-a - potpuno analogno tome kako referentni rad koristi
wav2vec2/BERT kao generisce, ne-emocijski-nadgledane reprezentacije.

Za svaki klip imamo N_FRAMES=8 isecenih lica (iz Koraka 3), fiksne velicine
160x160 - sto je bas ocekivana ulazna velicina ovog modela. Svaki frejm se
propusta kroz model i dobija se 512-dimenzionalni embedding, pa cuvamo
sekvencu (N_FRAMES, 512) po klipu - temporal pooling (mean+std) i "bez
poolinga" varijanta se racunaju kasnije, u Koraku 5, iz ovog istog kesa.

Rezultat: data/video_features/{split}/dia{D}_utt{U}.npy  oblika (8, 512)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm

# ------------------------------------------------------------------
MELD_ROOT = Path(r"C:\Users\XYZ\Desktop\MELD.Raw")
# ------------------------------------------------------------------

MANIFEST = MELD_ROOT / "manifest.csv"
FACES_DIR = MELD_ROOT / "data" / "faces"
OUT_DIR = MELD_ROOT / "data" / "video_features"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Koristim device: {device}")
print("Ucitavam InceptionResnetV1 (vggface2) - prvi put ce skinuti tezine, ~107MB...")

model = InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(device)
for p in model.parameters():
    p.requires_grad_(False)


def fixed_image_standardization(x):
    # standardna facenet normalizacija piksela iz [0,255] u priblizno [-1,1]
    return (x - 127.5) / 128.0


@torch.no_grad()
def extract_clip_embedding(face_path):
    arr = np.load(face_path)  # (N_FRAMES, 3, H, W) uint8
    faces = torch.from_numpy(arr).float().to(device)
    faces = fixed_image_standardization(faces)

    emb = model(faces)  # (N_FRAMES, 512)
    return emb.cpu().numpy()


def main():
    manifest = pd.read_csv(MANIFEST)
    for split in manifest["split"].unique():
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    skipped_missing = 0
    failed = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        split = row["split"]
        dia, utt = row["dialogue_id"], row["utterance_id"]
        face_path = FACES_DIR / split / f"dia{dia}_utt{utt}.npy"
        out_path = OUT_DIR / split / f"dia{dia}_utt{utt}.npy"

        if not face_path.exists():
            skipped_missing += 1
            continue
        if out_path.exists():
            continue

        try:
            emb = extract_clip_embedding(face_path)
            np.save(out_path, emb.astype(np.float32))
        except Exception as e:
            failed.append(f"{face_path} | {e}")

    print(f"\nPreskoceno (nema fajl lica): {skipped_missing}")
    print(f"Neuspesnih: {len(failed)}")
    if failed:
        fail_path = OUT_DIR / "failed.csv"
        pd.Series(failed).to_csv(fail_path, index=False)
        print(f"Detalji o greskama: {fail_path}")


if __name__ == "__main__":
    main()
