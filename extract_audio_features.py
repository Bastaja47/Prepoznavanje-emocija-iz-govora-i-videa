"""
Korak 4 (audio) - Ekstrakcija audio obelezja pomocu wav2vec2.

Koristimo facebook/wav2vec2-base (pretrained, BEZ fine-tuning-a - isto kao
u referentnom radu) kao fiksni ekstraktor obelezja. Model se ne dotrenirava,
samo se koristi njegov encoder da iz normalizovanog 6s audio signala izvuce
sekvencu latentnih reprezentacija dimenzije 768.

Cuvamo SIROVU vremensku sekvencu (T, 768) po klipu - temporal pooling
(mean+std) i "bez poolinga" varijanta se racunaju kasnije, u Koraku 5, iz
ovog istog keша, da ne bismo duplirali podatke na disku.

Kako je svaki audio fajl fiksne duzine (6s @ 16kHz), T (broj vremenskih
koraka na izlazu wav2vec2 enkodera) je isti za sve klipove.

Rezultat: data/audio_features/{split}/dia{D}_utt{U}.npy  oblika (T, 768)

Skripta automatski preskace klipove kojima audio fajl ne postoji (npr. onaj
1 korumpovani iz Koraka 3), i "resumable" je - ako izlazni fajl vec postoji,
preskace ga.
"""

import numpy as np
import pandas as pd
import torch
import soundfile as sf
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm

# ------------------------------------------------------------------
MELD_ROOT = Path(r"C:\Users\XYZ\Desktop\MELD.Raw")
# ------------------------------------------------------------------

MANIFEST = MELD_ROOT / "manifest.csv"
AUDIO_DIR = MELD_ROOT / "data" / "audio"
OUT_DIR = MELD_ROOT / "data" / "audio_features"
SAMPLE_RATE = 16000
MODEL_NAME = "facebook/wav2vec2-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Koristim device: {device}")
print(f"Ucitavam {MODEL_NAME} (prvi put ce skinuti tezine, ~360MB)...")

processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME).eval().to(device)
for p in model.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def extract_clip_embedding(wav_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == SAMPLE_RATE, f"Ocekivano {SAMPLE_RATE}Hz, dobijeno {sr}Hz"

    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    out = model(input_values)
    hidden = out.last_hidden_state.squeeze(0)  # (T, 768)
    return hidden.cpu().numpy()


def main():
    manifest = pd.read_csv(MANIFEST)
    for split in manifest["split"].unique():
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    skipped_missing = 0
    failed = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        split = row["split"]
        dia, utt = row["dialogue_id"], row["utterance_id"]
        audio_path = AUDIO_DIR / split / f"dia{dia}_utt{utt}.wav"
        out_path = OUT_DIR / split / f"dia{dia}_utt{utt}.npy"

        if not audio_path.exists():
            skipped_missing += 1
            continue
        if out_path.exists():
            continue

        try:
            emb = extract_clip_embedding(audio_path)
            np.save(out_path, emb.astype(np.float32))
        except Exception as e:
            failed.append(f"{audio_path} | {e}")

    print(f"\nPreskoceno (nema audio fajl): {skipped_missing}")
    print(f"Neuspesnih: {len(failed)}")
    if failed:
        fail_path = OUT_DIR / "failed.csv"
        pd.Series(failed).to_csv(fail_path, index=False)
        print(f"Detalji o greskama: {fail_path}")


if __name__ == "__main__":
    main()
