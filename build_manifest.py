"""
Cita CSV fajlove sa labelama (train/dev/test_sent_emo.csv) i za svaki red
proverava da li odgovarajuci .mp4 fajl postoji u odgovarajucem folderu
Rezultat je manifest.csv sa kolonama:
    split, dialogue_id, utterance_id, video_path, emotion, sentiment

Ovaj manifest koristimo u svim narednim koracima (ekstrakcija audija i
frejmova lica), tako da ne moramo ponovo da petljamo sa CSV fajlovima
"""

import pandas as pd
from pathlib import Path


MELD_ROOT = Path(r"C:\Users\XYZ\Desktop\MELD.Raw")


SPLITS = {
    "train": ("train_sent_emo.csv", "train_splits"),
    "dev":   ("dev_sent_emo.csv", "dev_splits_complete"),         #dev je val
    "test":  ("test_sent_emo.csv", "output_repeated_splits_test"),
}


def build():
    if not MELD_ROOT.exists():
        raise FileNotFoundError(
            f"MELD_ROOT ne postoji: {MELD_ROOT}. Otvori ovaj fajl i ispravi putanju."
        )

    rows = []                #Prazna lista za dataframe
    for split, (csv_name, video_dir) in SPLITS.items():
        csv_path = MELD_ROOT / csv_name
        if not csv_path.exists():
            print(f"[UPOZORENJE] Ne postoji CSV: {csv_path} - preskacem {split}")
            continue

        df = pd.read_csv(csv_path)          #dataframe sa svim redovima iz CSV-a
        video_folder = MELD_ROOT / video_dir

        found, missing = 0, 0
        missing_examples = []

        for _, row in df.iterrows():   #Iteracija korz svaki red uzima dialog i utterance
            dia = row["Dialogue_ID"]
            utt = row["Utterance_ID"]
            video_path = video_folder / f"dia{dia}_utt{utt}.mp4"

            if video_path.exists():
                found += 1
                rows.append({
                    "split": split,
                    "dialogue_id": dia,
                    "utterance_id": utt,
                    "video_path": str(video_path),
                    "emotion": row["Emotion"],
                    "sentiment": row["Sentiment"],
                })
            else:
                missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(video_path.name)

        print(f"[{split}] ukupno u CSV-u: {len(df)} | pronadjeno: {found} | nedostaje: {missing}")
        if missing_examples:
            print(f"    primeri fajlova koji nedostaju: {missing_examples}")

    manifest = pd.DataFrame(rows)
    out_path = MELD_ROOT / "manifest.csv"
    manifest.to_csv(out_path, index=False)

    print()
    print(f"Ukupno redova u manifestu: {len(manifest)}")
    print(f"Sacuvano u: {out_path}")
    print()
    print("Distribucija po split-u:")
    print(manifest["split"].value_counts())
    print()
    print("Distribucija emocija (ukupno, svi split-ovi):")
    print(manifest["emotion"].value_counts())


if __name__ == "__main__":
    build()
