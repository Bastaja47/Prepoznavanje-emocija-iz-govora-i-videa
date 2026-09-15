"""
Korak 4 (finalizacija) - Pravljenje finalnog manifesta za trening.

Uzima samo one klipove koji imaju USPESNO izvucena i audio i video
obelezja (data/audio_features i data/video_features). Ovo je jedini fajl
koji ce Dataset klasa u Koraku 5 citati - sve dalje (trening, evaluacija)
oslanja se na njega, tako da ne moramo nigde vise rucno da filtriramo
nedostajuce fajlove.

Rezultat: manifest_final.csv sa kolonama:
    split, dialogue_id, utterance_id, audio_feat_path, video_feat_path,
    emotion, sentiment
"""

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
MELD_ROOT = Path(r"C:\Users\XYZ\Desktop\MELD.Raw")
# ------------------------------------------------------------------

MANIFEST = MELD_ROOT / "manifest.csv"
AUDIO_FEAT_DIR = MELD_ROOT / "data" / "audio_features"
VIDEO_FEAT_DIR = MELD_ROOT / "data" / "video_features"


def main():
    manifest = pd.read_csv(MANIFEST)

    rows = []
    missing_audio, missing_video = 0, 0

    for _, row in manifest.iterrows():
        split = row["split"]
        dia, utt = row["dialogue_id"], row["utterance_id"]

        audio_feat = AUDIO_FEAT_DIR / split / f"dia{dia}_utt{utt}.npy"
        video_feat = VIDEO_FEAT_DIR / split / f"dia{dia}_utt{utt}.npy"

        a_ok = audio_feat.exists()
        v_ok = video_feat.exists()

        if not a_ok:
            missing_audio += 1
        if not v_ok:
            missing_video += 1

        if a_ok and v_ok:
            rows.append({
                "split": split,
                "dialogue_id": dia,
                "utterance_id": utt,
                "audio_feat_path": str(audio_feat),
                "video_feat_path": str(video_feat),
                "emotion": row["emotion"],
                "sentiment": row["sentiment"],
            })

    final = pd.DataFrame(rows)
    out_path = MELD_ROOT / "manifest_final.csv"
    final.to_csv(out_path, index=False)

    print(f"Ukupno u originalnom manifestu: {len(manifest)}")
    print(f"Nedostaje audio obelezja: {missing_audio}")
    print(f"Nedostaje video obelezja: {missing_video}")
    print(f"Finalno iskoristivih klipova: {len(final)}")
    print(f"Sacuvano u: {out_path}")
    print()
    print("Distribucija po split-u:")
    print(final["split"].value_counts())
    print()
    print("Distribucija emocija:")
    print(final["emotion"].value_counts())


if __name__ == "__main__":
    main()
