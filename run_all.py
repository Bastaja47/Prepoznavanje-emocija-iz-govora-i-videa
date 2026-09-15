"""
Pokrece SVE treninge i kasnu fuziju redom, jednom komandom:

    python run_all.py

Redosled:
  1) audio (base), audio (tp_cw)
  2) video (base), video (tp_cw)
  3) early_fusion (base), early_fusion (tp_cw)
  4) late fusion (base), late fusion (tp_cw)  <- koristi vec istrenirane
     audio/video modele iz koraka 1 i 2

Sve rezultate skuplja u results.csv, matrice konfuzije u
confusion_matrices/, checkpoint-e u checkpoints/.
"""

import subprocess
import sys
from pathlib import Path


MELD_ROOT = Path(r"C:\Users\XYZ\Desktop\MELD.Raw")


CKPT_DIR = MELD_ROOT / "checkpoints"

TRAIN_RUNS = [
    ("audio", "base"),
    ("audio", "tp_cw"),
    ("video", "base"),
    ("video", "tp_cw"),
    ("early_fusion", "base"),
    ("early_fusion", "tp_cw"),
]

LATE_FUSION_VARIANTS = ["base", "tp_cw"]


def main():
    for model, variant in TRAIN_RUNS:
        ckpt_path = CKPT_DIR / f"{model}_{variant}.pt"
        if ckpt_path.exists():
            print(f"[PRESKACEM] {model} ({variant}) - checkpoint vec postoji: {ckpt_path}")
            continue

        print("=" * 90)
        print(f"TRENING: model={model} varijanta={variant}")
        print("=" * 90)
        subprocess.run(
            [sys.executable, "train.py", "--model", model, "--variant", variant],
            check=True,
        )

    for variant in LATE_FUSION_VARIANTS:
        print("=" * 90)
        print(f"KASNA FUZIJA: varijanta={variant}")
        print("=" * 90)
        subprocess.run(
            [sys.executable, "late_fusion_eval.py", "--variant", variant],
            check=True,
        )

    print("\nSVE ZAVRSENO.")
    print(f"Rezultati: {MELD_ROOT / 'results.csv'}")
    print(f"Matrice konfuzije: {MELD_ROOT / 'confusion_matrices'}")


if __name__ == "__main__":
    main()
