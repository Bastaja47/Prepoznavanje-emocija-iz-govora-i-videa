"""
common.py - deljene komponente za Korak 5: Dataset klasa, arhitekture
modela i pomocne funkcije (matrica konfuzije, upis rezultata).

Koristi se iz train.py i late_fusion_eval.py - ne pokrece se direktno.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "fear", "disgust"]
LABEL2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL = {i: e for e, i in LABEL2IDX.items()}
NUM_CLASSES = len(EMOTIONS)


class MELDFeatureDataset(Dataset):
    """
    modality: "audio", "video" ili "both" (za ranu fuziju)
    pooling:  "none" (flatten sirove sekvence, "bazna" varijanta iz rada)
              ili "tp" (mean+std temporal pooling, druga varijanta)
    """

    def __init__(self, df, modality, pooling):
        self.df = df.reset_index(drop=True)
        self.modality = modality
        self.pooling = pooling

    def __len__(self):
        return len(self.df)

    def _load_and_pool(self, path):
        arr = np.load(path)  # (T, D)
        t = torch.from_numpy(arr).float()
        if self.pooling == "tp":
            mean = t.mean(dim=0)
            std = t.std(dim=0)
            return torch.cat([mean, std], dim=0)  # (2D,)
        return t.flatten()  # (T*D,)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = LABEL2IDX[row["emotion"]]

        if self.modality == "audio":
            feat = self._load_and_pool(row["audio_feat_path"])
        elif self.modality == "video":
            feat = self._load_and_pool(row["video_feat_path"])
        elif self.modality == "both":
            a = self._load_and_pool(row["audio_feat_path"])
            v = self._load_and_pool(row["video_feat_path"])
            feat = torch.cat([a, v], dim=0)
        else:
            raise ValueError(f"Nepoznat modality: {self.modality}")

        return feat, label


class UnimodalMLP(torch.nn.Module):
    """Isto za audio i video granu: dva potpuno povezana sloja (kao u radu)."""

    def __init__(self, input_dim, hidden_dim=256, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EarlyFusionMLP(torch.nn.Module):
    """Tri potpuno povezana sloja (kao rana fuzija u referentnom radu)."""

    def __init__(self, input_dim, fusion_hidden_dim=512, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, fusion_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(fusion_hidden_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def plot_confusion_matrix(cm, title, out_path):
    """Matrica konfuzije: broj uzoraka + procenat po redu (stil kao u referentnom radu)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    row_pct = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    ) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(
                j, i, f"{cm[i, j]}\n({row_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def append_result_row(results_csv, row_dict):
    row_df = pd.DataFrame([row_dict])
    if results_csv.exists():
        existing = pd.read_csv(results_csv)
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(results_csv, index=False)
