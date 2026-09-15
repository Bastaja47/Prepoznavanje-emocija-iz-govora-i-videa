"""
Izvlacenje i pretprocesiranje audio signala iz MELD videa.

Za svaki video iz manifest.csv:
  1) ffmpeg izvuce audio kao mono, 16kHz wav (16kHz je frekvencija koju
     wav2vec2 ocekuje na ulazu)
  2) normalizacija amplitude u opseg [-1, 1]
  3) obrezivanje/dopunjavanje na tacno 6 sekundi

Rezultat: data/audio/{split}/dia{D}_utt{U}.wav

"""

import subprocess
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


MELD_ROOT = Path(r"C:\Users\XYZ\Desktop\MELD.Raw")


MANIFEST = MELD_ROOT / "manifest.csv"
OUT_DIR = MELD_ROOT / "data" / "audio"
SAMPLE_RATE = 16000                          #Hz
CLIP_SECONDS = 6.0
TARGET_LEN = int(SAMPLE_RATE * CLIP_SECONDS)


def extract_wav_ffmpeg(video_path, tmp_wav_path):  #Iz video fajla izvlaci audio i cuva ga u tmp_wav_path
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),   #-y preporuka da se prekine ako fajl već postoji | -i input file
        "-vn",                                   #-vn ignorise video deo
        "-ac", "1",                              #-ac 1 postavlja broj audio kanala na 1 (mono)
        "-ar", str(SAMPLE_RATE),                 #sample rate (Hz)
        "-loglevel", "error",                    #-loglevel error prikazuje samo greške
        str(tmp_wav_path), 
    ]
    subprocess.run(cmd, check=True)


def normalize_and_fix_length(wav):                        #normalizuje audio signal ceo signal/amplituda u opseg [-1, 1] i obrezivanje/dopunjavanje na tacno 6 sekundi
    max_val = np.max(np.abs(wav)) if len(wav) > 0 else 0.0
    if max_val > 1e-8:
        wav = wav / max_val

    if len(wav) > TARGET_LEN:
        wav = wav[:TARGET_LEN]
    elif len(wav) < TARGET_LEN:
        wav = np.pad(wav, (0, TARGET_LEN - len(wav)))      #ako je signal duzi sklanja se visak ako je kraci dodamo nule
    return wav.astype(np.float32)


def main():
    manifest = pd.read_csv(MANIFEST)
    for split in manifest["split"].unique():
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)  #za svaki split pravi folder ako ne postoji, parents=True pravi i roditeljske foldere ako ne postoje, exist_ok=True ne baca gresku ako folder vec postoji

    failed = []
    tmp_path = MELD_ROOT / "_tmp_audio.wav"

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        split = row["split"]
        dia, utt = row["dialogue_id"], row["utterance_id"]
        out_path = OUT_DIR / split / f"dia{dia}_utt{utt}.wav"
        if out_path.exists():
            continue
        try:
            extract_wav_ffmpeg(row["video_path"], tmp_path)        #Izvlači audio iz videa pomoću ffmpeg-a
            wav, sr = sf.read(tmp_path, dtype="float32")           #Čita privremeni .wav fajl
            if wav.ndim > 1:
                wav = wav.mean(axis=1)                             #Ako je stereo (wav.ndim > 1), pretvara ga u mono (uzima srednju vrednost kanala)
            wav = normalize_and_fix_length(wav)                    #Normalizuje i podešava dužinu na tačno 6 sekundi
            sf.write(out_path, wav, SAMPLE_RATE, subtype="FLOAT")
        except Exception as e:
            failed.append((str(row["video_path"]), str(e)))
   

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"\nGotovo. Neuspesnih: {len(failed)}")
    if failed:
        fail_path = OUT_DIR / "failed.csv"
        pd.DataFrame(failed, columns=["video_path", "error"]).to_csv(fail_path, index=False)
        print(f"Detalji o greskama: {fail_path}")


if __name__ == "__main__":
    main()
