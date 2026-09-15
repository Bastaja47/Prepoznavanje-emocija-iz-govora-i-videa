# Multimodalno prepoznavanje emocija - MELD (audio + video)

Cilj projekta je prepoznavanje emocija na MELD korpusu koriscenjem audio i
video modaliteta, uz poredjenje jednomodalnih modela i rane/kasne fuzije.

## Struktura projekta

```
.
├── build_manifest.py            # 1. gradi manifest.csv iz MELD CSV + video fajlova
├── extract_audio.py             # 2. izdvaja .wav (16kHz, mono, 6s) iz videa
├── extract_faces.py             # 3. detekcija/kropovanje lica (MTCNN), 8 frejmova po klipu
├── extract_audio_features.py    # 4. wav2vec2 embeddings (768-dim)
├── extract_video_features.py    # 5. InceptionResnetV1/VGGFace2 embeddings (512-dim po frejmu)
├── build_final_manifest.py      # 6. filtrira na uzorke sa oba modaliteta -> manifest_final.csv
├── common.py                    # Dataset klasa, arhitekture modela, pomocne funkcije
├── train.py                     # trening jednog modela (audio/video/early_fusion)
├── late_fusion_eval.py          # evaluacija kasne fuzije iz vec istreniranih modela
├── run_all.py                   # pokrece sve treninge + kasnu fuziju redom
├── results.csv                  # sabrani rezultati svih varijanti (generise se)
├── confusion_matrices/          # matrice konfuzije (.png/.npy, generise se)
├── checkpoints/                 # najbolji checkpoint po modelu/varijanti (generise se, NIJE u git-u)
└── data/                        # izdvojena audio/video obelezja (generise se, NIJE u git-u)
```

Sirovi MELD video/audio fajlovi, izvedena obelezja (`data/`) i checkpoint-i
(`checkpoints/`) **nisu deo repozitorijuma** (prevelik su za git) - generisu
se lokalno pokretanjem skripti ispod. Repozitorijum sadrzi samo kod, konacne
rezultate (`results.csv`) i matrice konfuzije.

## Podaci

Koristi se [MELD](https://affective-meld.github.io/) (Multimodal
EmotionLines Dataset) - dijalozi iz serije "Friends", anotirani sa 7 emocija
(neutral, joy, surprise, anger, sadness, fear, disgust). Skup treba preuzeti
posebno i raspakovati lokalno pre pokretanja pipeline-a (`MELD_ROOT` promenljiva
na vrhu svake skripte).

## Pokretanje

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

python build_manifest.py
python extract_audio.py
python extract_faces.py
python extract_audio_features.py
python extract_video_features.py
python build_final_manifest.py

python run_all.py              # trenira sve modele i racuna kasnu fuziju
```

Svaki korak se moze ponovo pokrenuti - vec obradjeni/istrenirani fajlovi se
preskacu (proverava se da li izlazni fajl vec postoji).

## Metodologija (ukratko)

- **Audio**: wav2vec 2.0 (`facebook/wav2vec2-base`), zamrznut, kao ekstraktor obelezja (768-dim).
- **Video**: MTCNN (detekcija/kropovanje lica) + InceptionResnetV1 pretreniran na VGGFace2, zamrznut, kao ekstraktor obelezja (512-dim po frejmu, 8 frejmova po klipu).
- **Fuzija**: rana (konkatenacija + MLP) i kasna (ponderisano usrednjavanje softmax izlaza, α=0.5).
- Za svaki model testirane su dve varijante: **base** (bez temporal pooling-a, bez pondera klasa) i **tp_cw** (temporal pooling mean+std, sa ponderisanjem klasa).

## Rezultati

| Model         | TP | CW | Accuracy | Weighted F1 |
|---------------|:--:|:--:|:--------:|:-----------:|
| Audio         | ✗  | ✗  | 0.4664   | 0.3571      |
| Video         | ✗  | ✗  | 0.4570   | 0.3252      |
| Rana fuzija   | ✗  | ✗  | 0.4497   | 0.3691      |
| Kasna fuzija  | ✗  | ✗  | 0.4669   | 0.3205      |
| Audio         | ✓  | ✓  | 0.2537   | 0.2616      |
| Video         | ✓  | ✓  | 0.2361   | 0.2615      |
| Rana fuzija   | ✓  | ✓  | 0.2480   | 0.2598      |
| Kasna fuzija  | ✓  | ✓  | 0.2864   | 0.2970      |

Detaljnija diskusija rezultata (ukljucujuci matrice konfuzije) nalazi se u
pratecem izvestaju (`.docx`).

## Autor

Stojan Bastaja
