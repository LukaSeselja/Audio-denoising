import os
import time
import numpy as np
import matplotlib.pyplot as plt

from config import (WINDOW_SIZE, HOP_SIZE, HIDDEN_DIM, INPUT_DIM,
                    EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_PATH, RANDOM_SEED,
                    IZLAZ_TRENING_DIR)
from data import (ucitaj_noizeus_skup, rekonstruisi_ola, 
                  rekonstruisi_iz_spektra, sacuvaj_wav, ucitaj_audio)
from model import DenoisingAutoencoder
from metrics import ispisi_metrike

def treniraj(tip_suma: str, snr: str, model_path: str = MODEL_PATH) -> None:
    np.random.seed(RANDOM_SEED)
    os.makedirs(IZLAZ_TRENING_DIR, exist_ok=True)
    
    print(f"Korak 1/4 - Učitavanje NOIZEUS ({tip_suma}, {snr} dB)...")
    X_tr, Y_tr, _, _, sr = ucitaj_noizeus_skup(tip_suma, snr, WINDOW_SIZE, HOP_SIZE)

    X_fft = np.fft.rfft(X_tr, axis=1)
    Y_fft = np.fft.rfft(Y_tr, axis=1)
    X_mag = np.log1p(np.abs(X_fft))
    Y_mag = np.log1p(np.abs(Y_fft))

    X_faktor = float(np.max(X_mag)) or 1.0
    X_tr_norm = X_mag / X_faktor
    Y_tr_norm = Y_mag / X_faktor
    X_faze = np.angle(X_fft)

    print(f"\nKorak 2/4 - Trening ({EPOCHS} epoha)...")
    mreza = DenoisingAutoencoder(INPUT_DIM, HIDDEN_DIM)
    hist_tr = []
    t0 = time.time()

    for epoha in range(1, EPOCHS + 1):
        idx = np.random.permutation(len(X_tr_norm))
        n_batch = len(X_tr_norm) // BATCH_SIZE
        greska = 0.0

        for i in range(n_batch):
            b = idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            izlaz = mreza.forward(X_tr_norm[b])
            greska += float(np.mean((izlaz - Y_tr_norm[b]) ** 2))
            mreza.backward(X_tr_norm[b], Y_tr_norm[b])
            mreza.step(LEARNING_RATE)
        hist_tr.append(greska / n_batch)

        if epoha % 50 == 0 or epoha == 1:
            print(f"  Epoha {epoha:>3}/{EPOCHS} | MSE: {greska / n_batch:.6f}")

    print(f"  Trening završen za {time.time() - t0:.1f}s")
    mreza.sacuvaj(model_path, norm_faktor=X_faktor)

    print("\nKorak 3/4 - Rekonstrukcija audio-a...")
    pred_tr = mreza.forward(X_tr_norm)
    audio_sumovit = rekonstruisi_ola(X_tr, WINDOW_SIZE, HOP_SIZE)
    audio_cist = rekonstruisi_ola(Y_tr, WINDOW_SIZE, HOP_SIZE)
    audio_ociscen = rekonstruisi_iz_spektra(pred_tr, X_faktor, X_faze, WINDOW_SIZE, HOP_SIZE)

    sacuvaj_wav(os.path.join(IZLAZ_TRENING_DIR, "sumovit.wav"),  sr, audio_sumovit)
    sacuvaj_wav(os.path.join(IZLAZ_TRENING_DIR, "cist.wav"),     sr, audio_cist)
    sacuvaj_wav(os.path.join(IZLAZ_TRENING_DIR, "ociscen.wav"),  sr, audio_ociscen)

    ispisi_metrike("Trening", audio_cist, audio_ociscen)

    plt.figure(figsize=(10, 3))
    plt.plot(hist_tr, color='steelblue', label='Train MSE')
    plt.title('Learning curve'); 
    plt.xlabel('Epoch'); 
    plt.ylabel('MSE')
    plt.legend(); 
    plt.grid(True, alpha=0.4)
    plt.savefig(os.path.join(IZLAZ_TRENING_DIR, "rezultat.png"), dpi=120)