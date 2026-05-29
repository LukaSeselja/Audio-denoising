import os
import time
import numpy as np
from config import (WINDOW_SIZE, HOP_SIZE, HIDDEN_DIM, INPUT_DIM,
                    EPOCHS, LEARNING_RATE, BATCH_SIZE, MODEL_PATH, RANDOM_SEED)
from data import ucitaj_noizeus_skup, rekonstruisi_iz_spektra, sacuvaj_wav
from model import DenoisingAutoencoder

def treniraj(tip_suma: str, snr: str, model_path: str = MODEL_PATH) -> None:
    np.random.seed(RANDOM_SEED)
    
    print(f"\nUcitavanje podataka (sum: {tip_suma}, SNR: {snr} dB)...")
    X_tr, Y_tr, _, _, sr = ucitaj_noizeus_skup(tip_suma, snr, WINDOW_SIZE, HOP_SIZE)

    X_fft = np.fft.rfft(X_tr, axis=1)
    Y_fft = np.fft.rfft(Y_tr, axis=1)
    X_mag = np.log1p(np.abs(X_fft))
    Y_mag = np.log1p(np.abs(Y_fft))

    X_faktor = float(np.max(X_mag)) or 1.0
    X_tr_norm = X_mag / X_faktor
    Y_tr_norm = Y_mag / X_faktor
    X_faze = np.angle(X_fft)


    print(f"Trening pocinje ({EPOCHS} epoha)...")
    mreza = DenoisingAutoencoder(INPUT_DIM, HIDDEN_DIM)
    
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

        if epoha % 50 == 0 or epoha == 1:
            print(f"  Epoha {epoha:>3}/{EPOCHS} | Train MSE: {greska / n_batch:.6f}")

    print(f"Trening zavrsen za {time.time() - t0:.1f}s")
    mreza.sacuvaj(model_path, norm_faktor=X_faktor)

    pred_tr = np.maximum(0, mreza.forward(X_tr_norm))
    audio_ociscen = rekonstruisi_iz_spektra(pred_tr, X_faktor, X_faze, WINDOW_SIZE, HOP_SIZE)

    if not os.path.exists("izlaz_trening"):
        os.makedirs("izlaz_trening")

    sacuvaj_wav(os.path.join("izlaz_trening", "trening_ociscen.wav"), sr, audio_ociscen)