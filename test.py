import os
import numpy as np
from config import WINDOW_SIZE, HOP_SIZE, MODEL_PATH
from data import (ucitaj_noizeus_skup, rekonstruisi_iz_spektra, sacuvaj_wav)
from model import DenoisingAutoencoder

def testiraj(tip_suma: str, snr: str, model_path: str = MODEL_PATH) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' nije pronadjen!")
        
    print("Ucitavanje modela...")
    mreza, norm_faktor = DenoisingAutoencoder.ucitaj(model_path)

    print(f"Ucitavanje test podataka ({tip_suma}_{snr}dB)...")
    _, _, X_te, _, sr = ucitaj_noizeus_skup(tip_suma, snr, WINDOW_SIZE, HOP_SIZE)

    X_fft = np.fft.rfft(X_te, axis=1)
    X_mag = np.log1p(np.abs(X_fft))
    X_norm = X_mag / norm_faktor
    faze = np.angle(X_fft)

    print("Pokretanje inferencije (ciscenja)...")
    pred = np.maximum(0, mreza.forward(X_norm))

    audio_ociscen = rekonstruisi_iz_spektra(pred, norm_faktor, faze, WINDOW_SIZE, HOP_SIZE)

    if not os.path.exists("izlaz_test"):
        os.makedirs("izlaz_test")
        
    prefix = f"{tip_suma}_{snr}dB"
    sacuvaj_wav(os.path.join("izlaz_test", f"{prefix}_ociscen.wav"), sr, audio_ociscen)