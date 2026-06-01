import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

from config import (WINDOW_SIZE, HOP_SIZE, MODEL_PATH, IZLAZ_TEST_DIR, REZULTATI_PATH)
from data import (ucitaj_noizeus_skup, rekonstruisi_iz_spektra, sacuvaj_wav, rekonstruisi_ola)
from model import DenoisingAutoencoder
from metrics import ispisi_metrike

def testiraj(tip_suma: str, snr: str, model_path: str = MODEL_PATH) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model '{model_path}' nije pronadjen!")

    os.makedirs(IZLAZ_TEST_DIR, exist_ok=True)
        
    print("\nKorak 1/3 - Učitavanje modela...")
    mreza, norm_faktor = DenoisingAutoencoder.ucitaj(model_path)

    try:
        with open(REZULTATI_PATH) as f:
            podaci = json.load(f)
        trenirani_sum = podaci.get("tip_suma", "?")
        trenirani_snr = podaci.get("snr", "?")
    except FileNotFoundError:
        podaci = {}
        trenirani_sum, trenirani_snr = "?", "?"

    matched = (tip_suma == trenirani_sum)
    scenario = ("matched" if matched
                else f"mismatched (treniran na {trenirani_sum}_{trenirani_snr}dB)")
    print(f"  Scenario: {scenario} | Test skup: {tip_suma}_{snr}dB")

    print(f"\nKorak 2/3 - Učitavanje test podataka...")
    _, _, X_test, Y_test, sr = ucitaj_noizeus_skup(tip_suma, snr, WINDOW_SIZE, HOP_SIZE)

    X_fft = np.fft.rfft(X_test, axis=1)
    X_mag = np.log1p(np.abs(X_fft))
    X_norm = X_mag / norm_faktor
    faze = np.angle(X_fft)

    t0 = time.time()
    pred = mreza.forward(X_norm)
    print(f"  Test završen za: {(time.time()-t0)*1000:.1f} ms")

    audio_sumovit = rekonstruisi_ola(X_test, WINDOW_SIZE, HOP_SIZE)
    audio_cist = rekonstruisi_ola(Y_test, WINDOW_SIZE, HOP_SIZE)
    audio_ociscen = rekonstruisi_iz_spektra(pred, norm_faktor, faze,
                                         WINDOW_SIZE, HOP_SIZE)

    prefix = f"{tip_suma}_{snr}dB"
    sacuvaj_wav(os.path.join(IZLAZ_TEST_DIR, f"{prefix}_sumovit.wav"), sr, audio_sumovit)
    sacuvaj_wav(os.path.join(IZLAZ_TEST_DIR, f"{prefix}_cist.wav"),    sr, audio_cist)
    sacuvaj_wav(os.path.join(IZLAZ_TEST_DIR, f"{prefix}_ociscen.wav"), sr, audio_ociscen)

    print("\nKorak 3/3 - Metrike")
    print("=" * 62)
    rezultati = ispisi_metrike(f"TEST - {tip_suma}_{snr}dB ({scenario})", 
                               audio_cist, audio_sumovit, audio_ociscen, sr=sr)
    print("=" * 62)

    kljuc = "test_matched" if matched else f"test_mismatch_{tip_suma}_{snr}dB"
    podaci[kljuc] = {**rezultati, "tip_suma": tip_suma, "snr_ulaz": snr,
                     "scenario": scenario}
    with open(REZULTATI_PATH, 'w') as f:
        json.dump(podaci, f, indent=2)
    print(f"\n  Metrike sačuvane: {REZULTATI_PATH}")

    _, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    t = np.arange(4000)
    for ax, sig, title, color in zip(
        axes,
        [audio_cist, audio_sumovit, audio_ociscen],
        ['Čist', f'Šumovit ({snr} dB)', 'Očišćen'],
        ['green', 'red', 'dodgerblue']
    ):
        sred = len(sig) // 2
        ax.plot(t, sig[sred:sred+4000], color=color, linewidth=0.8)
        ax.set_title(title); ax.set_ylabel('Amplituda')
        ax.set_ylim([-0.3, 0.3]); ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel('Sample')
    plt.tight_layout()
    plt.savefig(os.path.join(IZLAZ_TEST_DIR, f"{prefix}_rezultat.png"), dpi=120)