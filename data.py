import os
import numpy as np
from scipy.io import wavfile
from config import WINDOW_SIZE, HOP_SIZE, NOIZEUS_DIR

def ucitaj_signal(putanja: str) -> tuple[np.ndarray, int]:
    sample_rate, signal = wavfile.read(putanja)

    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    signal = signal.astype(np.float32)
    maks = np.max(np.abs(signal))

    if maks > 0:
        signal /= maks

    return signal, sample_rate

def ucitaj_audio(putanja: str, window_size: int = WINDOW_SIZE, 
                 hop_size: int = HOP_SIZE) -> tuple[np.ndarray, int]:
    signal, sample_rate = ucitaj_signal(putanja)
    hann = np.hanning(window_size)

    prozori = [
        signal[i:i + window_size] * hann
        for i in range(0, len(signal) - window_size + 1, hop_size)
    ]
 
    return np.array(prozori, dtype=np.float32), sample_rate

def ucitaj_noizeus_skup(tip_suma: str, snr: str, window_size: int = WINDOW_SIZE, 
                        hop_size: int = HOP_SIZE):
    folder_sum  = os.path.join(NOIZEUS_DIR, f"{tip_suma}_{snr}dB")
    folder_cist = os.path.join(NOIZEUS_DIR, "clean")

    svi_fajlovi = sorted([f for f in os.listdir(folder_sum) if f.endswith('.wav')])
    
    trening_fajlovi = svi_fajlovi[:20]
    test_fajlovi    = svi_fajlovi[20:]

    def ucitaj_listu(fajlovi, folder, je_cist=False):
        sve = []
        sr  = None
        for ime in fajlovi:
            base = ime.split('_')[0] + '.wav' if je_cist else ime
            p, s = ucitaj_audio(os.path.join(folder, base), window_size, hop_size)
            sve.append(p)
            sr = s
        return np.concatenate(sve, axis=0), sr

    X_tr, sr = ucitaj_listu(trening_fajlovi, folder_sum,  je_cist=False)
    Y_tr, _  = ucitaj_listu(trening_fajlovi, folder_cist, je_cist=True)
    X_te, _  = ucitaj_listu(test_fajlovi,    folder_sum,  je_cist=False)
    Y_te, _  = ucitaj_listu(test_fajlovi,    folder_cist, je_cist=True)

    return X_tr, Y_tr, X_te, Y_te, sr

def rekonstruisi_ola(prozori: np.ndarray, window_size: int = WINDOW_SIZE, 
                     hop_size: int = HOP_SIZE) -> np.ndarray:
    n = prozori.shape[0]
    duzina = (n - 1) * hop_size + window_size
    out  = np.zeros(duzina, dtype=np.float32)
    suma = np.zeros(duzina, dtype=np.float32)
    hann = np.hanning(window_size)

    for i, p in enumerate(prozori):
        s = i * hop_size
        out[s:s + window_size]  += p * hann
        suma[s:s + window_size] += hann ** 2

    return out / suma

def sacuvaj_wav(putanja: str, sample_rate: int, signal: np.ndarray) -> None:
    data = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    wavfile.write(putanja, sample_rate, data)