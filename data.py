import os
import numpy as np
from scipy.io import wavfile
from config import WINDOW_SIZE, HOP_SIZE, NOIZEUS_DIR, NOIZEUS_N_TRAIN, NOIZEUS_TOTAL

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
    folder_sum = os.path.join(NOIZEUS_DIR, f"{tip_suma}_{snr}dB")
    folder_cist = os.path.join(NOIZEUS_DIR, "clean")

    if not os.path.exists(folder_sum):
        raise FileNotFoundError(
            f"Folder '{folder_sum}' nije pronađen.\n"
        )
    
    if not os.path.exists(folder_cist):
        raise FileNotFoundError(
            f"Folder sa čistim fajlovima '{folder_cist}' nije pronađen."
        )

    svi_fajlovi = sorted([f for f in os.listdir(folder_sum) if f.endswith('.wav')])

    if len(svi_fajlovi) != NOIZEUS_TOTAL:
        raise ValueError(
            f"Očekivano {NOIZEUS_TOTAL} fajlova u '{folder_sum}', "
            f"pronađeno {len(svi_fajlovi)}."
        )
    
    trening_fajlovi = svi_fajlovi[:NOIZEUS_N_TRAIN]
    test_fajlovi = svi_fajlovi[NOIZEUS_N_TRAIN:]

    print(f"  Trening fajlovi ({len(trening_fajlovi)}): "
          f"{trening_fajlovi[0]} {trening_fajlovi[-1]}")
    print(f"  Test fajlovi ({len(test_fajlovi)}): "
          f"{test_fajlovi[0]} {test_fajlovi[-1]}")

    def ucitaj_listu(fajlovi, folder, je_cist=False):
        sve = []
        sr = None
        for ime in fajlovi:
            base = ime.split('_')[0] + '.wav' if je_cist else ime
            p, s = ucitaj_audio(os.path.join(folder, base), window_size, hop_size)
            sve.append(p)
            sr = s
        return np.concatenate(sve, axis=0), sr

    X_tr, sr = ucitaj_listu(trening_fajlovi, folder_sum, je_cist=False)
    Y_tr, _  = ucitaj_listu(trening_fajlovi, folder_cist, je_cist=True)
    X_te, _  = ucitaj_listu(test_fajlovi, folder_sum, je_cist=False)
    Y_te, _  = ucitaj_listu(test_fajlovi, folder_cist, je_cist=True)

    print(f"  Trening prozora: {len(X_tr)} | Test prozora: {len(X_te)}")
    return X_tr, Y_tr, X_te, Y_te, sr

def rekonstruisi_ola(prozori: np.ndarray, window_size: int = WINDOW_SIZE, 
                     hop_size: int = HOP_SIZE) -> np.ndarray:
    n = prozori.shape[0]
    duzina = (n - 1) * hop_size + window_size
    out = np.zeros(duzina, dtype=np.float32)
    suma = np.zeros(duzina, dtype=np.float32)
    hann = np.hanning(window_size)

    for i, p in enumerate(prozori):
        s = i * hop_size
        out[s:s + window_size]  += p * hann
        suma[s:s + window_size] += hann ** 2
    suma[suma < 1e-4] = 1.0
    return out / suma

def sacuvaj_wav(putanja: str, sample_rate: int, signal: np.ndarray) -> None:
    data = (np.clip(signal, -1, 1) * 32767).astype(np.int16)
    wavfile.write(putanja, sample_rate, data)
    print(f"  Sačuvan: {putanja}")

def spektralna_normalizacija(prozori: np.ndarray, faktor: float | None = None):
    fft = np.fft.rfft(prozori, axis=1)
    mag = np.log1p(np.abs(fft))
    if faktor is None:
        faktor = float(np.max(mag)) or 1.0
    return mag / faktor, faktor, np.angle(fft)

def rekonstruisi_iz_spektra(mag_norm: np.ndarray, faktor: float, faze: np.ndarray, 
                            window_size: int = WINDOW_SIZE, 
                            hop_size: int = HOP_SIZE) -> np.ndarray:
    mag = np.expm1(mag_norm * faktor)
    spektar = mag * np.exp(1j * faze)
    prozori = np.fft.irfft(spektar, n=window_size)
    return rekonstruisi_ola(prozori, window_size, hop_size)