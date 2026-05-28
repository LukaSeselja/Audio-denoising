import numpy as np

def izracunaj_lsd(p1: np.ndarray, p2: np.ndarray) -> float:
    mag1 = np.abs(np.fft.rfft(p1, axis=1))
    mag2 = np.abs(np.fft.rfft(p2, axis=1))
    db1 = 20 * np.log10(mag1)
    db2 = 20 * np.log10(mag2)
    return float(np.mean(np.sqrt(np.mean((db1 - db2) ** 2, axis=1))))

def izracunaj_snr(signal: np.ndarray, suma: np.ndarray) -> float:
    razlika = signal - suma
    snaga_signala = np.mean(signal ** 2)
    snaga_suma = np.mean(razlika ** 2)
    return float(10 * np.log10(snaga_signala / snaga_suma))