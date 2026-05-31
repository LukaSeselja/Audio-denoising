import numpy as np

def izracunaj_snr(signal: np.ndarray, suma: np.ndarray) -> float:
    razlika = signal - suma
    snaga_signala = np.mean(signal ** 2)
    snaga_suma = np.mean(razlika ** 2)
    return float(10 * np.log10(snaga_signala / snaga_suma))

def ispisi_metrike(tag: str, cist: np.ndarray, ociscen: np.ndarray):     
    snr_vrednost = izracunaj_snr(cist, ociscen)   
    print(f"  [{tag}]")     
    print(f"    SNR  | Očišćen vs Čist: {snr_vrednost:.2f} dB")