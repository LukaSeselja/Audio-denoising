import numpy as np
from pesq import pesq
from pystoi import stoi

def izracunaj_snr(signal: np.ndarray, sum: np.ndarray, eps: float = 1e-10) -> float:
    razlika = signal - sum
    snaga_signala = np.mean(signal ** 2)
    snaga_suma = np.mean(razlika ** 2)
    return float(10 * np.log10(snaga_signala / (snaga_suma + eps)))

def ispisi_metrike(tag: str, cist: np.ndarray, sumovit: np.ndarray, ociscen: np.ndarray, sr: int = 8000) -> dict:
    snr_vrednost = izracunaj_snr(cist, ociscen)

    pesq_pre = None
    pesq_post = None
        
    pesq_pre = pesq(sr, cist, sumovit, mode='nb')
    pesq_post = pesq(sr, cist, ociscen, mode='nb')

    stoi_pre = stoi(cist, sumovit, sr, extended=False)
    stoi_post = stoi(cist, ociscen, sr, extended=False)

    print(f"  [{tag}]")
    print(f"    PESQ | PRE: {pesq_pre:.4f}  ->  POSLE: {pesq_post:.4f}  |  Dobitak: {pesq_post - pesq_pre:+.4f}")
    print(f"    STOI | PRE: {stoi_pre:.4f}  ->  POSLE: {stoi_post:.4f}  |  Dobitak: {stoi_post - stoi_pre:+.4f}")
    print(f"    SNR  | Očišćen vs Čist: {snr_vrednost:.2f} dB")

    return {
        "pesq_pre": round(float(pesq_pre), 4),
        "pesq_post": round(float(pesq_post), 4),
        "stoi_pre": round(float(stoi_pre), 4),
        "stoi_post": round(float(stoi_post), 4),
        "snr": round(snr_vrednost, 4)
    }