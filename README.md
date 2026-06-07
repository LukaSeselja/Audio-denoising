# Audio Denoising - Denoising Autoencoder

Implementacija autoenkodera za uklanjanje šuma iz audio snimaka. Model je treniran i evaluiran na NOIZEUS benchmark datasetu. Projekat je pisan od nule koristeći NumPy i SciPy, bez ML framework-a.

---

## Struktura projekta

```
├── src/
│   ├── main.py        # entry point
│   ├── config.py      # hiperparametri i konstante
│   ├── data.py        # učitavanje audio-a, prozoriranje, OLA rekonstrukcija
│   ├── model.py       # DenoisingAutoencoder klasa
│   ├── metrics.py     # SNR, PESQ, STOI metrike
│   ├── train.py       # trening petlja
│   ├── test.py        # evaluacija modela
│   └── compare.py     # poređenje rezultata
├── noizeus/           # dataset (nije u repozitorijumu, preuzeti ručno)
├── Makefile
└── README.md
```

---

## Dataset

Projekat koristi [NOIZEUS](http://ecs.utdallas.edu/loizou/speech/noizeus/) - 30 IEEE rečenica snimljenih od 3 muška i 3 ženska govornika, korumpiranih sa 8 tipova realnog šuma pri SNR nivoima 0, 5, 10 i 15 dB. Svi snimci su 8 kHz, 16-bit mono.

**Preuzimanje formatiranog dataseta:** [Google Drive link](https://drive.google.com/drive/folders/1vDeB61BWCEaG3HqTHEYKu1UcRNJi8Zn9?usp=sharing)  
Preuzeti folder `noizeus/` i postaviti ga u root folder projekta.

Dostupni tipovi šuma: `car`, `babble`, `restaurant`, `street`, `airport`, `train`, `exhibition`, `station`

Dostupni SNR nivoi: `0`, `5`, `10`, `15` dB

---

## Instalacija zavisnosti

```bash
pip install numpy scipy matplotlib pesq pystoi
```

---

## Pokretanje

Sve komande mogu biti pokrenute na dva načina:
- **Python**
- **Makefile**

### Trening

Trenira model na prvih 20 fajlova (sp01-sp20, govornici 1-4)

**Python:**
```bash
python3 src/main.py --mode train --noise car --snr 10
```

**Makefile:**
```bash
make train noise=car snr=10
```

**Izlaz:**
- `model.npz` - Sačuvane težine mreže
- `rezultati.json` - Metrike za trening skup
- `izlaz_trening/rezultati.png` - Grafikon treninga i signala
- `izlaz_trening/{cist,sumovit,ociscen}.wav` - Audio snimci

### Test

Evaluacija na poslednjih 10 fajlova (sp21-sp30, govornici 5-6) koji nisu viđeni tokom treninga.

**Python - matched scenario** (isti tip šuma kao u treningu):
```bash
python3 src/main.py --mode test --noise car --snr 10
```

**Python - mismatched scenario** (drugačiji tip šuma):
```bash
python3 src/main.py --mode test --noise restaurant --snr 10
```

**Makefile:**
```bash
make test noise=car snr=10
make test noise=restaurant snr=10
```

**Izlaz:**
- `izlaz_test/{tip_suma}_{snr}dB_*.wav` - Audio snimci (cist, sumovit, ociscen)
- `izlaz_test/{tip_suma}_{snr}dB_rezultat.png` - Grafikon testa
- `rezultati.json` - Ažuriran sa test metrikama

### Poređenje rezultata

Generiše tabelu i grafikon poređenja svih evaluacija iz `rezultati.json`.

**Python:**
```bash
python3 src/main.py --mode compare
```

**Makefile:**
```bash
make compare
```

**Izlaz:**
- Ispisana tabela sa PESQ, STOI, SNR metrikama
- `poredjenje.png` - Grafikon poređenja

### Prilagođeni model

Ako želite da sacuvate model na drugoj lokaciji umesto podrazumevanog `model.npz`, koristite opciju `--model`:

**Python - trening sa prilagođenom putanjom:**
```bash
python3 src/main.py --mode train --noise car --snr 10 --model modeli/car_10dB
```

**Python - test sa prilagođenim modelom:**
```bash
python3 src/main.py --mode test --noise car --snr 10 --model modeli/car_10dB
```

**Makefile:**
```bash
make train noise=car snr=10 model=modeli/car_10dB
make test noise=car snr=10 model=modeli/car_10dB
```

**Napomena:** Direktorijum `modeli/` i ekstenzija `.npz` će biti kreirani automatski ako ne postoje.

### Čišćenje

Brisanje svih privremenih fajlova i rezultata:

```bash
make clean
```

---

## Metrike

| Metrika | Opis |
|---------|------|
| **SNR** [dB] | Signal-to-Noise Ratio - energetski odnos čistog i šumovitog signala |
| **PESQ** | Perceptual Evaluation of Speech Quality (-0.5 - 4.5) |
| **STOI** | Short-Time Objective Intelligibility (0.0 - 1.0) |

Više vrednosti su bolje za sve tri metrike.

---
