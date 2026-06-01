import os
import json
import matplotlib.pyplot as plt
import numpy as np
from config import REZULTATI_PATH

def uporedi() -> None:
    if not os.path.exists(REZULTATI_PATH):
        raise FileNotFoundError(
            f"'{REZULTATI_PATH}' nije pronađen.\n"
        )

    with open(REZULTATI_PATH) as f:
        podaci = json.load(f)

    tip_suma = podaci.get("tip_suma", "?")
    snr = podaci.get("snr", "?")

    skupovi = {}
    if "trening" in podaci:
        skupovi["Trening"] = podaci["trening"]
    if "test_matched" in podaci:
        skupovi["Test (matched)"] = podaci["test_matched"]
    for kljuc, vrednost in podaci.items():
        if kljuc.startswith("test_mismatch_"):
            t = vrednost.get('tip_suma', '?')
            s = vrednost.get('snr_ulaz', '?')
            skupovi[f"Mismatch {t}_{s}dB"] = vrednost

    print("=" * 72)
    print(f"Model treniran na: {tip_suma}_{snr}dB")
    print("-" * 72)
    for labela, v in skupovi.items():
        lab = labela.replace('\n', ' ')
        pesq_pre = v.get('pesq_pre',  None)
        pesq_post = v.get('pesq_post', None)
        pesq_str = (f"PESQ: {pesq_pre:.2f} -> {pesq_post:.2f} ({pesq_post-pesq_pre:+.2f})"
                     if pesq_pre is not None else "PESQ: N/A")
        stoi_pre = v.get('stoi_pre',  None)
        stoi_post = v.get('stoi_post', None)
        stoi_str = (f"STOI: {stoi_pre:.4f} -> {stoi_post:.4f} ({stoi_post-stoi_pre:+.4f})"
                     if stoi_pre is not None else "STOI: N/A")
        print(f"  {lab:25s} | {pesq_str} | {stoi_str} | SNR: {float(v['snr']):+.2f} dB")
    print("=" * 72)

    # Grafik
    labele = list(skupovi.keys())
    x = np.arange(len(labele))

    pesq_pre_vals, pesq_post_vals = [], []
    stoi_pre_vals, stoi_post_vals = [], []
    snr_vals = []

    for v in skupovi.values():
        pesq_pre_vals.append(v.get('pesq_pre', 0))
        pesq_post_vals.append(v.get('pesq_post', 0))
        stoi_pre_vals.append(v.get('stoi_pre', 0))
        stoi_post_vals.append(v.get('stoi_post', 0))
        snr_vals.append(float(v.get('snr', 0)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    w = 0.35

    # PESQ
    bars_pesq_pre = axes[0].bar(x - w/2, pesq_pre_vals,  w, label='Pre',  color='salmon',    edgecolor='black', linewidth=0.6)
    bars_pesq_post = axes[0].bar(x + w/2, pesq_post_vals, w, label='Posle', color='steelblue', edgecolor='black', linewidth=0.6)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labele, fontsize=8)
    axes[0].set_title('PESQ', fontsize=10, fontweight='bold')
    axes[0].set_ylim([0, 5.5])
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.4)

    for b_pre, b_post in zip(bars_pesq_pre, bars_pesq_post):
        v_pre, v_post = b_pre.get_height(), b_post.get_height()
        axes[0].text(b_pre.get_x() + b_pre.get_width()/2, v_pre + 0.05,
                     f'{v_pre:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        axes[0].text(b_post.get_x() + b_post.get_width()/2, v_post + 0.05,
                     f'{v_post:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # STOI
    bars_stoi_pre = axes[1].bar(x - w/2, stoi_pre_vals,  w, label='Pre',  color='salmon',    edgecolor='black', linewidth=0.6)
    bars_stoi_post = axes[1].bar(x + w/2, stoi_post_vals, w, label='Posle', color='steelblue', edgecolor='black', linewidth=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labele, fontsize=8)
    axes[1].set_title('STOI', fontsize=10, fontweight='bold')
    axes[1].set_ylim([0, 1.2])
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.4)

    for b_pre, b_post in zip(bars_stoi_pre, bars_stoi_post):
        v_pre, v_post = b_pre.get_height(), b_post.get_height()
        axes[1].text(b_pre.get_x() + b_pre.get_width()/2, v_pre + 0.01,
                     f'{v_pre:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        axes[1].text(b_post.get_x() + b_post.get_width()/2, v_post + 0.01,
                     f'{v_post:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # SNR
    boje = plt.cm.tab10(np.linspace(0, 1, len(labele)))
    bars_snr = axes[2].bar(x, snr_vals, color=boje, width=0.5, edgecolor='black', linewidth=0.6)
    axes[2].set_xticks(x); axes[2].set_xticklabels(labele, fontsize=8)
    axes[2].set_title('SNR [dB]', fontsize=10, fontweight='bold')
    axes[2].set_ylim([0, max(snr_vals) * 1.35 + 0.5])
    axes[2].grid(axis='y', alpha=0.4)

    for bar, v in zip(bars_snr, snr_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, max(v, 0) + 0.02,
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle(f'Poređenje evaluacija - model: {tip_suma}_{snr}dB',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('poredjenje.png', dpi=120)
    plt.close()
    print("  Grafikon sačuvan: poredjenje.png")