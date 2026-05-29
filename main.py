import argparse

def main():
    parser = argparse.ArgumentParser(description="Denoising Autoencoder")
    parser.add_argument("--mode", required=True, choices=["train", "test"], 
                        help="train ili test")
    
    default_noise = "car"
    default_snr = "10"

    args = parser.parse_args()

    if args.mode == "train":
        from train import treniraj
        treniraj(tip_suma=default_noise, snr=default_snr)

    elif args.mode == "test":
        from test import testiraj
        testiraj(tip_suma=default_noise, snr=default_snr)

if __name__ == "__main__":
    main()