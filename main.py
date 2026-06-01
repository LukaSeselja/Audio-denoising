import argparse
from config import MODEL_PATH, NOIZEUS_NOISE_TYPES

def main():
    parser = argparse.ArgumentParser(
        description="Denoising Autoencoder", 
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode", required=True, 
        choices=["train", "test", "compare"], 
        help=(
            "train -> trenira model na NOIZEUS skupu\n"
            "test -> testira model\n"
            "compare -> Uporedjivanje rezultata\n"
        ),
    )
    
    parser.add_argument(
        "--noise", default=None,
        choices=NOIZEUS_NOISE_TYPES,
        help=f"Tip šuma iz NOIZEUS-a.\nDostupni: {', '.join(NOIZEUS_NOISE_TYPES)}\n"
             "(obavezno za --mode train i --mode test)\n"
    )

    parser.add_argument(
        "--snr", default=None,
        choices=["0", "5", "10", "15"],
        help="Nivo ulaznog SNR-a [dB]\n(obavezno za --mode train i --mode test)"
    )

    parser.add_argument(
        "--model", default=MODEL_PATH,
        help=f"Putanja do NPZ fajla sa težinama (default: {MODEL_PATH})"
    )

    args = parser.parse_args()

    if not args.model.endswith('.npz'):
        args.model += '.npz'  

    if args.mode in ("train", "test"):
        if args.noise is None:
            parser.error(f'--noise je obavezan za --mode {args.mode}')
        if args.snr is None:
            parser.error(f'--snr je obavezan za --mode {args.mode}')
    
    if args.mode == "train":
        from train import treniraj
        treniraj(tip_suma=args.noise, snr=args.snr, model_path=args.model)

    elif args.mode == "test":
        from test import testiraj
        testiraj(tip_suma=args.noise, snr=args.snr, model_path=args.model)

    elif args.mode == "compare":
        from compare import uporedi
        uporedi()

if __name__ == "__main__":
    main()