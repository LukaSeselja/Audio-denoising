clean:
	rm -rf izlaz_trening/ izlaz_test/ rezultati.json poredjenje.png *.npz src/__pycache__

train:
ifdef model
	python3 src/main.py --mode train --noise $(noise) --snr $(snr) --model $(model)
else
	python3 src/main.py --mode train --noise $(noise) --snr $(snr)
endif

test:
ifdef model
	python3 src/main.py --mode test --noise $(noise) --snr $(snr) --model $(model)
else
	python3 src/main.py --mode test --noise $(noise) --snr $(snr)
endif

compare:
	python3 src/main.py --mode compare