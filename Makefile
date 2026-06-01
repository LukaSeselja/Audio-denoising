clean:
	rm -rf izlaz_trening/ izlaz_test/ rezultati.json poredjenje.png *.npz

train:
ifdef model
	python3 main.py --mode train --noise $(noise) --snr $(snr) --model $(model)
else
	python3 main.py --mode train --noise $(noise) --snr $(snr)
endif

test:
ifdef model
	python3 main.py --mode test --noise $(noise) --snr $(snr) --model $(model)
else
	python3 main.py --mode test --noise $(noise) --snr $(snr)
endif

compare:
	python3 main.py --mode compare