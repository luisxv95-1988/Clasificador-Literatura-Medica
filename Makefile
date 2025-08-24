install:
	pip install -r requirements.txt

train:
	python src/train.py

predict:
	python src/predict.py --input data/sample_input.csv --output data/sample_output.csv
