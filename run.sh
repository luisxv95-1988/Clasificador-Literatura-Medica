#!/bin/bash

set -e

if [ "$1" == "--retrain" ]; then
  echo 'Entrenando el modelo...'
  python src/train.py
fi

echo 'Ejecutando predicción...'
python src/predict.py --input data/sample_input.csv --output data/sample_output.csv
