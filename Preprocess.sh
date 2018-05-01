#!/bin/sh

python3 Preprocess.py training_set.json training_set_preprocess.json
python3 Preprocess.py test_set.json test_set_preprocess.json

