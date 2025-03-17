# FocusFarm

This repository contains the code used in my (Karen's) [Signature Work](https://signature-work.dukekunshan.edu.cn/signature-work-overview/) project. 

## Setup

A virtual environment can be created by running:

```bash
pip install -r requirements.txt
```

## Usage

```python
# run for help message to see parameters of the module to generate synthetic data
python generate.py -h

# run to reproduce the sample dataset (which had 7315 total sequences)
python generate.py -n 200 -seed 11 -plot True

# now that we have our dataset, run this for help message to see parameters of the module that performs classification
python classify.py -h

# if creating the models, run this (replace the directory name with the one just generated--format is 'yyyymmdd_n')
python classify.py -f 20250302_200/synthetic_200.csv -n 10 -m all -seed 11

# if running classification using the presaved models
python classify.py -f 20250302_200/synthetic_200.csv -p 20250302_200/
# ensure the argument for -p is the folder (with the forward slash) that contains bayes_tsf.pkl and bayes_TSBF.pkl
```

```python
# additionally, if you wish to try the PAR synthesizer you can run this
python synthesize.py -h
python synthesize.py -f 20250302_200/synthetic_200.csv -n 200 -seed 11 -plot True
```

For `synthesize.py`, the SDV library may give a warning message which doesn't affect functionality; the synthesizer has already been saved and the only way to ensure the same synthetic dataset as in the paper is produced is to use the same synthesizer object (`synthesizer.pkl`) that's been fitted with the same input dataset as in the paper.

```python
# if you only want to visualize some plots on a synthetic dataset of your choice, run this and replace the filepath with the relevant one
python synthesize.py -f 20250302_200/synthetic_200 -n 200 -p 20250302_200/synthesizer.pkl -seed 11 -plot True
```