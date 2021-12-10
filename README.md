# Conformer OCR
## Introduction
Conformer OCR is an Optical Character Recognition toolkit built for researchers working on both OCR for both Vietnamese and English.
This project only focused on variants of vanilla Transformer and Feature Extraction (CNN-based approach).

This is also the first repo to utilize ConformerNet (https://arxiv.org/abs/2005.08100) for OCR.

## Key Features

## Install dependencies
```
cd transformer_ocr
pip install -r requirements/requirements.txt
```

## Directory structure
To modulize the repo, the current structure is adopted as follows:
```bash 
├── conf # configurations
│   ├── dataset
│   ├── model
│   ├── optimizer
│   ├── pl_params
│   └── config.yaml
├── requirements # Where store different requirements if needed
│   └── requirements.txt
├── scripts # Where start your training/evaluation/testing models 
│   ├── train.py
│   └── train_PT.py
├── transformer_ocr # Main resource
├── README.md
└── run_CXR.py # main file will be store here 
```

## Tutorials

## Quick start
```
cd scripts
python train_PT.py
```

## Pre-trained models
Coming soon...