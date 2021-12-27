# Conformer OCR
## Introduction
Conformer OCR is an Optical Character Recognition toolkit built for researchers working on both OCR for both Vietnamese and English.
This project only focused on variants of vanilla Transformer and Feature Extraction (CNN-based approach).

This is also the first repo to utilize ConformerNet (https://arxiv.org/abs/2005.08100) for OCR.

## Architecture

<p align="center">
<img src="https://raw.githubusercontent.com/hoangtuanvu/conformer_ocr/master/visualization/architecture.png" width="512" height="614">
</p>

## Key Features
- Variants of Transformer (e.g., Vanilla, Conformer) encoder with CTC decoder.
- Both naive Pytorch and Pytorch Lightning are provided
- Beam search with N-gram Language model
- Accumulation gradient training

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
└── README.md 
```

## Tutorials

## Quick start
Train with naive Pytorch mode
```
cd scripts
python train.py
```

Train with Pytorch Lightning mode
```
cd scripts
python train_PT.py
```

## Pre-trained models
Coming soon...