# DeepHECT

DeepHECT is a lightweight deep learning model for identifying HECT family E3 ligases from protein sequences.

## Dataset
The `data/` folder contains:
- `train.fasta` – training set
- `val.fasta` – validation set
- `test.fasta` – independent test set

All sequences are in FASTA format. Sequence headers must be like:
HECT|protein_id
RING|protein_id
UBOX|protein_id
RANDOM|protein_id

## Pre-trained Model
The pre-trained model weights are in `model/best_model.pth`.

## Quick Start
1.Install dependencies:
pip install -r requirements.txt
2.Run inference on the test set:
python main.py
The script will output accuracy, precision, recall, F1‑score, MCC, AUC, and AUPRC on the test set.

## Requirements
Python 3.8+
PyTorch, scikit-learn, numpy (see requirements.txt)