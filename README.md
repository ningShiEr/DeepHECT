# DeepHECT

## Description

DeepHECT is a lightweight deep learning model for identifying HECT family E3 ligases from protein sequences. The repository provides the source code, benchmark datasets, and pre-trained model weights used for sequence-based HECT protein identification.

## Dataset Information

The `data/` folder contains three FASTA files:

- `train.fasta`: training set
- `val.fasta`: validation set
- `test.fasta`: independent test set

All sequences are stored in FASTA format. Sequence headers should use one of the following label formats:

```text
>HECT|protein_id
>RING|protein_id
>UBOX|protein_id
>RANDOM|protein_id
```

`HECT` sequences are treated as positive samples. `RING`, `UBOX`, and `RANDOM` sequences are treated as negative samples.

## Code Information

The main files are:

- `config.py`: dataset paths, label mapping, model hyperparameters, and device settings
- `data_loader.py`: FASTA parsing and PyTorch data loading
- `models.py`: DeepHECT model architecture
- `utils.py`: evaluation metric calculation
- `main.py`: inference and evaluation on the test set
- `requirements.txt`: required Python packages
- `model/best_model.pth`: pre-trained model weights

## Usage Instructions

Install the required packages:

```bash
pip install -r requirements.txt
```

Run inference and evaluation on the test set:

```bash
python main.py
```

The script reports loss, accuracy, precision, recall, F1-score, MCC, AUC, and AUPRC on the test set.

## Requirements

- Python 3.8 or later
- PyTorch
- scikit-learn
- numpy

The exact package list is provided in `requirements.txt`.

## Methodology

Protein sequences are read from FASTA files and converted into amino acid indices. Unknown or non-standard residues are mapped to `X`. The model uses the sequence labels in the FASTA headers to assign binary classes, where HECT proteins are positive samples and all other classes are negative samples.

The provided `main.py` script loads the independent test set, restores the pre-trained model weights from `model/best_model.pth`, performs inference, and calculates classification metrics.

## Citations

If you use this repository, please cite the associated manuscript:

DeepHECT: A Lightweight Deep Learning Model for Sequence-Based Identification of HECT Family E3 Ligases.

## License and Contribution Guidelines

This repository is provided for academic review and research use. Please contact the authors before reusing or redistributing the code or dataset for other purposes.
