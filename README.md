# DeepHECT

DeepHECT is a lightweight sequence classifier for identifying HECT-family E3 ubiquitin ligases. This repository snapshot corresponds to the final, PAD-separated model used in the accompanying manuscript: 100-dimensional amino-acid embeddings, a 1D convolution kernel size of 7, and a two-layer BiLSTM with attention.

## Final model and evaluation

- Vocabulary: PAD is index 0; the 20 standard amino acids and unknown residue `X` use indices 1–21 (vocabulary size 22).
- Architecture: embedding dimension 100; 16 convolution filters with kernel size 7; two BiLSTM layers with 64 hidden units per direction; masked attention.
- Independent test set: 769 sequences (85 HECT, 684 non-HECT). Accuracy 0.9961, precision 0.9767, recall 0.9882, F1 0.9825, MCC 0.9803, AUROC 0.9999, and AUPRC 0.9995.
- Five-fold validation on the training set: accuracy 0.9981 ± 0.0010, precision 0.9951 ± 0.0036, recall 0.9951 ± 0.0017, F1 0.9951 ± 0.0025, and MCC 0.9901 ± 0.0049. These are internal training-set cross-validation results, not independent-test estimates.
- The final checkpoint contains 163,755 parameters.

## Repository contents

- `config.py`, `data.py`, `models.py`, `trainer.py`, `kfold.py`, `utils.py`, `visualize.py`, `main.py`: model, data, training, evaluation, and visualization code.
- `data/`: train, validation, and independent-test FASTA files. Labels are encoded in FASTA headers (`HECT`, `RING`, `UBOX`, or `RANDOM`).
- `model/DeepHECT/best_model.pth`: final 100×7 checkpoint used by the current code.
- `model/best_model.pth`: identical copy kept at the legacy path for compatibility with earlier scripts.
- `results/`: independent-test metrics, training history, and five-fold results.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

From the repository root, run:

```bash
python main.py
```

Choose `2` for test-only evaluation using the included checkpoint, `1` to train on the supplied train/validation sets and evaluate on the test set, `3` for five-fold cross-validation, or `4` to generate t-SNE visualizations. The code uses paths relative to this repository, so no machine-specific path editing is needed. A CUDA-capable GPU is optional; inference also runs on CPU.

## Data split

The dataset contains 7,704 clustered sequences: 6,164 training, 771 validation, and 769 independent-test sequences. The complete class counts are reported in the manuscript and its supplementary materials. The independent test set was reserved for final evaluation and was not used to choose the 100×7 hyperparameters.

## Citation

Please cite the associated manuscript:

> DeepHECT: A Lightweight Deep Learning Model for Sequence-Based Identification of HECT Family E3 Ligases.

## Reuse

This repository is provided for academic review and research use. Please contact the authors before reusing or redistributing the code or dataset for other purposes.
