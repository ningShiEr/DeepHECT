from pathlib import Path

import torch

class Config:
    data_root = Path(__file__).resolve().parent / "data"
    train_fasta = f"{data_root}/train.fasta"
    val_fasta = f"{data_root}/val.fasta"
    test_fasta = f"{data_root}/test.fasta"
    train_labels = f"{data_root}/train_labels.tsv"
    val_labels = f"{data_root}/val_labels.tsv"
    test_labels = f"{data_root}/test_labels.tsv"

    # Reserve PAD=0; encode 20 amino acids plus unknown X at 1..21.
    _tokens = "ARNDCQEGHILKMFPSTWYVX"
    AA_dict = {aa: i + 1 for i, aa in enumerate(_tokens)}
    PAD_IDX = 0
    vocab_size = len(AA_dict) + 1

    label_map = {"HECT": 1, "RING": 0, "UBOX": 0, "RANDOM": 0}
    num_classes = 2
    class_names = ["non-HECT", "HECT"]

    model_name = "DeepHECT"
    embedding_dim = 100
    num_filters = 16
    kernel_size = 7
    lstm_hidden = 64
    num_lstm_layers = 2
    dropout_conv = 0.2
    dropout_lstm = 0.3
    dropout_fc = 0.5

    batch_size = 64
    num_workers = 0
    learning_rate = 0.001
    num_epochs = 100
    weight_decay = 1e-5
    patience = 10
    gradient_clip = 1.0

    lr_scheduler = "CosineAnnealing"
    step_size = 30
    gamma = 0.1
    T_max = 50
    eta_min = 1e-6
    min_lr = 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
