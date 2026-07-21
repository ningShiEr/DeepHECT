import torch

class Config:
    # Dataset file paths
    train_fasta = './data/train.fasta'
    val_fasta   = './data/val.fasta'
    test_fasta  = './data/test.fasta'

    # Amino acid index: 20 standard amino acids plus X for unknown residues
    AA_dict = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYVX')}
    vocab_size = len(AA_dict)

    # Binary label mapping: HECT is positive, other classes are negative
    label_map = {'HECT': 1, 'RING': 0, 'UBOX': 0, 'RANDOM': 0}
    num_classes = 2

    # Model hyperparameters
    embedding_dim = 130
    num_filters = 16
    kernel_size = 5
    lstm_hidden = 64
    num_lstm_layers = 2
    dropout_conv = 0.2
    dropout_lstm = 0.3
    dropout_fc = 0.5

    # Evaluation parameters
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
