from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import Config


def read_fasta(fasta_path):
    sequences = []
    current_header = None
    current_seq = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

    if current_header is not None:
        sequences.append((current_header, "".join(current_seq)))
    return sequences


def load_label_map(labels_path):
    labels_path = Path(labels_path)
    if not labels_path.exists():
        return None

    df = pd.read_csv(labels_path, sep="\t")
    if "sequence_id" not in df.columns:
        return None

    if "label" not in df.columns and "label_name" not in df.columns:
        return None

    mapping = {}
    for _, row in df.iterrows():
        seq_id = str(row["sequence_id"]).strip()
        if "label" in df.columns and pd.notna(row["label"]):
            mapping[seq_id] = int(row["label"])
            continue

        label_key = str(row["label_name"]).strip().upper()
        if label_key == "STRICT_DOMAIN_POSITIVE":
            label_key = "HECT"
        elif label_key in {"HARD_NEGATIVE", "CLEAN_NEGATIVE", "BACKGROUND_NEGATIVE"}:
            label_key = "RING"
        if label_key in Config.label_map:
            mapping[seq_id] = Config.label_map[label_key]
    return mapping


def parse_sequence_id(header):
    return header.split()[0]


def infer_label_from_header(header):
    return header.split("|", 1)[0]


class ProteinDataset(Dataset):
    def __init__(self, fasta_path, labels_path=None):
        self.data = []
        label_map = load_label_map(labels_path) if labels_path else None

        for header, seq in read_fasta(fasta_path):
            seq_id = parse_sequence_id(header)
            if label_map is not None and seq_id in label_map:
                label_idx = label_map[seq_id]
            else:
                label_key = infer_label_from_header(header)
                if label_key not in Config.label_map:
                    raise ValueError(f"Unknown label '{label_key}' in {fasta_path}")
                label_idx = Config.label_map[label_key]

            seq_indices = [Config.AA_dict.get(aa, Config.AA_dict["X"]) for aa in seq]
            self.data.append((torch.tensor(seq_indices, dtype=torch.long), label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=Config.PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, lengths, labels


def get_loader(fasta_path, labels_path=None, shuffle=False):
    dataset = ProteinDataset(fasta_path, labels_path)
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=getattr(Config, "num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def get_dataloader(fasta_path, config=None, shuffle=False, labels_path=None):
    if labels_path is None and config is not None:
        fasta_str = str(fasta_path)
        if "train" in fasta_str:
            labels_path = getattr(config, "train_labels", None)
        elif "val" in fasta_str:
            labels_path = getattr(config, "val_labels", None)
        elif "test" in fasta_str:
            labels_path = getattr(config, "test_labels", None)
    return get_loader(fasta_path, labels_path=labels_path, shuffle=shuffle)
