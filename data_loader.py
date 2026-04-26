import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import Config

def read_fasta(fasta_path):
    sequences = []
    current_label = None
    current_seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_label is not None:
                    sequences.append((current_label, ''.join(current_seq)))
                header = line[1:]
                parts = header.split('|')
                current_label = parts[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_label is not None:
            sequences.append((current_label, ''.join(current_seq)))
    return sequences

class ProteinDataset(Dataset):
    def __init__(self, fasta_path):
        self.data = []
        for label, seq in read_fasta(fasta_path):
            if label not in Config.label_map:
                raise ValueError(f"未知标签 '{label}' 在文件 {fasta_path}")
            label_idx = Config.label_map[label]
            seq_indices = [Config.AA_dict.get(aa, Config.AA_dict['X']) for aa in seq]
            self.data.append((torch.tensor(seq_indices, dtype=torch.long), label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, lengths, labels

def get_loader(fasta_path, shuffle=False):
    """数据加载接口（供 main.py 调用）"""
    dataset = ProteinDataset(fasta_path)
    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader

