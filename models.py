import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import Config

class DeepHECT(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_size = Config.vocab_size
        embed_dim = Config.embedding_dim
        num_filters = Config.num_filters
        kernel_size = Config.kernel_size
        lstm_hidden = Config.lstm_hidden
        num_layers = Config.num_lstm_layers
        dropout_conv = Config.dropout_conv
        dropout_lstm = Config.dropout_lstm
        dropout_fc = Config.dropout_fc

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size,
                              padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout_conv)

        self.lstm = nn.LSTM(num_filters, lstm_hidden, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout_lstm if num_layers > 1 else 0)

        self.attn = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.bn = nn.BatchNorm1d(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, Config.num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)                           # (B, L, E)
        x = x.transpose(1, 2)                           # (B, E, L)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)                           # (B, L, C)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, L, H*2)

        attn_scores = self.attn(lstm_out).squeeze(-1)    # (B, L)
        mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)    # (B, L)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H*2)

        out = self.bn(context)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out