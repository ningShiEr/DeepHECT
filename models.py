import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ==================== DeepHECT（原模型）====================
class DeepHECT(nn.Module):
    """基础模型，支持变长序列"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_dim = config.embedding_dim
        num_filters = config.num_filters
        kernel_size = config.kernel_size
        lstm_hidden = config.lstm_hidden
        num_layers = config.num_lstm_layers
        dropout_conv = config.dropout_conv
        dropout_lstm = config.dropout_lstm
        dropout_fc = config.dropout_fc

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
        self.fc2 = nn.Linear(64, config.num_classes)

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


# ==================== 消融模型（基于DeepHECT架构）====================
class DeepHECTNoConv(nn.Module):
    """消融：移除卷积层"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_dim = config.embedding_dim
        lstm_hidden = config.lstm_hidden
        num_layers = config.num_lstm_layers
        dropout_lstm = config.dropout_lstm
        dropout_fc = config.dropout_fc

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout_lstm if num_layers > 1 else 0)

        self.attn = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.bn = nn.BatchNorm1d(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, config.num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)                           # (B, L, E)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, L, H*2)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        out = self.bn(context)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class DeepHECTNoAttention(nn.Module):
    """消融：移除注意力，使用LSTM最后有效时间步输出"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_dim = config.embedding_dim
        num_filters = config.num_filters
        kernel_size = config.kernel_size
        lstm_hidden = config.lstm_hidden
        num_layers = config.num_lstm_layers
        dropout_conv = config.dropout_conv
        dropout_lstm = config.dropout_lstm
        dropout_fc = config.dropout_fc

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size,
                              padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout_conv)

        self.lstm = nn.LSTM(num_filters, lstm_hidden, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout_lstm if num_layers > 1 else 0)

        self.bn = nn.BatchNorm1d(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, config.num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, L, H*2)

        # 取最后有效时间步的输出
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, lstm_out.size(2))
        context = torch.gather(lstm_out, 1, idx).squeeze(1)  # (B, H*2)

        out = self.bn(context)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class DeepHECTUnidirectional(nn.Module):
    """消融：单向LSTM"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_dim = config.embedding_dim
        num_filters = config.num_filters
        kernel_size = config.kernel_size
        lstm_hidden = config.lstm_hidden
        num_layers = config.num_lstm_layers
        dropout_conv = config.dropout_conv
        dropout_lstm = config.dropout_lstm
        dropout_fc = config.dropout_fc

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size,
                              padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout_conv)

        self.lstm = nn.LSTM(num_filters, lstm_hidden, num_layers,
                            batch_first=True, bidirectional=False,
                            dropout=dropout_lstm if num_layers > 1 else 0)

        self.attn = nn.Linear(lstm_hidden, 1, bias=False)
        self.bn = nn.BatchNorm1d(lstm_hidden)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.fc2 = nn.Linear(64, config.num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, L, H)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)

        out = self.bn(context)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class DeepHECTNoBatchNorm(nn.Module):
    """消融：无批归一化"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_dim = config.embedding_dim
        num_filters = config.num_filters
        kernel_size = config.kernel_size
        lstm_hidden = config.lstm_hidden
        num_layers = config.num_lstm_layers
        dropout_conv = config.dropout_conv
        dropout_lstm = config.dropout_lstm
        dropout_fc = config.dropout_fc

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size,
                              padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout_conv)

        self.lstm = nn.LSTM(num_filters, lstm_hidden, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout_lstm if num_layers > 1 else 0)

        self.attn = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, config.num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class DeepHECTNoDropout(nn.Module):
    """消融：无任何Dropout"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_dim = config.embedding_dim
        num_filters = config.num_filters
        kernel_size = config.kernel_size
        lstm_hidden = config.lstm_hidden
        num_layers = config.num_lstm_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size,
                              padding=kernel_size // 2)
        self.relu = nn.ReLU()
        # 无conv_dropout

        self.lstm = nn.LSTM(num_filters, lstm_hidden, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=0)  # LSTM内部无dropout

        self.attn = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.bn = nn.BatchNorm1d(lstm_hidden * 2)
        # 无全连接dropout
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, config.num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        # 跳过conv_dropout
        x = x.transpose(1, 2)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        out = self.bn(context)
        # 无dropout
        out = F.relu(self.fc1(out))
        # 无dropout
        out = self.fc2(out)
        return out


class DeepHECTSimple(nn.Module):
    """消融：极简模型（嵌入+平均池化+全连接）"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_dim = config.embedding_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.fc2 = nn.Linear(64, config.num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)  # (B, L, E)
        # 掩码平均池化
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / lengths.float().unsqueeze(1)  # (B, E)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class DeepHECTSingleLSTM(nn.Module):
    """消融：单层LSTM（而非多层）"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        embed_dim = config.embedding_dim
        num_filters = config.num_filters
        kernel_size = config.kernel_size
        lstm_hidden = config.lstm_hidden
        dropout_conv = config.dropout_conv
        dropout_fc = config.dropout_fc

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size,
                              padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv_dropout = nn.Dropout(dropout_conv)

        # 单层双向LSTM
        self.lstm = nn.LSTM(num_filters, lstm_hidden, num_layers=1,
                            batch_first=True, bidirectional=True,
                            dropout=0)

        self.attn = nn.Linear(lstm_hidden * 2, 1, bias=False)
        self.bn = nn.BatchNorm1d(lstm_hidden * 2)
        self.dropout = nn.Dropout(dropout_fc)
        self.fc1 = nn.Linear(lstm_hidden * 2, 64)
        self.fc2 = nn.Linear(64, config.num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_dropout(x)
        x = x.transpose(1, 2)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        mask = torch.arange(lstm_out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)

        out = self.bn(context)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# ==================== 模型选择器 ====================
def get_model(model_name):
    models = {
        'DeepHECT': DeepHECT,
        'DeepHECTNoConv': DeepHECTNoConv,
        'DeepHECTNoAttention': DeepHECTNoAttention,
        'DeepHECTUnidirectional': DeepHECTUnidirectional,
        'DeepHECTNoBatchNorm': DeepHECTNoBatchNorm,
        'DeepHECTNoDropout': DeepHECTNoDropout,
        'DeepHECTSimple': DeepHECTSimple,
        'DeepHECTSingleLSTM': DeepHECTSingleLSTM,
    }
    if model_name not in models:
        raise ValueError(f"未知模型名: {model_name}，可用模型: {list(models.keys())}")
    return models[model_name]