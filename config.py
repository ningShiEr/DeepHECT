import torch

class Config:
    # 数据路径
    train_fasta = './data/train.fasta'
    val_fasta   = './data/val.fasta'
    test_fasta  = './data/test.fasta'

    # 氨基酸索引（20种标准氨基酸 + X）
    AA_dict = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYVX')}
    vocab_size = len(AA_dict)

    # 标签映射（二分类：HECT=1，其他=0）
    label_map = {'HECT': 1, 'RING': 0, 'UBOX': 0, 'RANDOM': 0}
    num_classes = 2

    # 模型超参数（必须与训练时一致）
    embedding_dim = 130
    num_filters = 16
    kernel_size = 5
    lstm_hidden = 64
    num_lstm_layers = 2
    dropout_conv = 0.2
    dropout_lstm = 0.3
    dropout_fc = 0.5

    # 测试参数
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')