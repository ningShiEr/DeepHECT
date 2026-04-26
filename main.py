import torch
import time
import numpy as np
from config import Config
from data_loader import get_loader
from models import DeepHECT
from utils import compute_metrics

def test():
    print("加载测试数据...")
    test_loader = get_loader(Config.test_fasta)

    print("加载模型...")
    model = DeepHECT().to(Config.device)
    checkpoint = torch.load('./model/best_model.pth', map_location=Config.device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    print("开始测试...")
    total_loss = 0
    total_samples = 0
    all_probs, all_preds, all_labels = [], [], []
    criterion = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    with torch.no_grad():
        for seqs, lengths, labels in test_loader:
            seqs = seqs.to(Config.device)
            lengths = lengths.to(Config.device)
            labels = labels.to(Config.device)

            outputs = model(seqs, lengths)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    avg_loss = total_loss / total_samples
    metrics = compute_metrics(all_labels, all_probs, all_preds)

    print("\n========== 测试结果 ==========")
    print(f"损失: {avg_loss:.4f}")
    print(f"推理用时: {end_time - start_time:.2f} 秒")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    test()