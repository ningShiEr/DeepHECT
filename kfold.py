import os
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from torch.utils.data import Subset, DataLoader
from data import ProteinDataset, collate_fn
from config import Config
from models import get_model
from utils import get_scheduler
import json
from datetime import datetime

class KFoldCrossValidation:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.results = {}

    def get_labels(self, dataset):
        """从数据集中提取所有标签"""
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        return labels

    def run_cross_validation(self):
        """执行K折交叉验证"""
        print(f"\n开始 {self.n_splits} 折交叉验证...")
        full_dataset = ProteinDataset(Config.train_fasta)
        labels = self.get_labels(full_dataset)

        kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)), labels)):
            print(f"\n{'='*50}")
            print(f"训练第 {fold+1}/{self.n_splits} 折")
            print(f"{'='*50}")

            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=Config.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=Config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )

            fold_result = self.train_fold(fold, train_loader, val_loader)
            fold_results.append(fold_result)

            print(f"第 {fold+1} 折完成 - 最佳验证准确率: {fold_result['best_val_acc']:.4f} (epoch {fold_result['best_epoch']})")

        self.aggregate_results(fold_results)
        self.save_results()
        return self.results

    def train_fold(self, fold, train_loader, val_loader):
        """训练单个折，包含早停机制，记录最佳验证指标"""
        # 获取模型类并初始化
        model_class = get_model(Config.model_name)
        model = model_class(Config).to(Config.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate,
                                     weight_decay=Config.weight_decay)
        scheduler = get_scheduler(optimizer, Config)
        criterion = torch.nn.CrossEntropyLoss()

        # 早停相关变量
        best_val_acc = 0.0
        best_epoch = 0
        # 新增：记录最佳epoch对应的其他指标
        best_val_precision = 0.0
        best_val_recall = 0.0
        best_val_f1 = 0.0
        best_val_mcc = 0.0

        patience_counter = 0
        stopped_epoch = Config.num_epochs  # 默认全部训练完

        # 记录历史
        fold_history = {key: [] for key in [
            'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'train_precision', 'val_precision',
            'train_recall', 'val_recall',
            'train_f1', 'val_f1',
            'train_mcc', 'val_mcc'
        ]}

        for epoch in range(1, Config.num_epochs + 1):
            # ---------- 训练 ----------
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_preds, train_labels = [], []

            for seqs, lengths, labels in train_loader:
                seqs = seqs.to(Config.device)
                lengths = lengths.to(Config.device)
                labels = labels.to(Config.device)

                optimizer.zero_grad()
                outputs = model(seqs, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.gradient_clip)
                optimizer.step()

                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size
                train_total += batch_size
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()

                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # ---------- 验证 ----------
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for seqs, lengths, labels in val_loader:
                    seqs = seqs.to(Config.device)
                    lengths = lengths.to(Config.device)
                    labels = labels.to(Config.device)

                    outputs = model(seqs, lengths)
                    loss = criterion(outputs, labels)

                    batch_size = labels.size(0)
                    val_loss += loss.item() * batch_size
                    val_total += batch_size
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()

                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # 计算指标
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0

            def compute_fold_metrics(y_true, y_pred):
                precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                mcc = matthews_corrcoef(y_true, y_pred)
                return precision, recall, f1, mcc

            train_prec, train_rec, train_f1, train_mcc = compute_fold_metrics(train_labels, train_preds)
            val_prec, val_rec, val_f1, val_mcc = compute_fold_metrics(val_labels, val_preds)

            # 记录历史
            fold_history['train_loss'].append(train_loss / train_total if train_total > 0 else train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_loss / val_total if val_total > 0 else val_loss)
            fold_history['val_acc'].append(val_acc)
            fold_history['train_precision'].append(train_prec)
            fold_history['train_recall'].append(train_rec)
            fold_history['train_f1'].append(train_f1)
            fold_history['train_mcc'].append(train_mcc)
            fold_history['val_precision'].append(val_prec)
            fold_history['val_recall'].append(val_rec)
            fold_history['val_f1'].append(val_f1)
            fold_history['val_mcc'].append(val_mcc)

            # 更新学习率
            if scheduler:
                if Config.lr_scheduler == 'ReduceLROnPlateau':
                    scheduler.step(val_loss / val_total if val_total > 0 else val_loss)
                else:
                    scheduler.step()

            # 早停检查 & 记录最佳指标
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                # 同时记录该epoch的其他验证指标
                best_val_precision = val_prec
                best_val_recall = val_rec
                best_val_f1 = val_f1
                best_val_mcc = val_mcc
                self.save_fold_model(model, fold)   # 保存当前折的最佳模型
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.patience:
                    stopped_epoch = epoch
                    print(f"  折 {fold+1} 早停于 epoch {epoch} (最佳 epoch {best_epoch})")
                    break   # 提前终止

            if epoch % 10 == 0:
                print(f"  Fold {fold+1}, Epoch {epoch:03d} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # 返回结果，包含最佳指标
        return {
            'fold': fold,
            'best_epoch': best_epoch,
            'stopped_epoch': stopped_epoch,
            'best_val_acc': best_val_acc,
            'best_val_precision': best_val_precision,
            'best_val_recall': best_val_recall,
            'best_val_f1': best_val_f1,
            'best_val_mcc': best_val_mcc,
            'history': fold_history,   # 只包含实际训练的epoch
            # 保留final指标（可选），但汇总时不再使用
            'final_val_acc': fold_history['val_acc'][-1] if fold_history['val_acc'] else 0,
            'final_val_f1': fold_history['val_f1'][-1] if fold_history['val_f1'] else 0,
            'final_val_mcc': fold_history['val_mcc'][-1] if fold_history['val_mcc'] else 0,
        }

    def save_fold_model(self, model, fold):
        """保存当前折的最佳模型"""
        model_dir = os.path.join('model', Config.model_name, 'kfold')
        os.makedirs(model_dir, exist_ok=True)
        torch.save({
            'state_dict': model.state_dict(),
            'fold': fold,
            'model_name': Config.model_name
        }, os.path.join(model_dir, f'best_model_fold_{fold}.pth'))

    def aggregate_results(self, fold_results):
        """汇总所有折的最佳指标结果"""
        # 提取每折的最佳指标
        best_accs = [r['best_val_acc'] for r in fold_results]
        best_precs = [r['best_val_precision'] for r in fold_results]
        best_recs = [r['best_val_recall'] for r in fold_results]
        best_f1s = [r['best_val_f1'] for r in fold_results]
        best_mccs = [r['best_val_mcc'] for r in fold_results]

        self.results = {
            'fold_results': fold_results,
            'mean_best_val_acc': float(np.mean(best_accs)),
            'std_best_val_acc': float(np.std(best_accs)),
            'mean_best_val_precision': float(np.mean(best_precs)),
            'std_best_val_precision': float(np.std(best_precs)),
            'mean_best_val_recall': float(np.mean(best_recs)),
            'std_best_val_recall': float(np.std(best_recs)),
            'mean_best_val_f1': float(np.mean(best_f1s)),
            'std_best_val_f1': float(np.std(best_f1s)),
            'mean_best_val_mcc': float(np.mean(best_mccs)),
            'std_best_val_mcc': float(np.std(best_mccs)),
            'cross_validation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def save_results(self):
        """保存交叉验证结果到JSON文件"""
        results_dir = os.path.join('results', 'kfold')
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, f'{Config.model_name}_kfold_results.json')
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, float)) else x)
        print(f"\n交叉验证结果已保存至: {file_path}")

def run_kfold_validation():
    """运行K折交叉验证的便捷函数"""
    kfold = KFoldCrossValidation(n_splits=5)
    results = kfold.run_cross_validation()
    return results

if __name__ == "__main__":
    run_kfold_validation()