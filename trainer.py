import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import compute_metrics, get_scheduler

class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = get_scheduler(self.optimizer, config)
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.best_state = None

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_preds, all_labels, all_probs = [], [], []

        for seqs, lengths, labels in self.train_loader:
            seqs = seqs.to(self.config.device)
            lengths = lengths.to(self.config.device)
            labels = labels.to(self.config.device)

            self.optimizer.zero_grad()
            outputs = self.model(seqs, lengths)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total_samples
        metrics = compute_metrics(all_labels, all_probs, all_preds)
        return avg_loss, metrics

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for seqs, lengths, labels in loader:
                seqs = seqs.to(self.config.device)
                lengths = lengths.to(self.config.device)
                labels = labels.to(self.config.device)

                outputs = self.model(seqs, lengths)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / total_samples
        metrics = compute_metrics(all_labels, all_probs, all_preds)
        return avg_loss, metrics

    def fit(self, epochs):
        history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'train_mcc': [], 'val_mcc': []
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            if self.scheduler:
                if self.config.lr_scheduler == 'ReduceLROnPlateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auc'].append(train_metrics['auc'])
            history['val_auc'].append(val_metrics['auc'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1'])
            history['val_f1'].append(val_metrics['f1'])
            history['train_mcc'].append(train_metrics['mcc'])
            history['val_mcc'].append(val_metrics['mcc'])

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} AUC: {train_metrics['auc']:.4f} | "
                  f"Val Loss: {val_loss:.4f} AUC: {val_metrics['auc']:.4f}")

            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
                print(f"  -> 保存最佳模型 (AUC={self.best_val_auc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"早停于 epoch {epoch}")
                    break

        self.model.load_state_dict(self.best_state)
        return history