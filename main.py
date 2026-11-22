import json
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data import get_dataloader
from kfold import KFoldCrossValidation
from models import get_model
from trainer import Trainer
from visualize import visualize_multistage_tsne_combined


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def save_roc_data(model, data_loader, device, save_path):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for seqs, lengths, labels in data_loader:
            seqs = seqs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            outputs = model(seqs, lengths)
            probs = torch.softmax(outputs, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    np.savez(save_path, y_true=np.array(all_labels), y_prob=np.array(all_probs))
    print(f"ROC data saved to: {save_path}")


def run_regular_training():
    print("\n========== Regular Training (Train/Val/Test) ==========")
    train_loader = get_dataloader(Config.train_fasta, Config, shuffle=True)
    val_loader = get_dataloader(Config.val_fasta, Config, shuffle=False)
    test_loader = get_dataloader(Config.test_fasta, Config, shuffle=False)

    model_class = get_model(Config.model_name)
    model = model_class(Config)
    trainer = Trainer(model, Config, train_loader, val_loader)
    history = trainer.fit(Config.num_epochs)

    save_dir = os.path.join("model", Config.model_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(trainer.best_state, os.path.join(save_dir, "best_model.pth"))
    print(f"Best model saved to: {save_dir}/best_model.pth")

    history_serializable = {
        key: [convert_to_serializable(v) for v in value] if isinstance(value, list) else convert_to_serializable(value)
        for key, value in history.items()
    }
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=2, ensure_ascii=False)
    print(f"Training history saved to: {history_path}")

    start_time = time.time()
    test_loss, test_metrics = trainer.validate(test_loader)
    predict_time = time.time() - start_time

    print("\n========== Test Results ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"prediction_time_seconds: {predict_time:.2f}")

    test_metrics["prediction_time_seconds"] = predict_time
    test_result_path = os.path.join(save_dir, "test_results.json")
    with open(test_result_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"Test results saved to: {test_result_path}")

    save_roc_data(model, test_loader, Config.device, os.path.join(save_dir, "test_roc_data.npz"))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_auc"], label="Train AUC")
    plt.plot(history["val_auc"], label="Val AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()

    plt.suptitle(f"Training Curves - {Config.model_name}")
    plt.tight_layout()
    curve_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {curve_path}")

    return history, test_metrics


def run_test_only():
    print("\n========== Test Only ==========")
    model_path = os.path.join("model", Config.model_name, "best_model.pth")
    save_dir = os.path.join("model", Config.model_name)

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    test_loader = get_dataloader(Config.test_fasta, Config, shuffle=False)
    model_class = get_model(Config.model_name)
    model = model_class(Config)

    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(Config.device)

    trainer = Trainer(model, Config, None, None)

    start_time = time.time()
    test_loss, test_metrics = trainer.validate(test_loader)
    predict_time = time.time() - start_time

    print("\n========== Test Results ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"prediction_time_seconds: {predict_time:.2f}")

    test_metrics["prediction_time_seconds"] = predict_time
    test_result_path = os.path.join(save_dir, "test_results_latest.json")
    with open(test_result_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    print(f"Test results saved to: {test_result_path}")

    save_roc_data(model, test_loader, Config.device, os.path.join(save_dir, "test_roc_data.npz"))


def run_kfold():
    print("\n========== K-Fold Cross Validation ==========")
    kfold = KFoldCrossValidation(n_splits=5)
    results = kfold.run_cross_validation()
    print(f"Mean best val acc: {results['mean_best_val_acc']:.4f}")
    print(f"Mean best val precision: {results['mean_best_val_precision']:.4f}")
    print(f"Mean best val recall: {results['mean_best_val_recall']:.4f}")
    print(f"Mean best val f1: {results['mean_best_val_f1']:.4f}")
    print(f"Mean best val mcc: {results['mean_best_val_mcc']:.4f}")
    return results


def run_tsne():
    print("\n========== t-SNE Visualization ==========")
    model_path = os.path.join("model", Config.model_name, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model_class = get_model(Config.model_name)
    model = model_class(Config)
    state_dict = torch.load(model_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    elif isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.to(Config.device)

    test_loader = get_dataloader(Config.test_fasta, Config, shuffle=False)
    visualize_multistage_tsne_combined(model, test_loader, Config.device)


def main():
    set_seed(Config.seed)
    print(f"Device: {Config.device}")
    print(f"Model: {Config.model_name}")
    print(f"Train FASTA: {Config.train_fasta}")
    print(f"Val FASTA: {Config.val_fasta}")
    print(f"Test FASTA: {Config.test_fasta}")
    print("Choose mode:")
    print("1. Train + Val + Test")
    print("2. Test only")
    print("3. K-fold")
    print("4. t-SNE")

    choice = input("Enter 1-4: ").strip()
    if choice == "1":
        run_regular_training()
    elif choice == "2":
        run_test_only()
    elif choice == "3":
        run_kfold()
    elif choice == "4":
        run_tsne()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
