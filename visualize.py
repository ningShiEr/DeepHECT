import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
import os
import logging
import json
from datetime import datetime
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_multistage_features(model, dataloader, device):
    """提取特征（保持不变）"""
    model.eval()
    stage_features = {
        'embedding': [], 'conv': [], 'lstm': [], 'attention': []
    }
    all_labels = []

    with torch.no_grad():
        for seqs, lengths, labels in dataloader:
            seqs = seqs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            x_embed = model.embedding(seqs)
            embed_pooled = torch.mean(x_embed, dim=1)
            stage_features['embedding'].append(embed_pooled.cpu())

            x_conv = x_embed.transpose(1, 2)
            x_conv = model.conv(x_conv)
            x_conv = model.relu(x_conv)
            x_conv = model.conv_dropout(x_conv)
            x_conv = x_conv.transpose(1, 2)
            conv_pooled = torch.mean(x_conv, dim=1)
            stage_features['conv'].append(conv_pooled.cpu())

            lstm_out, _ = model.lstm(x_conv)
            lstm_pooled = torch.mean(lstm_out, dim=1)
            stage_features['lstm'].append(lstm_pooled.cpu())

            attn_scores = model.attn(lstm_out).squeeze(-1)
            mask = torch.arange(lstm_out.size(1), device=device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
            stage_features['attention'].append(context.cpu())

            all_labels.append(labels.cpu())

    for stage in stage_features:
        stage_features[stage] = torch.cat(stage_features[stage], dim=0).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return stage_features, all_labels

def visualize_multistage_tsne_combined(model, dataloader, device, save_dir='img/tsne_multistage', perplexity=30):
    """生成2x2 t-SNE图，并为每个阶段单独保存JSON文件"""
    stage_features, labels = extract_multistage_features(model, dataloader, device)
    if not any(len(features) > 0 for features in stage_features.values()):
        logging.warning("没有提取到特征，请检查数据或模型")
        return

    config = Config()
    class_names = config.class_names
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model_name

    stages = ['embedding', 'conv', 'lstm', 'attention']
    stage_titles = {
        'embedding': 'Embedding Layer',
        'conv': 'After 1D Convolution',
        'lstm': 'After BiLSTM',
        'attention': 'After Attention'
    }
    colors = ['#1f77b4', '#ff7f0e']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    reduced_coords = {}

    for i, stage in enumerate(stages):
        if stage in stage_features and len(stage_features[stage]) > 0:
            logging.info(f"生成 {stage} 阶段 t-SNE，样本数: {len(stage_features[stage])}")
            try:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                reduced = tsne.fit_transform(stage_features[stage])
                reduced_coords[stage] = reduced

                ax = axes[i]
                unique_labels = np.unique(labels)
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    class_name = class_names[label] if label < len(class_names) else f'Class_{label}'
                    ax.scatter(reduced[mask, 0], reduced[mask, 1],
                               c=colors[j % len(colors)], label=class_name,
                               alpha=0.8, s=40, edgecolors='white', linewidth=0.5)

                ax.set_title(stage_titles[stage], fontsize=12, fontweight='bold')
                ax.text(-0.1, 1.05, chr(65 + i), transform=ax.transAxes,
                        fontsize=16, fontweight='bold', va='top')
            except Exception as e:
                logging.error(f"{stage} t-SNE 计算失败: {e}")
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                             transform=axes[i].transAxes, fontsize=10)
                axes[i].set_title(stage_titles[stage])
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center',
                         transform=axes[i].transAxes, fontsize=10)
            axes[i].set_title(stage_titles[stage])

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # 保存图片
    svg_path = os.path.join(save_dir, f"{model_name}_multistage_tsne_{timestamp}.svg")
    png_path = os.path.join(save_dir, f"{model_name}_multistage_tsne_{timestamp}.png")
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', facecolor='white', dpi=300)
    logging.info(f"t-SNE 图已保存至 {save_dir}")

    # 保存 NPZ（完整二进制）
    npz_path = os.path.join(save_dir, f"{model_name}_tsne_data_{timestamp}.npz")
    np.savez_compressed(
        npz_path,
        labels=labels,
        class_names=np.array(class_names),
        stage_names=np.array(stages),
        embedding_coords=reduced_coords.get('embedding', np.array([])),
        conv_coords=reduced_coords.get('conv', np.array([])),
        lstm_coords=reduced_coords.get('lstm', np.array([])),
        attention_coords=reduced_coords.get('attention', np.array([])),
        perplexity=perplexity,
        timestamp=timestamp,
        model_name=model_name
    )
    logging.info(f"NPZ 数据已保存至 {npz_path}")

    # ---------- 修改：为每个阶段单独保存 JSON 文件 ----------
    for stage in stages:
        if stage in reduced_coords and len(reduced_coords[stage]) > 0:
            stage_json_path = os.path.join(save_dir, f"{model_name}_tsne_{stage}_{timestamp}.json")
            stage_data = {
                "meta": {
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "stage": stage,
                    "perplexity": perplexity,
                    "class_names": class_names,
                    "num_samples": int(len(labels))
                },
                "labels": labels.tolist(),
                "coordinates": reduced_coords[stage].tolist()  # 仅该阶段的坐标
            }
            with open(stage_json_path, 'w', encoding='utf-8') as f:
                json.dump(stage_data, f, indent=2, ensure_ascii=False)
            logging.info(f"JSON 数据（{stage}阶段）已保存至 {stage_json_path}")
    # ----------------------------------------------------------------

    plt.show()
    return fig