# DeepHECT

DeepHECT 是一个轻量级深度学习模型，用于从蛋白质序列中识别 HECT 家族 E3 连接酶。

## 数据集
data/ 文件夹包含：
train.fasta – 训练集
val.fasta – 验证集
test.fasta – 独立测试集

所有序列均为 FASTA 格式，序列头格式示例：

HECT|蛋白质ID
RING|蛋白质ID
UBOX|蛋白质ID
RANDOM|蛋白质ID

## 预训练模型
预训练模型权重文件为 model/best_model.pth。

## Quick Start
快速运行
1.安装依赖：
pip install -r requirements.txt
2.在测试集上运行推理：
python main.py

程序将输出准确率、精确率、召回率、F1 值、MCC、AUC 和 AUPRC。

## 环境依赖
Python 3.8+
PyTorch、scikit-learn、numpy（详见 requirements.txt）