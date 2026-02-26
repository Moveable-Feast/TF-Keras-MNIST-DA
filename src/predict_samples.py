import os
import yaml
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from DATA.preprocessing import load_mnist, preprocess

def set_seed(seed=42):
    np.random.seed(seed)

def plot_sample_images(images, true_labels, pred_labels, num_samples=10, save_path='reports/figures/sample_results.png'):
    """
    显示一组样本图片，标题显示真实值和预测值。
    
    Parameters:
    - images: 原始图片数据（形状为 [n, 28, 28] 或展平后的 [n, 784]）
    - true_labels: 真实标签列表
    - pred_labels: 预测标签列表
    - num_samples: 要显示的样本数量
    - save_path: 如果提供，保存图像到该路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 确保图片是 28x28 的形状
    if images.shape[1:] != (28, 28):
        # 假设是展平的 (784,)，需要重塑
        images = images.reshape(-1, 28, 28)
    
    # 随机选择 num_samples 个样本的索引
    indices = random.sample(range(len(images)), min(num_samples, len(images)))

    # 计算网格布局：尽量接近正方形
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))

    plt.figure(figsize=(cols*2, rows*2))

    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"True: {true_labels[idx]}\nPred: {pred_labels[idx]}")
        plt.axis('off')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description='展示测试集样本的预测结果')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='要显示的样本数量 (默认: 10)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='保存图像的文件路径（可选）')
    args = parser.parse_args()

    # 加载测试数据
    print("加载测试数据...")
    (_, _), (x_test, y_test) = load_mnist()          # x_test 原始形状 (10000, 28, 28)
    # 对测试数据进行预处理（用于模型预测）
    x_test_flat, y_test_onehot = preprocess(x_test, y_test)   # 展平为 (10000, 784), one-hot 编码

    # 加载模型
    print("加载模型...")
    model = keras.models.load_model('models/mnist_mlp.h5')

    # 预测
    print("进行预测...")
    y_pred_probs = model.predict(x_test_flat)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test  # 原始标签（未 one-hot）

    # 显示样本
    print(f"随机展示 {args.num_samples} 个样本...")
    plot_sample_images(x_test, y_true, y_pred, num_samples=args.num_samples)

if __name__ == '__main__':
    main()