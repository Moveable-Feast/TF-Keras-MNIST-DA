import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from DATA.preprocessing import load_mnist, preprocess

def plot_confusion_matrix(y_true, y_pred, classes, save_path='reports/figures/cm.png'):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    plt.savefig(save_path)
    plt.show()

def main():
    # 加载测试数据
    (_, _), (x_test, y_test) = load_mnist()
    x_test, y_test_onehot = preprocess(x_test, y_test)

    # 加载模型
    model = keras.models.load_model('models/mnist_mlp.h5')

    # 预测
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test  # 原始标签（未 one-hot）

    # 打印分类报告
    print(classification_report(y_true, y_pred, digits=4))

    # 生成混淆矩阵
    plot_confusion_matrix(y_true, y_pred, classes=list(range(10)))

if __name__ == '__main__':
    main()