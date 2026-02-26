import numpy as np
from tensorflow import keras

def load_mnist():
    """加载 MNIST 原始数据"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def preprocess(x, y, num_classes=10):
    """归一化、重塑、one-hot 编码"""
    x = x.reshape(x.shape[0], -1).astype("float32") / 255.0
    y = keras.utils.to_categorical(y, num_classes)
    return x, y