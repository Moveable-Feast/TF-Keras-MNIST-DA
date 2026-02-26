import argparse
import yaml
import numpy as np
import tensorflow as tf

import mlflow
import mlflow.tensorflow

from DATA.preprocessing import load_mnist, preprocess
from models.mlp import create_mlp

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config['data']['random_seed'])

    # 加载并预处理数据
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    # 构建模型
    model = create_mlp(
        hidden_units=config['model']['hidden_units'],
        activation=config['model']['activation']
    )
    model.compile(
        optimizer=config['training']['optimizer'],
        loss=config['training']['loss'],
        metrics=config['training']['metrics']
    )

    # ----- 新增：MLflow 实验设置 -----
    mlflow.set_experiment(config['logging']['experiment_name'])

    with mlflow.start_run():
        # 记录超参数
        mlflow.log_params({
            'hidden_units': config['model']['hidden_units'],
            'batch_size': config['training']['batch_size'],
            'epochs': config['training']['epochs'],
            'optimizer': config['training']['optimizer'],
            'validation_split': config['data']['validation_split']
        })

        # 训练
        history = model.fit(
            x_train, y_train,
            batch_size=config['training']['batch_size'],
            epochs=config['training']['epochs'],
            validation_split=config['data']['validation_split'],
            verbose=1
        )

        # 记录验证准确率（取最后一个 epoch 的值）
        val_acc = history.history['val_accuracy'][-1]
        mlflow.log_metric('val_accuracy', val_acc)

        # 记录模型到 MLflow
        mlflow.tensorflow.log_model(model, "model")

    # 保存模型
    model.save('models/mnist_mlp.h5')
    print("模型已保存至 models/mnist_mlp.h5")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    main(args.config)