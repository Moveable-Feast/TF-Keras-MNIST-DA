# TF-Keras-MNIST-DA

*AL generated. Don't be so serious.*

A professional deep learning project demonstrating best practices for building, training, evaluating, and tracking a Multi-Layer Perceptron (MLP) model on the MNIST handwritten digit dataset using TensorFlow/Keras.

This project serves as a template for **reproducible, well-structured, and production-ready** machine learning projects, including:

- Modular code organization (`src/`)
- Configuration-driven experiments (`configs/config.yaml`)
- Experiment tracking with MLflow
- Model evaluation with metrics and visualizations
- Sample prediction visualization
- Adherence to open-source standards (LICENSE, .gitignore, README)

---

## Features

- **Data Module**: Loads and preprocesses MNIST (normalization, reshaping, one-hot encoding).
- **Model Definition**: Configurable MLP with one hidden layer.
- **Training Script**: Accepts a YAML configuration file, logs parameters/metrics to MLflow, saves the model.
- **Evaluation Script**: Computes classification report and confusion matrix (with automatic directory creation).
- **Prediction Samples**: Randomly selects test images and displays true vs. predicted labels.
- **Experiment Tracking**: MLflow integration for logging hyperparameters, metrics, and model artifacts.
- **Reproducibility**: Fixed random seed and dependency management via `requirements.txt`.

---

## Project Structure

```
TF-Keras-MNIST-DA/
│
├── .github/                   # GitHub issue/PR templates (optional)
├── configs/                   
│   └── config.yaml             # Training configuration
├── models/                     # Saved trained models (ignored by git)
├── reports/                    
│   └── figures/                 # Generated plots (confusion matrix, samples)
├── src/                         # Source code
│   ├── data/                    
│   │   ├── __init__.py
│   │   ├── download.py          # (Optional) data download logic
│   │   └── preprocessing.py     # load_mnist, preprocess
│   ├── models/                  
│   │   ├── __init__.py
│   │   └── mlp.py               # create_mlp model definition
│   ├── train.py                 # Training script with MLflow
│   ├── evaluate.py               # Evaluation script (metrics + confusion matrix)
│   ├── predict_samples.py        # Visualize random test predictions
│   └── utils/                    # (Optional) helper functions
├── .gitignore
├── LICENSE                      # MIT License
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── setup.py                     # Editable install (optional)
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/TF-Keras-MNIST-DA.git
cd TF-Keras-MNIST-DA
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If you want to install the project in editable mode (for development):
```bash
pip install -e .
```

---

## Usage

All commands should be run from the **project root directory**.

### 1. Train the model
```bash
python src/train.py --config configs/config.yaml
```

This will:
- Load configuration from `configs/config.yaml`
- Train the MLP model
- Log hyperparameters and final validation accuracy to MLflow
- Save the trained model to `models/mnist_mlp.h5`

### 2. View experiment logs with MLflow
```bash
mlflow ui
```
Then open http://localhost:5000 in your browser.

### 3. Evaluate the model
```bash
python src/evaluate.py
```
Generates a classification report and saves the confusion matrix to `reports/figures/cm.png`.

### 4. Visualize sample predictions
```bash
python src/predict_samples.py --num_samples 16 --save_path reports/figures/samples.png
```
Displays 16 random test images with true and predicted labels, optionally saving the figure.

---

## Configuration

The training configuration is stored in `configs/config.yaml`. You can modify it to experiment with different hyperparameters:

```yaml
data:
  dataset: mnist
  validation_split: 0.2
  random_seed: 42

model:
  name: mlp
  hidden_units: 512
  activation: relu
  output_units: 10
  output_activation: softmax

training:
  batch_size: 128
  epochs: 10
  optimizer: rmsprop
  loss: categorical_crossentropy
  metrics: ["accuracy"]

logging:
  experiment_name: mnist_mlp
```

---

## Results

After training for 10 epochs with the default configuration, the model achieves:

- **Test Accuracy**: ~98.07%
- **Confusion Matrix**: Saved in `reports/figures/cm.png`
- **Classification Report**:

```
              precision    recall  f1-score   support
           0     0.9721    0.9939    0.9828       980
           1     0.9903    0.9938    0.9921      1135
           2     0.9911    0.9671    0.9789      1032
           3     0.9755    0.9851    0.9803      1010
           4     0.9836    0.9796    0.9816       982
           5     0.9931    0.9652    0.9790       892
           6     0.9833    0.9823    0.9828       958
           7     0.9730    0.9825    0.9777      1028
           8     0.9648    0.9836    0.9741       974
           9     0.9810    0.9713    0.9761      1009
    accuracy                         0.9807     10000
```

Sample predictions:

![Sample predictions](reports/figures/samples.png)  
*(Example image, you need to generate it first)*

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Please ensure your code adheres to the existing style and includes appropriate tests.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgments

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow](https://www.tensorflow.org/)
- [MLflow](https://mlflow.org/)

---

## Contact

Your Name – [@your_twitter](https://twitter.com/your_twitter) – email@example.com

Project Link: [https://github.com/yourusername/TF-Keras-MNIST-DA](https://github.com/yourusername/TF-Keras-MNIST-DA)
