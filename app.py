from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model('models/mnist_mlp.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img = np.array(data['image']).reshape(1, 784).astype('float32') / 255.0
    pred = model.predict(img)
    return jsonify({'prediction': int(np.argmax(pred))})

if __name__ == '__main__':
    app.run()