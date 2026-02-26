import unittest
from src.data.preprocessing import load_mnist, preprocess

class TestData(unittest.TestCase):
    def test_preprocess_shape(self):
        (x_train, y_train), _ = load_mnist()
        x, y = preprocess(x_train, y_train)
        self.assertEqual(x.shape[1], 784)
        self.assertEqual(y.shape[1], 10)

if __name__ == '__main__':
    unittest.main()