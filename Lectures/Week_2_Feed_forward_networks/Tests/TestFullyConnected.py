import unittest
import numpy as np
import torch

from scipy import stats

class TestFullyConnected(unittest.TestCase):
    FullyConnected = None

    def setUp(self):
        self.weights = torch.arange(18).reshape(3, 6).float()
        self.bias = torch.arange(6).float()
        self.input_tensor = torch.arange(9).reshape(3, 3).float() - 4
        self.output_tensor = torch.tensor([[-42., -50., -58., -66., -74., -82.],
        [ 12.,  13.,  14.,  15.,  16.,  17.],
        [ 66.,  76.,  86.,  96., 106., 116.]])


    def test_forward_size(self):
        layer = self.FullyConnected(3, 6)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], 6)
        self.assertEqual(output_tensor.shape[0], 3)

    def test_forward(self):
        layer = self.FullyConnected(3, 6)
        layer.weights = self.weights
        layer.bias = self.bias
        output_tensor = layer.forward(self.input_tensor).numpy()

        expected_tensor = self.output_tensor.numpy()
        self.assertAlmostEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

    def test_initialization_weight(self):
        layer = self.FullyConnected(1000, 1000)

        scale = 0.02
        p_value = stats.kstest(layer.weights.numpy().flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01)

    def test_initialization_bias(self):
        layer = self.FullyConnected(1, 1000)
        bias = layer.bias.numpy()
        self.assertAlmostEqual(np.sum(np.power(bias, 2)), 0)

if __name__ == "__main__":
    pass