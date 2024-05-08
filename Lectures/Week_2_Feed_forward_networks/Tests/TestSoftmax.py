import unittest
import torch
import numpy as np

class TestSoftmax(unittest.TestCase):
    Softmax = None

    def setUp(self):
        self.input_tensor = torch.tensor([[-4, -3, -2, -1, 0],
                                                [1, 2, 4, 5, 6]], dtype=torch.float)


    def test_forward(self):
        expected_tensor = torch.softmax(self.input_tensor, dim=1)
        layer = self.Softmax()
        output_tensor = layer.forward(self.input_tensor).numpy()
        expected_tensor = expected_tensor.numpy()
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = self.Softmax()
        output_tensor = layer.forward(self.input_tensor*5)
        output_tensor = output_tensor.numpy()
        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1.)
        self.assertGreaterEqual(out_min, 0.)


if __name__ == "__main__":
    pass