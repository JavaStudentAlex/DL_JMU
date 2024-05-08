import unittest
import torch
import numpy as np

class TestStableSoftmax(unittest.TestCase):
    StableSoftmax = None

    def setUp(self):
        self.input_tensor = torch.tensor([[-4, -3, -2, -1, 0],
                                                [1, 2, 4, 5, 6]], dtype=torch.float)


    def test_forward(self):
        expected_tensor = torch.softmax(self.input_tensor, dim=1)
        layer = self.StableSoftmax()
        output_tensor = layer.forward(self.input_tensor).numpy()
        expected_tensor = expected_tensor.numpy()
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = self.StableSoftmax()
        output_tensor = layer.forward(self.input_tensor*5)
        output_tensor = output_tensor.numpy()
        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1.)
        self.assertGreaterEqual(out_min, 0.)

    def test_nonan(self):
        layer = self.StableSoftmax()
        output_tensor = layer.forward(torch.tensor([[1000, 1, 1000]]))
        self.assertFalse(torch.isnan(output_tensor).any())
        output_tensor = layer.forward(torch.tensor([[0, 1, -100]]))
        self.assertFalse(torch.isnan(output_tensor).any())
        output_tensor = layer.forward(torch.tensor([[1000, 1, -10000]]))
        self.assertFalse(torch.isnan(output_tensor).any())
        output_tensor = layer.forward(torch.tensor([[0, -55000, 551000]]))
        self.assertFalse(torch.isnan(output_tensor).any())


if __name__ == "__main__":
    pass