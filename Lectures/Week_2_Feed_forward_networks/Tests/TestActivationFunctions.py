import unittest
import torch
import numpy as np

class TestSigmoid(unittest.TestCase):
    Sigmoid = None

    def setUp(self):
        self.input_tensor = torch.tensor([[-4, -3, -2, -1, 0],
                                                [1, 2, 4, 5, 6]], dtype=torch.float)
        self.output_tensor = torch.Tensor([[0.0180, 0.0474, 0.1192, 0.2689, 0.5000],
                                            [0.7311, 0.8808, 0.9820, 0.9933, 0.9975]])


    def test_forward(self):
        expected_tensor = self.output_tensor
        layer = self.Sigmoid()
        output_tensor = layer.forward(self.input_tensor).numpy()
        expected_tensor = expected_tensor.numpy()
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_range(self):
        layer = self.Sigmoid()
        output_tensor = layer.forward(self.input_tensor*10)
        output_tensor = output_tensor.numpy()
        out_max = np.max(output_tensor)
        out_min = np.min(output_tensor)

        self.assertLessEqual(out_max, 1.)
        self.assertGreaterEqual(out_min, 0.)


class TestSiLU(unittest.TestCase):
    SiLU = None

    def setUp(self):
        self.input_tensor = torch.tensor([[-4, -3, -2, -1, 0],
                                                [1, 2, 4, 5, 6]], dtype=torch.float)
        self.output_tensor = torch.Tensor([[-0.0719, -0.1423, -0.2384, -0.2689,  0.0000],
                                            [ 0.7311,  1.7616,  3.9281,  4.9665,  5.9852]])


    def test_forward(self):
        expected_tensor = self.output_tensor
        layer = self.SiLU()
        output_tensor = layer.forward(self.input_tensor).numpy()
        expected_tensor = expected_tensor.numpy()
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

class TestReLU(unittest.TestCase):
    ReLU = None

    def setUp(self):
        self.input_tensor = torch.tensor([[-4, -3, -2, -1, 0, 1],
                                                [-1, 1, 2, 4, 5, 6]], dtype=torch.float)


    def test_forward(self):
        expected_tensor = torch.relu(self.input_tensor)
        layer = self.ReLU()
        output_tensor = layer.forward(self.input_tensor).numpy()
        expected_tensor = expected_tensor.numpy()
        self.assertAlmostEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

if __name__ == "__main__":
    pass