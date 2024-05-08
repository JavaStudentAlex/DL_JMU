import unittest
import numpy as np
import torch


class TestSiGLUBlock(unittest.TestCase):
    SiGLUBlock = None

    def setUp(self):
        self.weights_in = torch.arange(18).reshape(3, 6).float() * 1 - 7.77
        self.weights_out = torch.arange(18).reshape(6, 3).float() * 1 - 6.9
        self.weights_gate = torch.arange(18).reshape(3, 6).float() * 1 - 9.6
        self.input_tensor = torch.arange(12).reshape(4, 3).float() - 6
        self.output_tensor = torch.tensor([[-23455.6504, -19666.2910, -15876.9336],
        [ -7365.3359,  -5884.2368,  -4403.1382],
        [  4159.3184,   5055.0024,   5950.6860],
        [ 15429.8564,  17670.2324,  19910.6074]])

    def test_forward_size(self):
        block = self.SiGLUBlock(3, 6, 3)
        output_tensor = block.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], 3)
        self.assertEqual(output_tensor.shape[0], 4)

    def test_forward(self):
        block = self.SiGLUBlock(3, 6, 3)
        block.fully_connected_in.weights = self.weights_in
        block.fully_connected_out.weights = self.weights_out
        block.fully_connected_gate.weights = self.weights_gate
        output_tensor = block.forward(self.input_tensor).numpy()

        expected_tensor = self.output_tensor.numpy()
        self.assertAlmostEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)


if __name__ == "__main__":
    pass