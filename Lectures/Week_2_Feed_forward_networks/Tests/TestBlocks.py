import unittest
import numpy as np
import torch

class TestSiLUBlock(unittest.TestCase):
    SiLUBlock = None

    def setUp(self):
        self.weights_in = torch.arange(18).reshape(3, 6).float() * 1 - 7.77
        self.weights_out = torch.arange(18).reshape(6, 3).float() * 1 - 6.9
        self.input_tensor = torch.arange(12).reshape(4, 3).float() - 6
        self.output_tensor = torch.tensor([[-365.5548, -294.9167, -224.2785],
        [-222.2984, -168.1457, -113.9930],
        [ 208.6437,  293.7747,  378.9058],
        [ 641.6269,  775.2617,  908.8965]])


    def test_forward_size(self):
        block = self.SiLUBlock(3, 6, 3)
        output_tensor = block.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], 3)
        self.assertEqual(output_tensor.shape[0], 4)

    def test_forward(self):
        block = self.SiLUBlock(3, 6, 3)
        block.fully_connected_in.weights = self.weights_in
        block.fully_connected_out.weights = self.weights_out
        output_tensor = block.forward(self.input_tensor).numpy()

        expected_tensor = self.output_tensor.numpy()
        self.assertAlmostEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

if __name__ == "__main__":
    pass