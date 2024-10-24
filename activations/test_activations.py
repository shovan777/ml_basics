import unittest
import numpy as np
from activations import relu
from activations import sigmoid
from activations import lrelu


class TestActivations(unittest.TestCase):
    """Test the activation functions."""

    def test_relu_positive_values(self):
        """Test ReLU with positive values."""
        x = np.array([1, 2, 3])
        expected_output = np.array([1, 2, 3])
        np.testing.assert_array_equal(relu(x), expected_output)

    def test_relu_negative_values(self):
        """Test ReLU with negative values."""
        x = np.array([-1, -2, -3])
        expected_output = np.array([0, 0, 0])
        np.testing.assert_array_equal(relu(x), expected_output)

    def test_relu_mixed_values(self):
        """Test ReLU with mixed values."""
        x = np.array([-1, 2, -3, 4])
        expected_output = np.array([0, 2, 0, 4])
        np.testing.assert_array_equal(relu(x), expected_output)

    def test_relu_zero_values(self):
        """Test ReLU with zero values."""
        x = np.array([0, 0, 0])
        expected_output = np.array([0, 0, 0])
        np.testing.assert_array_equal(relu(x), expected_output)

    def test_sigmoid_positive_values(self):
        """Test sigmoid with positive values."""
        x = np.array([1, 2, 3])
        expected_output = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(sigmoid(x), expected_output)

    def test_sigmoid_negative_values(self):
        """Test sigmoid with negative values."""
        x = np.array([-1, -2, -3])
        expected_output = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(sigmoid(x), expected_output)

    def test_sigmoid_mixed_values(self):
        """Test sigmoid with mixed values."""
        x = np.array([-1, 2, -3, 4])
        expected_output = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(sigmoid(x), expected_output)

    def test_sigmoid_zero_values(self):
        """Test sigmoid with zero values."""
        x = np.array([0, 0, 0])
        expected_output = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(sigmoid(x), expected_output)

    def test_sigmoid_values_between_zero_and_one(self):
        "Test sigmoid is bounded within the range (0, 1)."
        x = np.array([1, -2, 3])
        output = sigmoid(x)
        self.assertTrue(np.all((output >= 0) & (output <= 1)))

    def test_sigmoid_values_at_extremes(self):
        "Test sigmoid at the extremes."
        x = np.array([np.inf, -np.inf])
        output = sigmoid(x)
        expected_output = np.array([1, 0])
        np.testing.assert_array_almost_equal(output, expected_output)
        # self.assertTrue(np.allclose(output, [1, 0])) # this works too

    def test_lrelu_negative_values(self):
        x = np.array([-1, -2, -3])
        l_relu = lrelu(alpha=0.01) 
        output = l_relu(x)
        expected_output = np.array([-0.01, -0.02, -0.03])
        np.testing.assert_array_almost_equal(output, expected_output)

if __name__ == "__main__":
    unittest.main()
