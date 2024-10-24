import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Implement the ReLU activation function.
    Rectified Linear Units, or ReLUs, are a type of activation function that
    are linear in the positive dimension, but zero in the negative dimension.
    The kink in the function is the source of the non-linearity.

    The linearity in the positive dimension prevents the saturation of the
    function, which is a problem with other activation functions like sigmoid.
    This prevents vanishing gradients, which is a problem with training deep
    neural networks.
    Although, it is possible for ReLU to saturate in the negative dimension.

    Args:
        x (np.ndarrray): Input array.

    Returns:
        np.ndarray: Output array.
    """
    return np.maximum(0, x)

def lrelu(alpha: float = 0.01):
    """Implement the Leaky ReLU activation function.
    Leaky ReLUs are a type of activation function that are similar to ReLUs,
    but have a small slope in the negative direction. This small slope is
    typically set to a small value like 0.01.

    The small slope prevents the dying ReLU problem, which is when the ReLU
    units always output zero. This can happen when the input to the ReLU is
    always negative, causing the gradient to be zero.

    Args:
        x (np.ndarray): Input array.
        alpha (float): Slope in the negative direction.

    Returns:
        np.ndarray: Output array.
    """
    def lrelu(x: np.ndarray) -> np.ndarray:
        """Implement the Leaky ReLU activation function.
        Leaky ReLUs are a type of activation function that are similar to ReLUs,
        but have a small slope in the negative direction. This small slope is
        typically set to a small value like 0.01.

        The small slope prevents the dying ReLU problem, which is when the ReLU
        units always output zero. This can happen when the input to the ReLU is
        always negative, causing the gradient to be zero. This is useful with GANs
        where gradients are sparse.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array.
        """
        return np.maximum(alpha*x, x)
    # return lambda x: np.maximum(alpha*x, x)
    return lrelu


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Implement the sigmoid activation function.
    The sigmoid function is a type of activation function that is an S-shaped
    curve. It is defined as:
    f(x) = 1 / (1 + exp(-x))

    The sigmoid function is used in binary classification problems, where the
    output is a probability between 0 and 1. It squashes the output between 0
    and 1, which can be interpreted as a probability.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array.
    """
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.array([-1, -2, -3])
    l_relu = lrelu(alpha=0.01)
    print(l_relu(x))