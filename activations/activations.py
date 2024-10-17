import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """Discard negative values in the input array.

    Args:
        x (np.ndarrray): Input array.

    Returns:
        np.ndarray: Output array with negative values set to zero.
    """    
    return np.maximum(0, x)