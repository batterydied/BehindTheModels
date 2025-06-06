import numpy as np

def asserts(X, y):
    assert isinstance(X, np.ndarray), "X must be a NumPy array"
    assert isinstance(y, np.ndarray), "y must be a NumPy array"
    assert X.ndim == 2, "X must be a 2D array"
    assert y.ndim == 1, "y must be a 1D array"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"