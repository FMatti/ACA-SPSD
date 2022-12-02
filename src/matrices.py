import numpy as np

def get_A1(n, theta=0.3):
    """
    Obtain the matrix A_1 defined on the project manual.

    Parameters
    ----------
    n : int, n > 0
        Size of the matrix A_1
    theta : float, optional (default is 0.3)
        Angular parameter used in generating the matrix.

    Returns
    -------
    A1 : np.ndarray, shape (n, n)
        The generated matrix.
    """
    L = -np.tril(np.ones((n, n)), k=-1) * np.cos(theta) + np.eye(n)
    d = np.sin(theta)**np.arange(0, 2*n, 2)
    A1 = L @ np.diag(d) @ L.T
    return A1

def get_A2(n, seed=0):
    """
    Obtain the matrix A_2 defined on the project manual.

    Parameters
    ----------
    n : int, n > 0
        Size of the matrix A_1
    seed : int, optional (default is 0)
        The seed used to generate random matrix.

    Returns
    -------
    A2 : np.ndarray, shape (n, n)
        The generated matrix.
    """
    np.random.seed(seed)
    d = np.arange(1, n+1, dtype=float)**-2
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    A2 = Q @ np.diag(d) @ Q.T
    return A2

def get_A3(n):
    """
    Obtain the matrix A_3 defined on the project manual.

    Parameters
    ----------
    n : int, n > 0
        Size of the matrix A_1

    Returns
    -------
    A3 : np.ndarray, shape (n, n)
        The generated matrix.
    """
    A3 = 1 / (np.add.outer(np.arange(1, n+1), np.arange(1, n+1)) - 1)
    return A3

def get_A4(epsilon=1e-1):
    """
    Obtain the matrix A_4 used as a counter example for Task 4.

    Parameters
    ----------
    epsilon : float, optional, default is 1e-1
        Parameter used to control "non-orthonormality" of first submatrix.

    Returns
    -------
    A4 : np.ndarray, shape (n, n)
        The generated matrix.
    """
    A4 = np.array([[1, epsilon, epsilon, epsilon],
                   [epsilon, 1, epsilon, epsilon],
                   [epsilon, epsilon, 1, 0],
                   [epsilon, epsilon, 0, 1]])
    return A4