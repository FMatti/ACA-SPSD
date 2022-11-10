import numpy as np
from typing import Union

def argmax_perm(vector : np.ndarray, perm : np.ndarray, start : int) -> int:
    """
    Find maximum argument of permuted vector from an arbitrary starting point.

    Parameters
    ----------
    vector : np.ndarray, shape (n)
        Input vector.
    perm : np.ndarray, shape (n)
        Permutation vector.
    start : int, 0 <= start < n
        Starting index

    Returns
    -------
    idx_max : int
        Maximum index in permuted input vector.
         = argmax{vector[perm[j]], j = start, start+1, ..., n-1}
    """
    idx_max_perm = np.argmax(vector[perm[start:]])
    idx_max = np.where(perm == perm[start + idx_max_perm])
    idx_max = idx_max[0].squeeze()
    return idx_max

def swap_entries(vector : np.ndarray, idx1 : int, idx2 : int) -> None:
    """
    Swap two entries in a vector.

    Parameters
    ----------
    vector : np.ndarray, shape (n)
        Input vector.
    idx1 : int, 0 <= idx1 < n
        Index of first entry.
    idx2 : int, 0 <= idx2 < n
        Index of second entry.
    """
    tmp = vector[idx1]
    vector[idx1] = vector[idx2]
    vector[idx2] = tmp

def volume(matrix : np.ndarray) -> float:
    """
    Swap two entries in a vector.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        Square input matrix.

    Returns
    -------
    vol : float
        Volume of the input matrix.
    """
    vol = abs(np.linalg.det(matrix))
    return vol

def ACA_approximation(matrix : np.ndarray, I : np.ndarray):
    """
    Error of cross approximation.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    I : np.ndarray, shape (k)
        Selected index set from cross approximation.

    Returns
    -------
    matrix_approximation : np.ndarray, shape (n, n)
        The approximation of the matrix using cross approximation.
    """
    C = matrix[:, I]
    U_inv = matrix[np.ix_(I, I)]
    R = matrix[I, :]
    matrix_approximation = C @ np.linalg.solve(U_inv,  R)
    return matrix_approximation


def ACA_error(matrix : np.ndarray, I : np.ndarray, ord : Union[str, int] = 'fro') -> float:
    """
    Error of cross approximation.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    I : np.ndarray, shape (k)
        Selected index set from cross approximation.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm. 

    Returns
    -------
    vol : float
        Volume of the input matrix.
    """
    norm = np.linalg.norm(matrix - ACA_approximation(matrix, I), ord=ord)
    return norm