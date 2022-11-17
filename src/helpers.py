import numpy as np
from typing import Union
from itertools import combinations

def argmax_perm(vector : np.ndarray, perm : np.ndarray, start : int) -> int:
    """
    Find maximum argument of permuted vector from an arbitrary starting point.

        idx_max = argmax{vector[perm[j]], j = start, start+1, ..., n-1}

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
    """
    # Find maximum index in permuted vector
    idx_max_perm = np.argmax(vector[perm[start:]])

    # Find index in permutation vector corresponding to maximum
    idx_max = np.where(perm == perm[start + idx_max_perm])

    # Convert index to integer
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
    Compute the volume of a square matrix.

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

def max_volume_index_set(matrix : np.ndarray, k : int) -> np.ndarray:
    """
    Get index set of largest volume square submatrix of size k.

        J = argmax{vol(matrix[I, I]), |I| = k}

    Warning
    -------
    This function is of complexity O(c(k, n)). Don't use it for n >> k >> 1.

    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        Square input matrix.
    k : int
        Size of submatrix.

    Returns
    -------
    I : np.ndarray
        Index set of largest volume submatrix.
    """
    max_vol = -np.inf

    # Iterate through all possible index sets of size k
    for I in combinations(range(len(matrix)), k):

        # Memorize index set of maximum volume submatrix
        if (vol := volume(matrix[np.ix_(I, I)])) > max_vol:
            max_vol = vol
            max_I = I
    # Convert index set to a numpy array
    J = np.array(max_I)
    return J

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


def ACA_error(matrix : np.ndarray, I : np.ndarray, ord : Union[str, int] = np.inf) -> float:
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
    error : float
        Error of the cross approximation of the input matrix.
    """
    if ord == 'max':
        error = np.max(np.abs(matrix - ACA_approximation(matrix, I)))
    else:
        error = np.linalg.norm(matrix - ACA_approximation(matrix, I), ord=ord)
    return error

def ACA_upper_bounds(matrix : np.ndarray, k_max : int) -> np.ndarray:
    """
    Compute the theoretical upper bounds on the ACA for k = 1, ..., k_max.

        upper_bounds(k) = (k + 1)*sigma_{k+1}(matrix)

    Parameters
    ----------
    matrix : np.ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    k_max : int
        Cardinality of the input matrix.
    
    Returns
    -------
    upper_bounds : np.ndarray
        Theoretical upper bounds on the ACA for k = 1, ..., k_max.
    singular_values : np.ndarray
        The first k_max singular values of the matrix.
    """
    singular_values = np.linalg.svd(matrix, compute_uv = False)
    upper_bounds = (np.arange(1,k_max+1) + 1)*singular_values[1:(k_max+1)]
    return upper_bounds, singular_values[:k_max]

