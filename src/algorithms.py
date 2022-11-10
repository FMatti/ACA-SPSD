import numpy as np

from src.helpers import argmax_perm, swap_entries

def ACA_SPSD(A : np.ndarray, k : int) -> np.ndarray:
    """
    Adaptive cross approximation for symetric positive semidefinite matrices.

    Parameters
    ----------
    A : np.nbdarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    k : int, 0 < k <= n
        Number of indices to select.

    Returns
    -------
    I : np.ndarray, shape (k)
        Selected index set.

    Reference
    ---------
    [1] Harbrecht, H., Peters, M., & Schneider, R. (2012).
        On the low-rank approximation by the pivoted Cholesky decomposition.
        Applied Numerical Mathematics, 62(4), 428-440.
        doi:10.1016/j.apnum.2011.10.001
    """
    n = len(A)
    d = np.diag(A).copy()
    
    # Index permutation vector
    pi = np.arange(n)

    # Matrix holding the factorization matrix L
    L = np.zeros((n, k))

    # Algorithm 1 from [1] on page 431
    for m in range(k):
        i = argmax_perm(d, pi, m)
        swap_entries(pi, m, i)
        L[pi[m], m] = pow(d[pi[m]], 0.5)

        for i in range(m+1, n):
            L[pi[i], m] = (A[pi[m], pi[i]] - L[pi[m]] @ L[pi[i]]) / L[pi[m], m]
            d[pi[i]] -= pow(L[pi[i], m], 2)

    # Selected index set is first k entries in the index permutation vector
    I = pi[:k]
    return I

def Algorithm1(A : np.ndarray, k : int) -> np.ndarray:
    raise NotImplementedError