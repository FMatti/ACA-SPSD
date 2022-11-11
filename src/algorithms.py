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

        # Find argmax{d[pi[j]], j = m, m+1, ..., n-1}
        i = argmax_perm(d, pi, m)

        # Swap entries pi[m] with pi[i]
        swap_entries(pi, m, i)

        # Set entry L[m, pi[m]]
        L[pi[m], m] = pow(d[pi[m]], 0.5)

        # Compute remaining entries in L and update d
        L[pi[m+1:], m] = (A[pi[m], pi[m+1:]] - L[pi[m+1:]] @ L[pi[m]]) / L[pi[m], m]
        d[pi[m+1:]] -= L[pi[m+1:], m]**2

    # Selected index set is first k entries (pivots) in index permutation vector
    I = pi[:k]
    return I

def Algorithm1(A : np.ndarray, k : int) -> np.ndarray:
    raise NotImplementedError