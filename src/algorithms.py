import numpy as np

from src.helpers import argmax_perm, swap_entries
from src.helpers import volume

def ACA_SPSD(A : np.ndarray, k : int) -> np.ndarray:
    """
    Adaptive cross approximation for symetric positive semidefinite matrices.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
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
    """
    Attempts to improve the volume of the submatrix selected by ACA_SPSD
    in order to obtain a better cross approximation

     Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Symmetric positive semidefinite matrix.
    k : int, 0 < k <= n
        Number of indices to select.

    Returns
    -------
    I : np.ndarray, shape (k)
        Improved selection of index set.
    """
    n = len(A)
    I = ACA_SPSD(A, k)

    I_bool = np.zeros(n, dtype=bool)
    I_bool[I] = True
    volume_I = volume(A[np.ix_(I_bool, I_bool)])

    swapped = True
    while swapped:
        swapped = False
        J_bool = I_bool.copy()
        for i in np.arange(n)[I_bool]:
            for j in np.arange(n)[~I_bool]:
                # swapping i and j
                J_bool[i] = False
                J_bool[j] = True

                if (new_volume := volume(A[np.ix_(J_bool, J_bool)])) > volume_I:
                    # update indices if swapping i and j improves volume of selected submatrix
                    I_bool = J_bool.copy()
                    volume_I = new_volume
                    swapped = True
                    break
                J_bool = I_bool.copy()
            if swapped:
                break

    I = np.arange(n)[I_bool]
    return I
