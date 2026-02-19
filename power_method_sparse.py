import numpy as np
from scipy.sparse import csr_matrix, issparse

def power_method_sparse(P, tol=1e-30, maxit=10000):

    n = P.shape[0]
    pi = np.full(n, 1.0 / n)


    if issparse(P) and not isinstance(P, csr_matrix):
        P = P.tocsr()

    for _ in range(maxit):

        new = pi @ P


        if np.linalg.norm(new - pi, 1) < tol:
            pi = new / new.sum()
            break
        pi = new

    return pi / pi.sum()

