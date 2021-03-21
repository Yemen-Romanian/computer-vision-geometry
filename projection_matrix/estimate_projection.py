import numpy as np
from scipy.linalg import null_space
from itertools import combinations


def estimate_Q(U, X, ix):
    """Image projection matrix estimation (up to a non-zero scalar).
       The algorithm uses 6 image-world point correspondences for estimation.

    Args:
        U (numpy ndarray, n x 3): Matrix of homogeneous image coordinates
        X (numpy ndarray, n x 4): Matrix of correspondent points in the world, 
        also in homogeneous coordinates 
        ix (numpy ndarray m): Indices of points to choose from for estimation

    Returns:
        (best_Q, min_error, best_points): Returns best projection matrix estimation in terms
        of reprojection error, the min error and best points themselves.
    """
    
    u, x = U[ix], X[ix]
    min_error = np.inf
    for indices in combinations(np.arange(10), 6):

        indices = list(indices)
        sel_u, sel_x = u[indices], x[indices]
        coef_mat = build_coef_mat(sel_u, sel_x)
        Q = null_space(coef_mat).reshape((3, 4))
        error = compute_reprojection_error(Q, U, X)

        if error < min_error:
            best_Q = Q
            min_error = error
            best_points = indices

    return best_Q, min_error, best_points

def build_cross_mat(U):
    u, v, w = U
    return np.array([[1, 0, -u],
                     [0, 1, -v]])

def build_coef_mat(U, X):
    coef_mat = np.array([]).reshape(0, 12)
    for u, x in zip(U, X):
        cross_mat = build_cross_mat(u)
        coef_mat = np.vstack([coef_mat, np.kron(cross_mat, x.T)])

    return coef_mat[:-1]


def to_homogeneous(A):
    result = np.ones((A.shape[0], A.shape[1] + 1))
    result[:, :-1] = A
    return result


def compute_reprojection_error(Q, U, X):
    _U = Q.dot(X.T).T
    _U[:, 0] /= _U[:, 2]
    _U[:, 1] /= _U[:, 2]
    return np.linalg.norm(_U - U)