"""
This file contains all utility functions that are used by the factorization algorithms.
"""

import numpy as np
import torch
import tensorly as tl



def defactorizing_CP(A_js, X_shape):
    """
    This function calculates the tensor X from the A_js
    
    Args:
      A_js: list of N matrizes that factorize X
      X_shape: the shape of X
    
    Returns:
      X 
    """
    # calculated X_(0) (mode 0 matricization) using the formula from Kolda and then folds the tensor back up
    X_unfolded_0 = tl.matmul(A_js[0], tl.transpose(tl.tenalg.khatri_rao(A_js, skip_matrix=0)) )
    return tl.fold(X_unfolded_0, 0, X_shape)


def is_tensor_not_finite(tensor):
    """
    Checks if the tensor contains some non-finite elements, i.e. not inf or nan
    Returns true if it finds any non-finite elements and returns false if all elements are finite
    can use both numpy and pytorch tensor

    Is using in the tensor_factorization_cp_poisson function inside the step size calculation to make sure the gradient is finite
    """
    if tl.get_backend() == 'pytorch':
        return torch.any(~torch.isfinite(tensor))
    if tl.get_backend() == 'numpy':
        return np.any(~np.isfinite(tensor))
    else: # default to using numpy by first casting the array to numpy
        return np.any(~np.isfinite(np.array(tensor)))
    

def poisson_error(X_unfolded_n, M_unfolded_n):
    """
    Calculates the Poisson error given a tensor X and an approximation M. Using both as unfolded n
    
    Args:
        X_unfolded_n: The tensor X unfolded at dimension n
        M_unfolded_n: The approximation M unfolded at dimension n
    """
    return tl.sum( M_unfolded_n - X_unfolded_n * tl.log(M_unfolded_n))


def create_initial_data(X, F):
    """
    Create random initial A_ns for our algorithms
    """
    N = X.ndim # get dimension of X
    X_shape = X.shape
    # initialize A_j with random positive values if it was not given
    A_ns = []
    for i in range(N):
        # we use random.random_tensor as it returns a tensor
        A_ns.append(tl.random.random_tensor((X_shape[i], F), **tl.context(X)))
       
    # rescale
    # TODO figure out if we want to scale with max or norm
    norm_approx = tl.norm( defactorizing_CP(A_ns, X.shape) )
    scaling = (tl.norm(X) / norm_approx) ** (1.0/ X.ndim)
    for n in range(len(A_ns)):
        A_ns[n] = A_ns[n] * scaling
    return A_ns


def random_cp_with_noise(dimensions, F, noise_scaling=0.1, context={}):
    """
    Create a random CP tensor with some added noise
    """
    true_solution = tl.random.random_cp(dimensions, F, full=True, **context)
    noise = tl.random.random_tensor(dimensions, **context) * noise_scaling * tl.max(true_solution)
    X = true_solution + noise
    return X