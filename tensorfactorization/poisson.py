"""
This python file contains the algorithm for Poisson Family special case
"""


import time
import tensorly as tl
import numpy as np
import math


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



def tensor_factorization_cp_poisson(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False):
    """
    This function uses a multiplicative method to calculate a nonnegative tensor decomposition
    
    Args:
      X: The tensor of dimension N we want to decompose. X \in \RR^{I_1 x ... x I_N}
      F: The order of the apporximation
      error: stops iteration if difference between X and approximation with decomposition changes less then this
      max_iter: maximum number of iterations
      detailed: if false, function returns only G and the As. if true returns also all errors found during calculation 
      verbose: If True, prints additional information
    
    Returns:
      A list of tensors approximating X 
    """
    
    N = X.ndim # get dimension of X
    X_shape = X.shape
    norm_X = tl.norm(X)
    # initialize A_j with random positive values
    A_ns = []
    for i in range(N):
        # we use random.random_tensor as it returns a tensor
        A_ns.append(tl.random.random_tensor((X_shape[i], F), **tl.context(X)))
    
    # the reconstruction error
    approximated_X = defactorizing_CP(A_ns, X_shape)
        
    RE = [tl.norm(X-approximated_X)/norm_X]
    for _ in range(max_iter):
        for n in range(N):
            start = time.time()
            
            khatri_rao_product = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            approximated_X_unfolded_n = approximated_X #tl.matmul(A_ns[n], tl.transpose(khatri_rao_product))
            
            # regular * does componentwise multiplication
            step_size = 1 # TODO add step size calculation
            A_ns[n] = A_ns[n] * np.exp(-step_size * tl.matmul( tl.base.unfold(X, n) / approximated_X_unfolded_n , khatri_rao_product )  ) 
            
            end = time.time()
            if verbose:
                print("Current index: " + str(n))
                print("Calculculation time: " + str(end - start))
                
        # the reconstruction error
        approximated_X = defactorizing_CP(A_ns, X_shape)
        RE.append(tl.norm(X-approximated_X)/norm_X)
        
        # check if we have converged
        if abs(RE[-1] - RE[-2]) < error:
            break

    # TODO I think we can skip this for now
    """
    # Rescale the A_ns
    # TODO there should be a smarter way of calculating K
    K = []
    for n in range(N):
        K_j = []
        for a in range(F):
            sum = 0
            for i in range(X_shape[n]):
                sum += A_ns[n][i, a]**2
            K_j.append(math.sqrt(sum))
        K.append(K_j)
    K = tl.tensor(K, **tl.context(X))
    for n in range(N):
        for a in range(F):
            A_ns[n][:, a] = A_ns[n][:, a] * math.pow(tl.prod(K[:, a]), 1.0/N) / K[n, a]
    """

    if detailed:
        return A_ns, RE, approximated_X
    return A_ns