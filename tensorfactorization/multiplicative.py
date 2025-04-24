"""
This python file contains all code for multiplicative non-negative tensor factorization.
The main function you want to use is 'multiplicative_tesnor_factorization'. It calculates the tensor factoriazation.
To get the original tensor back use 'defactorizing_CP' from utils.
"""


import time
import tensorly as tl
import math

from copy import deepcopy

from .utils import defactorizing_CP



def tensor_factorization_cp_multiplicative(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, initial_A_ns=None):
    """
    This function uses a multiplicative method to calculate a nonnegative tensor decomposition
    
    Args:
      X: The tensor of dimension N we want to decompose. X \in \RR^{I_1 x ... x I_N}
      F: The order of the apporximation
      error: stops iteration if normed difference between X and approximation changes less then this number
      max_iter: maximum number of iterations
      detailed: if false, function returns only G and the As. if true returns also all errors found during calculation and final approximation
      verbose: If True, prints additional information
      initial_A_ns: List of initial A_ns has to be of length X.ndim and each element has to have the correct shape (X_shape[i], F) and the same context as X
    
    Returns:
      A_ns: A list of matrizes approximating X 
      RE (optional): list of all errors during optimization. Uses quadratic/Gauss error instead of poisson error currently
      approximated_X (optional): final approximation of X
    """
    
    N = X.ndim # get dimension of X
    X_shape = X.shape
    norm_X = tl.norm(X)
    # initialize A_j with random positive values
    # initialize A_j with random positive values if it was not given
    # TODO could consider adding checks if initial A_ns are given
    if initial_A_ns is None:
        A_ns = []
        for i in range(N):
            # we use random.random_tensor as it returns a tensor
            A_ns.append(tl.random.random_tensor((X_shape[i], F), **tl.context(X)))
    else:
        if len(initial_A_ns) != N:
            raise ValueError("initial A_ns given does not have to correct length")
        for i in range(N):
            if initial_A_ns[i].shape != (X_shape[i], F):
                raise ValueError("inital A_ns with index " + str(i) + " does not have correct dimension. Should be " + str((X_shape[i], F)) + " but is " + str(initial_A_ns[i].shape))
            if tl.context(initial_A_ns[i]) != tl.context(X):
                raise ValueError("inital A_ns with index " + str(i) + " does not have the same context as X. Should be " + str(tl.context(X)) + " but is " + str(tl.context(initial_A_ns[i])))
        A_ns = deepcopy(initial_A_ns) # use copy since that is how we want to later use it for testing
    
    # the reconstruction error
    approximated_X = defactorizing_CP(A_ns, X_shape)
        
    RE = [tl.norm(X-approximated_X)/norm_X]
    for _ in range(max_iter):
        for n in range(N):
            start = time.time()
            Y = tl.base.unfold(X, n)
            
            M = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            

            # regular * does componentwise multiplication
            A_ns[n] = A_ns[n] * tl.matmul(Y, M) / tl.matmul(tl.matmul(A_ns[n], tl.transpose(M)), M)
            
            end = time.time()
            if verbose:
                print("Current index: " + str(n))
                print("total Memory of Y: " + str(Y.size * 8 / 1e6) + "MB")
                print("total Memory of M: " + str(M.size * 8 / 1e6) + "MB")
                print("Calculculation time: " + str(end - start))
                
            
        # the reconstruction error
        approximated_X = defactorizing_CP(A_ns, X_shape)
        RE.append(tl.norm(X-approximated_X)/norm_X)
        
        # check if we have converged
        if abs(RE[-1] - RE[-2]) < error:
            break

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

    if detailed:
        return A_ns, RE, approximated_X
    return A_ns



def tensor_factorization_cp_multiplicative_poisson(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, initial_A_ns=None):
    """
    This function uses a multiplicative method to calculate a nonnegative tensor decomposition by minimizing the poisson error. See paper from Welling and Weber
    
    Args:
      X: The tensor of dimension N we want to decompose. X \in \RR^{I_1 x ... x I_N}
      F: The order of the apporximation
      error: stops iteration if normed difference between X and approximation changes less then this number
      max_iter: maximum number of iterations
      detailed: if false, function returns only G and the As. if true returns also all errors found during calculation and final approximation
      verbose: If True, prints additional information
      initial_A_ns: List of initial A_ns has to be of length X.ndim and each element has to have the correct shape (X_shape[i], F) and the same context as X
    
    Returns:
      A_ns: A list of matrizes approximating X 
      RE (optional): list of all errors during optimization. Uses quadratic/Gauss error instead of poisson error currently
      approximated_X (optional): final approximation of X
    """
    
    N = X.ndim # get dimension of X
    X_shape = X.shape
    norm_X = tl.norm(X)
    # initialize A_j with random positive values if it was not given
    if initial_A_ns is None:
        A_ns = []
        for i in range(N):
            # we use random.random_tensor as it returns a tensor
            A_ns.append(tl.random.random_tensor((X_shape[i], F), **tl.context(X)))
    else:
        if len(initial_A_ns) != N:
            raise ValueError("initial A_ns given does not have to correct length")
        for i in range(N):
            if initial_A_ns[i].shape != (X_shape[i], F):
                raise ValueError("inital A_ns with index " + str(i) + " does not have correct dimension. Should be " + str((X_shape[i], F)) + " but is " + str(initial_A_ns[i].shape))
            if tl.context(initial_A_ns[i]) != tl.context(X):
                raise ValueError("inital A_ns with index " + str(i) + " does not have the same context as X. Should be " + str(tl.context(X)) + " but is " + str(tl.context(initial_A_ns[i])))
        A_ns = deepcopy(initial_A_ns) # use copy since that is how we want to later use it for testing
    
    # the reconstruction error
    approximated_X = defactorizing_CP(A_ns, X_shape)
        
    RE = [tl.norm(X-approximated_X)/norm_X]
    for _ in range(max_iter):
        for n in range(N):
            start = time.time()
            Y = tl.base.unfold(X, n)
            
            M = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            

            # regular * does componentwise multiplication
            A_n_M_transposed = tl.matmul(A_ns[n], tl.transpose(M))
            for a in range(F):
                A_ns[n][:,a] = A_ns[n][:,a] / tl.sum(M[:,a]) * tl.dot( Y / A_n_M_transposed , M[:,a])
                
            
            end = time.time()
            if verbose:
                print("Current index: " + str(n))
                print("total Memory of Y: " + str(Y.size * 8 / 1e6) + "MB")
                print("total Memory of M: " + str(M.size * 8 / 1e6) + "MB")
                print("Calculculation time: " + str(end - start))
                
            
        # the reconstruction error
        approximated_X = defactorizing_CP(A_ns, X_shape)
        RE.append(tl.norm(X-approximated_X)/norm_X)
        
        # check if we have converged
        if abs(RE[-1] - RE[-2]) < error:
            break

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

    if detailed:
        return A_ns, RE, approximated_X
    return A_ns
