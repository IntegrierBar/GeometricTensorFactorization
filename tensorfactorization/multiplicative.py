"""
This python file contains all code for multiplicative non-negative tensor factorization.
The main function you want to use is 'multiplicative_tesnor_factorization'. It calculates the tensor factoriazation.
To get the original tensor back use 'defactorizing_PTF'
"""


import time
import tensorly as tl
import numpy as np
import math

from copy import deepcopy

# This version is deprecated as the one below is twice as fast
def defactorizing_CP_old(A_js, F, N, X_shape):
    """
    This function calculates the tensor X from the A_js
    
    Args:
      A_js: list of N matrizes that factorize X
      F: The order of the apporximation
      N: the dimension of X
      X_shape: the shape of X
    
    Returns:
      X 
    """
    
    X = tl.zeros(X_shape)
    for a in range(F):
        tensors = []
        for j in range(N):
            tensors.append(A_js[j][:, a])
        X += tl.tenalg.outer(tensors)

    return X


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


# this does the same as tl.tenalg.khatri_rao(A_js, skip_matrix=j) and is thus now deprecated
def multiply_varying_matrices(A_list):
    """
    This function multiplies matrices in a list with different first dimensions.
    
    Args:
      A_list: A list of numpy arrays representing the matrices A^(j).
    
    Returns:
      A numpy array representing the resulting matrix M.
    """
    a = A_list[0].shape[1]  # Assuming all matrices have the same second dimension 'a'
    total_rows = np.prod([matrix.shape[0] for matrix in A_list])  # Total rows in M

    # Create an index mapping for efficient multiplication
    all_shapes = [matrix.shape[0] for matrix in A_list]
    index_map = np.ndindex(*all_shapes)  # More efficient index generation
    # Iterate through each element in the resulting matrix and perform multiplication
    M = np.ones((total_rows, a))
    for i, indices in enumerate(index_map):
        #print(indices)
        for j, index in enumerate(indices):
            M[i] *= A_list[j][index]
    return M



def tensor_factorization_cp_multiplicative(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, initial_A_ns=None):
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
      error: stops iteration if difference between X and approximation with decomposition changes less then this
      max_iter: maximum number of iterations
      detailed: if false, function returns only G and the As. if true returns also all errors found during calculation 
      verbose: If True, prints additional information
    
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







def convert_diag_tucker_to_cp(G, A_ns):
    """convert a diagonal tucker decomposition to a CP decomposition by getting the diagonal matrix D and multiplying it with A_0

    Args:
        G (tensor): The diagonal core. Has to be diagonal to work.
        A_ns (list of matrices): The factors. Have to be mutable!

    Returns:
        A_ns : The factors are now a CP decomposition
    """
    A_ns[0] = tl.matmul(A_ns[0], tl.diag(get_diagonal(G)))
    return A_ns
    
    


def get_diagonal(arr):
  """Extracts the diagonal elements from an n-dimensional array.

  Args:
    arr: The input n-dimensional NumPy array.

  Returns:
    A 1D NumPy array containing the diagonal elements.
  """

  # Reduce the array to a 2D array by summing over all but the first two axes
  reduced_arr = np.sum(arr, axis=tuple(range(2, arr.ndim)))

  # Extract the diagonal of the reduced 2D array
  return np.diagonal(reduced_arr)