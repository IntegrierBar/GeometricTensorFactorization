"""
This python file contains the algorithm for Poisson Family special case
"""


import time
import tensorly as tl
import numpy as np
import math
from .multiplicative import defactorizing_CP


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

    # lets save the different m as well so that we can run some analysis there
    step_size_modifiers = []
    for i in range(N):
        step_size_modifiers.append([])
    for _ in range(max_iter):
        for n in range(N):
            start = time.time()
            
            khatri_rao_product = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            approximated_X_unfolded_n = tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)) # TODO consider if we want to use the same for all matrizes or update along the way
            
            ###### Step size calculation ######
            # TODO these values should be looked at more to determine which are best
            sigma = 0.5
            beta = 0.5
            alpha = 0.5
            m = 0 # TODO need to find some estimation for first m! otherwise need to compute too much
            # for now, just use 0.7 of what was used last time
            if len(step_size_modifiers[n]) > 0:
                m = int(step_size_modifiers[n][-1] * 0.7)
            step_size = math.pow(beta, m) * alpha
            f = lambda A: tl.sum( tl.matmul(A, tl.transpose(khatri_rao_product)) - tl.base.unfold(X, n) * tl.log( tl.matmul(A, tl.transpose(khatri_rao_product)) )) 
            function_value_at_iteration = tl.sum(approximated_X_unfolded_n - tl.base.unfold(X, n) * tl.log(approximated_X_unfolded_n)) 
            gradient_at_iteration = tl.matmul(tl.ones(approximated_X_unfolded_n.shape, **tl.context(X)) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )
            riemanndian_gradient_at_iteration = A_ns[n] * gradient_at_iteration # The "A_ns[n] *" is the inverse of the Riemannien metric tensor matrix thing
            norm_of_rg = tl.sum(gradient_at_iteration * riemanndian_gradient_at_iteration) # TODO maybe check if this is correct!
            next_iterate =  A_ns[n] * tl.exp(-step_size * riemanndian_gradient_at_iteration)
            # TODO get rid of just using numpy for checking for infinity, to also make use of pytorch as well.
            while is_tensor_not_finite(next_iterate) or ( function_value_at_iteration - sigma * step_size * norm_of_rg < f(next_iterate) ):
                m += 1
                step_size = math.pow(beta, m) * alpha
                next_iterate =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
            
            step_size_modifiers[n].append(m) # save the value of m for later inspection
            ###### update A_ns[n]
            A_ns[n] = next_iterate
            # OLD CODE, kept for safety
            # regular * does componentwise multiplication
            #A_ns[n] = A_ns[n] * tl.exp(-step_size * tl.matmul(tl.ones(approximated_X_unfolded_n.shape) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )  ) 
            
            end = time.time()
            if verbose:
                print("Current index: " + str(n))
                print("Calculculation time: " + str(end - start))
                print("New objective function value: " + str(f(A_ns[n])))
                print("step size was: " + str(step_size) + " with m = " + str(m))

                print("function_value_at_iteration = " + str(function_value_at_iteration))
                print("norm_of_rg = " + str(norm_of_rg))
                print("gradiend_at_iteration = ")
                print(gradient_at_iteration)
                print("riemannian_gradient_at_iteration = ")
                print(riemanndian_gradient_at_iteration)

                print("new A_ns[n]:")
                print(A_ns[n])
                
                
            
        # the reconstruction error
        approximated_X = defactorizing_CP(A_ns, X_shape)
        RE.append(tl.norm(X-approximated_X)/norm_X)

        if verbose:
            print("current apporximation error is: " + str(RE[-1]))
            #print("approximation is:")
            #print(approximated_X)
        
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
        return A_ns, RE, approximated_X, step_size_modifiers
    return A_ns