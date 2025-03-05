"""
This python file contains the algorithm for Poisson Family special case
"""


import time
import tensorly as tl
import numpy as np
import torch
import math
from .multiplicative import defactorizing_CP

from copy import deepcopy


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



def tensor_factorization_cp_poisson(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, update_approximation_everytime=True, initial_A_ns=None):
    """
    This function uses a multiplicative method to calculate a nonnegative tensor decomposition
    
    Args:
      X: The tensor of dimension N we want to decompose. X \in \RR^{I_1 x ... x I_N}
      F: The order of the apporximation
      error: stops iteration if normed difference between X and approximation changes less then this number
      max_iter: maximum number of iterations
      detailed: if false, function returns only G and the As. if true returns also all errors found during calculation, final approximation and all step size modifiers
      verbose: If True, prints additional information
      update_approximation_everytime: If True, update approximated_X after each time a matrix factor is changed. If false, update approximated_X only after all matrixfactors have been updated
      initial_A_ns: List of initial A_ns has to be of length X.ndim and each element has to have the correct shape (X_shape[i], F) and the same context as X
    
    Returns:
      A_ns: A list of matrizes approximating X 
      RE (optional): list of all errors during optimization. Uses quadratic/Gauss error instead of poisson error currently
      approximated_X (optional): final approximation of X
      step_size_modifiers (optional): list of all step-size-modifiers m used during iteration
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

    # lets save the different m as well so that we can run some analysis there
    step_size_modifiers = []
    for i in range(N):
        step_size_modifiers.append([])
        
    ## MAIN LOOP ##
    for _ in range(max_iter):
        for n in range(N):
            if verbose:
                print("Current index: " + str(n))
                
            start = time.time()
            
            khatri_rao_product = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            
            if update_approximation_everytime:
                approximated_X_unfolded_n = tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)) # use the new approximation using the matrizes we just updated
            else:
                approximated_X_unfolded_n = tl.unfold(approximated_X, n) # use the approximation from the previous iteration step, not using the matrix updates calculated in this iteration
            
            ###### Step size calculation ######
            # TODO these values should be looked at more to determine which are best
            sigma = 0.5
            beta = 0.5
            alpha = 0.5
            m = 0 # TODO need to find some estimation for first m! otherwise need to compute too much, at least add test for finiteness first!
            # for now, just use 0.7 of what was used last time
            if len(step_size_modifiers[n]) > 0:
                m = int(step_size_modifiers[n][-1] * 0.7)
                # TODO seems like we want to make sure that step_size * max(-grad) < 10
                
            step_size = math.pow(beta, m) * alpha # initial step size
            f = lambda A: tl.sum( tl.matmul(A, tl.transpose(khatri_rao_product)) - tl.base.unfold(X, n) * tl.log( tl.matmul(A, tl.transpose(khatri_rao_product)) ))  # lambda for function we actually want to minimize
            function_value_at_iteration = tl.sum(approximated_X_unfolded_n - tl.base.unfold(X, n) * tl.log(approximated_X_unfolded_n)) 
            gradient_at_iteration = tl.matmul(tl.ones(approximated_X_unfolded_n.shape, **tl.context(X)) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )
            riemanndian_gradient_at_iteration = A_ns[n] * gradient_at_iteration # The "A_ns[n] *" is the inverse of the Riemannian metric tensor matrix applied to the gradient, i.e. G(A)^{-1} (\nabla f)
            norm_of_rg = tl.sum(gradient_at_iteration * riemanndian_gradient_at_iteration) # TODO maybe check if this is correct! But it should be since we calculate the Riemmannian norm of the Riemannian gradient as \| grad f \|_g = (G^{-1} \nabla f)^T G G^{-1} \nabla f = \nabla f^T grad f
            next_iterate =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
            if verbose:
                print("Time from start to calculate gradients and first next iterate: " + str(time.time() - start))
            # if Armijo step size condition is not fullfilled, try again with smaller step size. Thanks to math, this is while loop will eventually finish
            while is_tensor_not_finite(next_iterate) or ( function_value_at_iteration - sigma * step_size * norm_of_rg < f(next_iterate) ):
                # TODO: instead of recalculating like this, we can also use (for beta=0.5) that exp(beta * stuff) = [exp(stuff)]^beta and if beta=0.5 this is just sqrt which is 3 times faster
                m += 1
                step_size = math.pow(beta, m) * alpha
                next_iterate =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
            if verbose:
                print("Time from start until end of step size calculation: " + str(time.time() - start))
                print("Biggest element in -gradient: " + str(tl.max(-gradient_at_iteration)))
                print("smallest element in -gradient: " + str(tl.min(-gradient_at_iteration)))
                print("Step Size: " + str(step_size))
                print("m: " + str(m))
                print("Step size * biggest element: " + str(step_size * tl.max(-gradient_at_iteration)))
                print("biggest element in A_n: " + str(tl.max(A_ns[n])))
                print("smallest element in A_n: " + str(tl.min(A_ns[n])))
                print("Shape of approximated_X_unfolded_n: " + str(approximated_X_unfolded_n.shape))
                print("Shape of khatri Rao product: " + str(khatri_rao_product.shape))
            
            step_size_modifiers[n].append(m) # save the value of m for later inspection
            # finally update A_n
            A_ns[n] = next_iterate
            # OLD CODE, kept for safety, but can be deleted
            # regular * does componentwise multiplication
            #A_ns[n] = A_ns[n] * tl.exp(-step_size * tl.matmul(tl.ones(approximated_X_unfolded_n.shape) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )  ) 
            
            end = time.time()
            if verbose:
                print("Calculculation time: " + str(end - start))
                print("New objective function value: " + str(f(A_ns[n])))

                print("function_value_at_iteration = " + str(function_value_at_iteration))
                print("norm_of_rg = " + str(norm_of_rg))
                print("biggest Element of X/M = " + str(tl.max(tl.abs(tl.base.unfold(X, n) / approximated_X_unfolded_n))))
                #print("gradiend_at_iteration = ")
                #print(gradient_at_iteration)
                #print("riemannian_gradient_at_iteration = ")
                #print(riemanndian_gradient_at_iteration)

                #print("new A_ns[n]:")
                #print(A_ns[n])
                print("\n")
                
                
            
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



def tensor_factorization_cp_poisson_fixed_step_size(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, update_approximation_everytime=True, initial_A_ns=None):
    """
    This function uses a multiplicative method to calculate a nonnegative tensor decomposition.
    This time with fixed step size
    
    Args:
      X: The tensor of dimension N we want to decompose. X \in \RR^{I_1 x ... x I_N}
      F: The order of the apporximation
      error: stops iteration if normed difference between X and approximation changes less then this number
      max_iter: maximum number of iterations
      detailed: if false, function returns only G and the As. if true returns also all errors found during calculation, final approximation and all step size modifiers
      verbose: If True, prints additional information
      update_approximation_everytime: If True, update approximated_X after each time a matrix factor is changed. If false, update approximated_X only after all matrixfactors have been updated
      initial_A_ns: List of initial A_ns has to be of length X.ndim and each element has to have the correct shape (X_shape[i], F) and the same context as X
    
    Returns:
      A_ns: A list of matrizes approximating X 
      RE (optional): list of all errors during optimization. Uses quadratic/Gauss error instead of poisson error currently
      approximated_X (optional): final approximation of X
      step_size_modifiers (optional): list of all step-size-modifiers m used during iteration
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

    # lets save the different m as well so that we can run some analysis there
    step_size_modifiers = []
    for i in range(N):
        step_size_modifiers.append([])
        
    ## MAIN LOOP ##
    for _ in range(max_iter):
        for n in range(N):
            if verbose:
                print("Current index: " + str(n))
                
            start = time.time()
            
            khatri_rao_product = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            
            if update_approximation_everytime:
                approximated_X_unfolded_n = tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)) # use the new approximation using the matrizes we just updated
            else:
                approximated_X_unfolded_n = tl.unfold(approximated_X, n) # use the approximation from the previous iteration step, not using the matrix updates calculated in this iteration
            
            
                
            
            
            f = lambda A: tl.sum( tl.matmul(A, tl.transpose(khatri_rao_product)) - tl.base.unfold(X, n) * tl.log( tl.matmul(A, tl.transpose(khatri_rao_product)) ))  # lambda for function we actually want to minimize
            function_value_at_iteration = tl.sum(approximated_X_unfolded_n - tl.base.unfold(X, n) * tl.log(approximated_X_unfolded_n)) 
            gradient_at_iteration = tl.matmul(tl.ones(approximated_X_unfolded_n.shape, **tl.context(X)) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )
            riemanndian_gradient_at_iteration = A_ns[n] * gradient_at_iteration # The "A_ns[n] *" is the inverse of the Riemannian metric tensor matrix applied to the gradient, i.e. G(A)^{-1} (\nabla f)
            norm_of_rg = tl.sum(gradient_at_iteration * riemanndian_gradient_at_iteration) # TODO maybe check if this is correct! But it should be since we calculate the Riemmannian norm of the Riemannian gradient as \| grad f \|_g = (G^{-1} \nabla f)^T G G^{-1} \nabla f = \nabla f^T grad f
            
            step_size = 5.0 / khatri_rao_product.shape[0] # fixed step size according to my estimates
            # still need to make sure we donÄt get problem in first iteration so we need to ensure that in the exponent there is nothing bigger then 10!
            largest_element_gradient = -tl.min(gradient_at_iteration)
            step_size = min(step_size, 2.0 / largest_element_gradient)
            
            next_iterate =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
            if verbose:
                print("Time from start to calculate gradients and first next iterate: " + str(time.time() - start))
            # if Armijo step size condition is not fullfilled, try again with smaller step size. Thanks to math, this is while loop will eventually finish
            #if is_tensor_not_finite(next_iterate) or ( function_value_at_iteration - 0.1 * step_size * norm_of_rg < f(next_iterate) ):
            #    # TODO: instead of recalculating like this, we can also use (for beta=0.5) that exp(beta * stuff) = [exp(stuff)]^beta and if beta=0.5 this is just sqrt which is 3 times faster
            #    print("step size too big")
                
                
            if verbose:
                print("Time from start until end of step size calculation: " + str(time.time() - start))
                print("Biggest element in -gradient: " + str(tl.max(-gradient_at_iteration)))
                print("smallest element in -gradient: " + str(tl.min(-gradient_at_iteration)))
                print("Step Size: " + str(step_size))
                #print("m: " + str(m))
                print("Step size * biggest element: " + str(step_size * tl.max(-gradient_at_iteration)))
                print("biggest element in A_n: " + str(tl.max(A_ns[n])))
                print("smallest element in A_n: " + str(tl.min(A_ns[n])))
                print("Shape of approximated_X_unfolded_n: " + str(approximated_X_unfolded_n.shape))
                print("Shape of khatri Rao product: " + str(khatri_rao_product.shape))
            
            step_size_modifiers[n].append(step_size) # save the value of m for later inspection
            # finally update A_n
            A_ns[n] = next_iterate
            # OLD CODE, kept for safety, but can be deleted
            # regular * does componentwise multiplication
            #A_ns[n] = A_ns[n] * tl.exp(-step_size * tl.matmul(tl.ones(approximated_X_unfolded_n.shape) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )  ) 
            
            end = time.time()
            if verbose:
                print("Calculculation time: " + str(end - start))
                print("New objective function value: " + str(f(A_ns[n])))

                print("function_value_at_iteration = " + str(function_value_at_iteration))
                print("norm_of_rg = " + str(norm_of_rg))
                print("biggest Element of X/M = " + str(tl.max(tl.abs(tl.base.unfold(X, n) / approximated_X_unfolded_n))))
                #print("gradiend_at_iteration = ")
                #print(gradient_at_iteration)
                #print("riemannian_gradient_at_iteration = ")
                #print(riemanndian_gradient_at_iteration)

                #print("new A_ns[n]:")
                #print(A_ns[n])
                print("\n")
                
                
            
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