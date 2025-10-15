"""
This python file contains the algorithm for Poisson Family special case
"""


import time
import tensorly as tl
import math
from .utils import defactorizing_CP, is_tensor_not_finite, poisson_error

from copy import deepcopy

import warnings


class BacktrackingWarning(UserWarning):
    """A special warning for when the backtracking fails."""
    pass



def tensor_factorization_cp_poisson(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, update_approximation_everytime=True, initial_A_ns=None, alpha=0.5, beta=0.5, sigma=0.5, eps=None):
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
      eps: all values lower than this value are lifted to this value
    
    Returns:
      A_ns: A list of matrizes approximating X 
      RE (optional): list of all errors during optimization. Uses quadratic/Gauss error instead of poisson error currently
      approximated_X (optional): final approximation of X
      step_size_modifiers (optional): list of all step-size-modifiers m used during iteration
    """
    # if no eps was specified, we use the machine epsilon for float32 (smallest number s.t. 1+eps>1)
    if eps==None:
        eps = tl.eps(tl.float32)
    
    # first of all we clamp X to eps so we do not get problems due to machine precission
    X[X<eps] = eps
    
    N = X.ndim # get dimension of X
    X_shape = X.shape
    norm_X = tl.norm(X)
    # initialize A_j with random positive values if it was not given
    if initial_A_ns is None:
        # TODO change this so it uses the function from utils and check which initialization is best
        A_ns = []
        for i in range(N):
            # we use random.random_tensor as it returns a tensor
            A_ns.append(tl.random.random_tensor((X_shape[i], F), **tl.context(X)))
    else:
        if len(initial_A_ns) != N:
            raise ValueError("initial A_ns given does not have to correct length")
        for i in range(N):
            if initial_A_ns[i].shape != (X_shape[i], F):
                raise ValueError(f"inital A_ns with index{i} does not have correct dimension. Should be {(X_shape[i], F)} but is {initial_A_ns[i].shape}")
            if tl.context(initial_A_ns[i]) != tl.context(X):
                raise ValueError(f"inital A_ns with index {i} does not have the same context as X. Should be {tl.context(X)} but is {tl.context(initial_A_ns[i])}")
        A_ns = deepcopy(initial_A_ns) # use copy since that is how we want to later use it for testing
    # the reconstruction error
    approximated_X = defactorizing_CP(A_ns, X_shape)
        
    RE = [tl.norm(X-approximated_X)/norm_X]

    # lets save the different m as well so that we can run some analysis there
    step_size_modifiers = []
    for i in range(N):
        step_size_modifiers.append([])
    
    # We use this to manualy stop the loop, if the backtracking failed for all matrix factors in one loop
    failed_to_backtrack = [False] * N
        
    ## MAIN LOOP ##
    for iteration in range(max_iter):
        for n in range(N):
            if verbose:
                print(f"Current index: {n}")
                
            start = time.time()
            
            khatri_rao_product = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            
            if update_approximation_everytime:
                approximated_X_unfolded_n = tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)) # use the new approximation using the matrizes we just updated
            else:
                approximated_X_unfolded_n = tl.unfold(approximated_X, n) # use the approximation from the previous iteration step, not using the matrix updates calculated in this iteration
            
            approximated_X_unfolded_n[approximated_X_unfolded_n<eps] = eps
            
            #f = lambda A: tl.sum( tl.matmul(A, tl.transpose(khatri_rao_product)) - tl.base.unfold(X, n) * tl.log( tl.matmul(A, tl.transpose(khatri_rao_product)) ))  # lambda for function we actually want to minimize
            function_value_at_iteration = poisson_error(tl.base.unfold(X, n), approximated_X_unfolded_n) #tl.sum(approximated_X_unfolded_n - tl.base.unfold(X, n) * tl.log(approximated_X_unfolded_n)) 
            gradient_at_iteration = tl.matmul(tl.ones(approximated_X_unfolded_n.shape, **tl.context(X)) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )
            riemanndian_gradient_at_iteration = A_ns[n] * gradient_at_iteration # The "A_ns[n] *" is the inverse of the Riemannian metric tensor matrix applied to the gradient, i.e. G(A)^{-1} (\nabla f)
            norm_of_rg = tl.sum(gradient_at_iteration * riemanndian_gradient_at_iteration) # TODO maybe check if this is correct! But it should be since we calculate the Riemmannian norm of the Riemannian gradient as \| grad f \|_g = (G^{-1} \nabla f)^T G G^{-1} \nabla f = \nabla f^T grad f
            
            
            ###### Step size calculation ######
            # TODO these values should be looked at more to determine which are best
            #sigma = 0.5
            #beta = 0.5
            #alpha = 0.5
            
            
            # for now, just use 0.7 of what was used last time
            # TODO need to find some estimation for first m! otherwise need to compute too much, at least add test for finiteness first!
            if len(step_size_modifiers[n]) > 0:
                m = int(step_size_modifiers[n][-1] * 0.7)
                # TODO seems like we want to make sure that step_size * max(-grad) < 10
            else:
                m = 0
            # first we make sure that the step size is small enough that we do not get any numerical problems with exp (exp(700)=infty)
            # TODO need to make this smarter for better speed
            biggest_negative_element_grad = max(0, -tl.min(gradient_at_iteration))
            while alpha * math.pow(beta, m) * biggest_negative_element_grad > 600:
                m += 1
            
            # TODO here is a massive problem. For some reason if our tensor is outside [0,1] we get problems because function_value_at_iteration can be negative?
            # For first few iterations norm_of_rg is larger then function_value_at_iteration. And since we need function_value_at_iteration - sigma*step_size*norm_of_rg > f(next_iteration) > 0, we need to choose initial m such that norm_of_rg is less then f(current_iterate)
            #print(function_value_at_iteration)
            #m = max(math.ceil(math.log(function_value_at_iteration / (sigma * alpha * norm_of_rg), beta)), m)
            if verbose:
                print(f"Initial m = {m}")
                print(f"Biggest element in -gradient: {tl.max(-gradient_at_iteration)}")
                print(f"smallest element in -gradient: {tl.min(-gradient_at_iteration)}")
                #print(math.log(function_value_at_iteration / (sigma * alpha * norm_of_rg)))
                #print(math.log(beta))
                
            step_size = math.pow(beta, m) * alpha # initial step size
            next_iterate =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
            if verbose:
                print("Time from start to calculate gradients and first next iterate: " + str(time.time() - start))
                
            # if Armijo step size condition is not fullfilled, try again with smaller step size. Thanks to math, this is while loop will eventually finish
            n_backtracks = 0 # how many backtrackings we did
            while is_tensor_not_finite(next_iterate) or ( function_value_at_iteration - sigma * step_size * norm_of_rg < poisson_error( tl.base.unfold(X, n), tl.matmul(next_iterate, tl.transpose(khatri_rao_product)) ) ):
                # TODO: instead of recalculating like this, we can also use (for beta=0.5) that exp(beta * stuff) = [exp(stuff)]^beta and if beta=0.5 this is just sqrt which is 3 times faster then recalculating exp. NOT WORTH IT
                # if we need more than 200 backtracks something is wrong and we give a warning and skip this update
                n_backtracks += 1
                if n_backtracks > 200:
                    failed_to_backtrack[n] = True
                    warnings.warn("Backtracking did not converge in time so we skip this update.", BacktrackingWarning)
                    if verbose:
                        print("#### PRINTING ADDITIONAL INFORMATION ####")
                        print("Current iteration: " + str(iteration))
                        print("Current index: " + str(n))
                        print("Step Size: " + str(step_size))
                        print("m: " + str(m))
                        print("Step size * biggest element of negative gradient: " + str(step_size * tl.max(-gradient_at_iteration)))
                        print("biggest element in A_n: " + str(tl.max(A_ns[n])))
                        print("smallest element in A_n: " + str(tl.min(A_ns[n])))
                        print("Shape of approximated_X_unfolded_n: " + str(approximated_X_unfolded_n.shape))
                        print("Shape of khatri Rao product: " + str(khatri_rao_product.shape))
                        print("\nPoisson error of current iteration: " + str(function_value_at_iteration))
                        print("norm of Riemannian Gradient: " + str(norm_of_rg))
                        print("poisson error of next iteration: " + str(poisson_error( tl.base.unfold(X, n), tl.matmul(next_iterate, tl.transpose(khatri_rao_product)) )))
                        print("\nPrinting additional tensor information")
                        tensors_to_print = {
                            "TENSOR" : X,
                            "GRADIENT" : gradient_at_iteration,
                            "NEXT ITERATE" : next_iterate,
                            "KHATRI RAO PRODUCT" : khatri_rao_product,
                            "APPROXIMATED X" : approximated_X_unfolded_n,
                        }
                        for index, A_n in enumerate(A_ns):
                            tensors_to_print["A_ns["+str(index)+"]"] = A_n
                        
                        for name, tensor_to_print in tensors_to_print.items():
                            print(name + ": smallest: " + str(tl.min(tensor_to_print)) + ", biggest: " + str(tl.max(tensor_to_print)) + ", average: " + str(tl.mean(tensor_to_print)))
                        #print("NEXT ITERATE: smallest element: " + str(tl.min(next_iterate)) + " biggest element: " + str(tl.max(next_iterate)))
                        #print("KHATRI RAO PRODUCT: smallest element: " + str(tl.min(khatri_rao_product)) + " biggest element: " + str(tl.max(khatri_rao_product)))
                        #print("APPROXIMATED X: smallest element: " + str(tl.min(approximated_X_unfolded_n)) + " biggest element: " + str(tl.max(approximated_X_unfolded_n)))
                        #for index, A_n in enumerate(A_ns):
                        #    print("A_ns["+str(index)+"]: smallest element: " + str(tl.min(A_n)) + " biggest element: " + str(tl.max(A_n)) + ", average: " + str(tl.mean(A_n)))
                        print("\n")
                    break
                
                m += 1
                step_size = math.pow(beta, m) * alpha
                next_iterate =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
            # if the backtracking did not converge we skip the update
            if n_backtracks > 200:
                continue
            failed_to_backtrack[n] = False # since we reached here, the backtracking worked
            
            if verbose:
                print("Time from start until end of step size calculation: " + str(time.time() - start))
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
                print("New objective function value: " + str(poisson_error(tl.base.unfold(X, n), tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)))))

                print("function_value_at_iteration = " + str(function_value_at_iteration))
                print("norm_of_rg = " + str(norm_of_rg))
                print("biggest Element of X/M = " + str(tl.max(tl.abs(tl.base.unfold(X, n) / approximated_X_unfolded_n))))
                print("\nPrinting additional tensor information")
                tensors_to_print = {
                    "TENSOR" : X,
                    "GRADIENT" : gradient_at_iteration,
                    "NEXT ITERATE" : next_iterate,
                    "KHATRI RAO PRODUCT" : khatri_rao_product,
                    "APPROXIMATED X" : approximated_X_unfolded_n,
                }
                for index, A_n in enumerate(A_ns):
                    tensors_to_print["A_ns["+str(index)+"]"] = A_n
                    
                for name, tensor_to_print in tensors_to_print.items():
                    print(name + ": smallest: " + str(tl.min(tensor_to_print)) + ", biggest: " + str(tl.max(tensor_to_print)) + ", average: " + str(tl.mean(tensor_to_print)))
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
        
        # if we had one round of backtracking not working, we abort the loop
        if all(failed_to_backtrack):
            warnings.warn("Backtracking has broken down, so we stop iterations", BacktrackingWarning)
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



def tensor_factorization_cp_poisson_fixed_step_size(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, update_approximation_everytime=True, initial_A_ns=None, eps=None):
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
    # if no eps was specified, we use the machine epsilon for float32 (smallest number s.t. 1+eps>1)
    if eps==None:
        eps = tl.eps(tl.float32)
    
    # first of all we clamp X to eps so we do not get problems due to machine precission
    X[X<eps] = eps
    
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

    # lets save the different step sizes
    step_sizes = []
    for i in range(N):
        step_sizes.append([])
    
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

            approximated_X_unfolded_n[approximated_X_unfolded_n<eps] = eps
            
            gradient_at_iteration = tl.matmul(tl.ones(approximated_X_unfolded_n.shape, **tl.context(X)) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )
            
            step_size = 1.0 * math.pow(khatri_rao_product.shape[0], -1) # fixed step size according to my estimates
            # still need to make sure we don't get problem in first iteration so we need to ensure that in the exponent there is nothing bigger then 3, since e^3=20 should be enough!
            largest_element_gradient = -tl.min(gradient_at_iteration)
            step_size = min(step_size, 3.0 / largest_element_gradient)
            
            A_ns[n] =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
                
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
            
            step_sizes[n].append(step_size) # save the value of m for later inspection
            
            end = time.time()
            if verbose:
                print("Calculculation time: " + str(end - start))
                #print("New objective function value: " + str(f(A_ns[n]))) # TODO fix the prints here

                #print("function_value_at_iteration = " + str(function_value_at_iteration))
                #print("norm_of_rg = " + str(norm_of_rg))
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
        return A_ns, RE, approximated_X, step_sizes
    return A_ns

    
    
    
def tensor_factorization_cp_poisson_exp(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, update_approximation_everytime=True, initial_A_ns=None, alpha=0.5, beta=0.5, sigma=0.5, eps=None):
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
      eps: all values lower than this value are lifted to this value
    
    Returns:
      A_ns: A list of matrizes approximating X 
      RE (optional): list of all errors during optimization. Uses quadratic/Gauss error instead of poisson error currently
      approximated_X (optional): final approximation of X
      step_size_modifiers (optional): list of all step-size-modifiers m used during iteration
    """
    # if no eps was specified, we use the machine epsilon for float32 (smallest number s.t. 1+eps>1)
    if eps==None:
        eps = tl.eps(tl.float32)
    
    # first of all we clamp X to eps so we do not get problems due to machine precission
    X[X<eps] = eps
    
    N = X.ndim # get dimension of X
    X_shape = X.shape
    norm_X = tl.norm(X)
    # initialize A_j with random positive values if it was not given
    if initial_A_ns is None:
        # TODO change this so it uses the function from utils and check which initialization is best
        A_ns = []
        for i in range(N):
            # we use random.random_tensor as it returns a tensor
            A_ns.append(tl.random.random_tensor((X_shape[i], F), **tl.context(X)))
    else:
        if len(initial_A_ns) != N:
            raise ValueError("initial A_ns given does not have to correct length")
        for i in range(N):
            if initial_A_ns[i].shape != (X_shape[i], F):
                raise ValueError(f"inital A_ns with index{i} does not have correct dimension. Should be {(X_shape[i], F)} but is {initial_A_ns[i].shape}")
            if tl.context(initial_A_ns[i]) != tl.context(X):
                raise ValueError(f"inital A_ns with index {i} does not have the same context as X. Should be {tl.context(X)} but is {tl.context(initial_A_ns[i])}")
        A_ns = deepcopy(initial_A_ns) # use copy since that is how we want to later use it for testing
    # the reconstruction error
    approximated_X = defactorizing_CP(A_ns, X_shape)
        
    RE = [tl.norm(X-approximated_X)/norm_X]

    # lets save the different m as well so that we can run some analysis there
    step_size_modifiers = []
    for i in range(N):
        step_size_modifiers.append([])
    
    # We use this to manualy stop the loop, if the backtracking failed for all matrix factors in one loop
    failed_to_backtrack = [False] * N
        
    ## MAIN LOOP ##
    for iteration in range(max_iter):
        for n in range(N):
            if verbose:
                print(f"Current index: {n}")
                
            start = time.time()
            
            khatri_rao_product = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            
            if update_approximation_everytime:
                approximated_X_unfolded_n = tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)) # use the new approximation using the matrizes we just updated
            else:
                approximated_X_unfolded_n = tl.unfold(approximated_X, n) # use the approximation from the previous iteration step, not using the matrix updates calculated in this iteration
            
            approximated_X_unfolded_n[approximated_X_unfolded_n<eps] = eps
            
            #f = lambda A: tl.sum( tl.matmul(A, tl.transpose(khatri_rao_product)) - tl.base.unfold(X, n) * tl.log( tl.matmul(A, tl.transpose(khatri_rao_product)) ))  # lambda for function we actually want to minimize
            function_value_at_iteration = poisson_error(tl.base.unfold(X, n), approximated_X_unfolded_n) #tl.sum(approximated_X_unfolded_n - tl.base.unfold(X, n) * tl.log(approximated_X_unfolded_n)) 
            gradient_at_iteration = tl.matmul(tl.ones(approximated_X_unfolded_n.shape, **tl.context(X)) - (tl.base.unfold(X, n) / approximated_X_unfolded_n) , khatri_rao_product )
            riemanndian_gradient_at_iteration = A_ns[n] * gradient_at_iteration # The "A_ns[n] *" is the inverse of the Riemannian metric tensor matrix applied to the gradient, i.e. G(A)^{-1} (\nabla f)
            norm_of_rg = tl.sum(gradient_at_iteration * riemanndian_gradient_at_iteration) # TODO maybe check if this is correct! But it should be since we calculate the Riemmannian norm of the Riemannian gradient as \| grad f \|_g = (G^{-1} \nabla f)^T G G^{-1} \nabla f = \nabla f^T grad f
            
            
            ###### Step size calculation ######
            # TODO these values should be looked at more to determine which are best
            #sigma = 0.5
            #beta = 0.5
            #alpha = 0.5
            
            
            # for now, just use 0.7 of what was used last time
            # TODO need to find some estimation for first m! otherwise need to compute too much, at least add test for finiteness first!
            if len(step_size_modifiers[n]) > 0:
                m = int(step_size_modifiers[n][-1] * 0.7)
                # TODO seems like we want to make sure that step_size * max(-grad) < 10
            else:
                m = 0
            # first we make sure that the step size is small enough that we do not get any numerical problems with exp (exp(700)=infty)
            # TODO need to make this smarter for better speed
            biggest_negative_element_grad = max(0, -tl.min(gradient_at_iteration))
            while alpha * math.pow(beta, m) * biggest_negative_element_grad > 600:
                m += 1
            
            # TODO here is a massive problem. For some reason if our tensor is outside [0,1] we get problems because function_value_at_iteration can be negative?
            # For first few iterations norm_of_rg is larger then function_value_at_iteration. And since we need function_value_at_iteration - sigma*step_size*norm_of_rg > f(next_iteration) > 0, we need to choose initial m such that norm_of_rg is less then f(current_iterate)
            #print(function_value_at_iteration)
            #m = max(math.ceil(math.log(function_value_at_iteration / (sigma * alpha * norm_of_rg), beta)), m)
            if verbose:
                print(f"Initial m = {m}")
                print(f"Biggest element in -gradient: {tl.max(-gradient_at_iteration)}")
                print(f"smallest element in -gradient: {tl.min(-gradient_at_iteration)}")
                #print(math.log(function_value_at_iteration / (sigma * alpha * norm_of_rg)))
                #print(math.log(beta))
                
            step_size = math.pow(beta, m) * alpha # initial step size
            next_iterate =  A_ns[n] * (tl.ones(A_ns[n].shape, **tl.context(X))- step_size/2.0 * gradient_at_iteration)**2
            if verbose:
                print("Time from start to calculate gradients and first next iterate: " + str(time.time() - start))
                
            # if Armijo step size condition is not fullfilled, try again with smaller step size. Thanks to math, this is while loop will eventually finish
            n_backtracks = 0 # how many backtrackings we did
            while is_tensor_not_finite(next_iterate) or ( function_value_at_iteration - sigma * step_size * norm_of_rg < poisson_error( tl.base.unfold(X, n), tl.matmul(next_iterate, tl.transpose(khatri_rao_product)) ) ):
                # TODO: instead of recalculating like this, we can also use (for beta=0.5) that exp(beta * stuff) = [exp(stuff)]^beta and if beta=0.5 this is just sqrt which is 3 times faster then recalculating exp. NOT WORTH IT
                # if we need more than 200 backtracks something is wrong and we give a warning and skip this update
                n_backtracks += 1
                if n_backtracks > 200:
                    failed_to_backtrack[n] = True
                    warnings.warn("Backtracking did not converge in time so we skip this update.", BacktrackingWarning)
                    if verbose:
                        print("#### PRINTING ADDITIONAL INFORMATION ####")
                        print("Current iteration: " + str(iteration))
                        print("Current index: " + str(n))
                        print("Step Size: " + str(step_size))
                        print("m: " + str(m))
                        print("Step size * biggest element of negative gradient: " + str(step_size * tl.max(-gradient_at_iteration)))
                        print("biggest element in A_n: " + str(tl.max(A_ns[n])))
                        print("smallest element in A_n: " + str(tl.min(A_ns[n])))
                        print("Shape of approximated_X_unfolded_n: " + str(approximated_X_unfolded_n.shape))
                        print("Shape of khatri Rao product: " + str(khatri_rao_product.shape))
                        print("\nPoisson error of current iteration: " + str(function_value_at_iteration))
                        print("norm of Riemannian Gradient: " + str(norm_of_rg))
                        print("poisson error of next iteration: " + str(poisson_error( tl.base.unfold(X, n), tl.matmul(next_iterate, tl.transpose(khatri_rao_product)) )))
                        print("\nPrinting additional tensor information")
                        tensors_to_print = {
                            "TENSOR" : X,
                            "GRADIENT" : gradient_at_iteration,
                            "NEXT ITERATE" : next_iterate,
                            "KHATRI RAO PRODUCT" : khatri_rao_product,
                            "APPROXIMATED X" : approximated_X_unfolded_n,
                        }
                        for index, A_n in enumerate(A_ns):
                            tensors_to_print["A_ns["+str(index)+"]"] = A_n
                        
                        for name, tensor_to_print in tensors_to_print.items():
                            print(name + ": smallest: " + str(tl.min(tensor_to_print)) + ", biggest: " + str(tl.max(tensor_to_print)) + ", average: " + str(tl.mean(tensor_to_print)))
                        #print("NEXT ITERATE: smallest element: " + str(tl.min(next_iterate)) + " biggest element: " + str(tl.max(next_iterate)))
                        #print("KHATRI RAO PRODUCT: smallest element: " + str(tl.min(khatri_rao_product)) + " biggest element: " + str(tl.max(khatri_rao_product)))
                        #print("APPROXIMATED X: smallest element: " + str(tl.min(approximated_X_unfolded_n)) + " biggest element: " + str(tl.max(approximated_X_unfolded_n)))
                        #for index, A_n in enumerate(A_ns):
                        #    print("A_ns["+str(index)+"]: smallest element: " + str(tl.min(A_n)) + " biggest element: " + str(tl.max(A_n)) + ", average: " + str(tl.mean(A_n)))
                        print("\n")
                    break
                
                m += 1
                step_size = math.pow(beta, m) * alpha
                next_iterate =  A_ns[n] * (tl.ones(A_ns[n].shape, **tl.context(X))- step_size/2.0 * gradient_at_iteration)**2
            # if the backtracking did not converge we skip the update
            if n_backtracks > 200:
                continue
            failed_to_backtrack[n] = False # since we reached here, the backtracking worked
            
            if verbose:
                print("Time from start until end of step size calculation: " + str(time.time() - start))
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
                print("New objective function value: " + str(poisson_error(tl.base.unfold(X, n), tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)))))

                print("function_value_at_iteration = " + str(function_value_at_iteration))
                print("norm_of_rg = " + str(norm_of_rg))
                print("biggest Element of X/M = " + str(tl.max(tl.abs(tl.base.unfold(X, n) / approximated_X_unfolded_n))))
                print("\nPrinting additional tensor information")
                tensors_to_print = {
                    "TENSOR" : X,
                    "GRADIENT" : gradient_at_iteration,
                    "NEXT ITERATE" : next_iterate,
                    "KHATRI RAO PRODUCT" : khatri_rao_product,
                    "APPROXIMATED X" : approximated_X_unfolded_n,
                }
                for index, A_n in enumerate(A_ns):
                    tensors_to_print["A_ns["+str(index)+"]"] = A_n
                    
                for name, tensor_to_print in tensors_to_print.items():
                    print(name + ": smallest: " + str(tl.min(tensor_to_print)) + ", biggest: " + str(tl.max(tensor_to_print)) + ", average: " + str(tl.mean(tensor_to_print)))
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
        
        # if we had one round of backtracking not working, we abort the loop
        if all(failed_to_backtrack):
            warnings.warn("Backtracking has broken down, so we stop iterations", BacktrackingWarning)
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



def smart_error(X_unfolded_n, M_unfolded_n):
    return tl.sum(M_unfolded_n * tl.log(M_unfolded_n / X_unfolded_n) - M_unfolded_n)


def tensor_factorization_cp_poisson_SMART(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, update_approximation_everytime=True, initial_A_ns=None, alpha=0.5, beta=0.5, sigma=0.5, eps=None):
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
      eps: all values lower than this value are lifted to this value
    
    Returns:
      A_ns: A list of matrizes approximating X 
      RE (optional): list of all errors during optimization. Uses quadratic/Gauss error instead of poisson error currently
      approximated_X (optional): final approximation of X
      step_size_modifiers (optional): list of all step-size-modifiers m used during iteration
    """
    # if no eps was specified, we use the machine epsilon for float32 (smallest number s.t. 1+eps>1)
    if eps==None:
        eps = tl.eps(tl.float32)
    
    # first of all we clamp X to eps so we do not get problems due to machine precission
    X[X<eps] = eps
    
    N = X.ndim # get dimension of X
    X_shape = X.shape
    norm_X = tl.norm(X)
    # initialize A_j with random positive values if it was not given
    if initial_A_ns is None:
        # TODO change this so it uses the function from utils and check which initialization is best
        A_ns = []
        for i in range(N):
            # we use random.random_tensor as it returns a tensor
            A_ns.append(tl.random.random_tensor((X_shape[i], F), **tl.context(X)))
    else:
        if len(initial_A_ns) != N:
            raise ValueError("initial A_ns given does not have to correct length")
        for i in range(N):
            if initial_A_ns[i].shape != (X_shape[i], F):
                raise ValueError(f"inital A_ns with index{i} does not have correct dimension. Should be {(X_shape[i], F)} but is {initial_A_ns[i].shape}")
            if tl.context(initial_A_ns[i]) != tl.context(X):
                raise ValueError(f"inital A_ns with index {i} does not have the same context as X. Should be {tl.context(X)} but is {tl.context(initial_A_ns[i])}")
        A_ns = deepcopy(initial_A_ns) # use copy since that is how we want to later use it for testing
    # the reconstruction error
    approximated_X = defactorizing_CP(A_ns, X_shape)
        
    RE = [tl.norm(X-approximated_X)/norm_X]

    # lets save the different m as well so that we can run some analysis there
    step_size_modifiers = []
    for i in range(N):
        step_size_modifiers.append([])
    
    # We use this to manualy stop the loop, if the backtracking failed for all matrix factors in one loop
    failed_to_backtrack = [False] * N
        
    ## MAIN LOOP ##
    for iteration in range(max_iter):
        for n in range(N):
            if verbose:
                print(f"Current index: {n}")
                
            start = time.time()
            
            khatri_rao_product = tl.tenalg.khatri_rao(A_ns, skip_matrix=n)
            
            if update_approximation_everytime:
                approximated_X_unfolded_n = tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)) # use the new approximation using the matrizes we just updated
            else:
                approximated_X_unfolded_n = tl.unfold(approximated_X, n) # use the approximation from the previous iteration step, not using the matrix updates calculated in this iteration
            
            approximated_X_unfolded_n[approximated_X_unfolded_n<eps] = eps
            
            #f = lambda A: tl.sum( tl.matmul(A, tl.transpose(khatri_rao_product)) - tl.base.unfold(X, n) * tl.log( tl.matmul(A, tl.transpose(khatri_rao_product)) ))  # lambda for function we actually want to minimize
            function_value_at_iteration = smart_error(tl.base.unfold(X, n), approximated_X_unfolded_n) #tl.sum(approximated_X_unfolded_n - tl.base.unfold(X, n) * tl.log(approximated_X_unfolded_n)) 
            gradient_at_iteration = tl.matmul(tl.log( approximated_X_unfolded_n / tl.base.unfold(X, n) ) , khatri_rao_product )
            riemanndian_gradient_at_iteration = A_ns[n] * gradient_at_iteration # The "A_ns[n] *" is the inverse of the Riemannian metric tensor matrix applied to the gradient, i.e. G(A)^{-1} (\nabla f)
            norm_of_rg = tl.sum(gradient_at_iteration * riemanndian_gradient_at_iteration) # TODO maybe check if this is correct! But it should be since we calculate the Riemmannian norm of the Riemannian gradient as \| grad f \|_g = (G^{-1} \nabla f)^T G G^{-1} \nabla f = \nabla f^T grad f
            
            
            ###### Step size calculation ######
            # TODO these values should be looked at more to determine which are best
            #sigma = 0.5
            #beta = 0.5
            #alpha = 0.5
            
            
            # for now, just use 0.7 of what was used last time
            # TODO need to find some estimation for first m! otherwise need to compute too much, at least add test for finiteness first!
            if len(step_size_modifiers[n]) > 0:
                m = int(step_size_modifiers[n][-1] * 0.7)
                # TODO seems like we want to make sure that step_size * max(-grad) < 10
            else:
                m = 0
            # first we make sure that the step size is small enough that we do not get any numerical problems with exp (exp(700)=infty)
            # TODO need to make this smarter for better speed
            biggest_negative_element_grad = max(0, -tl.min(gradient_at_iteration))
            while alpha * math.pow(beta, m) * biggest_negative_element_grad > 600:
                m += 1
            
            # TODO here is a massive problem. For some reason if our tensor is outside [0,1] we get problems because function_value_at_iteration can be negative?
            # For first few iterations norm_of_rg is larger then function_value_at_iteration. And since we need function_value_at_iteration - sigma*step_size*norm_of_rg > f(next_iteration) > 0, we need to choose initial m such that norm_of_rg is less then f(current_iterate)
            #print(function_value_at_iteration)
            #m = max(math.ceil(math.log(function_value_at_iteration / (sigma * alpha * norm_of_rg), beta)), m)
            if verbose:
                print(f"Initial m = {m}")
                print(f"Biggest element in -gradient: {tl.max(-gradient_at_iteration)}")
                print(f"smallest element in -gradient: {tl.min(-gradient_at_iteration)}")
                #print(math.log(function_value_at_iteration / (sigma * alpha * norm_of_rg)))
                #print(math.log(beta))
                
            step_size = math.pow(beta, m) * alpha # initial step size
            next_iterate =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
            if verbose:
                print("Time from start to calculate gradients and first next iterate: " + str(time.time() - start))
                
            # if Armijo step size condition is not fullfilled, try again with smaller step size. Thanks to math, this is while loop will eventually finish
            n_backtracks = 0 # how many backtrackings we did
            while is_tensor_not_finite(next_iterate) or ( function_value_at_iteration - sigma * step_size * norm_of_rg < smart_error( tl.base.unfold(X, n), tl.matmul(next_iterate, tl.transpose(khatri_rao_product)) ) ):
                # TODO: instead of recalculating like this, we can also use (for beta=0.5) that exp(beta * stuff) = [exp(stuff)]^beta and if beta=0.5 this is just sqrt which is 3 times faster then recalculating exp. NOT WORTH IT
                # if we need more than 200 backtracks something is wrong and we give a warning and skip this update
                n_backtracks += 1
                if n_backtracks > 200:
                    failed_to_backtrack[n] = True
                    warnings.warn("Backtracking did not converge in time so we skip this update.", BacktrackingWarning)
                    if verbose:
                        print("#### PRINTING ADDITIONAL INFORMATION ####")
                        print("Current iteration: " + str(iteration))
                        print("Current index: " + str(n))
                        print("Step Size: " + str(step_size))
                        print("m: " + str(m))
                        print("Step size * biggest element of negative gradient: " + str(step_size * tl.max(-gradient_at_iteration)))
                        print("biggest element in A_n: " + str(tl.max(A_ns[n])))
                        print("smallest element in A_n: " + str(tl.min(A_ns[n])))
                        print("Shape of approximated_X_unfolded_n: " + str(approximated_X_unfolded_n.shape))
                        print("Shape of khatri Rao product: " + str(khatri_rao_product.shape))
                        print("\nPoisson error of current iteration: " + str(function_value_at_iteration))
                        print("norm of Riemannian Gradient: " + str(norm_of_rg))
                        print("poisson error of next iteration: " + str(poisson_error( tl.base.unfold(X, n), tl.matmul(next_iterate, tl.transpose(khatri_rao_product)) )))
                        print("\nPrinting additional tensor information")
                        tensors_to_print = {
                            "TENSOR" : X,
                            "GRADIENT" : gradient_at_iteration,
                            "NEXT ITERATE" : next_iterate,
                            "KHATRI RAO PRODUCT" : khatri_rao_product,
                            "APPROXIMATED X" : approximated_X_unfolded_n,
                        }
                        for index, A_n in enumerate(A_ns):
                            tensors_to_print["A_ns["+str(index)+"]"] = A_n
                        
                        for name, tensor_to_print in tensors_to_print.items():
                            print(name + ": smallest: " + str(tl.min(tensor_to_print)) + ", biggest: " + str(tl.max(tensor_to_print)) + ", average: " + str(tl.mean(tensor_to_print)))
                        #print("NEXT ITERATE: smallest element: " + str(tl.min(next_iterate)) + " biggest element: " + str(tl.max(next_iterate)))
                        #print("KHATRI RAO PRODUCT: smallest element: " + str(tl.min(khatri_rao_product)) + " biggest element: " + str(tl.max(khatri_rao_product)))
                        #print("APPROXIMATED X: smallest element: " + str(tl.min(approximated_X_unfolded_n)) + " biggest element: " + str(tl.max(approximated_X_unfolded_n)))
                        #for index, A_n in enumerate(A_ns):
                        #    print("A_ns["+str(index)+"]: smallest element: " + str(tl.min(A_n)) + " biggest element: " + str(tl.max(A_n)) + ", average: " + str(tl.mean(A_n)))
                        print("\n")
                    break
                
                m += 1
                step_size = math.pow(beta, m) * alpha
                next_iterate =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
            # if the backtracking did not converge we skip the update
            if n_backtracks > 200:
                continue
            failed_to_backtrack[n] = False # since we reached here, the backtracking worked
            
            if verbose:
                print("Time from start until end of step size calculation: " + str(time.time() - start))
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
                print("New objective function value: " + str(poisson_error(tl.base.unfold(X, n), tl.matmul(A_ns[n], tl.transpose(khatri_rao_product)))))

                print("function_value_at_iteration = " + str(function_value_at_iteration))
                print("norm_of_rg = " + str(norm_of_rg))
                print("biggest Element of X/M = " + str(tl.max(tl.abs(tl.base.unfold(X, n) / approximated_X_unfolded_n))))
                print("\nPrinting additional tensor information")
                tensors_to_print = {
                    "TENSOR" : X,
                    "GRADIENT" : gradient_at_iteration,
                    "NEXT ITERATE" : next_iterate,
                    "KHATRI RAO PRODUCT" : khatri_rao_product,
                    "APPROXIMATED X" : approximated_X_unfolded_n,
                }
                for index, A_n in enumerate(A_ns):
                    tensors_to_print["A_ns["+str(index)+"]"] = A_n
                    
                for name, tensor_to_print in tensors_to_print.items():
                    print(name + ": smallest: " + str(tl.min(tensor_to_print)) + ", biggest: " + str(tl.max(tensor_to_print)) + ", average: " + str(tl.mean(tensor_to_print)))
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
        
        # if we had one round of backtracking not working, we abort the loop
        if all(failed_to_backtrack):
            warnings.warn("Backtracking has broken down, so we stop iterations", BacktrackingWarning)
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



def tensor_factorization_cp_SMART_fixed_step_size(X, F, error=1e-6, max_iter=500, detailed=False, verbose=False, update_approximation_everytime=True, initial_A_ns=None, eps=None):
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
    # if no eps was specified, we use the machine epsilon for float32 (smallest number s.t. 1+eps>1)
    if eps==None:
        eps = tl.eps(tl.float32)
    
    # first of all we clamp X to eps so we do not get problems due to machine precission
    X[X<eps] = eps
    
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

    # lets save the different step sizes
    step_sizes = []
    for i in range(N):
        step_sizes.append([])
    
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

            approximated_X_unfolded_n[approximated_X_unfolded_n<eps] = eps
            
            gradient_at_iteration = tl.matmul(tl.log( approximated_X_unfolded_n / tl.base.unfold(X, n) ) , khatri_rao_product )
            
            one_norm_khatri_rao = tl.norm(khatri_rao_product, order=1)
            step_size = 1.0 / one_norm_khatri_rao # fixed step size according to my estimates
            # still need to make sure we don't get problem in first iteration so we need to ensure that in the exponent there is nothing bigger then 600 to avoid infinity
            largest_element_gradient = -tl.min(gradient_at_iteration)
            step_size = min(step_size, 600.0 / largest_element_gradient)
            
            A_ns[n] =  A_ns[n] * tl.exp(-step_size * gradient_at_iteration)
                
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
            
            step_sizes[n].append(step_size) # save the value of m for later inspection
            
            end = time.time()
            if verbose:
                print("Calculculation time: " + str(end - start))
                #print("New objective function value: " + str(f(A_ns[n]))) # TODO fix the prints here

                #print("function_value_at_iteration = " + str(function_value_at_iteration))
                #print("norm_of_rg = " + str(norm_of_rg))
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
        return A_ns, RE, approximated_X, step_sizes
    return A_ns