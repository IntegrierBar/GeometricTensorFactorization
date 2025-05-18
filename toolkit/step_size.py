"""
This file contains the functions to analyze step sizes of our new algorithm.
Due to relative imports we can only call this script from top level.
"""

import tensorly as tl
import numpy as np
import time
import math
import matplotlib.pyplot as plt

from typing import List

from skimage import data

from .constants import (
    picture_folder, data_folder,
    error_label, iteration_label, tensor_dimension_label, time_label,
    xscale_convergence_data, xscale_convergence, yscale_convergence
) 
from tensorfactorization.utils import (create_initial_data, defactorizing_CP, random_cp_with_noise)
from data.data_imports import (load_indian_pines)

from tensorfactorization.poisson import (tensor_factorization_cp_poisson)


def show_step_sizes(tensor: tl.tensor, F: int = 3, alpha: float = 0.5, beta: float = 0.5, sigma: float = 0.5):
    """
    This function runs the geometric algorithm on the tensor once, without scaling, with mean=1, with max=1.
    Then plots the convergence lines and the step sizes.
    
    Args:
        tensor: the tensor to factorize
        F: the F for the factorization
    """  
    max_iter = 300 
    N = tensor.ndim
    
    scalings = {
        "unscaled" : 1.0,
        "mean=1" : 1.0/tl.mean(tensor),
        "max=1" : 1.0/tl.max(tensor),
    } 
    colors = {
        "unscaled" : "red",
        "mean=1" : "blue",
        "max=1" : "green",
    }
    
    all_step_sizes = [{} for _ in range(N)]
    
    plt.figure() # figure for the convergence lines
    # create initial data to use for all algorithms
    initial_A_ns = create_initial_data(tensor, F)
    
    for name, scaling in scalings.items(): 
        # scale the tensor and the initial data
        scaled_tensor = tensor * scaling 
        initial_A_ns = rescale_initial_data(initial_A_ns, scaled_tensor)
        start = time.time()
        _, RE, _, step_size_modifiers = tensor_factorization_cp_poisson(scaled_tensor, F, max_iter=max_iter, detailed=True, initial_A_ns=initial_A_ns, alpha=alpha, beta=beta, sigma=sigma)
        end = time.time()
        RE = tl.tensor(RE) # convert to numpy array
        time_total = end - start
        print(f"\n \n {name} Tensor")
        print(f"took {time_total:.3f} seconds with final error {RE[-1]}")
        for i in range(N):
            all_step_sizes[i][name] = [math.pow(beta, m) * alpha for m in step_size_modifiers[i]]
        plt.plot(tl.to_numpy(RE), label=name, color=colors[name])
        
    plt.title("Convergence Lines")
    plt.xlabel(iteration_label)
    plt.ylabel(error_label)
    # TODO das mal noch genauer anschauen, was da sinnvoll ist
    plt.yscale(yscale_convergence)
    #plt.xscale(xscale_convergence)
    plt.xscale(**xscale_convergence_data)
    plt.xlim(left=0)
    plt.legend(title='Variants', loc='upper right')
    plt.savefig(f"{picture_folder}step_size_convergence.png")

    for i in range(N):
        plt.figure()
        for name, step_sizes in all_step_sizes[i].items():
            plt.plot(step_sizes, label=name, color=colors[name])
        plt.legend()
        plt.title(f"Step Sizes for Matrix Factor {i}")
        plt.yscale(yscale_convergence)
        plt.savefig(f"{picture_folder}step_sizes_factor{i}.png")
        

    

def rescale_initial_data(initial_A_ns, tensor):
    """
    rescale the initial_A_ns so that the norm of the approximating tensor is the same as the tensor
    """
    norm_X = tl.norm(tensor)
    norm_approx = tl.norm( defactorizing_CP(initial_A_ns, tensor.shape) )
    # since we want to equally divide the scalling amongst the factors, we need to take the nth root
    scaling = (norm_X / norm_approx) ** (1.0/ tensor.ndim)

    for n in range(len(initial_A_ns)):  
        initial_A_ns[n] = initial_A_ns[n] * scaling

    return initial_A_ns