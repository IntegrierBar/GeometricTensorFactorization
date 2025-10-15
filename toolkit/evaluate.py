"""
This file contains the functions to evalute algorithms and generate convergence plots for both real and generated data.
Due to relative imports we can only call this script from top level.
"""

import tensorly as tl
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from typing import List

from skimage import data

from .classes import (Factorizer, IterationResult)
from .constants import (
    picture_folder, data_folder,
    error_label, iteration_label, tensor_dimension_label, time_label,
    xscale_convergence_data, xscale_convergence, yscale_convergence
) 
from tensorfactorization.utils import (create_initial_data, defactorizing_CP, random_cp_with_noise)
from data.data_imports import (load_indian_pines)




random_tensors = [
    {"dimensions": (10,10,10), "nreps": 20 }, # 1k
    {"dimensions": (20,20,5), "nreps": 20 }, # 2 k
    {"dimensions": (10,10,10,5), "nreps": 20 }, # 5k
    {"dimensions": (10,10,5,5,5), "nreps": 20 }, # 12,5k
    {"dimensions": (100,100,3), "nreps": 20 }, # 30k
]

def evaluate_on_random(factorizers: List[Factorizer], context={"dtype": tl.float64}):
    """
    Evalue the alogirhtm factorization_algorithm on images from skimage package

    Args:
        factorizers (List[Factorizers): List of Factorizers we want to evaluate
        context (dict): context for the tensors using tensorly
    """
    # testing on random data
    print("\nTesting on random generated tensors:")
    for tensor_data in random_tensors:
        dimension = tensor_data["dimensions"]
        F = random.randint(2, 5) # get random order between 2 and 5
        norm_of_tensor = random.uniform(1.0, 500.0) # get a random norm for our tensor
        noise_scaling = max(0, random.uniform(-0.1, 0.2))
        
        print(f"Dimension of tensor: {dimension}, noise: {noise_scaling}, F: {F}, norm: {norm_of_tensor}")
        
        tensor = random_cp_with_noise(dimension, F, noise_scaling=noise_scaling, context=context) # make it have no noise
        tensor = tensor * norm_of_tensor / tl.norm(tensor) # rescale the tensor
        # generate initial A_ns
        
        plt.figure()
        legend_handles = {} # Use a dictionary to store unique labels and handles
        alpha = max(1.0 - 0.2 * (tensor_data["nreps"] - 1) , 0.5)
        for _ in range(tensor_data["nreps"]):
            initial_A_ns = create_initial_data(tensor, F)
            for factorizer in factorizers:
                iteration_result = factorizer.factorize_cp(tensor, F, initial_A_ns)
            
                print(f"{factorizer.label} converged in {iteration_result.calculation_time:.3f} seconds and {len(iteration_result.reconstruction_errors)} iterations")
            
                if factorizer.label not in legend_handles:
                    # If not, plot with the label and store the handle
                    line, = plt.plot(iteration_result.reconstruction_errors, color=factorizer.color, label=factorizer.label, linestyle=factorizer.linestyle, alpha=alpha)
                    legend_handles[factorizer.label] = line
                else:
                    # If it is, plot without the label to avoid duplicates
                    plt.plot(iteration_result.reconstruction_errors, color=factorizer.color, linestyle=factorizer.linestyle, alpha=alpha)
                    
                #plt.plot(iteration_result.reconstruction_errors, color=factorizer.color, label=factorizer.label, linestyle=factorizer.linestyle, alpha=1.0/tensor_data["nreps"])
            
        plt.xlabel(iteration_label)
        plt.ylabel(error_label)
        # TODO das mal noch genauer anschauen, was da sinnvoll ist
        plt.yscale(yscale_convergence)
        #plt.xscale(xscale_convergence)
        plt.xscale(**xscale_convergence_data)
        plt.xlim(left=0)
        plt.legend(title='Algorithms', loc='upper right')
        plt.title(f"Tensor of Dimension {dimension} with {noise_scaling*100:.2f}% Noise")
        #plt.show()
        plt.savefig(f"{picture_folder}random_{dimension}_convergence.png", bbox_inches='tight')



# Which tensordata is usable:
# covid19: not non-negative so not usable
# IL2: needs masks but is ok otherwise
# indian pines: possible
# kinetic: not non-negative so not usable

# can also use "data/vaccine_tensor.npy" to get a integer tensor
def evaluate_on_data(factorizers: List[Factorizer], context={"dtype": tl.float64}, nrepetitions: int = 1):
    """
    Evalue the alogirhtm factorization_algorithm on images from skimage package

    Args:
        factorizers (List[Factorizers): List of Factorizers we want to evaluate
        context (dict): context for the tensors using tensorly
        nrepetitions: number of times we run the algorithms with random initial data.
    """
    # TODO IL2 needs mask
    # for each data tensor contains a dictionary with name, F and tensor
    data_tensors = []
    
    indian_pines = load_indian_pines()
    tensor_indian_pines = tl.tensor(indian_pines.tensor, **context)
    data_tensors.append({
        "name" : "indian_pines",
        "tensor" : tensor_indian_pines,
        "F" : 6
    })

    # TODO consider removing vaccine data, since it does not work well
    vaccine_data = np.load("data/vaccine_tensor.npy")
    tensor_vaccine = tl.tensor(vaccine_data, **context)
    #data_tensors.append({ "name" : "vaccine", "tensor" : tensor_vaccine, "F" : 4 })

    for data_tensor in data_tensors:
        name = data_tensor["name"]
        tensor = data_tensor["tensor"]
        F = data_tensor["F"]

        print(f"\nTesting on {name} data:")
        print(f"Tensor is of shape: {tensor.shape}")

        legend_handles = {} # Use a dictionary to store unique labels and handles
        alpha = max(1.0 - 0.2 * (nrepetitions - 1) , 0.5)
        plt.figure()
        for _ in range(nrepetitions): 
            # generate initial A_ns
            initial_A_ns = create_initial_data(tensor, F)
            for factorizer in factorizers:
                iteration_result = factorizer.factorize_cp(tensor, F, initial_A_ns)
                
                print(f"{factorizer.label} converged in {iteration_result.calculation_time:.3f} seconds and {len(iteration_result.reconstruction_errors)} iterations")
                
                if factorizer.label not in legend_handles:
                    # If not, plot with the label and store the handle
                    line, = plt.plot(iteration_result.reconstruction_errors, color=factorizer.color, label=factorizer.label, linestyle=factorizer.linestyle, alpha=alpha)
                    legend_handles[factorizer.label] = line
                else:
                    # If it is, plot without the label to avoid duplicates
                    plt.plot(iteration_result.reconstruction_errors, color=factorizer.color, linestyle=factorizer.linestyle, alpha=alpha)
                #plt.plot(iteration_result.reconstruction_errors, color=factorizer.color, label=factorizer.label, linestyle=factorizer.linestyle)
        plt.xlabel(iteration_label)
        plt.ylabel(error_label)
        # TODO das mal noch genauer anschauen, was da sinnvoll ist
        plt.yscale(yscale_convergence)
        #plt.xscale(xscale_convergence)
        plt.xscale(**xscale_convergence_data)
        plt.xlim(left=0)
        plt.legend(title='Algorithms', loc='upper right')
        plt.title(f"{name} data")
        #plt.show()
        plt.savefig(f"{picture_folder}data_{name}_convergence.png", bbox_inches='tight')



# all color images from skimage.data sorted by size
default_F = 3
image_names = [
    {"name": 'colorwheel', "F": 3}, # (370, 371, 3)
    {"name": 'cat', "F": 6}, # (300, 451, 3)
    {"name": 'coffee', "F": 3}, # (400, 600, 3)
    {"name": 'astronaut', "F": 6}, # (512, 512, 3)
    {"name": 'immunohistochemistry', "F": 6}, # (512, 512, 3)
    {"name": 'rocket', "F": 3}, # (427, 640, 3) maybe J=3?
    {"name": 'logo', "F": 6}, # (500, 500, 4)
    {"name": 'hubble_deep_field', "F": 6}, # (872, 1000, 3)
    {"name": 'retina', "F": 6}, # (1411, 1411, 3)
]

# TODO: I might want to change this to clamp instead of dividing by max
def to_image(tensor, clamp=False):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    if clamp:
        #im = np.clip(im, 0, 255)
        im /= im.max()
        im *= 255
    else:
        im -= im.min()
        im /= im.max()
        im *= 255
    return im.astype(np.uint8)

def evaluate_on_images(factorizers: List[Factorizer], context={"dtype": tl.float64}):
    """
    Evalue the alogirhtm factorization_algorithm on images from skimage package

    Args:
        factorizers (List[Factorizers): List of Factorizers we want to evaluate
        context (dict): context for the tensors using tensorly
    """
    # iterate over all images, use all factorizers. Display the approximations and the convergence lines
    for name in image_names:
        print(f"\nFactorizing image {name['name']} with F = {name['F']}")
        # get the image
        caller = getattr(data, name["name"])
        image = caller()
        tensor = tl.tensor(image, **context)

        F = name["F"]

        # generate initial A_ns
        initial_A_ns = create_initial_data(tensor, F)
        
        # Plot the resulting image and the convergence lines
        # TODO add feature, so that if we have more then 3 we use multiple rows
        fig, axes = plt.subplots(nrows=1, ncols=len(factorizers)+1, figsize=(20, 20))
        # create subplots for the convergence lines
        plt.figure()
        axes_index = 0 # index of the axes
        # original image
        axes[axes_index].set_title(name["name"])
        axes[axes_index].set_axis_off()
        if tensor.ndim == 2:
            axes[axes_index].imshow(to_image(tensor), cmap=plt.cm.gray)
        else:
            axes[axes_index].imshow(to_image(tensor))
        
        # iterate over all factorizers, let them run, save the data
        for factorizer in factorizers:
            axes_index += 1
            iteration_result = factorizer.factorize_cp(tensor, F, initial_A_ns)
            reconstruction = defactorizing_CP(iteration_result.A_ns, tensor.shape)

            # Show the individual components of the reconstruction
            show_individual_components(tensor, name['name'], factorizer_name=factorizer.label, J=F, result=iteration_result)

            print(f"{factorizer.label} converged in {iteration_result.calculation_time:.3f} seconds and {len(iteration_result.reconstruction_errors)} iterations")
            
            axes[axes_index].set_title(factorizer.label)
            axes[axes_index].set_xticks([])
            axes[axes_index].set_yticks([])
            if tensor.ndim == 2:
                axes[axes_index].imshow(to_image(reconstruction), cmap=plt.cm.gray)
            else:
                axes[axes_index].imshow(to_image(reconstruction))

            plt.plot(iteration_result.reconstruction_errors, color=factorizer.color, label=factorizer.label, linestyle=factorizer.linestyle)

        #fig.show()
        fig.savefig(f"{picture_folder}image_{name['name']}_reconstruction.png", bbox_inches='tight')

        plt.xlabel(iteration_label)
        plt.ylabel(error_label)
        # TODO das mal noch genauer anschauen, was da sinnvoll ist
        plt.yscale(yscale_convergence)
        #plt.xscale(xscale_convergence)
        plt.xscale(**xscale_convergence_data)
        plt.xlim(left=0)
        plt.legend(title='Algorithms', loc='upper right')
        plt.title(f"{name['name']}")
        #plt.show()
        plt.savefig(f"{picture_folder}image_{name['name']}_convergence.png", bbox_inches='tight')
        plt.close(fig)

        
def show_individual_components(tensor: tl.tensor, tensor_name: str, factorizer_name: str, J: int, result: IterationResult):
    fig, axes = plt.subplots(nrows=1, ncols=J+2, figsize=(20,20))
    axes[0].set_title("Original")
    axes[0].set_axis_off()
    axes[0].imshow(to_image(tensor))

    axes[1].set_title(factorizer_name)
    axes[1].set_axis_off()
    axes[1].imshow(to_image(defactorizing_CP(result.A_ns, tensor.shape)))

    for index in range(2,J+2):
        columns = []
        for A_n in result.A_ns:
            columns.append(A_n[:,index-2])
        rank1_tensor = tl.tenalg.outer(columns)
        axes[index].set_title(f"Component {index-1}")
        axes[index].set_xticks([])
        axes[index].set_yticks([])
        axes[index].imshow(to_image(rank1_tensor, clamp=True))
    
    fig.savefig(f"{picture_folder}image_{tensor_name}_{factorizer_name}_individual_factors.png", bbox_inches='tight')
    plt.close(fig)

def print_mean_and_variance(times, name):
    times = tl.tensor(times)
    mean = tl.mean(times)
    variance = tl.sum((times - mean) * (times - mean)) / len(times)
    print(f"Algorithm {name} took {mean} seconds on average with a variance of {variance}")
    

def plot_calculation_times_and_niter(factorizers: List[Factorizer]):
    """
    Plots the calculation times, calculation times per iteration and the number of iterations for all algorithms along all data they encountered.
    """
    # calculation times
    print("calculation times")
    plt.figure()
    for factorizer in factorizers:
        plt.plot(factorizer.get_calculation_times(), color=factorizer.color, label=factorizer.label, linestyle=factorizer.linestyle)
        print_mean_and_variance(factorizer.get_calculation_times(), factorizer.label)
    plt.legend(title='Algorithms', loc='upper right')
    plt.ylabel(time_label)
    plt.title("Calculation Times")
    plt.savefig(f"{picture_folder}calculation_times.png", bbox_inches='tight')
    # calculation times per iteration
    print("\ncalculation times per iteration") 
    plt.figure()
    for factorizer in factorizers:
        plt.plot(factorizer.get_calculation_times_per_iteration(), color=factorizer.color, label=factorizer.label, linestyle=factorizer.linestyle)
        print_mean_and_variance(factorizer.get_calculation_times_per_iteration(), factorizer.label)
    plt.legend(title='Algorithms', loc='upper right')
    plt.ylabel(time_label)
    plt.title("Calculation Times Per Iteration")
    plt.savefig(f"{picture_folder}calculation_times_per_iteration.png", bbox_inches='tight')
    # number of iterations
    print("\nnumber of iterations")
    plt.figure()
    for factorizer in factorizers:
        plt.plot(factorizer.get_number_of_iterations(), color=factorizer.color, label=factorizer.label, linestyle=factorizer.linestyle)
        print_mean_and_variance(factorizer.get_number_of_iterations(), factorizer.label)
    plt.legend(title='Algorithms', loc='upper right')
    plt.title("Number of Iterations")
    plt.savefig(f"{picture_folder}number_of_iterations.png", bbox_inches='tight')


def plot_trajectories(factorizers: List[Factorizer], tensor, F, indices, max_iter=1000):
    """
    Plots the trajectories the individual factorizers take.
    
    Args:
        factorizers (List[Factorizers): List of Factorizers we want to evaluate
        tensor: The tensor we want to factorize
        dimensions: the indices of the dimensions we want to plot. Has to be a list of 2 tuples that define the index of the tensor we want to plot
        max_iter: maximum number of iterations. Defaults to 1000
    """
    index0 = indices[0]
    index1 = indices[1]
    
    initial_A_ns = create_initial_data(tensor, F)
    # For each factorizer we store the matrix factors of the current step of the factorization
    A_ns = {}
    for factorizer in factorizers:
        A_ns[factorizer.label] = deepcopy(initial_A_ns)
    
    intital_approx = tl.to_numpy(defactorizing_CP(initial_A_ns, tensor.shape))

    # list of tuples (index0, index1) during the iteration for the 3 algorithms
    graphs = {}
    for factorizer in factorizers:
        graphs[factorizer.label] = [(intital_approx[index0], intital_approx[index1])]

    reconstruction_errors = {}
    for factorizer in factorizers:
        reconstruction_errors[factorizer.label] = [tl.to_numpy(tl.norm(tensor - defactorizing_CP(initial_A_ns, tensor.shape)) / tl.norm(tensor))]

    # achive this by always doing one iteration step from each algorithm
    for i in range(max_iter):
        for factorizer in factorizers:
            iteration_result = factorizer.algorithm(tensor, F, A_ns[factorizer.label], 1)
            approx = tl.to_numpy(defactorizing_CP(iteration_result.A_ns, tensor.shape))
            A_ns[factorizer.label] = iteration_result.A_ns
            graphs[factorizer.label].append( (approx[index0], approx[index1]) )
            #print(approx[index0])
            reconstruction_errors[factorizer.label].append(tl.to_numpy(iteration_result.reconstruction_errors[-1]))
         
    # Plot the trajectories of the individual factorizers
    plt.figure()
    for factorizer in factorizers:
        #print([e for e in graphs[factorizer.label]])
        plt.plot([e[0] for e in graphs[factorizer.label]], [e[1] for e in graphs[factorizer.label]], marker='.', label=factorizer.label, color=factorizer.color)
    
    # plot true solution
    plt.plot(tl.to_numpy(tensor[index0]), tl.to_numpy(tensor[index1]), 'o', label='solution', color='black')
    plt.legend(title='Algorithms')
    plt.savefig(f"{picture_folder}trajectories.png", bbox_inches='tight')

    plt.figure()
    for factorizer in factorizers:
        plt.plot(reconstruction_errors[factorizer.label], label=factorizer.label, color=factorizer.color)
    plt.xlabel(iteration_label)
    plt.ylabel(error_label)
    plt.yscale(yscale_convergence)
    plt.xscale(**xscale_convergence_data)
    plt.xlim(left=0)
    plt.legend(title='Algorithms', loc='upper right')
    plt.savefig(f"{picture_folder}trajectories_REs.png", bbox_inches='tight')
    



## Deprecated Use the individual once instead
def evaluate_algorithms(factorizers: List[Factorizer], context={"dtype": tl.float64}):
    """
    Evalue the alogirhtm factorization_algorithm on actual data.

    Args:
        factorizers (List[Factorizers): List of Factorizers we want to evaluate
        context (dict): context for the tensors using tensorly
    """

    # testing on random data
    print("\nTesting on random generated tensors:")
    for dimension in random_tensors:
        F = random.randint(2, 5) # get random order between 2 and 5
        norm_of_tensor = random.uniform(1.0, 500.0) # get a random norm for our tensor
        noise_scaling = max(0, random.uniform(-0.05, 0.2))
        
        print(f"Dimension of tensor: {dimension}, noise: {noise_scaling}, F: {F}, norm: {norm_of_tensor}")
        
        tensor = random_cp_with_noise(dimension, F, noise_scaling=0.0, context=context) # make it have no noise
        tensor = tensor * norm_of_tensor / tl.norm(tensor) # rescale the tensor
        # generate initial A_ns
        initial_A_ns = create_initial_data(tensor, F)
        
        plt.figure()
        for factorzer in factorizers:
            iteration_result = factorzer.factorize_cp(tensor, F, initial_A_ns)
            reconstruction = defactorizing_CP(iteration_result.A_ns, tensor.shape)
            
            print(f"{factorzer.label} converged in {iteration_result.calculation_time:.3f} seconds and {len(iteration_result.reconstruction_errors)} iterations")
            
            plt.plot(iteration_result.reconstruction_errors, color=factorzer.color, label=factorzer.label, linestyle=factorzer.linestyle)
            
        plt.xlabel(iteration_label)
        plt.ylabel(error_label)
        # TODO das mal noch genauer anschauen, was da sinnvoll ist
        plt.yscale(yscale_convergence)
        #plt.xscale(xscale_convergence)
        plt.xscale(**xscale_convergence_data)
        plt.xlim(left=0)
        plt.legend(title='Algorithms', loc='upper right')
        plt.title('Tensor of Dimension ' + str(dimension))
        #plt.show()
        plt.savefig(picture_folder+'random_'+str(dimension)+'_convergence.png')

    # TODO IL2 needs mask

    # test on indian pines data
    print("\nTesting on indian pines data:")
    indian_pines = load_indian_pines()
    tensor = tl.tensor(indian_pines.tensor, **context)
    print("Tensor is of shape: " + str(tensor.shape))
    F = 4 # TODO find good F here
    # generate initial A_ns
    initial_A_ns = create_initial_data(tensor, F)
    plt.figure()
    for factorzer in factorizers:
        iteration_result = factorzer.factorize_cp(tensor, F, initial_A_ns)
        reconstruction = defactorizing_CP(iteration_result.A_ns, tensor.shape)
        
        print(f"{factorzer.label} converged in {iteration_result.calculation_time:.3f} seconds and {len(iteration_result.reconstruction_errors)} iterations")
        
        plt.plot(iteration_result.reconstruction_errors, color=factorzer.color, label=factorzer.label, linestyle=factorzer.linestyle)
    plt.xlabel(iteration_label)
    plt.ylabel(error_label)
    # TODO das mal noch genauer anschauen, was da sinnvoll ist
    plt.yscale(yscale_convergence)
    #plt.xscale(xscale_convergence)
    plt.xscale(**xscale_convergence_data)
    plt.xlim(left=0)
    plt.legend(title='Algorithms', loc='upper right')
    plt.title('Indian Pines Data')
    #plt.show()
    plt.savefig(picture_folder+'data_indian_pines_convergence.png')


    # test on vaccines data
    print("\nTesting on vaccine data:")
    vaccine_data = np.load("data/vaccine_tensor.npy")
    tensor = tl.tensor(vaccine_data, **context)
    print("Tensor is of shape: " + str(tensor.shape))
    F = 4 # TODO find good F here
    # generate initial A_ns
    initial_A_ns = create_initial_data(tensor, F)
    plt.figure()
    for factorzer in factorizers:
        iteration_result = factorzer.factorize_cp(tensor, F, initial_A_ns)
        reconstruction = defactorizing_CP(iteration_result.A_ns, tensor.shape)
        
        print(f"{factorzer.label} converged in {iteration_result.calculation_time:.3f} seconds and {len(iteration_result.reconstruction_errors)} iterations")
        
        plt.plot(iteration_result.reconstruction_errors, color=factorzer.color, label=factorzer.label, linestyle=factorzer.linestyle)
    plt.xlabel(iteration_label)
    plt.ylabel(error_label)
    # TODO das mal noch genauer anschauen, was da sinnvoll ist
    plt.yscale(yscale_convergence)
    #plt.xscale(xscale_convergence)
    plt.xscale(**xscale_convergence_data)
    plt.xlim(left=0)
    plt.legend(title='Algorithms', loc='upper right')
    plt.title('Vaccine Data')
    #plt.show()
    plt.savefig(picture_folder+'data_vaccines_convergence.png')
    
    
    # test on images
    for name in image_names:
        print("\nFactorizing image "+name["name"]+"with F = "+str(name["F"]))
        # get the image
        caller = getattr(data, name["name"])
        image = caller()
        tensor = tl.tensor(image, **context)

        F = name["F"]

        # generate initial A_ns
        initial_A_ns = create_initial_data(tensor, F)
        
        # Plot the resulting image and the convergence lines
        fig, axes = plt.subplots(nrows=1, ncols=len(factorizers)+1, figsize=(20, 20))
        # create subplots for the convergence lines
        plt.figure()
        axes_index = 0 # index of the axes
        # original image
        axes[axes_index].set_title(name["name"])
        axes[axes_index].set_axis_off()
        if tensor.ndim == 2:
            axes[axes_index].imshow(to_image(tensor), cmap=plt.cm.gray)
        else:
            axes[axes_index].imshow(to_image(tensor))
        
        # iterate over all factorizers, let them run, save the data
        for factorzer in factorizers:
            axes_index += 1
            iteration_result = factorzer.factorize_cp(tensor, F, initial_A_ns)
            reconstruction = defactorizing_CP(iteration_result.A_ns, tensor.shape)

            print(f"{factorzer.label} converged in {iteration_result.calculation_time:.3f} seconds and {len(iteration_result.reconstruction_errors)} iterations")
            
            axes[axes_index].set_title(factorzer.label)
            axes[axes_index].set_xticks([])
            axes[axes_index].set_yticks([])
            if tensor.ndim == 2:
                axes[axes_index].imshow(to_image(reconstruction), cmap=plt.cm.gray)
            else:
                axes[axes_index].imshow(to_image(reconstruction))

            plt.plot(iteration_result.reconstruction_errors, color=factorzer.color, label=factorzer.label, linestyle=factorzer.linestyle)

        #fig.show()
        fig.savefig(picture_folder+'image_'+name["name"]+'_reconstruction.png', bbox_inches='tight')

        plt.xlabel(iteration_label)
        plt.ylabel(error_label)
        # TODO das mal noch genauer anschauen, was da sinnvoll ist
        plt.yscale(yscale_convergence)
        #plt.xscale(xscale_convergence)
        plt.xscale(**xscale_convergence_data)
        plt.xlim(left=0)
        plt.legend(title='Algorithms', loc='upper right')
        plt.title(name["name"]+" with F = "+str(name["F"]))
        #plt.show()
        plt.savefig(picture_folder+'image_'+name["name"]+'_convergence.png')
        plt.close(fig)

    # finally we how many iterations and how much time each algorithm took for each tensor
    # calculation times
    plt.figure()
    for factorizer in factorizers:
        plt.plot(factorizer.get_calculation_times(), color=factorzer.color, label=factorzer.label, linestyle=factorzer.linestyle)
    plt.legend(title='Algorithms', loc='upper right')
    plt.title("Calculation Times")
    plt.savefig(f"{picture_folder}calculation_times.png")
    plt.close(fig)
     # calculation times per iteration
    plt.figure()
    for factorizer in factorizers:
        plt.plot(factorizer.get_calculation_times_per_iteration(), color=factorzer.color, label=factorzer.label, linestyle=factorzer.linestyle)
    plt.legend(title='Algorithms', loc='upper right')
    plt.title("Calculation Times")
    plt.savefig(f"{picture_folder}calculation_times_per_iteration.png")
    plt.close(fig)   
    # number of iterations
    plt.figure()
    for factorizer in factorizers:
        plt.plot(factorizer.get_number_of_iterations(), color=factorzer.color, label=factorzer.label, linestyle=factorzer.linestyle)
    plt.legend(title='Algorithms', loc='upper right')
    plt.title("Calculation Times")
    plt.savefig(f"{picture_folder}number_of_iterations.png")

# TODO add saving of data