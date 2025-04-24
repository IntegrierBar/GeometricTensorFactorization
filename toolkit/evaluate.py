"""
This file contains the functions to evalute algorithms and generate convergence plots for both real and generated data.
"""

import tensorly as tl
import numpy as np
import random
import matplotlib.pyplot as plt

from typing import List

from skimage import data

from .classes import (Factorizer)
from .constants import (
    folder, data_folder,
    error_label, iteration_label, tensor_dimension_label, time_label,
    xscale_convergence_data, xscale_convergence, yscale_convergence
) 
from ..tensorfactorization.utils import (create_initial_data, defactorizing_CP, random_cp_with_noise)
from ..data.data_imports import (load_indian_pines)


# Which tensordata is usable:
# covid19: not non-negative so not usable
# IL2: needs masks but is ok otherwise
# indian pines: possible
# kinetic: not non-negative so not usable

# can also use "data/vaccine_tensor.npy" to get a integer tensor

dimensions = [
    (10,10,10),
    (20,20,5),
    (10,10,10,5),
    (10,10,5,5,5),
    (100,100,3),
]


# all color images from skimage.data sorted by size
default_F = 3
image_names = [
    {"name": 'colorwheel', "F": default_F}, # (370, 371, 3)
    {"name": 'cat', "F": 4}, # (300, 451, 3)
    {"name": 'coffee', "F": 4}, # (400, 600, 3)
    {"name": 'astronaut', "F": 5}, # (512, 512, 3)
    {"name": 'immunohistochemistry', "F": 5}, # (512, 512, 3)
    {"name": 'rocket', "F": 4}, # (427, 640, 3)
    {"name": 'logo', "F": 4}, # (500, 500, 4)
    {"name": 'hubble_deep_field', "F": 5}, # (872, 1000, 3)
#    {"name": 'skin', "F": default_F}, # (960, 1280, 3)
#    {"name": 'lily', "F": default_F}, # (922, 922, 4)
    {"name": 'retina', "F": 5}, # (1411, 1411, 3)
]

#image_names = [
#    {"name": 'colorwheel', "F": default_F}, # (370, 371, 3)
#    {"name": 'cat', "F": default_F}, # (300, 451, 3)
#]

max_iter = 2000

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)


def evaluate_algorithms(factorizers: List[Factorizer], context={"dtype": tl.float64}):
    """
    Evalue the alogirhtm factorization_algorithm on actual data.

    Args:
        factorizers (List[Factorizers): List of Factorizers we want to evaluate
        context (dict): context for the tensors using tensorly
        algorithm_args (dict): additional arguments for the algorithm
    """

    # testing on random data
    print("\nTesting on random generated tensors:")
    for dimension in dimensions:
        F = random.randint(2, 5) # get random order between 2 and 5
        norm_of_tensor = random.uniform(1.0, 500.0) # get a random norm for our tensor
        noise_scaling = max(0, random.uniform(-0.05, 0.2))
        
        print("Dimension of tensor: " + str(dimension) + ", noise: " + str(noise_scaling) + ", F: " + str(F) + ", norm: " + str(norm_of_tensor))
        
        tensor = random_cp_with_noise(dimension, F, noise_scaling=0.0, context=context) # make it have no noise
        tensor = tensor * norm_of_tensor / tl.norm(tensor) # rescale the tensor
        # generate initial A_ns
        initial_A_ns = create_initial_data(tensor, F)
        
        plt.figure()
        for factorzer in factorizers:
            iteration_result = factorzer.factorize_cp(tensor, F, initial_A_ns)
            reconstruction = defactorizing_CP(iteration_result.A_ns, tensor.shape)
            
            print(factorzer.label + " converged in " + str(iteration_result.calculation_time) + "seconds and " + str(len(iteration_result.reconstruction_errors)) + " iterations")
            
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
        plt.savefig(folder+'random_'+str(dimension)+'_convergence.png')

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
        
        print(factorzer.label + " converged in " + str(iteration_result.calculation_time) + "seconds and " + str(len(iteration_result.reconstruction_errors)) + " iterations")
        
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
    plt.savefig(folder+'data_indian_pines_convergence.png')


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
        
        print(factorzer.label + " converged in " + str(iteration_result.calculation_time) + "seconds and " + str(len(iteration_result.reconstruction_errors)) + " iterations")
        
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
    plt.savefig(folder+'data_vaccines_convergence.png')
    
    
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

            print(factorzer.label + " converged in " + str(iteration_result.calculation_time) + " seconds and " + str(len(iteration_result.reconstruction_errors)) + " iterations")
            
            axes[axes_index].set_title(factorzer.label)
            axes[axes_index].set_xticks([])
            axes[axes_index].set_yticks([])
            if tensor.ndim == 2:
                axes[axes_index].imshow(to_image(reconstruction), cmap=plt.cm.gray)
            else:
                axes[axes_index].imshow(to_image(reconstruction))

            plt.plot(iteration_result.reconstruction_errors, color=factorzer.color, label=factorzer.label, linestyle=factorzer.linestyle)

        #fig.show()
        fig.savefig(folder+'image_'+name["name"]+'_reconstruction.png', bbox_inches='tight')

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
        plt.savefig(folder+'image_'+name["name"]+'_convergence.png')
        plt.close(fig)

# TODO add saving of data