"""
This file contains all classes used in the jupyter notebooks.
"""

import tensorly as tl

from dataclasses import dataclass, field
from typing import List, Callable

# TODO add methods for loading and saving the dataclasses

@dataclass
class IterationResult:
    """ data class that holds the plotting information for one algorithm execution. Should only be used as a part of SolverData
    
    Attributes
    ----------
    reconstruction_errors (list): List or array of the reconstruction errors
    calculation_time (float): Total time it took to run the algorithm
    A_ns: final approximation factors
    
    """
    reconstruction_errors: tl.tensor
    calculation_time: float
    A_ns: List


@dataclass
class Factorizer:
    """ data class that holds the plotting results of all executions for one algorithm
    
    Attributes
    ----------
    label (str): label for the plot
    algorithm (Callable): the actual factorization algorithm. Takes a tensor, an integer and initial_A_ns as input and has an IterantionResult as output
    args (List): List of arguments for the solver.
    color (str): color for the plot. Defaults to 'red'.
    linestyle (str): linestyle for the plot. Defaults to 'solid'.
    data (List): list of IterationResults for the individual executions of the algorithm
    
    """
    label: str
    algorithm: Callable # input tensor, int and initial_A_ns, output IterationResult
    #args: List = field(default_factory=list)
    color: str = 'red'
    linestyle: str = 'solid'
    data: List[IterationResult] = field(default_factory=list) 

    # If args was not given, then default it to [self.label]
    #def __post_init__(self):
    #    if not self.args:
    #        self.args = [self.label]

    def factorize_cp(self, tensor, F, initial_A_ns) -> IterationResult:
        """ Runs the algorithm and saves the result and returns it as well
        """
        result = self.algorithm(tensor, F, initial_A_ns)
        self.add_data(result)
        return result
    
    def add_data(self, iteration_result):
        self.data.append(iteration_result)
    
    def get_calculation_times(self):
        """returns a list of the calculation times of all IterationResults in data
        """
        return [ir.calculation_time for ir in self.data]

    def get_number_of_iterations(self):
        """returns a list of the calculation times of all IterationResults in data
        """
        return [len(ir.reconstruction_errors)-1 for ir in self.data]

    def average_calculation_time(self):
        calc_times = self.get_calculation_times()
        return sum(calc_times)/len(calc_times)

    def get_calculation_times_per_iteration(self):
        """ If there where no iterations use 0
        """
        return [ir.calculation_time/(len(ir.reconstruction_errors)-1) if len(ir.reconstruction_errors) > 1 else 0 for ir in self.data]
