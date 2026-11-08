#!/usr/bin/python3


import json
import pickle
from scipy.stats import qmc
from ActiveLearning.parameter import *
from copy import deepcopy
import pickle
from utils import *


class ParameterSet:
    """
    A class that manages collections of MESA parameters providing functionality for 
    parameter sampling, data generation, and prior probability calculations across 
    multiple parameters simultaneously.

    Attributes
    ----------
    parameters : dict
        Dictionary storing Parameter objects with parameter names as keys

    Methods
    -------
    __init__(parameters={})
        Initialize ParameterSet with optional parameter dictionary
    specify_parameters(parameters)
        Set parameters from a list of Parameter objects
    add_parameters(parameters)
        Add new parameters to existing parameter set
    generate_datapoints(spacing="grid", param_step_dict={}, num_samples=None, seed=None)
        Generate parameter combinations using grid or Latin Hypercube sampling
    save_parameter_set(file_path, parameter_set_name="")
        Save ParameterSet to file
    read_parameter_set(file_path)
        Load ParameterSet from file
    read_params_from_file(file_list=[])
        Load multiple parameters from individual parameter files
    collective_prior(datapoint, uniform_override=False)
        Calculate combined log prior probability for parameter combination
    group_midpoints()
        Calculate midpoint values for all parameters
    num_parameters()
        Return number of parameters in set
    param_names()
        Return list of parameter names
    __str__()
        String representation of parameter set
    """

   
    def __init__(self, parameters={}):
        """
        Initializes a ParameterSet object with the provided parameters dictionary. Creates a container
        for managing multiple MESA parameters that can be used together for sampling, prior calculation,
        and data generation in active learning applications.

        Parameters
        ----------
        parameters : dict, optional
            Dictionary of Parameter objects with parameter names as keys. Default is empty dict.

        Returns
        -------
        None            
        """
        
        self._parameters = parameters

        
    def specify_parameters(self, parameters):
        """
        Replaces the current parameter set with a new set of parameters. Clears any existing parameters
        and creates a new dictionary using the parameter names as keys and Parameter objects as values.
        This method completely overwrites the existing parameter collection.

        Parameters
        ----------
        parameters : list
            List of Parameter objects to be added to the set

        Returns
        -------
        None
        """       
        self._parameters = {}        
        for param in parameters:
            self._parameters[param.param_name] = param


    def add_parameter(self, param):
        """
        Adds new parameters to the existing parameter set without removing current parameters.
        Uses parameter names as dictionary keys. If a parameter with the same name already exists,
        it will be overwritten with the new Parameter object.

        Parameters
        ----------
        parameters : list
            Parameter object to add to the existing set

        Returns
        -------
        None
        """
        self._parameters[param.param_name] = param
        return None
    
        
    def generate_datapoints(self, num_samples=None, steps_per_dim=None, spacing="grid", param_step_dict=None, seed=None):
        """
        Generates parameter value combinations using either grid sampling or Latin Hypercube sampling.
        Grid sampling creates all possible combinations of parameter values. LHC sampling generates
        space-filling samples with proper handling of continuous, discrete, and log-spaced parameters.
        Supports discrete parameters by mapping continuous samples to discrete values.

        Parameters
        ----------
        spacing : str, optional
            Sampling method, either "grid" or "LHC" (Latin Hypercube). Default is "grid".
        param_step_dict : dict, optional
            Dictionary specifying number of steps for each continuous parameter. Default is empty dict.
        num_samples : int, optional
            Number of samples for LHC sampling. Default is None.
        seed : int, optional
            Random seed for LHC sampling reproducibility. Default is None.

        Returns
        -------
        list or numpy.ndarray
            Deep copy of generated parameter combinations as coordinate arrays
        
        Notes
        -----
        This method currently does not support categorical variables. For these it is currently advised to simply create a class for each 
        value of the categorical variable pending completion of this upgrade.
        """
        # TODO: Fix for Categorical Parameters


        if spacing == "grid":
            if param_step_dict is None and num_samples is None and steps_per_dim is None:
                raise ValueError("Either num_samples and ratio_of_steps or param_step_dict must be passed")

            elif param_step_dict is None and num_samples is not None and steps_per_dim is not None:
                param_step_dict = {}

                for param, steps in zip(self, steps_per_dim):
                    param_step_dict[param.param_name] = steps

            param_dict = {}
            for param in self:
                if param.continuous:
                    param_dict[param.param_name] = param.generate_grid(param_step_dict[param.param_name])
                else:
                    param_dict[param.param_name] = param.generate_grid()

            data = recursive_pairing(param_dict)

                
        elif spacing == "LHC":
            sampler = qmc.LatinHypercube(d=len(self._parameters), optimization="random-cd", seed=seed)

            sample = sampler.random(num_samples)
            
            u_bounds = []
            l_bounds = []

            discrete = []
            log_spacing = []
            
            for param in self:
                if param.continuous == False:
                    u_bounds.append(len(param.discrete_steps))
                    l_bounds.append(0)
                    discrete.append(True)
                else:
                    u_bounds.append(param.u_bound)
                    l_bounds.append(param.l_bound)
                    discrete.append(False)

                log_spacing.append(param.log_space)
                if param.log_space:
                    u_bounds[-1] = np.log10(u_bounds[-1])
                    l_bounds[-1] = np.log10(l_bounds[-1])


            scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
            key_list = list(self._parameters)
            
            for i in range(len(self._parameters)):
                if discrete[i]:
                    scaled_sample[:, i] = np.floor(scaled_sample[:, i]).astype(int)
                    for j in range(len(self._parameters[key_list[i]].discrete_steps)):
                        scaled_sample[:, i][scaled_sample[:, i] == j] = self._parameters[key_list[i]].discrete_steps[j]
                        
                if log_spacing[i]:
                    scaled_sample[:, i] = 10**scaled_sample[:, i]

                    
            data = scaled_sample[:]
        else:
            data = None

            
        return deepcopy(data)


    # ------------------------------------------------------------------
    # Save / load  (JSON, replaces the old pickle versions)
    # ------------------------------------------------------------------

    def save_parameter_set(self, file_path, pkl=False, parameter_set_name=""):
        """
        Saves the ParameterSet to a lightweight JSON file (<parameter_set_name>.param_set).
        Each Parameter is serialized via Parameter._to_dict(), so only plain scalars and
        lists are written — the file contains no object graph and no numpy dtype overhead.
        This replaces the previous pickle-based implementation.

        Parameters
        ----------
        file_path : str
            Directory prefix for the output file.
        parameter_set_name : str, optional
            Base name for the output file. Default is empty string.

        Returns
        -------
        None
        """
        if len(self._parameters) == 0:
            return None

        if pkl:
            with open(file_path+parameter_set_name+'.param_set', 'wb') as f:
                pickle.dump(self, f)
        else:
            data = {name: param._to_dict() for name, param in self._parameters.items()}
            out_path = file_path + parameter_set_name + ".param_set"
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)

        return None


    def read_parameter_set(file_path):
        """
        Static method that loads a ParameterSet from a JSON file previously written by
        save_parameter_set. Falls back to pickle for files created by the old implementation
        so that existing saved parameter sets can still be read without regenerating them.

        Parameters
        ----------
        file_path : str
            Full path to the .param_set file.

        Returns
        -------
        ParameterSet
            The reconstructed ParameterSet object.
        """
        with open(file_path, "rb") as f:
            header = f.read(1)

        # JSON files start with '{'; pickle files start with 0x80
        if header == b"{":
            with open(file_path, "r") as f:
                raw = json.load(f)
                params = {name: Parameter._from_dict(d) for name, d in raw.items()}
            return ParameterSet(parameters=params)
        else:
            # Legacy pickle fallback
            with open(file_path, "rb") as f:
                return pickle.load(f)


    def read_params_from_file(self, file_list = []):
        """
        Loads multiple Parameter objects from individual parameter files and adds them to the
        parameter set. Each file should contain a single Parameter object. Uses the parameter
        name from each loaded Parameter as the dictionary key.

        Parameters
        ----------
        file_list : list, optional
            List of file paths to parameter files to be loaded. Default is empty list.

        Returns
        -------
        None
        """

        
        for filename in file_list:
            temp = Parameter().read_parameter(filename)
            self._parameters[temp.param_name] = temp

        return None


    def collective_prior(self, datapoint, uniform_override=False):
        """
        Calculates the collective log prior probability for a parameter combination by summing
        individual parameter log priors. The datapoint order must match the parameter key order.
        Returns the total log probability, which can be used in Bayesian inference applications.

        Parameters
        ----------
        datapoint : array-like
            Parameter values in the same order as parameters were added
        uniform_override : bool, optional
            If True, forces uniform priors for all parameters. Default is False.

        Returns
        -------
        float 
            Sum of log prior probabilities across all parameters
        """

        
        logp = 0
        for param in self:
            logp += param.log_prior(datapoint[i], uniform_override=uniform_override)
        return logp
    

    def group_midpoints(self):
        """
        Calculates midpoint values for all parameters in the set. For finite bounds, uses the
        arithmetic mean of upper and lower bounds. For parameters with one infinite bound,
        uses the finite bound value. For parameters with both infinite bounds, uses zero.
        Returns values in the same order as parameter keys.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Array of midpoint values for each parameter
        """

        
        pos = np.zeros((self.num_parameters()))
        for param in self)):
            if np.isfinite(param.l_bound) and np.isfinite(param.u_bound):
                pos[i] = param.l_bound + (param.u_bound - param.l_bound) / 2
            elif np.isfinite(param.l_bound) and not np.isfinite(param.u_bound):
                pos[i] = param.l_bound
            elif not np.isfinite(param.l_bound) and np.isfinite(param.u_bound):
                pos[i] = param.u_bound    
            else:
                pos[i] = 0
        return pos


    def get_bounds(self):
        """


        """
        bounds = []
        for param in self:
            temp = []
            if np.isfinite(param.l_bound):
                temp.append(param.l_bound)
            else:
                temp.append(None)

            if np.isfinite(param.u_bound):
                temp.append(param.u_bound)
            else:
                temp.append(None)
                bounds.append(temp)

        bounds = np.array(bounds)
        return bounds


    def num_parameters(self):
        """
        Returns the total count of parameters currently stored in the parameter set.
        Useful for determining dimensionality for sampling and analysis operations.

        Parameters
        ----------
        None

        Returns
        -------
        int 
            Number of parameters in the parameter set
        """
        return len(self._parameters)


    def num_continuous_parameters(self):
        """


        """
        count = 0
        for param in self:
            if param.continuous:
                count += 1
        return count



    def get_param_names(self):
        """
        Returns a list of parameter names from all Parameter objects in the set. Note: There
        appears to be a bug in the original code - it accesses param.name instead of 
        param.param_name, which should cause an AttributeError.

        Parameters
        ----------
        None

        Returns
        -------
        list
            List of parameter names
        """        
        names = []
        for param in self:
            names.append(param.param_name)
        return names


    def __str__(self):
        """
        String representation method that prints information for all parameters in the set
        by calling each Parameter object's __str__ method. Returns a tab character after
        printing all parameter details to stdout.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Tab character (after printing parameter information)            
        """
        for param in self:
            print(param)
        return "\t"


    def __iter__(self):
        """


        """
        return iter(self._parameters)
    
                
def test_init():
    """
    Test function for ParameterSet initialization. Loads three parameter files (mass, helium,
    metallicity) and creates a ParameterSet with them. Prints the parameter set information
    for verification. Used for testing basic initialization functionality.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    
    param1 = Parameter().read_parameter("mass.param")
    param2 = Parameter().read_parameter("helium.param")
    param3 = Parameter().read_parameter("metallicity.param")
    # if test_set.parameters == [param1, param2, param3]:
    test_set = ParameterSet({param1.param_name:param1, param2.param_name:param2, param3.param_name:param3})
    print(test_set)
    

def test_specify_parameters():
    """
    Test function for specify_parameters method. Creates empty ParameterSet and tests
    adding parameters through specify_parameters method with different parameter types.
    Tests parameter replacement functionality and saves result for verification.
    Includes TODO note indicating incomplete testing.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    
    # TODO: This function does not introduce significant untested code. Test will be finished later
    test_set = ParameterSet()
    test_set.specify_parameters([Parameter("Mass", 0.7, 2.25, True, False, False, mass_prior), Parameter("Nuc_Rate", continuous=False)])

    print(test_set)
    
    param_2 = Parameter("Nuc_Rate", continuous=False, discrete_steps=['0.5', '0.0', '0.3'])
    param_3 = Parameter().read_parameter("helium.param")

    print(test_set)
    
    test_set.specify_parameters([param_2, param_3])

    print(test_set)
    test_set.save_parameter_set("test")
    return None


def test_add_parameter():
    """
    Placeholder test function for add_parameter method. Currently returns None with TODO
    note indicating that testing is incomplete and will be finished later. This function
    does not currently test any functionality.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    
    # TODO: This function does not introduce significant untested code. Test will be finished later
    return None


def test_generate_datapoints():
    """
    Test function for generate_datapoints method. Creates ParameterSet from file, adds
    discrete parameter, tests both grid and LHC sampling methods. Demonstrates parameter
    loading, mixed parameter types, and different sampling strategies. Prints results
    for verification of sampling functionality.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """


    test_class = ParameterSet()
    test_class.read_params_from_file(file_list=["mass.param", "helium.param", "metallicity.param"])
    test_class.add_parameters([Parameter("nuclear", continuous=False, discrete_steps=['0.5', '0.0', '0.3'])])

    data = test_class.generate_datapoints(spacing="grid", param_step_dict={"mass": 2, "helium":1, "metallicity":3})

    print(test_class)
    
    print(data)

    print("----------LHC----------")

    data = test_class.generate_datapoints(spacing="LHC", param_step_dict={"mass": 2, "helium":3, "metallicity":3}, num_samples=30)

    print(data)
    return None


if __name__ == "__main__":
    test_init()
    print("-----------------------")
    test_specify_parameters()
    print("-----------------------")
    test_generate_datapoints()
    print("-----------------------")
