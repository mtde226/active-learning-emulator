#!/bin/bash/python3


import inspect

def kwargs_to_args(func, kwargs):
    """
    
    
    """
    bound = inspect.signature(func).bind(**kwargs)
    bound.apply_defaults()
    args = bound.args()
    return args


def recursive_pairing(parameter_dictionary, coord_list=None):
    """
    Recursively generates all possible combinations of parameter values from a dictionary
    of parameter arrays. Creates a Cartesian product of all parameter value sets. Handles
    single parameter case, two parameter base case, and recursive combination for multiple
    parameters. Used internally by generate_datapoints for grid sampling.

    Parameters
    ----------
    parameter_dictionary : dict
        Dictionary of parameter names and their value arrays
    coord_list : list, optional
        Current list of coordinate combinations for recursion. Default is None.

    Returns
    -------
    list: List of coordinate arrays representing all parameter value combinations            
    """
    
    if coord_list is None and len(parameter_dictionary.keys())==1:
        key1 = list(parameter_dictionary.keys())[0]
        values1 = parameter_dictionary.pop(key1)
        coordinates = [[x] for x in values1]
        return coordinates
    elif coord_list is None:
        key1 = list(parameter_dictionary.keys())[0]
        key2 = list(parameter_dictionary.keys())[1]
        values1 = parameter_dictionary.pop(key1)
        values2 = parameter_dictionary.pop(key2)
        coordinates = [[x, y] for x in values1 for y in values2]
        if len(parameter_dictionary.keys()):
            return recursive_pairing(parameter_dictionary, coord_list=coordinates)
        else:
            return coordinates
    else:
        key1 = list(parameter_dictionary.keys())[0]
        values1 = parameter_dictionary.pop(key1)
        coordinates = [[*y, x] for y in coord_list for x in values1]
        if len(parameter_dictionary.keys()):
            return recursive_pairing(parameter_dictionary, coord_list=coordinates)
        else:
            return coordinates
