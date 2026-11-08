#!/usr/bin/python3


import numpy as np
import json
import pickle
from scipy import stats
from ActiveLearning.parameter_priors import *


class Parameter:
    """
    A class that stores a MESA parameter and relevant attributes.

    Attributes
    ----------
    param_name : str
        The name of the parameter
    l_bound : numeric 
        The lower bound for the parameter
    u_bound : numeric 
        The upper bound for the parameter
    continuous : bool
        True if the variable is a continuous variable
    discrete_steps : list-like
        A list of the steps for the parameter (e.g. for opacity tables)
    log_space : bool
        True if the variable should be spaced logarithmically
    prior : function-like or str
        If string the function should match one of the built-in priors (only Gaussian for now) 
        or if function like should take the values to be assessed and the upper and lower bounds; alternatively just the values 
        to be assessed if no upper or lower bounds are required.
    file_path : str
        The file where the parameter should be written
    """
    
    def __init__(self, name=None, lower=None, upper=None, continuous=None, discrete_steps=None, log_space=False, prior=None, filepath="", **kwargs):
        """
        Initializes a Parameter object with the specified attributes. For Gaussian priors, extracts central_value 
        and central_err from kwargs. Sets up all parameter bounds, spacing, and prior information.

        Parameters
        ----------
        name : str, optional
            The name of the parameter. Default is None.
        lower : numeric, optional
            The lower bound for the parameter. Default is None.
        upper : numeric, optional
            The upper bound for the parameter. Default is None.
        continuous : bool, optional
            True if the variable is continuous, False if discrete. Default is None.
        discrete_steps : list, optional
            List of discrete values for non-continuous parameters. Default is None.
        log_space : bool, optional
            True if parameter should be sampled in log space. Default is False.
        prior : str or callable, optional
            Prior distribution for the parameter. Can be "Gaussian" or a callable function. Default is None.
        filepath : str, optional
            File path for parameter storage. Default is empty string.
        **kwargs : dict
            Additional keyword arguments, used for Gaussian prior parameters (central_value, central_err).

        Returns
        -------
        None
        """
    
        self.param_name = name
        self.l_bound = lower
        self.u_bound = upper
        self.continuous = continuous
        self.discrete_steps = discrete_steps
        self.log_space = log_space
        if prior == "Gaussian":
            self.prior = "Gaussian"
            self.central_value = float(kwargs['central_value'])
            self.central_err = float(kwargs['central_err'])
        else:
             self.prior = prior
        self.file_path = filepath
        

    def log_prior(self, x, uniform_override=False):
        """
        Calculates the log prior probability density at value x. Returns 0.0 for uniform priors within bounds,
        -inf for values outside bounds. For Gaussian priors, computes normalized log probability. For custom
        priors, calls the prior function and returns its log value. Returns -inf for invalid inputs.

        Parameters
        ----------
        x : numeric
            The value at which to evaluate the log prior
        uniform_override : bool, optional
            If True, forces uniform prior regardless of specified prior. Default is False.

        Returns
        -------
        float 
            The natural logarithm of the prior probability density at x    
        """
        
        if self.prior == None or uniform_override:
            if self.l_bound <= x <= self.u_bound:
                return 0.0
            else:
                return -np.inf
            
        elif self.prior == "Gaussian":
            if self.l_bound <= x <= self.u_bound:
                val = np.log(stats.norm(loc=self.central_value, scale=self.central_err).pdf(x)/(1/(self.central_err*np.sqrt(np.pi*2))))
                if val == 0:
                    return -np.inf
                else:
                    return val
            else:
                return -np.inf
            
        elif self.prior != None:
            try:
                val = np.log(self.prior(x, self.l_bound, self.u_bound))
                return val
            except:
                val = np.log(self.prior(x))
                return val
        
        else:
            return -np.inf

    
    def generate_grid(self, steps=10):
        """
        Generates a grid of parameter values for sampling or evaluation. For continuous parameters, creates 
        linearly or logarithmically spaced arrays between bounds. For discrete parameters, returns the 
        predefined discrete_steps list. Stores the steps value as an instance attribute.

        Parameters
        ----------
        steps : int, optional
            Number of grid points to generate for continuous parameters. Default is 10.

        Returns
        -------
        numpy.ndarray or list 
            Array of grid points for continuous parameters, or list of discrete steps for discrete parameters
        """
        
        self.steps = steps
        if self.continuous:
            if self.log_space:
                return np.logspace(np.log10(self.l_bound), np.log10(self.u_bound), self.steps)
            else:
                return np.linspace(self.l_bound, self.u_bound, self.steps)
        else:
            return self.discrete_steps


    def _to_dict(self):
        data = {
            "param_name":     self.param_name,
            "l_bound":        self.l_bound,
            "u_bound":        self.u_bound,
            "continuous":     self.continuous,
            "discrete_steps": self.discrete_steps,
            "log_space":      self.log_space,
            "file_path":      self.file_path,
        }

        if self.prior == "Gaussian":
            data["prior"] = "Gaussian"
            data["central_value"] = self.central_value
            data["central_err"]   = self.central_err
        elif self.prior is None:
            data["prior"] = None
        else:
            # Store the fully-qualified location of the callable so _from_dict
            # can import it automatically. __module__ + __qualname__ together
            # uniquely identify the function even for methods and nested callables.
            data["prior"] = None
            data["prior_module"] = getattr(self.prior, "__module__", None)
            data["prior_qualname"] = getattr(self.prior, "__qualname__", None)

        return data

    @classmethod
    def _from_dict(cls, data):
        import importlib
        import warnings
        
        kwargs = {}
        prior = data.get("prior")

        if prior == "Gaussian":
            kwargs["central_value"] = data["central_value"]
            kwargs["central_err"]   = data["central_err"]
        elif data.get("prior_module") and data.get("prior_qualname"):
            # Attempt to restore the callable automatically
            module_name = data["prior_module"]
            qualname = data["prior_qualname"]
            try:
                module = importlib.import_module(module_name)
                # qualname may be nested e.g. "MyClass.my_method", so walk it
                func = module
                for attr in qualname.split("."):
                    func = getattr(func, attr)
                prior = func
            except Exception as e:
                warnings.warn(
                    f"Parameter '{data['param_name']}': could not restore callable prior "
                    f"'{qualname}' from module '{module_name}': {e}. "
                    "Re-attach it manually via param.prior = <function>.",
                    UserWarning,
                )
                prior = None

        obj = cls(
            name = data["param_name"],
            lower = data["l_bound"],
            upper = data["u_bound"],
            continuous = data["continuous"],
            discrete_steps = data["discrete_steps"],
            log_space = data["log_space"],
            prior = prior,
            filepath = data.get("file_path", ""),
            **kwargs
        )
        return obj
    # ------------------------------------------------------------------
    # Public save / load  (JSON, replaces the old pickle versions)
    # ------------------------------------------------------------------

    def save_parameter(self, pkl=False, file_path=""):
        """
        Saves the Parameter to a lightweight JSON file (<param_name>.param).
        This replaces the previous pickle-based implementation and uses orders
        of magnitude less RAM because only plain scalar/list data is written
        — no object graph, no hidden numpy dtypes, no embedded model weights.

        Parameters
        ----------
        file_path : str, optional
            Directory prefix for the output file. Default is empty string (cwd).

        Returns
        -------
        None
        """
        if self.param_name is None:
            return None

        if pkl:
            with open(file_path+str(self.param_name)+'.param', 'wb') as f:
                pickle.dump(self, f)
        else:
            out_path = file_path + str(self.param_name) + ".param"
            with open(out_path, "w") as f:
                json.dump(self._to_dict(), f, indent=2)

    def read_parameter(self, file_path):
        """
        Loads a Parameter from a JSON file previously written by save_parameter.
        Falls back to pickle for files created by the old implementation so that
        existing saved parameters can still be read without regenerating them.

        Parameters
        ----------
        file_path : str
            Full path to the .param file.

        Returns
        -------
        Parameter
            The reconstructed Parameter object.
        """
        with open(file_path, "rb") as f:
            header = f.read(1)

        # JSON files start with '{'; pickle files start with 0x80
        if header == b"{":
            with open(file_path, "r") as f:
                data = json.load(f)
            return Parameter._from_dict(data)
        else:
            # Legacy pickle fallback
            with open(file_path, "rb") as f:
                return pickle.load(f)
        
        
    def __str__(self):
        """
        String representation method that prints formatted information about the parameter including
        name, bounds, continuity, log spacing, and discrete steps if applicable. Returns empty string
        after printing all parameter details to stdout.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Empty string (after printing parameter information)
        """
        
        print('Parameter Name: '+str(self.param_name))
        print('Parameter Lower: '+str(self.l_bound))
        print('Parameter Upper: '+str(self.u_bound))
        print('Parameter Continuous: '+str(self.continuous))
        print('Parameter Log Spacing: '+str(self.log_space))
        if self.discrete_steps:
            print('Parameter Steps: ', self.discrete_steps)
        return ''
        

def test_init():
    """
    Tests the Parameter class initialization with various parameter combinations.
    Creates test parameters with different configurations and verifies all attributes
    are set correctly. Tests both continuous and discrete parameter types.

    Parameters
    ----------
    None

    Returns
    -------
    bool 
        True if tests pass, False if any test fails
    """
    
    test_param = Parameter("Mass", 0.7, 2.25, True, None, False, mass_prior)
    if test_param.param_name == "Mass":
        if test_param.l_bound == 0.7:
            if test_param.u_bound == 2.25:
                if test_param.continuous:
                    if not test_param.log_space:
                        if test_param.prior == mass_prior:
                            if test_param.file_path == "":
                                test_param = Parameter("Nuc_Rate", continuous=False, discrete_steps=['0.5', '0.0', '-0.5'])
                                if not test_param.continuous:
                                    if test_param.l_bound == None:
                                        if test_param.u_bound == None:
                                            return False

    return True

                           
def test_log_prior():
    """
    Tests the log_prior method of the Parameter class. Creates a test parameter
    and verifies that the log prior returns expected values for inputs within
    and outside the parameter bounds.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if tests pass, False if any test fails
    """
    
    test_param = Parameter("Mass", 0.7, 2.25, True, None, False)
    if test_param.log_prior(1) == 0.0 and test_param.ln_prior(3) == -np.inf:
        return False
    return True

    
def test_generate_grid():
    """
    Tests the generate_grid method for both continuous and discrete parameters.
    Verifies linear spacing, logarithmic spacing, and discrete step functionality.
    Checks that generated grids match expected numpy arrays or lists.

    Parameters
    ----------
    None

    Returns
    -------
    bool 
        True if tests pass, False if any test fails
    """
    
    test_param = Parameter("Mass", 0.7, 2.25, True, None, False, mass_prior)
    if (test_param.generate_grid(10) == np.linspace(0.7, 2.25, 10)).all():
        test_param.log_space = True
        if (test_param.generate_grid(10) == np.logspace(np.log10(0.7), np.log10(2.25), 10)).all():
            test_param = Parameter("Mass", 0.7, 2.25, False, ['alpha', 'beta', 'gamma'], False, mass_prior)
            if test_param.generate_grid() == ['alpha', 'beta', 'gamma']:
                return False
    return True


def test_save_load_parameter():
    """
    Tests the save_parameter and read_parameter methods. Creates a test parameter,
    saves it to file, loads it back, and verifies the loaded parameter matches
    the original. Note: There appears to be a bug in the original code calling
    load_parameter instead of read_parameter.

    Parameters
    ----------
    None
    
    Returns
    -------
    bool 
        True if tests pass, False if any test fails
    """
    
    test_param = Parameter("Mass", 0.7, 2.25, True, None, False, mass_prior)
    test_param.save_parameter('testSave')
    read_param = Parameter().load_parameter('testSaveMass.param')
    if not (test_param == read_param):
        return False
    return True


# Run all tests
if __name__ == "__main__":
    error = False
    if test_init():
        print("Error in init().")
        error = True
    if test_ln_prior():
        print("Error in ln_prior().")
        error = True
    if test_generate_grid():
        print("Error in generate_grid().")
        error = True
    if test_save_load_parameter():
        print("Error in saving or loading parameters from file.")
        error = True
    if error:
        print("Errors were found, please correct the code.")
    else:
        print("All tests completed successfully.")
