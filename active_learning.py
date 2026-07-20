#!/usr/bin/python3

import numpy as np
import random
from scipy import stats, signal, integrate

import os, json, zipfile, tempfile, pickle
from PTMCMCSampler import PTMCMCSampler
from ActiveLearning.mesa_training import *
from ActiveLearning.parameter_set import *
from copy import deepcopy
import pickle

class ActiveLearning:
    """
    Orchestrates an active-learning loop for stellar parameter inference.

    Wraps a MESA-trained ML model (classifier and/or regressor) together with
    a parameter prior and observational data to drive iterative MCMC sampling.
    At each iteration the sampler is guided by an acquisition heuristic that
    balances classification uncertainty and regression improvement, producing
    new candidate stellar models for subsequent MESA evaluation.

    Attributes
    ----------
    MESA_parameters : ParameterSet or None
        Prior parameter ranges and distributions.
    ML_Data : MESATraining or None
        Normalised dataset, classifier, regressor, and residual interpolators.
    observational_data : pd.DataFrame or None
        Observational targets used in the likelihood.
    classify : bool
        Whether to use the classifier in the heuristic.
    regress : bool
        Whether to use the regressor in the heuristic.
    num_dim : int or None
        Dimensionality of the parameter space.
    init_pos : array-like
        Initial position for the MCMC sampler.
    covariance : np.ndarray or None
        Proposal covariance matrix for the MCMC sampler.
    burnin : int
        Number of burn-in steps (default 5000).
    num_samples : int
        Total number of MCMC samples to draw (default 10000).
    output_directory : str
        Directory where sampler output files are written.
    obs_data_cols : list of str or None
        Observational data column names.
    obs_err_cols : list of str or None
        Observational uncertainty column names.
    data_prior_scaling : float
        Scale factor applied to the data prior term.
    obs_likelihood_scaling : float
        Scale factor applied to the observational likelihood.
    classify_scaling : float
        Scale factor applied to the classifier heuristic.
    regress_scaling : float
        Scale factor applied to the regressor heuristic.
    track_error : bool
        If True, residual ML errors are recorded during sampling.
    error_values : list
        Accumulated residual errors when track_error is True.
    """
    def __init__(self, classify, regress, output_directory="", directory=None, **kwargs):
        """
        Initialise an ActiveLearning instance, optionally loading state from disk
        or accepting pre-built data objects via keyword arguments.

        Parameters
        ----------
        classify : bool
            Enable the classifier component of the acquisition heuristic.
        regress : bool
            Enable the regressor component of the acquisition heuristic.
        output_directory : str, optional
            Directory for sampler output files (default current directory).
        directory : str or None, optional
            Path to a saved .active_learner archive to restore; if provided,
            all other state is loaded from that file.
        **kwargs
            Param_Dir : str
                Path to a .param_set file; loaded via import_MESA_Parameters.
            Param_Data : ParameterSet
                Pre-built ParameterSet (used directly, skips file I/O).
            ML_Dir : str
                Path to a .training_set archive; loaded via import_ML_Data.
            ML_Data : MESATraining
                Pre-built MESATraining object (used directly, skips file I/O).
            Obs_Dir : str
                Path to an observational data file; loaded via import_Observational_Data.
            pos : array-like
                Explicit initial MCMC position vector.
            cov : np.ndarray
                Explicit proposal covariance matrix.
            nburn : int
                Number of burn-in steps (overrides default of 5000).
            nwalkers : int
                Number of parallel MCMC walkers.
            nsamples : int
                Total number of MCMC samples (overrides default of 10000).
            neffective : int
                Target number of effective samples.
            track_error : bool
                If True, record ML residual errors during sampling.
        """
        self.MESA_parameters = None # A parameter set
        self.ML_Data = None # The final data
        self.observational_data = None 
        self.classify = classify
        self.regress = regress
        self.num_dim = None
        self.init_pos = None
        self.covariance = None
        self.burnin = 5000
        self.num_samples = 10000
        # self.num_effective = 10000
        self.output_directory = output_directory
        self.obs_data_cols = None
        self.obs_err_cols = None
        self.data_prior_scaling = 1
        self.obs_likelihood_scaling = 1
        self.classify_scaling = 1
        self.regress_scaling = 1
        self.track_error=False
        
       
        if directory != None:
            self.load_class(filepath=directory)

        if 'Param_Dir' in kwargs.keys():
            self.import_MESA_Parameters(kwargs['Param_Dir'])

        if 'Param_Data' in kwargs.keys():
            self.MESA_parameters = kwargs['Param_Data']
            self.num_dim = self.MESA_parameters.num_parameters()
            
        if 'ML_Dir' in kwargs:
            self.import_ML_Data(kwargs['ML_Dir'])
       
        if 'ML_Data' in kwargs.keys():
            self.ML_Data = kwargs['ML_Data']
            try:
                self.covariance = np.cov(self.ML_Data.dataset.query('Class == 0').sample(frac=0.1)[self.ML_Data.parameter_columns].to_numpy())
            except:
                matrix = np.zeros((len(self.ML_Data.parameter_columns), len(self.ML_Data.parameter_columns)))
                for i in range(0, len(self.ML_Data.parameter_columns)):
                    matrix[i, i] = np.var(self.ML_Data.dataset[self.ML_Data.parameter_columns[i]])
                    matrix[i-1, i] = self.ML_Data.dataset[self.ML_Data.parameter_columns[i]].cov(self.ML_Data.dataset[self.ML_Data.parameter_columns[i-1]])
                    matrix[i, i-1] = matrix[i, i-1]
                self.covariance = copy.deepcopy(matrix)
                
        if 'Obs_Dir' in kwargs:
            self.import_Observational_Data(kwargs['Obs_Dir'])

        if 'pos' in kwargs:
            self.init_pos = kwargs['pos']
        elif self.MESA_parameters:
            self.init_pos = self.ML_Data.normalise_data(data_point=self.MESA_parameters.group_midpoints(), parameters=True)
        else:
            self.init_pos = []

        if 'cov' in kwargs:
            self.covariance = kwargs['cov']
            # elif calculate from parameterset            
        elif self.num_dim != None:
            self.covariance = np.eye(self.num_dim)
            
        if 'nburn' in kwargs:
            self.burniin = kwargs['nburn']

        if 'nwalkers' in kwargs:
            self.num_walkers = kwargs['nwalkers']

        if 'nsamples' in kwargs:
            self.num_samples = kwargs['nsamples']

        if 'neffective' in kwargs:
            self.num_effective = kwargs['neffective']

        if 'track_error' in kwargs:
            self.track_error = kwargs['track_error']

        if self.track_error:
            self.error_values = []

    def import_MESA_Parameters(self, data_path):
        """
        Load a ParameterSet from a .param_set file and store it on the instance.

        Also sets self.num_dim to the number of parameters in the loaded set.

        Parameters
        ----------
        data_path : str
            Path to the .param_set file to read.

        Returns
        -------
        None
        """
        self.MESA_parameters = Parameter_Set().read_parameter_set(data_path)
        self.num_dim = MESA_parameters.num_parameters()
        

    def import_ML_Data(self, data_path):
        """
        Load a MESATraining dataset from a .training_set archive and store it
        on the instance.

        Also computes and stores the parameter covariance matrix from the full
        (unnormalised) dataset.

        Parameters
        ----------
        data_path : str
            Path to the .training_set archive to read.

        Returns
        -------
        None
        """
        self.ML_Data = MESA_Training.read_MESA_Training(filepath=data_path)
        self.covariance = np.cov(self.ML_Data.dataset[self.ML_Data.parameter_columns].to_numpy())        

    def import_Observational_Data(self, observational_columns, error_columns, data=None, data_path=None, read=False, pandas_kwargs={}):
        """
        Set the observational data used in likelihood evaluations.

        Parameters
        ----------
        observational_columns : list of str
            Column names identifying the observed quantities of interest.
        error_columns : list of str
            Column names identifying the measurement uncertainties.
        data : pd.DataFrame or None, optional
            Pre-loaded DataFrame to use directly (ignored when read=True).
        data_path : str or None, optional
            CSV file path; required when read=True.
        read : bool, optional
            If True, read the DataFrame from data_path via pd.read_csv.
            If False (default), use the data argument directly.
        pandas_kwargs : dict, optional
            Additional keyword arguments forwarded to pd.read_csv when read=True.

        Returns
        -------
        None
        """
        self.obs_data_cols = observational_columns
        self.obs_err_cols = error_columns
        if read:
            self.observational_data = pd.read_csv(data_path, names=self.obs_data_cols, **pandas_kwargs)
        else:
            self.observational_data = data.copy(deep=True)

   
    def save_ActiveLearning(self, ActiveLearning_name, pkl=False, file_path=""):
        """
        Saves the ActiveLearning object to a zip archive
        (<ActiveLearning_name>.active_learner) by delegating each heavy
        component to its own efficient serializer:
 
        * ML_Data        (MESATraining)  → nested .training_set archive
                                        (Parquet + joblib + Keras SavedModel)
        * MESA_parameters (ParameterSet) → JSON .param_set file
        * All scalar / list attributes   → JSON meta file
    
        Compared to pickling the whole ActiveLearning object, this avoids
        loading every numpy array inside every sklearn tree and every TensorFlow
        variable into RAM simultaneously.  Each component is deserialized
        independently, so peak memory during load equals the size of the largest
        single component, not the sum of all components.
    
        Parameters
        ----------
        ActiveLearning_name : str
            Base name for the output archive (no extension needed).
        file_path : str, optional
            Directory in which to write the archive.  Default is cwd.
    
        Returns
        -------
        None
        """
        
        if pkl:
            with open(file_path+ActiveLearning_name+'.active_learner', 'wb'):
                pkl.dump(self, f)

        else:
            archive_path = os.path.join(file_path, ActiveLearning_name + ".active_learner")
    
            with tempfile.TemporaryDirectory() as tmp:
    
                # ── 1. ML_Data (MESATraining) ─────────────────────────────────────
                # save_MESA_Training already produces a self-contained zip archive;
                # we just embed that archive inside our outer zip.
                if self.ML_Data is not None:
                    self.ML_Data.save_MESA_Training("ML_Data", file_path=tmp + os.sep)
    
                # ── 2. MESA_parameters (ParameterSet) ─────────────────────────────
                if self.MESA_parameters is not None:
                    self.MESA_parameters.save_parameter_set(tmp + os.sep, "MESA_parameters")
    
                # ── 3. observational_data DataFrame ───────────────────────────────
                if self.observational_data is not None:
                    self.observational_data.to_parquet(
                        os.path.join(tmp, "observational_data.parquet"), index=True
                    )
    
                # ── 4. Scalar / list metadata → JSON ─────────────────────────────
                def _safe(v):
                    """Return v if JSON-serializable, else None."""
                    try:
                        json.dumps(v)
                        return v
                    except (TypeError, ValueError):
                        return None
    
                meta = {
                    "classify":               self.classify,
                    "regress":                self.regress,
                    "num_dim":                self.num_dim,
                    "init_pos":               _safe(
                        self.init_pos.tolist()
                        if hasattr(self.init_pos, "tolist")
                        else self.init_pos
                    ),
                    "covariance":             _safe(
                        self.covariance.tolist()
                        if hasattr(self.covariance, "tolist")
                        else self.covariance
                    ),
                    "burnin":                 self.burnin,
                    "num_samples":            self.num_samples,
                    "output_directory":       self.output_directory,
                    "obs_data_cols":          self.obs_data_cols,
                    "obs_err_cols":           self.obs_err_cols,
                    "data_prior_scaling":     self.data_prior_scaling,
                    "obs_likelihood_scaling": self.obs_likelihood_scaling,
                    "classify_scaling":       self.classify_scaling,
                    "regress_scaling":        self.regress_scaling,
                    "track_error":            self.track_error,
                    "error_values":           _safe(
                        getattr(self, "error_values", None)
                    ),
                }
    
                with open(os.path.join(tmp, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
    
                # ── 5. Pack everything into a single zip ──────────────────────────
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(tmp):
                        for file in files:
                            abs_path = os.path.join(root, file)
                            arc_name = os.path.relpath(abs_path, tmp)
                            zf.write(abs_path, arc_name)
 
 
    def read_ActiveLearning(file_path):
        """
        Static method that loads an ActiveLearning object from an archive written
        by the new save_ActiveLearning.  Falls back to pickle for archives created
        by the old implementation.
    
        Parameters
        ----------
        file_path : str
            Full path to the .active_learner archive.
    
        Returns
        -------
        ActiveLearning
            The reconstructed ActiveLearning object.
        """

    
        # ── Detect format: zip (new) vs pickle (old) ─────────────────────────
        with open(file_path, "rb") as f:
            magic = f.read(4)
    
        if magic != b"PK\x03\x04":          # not a zip → old pickle
            with open(file_path, "rb") as f:
                return pickle.load(f)
    
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(tmp)
    
            # ── 1. Metadata ───────────────────────────────────────────────────
            with open(os.path.join(tmp, "meta.json")) as f:
                meta = json.load(f)
    
            # Construct with the minimal positional args; we overwrite everything
            obj = ActiveLearning(
                classify=meta["classify"],
                regress=meta["regress"],
                output_directory=meta.get("output_directory", ""),
            )
    
            obj.num_dim               = meta["num_dim"]
            obj.burnin                = meta["burnin"]
            obj.num_samples           = meta["num_samples"]
            obj.obs_data_cols         = meta["obs_data_cols"]
            obj.obs_err_cols          = meta["obs_err_cols"]
            obj.data_prior_scaling    = meta["data_prior_scaling"]
            obj.obs_likelihood_scaling = meta["obs_likelihood_scaling"]
            obj.classify_scaling      = meta["classify_scaling"]
            obj.regress_scaling       = meta["regress_scaling"]
            obj.track_error           = meta["track_error"]
            obj.error_values          = meta.get("error_values") or []
    
            raw_pos = meta.get("init_pos")
            obj.init_pos = np.array(raw_pos) if raw_pos is not None else []
    
            raw_cov = meta.get("covariance")
            obj.covariance = np.array(raw_cov) if raw_cov is not None else None
    
            # ── 2. ML_Data (MESATraining) ─────────────────────────────────────
            ml_archive = os.path.join(tmp, "ML_Data.training_set")
            if os.path.exists(ml_archive):
                from ActiveLearning.mesa_training import MESATraining
                obj.ML_Data = MESATraining.read_MESA_Training(ml_archive)
            else:
                obj.ML_Data = None
    
            # ── 3. MESA_parameters (ParameterSet) ─────────────────────────────
            ps_path = os.path.join(tmp, "MESA_parameters.param_set")
            if os.path.exists(ps_path):
                from ActiveLearning.parameter_set import ParameterSet
                obj.MESA_parameters = ParameterSet.read_parameter_set(ps_path)
            else:
                obj.MESA_parameters = None
    
            # ── 4. observational_data DataFrame ───────────────────────────────
            obs_path = os.path.join(tmp, "observational_data.parquet")
            obj.observational_data = (
                pd.read_parquet(obs_path) if os.path.exists(obs_path) else None
            )
    
        return obj
            
    def classifier_Heuristic(self, theta, method=None, factor=1):
        """
        Compute a classification-based acquisition score for a parameter point.

        Measures how uncertain the classifier is about the class of theta; higher
        scores indicate regions where the classifier is least confident.

        Parameters
        ----------
        theta : array-like of shape (n_parameters,)
            Normalised parameter point to evaluate.
        method : str or None, optional
            Scoring strategy:
            - None (default): 1 minus the maximum class probability.
            - 'new_class': corrected score that down-weights the mean probability
            of non-dominant classes.
            - 'softmax': applies a softmax transform before scoring.
        theta : numeric
            A factor used to scale the heuristic function.
            
        Returns
        -------
        float or None
            Acquisition score, or None if no classifier is available.
        """
        score = None
        
        if self.ML_Data.classifier:
            prediction = self.ML_Data.classifier(np.reshape(np.array(theta), (1, len(theta))))            
            # factor = np.shape(prediction)[1] / (np.shape(prediction)[1] - 1)
            if method == None:
                score_index = np.argmax(prediction[0])
                score = (1 - prediction[0, score_index])
            elif method == "new_class":
                score_index = np.argmax(prediction[0])
                score = factor * (1 - prediction[0, score_index] - np.mean(np.delete(prediction, score_index)))
            elif method == "softmax":
                softmax_value = np.exp(prediction[0]) / np.sum(np.exp(prediction[0]))
                score_index = np.argmax(softmax_value)
                score = factor * (1 - softmax_value[score_index])
            # print("CLASSIFIER SCORE: ", score)
            return score
        else:
            return None


    def regressor_Heuristic(self, theta, regressor_index=0, method="EI", **kwargs):
        """
        Compute a regression-based acquisition score using a Bayesian optimisation
        heuristic (Expected Improvement, Probability of Improvement, or Sigma).

        Uses Monte-Carlo dropout over `permutations` forward passes to estimate the
        predictive mean and standard deviation of the regressor at theta.

        Parameters
        ----------
        theta : np.ndarray of shape (n_parameters,)
            Normalised parameter point to evaluate.
        regressor_index : int, optional
            Index of the regressor to use (currently unused; kept for API
            compatibility).
        method : {'EI', 'PI', 'Sigma'}
            Acquisition function:
            - 'EI'    : Expected Improvement over the current prediction.
            - 'PI'    : Probability of Improvement (log-scale).
            - 'Sigma' : Mean normalised standard deviation across outputs.
        **kwargs
            permutations : int
                Number of stochastic forward passes for uncertainty estimation
                (default 30).
            obs_err : float
                Target observational error used to normalise the Sigma score
                (default 1).
            xi_ei : float
                Exploration-exploitation trade-off parameter for EI (default 0).

        Returns
        -------
        float or None
            Acquisition score, or None if no ML data or regressor is available.

        Raises
        ------
        ValueError
            If method is not one of 'EI', 'PI', or 'Sigma'.
        """
        if method not in ['EI', 'PI', 'Sigma']:
            raise ValueError("Invalid value for method parameter")
            return None
        
        if not self.ML_Data:
            return None

        if not self.ML_Data.regressor:
            return None

        if 'permutations' in kwargs:
            num_permutations = kwargs['permutations']
        else:
            num_permutations = 30
            
        if 'obs_err' in kwargs:
            accepted_error = kwargs['obs_err']
        else:
            accepted_error = 1

        if 'xi_ei' in kwargs:
            xi_ei = kwargs['xi_ei']
        else:
            xi_ei = 0

        score = 0
        
        if method == "EI":
            data  = np.reshape(np.asarray(theta.tolist()*num_permutations), (num_permutations, len(self.ML_Data.parameter_columns)))
            theta_prediction = np.zeros((len(self.ML_Data.regression_columns)))
            for i in range(len(self.ML_Data.regression_columns)):
                theta_prediction[i] = self.ML_Data.regressor[self.ML_Data.regression_columns[i]](np.reshape(theta, (1, len(self.ML_Data.parameter_columns))))
            if self.track_error:
                errors = np.zeros(len(self.ML_Data.regression_columns))
                if self.ML_Data.interpolator_type == "random_forest":
                    for i in range(len(self.ML_Data.regression_columns)):
                        errors[i] = self.ML_Data.residual_interpolator[self.ML_Data.regression_columns[i]].predict(np.reshape(self.ML_Data.denormalise_point(theta, parameters=True), (1, len(self.ML_Data.parameter_columns))))
                elif self.ML_Data.interpolator_type == "dnn":
                    for i in range(len(self.ML_Data.regression_columns)):
                        errors[i] = self.ML_Data.residual_interpolator[self.ML_Data.regression_columns[i]](np.reshape(self.ML_Data.denormalise_point(theta, parameters=True), (1, len(self.ML_Data.parameter_columns))))
                self.error_values.append(errors)
            group_prediction = np.zeros((num_permutations, len(self.ML_Data.regression_columns)))
            for i in range(len(self.ML_Data.regression_columns)):
                group_prediction[:, i] = self.ML_Data.regressor[self.ML_Data.regression_columns[i]](data, training=True).numpy().reshape((num_permutations,))

            denorm_theta = self.ML_Data.denormalise_regression_predictions(theta_prediction)
            denorm_group = self.ML_Data.denormalise_regression_predictions(group_prediction)
            mu_group, std_group = np.mean(denorm_group, axis=0), np.std(denorm_group, axis=1)
            for i in range(len(self.ML_Data.regression_columns)):
                numerator = mu_group[i] - denorm_theta[i] - xi_ei
                score += numerator * stats.norm.cdf(numerator / std_group[i]) + std_group[i] * stats.norm.pdf(numerator / std_group[i])
                
            
        elif method == "PI":
            data  = np.reshape(np.asarray(theta.tolist()*num_permutations), (num_permutations, len(self.ML_Data.parameter_columns)))
            theta_prediction = self.ML_Data.regressor(np.reshape(theta, (1, len(self.ML_Data.parameter_columns))))
            group_prediction = self.ML_Data.regressor(data, training=True).numpy()
            denorm_theta = self.ML_Data.denormalise_regression_predictions(theta_prediction)[0]
            denorm_group = self.ML_Data.denormalise_regression_predictions(group_prediction)
            mu_group, std_group = np.mean(denorm_group, axis=1), np.std(denorm_group, axis=1)
            for i in range(len(self.regression_columns)):
                score += np.log(stats.norm.cdf((mu_group[i] - denorm_theta[i])/std_group[i]))
            
        elif method == "sigma":
            data  = np.reshape(np.asarray(theta.tolist()*num_permutations), (num_permutations, len(self.ML_Data.parameter_columns)))
            prediction = self.ML_Data.regressor(data, training=True).numpy()
            denorm_prediction = self.ML_Data.denormalise_regression_predictions(prediction)
            std_devs = np.std(prediction, axis=0)
            score = np.mean(std_devs / accepted_error)

        return score


    def evaluate_Heuristic(self, theta, active_class, classify=False, regress=False, classifier_method=None, regressor_method=None, regressor_kwargs={}, class_scale=1, regress_scale=1.0):
        """
        Combine classifier and regressor acquisition scores into a single heuristic
        value used to guide MCMC sampling toward informative regions.

        Parameters
        ----------
        theta : np.ndarray of shape (n_parameters,)
            Normalised parameter point to evaluate.
        active_class : ActiveLearning
            The ActiveLearning instance whose heuristic methods are called.
        classify : bool, optional
            If True, include the classification heuristic score (default False).
        regress : bool, optional
            If True, include the regression heuristic score (default False).
        classifier_method : str or None, optional
            Method forwarded to classifier_Heuristic (default None).
        regressor_method : str or None, optional
            Method forwarded to regressor_Heuristic; if None, regression score is
            not computed even when regress=True.
        regressor_kwargs : dict, optional
            Additional keyword arguments forwarded to regressor_Heuristic.
        class_scale : float, optional
            Weight applied to the log classification score (default 1).
        regress_scale : float, optional
            Weight applied to the log regression score (default 1.0).

        Returns
        -------
        float or None
            Combined log-heuristic score, -inf if a zero score is encountered, or
            None if neither classify nor regress is True.
        """
        if classify:
            class_score = active_class.classifier_Heuristic(theta, method=classifier_method)
        if regress:
            if regressor_method is not None:
                regress_score = active_class.regressor_Heuristic(theta, method=regressor_method, **regressor_kwargs)                
                
        if classify and regress:
            if class_score == 0:
                class_score = -np.inf
            else:
                class_score = np.log(class_score)
            if regress_score == 0:
                regress_score = -np.inf
            else:
                regress_score = np.log(regress_score)
                heuristic_score = class_score*class_scale + regress_score*regress_scale
            return heuristic_score
        elif classify:
            if class_score == 0:
                return -np.inf
            else:
                return np.log(class_score)
        elif regress:
            return np.log(regress_score)
        return None
    

    def collective_Data_Prior(self, theta, active_class=None, prior=True, location=False, location_function=None, prior_kwargs={}, location_kwargs={}, correction=None, correction_kwargs={}):
        """
        Accumulate the total log-prior used as the MCMC 'logp' function.

        Optionally combines a parameter-space prior with a data-location likelihood
        term, allowing the sampler to be steered toward observationally consistent
        regions even during the active-learning (non-final) phase.

        Parameters
        ----------
        theta : array-like of shape (n_parameters,)
            Normalised parameter point to evaluate.
        active_class : ActiveLearning or None, optional
            Instance whose log_Prior and log_Data_Likelihood methods are called.
        prior : bool, optional
            If True (default), add the parameter prior via active_class.log_Prior.
        location : bool, optional
            If True, add the observational likelihood via
            active_class.log_Data_Likelihood.
        location_function : callable or None, optional
            Likelihood function forwarded to log_Data_Likelihood when location=True.
        prior_kwargs : dict, optional
            Additional keyword arguments forwarded to log_Prior.
        location_kwargs : dict, optional
            Additional keyword arguments forwarded to log_Data_Likelihood.
        correction : callable or None, optional
            Optional correction function forwarded to log_Data_Likelihood.
        correction_kwargs : dict, optional
            Additional keyword arguments forwarded to the correction function.

        Returns
        -------
        float
            Combined log-prior (and optionally log-likelihood) value.
        """
        collective_data_prior = 0
        if prior:
            collective_data_prior += active_class.log_Prior(theta, **prior_kwargs)
        if location:
            collective_data_prior += active_class.log_Data_Likelihood(theta, likelihood_function=location_function, likelihood_function_kwargs=location_kwargs, correction_function=correction, correction_function_kwargs=correction_kwargs)
        return collective_data_prior


    def log_Prior(self, theta, classify=False, accepted_classes=None, scaling_factor=1, uniform_override=False):
        """
        Evaluate the log-prior probability of a normalised parameter point.

        Optionally gates the prior with a hard classifier rejection: if the
        predicted class is not in accepted_classes, returns -inf immediately.

        Parameters
        ----------
        theta : array-like of shape (n_parameters,)
            Normalised parameter point.
        classify : bool, optional
            If True, run the classifier and reject points outside accepted_classes
            (default False).
        accepted_classes : list of int or None, optional
            Class indices considered physically valid; required when classify=True.
        scaling_factor : float, optional
            Multiplicative factor applied to the log-prior value (default 1).
        uniform_override : bool, optional
            If True, treat all parameter priors as uniform regardless of their
            configured distributions (default False).

        Returns
        -------
        float
            Scaled log-prior value, or -inf if the point is out of bounds or
            rejected by the classifier.
        """
        if classify:
            result_class = np.argmax(self.ML_Data.classifier(np.asarray([theta])))
            if result_class not in accepted_classes:
                return -np.inf

        prior = self.MESA_parameters.collective_prior(self.ML_Data.denormalise_point(theta, parameters=True), uniform_override=uniform_override)
        
        if prior == -np.inf:
            return -np.inf
        log_prior = prior
        return log_prior * scaling_factor
    
            
    def log_Data_Likelihood(self, theta, likelihood_function, likelihood_function_kwargs={}, correction_function=None, correction_function_kwargs={}, uncertainty_method="parameterisation"):
        """
        Evaluate the observational log-likelihood at a normalised parameter point.

        Runs the regressor over all output columns, de-normalises predictions,
        optionally applies an ML uncertainty model and a correction function,
        then delegates to the supplied likelihood_function.

        Parameters
        ----------
        theta : array-like of shape (n_parameters,)
            Normalised parameter point.
        likelihood_function : callable
            Function with signature
            (theta, ml_outputs, obs_data, obs_data_cols, obs_err_cols, **kwargs)
            that returns a log-likelihood scalar.
        likelihood_function_kwargs : dict, optional
            Extra keyword arguments forwarded to likelihood_function.
        correction_function : callable or None, optional
            Post-processing function applied to de-normalised ML outputs before
            passing them to likelihood_function.
        correction_function_kwargs : dict, optional
            Extra keyword arguments forwarded to correction_function.
        uncertainty_method : str, optional
            Method passed to apply_regression_uncertainty for quantifying ML
            prediction uncertainty (default 'parameterisation').

        Returns
        -------
        float or None
            Log-likelihood value, or None if no regressor is available.
        """
        log_likelihood = 0
                
        if self.ML_Data.regressor is not None:
            output_values = np.zeros((len(self.ML_Data.regression_columns)))
            for i in range(len(output_values)):
                output_values[i] = self.ML_Data.regressor[self.ML_Data.regression_columns[i]](np.reshape(theta, (1, len(self.ML_Data.parameter_columns))))
            output_values = self.ML_Data.denormalise_regression_predictions(output_values)
            output_values = self.ML_Data.apply_regression_uncertainty(output_values, uncertainty_method)
            if correction_function is not None:
                if correction_function_kwargs is not None:
                    output_values = correction_function(output_values, **correction_function_kwargs)
                else:
                    output_values = correction_function(output_values)

            log_likelihood = likelihood_function(theta, output_values, self.observational_data, self.obs_data_cols, self.obs_err_cols, **likelihood_function_kwargs)
            # print("LOG LIKELIHOOD: ", log_likelihood)
            return log_likelihood
        else:
            return None      

    
    def run_PT_sampler(self, final=False, prior_kwargs={}, likelihood_kwargs={}, likelihood_func=None):
        """
        Initialise and run a Parallel-Tempering MCMC sampler (PTMCMCSampler).

        In active-learning mode (final=False) the sampler uses evaluate_Heuristic
        as its log-likelihood and collective_Data_Prior as its log-prior, guiding
        new simulations toward informative parameter regions.

        In final mode (final=True) the sampler uses the supplied likelihood_func
        and log_Prior, performing a conventional posterior inference run.

        Parameters
        ----------
        final : bool, optional
            If False (default), run the acquisition heuristic sampler.
            If True, run a final posterior sampler using likelihood_func.
        prior_kwargs : dict, optional
            Keyword arguments forwarded to the log-prior function.
        likelihood_kwargs : dict, optional
            Keyword arguments forwarded to the log-likelihood function.
        likelihood_func : callable or None, optional
            Observational likelihood function; required when final=True.

        Returns
        -------
        None
            Returns None early if MESA_parameters is unset, or if final=True and
            no likelihood_func is provided.
        """
        save_freq = int(self.num_samples // 100)
        thinning = 1
        if self.MESA_parameters is None:
            return None
        elif final:
            if likelihood_func is None:
                print("Error no likelihood function assigned")
                return None
            self.sampler = PTMCMCSampler.PTSampler(self.num_dim, logl=likelihood_func, logp=self.log_Prior, cov=self.covariance, loglkwargs=likelihood_kwargs, logpkwargs=prior_kwargs, outDir=self.output_directory)
            self.sampler.sample(p0=self.init_pos, Niter=self.num_samples, burn=self.burnin, isave=save_freq, thin=thinning)
        else:
            self.sampler = PTMCMCSampler.PTSampler(self.num_dim, logl=self.evaluate_Heuristic, logp=self.collective_Data_Prior, cov=self.covariance, loglkwargs=likelihood_kwargs, logpkwargs=prior_kwargs, outDir=self.output_directory)
            self.sampler.sample(p0=self.init_pos, Niter=self.num_samples, burn=self.burnin, isave=save_freq, thin=thinning)


    def evaluate_chains(self, method="random", num_samples=1, chains_dir=None, **kwargs):
        """
        Extract new candidate parameter points from the MCMC chain for the next
        active-learning iteration.

        Parameters
        ----------
        method : {'random', 'kde_individual', 'kde_ndim'}, optional
            Sampling strategy:
            - 'random' (default): draw num_samples points at random from each
            marginal chain dimension and de-normalise them.
            - 'kde_individual': fit a 1-D KDE per dimension, locate local maxima
            and minima, then allocate samples proportionally to peak areas.
            - 'kde_ndim': fit an N-dimensional KDE (not yet fully implemented).
        num_samples : int, optional
            Number of new candidate points to return (default 1).
        chains_dir : str or None, optional
            Path to a saved chain file (currently unused; reserved for future use).
        **kwargs
            Additional arguments reserved for future use.

        Returns
        -------
        np.ndarray of shape (num_samples, num_dim) or None
            De-normalised parameter points for the next simulation batch.
            Returns None or partial results for unfinished method branches.
        """
        data = self.sampler._chain
        new_points = np.zeros((num_samples, self.num_dim))

        if method == "kde_individual":
            kde_kernels = []        
        
            domain_resolution = 1000
            domains = np.zeros((domain_resolution, self.num_dim))
            ranges = np.zeros((domain_resolution, self.num_dim))
            
            for i in range(self.num_dim):
                if MESA_parameters.parameters[i].continuous:
                    domains[:, i] = MESA_parameters.parameters[i].generate_grid(self.domain_resolution)
                else:
                    domains[:, i] = np.random.choice(MESA_parameters.parameters[i].discrete_steps)

                kde_kernels.append(stats.gaussian_kde(data[:, i]))
                ranges[:, i] = np.reshape(kde_kernels[-1](domains[:, i]).T, domains[:, i].shape)

                maxima = signal.argrelextrema(ranges[:, i])
                minima = signal.argrelextrema(ranges[:, i])

                areas = []

                if len(maxima) == 0 and len(minima) == 0:
                    pass
                elif len(maxima) == 0 and len(minima) != 0:
                    pass
                elif len(maxima) != 0 and len(minima) == 0:
                    pass
                else:
                    for j in range(len(minima)):
                        if j == 0:
                            areas.append(integrate.simpson(ranges[:, 0:minima[j]]))
                        elif j == len(minima)-1:
                            areas.append(integrate.simpson(ranges[:, minima[j]:]))
                        else:
                            areas.append(integrate.simpson(ranges[:, minima[j-1]:minima[j]]))

                    area_sum = np.sum(areas)
                    pts_per_area = np.zeros((len(areas)))

                    count = 0
                    maxima_order = np.argsort(maxima)
                    for j in range(len(maxima)):
                        new_points[count, i] = domains[maxima[maxima_order[j]]]
                        count += 1
                    
                    for j in range(len(areas)):
                        rel_area = areas[j] / area_sum
                        pts_per_area[j] = np.round(rel_area * (num_samples - len(maxima)))

                    pt_difference = num_samples - (np.sum(pts_per_area) + len(maxima))

                    area_args_sort = np.argsort(pts_per_area)

                    
                    if pt_difference > 0:
                        pts_per_area[np.argmax(pts_per_area)] += num_samples - np.sum(pts_per_area) - len(maxima)
                    elif pt_difference < 0:
                        for j in range(pt_difference):
                            pts_per_area[area_args_sort[j]] -= 1

                    for j in range(pts_per_area):
                        if j == 0:
                            new_points[count:count+pts_per_area[j], i] = np.random.choice(domains[0:minima[j]], size=pts_per_area[j])
                        elif j == len(pts_per_area) - 1:
                            new_points[count:count+pts_per_area[j], i] = np.random.choice(domains[minima[j]:], size=pts_per_area[j])
                        else:
                            new_points[count:count+pts_per_area[j], i] = np.random.choice(domains[minima[j]:minima[j+1]], size=pts_per_area[j])

                        count += pts_per_area[j]

        elif method == "kde_ndim":
            # TODO: Finish
            nd_kernel = stats.gaussian_kde(data.T)
            domain_resolution = 1000
            domains = np.zeros((domain_resolution, self.num_dim))

            for i in range(self.num_dim):
                if MESA_parameters.parameters[i].continuous:
                    domains[:, i] = MESA_parameters.parameters[i].generate_grid(self.domain_resolution)

            nd_curve = np.reshape(nd_kernel(domains.T))

        elif method == "random":
            for i in range(self.num_dim):
                new_points[:, i] = self.ML_Data.denormalise_point(np.asarray(np.random.choice(np.asarray(data[:, i]), num_samples)), chains=True, index=i)
                
            return new_points

def test():
    param_set = ParameterSet.read_parameter_set('test.param_set')
    data = pd.read_csv('testData.csv')
    data = data[['ID', 'flag', 'mass', 'helium', 'metallicity', 'IBand', 'Ierr', 'VI', 'VIerr']]
    ML_Data_test = MESATraining.read_MESA_Training("test.training_set")
    active_kwargs = {'ML_Data': ML_Data_test, 'Param_Data': param_set}
    test_Active = ActiveLearning(True, True, kwargs=active_kwargs)
    return None


if __name__ == "__main__":
    test()
    print("----------------------")
