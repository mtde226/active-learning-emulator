#!/usr/bin/python3


import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import copy
import pickle

import os
import json
import shutil
import joblib
import tempfile
import zipfile

from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import fbeta_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from ActiveLearning.parameter_set import *


def get_1D_error(data, param1, error_col, num_bins=50):
    """
    This function calculates the average error in a given subsection of the provided parameter column. 
    The values from the error column in data are averaged after being divided into bins of equal size based on the param1 column.

    Parameters
    ----------
    data : pandas DataFrame object
        A pandas DataFrame object that contains at least two columns, one for the provided parameter column and one for the provided error column.
    param1 : str
        A column name in data.
    error_col : str
        A column name in data.
    num_bins : int
        The number of bins to equally divide the param1 column into.

    Returns
    -------
    coord_1 : list
        A list of the midpoints created by the equally sized bins of param1.
    coord_err : list
        A list of the error value of the bins centered at the coord_1 midpoints.

    """
    coord_1 = []
    coord_err = []
   
    _, param1_bin_edges = np.histogram(data[param1], bins=num_bins)
    
    for i in range(len(param1_bin_edges)-1):
        param1_left = param1_bin_edges[i]
        param1_right = param1_bin_edges[i+1]
        
        if i+1 == len(param1_bin_edges):
            temp = data.query("{0} <= {1} <= {2}".format(param1_left, param1, param1_right))[error_col]
        else:
            temp = data.query("{0} <= {1} < {2}".format(param1_left, param1, param1_right))[error_col]
                
                
        if len(temp) == 0:
            coord_mean = np.nan
        elif len(temp) == 1:
            coord_mean = np.median(temp)
        else:
            coord_mean = np.ma.median(np.ma.masked_invalid(temp.to_numpy()))

        coord_1.append((param1_right + param1_left) / 2)
        coord_err.append(coord_mean)

    return coord_1, coord_err
    

def get_2D_error(data, param1, param2, error_col, num_bins=50):
    """
    This function calculates the average error in a given subsection of the provided parameter column. 
    The values from the error column in data are averaged after being divided into a number of grid 
    sections of equal size based on the param1 and param2 column.

    Parameters
    ----------
    data : pandas DataFrame object
        A pandas DataFrame object that contains at least two columns, one for the provided parameter column and one for the provided error column.
    param1 : str 
        A column name in data.
    param2 : str
        A column name in data.
    error_col : str
        A column name in data.
    num_bins : int
        The number of bins to equally divide the param1 column into.

    Returns
    -------
    coord_1 : list
        A list of the midpoints created by the equally sized bins of param1.
    coord_2 : list
        A list of the midpoints created by the equally sized bins of param2.
    coord_err : list
        A list of the error value of the bins centered at the coord_1 midpoints.

    """
    coord_1 = []
    coord_2 = []
    coord_err = []
        
    _, param1_bin_edges = np.histogram(data[param1], bins=num_bins)
    _, param2_bin_edges = np.histogram(data[param2], bins=num_bins)
        
    for i in range(len(param1_bin_edges)-1):
        for j in range(len(param2_bin_edges)-1):
            param1_left = param1_bin_edges[i]
            param1_right = param1_bin_edges[i+1]
            param2_left = param2_bin_edges[j]
            param2_right = param2_bin_edges[j+1]

            
            if i+1 == len(param1_bin_edges):
                if j+1 == len(param2_bin_edges):
                    temp = data.query("{0} <= {1} <= {2} and {3} <= {4} <= {5}".format(param1_left, param1, param1_right, param2_left, param2, param2_right))[error_col]
                else:
                    temp = data.query("{0} <= {1} <= {2} and {3} <= {4} < {5}".format(param1_left, param1, param1_right, param2_left, param2, param2_right))[error_col]
            elif j+1 == len(param2_bin_edges):
                temp = data.query("{0} <= {1} < {2} and {3} <= {4} <= {5}".format(param1_left, param1, param1_right, param2_left, param2, param2_right))[error_col]

            else:
                temp = data.query("{0} <= {1} < {2} and {3} <= {4} < {5}".format(param1_left, param1, param1_right, param2_left, param2, param2_right))[error_col]
                

            if len(temp) == 0:
                coord_mean = np.nan
            elif len(temp) == 1:
                coord_mean = np.median(temp)
            else:
                coord_mean = np.ma.median(np.ma.masked_invalid(temp.to_numpy()))


            coord_1.append((param1_right + param1_left) / 2)
            coord_2.append((param2_right + param2_left) / 2)
            coord_err.append(coord_mean)

    return coord_1, coord_2, coord_err


def make_corner(data, param_cols, plot_col, num_bins, save_path=""):
    """
    Create a corner plot of the provided plot column (the resulting scatterplot point colours) versus the provided paramter columns. 
    The corner plot is created by creating a grid of subplots where the x and y axes of each subplot are shared vertically and horizontally. 
    The upper triangle of the corner diagram is discarded, and the diagonal plots share only their x-axes with other plots. 
    The y-axes are the same as the provided plot column.

    Parameters
    ----------
    data : pandas DataFrame object 
        A pandas DataFrame object that contains at least two columns, one for the provided parameter column and one for the provided error column.
    param1 : str
        A column name in data.
    param2 : str
       A column name in data.
    error_col : str
        A column name in data.
    num_bins : int 
        The number of bins to equally divide the param1 column into.
    save_path : str
        The file path to save the corner plots to. 

   Returns
    -------
    coord_1 : list
        A list of the midpoints created by the equally sized bins of param1.
    coord_2 : list
        A list of the midpoints created by the equally sized bins of param2.
    coord_err : list
        A list of the error value of the bins centered at the coord_1 midpoints. 
    """
    
    fig, ax = plt.subplots(len(param_cols), len(param_cols), sharex=False, sharey=False)
    for i in range(len(param_cols)):
        midpoints, error = get_1D_error(data, param_cols[i], plot_col, num_bins)
        image = ax[i, i].scatter(midpoints, error, c=error, cmap='inferno')
        plt.colorbar(image, ax=ax[i, i])
        for j in range(i, len(param_cols)):
            if j == i:
                continue

            midpoints_x, midpoints_y, error = get_2D_error(data, param_cols[i], param_cols[j], plot_col, num_bins)
            image = ax[j, i].scatter(midpoints_x, midpoints_y, c=error, cmap='inferno')
            plt.colorbar(image, ax=ax[j, i])
            if i == 0:
                ax[j, i].set_ylabel(param_cols[j])
            if j == len(param_cols)-1:
                ax[j, i].set_xlabel(param_cols[i])


        for ax_temp in ax[i, i+1:]:
            ax_temp.axis("off")

    fig.set_size_inches(12, 9)
    fig.suptitle(str(plot_col)+' Corner Plot')
    fig.savefig(save_path+str(plot_col)+'_corner.png', dpi=100)
    

class MESATraining:
    """
    A comprehensive machine learning framework for training classification and regression models on MESA 
    stellar evolution data, with support for data normalization, uncertainty quantification, error 
    parameterization, and visualization tools for active learning applications.

    Attributes
    ----------
    MESA_Parameters : ParameterSet
        Parameter set containing MESA parameter definitions
    classification_model : object
        Deprecated classifier attribute
    parameter_columns : list
        List of column names for input parameters
    classification_columns : list
        List of column names for classification targets
    regression_columns : list
        List of column names for regression targets
    dataset : pandas.DataFrame
        Main dataset containing all training data
    regressor : dict
        Dictionary of trained regression models keyed by column name
    regressor_error_dist : dict
        Dictionary of error distributions for each regression target
    regressor_error_rmse : dict
        Dictionary of RMSE values for each regression model
    classification_error_accuracy : float
        Classification error accuracy metric (deprecated)
    classifier_accuracy : float
        Overall classification accuracy score
    residual_interpolator : dict
        Dictionary of models for predicting regression residuals
    normalise_dict : dict
        Dictionary storing normalization parameters for each column
    num_classes : int
        Number of classes for classification tasks
    resample_flag : bool
        Flag indicating whether SMOTE resampling was applied
    oversampler : SMOTE
        SMOTE oversampling object for handling class imbalance
    classifier_mode : str
        Type of classifier being used ('dnn' or 'random_forest')
    regressor_mode : str
        Type of regressor being used ('dnn' or 'random_forest')
    interpolator_type : str
        Type of interpolator for residual prediction
    X_train_classifier, X_test_classifier, X_val_classifier : pandas.Index
        Data splits for classifier training
    y_train_classifier, y_test_classifier, y_val_classifier : pandas.Index
        Target splits for classifier training
    X_train_regressor, X_test_regressor, X_val_regressor : pandas.Index
        Data splits for regressor training
    y_train_regressor, y_test_regressor, y_val_regressor : pandas.Index
        Target splits for regressor training
    train_residuals, val_residuals, test_residuals : pandas.DataFrame
        Residual DataFrames for training, validation, and test sets
    classifier : object
        Trained classification model

    Methods
    -------
    __init__(parameter_set=None, classifier=None, regressor=None, datafile_path=None, pandas_kwargs={})
        Initialize MESATraining object with optional parameter set and data loading
    set_classification_columns(column_names)
        Set the column names for classification targets
    set_regression_columns(column_names)
        Set the column names for regression targets
    set_parameter_columns(column_names)
        Set the column names for input parameters
    load_dataset(datafile_path, class_columns=None, regress_columns=None, parameter_columns=None, all_columns=None, pandas_df=True, pandas_kwargs={})
        Load dataset from file with column specification
    save_MESA_Training(MESATraining_name, file_path="")
        Save MESATraining object to zipped file set
    read_MESA_Training(file_path)
        Load MESATraining object from pickle file (deprecated) or zipped file set
    normalise_data(data_point=[], new_dataset=False, parameters=False, regression_output=False)
        Normalize dataset or individual data points
    normalise(data, minimum, maximum)
        Apply min-max normalization to data
    denormalise_point(theta, parameters=False, chains=False, index=0)
        Denormalize individual parameter points
    denormalise_data(data=None)
        Denormalize entire dataset or specified data
    denormalise_regression_predictions(predictions, column=None)
        Denormalize regression model predictions
    create_model(style, inputDim, outputDim, numLayers, hiddenInit, dropoutFrac, activation, parameterise=False)
        Create neural network models for classification or regression
    train_classifier(layers=3, init_nodes=64, dropout=0.5, activation='relu', learning=0.001, training=0.8, validation=0.5, callbacks=None, tf_fit_kwargs={}, num_classes=None, mode='dnn')
        Train classification models using DNN or Random Forest
    train_regressor(layers=3, init_nodes=64, dropout=0.5, activation='relu', learning=0.001, training=0.8, validation=0.5, test_criteria=None, test_metric_val=None, classifier_class=None, classifier_column=None, callbacks=None, tf_fit_kwargs={}, mode='dnn')
        Train regression models using DNN or Random Forest
    evaluate_classification_accuracy(test_data, prediction_data, criteria=None, metric_val=None)
        Evaluate classification model accuracy
    evaluate_regression_accuracy(test_data, prediction_data, criteria=None, metric_val=None)
        Evaluate regression model accuracy
    apply_regression_uncertainty(data_points, method="distribution", parameterisation_mode="dnn")
        Apply uncertainty quantification to regression predictions
    set_residuals(return_values=False)
        Calculate and store regression residuals for all data splits
    parameterise_error(mode="random_forest", corner=True, num_bins=20, save_path="", kwargs={})
        Train models to predict regression residuals and create corner plots
    """
    
    def __init__(self, parameter_set=None, classifier=None, regressor=None, datafile_path=None, pandas_kwargs={}):
        """
        Initializes a MESATraining object with optional parameter set and automatic data loading.
        Sets up all necessary attributes for machine learning workflows including model containers,
        error tracking, and data normalization. If datafile_path is provided, automatically loads
        the dataset using load_dataset method.

        Parameters
        ----------
        parameter_set : ParameterSet, optional 
            ParameterSet object containing MESA parameter definitions. Default is None.
        classifier : object, optional 
            Deprecated classifier parameter. Default is None.
        regressor : object, optional
            Deprecated regressor parameter. Default is None.
        datafile_path : str, optional
            Path to data file to load upon initialization. Default is None.
        pandas_kwargs : dict, optional
            Additional keyword arguments for pandas file reading. Default is empty dict.

        Returns
        -------
        None

        """
        
        self.MESA_Parameters = parameter_set
        self.classification_model = classifier
        self.parameter_columns=None
        self.classification_columns=None
        self.regression_columns=None
        if datafile_path != None:
            self.load_dataset(datafile_path, pandas_kwargs=pandas_kwargs)
        else:
            self.dataset = None
        self.regressor = {}
        self.regressor_error_dist = {}
        self.regressor_error_rmse = {}
        self.classification_error_accuracy = 0
        self.classifier_accuracy = 0
        self.residual_interpolator = {}
        
            
    def set_classification_columns(self, column_names):
        """
        Sets the classification target columns for the training object. Creates a copy of the
        provided list to avoid reference issues. Only sets the columns if column_names is not None.
        These columns will be used as targets for classification model training.

        Parameters
        ----------
        column_names : list 
            List of column names to use as classification targets
        
        Returns
        -------
        None
        """
        
        if column_names != None:
            self.classification_columns=column_names[:]
        

    def set_regression_columns(self, column_names):
        """
        Sets the regression target columns for the training object. Creates a copy of the
        provided list to avoid reference issues. Only sets the columns if column_names is not None.
        These columns will be used as targets for regression model training.

        Parameters
        ----------
        column_names : list 
            List of column names to use as regression targets

        Returns
        -------
        None
        """
        
        if column_names != None:
            self.regression_columns=column_names[:]
            
        
    def set_parameter_columns(self, column_names):
        """
        Sets the input parameter columns for the training object. Creates a copy of the
        provided list to avoid reference issues. Only sets the columns if column_names is not None.
        These columns will be used as features for both classification and regression models.

        Parameters
        ----------
        column_names : list
            List of column names to use as input parameters

        Returns
        -------
        None
        """
        
        if column_names != None:
            self.parameter_columns=column_names[:]    


    def load_dataset(self, datafile_path, class_columns=None, regress_columns=None, parameter_columns=None, all_columns=None, pandas_df=True, pandas_kwargs={}):
        """
        Loads dataset from file and sets column specifications. Uses pandas.read_csv by default
        with header=0, but can use numpy.loadtxt with provided column names. Automatically sets
        classification, regression, and parameter columns if provided. Supports additional
        pandas reading options through pandas_kwargs.

        Parameters
        ----------
        datafile_path : str 
            Path to the data file to load
        class_columns : list, optional 
            Column names for classification targets. Default is None.
        regress_columns : list, optional 
            Column names for regression targets. Default is None.
        parameter_columns : list, optional
            Column names for input parameters. Default is None.
        all_columns : list, optional 
            All column names when pandas_df is False. Default is None.
        pandas_df : bool, optional 
            Whether to use pandas CSV reader or numpy loadtxt. Default is True.
        pandas_kwargs : dict, optional
            Additional arguments for pandas.read_csv. Default is empty dict.

        Returns
        -------
        None
        """
        
        if pandas_df:
            self.dataset=pd.read_csv(datafile_path, header=0, **pandas_kwargs)
        else:
            self.dataset=pd.DataFrame(np.loadtxt(datafile_path), columns=all_columns)
            

        self.set_classification_columns(class_columns)
        self.set_regression_columns(regress_columns)
        self.set_parameter_columns(parameter_columns)

        
    def save_MESA_Training(self, MESATraining_name, pkl=False, file_path=""):
        """
        Saves the MESATraining object to a zip archive (<MESATraining_name>.training_set)
        that contains one sub-file per logical component, each stored in its most
        memory-efficient native format:
 
        * dataset / residual DataFrames  → Parquet  (columnar, compressed)
        * regressor_error_dist arrays    → NumPy npz (one array per column)
        * Keras / TF models              → SavedModel directories (then zipped)
        * sklearn models (RF etc.)       → joblib   (memory-mappable NumPy arrays)
        * normalise_dict and all scalar  → JSON     (tiny, human-readable)
        metadata (column names, flags,
        accuracy scores, train/val/test
        index lists, …)
 
        Why this is better than pickle
        ------------------------------
        pickle.dump/load deserializes the *entire* object graph into RAM at once,
        including all nested numpy arrays inside every sklearn tree and every
        TensorFlow variable — even if you only need one attribute.  The formats
        used here let the OS stream or memory-map data on demand, so peak RAM
        during load is proportional to what you actually *use*, not to the total
        file size.
 
        Parameters
        ----------
        MESATraining_name : str
        Base name for the archive (no extension needed).
        file_path : str, optional
        Directory in which to write the archive.  Default is cwd.
 
        Returns
        -------
        None
        """

        if pkl:
            with open(file_path+MESATraining_name+'.training_set', 'wb') as f:
                pickle.dump(self, f)
        else:
            archive_path = os.path.join(file_path, MESATraining_name + ".training_set")
 
            with tempfile.TemporaryDirectory() as tmp:
 
                # ── 1. DataFrames ────────────────────────────────────────────────
                for attr, fname in [
                        ("dataset",         "dataset.parquet"),
                        ("train_residuals", "train_residuals.parquet"),
                        ("val_residuals",   "val_residuals.parquet"),
                        ("test_residuals",  "test_residuals.parquet"),
                ]:
                    df = getattr(self, attr, None)
                    if df is not None and not df.empty:
                        df.to_parquet(os.path.join(tmp, fname), index=True)
 
                # ── 2. ML models ─────────────────────────────────────────────────
                self._save_models(tmp)
 
                # ── 3. regressor_error_dist (arrays keyed by regression column) ──
                if self.regressor_error_dist:
                    np.savez(
                        os.path.join(tmp, "regressor_error_dist.npz"),
                        **{col: arr for col, arr in self.regressor_error_dist.items()}
                    )
 
                # ── 4. ParameterSet (uses its own JSON serializer) ───────────────
                if self.MESA_Parameters is not None:
                    self.MESA_Parameters.save_parameter_set(tmp + os.sep, "MESA_Parameters")
 
                # ── 5. Everything else as JSON ───────────────────────────────────
                def _index_to_list(idx):
                    """pandas Index → plain Python list for JSON."""
                    return None if idx is None else list(idx)
 
                meta = {
                    # column lists
                    "parameter_columns":      self.parameter_columns,
                    "classification_columns": self.classification_columns,
                    "regression_columns":     self.regression_columns,
                    # normalisation dict  (values are [min, max] floats — JSON-safe)
                    "normalise_dict": getattr(self, "normalise_dict", None),
                    # scalar metrics
                    "classification_error_accuracy": self.classification_error_accuracy,
                    "classifier_accuracy":           self.classifier_accuracy,
                    "regressor_error_rmse":          self.regressor_error_rmse,
                    # mode strings
                    "classifier_mode":   getattr(self, "classifier_mode",   None),
                    "regressor_mode":    getattr(self, "regressor_mode",     None),
                    "interpolator_type": getattr(self, "interpolator_type",  None),
                    "resample_flag":     getattr(self, "resample_flag",      None),
                    "num_classes":       getattr(self, "num_classes",        None),
                    # train/val/test index splits (stored as lists)
                    "y_train_classifier": _index_to_list(getattr(self, "y_train_classifier", None)),
                    "y_val_classifier":   _index_to_list(getattr(self, "y_val_classifier",   None)),
                    "y_test_classifier":  _index_to_list(getattr(self, "y_test_classifier",  None)),
                    "y_train_regressor": _index_to_list(getattr(self, "y_train_regressor", None)),
                    "y_val_regressor":   _index_to_list(getattr(self, "y_val_regressor",   None)),
                    "y_test_regressor":  _index_to_list(getattr(self, "y_test_regressor",  None)),
                }
                
                with open(os.path.join(tmp, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
 
                # ── 6. Pack everything into a zip archive ─────────────────────────
                # We use ZIP_DEFLATED (gzip) which adds ~10-30 % further size
                # reduction on top of Parquet's internal compression.
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(tmp):
                        for file in files:
                            abs_path = os.path.join(root, file)
                            arc_name = os.path.relpath(abs_path, tmp)
                            zf.write(abs_path, arc_name)
 
 
    def read_MESA_Training(file_path):
        """
        Static method that loads a MESATraining object from an archive written by
        the new save_MESA_Training.  Falls back to pickle for archives produced by
        the old implementation so that existing saved training sets continue to work.
 
        Each component is loaded independently from its native format, so only the
        data you actually access needs to be paged into RAM.
 
        Parameters
        ----------
        file_path : str
        Full path to the .training_set archive.
 
        Returns
        -------
        MESATraining
        The reconstructed MESATraining object.
        """
 
        # ── Detect format: zip (new) vs pickle (old) ─────────────────────────
        with open(file_path, "rb") as f:
            magic = f.read(4)
 
        if magic != b"PK\x03\x04":          # not a zip → old pickle format
            with open(file_path, "rb") as f:
                return pickle.load(f)
 
        obj = MESATraining()                 # blank object — we fill it below
 
        with tempfile.TemporaryDirectory() as tmp:
 
            # Unzip into a temp directory
            with zipfile.ZipFile(file_path, "r") as zf:
                zf.extractall(tmp)
 
            # ── 1. Metadata ───────────────────────────────────────────────────
            with open(os.path.join(tmp, "meta.json")) as f:
                meta = json.load(f)
 
            obj.parameter_columns      = meta["parameter_columns"]
            obj.classification_columns = meta["classification_columns"]
            obj.regression_columns     = meta["regression_columns"]
            obj.normalise_dict         = meta.get("normalise_dict")
            obj.classification_error_accuracy = meta["classification_error_accuracy"]
            obj.classifier_accuracy           = meta["classifier_accuracy"]
            obj.regressor_error_rmse          = meta["regressor_error_rmse"]
            obj.classifier_mode    = meta.get("classifier_mode")
            obj.regressor_mode     = meta.get("regressor_mode")
            obj.interpolator_type  = meta.get("interpolator_type")
            obj.resample_flag      = meta.get("resample_flag")
            obj.num_classes        = meta.get("num_classes")
            
            def _list_to_index(lst):
                import pandas as pd
                return None if lst is None else pd.Index(lst)
 
            obj.y_train_classifier = _list_to_index(meta.get("y_train_classifier"))
            obj.y_val_classifier   = _list_to_index(meta.get("y_val_classifier"))
            obj.y_test_classifier  = _list_to_index(meta.get("y_test_classifier"))
            obj.y_train_regressor  = _list_to_index(meta.get("y_train_regressor"))
            obj.y_val_regressor    = _list_to_index(meta.get("y_val_regressor"))
            obj.y_test_regressor   = _list_to_index(meta.get("y_test_regressor"))
            dist_path = os.path.join(tmp, "regressor_error_dist.npz")
            if os.path.exists(dist_path):
                loaded = np.load(dist_path)
                obj.regressor_error_dist = {col: loaded[col] for col in loaded.files}
            else:
                obj.regressor_error_dist = {}
 
            # ── 2. DataFrames ─────────────────────────────────────────────────
            for attr, fname in [
                ("dataset",         "dataset.parquet"),
                ("train_residuals", "train_residuals.parquet"),
                ("val_residuals",   "val_residuals.parquet"),
                ("test_residuals",  "test_residuals.parquet"),
            ]:
                p = os.path.join(tmp, fname)
                setattr(obj, attr, pd.read_parquet(p) if os.path.exists(p) else None)
 
            # ── 3. ML models ──────────────────────────────────────────────────
            obj._load_models(tmp)
 
            # ── 4. ParameterSet ───────────────────────────────────────────────
            ps_path = os.path.join(tmp, "MESA_Parameters.param_set")
            if os.path.exists(ps_path):
                obj.MESA_Parameters = ParameterSet.read_parameter_set(ps_path)
            else:
                obj.MESA_Parameters = None
 
        return obj
  
 
    def _save_models(self, directory):
        """
        Writes all trained models in self.classifier, self.regressor, and
        self.residual_interpolator to *directory* using their native, most
        RAM-efficient serialization format.
        
        * Keras / TF models  →  SavedModel subdirectory  (tf.saved_model.save)
        * sklearn models     →  joblib file              (memory-mappable)
 
        Parameters
        ----------
        directory : str
        Absolute path to an existing directory that will receive the model files.
        
        Returns
        -------
        None
        """
 
        model_manifest = {}   # records what type each saved model is
        
        # ── classifier ────────────────────────────────────────────────────────
        clf = getattr(self, "classifier", None)
        if clf is not None:
            if isinstance(clf, tf.keras.Model):
                save_dir = os.path.join(directory, "classifier_keras")
                clf.save(save_dir)
                model_manifest["classifier"] = {"format": "keras", "path": "classifier_keras"}
            else:
                save_path = os.path.join(directory, "classifier.joblib")
                joblib.dump(clf, save_path, compress=3)
                model_manifest["classifier"] = {"format": "joblib", "path": "classifier.joblib"}
 
        # ── regressors (one per regression column) ────────────────────────────
        regressors_meta = {}
        for col, model in (self.regressor or {}).items():
            safe_col = col.replace("/", "_").replace(" ", "_")
            if isinstance(model, tf.keras.Model):
                save_dir = os.path.join(directory, f"regressor_{safe_col}_keras")
                model.save(save_dir)
                regressors_meta[col] = {"format": "keras", "path": f"regressor_{safe_col}_keras"}
            else:
                save_path = os.path.join(directory, f"regressor_{safe_col}.joblib")
                joblib.dump(model, save_path, compress=3)
                regressors_meta[col] = {"format": "joblib", "path": f"regressor_{safe_col}.joblib"}
        model_manifest["regressors"] = regressors_meta
 
        # ── residual interpolators (one per regression column) ────────────────
        interp_meta = {}
        for col, model in (self.residual_interpolator or {}).items():
            safe_col = col.replace("/", "_").replace(" ", "_")
            if isinstance(model, tf.keras.Model):
                save_dir = os.path.join(directory, f"interp_{safe_col}_keras")
                model.save(save_dir)
                interp_meta[col] = {"format": "keras", "path": f"interp_{safe_col}_keras"}
            else:
                save_path = os.path.join(directory, f"interp_{safe_col}.joblib")
                joblib.dump(model, save_path, compress=3)
                interp_meta[col] = {"format": "joblib", "path": f"interp_{safe_col}.joblib"}
        model_manifest["residual_interpolators"] = interp_meta
 
        with open(os.path.join(directory, "model_manifest.json"), "w") as f:
            json.dump(model_manifest, f, indent=2)
 
 
    def _load_models(self, directory):
        """
        Reads all model files from *directory* using the manifest written by
        _save_models and populates self.classifier, self.regressor, and
        self.residual_interpolator.
 
        Parameters
        ----------
        directory : str
        Absolute path to the directory containing the model files and
        model_manifest.json.
 
        Returns
        -------
        None
        """
 
        manifest_path = os.path.join(directory, "model_manifest.json")
        if not os.path.exists(manifest_path):
            self.classifier = None
            self.regressor = {}
            self.residual_interpolator = {}
            return
 
        with open(manifest_path) as f:
            manifest = json.load(f)
 
        def _load_one(info):
            path = os.path.join(directory, info["path"])
            if info["format"] == "keras":
                return tf.keras.models.load_model(path)
            else:
                return joblib.load(path)
 
        # classifier
        clf_info = manifest.get("classifier")
        self.classifier = _load_one(clf_info) if clf_info else None
 
        # regressors
        self.regressor = {}
        for col, info in manifest.get("regressors", {}).items():
            self.regressor[col] = _load_one(info)
 
        # residual interpolators
            self.residual_interpolator = {}
            for col, info in manifest.get("residual_interpolators", {}).items():
                self.residual_interpolator[col] = _load_one(info)
         
        
    def normalise_data(self, data_point=[], new_dataset=False, parameters=False, regression_output=False):
        """
        Normalizes dataset or individual data points using min-max normalization. If no data_point
        is provided, creates normalization dictionary and normalizes the entire dataset. If data_point
        is provided, applies existing normalization parameters to the point. Supports selective
        normalization of parameters and regression outputs.

        Parameters
        ----------
        data_point : list, optional 
            Individual data point to normalize. Default is empty list.
        new_dataset : bool, optional
            Flag for new dataset normalization (unused). Default is False.
        parameters : bool, optional
            Whether to normalize parameter columns in data_point. Default is False.
        regression_output : bool, optional 
            Whether to normalize regression columns in data_point. Default is False.
        
        Returns
        -------
        numpy.ndarray or None 
            Normalized data point or matrix of data points if data_point provided, None otherwise
        """
        if len(data_point) == 0:
            self.normalise_dict = {}
            if self.parameter_columns != None:
                for column in self.parameter_columns:
                    self.normalise_dict[column] = [self.dataset[column].min(), self.dataset[column].max()]
                    self.dataset[column] = self.normalise(self.dataset[column], *self.normalise_dict[column])

            if self.regression_columns != None:
                for column in self.regression_columns:
                    self.normalise_dict[column] = [self.dataset[column].min(), self.dataset[column].max()]
                    self.dataset[column] = self.normalise(self.dataset[column], *self.normalise_dict[column])                   
        else:
            return_values = np.empty((np.shape(data_point)))
            if self.normalise_dict:
                if parameters and regression_output:
                    for i in range(len(self.parameter_columns)):
                        if np.ndim(data_point) == 1:
                            return_values[i] = self.normalise(data_point[i], *self.normalise_dict[self.parameter_columns[i]])
                        else:
                            return_values[:, i] = self.normalise(data_point[:, i], *self.normalise_dict[self.parameter_columns[i]])
                    for i in range(len(self.regression_columns)):
                        if np.ndim(data_point) == 1:
                            return_values[i+len(self.parameter_columns)] = self.normalise(data_point[i+len(self.parameter_columns)], *self.normalise_dict[self.regression_columns[i]])
                        else:
                            return_values[:, i+len(self.parameter_columns)] = self.normalise(data_point[:, i+len(self.parameter_columns)], *self.normalise_dict[self.regression_columns[i]])
                elif parameters:
                    for i in range(len(self.parameter_columns)):
                        if np.ndim(data_point) == 1:
                            return_values[i] = self.normalise(data_point[i], *self.normalise_dict[self.parameter_columns[i]])
                        else:
                            return_values[:, i] = self.normalise(data_point[:, i], *self.normalise_dict[self.parameter_columns[i]])
                elif regression_output:
                    for i in range(len(self.regression_columns)):
                        if np.ndim(data_point) == 1:
                            return_values[i] = self.normalise(data_point[i], *self.normalise_dict[self.regression_columns[i]])
                        else:
                            return_values[:, i] = self.normalise(data_point[:, i], *self.normalise_dict[self.regression_columns[i]])
                return return_values
            else:
                return None

            
    def normalise(self, data, minimum, maximum):
        """
        Applies min-max normalization to scale data to the range [0, 1]. Uses the formula:
        (data - minimum) / (maximum - minimum). Works with both individual values and arrays.
        Used internally by normalise_data for consistent scaling across the dataset.

        Parameters
        ----------
        data : numeric or array-like 
            Data to be normalized
        minimum : float 
            Minimum value for normalization range
        maximum : float 
            Maximum value for normalization range

        Returns
        -------
        numeric or array-like
            Normalized data scaled to [0, 1] range
        """
        normal_data = (data - minimum) / (maximum - minimum)
        return normal_data


    def denormalise_point(self, theta, parameters=False, chains=False, index=0):
        """
        Converts normalized data points back to original scale using stored normalization parameters.
        Supports 1D and 2D arrays for parameter denormalization, and single-parameter chain denormalization.
        Uses inverse of min-max normalization: data * (max - min) + min.

        Parameters
        ----------
        theta : array-like
            Normalized data points to denormalize
        parameters : bool, optional
            Whether to denormalize as parameter columns. Default is False.
        chains : bool, optional
            Whether to denormalize as chain data for specific index. Default is False.
        index : int, optional
            Parameter index for chain denormalization. Default is 0.
        
        Returns
        -------
        numpy.ndarray 
            Denormalized data points in original scale
        """
        values = np.empty((np.shape(theta)))
        if parameters:
            if values.ndim == 1:
                for i in range(len(self.parameter_columns)):
                    minimum, maximum = self.normalise_dict[self.parameter_columns[i]]
                    values[i] = theta[i] * (maximum - minimum) + minimum
            else:
                for i in range(len(self.parameter_columns)):
                    minimum, maximum = self.normalise_dict[self.parameter_columns[i]]
                    values[:, i] = theta[:, i] * (maximum - minimum) + minimum

        if chains:
            minimum, maximum = self.normalise_dict[self.parameter_columns[index]]            
            values= theta * (maximum - minimum) + minimum

        return values
        

    def denormalise_data(self, data=None):
        """
        Denormalizes the entire dataset or specified data back to original scale. If data is None,
        denormalizes the main dataset in-place. Otherwise, denormalizes the provided data. Applies
        inverse normalization to both parameter and regression columns using stored normalization
        parameters. Modifies data in-place.

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Specific dataset to denormalize. Default is None.

        Returns
        -------
        None
        """
        if data is None:
            for i in range(len(self.parameter_columns)):
                minimum, maximum = self.normalise_dict[self.parameter_columns[i]]
                self.dataset[self.parameter_columns[i]] = self.dataset[self.parameter_columns[i]] * (maximum - minimum) + minimum

            for i in range(len(self.regression_columns)):
                minimum, maximum = self.normalise_dict[self.regression_columns[i]]
                self.dataset[self.regression_columns[i]] = self.dataset[self.regression_columns[i]] * (maximum - minimum) + minimum
        else:
            for i in range(len(self.parameter_columns)):
                minimum, maximum = self.normalise_dict[self.parameter_columns[i]]
                data[self.parameter_columns[i]] = self.dataset[self.parameter_columns[i]] * (maximum - minimum) + minimum

            for i in range(len(self.regression_columns)):
                minimum, maximum = self.normalise_dict[self.regression_columns[i]]
                data[self.regression_columns[i]] = self.dataset[self.regression_columns[i]] * (maximum - minimum) + minimum

            
    def denormalise_regression_predictions(self, predictions, column=None):
        """
        Converts normalized regression predictions back to original scale. If column is specified,
        denormalizes for that specific regression target. Otherwise, denormalizes all regression
        columns. Works with both 1D and 2D prediction arrays. Creates deep copy to avoid modifying
        input predictions.

        Parameters
        ----------
        predictions : array-like 
            Normalized regression predictions to denormalize
        column : str, optional 
            Specific column name for single-column denormalization. Default is None.

        Returns
        -------
        numpy.ndarray 
            Denormalized predictions in original scale
        """
        if np.ndim(predictions) > 1:
            denorm_data = copy.deepcopy(predictions[:])
        else:
            denorm_data = copy.deepcopy(predictions)
            
        if column is not None:
            minimum, maximum = self.normalise_dict[column]
            denorm_data = denorm_data * (maximum - minimum) + minimum
            return denorm_data
            
        for i in range(len(self.regression_columns)):
            minimum, maximum = self.normalise_dict[self.regression_columns[i]]
            if np.ndim(predictions) > 1:
                denorm_data[:, i] = denorm_data[:, i] * (maximum - minimum) + minimum                
            else:
                denorm_data[i] = denorm_data[i] * (maximum - minimum) + minimum


        return denorm_data
    

    def create_model(self, style, inputDim, outputDim, numLayers, hiddenInit, dropoutFrac, activation, parameterise=False):
        """
        Creates TensorFlow/Keras neural network models for classification or regression. Uses dense
        layers with ReLU activation and dropout for regularization. Output activation depends on
        model type: softmax/sigmoid for classification, ReLU/linear for regression. Supports
        multi-class classification and parameterized regression outputs.

        Parameters
        ----------
        style : str 
            Model type, either 'Classifier' or 'Regressor'
        inputDim : int 
            Number of input features
        outputDim : int 
            Number of output units
        numLayers : int 
            Number of hidden layers
        hiddenInit : int
            Number of neurons in each hidden layer
        dropoutFrac : float
            Dropout fraction for regularization
        activation : str 
            Activation function for hidden layers (not used in current implementation)
        parameterise : bool, optional
            Whether to use ReLU output activation for regression. Default is False.

        Returns
        -------
        tensorflow.keras.Model 
            Compiled neural network model
        """
        inputs = tf.keras.Input(shape=(inputDim,))
        x = tf.keras.layers.Dense(hiddenInit, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(dropoutFrac)(x)
        for i in range(numLayers-1):
            x = tf.keras.layers.Dense(hiddenInit, activation='relu')(x)
            x = tf.keras.layers.Dropout(dropoutFrac)(x)
        if (style == 'Classifier'):
            if self.num_classes > 2:
                outputs = tf.keras.layers.Dense(outputDim, activation='softmax')(x)
            else:
                outputs = tf.keras.layers.Dense(outputDim, activation='sigmoid')(x)
        elif (style == 'Regressor'):
            if parameterise:
                outputs = tf.keras.layers.Dense(outputDim, activation='relu')(x)
            else:                    
                outputs = tf.keras.layers.Dense(outputDim, activation='linear')(x)

                
        model = tf.keras.Model(inputs, outputs)
        return model
    
            
    def train_classifier(self, layers=3, init_nodes=64, dropout=0.5, activation='relu', learning=0.001, training=0.8, validation=0.5, tf_fit_kwargs={}, num_classes=None, mode='dnn'):
        """
        Creates and trains classification models using DNN or Random Forest with automatic data splitting and 
        SMOTE resampling. Creates and stores lists of indices as training lists to avoid consuming unnecessary 
        amounts of memory. Also stores a string identifying what the network was trained on.

        Parameters
        ----------
        layers : int, optional 
            Number of hidden layers for DNN (default: 3)
        init_nodes : int, optional
            Number of neurons in hidden layers (default: 64)
        dropout : float, optional
            Dropout fraction for regularization (default: 0.5)
        activation : str, optional
            Activation function for hidden layers (default: 'relu')
        learning : float, optional 
            Learning rate for optimizer (default: 0.001)
        training : float, optional 
            Fraction of data for training (default: 0.8)
        validation : float, optional
            Fraction of remaining data for validation (default: 0.5)
        callbacks : list, optional
            Keras callbacks for training (default: None)
        tf_fit_kwargs : dict, optional
            Additional arguments for model.fit() (default: {})
        num_classes : int, optional 
            Number of classes, auto-detected if None (default: None)
        mode : str, optional
            Model type: 'dnn' or 'random_forest' (default: 'dnn')

        Returns
        -------
        None
            Stores trained classifier in self.classifier
        """
        if not self.parameter_columns:
            try:
                self.set_parameter_column([parameter.name for parameter in self.MESA_Parameters.parameters])
            except:
                return None

        if not self.classification_columns:
            return None

        if self.dataset.empty:
            return None          

        self.X_train_classifier, self.X_test_classifier, self.y_train_classifier, self.y_test_classifier = train_test_split(self.dataset[self.parameter_columns], self.dataset[self.classification_columns], train_size=training, stratify=self.dataset[self.classification_columns])
        self.X_test_classifier, self.X_val_classifier, self.y_test_classifier, self.y_val_classifier = train_test_split(self.X_test_classifier, self.y_test_classifier, train_size=validation, stratify=self.y_test_classifier)

        self.X_train_classifier = self.X_train_classifier.index
        self.X_val_classifier = self.X_val_classifier.index
        self.X_test_classifier = self.X_test_classifier.index

        self.y_train_classifier = self.y_train_classifier.index
        self.y_val_classifier = self.y_val_classifier.index
        self.y_test_classifier = self.y_test_classifier.index
        
        counts = self.dataset[self.classification_columns].value_counts()
        if np.min(counts) / np.max(counts) < 0.3:
            self.resample_flag = True
            self.oversampler = RandomOverSampler()
            self.oversampler.fit_resample(self.dataset[self.parameter_columns].loc[self.X_train_classifier], self.dataset[self.classification_columns].loc[self.y_train_classifier])

            self.X_train_classifier = self.oversampler.sample_indices_
            self.y_train_classifier = self.oversampler.sample_indices_
            
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = len(np.unique(self.dataset[self.classification_columns]))
        
        if mode == 'dnn':
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)]
            self.classifier_mode = 'dnn'
            model = self.create_model(style='Classifier', inputDim=len(self.parameter_columns), outputDim=self.num_classes, numLayers=layers, hiddenInit=init_nodes, dropoutFrac=dropout, activation=activation)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        
            model.compile(optimizer, loss, metrics=metrics)

            train_dataset = tf.data.Dataset.from_tensor_slices((self.dataset[self.parameter_columns].loc[self.X_train_classifier].to_numpy(), self.dataset[self.classification_columns].loc[self.y_train_classifier].to_numpy()))

            if not tf_fit_kwargs['batch_size']:
                train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
            else:
                train_dataset = train_dataset.batch(tf_fit_kwargs['batch_size']).prefetch(tf.data.AUTOTUNE)

            val_dataset = tf.data.Dataset.from_tensor_slices((self.dataset[self.parameter_columns].loc[self.X_val_classifier].to_numpy(), self.dataset[self.classification_columns].loc[self.y_val_classifier].to_numpy()))
            val_dataset = val_dataset.batch(4096).prefetch(tf.data.AUTOTUNE)

            test_dataset = tf.data.Dataset.from_tensor_slices((self.dataset[self.parameter_columns].loc[self.X_test_classifier].to_numpy(), self.dataset[self.classification_columns].loc[self.y_test_classifier].to_numpy()))
            test_dataset = test_dataset.batch(4096).prefetch(tf.data.AUTOTUNE)
                
                
            history = model.fit(train_dataset, callbacks=callbacks, validation_data=val_dataset, **tf_fit_kwargs)
            self.classifier_accuracy = model.evaluate(test_dataset)[1]
            self.classifier = model
            
        elif mode == 'random_forest':
            self.classifier_mode = 'random_forest'
            model = RandomForestClassifier(n_estimators=5)
            model.fit(self.dataset[self.parameter_columns].loc[self.X_train_classifier].values, self.dataset[self.classification_columns].loc[self.y_train_classifier].values)

            self.classifier_accuracy = model.score(self.dataset[self.parameter_columns].loc[self.X_test_classifier], self.dataset[self.classification_columns].loc[self.y_test_classifier])
            self.classifier = model


    def predict_classifier(self, theta):
        """
        Make predictions using the trained classifier model.
    
        This method uses the stored regressor to predict class based on input
        parameters. It supports both random forest and deep neural network 
        (DNN) regressor modes.
    
        Parameters
        ----------
        theta : array-like
            Input parameters/features for making predictions. The shape and format
            should match what the regressor was trained on.
    
        Returns
        -------
        numeric
            The class predicted by the classifer
        """

        if self.ml_data.classifier_mode == 'random_forest':
            result_class = np.argmax(self.ml_data.classifier.predict_proba(np.asarray([theta])))
        elif self.ml_data.classifier_mode == 'dnn':
            result_class = np.argmax(self.ml_data.classifier(np.asarray([theta])))
        return result_class
            

    def train_regressor(self, layers=3, init_nodes=64, dropout=0.5, activation='relu', learning=0.001, training=0.8, validation=0.5, test_criteria=None, test_metric_val=None, classifier_class=None, classifier_column=None, tf_fit_kwargs={}, mode='dnn', multiple=False, multiple_column=''):
        """
        Creates and trains regression models using DNN or Random Forest with automatic data splitting. 
        The models are stored in a dictionary object that can be accessed with the relevant regression colun. 
        Training data as lists of indices to avoid consuming unnecessary amounts of memory. 
        Also stores a string identifying what the network was trained on.

        Parameters
        ----------
        layers : int, optional
            Number of hidden layers for DNN (default: 3)
        init_nodes : int, optional
            Number of neurons in hidden layers (default: 64)
        dropout : float, optional
            Dropout fraction for regularization (default: 0.5)
        activation : str, optional
            Activation function for hidden layers (default: 'relu')
        learning : float, optional
            Learning rate for optimizer (default: 0.001)
        training : float, optional
            Fraction of data for training (default: 0.8)
        validation : float, optional
            Fraction of remaining data for validation (default: 0.5)
        test_criteria : str, optional
            Test evaluation criteria (default: None)
        test_metric_val : float, optional
            Test metric value threshold (default: None)
        classifier_class : str, optional
            Specific class to train regressor for (default: None)
        classifier_column : str, optional
            Column name for class filtering (default: None)
        callbacks : list, optional
            Keras callbacks for training (default: None)
        tf_fit_kwargs : dict, optional
            Additional arguments for model.fit() (default: {})
        mode : str, optional
            Model type: 'dnn' or 'random_forest' (default: 'dnn')

        Returns
        -------
        None
            Stores trained regressors in self.regressor dictionary
        """        
        if not self.parameter_columns:
            try:
                self.set_parameter_columns([parameter.name for parameter in self.MESA_Parameters.parameters])
            except:
                return None

        if not self.regression_columns:
            return None

        if self.dataset.empty:
            return None

        if multiple:
            self.regressor_sort = self.dataset[multiple_column].unique()
            self.regressor_multiple = True
            self.train_regressor_indices = {}
            self.val_regressor_indices = {}
            self.test_regressor_indices = {}

            for cat in self.regressor_sort:
                self.regressor[cat] = {}
                self.regressor_error_rmse[cat] = {}
                self.regressor_error_dist[cat] = {}
                if classifier_class != None:
                    self.train_regressor_indices[cat], self.test_regressor_indices[cat] = train_test_split(self.dataset[self.dataset[classifier_column] == classifier_class].query('{0} == {1}'.format(multiple_column, cat)).index, train_size=training)
                    self.test_regressor_indices[cat], self.val_regressor_indices[cat] = train_test_split(self.test_regressor_indices[cat], train_size=validation)
                else:
                    self.train_regressor_indices[cat], self.test_regressor_indices[cat] = train_test_split(self.dataset.query('{0} == {1}'.format(multiple_column, cat)).index, train_size=training)
                    self.test_regressor_indices[cat], self.val_regressor_indices[cat] = train_test_split(self.test_regressor_indices[cat], train_size=validation)                                                                             
                
        else:
            if classifier_class != None:
                self.train_regressor_indices, self.test_regressor_indices = train_test_split(self.dataset[self.dataset[classifier_column] == classifier_class].index, train_size=training)
                self.test_regressor_indices, self.val_regressor_indices = train_test_split(self.test_regressor_indices, train_size=validation)

            else:
                self.train_regressor_indices, self.test_regressor_indices = train_test_split(self.dataset.index, train_size=training)
                self.test_regressor_indices, self.val_regressor_indices = train_test_split(self.test_regressor_indices, train_size=validation)
                
        for i in range(len(self.regression_columns)):
            if mode == 'dnn':
                self.regressor_mode = 'dnn'
                if multiple:
                    # TODO: UPDATE
                    for cat in self.regressor_sort:
                        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)]
                        model = self.create_model(style='Regressor', inputDim=len(self.parameter_columns), outputDim=1, numLayers=layers, hiddenInit=init_nodes, dropoutFrac=dropout, activation=activation)
                        optimizer = tf.keras.optimizers.Adam(learning_rate=learning)
                        loss = tf.keras.losses.MeanSquaredError()

                        model.compile(optimizer, loss)

                        self.regressor[cat][self.regression_columns[i]] = model
                        
                        history = self.regressor[cat][self.regression_columns[i]].fit(x=self.dataset[self.parameter_columns].loc[self.train_regressor_indices[cat]], y=self.dataset[self.regression_columns[i]].loc[self.train_regressor_indices[cat]], callbacks=callbacks, validation_data=(self.dataset[self.parameter_columns].loc[self.val_regressor_indices[cat]], self.dataset[self.regression_columns[i]].loc[self.val_regressor_indices[cat]]), **tf_fit_kwargs)

                        test_prediction = self.regressor[cat][self.regression_columns[i]](self.dataset[self.parameter_columns].loc[self.test_regressor_indices[cat]])
                        denorm_data = self.denormalise_regression_predictions(test_prediction, self.regression_columns[i]) # This yields a numpy array
                        y_test = self.denormalise_regression_predictions(copy.deepcopy(self.dataset[self.regression_columns[i]].loc[self.test_regressor_indices].to_numpy()), self.regression_columns[i]).reshape(-1, 1)
                        print(denorm_data)
                        print(y_test)
                        self.regressor_error_rmse[cat][self.regression_columns[i]] = mean_squared_error(y_test, denorm_data, squared=False)
                        self.regressor_error_dist[cat][self.regression_columns[i]] = denorm_data - y_test

                        print("RMSE", "Category:", str(cat), str(self.regression_columns[i])+":", self.regressor_error_rmse[cat][self.regression_columns[i]])
                                                                                      
                else:
                    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)]
                    model = self.create_model(style='Regressor', inputDim=len(self.parameter_columns), outputDim=1, numLayers=layers, hiddenInit=init_nodes, dropoutFrac=dropout, activation=activation)
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning)
                    loss = tf.keras.losses.MeanSquaredError()

                    model.compile(optimizer, loss)

                    self.regressor[self.regression_columns[i]] = model

                    train_dataset = tf.data.Dataset.from_tensor_slices((self.dataset[self.parameter_columns].loc[self.train_regressor_indices].to_numpy(), self.dataset[self.regression_columns[i]].loc[self.train_regressor_indices].to_numpy()))
                    if not tf_fit_kwargs['batch_size']:
                        train_dataset = train_dataset.shuffle(42).batch(32).prefetch(tf.data.AUTOTUNE)
                    else:
                        train_dataset = train_dataset.batch(tf_fit_kwargs['batch_size']).prefetch(tf.data.AUTOTUNE)
                        
                    val_dataset = tf.data.Dataset.from_tensor_slices((self.dataset[self.parameter_columns].loc[self.val_regressor_indices].to_numpy(), self.dataset[self.regression_columns[i]].loc[self.val_regressor_indices].to_numpy()))
                    val_dataset = val_dataset.batch(4096).prefetch(tf.data.AUTOTUNE)

                    test_dataset = tf.data.Dataset.from_tensor_slices(self.dataset[self.parameter_columns].loc[self.test_regressor_indices].to_numpy())
                    test_dataset = test_dataset.batch(4096).prefetch(tf.data.AUTOTUNE)
                    
                    history = self.regressor[self.regression_columns[i]].fit(train_dataset, callbacks=callbacks, validation_data=val_dataset, **tf_fit_kwargs)
                    
                    test_prediction = self.regressor[self.regression_columns[i]].predict(test_dataset)
                    denorm_data = self.denormalise_regression_predictions(test_prediction, self.regression_columns[i]) # This yields a numpy array
                    y_test = self.denormalise_regression_predictions(copy.deepcopy(self.dataset[self.regression_columns[i]].loc[self.test_regressor_indices].to_numpy()), self.regression_columns[i]).reshape(-1, 1)
                    self.regressor_error_rmse[self.regression_columns[i]] = mean_squared_error(y_test, denorm_data, squared=False)
                    self.regressor_error_dist[self.regression_columns[i]] = denorm_data - y_test

                    print("RMSE", str(self.regression_columns[i])+":", self.regressor_error_rmse[self.regression_columns[i]])
            
            elif mode == 'random_forest':
                self.regressor_mode = 'random_forest'
                model = RandomForestRegressor(n_estimators=5)
                self.regressor[self.regression_columns[i]] = model
                model.fit(self.dataset[self.parameter_columns].loc[self.train_regressor_indices].values, self.dataset[self.regression_columns[i]].loc[self.train_regressor_indices].values)
                test_prediction = self.regressor[self.regression_columns[i]].predict(self.dataset[self.parameter_columns].loc[self.test_regressor_indices])
                denorm_data = self.denormalise_regression_predictions(test_prediction, self.regression_columns[i]) # This yields a numpy array
                y_test = self.denormalise_regression_predictions(copy.deepcopy(self.dataset[self.regression_columns[i]].loc[self.test_regressor_indices].to_numpy()), self.regression_columns[i]).reshape(-1, 1)
                self.regressor_error_rmse[self.regression_columns[i]] = mean_squared_error(y_test, denorm_data, squared=False)
                self.regressor_error_dist[self.regression_columns[i]] = np.reshape(denorm_data, (len(y_test),)) - np.reshape(y_test, (len(y_test),))

                print("RMSE", str(self.regression_columns[i])+":", self.regressor_error_rmse[self.regression_columns[i]])


    def predict_regressor(self, theta, cat_val):
        """
        Make predictions using the trained regressor model(s).
    
        This method uses the stored regressor(s) to predict output values based on input
        parameters. It handles both single regressors and dictionaries of multiple 
        regressors (one per regression column), supporting both random forest and deep 
        neural network (DNN) regressor modes.
    
        Parameters
        ----------
        theta : array-like
            Input parameters/features for making predictions. The shape and format
            should match what the regressor was trained on.
    
        Returns
        -------
        np.ndarray or None
            - If self.regressor is a dictionary: Returns a numpy array of shape 
              (len(self.regression_columns),) containing predictions for each 
              regression column.
            - If self.regressor is a single model: Returns the direct output from 
              the regressor's predict method (random forest) or direct call (DNN).
            - If no regressor exists: Returns None.
        """
        
        if not self.regressor:
            return None

        if self.regressor_multiple:
            # DNNs only
            if isinstance(self.regressor[cat_val], dict):
                output_values = np.zeros((len(self.regression_columns),))
                for i in range(len(self.regressor[cat_val])):
                    output_values[i] = self.regressor[cat_val][self.regression_columns[i]](theta)
                else:
                    return self.regressor[cat_val](theta)


        elif isinstance(self.regressor, dict):
            output_values = np.zeros((len(self.regression_columns),))
            for i in range(len(self.regressor)):
                if self.regressor_mode == 'random_forest':
                    output_values[i] = self.regressor[self.regression_columns[i]].predict(theta)
                elif self.regressor_mode == 'dnn':
                    output_values[i] = self.regressor[self.regression_columns[i]](theta)

            return output_values
        else:
            if self.regressor_mode == 'random_forest':
                return self.regressor.predict(theta)
            elif self.regressor_mode == 'dnn':
                return self.regressor(theta)


    def evaluate_classification_accuracy(self, test_data, prediction_data, criteria=None, metric_val=None):
        """
        Evaluates classification model accuracy using specified criteria.
        Currently unfinished and requires completion before use.

        Parameters
        ----------
        test_data : array-like
            True classification labels
        prediction_data : array-like
            Predicted classification labels or probabilities
        criteria : str, optional
            Evaluation criteria (default: None)
        metric_val : float, optional
            Metric value threshold (default: None)

        Returns
        -------
        None
            Current implementation incomplete
"""
        if criteria == "original":
            n = prediction_data.shape()[1]
        return None
            

        
    def evaluate_regression_accuracy(self, test_data, prediction_data, criteria=None, metric_val=None):
        """
        Evaluates regression model accuracy using mean absolute error or percentile-based intervals.

        Parameters
        ----------
        test_data : array-like
            True regression values
        prediction_data : array-like
            Predicted regression values
        criteria : str, optional
            Evaluation method: 'mean' or 'interval' (default: 'interval')
        metric_val : float, optional
            Metric threshold value (default: 68 for interval method)

        Returns
        -------
        bool
            Whether accuracy meets the specified criteria
        """
        if not criteria:
            evaluate_method = 'interval'
        else:
            evaluate_method = criteria


            
        if evaluate_method == 'mean':
            status = np.mean(np.abs(test_data - prediction_data)) < evaluate_metric            
        elif evaluate_method == 'interval':
            if metric_val == None:
                evaluate_metric = 68
            else:
                evaluate_metric = metric_val

            status = (np.percentile(test_data - prediction_data, evaluate_metric) < metric_val)
            
        return status

        
    def apply_regression_uncertainty(self, data_points, method="distribution", parameterisation_mode="dnn"):
        """
        Applies uncertainty quantification to regression predictions by randomly sampling error distributions or 
        by using the parameterization of errors.

        Parameters
        ----------
        data_points : array-like 
            Denormalized regression predictions to add uncertainty to
        method : str, optional 
            Uncertainty method: 'distribution' or 'parameterisation' (default: 'distribution')
        parameterisation_mode : str, optional 
            Mode for parameterisation method: 'dnn' (default: 'dnn')

        Returns
        -------
        numpy.ndarray or None
            Predictions with added uncertainty, None if error distributions unavailable
        """
        if len(data_points) != len(self.regression_columns) and np.shape(data_points)[1] != len(self.regression_columns):
            return None


        if method == "distribution":
            if self.regressor_error_dist:
                return_values = np.zeros((np.shape(data_points)))
                if np.ndim(data_points) == (1,):
                    for i in range(len(self.regression_columns)):
                        return_values[i] = data_points[i] + np.random.choice(self.regressor_error_dist[self.regression_columns[i]])
                else:
                    for i in range(len(self.regression_columns)):
                        return_values[:, i] = data_points[:, i] + np.random.choice(self.regressor_error_dist[self.regression_columns[i]], len(data_points))
                return return_values
            else:
                return None
        elif method == "parameterisation":
            if parameterisation_mode == "dnn":
                if np.ndim(data_points) == (1,):
                    errors = np.zeros((1, len(self.regression_columns)))
                    for i in range(len(self.regression_columns)):
                        errors[i] = self.residual_interpolator[self.regression_columns[i]].predict(np.array(data_points).reshape(1, -1)).reshape(len(self.regression_columns),)
                    return_values = data_points + errors
                else:
                    errors = np.zeros((len(data_points), len(self.regression_columns)))
                    for i in range(len(self.regression_columns)):
                        errors[:, i] = self.residual_interpolator[self.regression_columns[i]].predict(np.array(data_points)).reshape(len(data_points), 1)
                    return_values = data_points + self.residual_interpolator[self.regression_columns[i]].predict(data_points)
            return return_values
            

    def set_residuals(self, return_values=False):
        """
        Calculates and stores regression residuals for training, validation, and test sets.

        Parameters
        ----------
        return_values : bool, optional
            Whether to return residual DataFrames (default: False)

        Returns
        -------
        tuple or None 
            If return_values=True, returns (train_residuals, val_residuals, test_residuals)
        """
        self.train_residuals = pd.DataFrame()
        self.val_residuals = pd.DataFrame()
        self.test_residuals = pd.DataFrame()        

        train_predictions = np.zeros((len(self.train_regressor_indices), len(self.regression_columns)))
        val_predictions = np.zeros((len(self.val_regressor_indices), len(self.regression_columns)))
        test_predictions = np.zeros((len(self.test_regressor_indices), len(self.regression_columns)))
        
        for i in range(len(self.regression_columns)):
            train_predictions[:, i] = self.regressor[self.regression_columns[i]].predict(self.dataset[self.parameter_columns].loc[self.train_regressor_indices]).reshape(-1)
            val_predictions[:, i] = self.regressor[self.regression_columns[i]].predict(self.dataset[self.parameter_columns].loc[self.val_regressor_indices]).reshape(-1)
            test_predictions[:, i] = self.regressor[self.regression_columns[i]].predict(self.dataset[self.parameter_columns].loc[self.test_regressor_indices]).reshape(-1)

        train_predictions = self.denormalise_regression_predictions(train_predictions)
        val_predictions = self.denormalise_regression_predictions(val_predictions)
        test_predictions = self.denormalise_regression_predictions(test_predictions)

        train_true = self.denormalise_regression_predictions(self.dataset[self.regression_columns].loc[self.train_regressor_indices].to_numpy())        
        val_true = self.denormalise_regression_predictions(self.dataset[self.regression_columns].loc[self.val_regressor_indices].to_numpy())
        test_true = self.denormalise_regression_predictions(self.dataset[self.regression_columns].loc[self.test_regressor_indices].to_numpy())
        
        for i in range(len(self.regression_columns)):
            self.train_residuals = pd.DataFrame(train_true - train_predictions, columns=[col+'_Residual' for col in self.regression_columns])
            self.val_residuals = pd.DataFrame(val_true - val_predictions, columns=[col+'_Residual' for col in self.regression_columns])
            self.test_residuals = pd.DataFrame(test_true - test_predictions, columns=[col+'_Residual' for col in self.regression_columns])
    
        del train_predictions, val_predictions, test_predictions

        if return_values:
            return self.train_predictions, self.val_residuals.copy(deep=True), self.test_residuals.copy(deep=True)

        
    def parameterise_error(self, mode="random_forest", corner=True, num_bins=20, save_path="", kwargs={}):
        """
        Trains models to predict regression residuals as a function of input parameters and creates visualization plots.
        Currently includes options but lacks actual implementations for linear interpolation, nearest interpolator, and
        support vector machines. It will optionally produce corner plot that visualize the errors as a function of 
        parameter space. 

        Parameters
        ----------
        mode : str, optional 
            Interpolator type: 'linear_interpolator', 'nearest_interpolator', 'svm', 'decision_tree', 'random_forest', or 'dnn' (default: 'random_forest')
        corner : bool, optional 
            Whether to create corner plots of residuals (default: True)
        num_bins : int, optional 
            Number of bins for corner plots (default: 20)
        save_path : str, optional 
            Directory path to save plots (default: "")
        kwargs : dict, optional 
            Additional arguments for the interpolator model (default: {})

        Returns
        -------
        None     
        """
        
        X_val_temp = self.dataset[self.parameter_columns].loc[self.val_regressor_indices].copy(deep=True).to_numpy()
        X_test_temp = self.dataset[self.parameter_columns].loc[self.test_regressor_indices].copy(deep=True).to_numpy()
        X_val_temp = pd.DataFrame(self.denormalise_point(X_val_temp, parameters=True), columns=self.parameter_columns)
        X_test_temp = pd.DataFrame(self.denormalise_point(X_test_temp, parameters=True), columns=self.parameter_columns)
        
        if mode == "linear_interpolator":
            pass
        
        elif mode == "nearest_interpolator":
            pass
        
        elif mode == "svm":
            pass
        
        elif mode == "decision_tree":
            self.interpolator_type = "decision_tree"
            self.residual_interpolator = RandomForestRegressor()
            self.residual_interpolator.fit(X_val_temp, self.val_residuals)
            
        elif mode == "random_forest":
            self.interpolator_type = "random_forest"
            for i in range(len(self.regression_columns)):
                self.residual_interpolator[self.regression_columns[i]] = RandomForestRegressor(**kwargs)
                self.residual_interpolator[self.regression_columns[i]].fit(X_val_temp.to_numpy(), self.val_residuals.iloc[:, i])

        elif mode == "dnn":
            self.interpolator_type = "dnn"
            for i in range(len(self.regression_columns)):
                model = self.create_model(style='Regressor', inputDim=len(self.parameter_columns), outputDim=1, numLayers=3, hiddenInit=64, dropoutFrac=0.2, activation='relu', parameterise=True)
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                loss = tf.keras.losses.MeanSquaredError()
                metrics = [tf.keras.metrics.MeanSquaredError()]
            
                model.compile(optimizer, loss, metrics=metrics)
                self.residual_interpolator[self.regression_columns[i]] = model                
                
                callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10), tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, monitor='val_loss', patience=5)]
                
                self.residual_interpolator[self.regression_columns[i]].fit(X_val_temp.to_numpy(), self.val_residuals.iloc[:, i], callbacks=callbacks, validation_data=(X_test_temp, self.test_residuals.iloc[:, i]), epochs=50)

                
        if corner:
            if mode == "linear_interpolator" and mode == "nearest_interpolator":
                pass
            elif mode == "svm" or mode == "decision_tree" or mode == "random_forest":
                residual_cols = [col+'_Residual' for col in self.regression_columns]
                for i in range(len(self.regression_columns)):
                    make_corner(pd.concat([X_test_temp, self.test_residuals.copy(deep=True)], axis=1), self.parameter_columns, residual_cols[i], num_bins, save_path)
                    X_test_temp['Predicted_'+residual_cols[i]] = self.residual_interpolator[self.regression_columns[i]].predict(X_test_temp[self.parameter_columns])
                    make_corner(X_test_temp, self.parameter_columns, 'Predicted_'+residual_cols[i], num_bins, save_path)
                    print(mean_squared_error(self.test_residuals[residual_cols[i]], X_test_temp['Predicted_'+residual_cols[i]], squared=False))
            elif mode == "dnn":
                residual_cols = [col+'_Residual' for col in self.regression_columns]
                for i in range(len(self.regression_columns)):
                    make_corner(pd.concat([X_test_temp, self.test_residuals.copy(deep=True)], axis=1), self.parameter_columns, residual_cols[i], num_bins, save_path)
                    X_test_temp['Predicted_'+residual_cols[i]] = self.residual_interpolator[self.regression_columns[i]](X_test_temp[self.parameter_columns])
                    make_corner(X_test_temp, self.parameter_columns, 'Predicted_'+residual_cols[i], num_bins, save_path)
                    print(mean_squared_error(self.test_residuals[residual_cols[i]], X_test_temp['Predicted_'+residual_cols[i]], squared=False))
        del X_val_temp, X_test_temp
                    

def test():
    param_set = ParameterSet.read_parameter_set('test.param_set')
    data = pd.read_csv('testData.csv')
    data = data[['ID', 'flag', 'mass', 'helium', 'metallicity', 'IBand', 'Ierr', 'VI', 'VIerr']]
    test = MESATraining(parameter_set=param_set, datafile_path="testData.csv")
    test.set_classification_columns(['flag'])
    test.set_regression_columns(['IBand', 'Ierr', 'VI', 'VIerr'])
    test.set_parameter_columns(['mass', 'helium', 'metallicity'])
    test.normalise_data()
    test.train_regressor()
    test.train_classifier()
    print(test.regressor)
    print(test.classifier)
    test_prediction = test.regressor(np.reshape(np.array([[1,1,1]*100]), (100, len(test.parameter_columns))), training=True) # This yields a Tensor object
    print(type(test_prediction))
    print(test.apply_regression_uncertainty(test.denormalise_regression_predictions(test_prediction.numpy())))
    return None


if __name__ == "__main__":
    test()
    print("-----------------------")

