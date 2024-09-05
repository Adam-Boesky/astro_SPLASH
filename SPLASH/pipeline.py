import os
import bz2
import torch
import pickle
import numpy as np
import sklearn
import pkg_resources

from typing import Union, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, MissingIndicator
from .network_utils import resume, get_model
from .rf_registry import get_goodboy

torch.set_default_dtype(torch.float64)  # set the pytorch default to a float64
MODELS_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Splash_Pipeline:
    """Our classification pipeline. The defaul pipeline classifies supernovae with a 3-step process:
            0. Impute the nan values of the given photometry and errors.
            1. Given grizy photometry of a SN host, conduct domain transfer by predicted 13 more
               photometric bands with an MLP.
            2. Predict the SN host's stellar mass, SFR, and redshift from its 18-band photometry
               using a MLP.
            3. Give the properties and the SN-host angular separation to a random forest to infer
               SN class.

       Units:
            Photometry          [mJy]
            Angular Separation  [arcseconds]
            Stellar Mass        [log(solar masses)]
            Star Formation Rate [log(solar masses/yr)]
            Redshift            [redshift]
    """
    def __init__(self, pipeline_version: str = 'weighted_full_band', pre_transformed: bool = False, within_4sigma: bool = True, nan_thresh_ratio: int = 0.5, random_seed: int = 22):

        # Get users choice of host prop predictor
        if pipeline_version not in ['full_band', 'best_band', 'weighted_full_band']:
            raise ValueError(f'{pipeline_version} is not an option for your pipeline version, it must either be \'weighted_full_band\' or \'best_band\' or \'full_band\'')
        self.version = pipeline_version

        # Load the hyperparameters for given pipeline version
        if self.version == 'weighted_full_band':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':13, 'nodes_per_layer': [7, 9, 11, 13], 'num_linear_output_layers': 1,
                                           'weights_fname': os.path.join(MODELS_DIR_PATH, 'trained_models/best_sed_model.pkl')}
            host_prop_hyperparams = {'num_inputs': 18, 'num_outputs': 3, 'nodes_per_layer': [18, 15, 12, 9, 6, 4], 'num_linear_output_layers': 3,
                                     'weights_fname': os.path.join(MODELS_DIR_PATH, 'trained_models/powlaw_n6_weighted_host_prop_best_model.pkl')} 
        elif self.version == 'full_band':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':13, 'nodes_per_layer': [7, 9, 11, 13], 'num_linear_output_layers': 1,
                                           'weights_fname': os.path.join(MODELS_DIR_PATH, 'trained_models/best_sed_model.pkl')}
            host_prop_hyperparams = {'num_inputs': 18, 'num_outputs': 3, 'nodes_per_layer': [18, 15, 12, 9, 6, 4], 'num_linear_output_layers': 3,
                                     'weights_fname': os.path.join(MODELS_DIR_PATH, 'trained_models/best_model.pkl')}
        elif self.version == 'best_band':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':9, 'nodes_per_layer': [6, 7, 7, 8], 'num_linear_output_layers': 2,
                                           'weights_fname':  os.path.join(MODELS_DIR_PATH, 'trained_models/V2_best_sed_model.pkl')}
            host_prop_hyperparams = {'num_inputs': 14, 'num_outputs': 3, 'nodes_per_layer': [12, 10, 8, 6, 4], 'num_linear_output_layers': 3,
                                     'weights_fname':  os.path.join(MODELS_DIR_PATH, 'trained_models/V2_host_prop_best_model.pkl')}

        # Get the pooch object for loading RFs
        self.pooch = get_goodboy()

        # Import imputer, domain transfer MLP, host prop NN, and classifier
        # 0. KNN imputers
        with open(os.path.join(MODELS_DIR_PATH, 'trained_models/knn_imputers.pkl'), 'rb') as f:
            (photo_imputer, photoerr_imputer) = pickle.load(f)
            self.photo_imputer: KNNImputer = photo_imputer
            self.photoerr_imputer: KNNImputer = photoerr_imputer
        # 1. Domain transfer network
        self.domain_transfer_net = get_model(num_inputs=domain_transfer_hyperparams['num_inputs'], num_outputs=domain_transfer_hyperparams['num_outputs'], nodes_per_layer=domain_transfer_hyperparams['nodes_per_layer'], num_linear_output_layers=domain_transfer_hyperparams['num_linear_output_layers'])
        resume(self.domain_transfer_net, domain_transfer_hyperparams['weights_fname'])
        self.domain_transfer_net.eval()
        # 2. Host property network
        self.property_predicting_net = get_model(num_inputs=host_prop_hyperparams['num_inputs'], num_outputs=host_prop_hyperparams['num_outputs'], nodes_per_layer=host_prop_hyperparams['nodes_per_layer'], num_linear_output_layers=host_prop_hyperparams['num_linear_output_layers'])
        resume(self.property_predicting_net, host_prop_hyperparams['weights_fname'])
        self.property_predicting_net.eval()
        # 3. Classifier
        if pkg_resources.parse_version(sklearn.__version__) >= pkg_resources.parse_version('1.3.0'):
            self.rf_fname = 'rf_classifier_new_version.pbz2'
        else:
            self.rf_fname = 'rf_classifier_old_version.pbz2'
        self.random_forest = self._load_rf(self.rf_fname)

        # Pipeline configuration
        self.class_labels = ['Ia', 'Ib/c', 'SLSN', 'IIn', 'II (P/L)']                               # labels of classes that SPLASH classifies
        self.nan_thresh_ratio = nan_thresh_ratio                                                    # the maximum proprotion of nan photometric measurements we will impute
        self.nan_indicators = None                                                                  # a matrix of indicators for nans before imputation
        self.pre_transformed = pre_transformed                                                      # whether data is pre-transformed by user or not
        self.host_props = None                                                                      # variable for storing host props
        self.host_props_err = None                                                                  # errors on above
        self.within_4sigma = within_4sigma                                                          # only return classes for galaxies within 4 sigma of train data (return -1 if outside)
        self.mu_props = np.array([8.87088133, -0.46037044, 0.58991822])                             # means of the train props
        self.std_props = np.array([1.08494612, 1.04024203, 0.2700682 ])                             # stds of the train props
        self.mu_phot = np.array([-2.87849839, -2.79518071, -2.78359176, -2.63765973, -2.66592091,   # means of the train photometry
                           -1.33953981, 0.06383305, 0.67075717, 1.34771103, 0.91141463,
                           1.33652476, 1.2116005, 1.32755404, -2.41376848, -2.20197162,
                           -1.97797089, -3.03436027, -3.03643454])
        self.std_phot = np.array([0.63162979, 0.70187988, 0.68387317, 0.72578209, 0.7777774,        # stds of the train photometry
                            0.88136263, 1.62222597, 2.06392269, 0.43210206, 1.36265575,
                            0.50050502, 0.43821229, 0.46058673, 2.21595863, 2.13270738,
                            2.02278681, 0.67246779, 0.70040306])
        self._ovr_classifiers = None                                                                # one vs. rest RF classifiers
        np.random.seed(random_seed)                                                                 # why 22? my lucky number

    @property
    def ovr_classifiers(self) -> Dict[str, RandomForestClassifier]:
        """The one-versus-rest random forest classifiers."""
        if not self._ovr_classifiers:
            self._ovr_classifiers = {}
            for lab in self.class_labels:
                fname_lab = f"{lab.lower().replace(' ', '_').replace('/', '&').replace('i', 'I')}_rf.pbz2"
                self._ovr_classifiers[lab] = self._load_rf(f'ovr_rfs/{fname_lab}')

        return self._ovr_classifiers

    def _load_rf(self, rf_fname: str) -> RandomForestClassifier:
        """Load random forest classifier."""
        cached_fname = self.pooch.fetch(f'trained_models/{rf_fname}')
        with bz2.BZ2File(os.path.join(cached_fname), 'rb') as f:
            return pickle.load(f)

    def _check_nans(self, arr: np.ndarray):
        """Raise an error if the given arr has a ratio of nans greater than self.nan_thresh_ratio.

        Args:
            arr (np.ndarray): The array we want to check NaNs from.
        """
        nan_counts = np.isnan(arr).sum(axis=1)
        nan_ratios = nan_counts / arr.shape[-1]

        if np.any(nan_ratios > self.nan_thresh_ratio):
            raise ValueError(f'The ratio of NaNs to total values exceeds the acceptable ratio of {self.nan_thresh_ratio} for some \
                             photometry. Please handle these values before passing into the pipeline.')

    def _transfer_domain(self, X_grizy: np.ndarray) -> np.ndarray:
        """Helper function for conduction domain transfer.

        Args:
            X_grizy (np.ndarray): The grizy data for the give galaxies.
        Returns:
            The full photometric band of the galaxy as a np.ndarray.
        """
        other_bands = self.domain_transfer_net(X_grizy).detach().numpy()
        X_full_band = np.hstack((X_grizy, other_bands))
        return X_full_band

    def _transform_photometry(self, X: np.ndarray, X_err: np.ndarray = None, just_grizy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Log transform and normalize the photometry and photometry error using the mean and variance of the
        train data. Photometry is in units of mJy.

        Args:
            X (np.ndarray): Data to transform.
            X_err (np.ndarray): Data errors to transform.
            just_grizy (bool): Whether the data is just grizy or all 18 bands.
        Returns:
            Log transformed and normalized X and X_err.
        
        NOTE: Bands are assumed to be in the order: ['G', 'R', 'I', 'Z', 'Y', 'J', 'H',
                                                     'Ks', 'CH1', 'CH2', 'MIPS24','MIPS70',
                                                     'PACS100', 'MIPS160', 'PACS160',
                                                     'SPIRE250', 'SPIRE350', 'SPIRE500']
        """
        n_bands = 5 if just_grizy else 18

        # Log transform, then normalize
        if X_err is not None:
            X_err = np.abs(X_err[:, : n_bands] / (X[:, : n_bands] * np.log(10)))
            X_err = X_err / self.std_phot[: n_bands]
        X = np.log10(X)
        X = (X - self.mu_phot[: n_bands]) / self.std_phot[: n_bands]

        return X, X_err

    def _inverse_transform_photometry(self, X: np.ndarray, X_err: np.ndarray = None, just_grizy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse of above transformation.

        Args:
            X (np.ndarray): Data to transform.
            X_err (np.ndarray): Data errors to transform.
            just_grizy (bool): Whether the data is just grizy or all 18 bands.
        Returns:
            Inverse transformed X and X_err.
        
        NOTE: Bands are assumed to be in the order: ['G', 'R', 'I', 'Z', 'Y', 'J', 'H',
                                                     'Ks', 'CH1', 'CH2', 'MIPS24','MIPS70',
                                                     'PACS100', 'MIPS160', 'PACS160',
                                                     'SPIRE250', 'SPIRE350', 'SPIRE500']
        """
        n_bands = 5 if just_grizy else 18

        # Inverse normalize
        X = X * self.std_phot[:n_bands] + self.mu_phot[:n_bands]

        # Inverse log transform
        X = 10 ** X

        # Inverse the transformation for errors if provided
        if X_err is not None:
            X_err = X_err * self.std_phot[:n_bands]
            X_err = np.abs(X_err[:, :n_bands] * (X[:, :n_bands] * np.log(10)))

        return X, X_err

    def _inverse_tranform_properties(self, X: np.ndarray, X_err: np.ndarray = None) -> np.ndarray:
        """Un-normalize the properties M_*, SFR, and redshift using the train mean and variances.

        Args:
            X (np.ndarray): Properties of galaxies.
        Returns:
            The un-normalized properties of the galaxies.
        """
        X = X * self.std_props + self.mu_props
        X_err = X_err * self.std_props
        return X, X_err

    def _tranform_properties(self, X: np.ndarray) -> np.ndarray:
        """Normalize the properties M_*, SFR, and redshift using the train mean and variances.

        Args:
            X (np.ndarray): Properties of galaxies.
        Returns:
            The un-normalized properties of the galaxies.
        """
        return (X - self.mu_props) / self.std_props

    def _impute_photometry(self, photo: np.ndarray, photoerr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Impute the photometry.

        Args:
            photo (np.ndarray): grizy data.
            photoerr (np.ndarray): grizy errors.
        """
        self.nan_indicator_matrix = MissingIndicator(missing_values=np.NaN, features="all").fit_transform(photo)
        return self.photo_imputer.transform(photo), self.photoerr_imputer.transform(photoerr)

    def infer_sed(self, X_grizy: np.ndarray, X_grizy_err: Optional[np.ndarray] = None, n_resamples: int = 10, return_normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the full 18-wavelength SED given the grizy data.

        Bands are ['G', 'R', 'I', 'Z', 'Y', 'J', 'H', 'Ks', 'CH1', 'CH2', 'MIPS24','MIPS70', 'PACS100', 'MIPS160',
                    'PACS160','SPIRE250', 'SPIRE350', 'SPIRE500']

        Args:
            X_grizy (np.ndarray): Either the grizy (n, 5) or full band photometry for the given galaxies (n, 18) in mJy.
            X_grizy_err (np.ndarray): Option for the errors on X_grizy. If given, we will resample and take 
                the median for the properties.
            n_resamples (int): The number of samples we should produce if we are going to resample from the
                photemetry with its given error.
        Returns:
            1. The 18 band photometry.
            2. The errors on the 18 bands.
        """
        # Check nans and grizy dimensions
        self._check_nans(X_grizy)
        if X_grizy.shape[1] != 5:
            raise ValueError(f'Input grizy dimensions are {X_grizy.shape} when they should be either (n, 5).')

        # Preprocess the data
        if self.version != 'best_band':  # imputation not implemented for best_band version
            X_grizy, X_grizy_err = self._impute_photometry(X_grizy, X_grizy_err)
        if not self.pre_transformed:
            X_grizy, X_grizy_err = self._transform_photometry(X_grizy, X_grizy_err, just_grizy=True)

        all_seds = []
        if X_grizy_err is not None:
            # Check grizy error dimensions
            if X_grizy_err.shape[1] not in [5, 18]:
                raise ValueError(f'Input grizy error dimensions are {X_grizy_err.shape} when they should be either (n, 5) or (n, 18).')

            for _ in range(n_resamples):
                # Resample X_grizy from a normal distribution with mu=X_grizy and sigma=X_grizy_err
                X_resampled = np.random.normal(X_grizy, X_grizy_err, )
                X_resampled = self._transfer_domain(torch.from_numpy(X_resampled))
                all_seds.append(X_resampled)
            all_seds = np.stack(all_seds)
            all_seds_err = np.std(all_seds, axis=0)
            all_seds = np.median(all_seds, axis=0)

        if not return_normalized:
            return self._inverse_transform_photometry(all_seds, all_seds_err, just_grizy=False)
        return all_seds, all_seds_err

    def predict_host_properties(self, X_grizy: np.ndarray, X_grizy_err: Optional[np.ndarray] = None, n_resamples: int = 10, return_normalized: bool = False) -> np.ndarray:
        """Predict the mass, SFR, and redshift of host galaxies from their photometry. Based on the dimensions of 
        the given photometry, this function will either conduct domain transfer or not.
        NOTE: 
            (i) All photometry is in units of mJy.
            (ii) This function sets self.host_props to whatever it predicts.

        Args:
            X_grizy (np.ndarray): Either the grizy (n, 5) or full band photometry for the given galaxies (n, 18) in mJy.
            X_grizy_err (np.ndarray): Option for the errors on X_grizy. If given, we will resample and take 
                the median for the properties.
            n_resamples (int): The number of samples we should produce if we are going to resample from the
                photemetry with its given error.
        Returns:
            1. Stellar mass [log10(solar masses)], SFR [log10(solar masses / yr)], and redshift of the given galaxies
               in an (n, 3) np.ndarray.
            2. The uncertainties on the above properties as the standard deviation across the resamples inferences.
        """

        # Check nans and grizy dimensions
        self._check_nans(X_grizy)
        if X_grizy.shape[1] not in [5, 18]:
            raise ValueError(f'Input grizy dimensions are {X_grizy.shape} when they should be either (n, 5) or (n, 18).')

        # Boolean for if we are doing domain transfer
        domain_transfer = (X_grizy.shape[1] == 5)

        # Preprocess the data
        if self.version != 'best_band' and domain_transfer:  # imputation not implemented for best_band version
            X_grizy, X_grizy_err = self._impute_photometry(X_grizy, X_grizy_err)
        if not self.pre_transformed:
            X_grizy, X_grizy_err = self._transform_photometry(X_grizy, X_grizy_err, just_grizy=domain_transfer)

        all_predictions = []
        if X_grizy_err is not None:
            # Check grizy error dimensions
            if X_grizy_err.shape[1] not in [5, 18]:
                raise ValueError(f'Input grizy error dimensions are {X_grizy_err.shape} when they should be either (n, 5) or (n, 18).')

            for _ in range(n_resamples):
                # Resample X_grizy from a normal distribution with mu=X_grizy and sigma=X_grizy_err
                X_resampled = np.random.normal(X_grizy, X_grizy_err, )
                if domain_transfer:
                    X_resampled = self._transfer_domain(torch.from_numpy(X_resampled))

                # Predict host properties for the resampled data
                props_resampled = self.property_predicting_net(torch.from_numpy(X_resampled)).detach().numpy()
                all_predictions.append(props_resampled)

            # Property predictions are the median of the resampled results
            all_predictions_stacked = np.stack(all_predictions)
            host_props_norm = np.median(all_predictions_stacked, axis=0)
            host_props_err_norm = np.std(all_predictions_stacked, axis=0)
        else:
            if domain_transfer:
                X_grizy = self._transfer_domain(torch.from_numpy(X_grizy))

            # Predict host properties
            host_props_norm = self.property_predicting_net(torch.from_numpy(X_grizy)).detach().numpy()
            host_props_err_norm = None
        self.host_props, self.host_props_err = self._inverse_tranform_properties(host_props_norm, X_err=host_props_err_norm)

        if return_normalized:
            return host_props_norm, host_props_err_norm
        else:
            return self.host_props, self.host_props_err

    def predict_classes(self, X_grizy: np.ndarray, angular_sep: np.ndarray, X_grizy_err: np.ndarray = None, n_resamples: int = 10):
        """Predict the class of a supernova given its host photometry. Based on the dimensions of the given
        photometry, this function will either conduct domain transfer or not. Returns -1 if hosts are outside
        of the train galaxy properties 4 sigma and within_4sigma is True.

        We use the following class labels:
            0=Ia
            1=Ib/c
            2=SLSN
            3=IIn
            4=II (P/L)
            -1=Outside train properties 4 sigma

        Args:
            X_grizy (np.ndarray): Either the grizy (n, 5) or full band photometry for the given galaxies (n, 18).
            angular_sep (np.ndarray): The angular separations of the SNe from the given galaxies in 
                arcseconds (n, 1).
            X_grizy_err (np.ndarray): Option for the errors on X_grizy. If given, we will resample and take 
                the median for the properties.
            n_resamples (int): The number of samples we should produce if we are going to resample from the
                photemetry with its given error.
        Returns:
            The supernova classes for the given host photometry.
        """
        # Get the host props
        host_props_norm, host_props_err_norm = self.predict_host_properties(X_grizy, X_grizy_err, n_resamples, return_normalized=True)
        host_props = np.hstack((np.log10(angular_sep.reshape(-1, 1)), self.host_props))  # in order: (log10(angular separation), mass, SFR, redshift)

        # Get the classes
        classes = self.random_forest.predict(host_props)
        if self.within_4sigma:
            within_training_mask = np.all((host_props_norm < 4) & (host_props_norm > -4), axis=1)  # hosts within 4 sigma of training data
            classes[~within_training_mask] = -1

        return classes

    def predict_probs(self, X_grizy: np.ndarray, angular_sep: np.ndarray, X_grizy_err: np.ndarray = None, n_resamples: int = 10, ovr: bool = False) -> Union[np.ndarray, dict]:
        """Predict the class of a supernova given its host photometry. Based on the dimensions of the given
        photometry, this function will either conduct domain transfer or not. Returns -1 if hosts are outside
        of the train galaxy properties 4 sigma and within_4sigma is True.

        We use the following class labels:
            0=Ia
            1=Ib/c
            2=SLSN
            3=IIn
            4=II (P/L)
            -1=Outside train properties 4 sigma

        Args:
            X_grizy (np.ndarray): Either the grizy (n, 5) or full band photometry for the given galaxies (n, 18).
            angular_sep (np.ndarray): The angular separations of the SNe from the given galaxies in 
                arcseconds (n, 1).
            X_grizy_err (np.ndarray): Option for the errors on X_grizy. If given, we will resample and take 
                the median for the properties.
            n_resamples (int): The number of samples we should produce if we are going to resample from the
                photemetry with its given error.
            ovr (bool): If true, return the one-versus-rest probability for each class in the form of a 
                dictionary.     e.g. {'Ia': 0.9, 'Ib/c': 0.2, ...}
        Returns:
            The supernova classes for the given host photometry.
        """
        # Get the host props
        host_props_norm, _ = self.predict_host_properties(X_grizy, X_grizy_err, n_resamples, return_normalized=True)
        host_props = np.hstack((np.log10(angular_sep.reshape(-1, 1)), self.host_props))  # in order: (log10(angular separation), mass, SFR, redshift)

        # Get the class probabilities
        if not ovr:
            all_probs = self.random_forest.predict_proba(host_props)
            if self.within_4sigma:
                within_training_mask = np.all((host_props_norm < 4) & (host_props_norm > -4), axis=1)
                all_probs[~within_training_mask] = np.nan
        else:
            all_probs = {}
            for class_lab, class_rf in self.ovr_classifiers.items():
                all_probs[class_lab] = class_rf.predict_proba(host_props)[:, np.where(class_rf.classes_ == 1)[0][0]]

        return all_probs
