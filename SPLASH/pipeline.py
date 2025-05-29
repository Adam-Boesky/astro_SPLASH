"""The SPLASH pipeline object."""
import bz2
import hashlib
import json
import os
import pickle
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pkg_resources
import sklearn
import torch
from astro_prost.associate import associate_sample
from astro_prost.helpers import SnRateAbsmag, fetch_panstarrs_sources
from astropy.coordinates import SkyCoord, Angle
from scipy.stats import gamma, halfnorm, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, MissingIndicator

from .network_utils import (ab_mag_to_flux, flux_to_ab_mag, get_intrinsic_mags,
                            get_mag_at_z, get_model, resume)
from .rf_registry import get_goodboy

torch.set_default_dtype(torch.float64)  # set the pytorch default to a float64
MODELS_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Splash_Pipeline:
    """The SPLASH classification pipeline. The default pipeline classifies supernovae with a 3-step process:
            0. Impute the nan values of the given photometry and errors.
            1. Predict the SN host's stellar mass, SFR, and redshift from its grizy absolute photometry and redshift
               using a MLP.
            2. Given SN host properties (stellar mass, SFR, redshift, angular separation), classify SNe using a
               random forest.

       Units:
            Photometry          [mJy]
            Angular Separation  [arcseconds]
            Stellar Mass        [log(solar masses)]
            Star Formation Rate [log(solar masses/yr)]
            Redshift            [redshift]
    """
    def __init__(self, pre_transformed: bool = False, within_4sigma: bool = True, nan_thresh_ratio: int = 0.5, random_seed: int = 22):

        np.random.seed(random_seed)                                                                 # why 22? Adam's lucky number

        # Pipeline configuration
        self.class_labels = ['Ia', 'Ib/c', 'SLSN', 'IIn', 'II (P/L)']                               # labels of classes that SPLASH classifies
        self.nan_thresh_ratio = nan_thresh_ratio                                                    # the maximum proprotion of nan photometric measurements we will impute
        self.nan_indicators = None                                                                  # a matrix of indicators for nans before imputation
        self.pre_transformed = pre_transformed                                                      # whether data is pre-transformed by user or not
        self.host_props = None                                                                      # variable for storing host props
        self.host_props_err = None                                                                  # errors on above
        self.within_4sigma = within_4sigma                                                          # only return classes for galaxies within 4 sigma of train data (return -1 if outside)

        # Mean and std for normalization
        self.mu_props = np.array([9.99878348, -0.13575662])  # stellar mass, SFR
        self.std_props = np.array([0.83429481, 1.04090293])
        self.mu_X = np.array([14.82211687, 15.05355227, 14.98872896, 15.23683768, 15.26239736, -0.3170728954166741])  # g,r,i,z,y,redshift
        self.std_X = np.array([0.73430271, 0.70560987, 0.7404269, 0.71824696, 0.79342847, 0.26033958098503646])

        # Hyperparameters for the MLP
        nn_hyperparams = {
            'num_inputs': 6,
            'num_outputs': 2,
            'nodes_per_layer': [6, 5, 4, 3],
            'num_linear_output_layers': 2,
            'weights_fname': os.path.join(MODELS_DIR_PATH, 'trained_models/host_prop_nn_abs_mag.pkl')
        }

        # Get the pooch object for loading RFs
        self.pooch = get_goodboy()

        # Handle some versioning due to sklearn incompatibilities
        if pkg_resources.parse_version(sklearn.__version__) >= pkg_resources.parse_version('1.3.0'):
            self.sklearn_version = 'new_skl'
            self.imputer_fname = 'knn_imputers_new_version.pkl'
            self.rf_fname = 'rf_classifier_new_version.pbz2'
        else:
            self.sklearn_version = 'old_skl'
            self.imputer_fname = 'knn_imputers_old_version.pkl'
            self.rf_fname = 'rf_classifier_old_version.pbz2'

        # Import imputer, domain transfer MLP, host prop NN, and classifier
        # 0. KNN imputers
        with open(os.path.join(MODELS_DIR_PATH, f'trained_models/{self.imputer_fname}'), 'rb') as f:
            (photo_imputer, photoerr_imputer) = pickle.load(f)
            self.photo_imputer: KNNImputer = photo_imputer
            self.photoerr_imputer: KNNImputer = photoerr_imputer
        # 1. Host property network
        self.property_predicting_net = get_model(
            num_inputs=nn_hyperparams['num_inputs'],
            num_outputs=nn_hyperparams['num_outputs'],
            nodes_per_layer=nn_hyperparams['nodes_per_layer'],
            num_linear_output_layers=nn_hyperparams['num_linear_output_layers'],
        )
        resume(self.property_predicting_net, nn_hyperparams['weights_fname'])
        self.property_predicting_net.eval()
        # 2. Classifier
        self.random_forest = self._load_rf(self.rf_fname)

        # Initialize a cache for transient catalogs so that we don't call get_transient_catalog multiple times for the
        # same input values. Also initialize a current hash for the cache so we can refer to the one just retrieved
        # more easily.
        self._current_transient_catalog_hash: Optional[str] = None
        self._transient_catalog_cache: Dict[str, pd.DataFrame] = {}

    @property
    def transient_catalog(self) -> Optional[pd.DataFrame]:
        return self._transient_catalog_cache.get(self._current_transient_catalog_hash)

    def _hash_transient_catalog_inputs(
        self,
        ra: np.ndarray,
        dec: np.ndarray,
        redshift: Optional[np.ndarray],
        names: Optional[np.ndarray],
        kwargs: Dict[str, Any]
    ) -> str:
        """Create a hash for the inputs of a given transient catalog."""
        hasher = hashlib.sha256()

        # Hash all array inputs
        for array in (ra, dec, redshift, names):
            if array is None:
                hasher.update(b"None")
            else:
                hasher.update(array.tobytes())
                hasher.update(str(array.shape).encode())
                hasher.update(str(array.dtype).encode())

        # Hash kwargs deterministically
        kwargs_serialized = json.dumps(kwargs, sort_keys=True, default=str)
        hasher.update(kwargs_serialized.encode())

        return hasher.hexdigest()

    def _load_rf(self, rf_fname: str) -> RandomForestClassifier:
        """Load random forest classifier."""
        cached_fname = self.pooch.fetch(f'trained_models/{rf_fname}')
        with bz2.BZ2File(os.path.join(cached_fname), 'rb') as f:
            return pickle.load(f)

    def _get_too_nan_rows(self, arr: np.ndarray):
        """Return mask for the array that have too many nans (>nan_thresh_ratio).

        Args:
            arr (np.ndarray): The array we want to check NaNs from.
        """
        nan_counts = np.isnan(arr).sum(axis=1)
        nan_ratios = nan_counts / arr.shape[-1]

        return nan_ratios > self.nan_thresh_ratio

    def _transform_photometry(self, X: np.ndarray, X_err: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Log transform and normalize the photometry and photometry error using the mean and variance of the
        train data. Photometry is in units of mJy.

        Args:
            X (np.ndarray): Data to transform.
            X_err (np.ndarray): Data errors to transform.
        Returns:
            Log transformed and normalized X and X_err.

        NOTE: Bands are assumed to be in the order: ['G', 'R', 'I', 'Z', 'Y']
        """

        # Convert to intrinsic magnitudes
        mags = flux_to_ab_mag(X[:, :-1])
        intrinsic_mags = get_intrinsic_mags(mags, z=X[:, -1])
        X[:, :-1] = ab_mag_to_flux(intrinsic_mags)

        # Log transform, then normalize
        if X_err is not None:
            X_err = np.abs(X_err / (X * np.log(10)))
            X_err = X_err / self.std_X
        X = np.log10(X)
        X = (X - self.mu_X) / self.std_X

        return X, X_err

    def _inverse_transform_photometry(self, X: np.ndarray, X_err: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse of above transformation.

        Args:
            X (np.ndarray): Data to transform.
            X_err (np.ndarray): Data errors to transform.
        Returns:
            Inverse transformed X and X_err.
        
        NOTE: Bands are assumed to be in the order: ['G', 'R', 'I', 'Z', 'Y']
        """

        # Inverse normalize
        X = X * self.std_X + self.mu_X

        # Inverse log transform
        X = 10 ** X

        # Inverse the transformation for errors if provided
        if X_err is not None:
            X_err = X_err * self.std_X
            X_err = np.abs(X_err * (X * np.log(10)))

        # Convert from intrinsic magnitudes
        mags = get_mag_at_z(X[:, :-1], X[:, -1])
        X[:, :-1] = mags

        return X, X_err

    def _inverse_transform_properties(self, X: np.ndarray, X_err: Optional[np.ndarray] = None) -> np.ndarray:
        """Un-normalize the properties M_* and SFR using the train mean and variances.

        Args:
            X (np.ndarray): Properties of galaxies.
            X_err (np.ndarray): Errors on the properties.
        Returns:
            The un-normalized properties of the galaxies and property errors if given.
        """
        X = X * self.std_props + self.mu_props
        if X_err is not None:
            X_err = X_err * self.std_props
            return X, X_err
        return X

    def _transform_properties(self, X: np.ndarray) -> np.ndarray:
        """Normalize the properties M_* and SFR using the train mean and variances.

        Args:
            X (np.indarray): Properties of galaxies.
        Returns:
            The un-normalized properties of the galaxies.
        """
        return (X - self.mu_props) / self.std_props

    def _impute_photometry(self, photo: np.ndarray, photoerr: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Impute the photometry.

        Args:
            photo (np.ndarray): grizy data.
            photoerr (np.ndarray): grizy errors.
        """
        self.nan_indicator_matrix = MissingIndicator(missing_values=np.NaN, features="all").fit_transform(photo)
        if photoerr is None:
            return self.photo_imputer.transform(photo), None
        return self.photo_imputer.transform(photo), self.photoerr_imputer.transform(photoerr)

    def infer_host_properties(
            self,
            ra: np.ndarray = None,
            dec: np.ndarray = None,
            grizy: np.ndarray = None,
            redshift: np.ndarray = None,
            grizy_err: Optional[np.ndarray] = None,
            redshift_err: Optional[np.ndarray] = None,
            n_resamples: int = 10,
            return_normalized: bool = False,
            return_no_host_mask: bool = False,
        ) -> np.ndarray:
        """Infer the mass, SFR, and redshift of host galaxies from either their photometry or coordinates.
        NOTE:
            (i) All photometry is in units of mJy.
            (ii) This function sets self.host_props to whatever it predicts.

        Args:
            ra (np.ndarray): The right ascension of the SNe.
            dec (np.ndarray): The declination of the SNe.
            grizy (np.ndarray): The grizy (n, 5) in mJy.
            redshift (np.ndarray): The redshift of the SNe.
            grizy_err (np.ndarray): Option for the errors on X_grizy. If given, we will resample and take 
                the median for the properties.
            redshift_err (Optional[np.ndarray]): Option for the errors on redshift. If given, we will resample
                and take the median for the properties.
            n_resamples (int): The number of samples we should produce if we are going to resample from the
                photemetry with its error.
            return_normalized (bool): Whether to return the properties in normalized form or not.
        Returns:
            1. Stellar mass [log10(solar masses)], SFR [log10(solar masses / yr)], and redshift of the given galaxies
               in an (n, 3) np.ndarray.
            2. The uncertainties on the above properties as the standard deviation across the resamples inferences.
        """
        if (ra is not None) and (dec is not None) and (grizy is None) and (grizy_err is None):
            # If ra and dec are given, use Prost to get the host photometry
            transient_catalog = self.get_transient_catalog(ra, dec, redshift=redshift, cat_cols=True, calc_host_props=True)


            def choose_column(name, df):
                if f"{name}_y" in df.columns and not df[f"{name}_y"].isnull().all():
                    return f"{name}_y"
                elif f"{name}_x" in df.columns:
                    return f"{name}_x"
                elif name in df.columns:
                    return name
                else:
                    raise KeyError(f"Could not find {name} in DataFrame")

            grizy = transient_catalog[
                [choose_column(f"{f}KronMag", transient_catalog) for f in ['g', 'r', 'i', 'z', 'y']]
            ].to_numpy()

            grizy_err = transient_catalog[
                [choose_column(f"{f}KronMagErr", transient_catalog) for f in ['g', 'r', 'i', 'z', 'y']]
            ].to_numpy()




            if redshift is None:
                redshift = transient_catalog['host_redshift_mean'].to_numpy()
                redshift_err = transient_catalog['host_redshift_std'].to_numpy()

            # Transform mag -> mJy
            grizy, grizy_err = ab_mag_to_flux(grizy, magerr=grizy_err)

            # Get the mask for the rows that had no hosts
            no_host_mask = self.transient_catalog['best_cat'].isna()
            grizy = grizy.astype(float)

            no_host_mask |= np.isnan(grizy).all(axis=1)
        else:
            if grizy.shape[1] != 5:
                raise ValueError(f'Input grizy dimensions are {X.shape} when they should be (n, 5).')
            no_host_mask = np.zeros(grizy.shape[0], dtype=bool)

        # If grizy is not given, raise an error
        if grizy is None:
            raise ValueError('Please provide the grizy photometry of the galaxies.')

        # Stack 'em up
        X = np.hstack((grizy, redshift.reshape(-1, 1)))
        if grizy_err is not None and redshift_err is not None:
            X_err = np.hstack((grizy_err, redshift_err.reshape(-1, 1)))
        elif grizy_err is None and redshift_err is not None:
            X_err = np.hstack((np.zeros(grizy.shape), redshift_err.reshape(-1, 1)))
        elif grizy_err is not None and redshift_err is None:
            X_err = np.hstack((grizy_err, np.zeros((redshift.shape[0], 1))))
        else:
            X_err = None

        # Check nans and grizy dimensions
        too_nan_mask = self._get_too_nan_rows(X)

        # Preprocess the data
        if X_err is None:
            X[:, :-1], X_err = self._impute_photometry(X[:, :-1], None)
        else:
            X[:, :-1], X_err[:, :-1] = self._impute_photometry(X[:, :-1], X_err[:, :-1])
        if not self.pre_transformed:
            X, X_err = self._transform_photometry(X, X_err)

        all_predictions = []
        if X_err is not None and n_resamples > 0:

            for _ in range(n_resamples):
                # Resample X_grizy from a normal distribution with mu=X_grizy and sigma=X_grizy_err
                X_resampled = np.random.normal(X, X_err, )

                # Predict host properties for the resampled data
                props_resampled = self.property_predicting_net(torch.from_numpy(X_resampled)).detach().numpy()
                all_predictions.append(props_resampled)

            # Property predictions are the median of the resampled results
            all_predictions_stacked = np.stack(all_predictions)
            host_props_norm = np.median(all_predictions_stacked, axis=0)
            host_props_err_norm = np.std(all_predictions_stacked, axis=0)
        else:
            host_props_norm = self.property_predicting_net(torch.from_numpy(X)).detach().numpy()
            host_props_err_norm = None

        # Fill in the rows that had too many nans or had no hosts
        host_props_norm[too_nan_mask | no_host_mask] = np.nan
        if host_props_err_norm is not None:
            host_props_err_norm[too_nan_mask | no_host_mask] = np.nan

        # Inverse transform the properties
        self.host_props, self.host_props_err = self._inverse_transform_properties(host_props_norm, X_err=host_props_err_norm)

        if return_normalized:
            if return_no_host_mask:
                return host_props_norm, host_props_err_norm, no_host_mask
            else:
                return host_props_norm, host_props_err_norm
        else:
            if return_no_host_mask:
                return self.host_props, self.host_props_err, no_host_mask
            else:
                return self.host_props, self.host_props_err

    def infer_classes(
            self,
            ra: np.ndarray = None,
            dec: np.ndarray = None,
            grizy: np.ndarray = None,
            angular_sep: np.ndarray = None,
            redshift: np.ndarray = None,
            grizy_err: Optional[np.ndarray] = None,
            redshift_err: Optional[np.ndarray] = None,
            n_resamples: int = 10,
        ):
        """Infer the classes of supernovae given their host photometry or coordinates. SNe will first be associated with
        host galaxies if just the coordinates are given. Returns -1 if hosts are outside of the train galaxy properties
        4 sigma and within_4sigma is True.

        We use the following class labels:
            0=Ia
            1=Ib/c
            2=SLSN
            3=IIn
            4=II (P/L)
            -1=Outside train properties 4 sigma
            -2=No host found
        Args:
            ra (np.ndarray): Right ascension coordinates of the SNe in degrees.
            dec (np.ndarray): Declination coordinates of the SNe in degrees.
            grizy (np.ndarray): Host grizy (n, 5).
            angular_sep (np.ndarray, optional): The angular separations of the SNe from the given galaxies in 
                arcseconds (n, 1). If not provided, will be calculated from ra/dec.
            redshift (np.ndarray, optional): Redshifts of the host galaxies. If not provided, will use values
                from transient catalog.
            grizy_err (np.ndarray, optional): Errors on the grizy photometry. If given, will resample and take
                the median for the properties.
            redshift_err (np.ndarray, optional): Errors on the redshifts. Used for resampling if provided.
            n_resamples (int, optional): The number of samples to produce when resampling from the
                photometry/redshift with their given errors. Defaults to 10.
        Returns:
            The supernova classes for the given host photometry.
        """
        # Get the host props
        host_props_norm, _, no_host_mask = self.infer_host_properties(
            ra=ra,
            dec=dec,
            grizy=grizy,
            redshift=redshift,
            grizy_err=grizy_err,
            redshift_err=redshift_err,
            n_resamples=n_resamples,
            return_normalized=True,
            return_no_host_mask=True,
        )

        # Get the angular separations and redshifts if not given
        if angular_sep is None:
            angular_sep = SkyCoord(ra, dec, unit='deg').separation(
                SkyCoord(
                    self.transient_catalog['host_ra'],
                    self.transient_catalog['host_dec'],
                    unit='deg',
                )
            ).arcsecond
        if redshift is None:
            redshift = self.transient_catalog['host_redshift_mean'].to_numpy()

        # in order: (log10(angular separation), mass, SFR, redshift)
        host_props = np.hstack((np.log10(angular_sep.reshape(-1, 1)), self.host_props, redshift.reshape(-1, 1)))

        # Get the classes
        classes = np.zeros(host_props.shape[0], dtype=int) - 2
        classes[~no_host_mask] = self.random_forest.predict(host_props[~no_host_mask])
        if self.within_4sigma:
            within_training_mask = np.all((host_props_norm < 4) & (host_props_norm > -4), axis=1)  # hosts within 4 sigma of training data
            classes[(~within_training_mask) & (~no_host_mask)] = -1

        return classes

    def infer_probs(
            self,
            ra: np.ndarray = None,
            dec: np.ndarray = None,
            grizy: np.ndarray = None,
            angular_sep: np.ndarray = None,
            redshift: np.ndarray = None,
            grizy_err: Optional[np.ndarray] = None,
            redshift_err: Optional[np.ndarray] = None,
            n_resamples: int = 10,
        ) -> Union[np.ndarray, dict]:
        """Infer the probabilities for SNe being each class from either their host photometry or SN coords. Returns -1
        if hosts are outside of the train galaxy properties 4 sigma and within_4sigma is True.

        We use the following class labels:
            0=Ia
            1=Ib/c
            2=SLSN
            3=IIn
            4=II (P/L)
            -1=Outside train properties 4 sigma

        Args:
            ra (np.ndarray, optional): Right ascension coordinates of the SNe in degrees.
            dec (np.ndarray, optional): Declination coordinates of the SNe in degrees.
            grizy (np.ndarray, optional): The grizy photometry for the given galaxies with shape (n, 5).
            angular_sep (np.ndarray, optional): The angular separations of the SNe from the given galaxies in
                arcseconds. If not provided, will be calculated from ra/dec.
            redshift (np.ndarray, optional): Redshifts of the host galaxies. If not provided, will use values
                from transient catalog.
            grizy_err (np.ndarray, optional): Errors on the grizy photometry. If given, will resample and take
                the median for the properties.
            redshift_err (np.ndarray, optional): Errors on the redshifts. Used for resampling if provided.
            n_resamples (int, optional): The number of samples to produce when resampling from the
                photometry/redshift with their given errors. Defaults to 10.
        Returns:
            The supernova classes for the given host photometry.
        """
        # Get the host props
        host_props_norm, _, no_host_mask = self.infer_host_properties(
            ra=ra,
            dec=dec,
            grizy=grizy,
            redshift=redshift,
            grizy_err=grizy_err,
            redshift_err=redshift_err,
            n_resamples=n_resamples,
            return_normalized=True,
            return_no_host_mask=True,
        )

        # Get the angular separations and redshifts if not given
        if angular_sep is None:
            angular_sep = SkyCoord(ra, dec, unit='deg').separation(
                SkyCoord(
                    self.transient_catalog['host_ra'],
                    self.transient_catalog['host_dec'],
                    unit='deg',
                )
            ).arcsecond
        if redshift is None:
            redshift = self.transient_catalog['host_redshift_mean'].to_numpy()

        # in order: (log10(angular separation), mass, SFR, redshift)
        host_props = np.hstack((np.log10(angular_sep.reshape(-1, 1)), self.host_props, redshift.reshape(-1, 1)))

        # Get the class probabilities
        all_probs = np.zeros((host_props.shape[0], 5)) * np.nan
        all_probs[~no_host_mask] = self.random_forest.predict_proba(host_props[~no_host_mask])
        if self.within_4sigma:
            within_training_mask = np.all((host_props_norm < 4) & (host_props_norm > -4), axis=1)
            all_probs[~within_training_mask] = np.nan

        return all_probs

    def get_panstarrs_photometry(
            self,
            ra: Iterable[float],
            dec: Iterable[float],
            query_radius_arcsec: float = 1.0,
        ) -> pd.DataFrame:
        """Get the photometry for the given RA and DEC from Panstarrs."""
        # Set up the photometry dataframe
        photo_cols = ['gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag', 'gKronMagErr', 'rKronMagErr',
                      'iKronMagErr', 'zKronMagErr', 'yKronMagErr']
        pstarr_photo_df = pd.DataFrame(columns=photo_cols)

        # Get the photometry for each coordinate and append to the dataframe
        coords = SkyCoord(ra, dec, unit='deg')
        search_rad = Angle(query_radius_arcsec, unit='arcsec')
        for coord in coords:

            # If the coordinate is nan, append a row of nans
            if np.isnan(coord.ra) or np.isnan(coord.dec):
                pstarr_photo_df = pd.concat([pstarr_photo_df, pd.DataFrame(data={col: [np.nan] for col in photo_cols})])
                continue

            # Get the photometry and append to the dataframe
            new_src_df = fetch_panstarrs_sources(
                coord,
                search_rad,
                cat_cols=True,
                calc_host_props=[],
                release='dr2',
            )
            if new_src_df is None:
                pstarr_photo_df = pd.concat([pstarr_photo_df, pd.DataFrame(data={col: [np.nan] for col in photo_cols})])
            else:
                pstarr_photo_df = pd.concat([pstarr_photo_df, new_src_df[photo_cols].iloc[0:1]])

        # Replace negative fluxes with nans
        if pstarr_photo_df is not None:
            pstarr_photo_df.replace(-999, np.nan, inplace=True)
            pstarr_photo_df.reset_index(drop=True, inplace=True)

        return pstarr_photo_df

    def get_transient_catalog(
            self,
            ra: np.ndarray,
            dec: np.ndarray,
            redshift: Optional[np.ndarray] = None,
            names: Optional[np.ndarray] = None,
            **kwargs,
        ) -> pd.DataFrame:
        """Given the RA and DEC of the transients in degrees, get the transient catalog using Prost."""
        # Check if we have already gotten the catalog for the given inputs
        cat_hash = self._hash_transient_catalog_inputs(ra, dec, redshift, names, kwargs)
        if cat_hash in self._transient_catalog_cache:
            return self._transient_catalog_cache[cat_hash]

        # Define priors and likelihoods
        priorfunc_z = halfnorm(loc=0.0001, scale=0.5)
        priorfunc_offset = uniform(loc=0, scale=10)
        priorfunc_absmag = uniform(loc=-30, scale=20)
        likefunc_offset = gamma(a=0.75)
        likefunc_absmag = SnRateAbsmag(a=-25, b=20)
        priors = {"offset": priorfunc_offset, "absmag": priorfunc_absmag, "redshift": priorfunc_z}
        likes = {"offset": likefunc_offset, "absmag": likefunc_absmag}

        if names is None:
            names = [f'SN{i}' for i in range(len(ra))]

        # Set up the catalog
        transient_catalog_no_hosts = pd.DataFrame(data={
            'name': names,  # arbitrary names
            'RA': ra,
            'DEC': dec,
        })
        if redshift is not None:
            transient_catalog_no_hosts['redshift'] = redshift

        # Associate the sample
        print('Associating the catalog!')

        transient_catalog: pd.DataFrame = associate_sample(
            transient_catalog_no_hosts,
            name_col='name',
            redshift_col='redshift',
            coord_cols=('RA', 'DEC'),
            priors=priors,
            likes=likes,
            catalogs=['glade', 'decals','panstarrs'],
            save=False,
            verbose=0,
            **kwargs,
        )

        # Query panstarrs for the photometry
        panstarrs_photo_df = self.get_panstarrs_photometry(
            transient_catalog['host_ra'],
            transient_catalog['host_dec'],
            query_radius_arcsec=3.0,
        )

        # Merge the photometry with the transient catalog
        transient_catalog = pd.merge(
            transient_catalog,
            panstarrs_photo_df,
            how='left',
            left_index=True,
            right_index=True,
        )

        # Make sure all the names are in the catalog
        transient_catalog = pd.merge(
            transient_catalog_no_hosts,
            transient_catalog,
            on='name',
            how='left'
        )

        # Reorder the catalog by the initial order of input names
        transient_catalog = transient_catalog.set_index('name', drop=True).loc[names].reset_index(drop=True)

        # Cache the catalog
        self._transient_catalog_cache[cat_hash] = transient_catalog
        self._current_transient_catalog_hash = cat_hash

        return transient_catalog