"""File used for preprocessing data from https://zenodo.org/communities/ddfdata/?page=1&size=20
Number of observations in each field:
es1     826242
wcdfs   799607
xmmlss  1247954
"""
import os
import sys

sys.path.append('/Users/adamboesky/Research/ay98/Weird_Galaxies')

import pickle
from pathlib import Path
from typing import Callable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from sklearn.impute import KNNImputer

from logger import get_clean_logger

SURVEY = 'wcdfs'
LOG = get_clean_logger(logger_name = Path(__file__).name, log_filename=f'/Users/adamboesky/Research/ay98/Weird_Galaxies/Paper/preprocessing/preprocessing_{SURVEY}.log')
PATH_TO_DATA = '/Volumes/T7/ay_98_data/Zou_data'
PATH_TO_TRAINING_DATA = 'Volumes/T7/ay_98_data/pickled_data'
FILTER_WAVELENGTHS = {
    'f_MIPS24': 24000,
    'f_MIPS70': 70000,
    'f_MIPS160': 160000,
    'f_PACS100': 100000,
    'f_PACS160': 160000,
    'f_SPIRE250': 250000,
    'f_SPIRE350': 350000,
    'f_SPIRE500': 500000,
    'mag_U_VOICE': 365, #nm
    'mag_G_DES': 480,
    'mag_R_DES': 630,
    'mag_R_VOICE': 630,
    'mag_I_DES': 775,
    'mag_Z_DES': 900,
    'mag_Y_DES': 980,
    'mag_Z_VIDEO': 900,
    'mag_Y_VIDEO': 980,
    'mag_J_VIDEO': 1250,
    'mag_H_VIDEO': 1650,
    'mag_Ks_VIDEO': 2150,
    'mag_CH1': 3550,
    'mag_CH2': 4500,
    'mag_U_CFHT': 365,
    'mag_G_HSC': 480,
    'mag_R_HSC': 630,
    'mag_I_HSC': 775,
    'mag_Z_HSC': 900,
    'mag_Y_HSC': 980
}
# {
#     'f_MIPS24': 24000,
#     'f_MIPS70': 70000,
#     'f_MIPS160': 160000,
#     'f_PACS100': 100,
#     'f_PACS160': 160,
#     'f_SPIRE250': 250,
#     'f_SPIRE350': 350,
#     'f_SPIRE500': 500,
#     'mag_U_VOICE': 365,   # U band
#     'mag_G_DES': 480,     # G band
#     'mag_R_DES': 640,     # R band
#     'mag_I_DES': 800,     # I band
#     'mag_Z_DES': 920,     # Z band for DES
#     'mag_Y_DES': 1000,    # Y band for DES
#     'mag_Z_VIDEO': 900,   # Z band for VIDEO
#     'mag_Y_VIDEO': 1020,  # Y band for VIDEO
#     'mag_J_VIDEO': 1250,  # J band
#     'mag_H_VIDEO': 1630,  # H band
#     'mag_Ks_VIDEO': 2190, # Ks band
#     'mag_CH1': 3560,      # IRAC Channel 1
#     'mag_CH2': 4510       # IRAC Channel 2
#}  # units are nm
ALL_SURVEY_KEYS = {
    'es1': [
        'f_MIPS24',
        'f_MIPS70',
        'f_MIPS160',
        'f_PACS100',
        'f_PACS160',
        'f_SPIRE250',
        'f_SPIRE350',
        'f_SPIRE500',
        'mag_U_VOICE',
        'mag_G_DES',
        'mag_R_DES',
        'mag_I_DES',
        'mag_Z_DES',
        'mag_Y_DES',
        'mag_Z_VIDEO',
        'mag_Y_VIDEO',
        'mag_J_VIDEO',
        'mag_H_VIDEO',
        'mag_Ks_VIDEO',
        'mag_CH1',
        'mag_CH2'],
    'wcdfs': [
        'f_MIPS24',
        'f_MIPS70',
        'f_MIPS160',
        'f_PACS100',
        'f_PACS160',
        'f_SPIRE250',
        'f_SPIRE350',
        'f_SPIRE500',
        # 'mag_U_VOICE',  # over 50% nans
        # 'mag_G_VOICE',  # over 50% nans
        'mag_R_VOICE',
        # 'mag_I_VOICE',  # over 50% nans
        'mag_G_HSC',
        # 'mag_R_HSC',    # over 50% nans
        'mag_I_HSC',
        'mag_Z_HSC',
        # 'mag_Z_VIDEO',  # over 50% nans
        'mag_Y_VIDEO',
        'mag_J_VIDEO',
        'mag_H_VIDEO',
        'mag_Ks_VIDEO',
        'mag_CH1',
        'mag_CH2'
    ],
    'xmmlss': [
        'f_MIPS24',
        'f_MIPS70',
        'f_MIPS160',
        'f_PACS100',
        'f_PACS160',
        'f_SPIRE250',
        'f_SPIRE350',
        'f_SPIRE500',
        'mag_U_CFHT',
        'mag_G_HSC',
        'mag_R_HSC',
        'mag_I_HSC',
        'mag_Z_HSC',
        'mag_Y_HSC',
        'mag_Z_VIDEO',
        'mag_Y_VIDEO',
        'mag_J_VIDEO',
        'mag_H_VIDEO',
        'mag_Ks_VIDEO',
        'mag_CH1',
        'mag_CH2'
    ]
}


def check_same_order(arr1: np.ndarray, arr2: np.ndarray, key: str = 'Tractor_ID'):
    """Check that two numpy arrays are ordered in the same way based on some ID value."""
    if len(arr1) != len(arr2):
        raise Exception(f"Data arrays are different lengths: {len(arr1)} != {len(arr2)}")
    if not np.all(arr1[key].astype(int) == arr2[key].astype(int)):
        raise Exception("Data arrays are not the same order!")
    pass


def impute(data: np.ndarray, n_batches=1, verbose: bool = True) -> np.ndarray:
    """Impute given np array. Option of imputing with batches"""
    batch_size = int(len(data) / n_batches)
    imputer = KNNImputer(n_neighbors=5)
    imputed = None
    for i_batch in range(n_batches):
        if i_batch == n_batches - 1:
            batch = data[i_batch * batch_size:]  # Last batch gets everything
        else:
            batch = data[i_batch * batch_size : (i_batch + 1) * batch_size]
        imputed_batch = imputer.fit_transform(batch)

        # If no batches have been imputed yet
        if imputed is None:
            imputed = imputed_batch
        else:
            imputed = np.concatenate((imputed, imputed_batch))
        if verbose: LOG.info('Done imputing batch %i / %i', i_batch + 1, n_batches)
    return imputed


def get_photometry_np_from_fits(filepath: Union[str, Path], columns: list, transforms: List[Union[None, Callable]], vector_key: Optional[str] = None):
    """Get given photometry fields in a numpy array from fits file path. Apply given transforms to the data. Return a vector of a column if given."""
    fits_data = np.array(fits.open(os.path.join(PATH_TO_DATA, filepath))[1].data)
    LOG.info('Importing %s', filepath)

    # # Add ids
    n = len(fits_data)
    # fits_data[vector_key] = np.arange(0, n, 1, dtype=int)  # unique id values

    # Define new dtype with additional column for unique IDs
    new_dtype = np.dtype(fits_data.dtype.descr + [(vector_key, int)])
    new_data = np.zeros(len(fits_data), dtype=new_dtype)
    for name in fits_data.dtype.names:
        new_data[name] = fits_data[name]
    new_data[vector_key] = np.arange(len(fits_data))
    fits_data = new_data



    # Drop any severely Nan columns (>50%)
    good_columns = []
    good_transforms = []
    for col, trans in zip(columns, transforms):
        num_nans = np.count_nonzero(np.isnan(fits_data[col]))
        if num_nans > 0.5 * n:
            LOG.info('Dropping column %s with %i / %i = %.2f%% NaNs', col, num_nans, n, (num_nans / n) * 100)
        else:
            LOG.info('Keeping column %s with %i / %i = %.2f%% NaNs', col, num_nans, n, (num_nans / n) * 100)
            good_columns.append(col)
            good_transforms.append(trans)
    
    # Drop severly Nan rows (>50%)
    temp_arr = np.array([fits_data[col] for col in good_columns]).T
    nan_mask = np.sum(np.isnan(temp_arr), axis=1) <= temp_arr.shape[1] / 2
    fits_data = fits_data[nan_mask]
    LOG.info('Dropped %i rows because more than half of the values were NaN', np.sum(~nan_mask))

    # Transform the data
    out_arr = np.array([fits_data[col] for col in good_columns]).T
    out_arr_imputed = impute(out_arr, n_batches=50)
    if out_arr.shape != out_arr_imputed.shape:
        raise ValueError(f'Shape of imputed ({out_arr_imputed.shape}) is different from original ({out_arr.shape})')

    for idx, trans in enumerate(good_transforms):
        if trans is not None:
            out_arr_imputed[:,idx] = trans(out_arr_imputed[:,idx])

    # If given vector key, return vector as well as data
    if vector_key is not None:
        return out_arr_imputed, fits_data[[vector_key]]
    return out_arr_imputed


def get_catalog_np_from_fits(filepath: Union[str, Path], columns: list, transforms: List[Union[None, Callable]], vector_key: Optional[str] = None):
    """Get given catalog fields in a numpy array from fits file path. Apply given transforms to the data. Return a vector of a column if given."""
    fits_data = np.array(fits.open(os.path.join(PATH_TO_DATA, filepath))[1].data)
    LOG.info('Importing %s', filepath)

    # Add ids
    n = len(fits_data)
    # fits_data[vector_key] = np.arange(0, n, 1, dtype=int)  # unique id values
    new_dtype = np.dtype(fits_data.dtype.descr + [(vector_key, int)])
    new_data = np.zeros(len(fits_data), dtype=new_dtype)
    for name in fits_data.dtype.names:
        new_data[name] = fits_data[name]
    new_data[vector_key] = np.arange(len(fits_data))
    fits_data = new_data

    # Drop any severely Nan arrays (>50%)
    temp_arr = np.array([fits_data[col] for col in columns]).T
    nan_row_inds = np.where(np.any(np.isnan(temp_arr), axis=1))[0]
    LOG.info('Dropping %i rows because they contain nans', len(nan_row_inds))
    fits_data = np.delete(fits_data, nan_row_inds, axis=0)

    # Transform the data
    out_arr = np.array([fits_data[col] for col in columns]).T
    for idx, trans in enumerate(transforms):
        if trans is not None:
            out_arr[:,idx] = trans(out_arr[:,idx])
    LOG.info('Done imputing!')

    # If given vector key, return vector as well as data
    if vector_key is not None:
        return out_arr, fits_data[[vector_key]]
    return out_arr


def ab_mag_to_flux(AB_mag: np.ndarray) -> np.ndarray:
    """Convert AB magnitude to flux in units of mJy"""
    return 10**((AB_mag - 8.9) / -2.5) * 1000


def flux_to_ab_mag(flux: np.ndarray) -> np.ndarray:
    """Convert flux in units of mJy to AB magnitude"""
    return -2.5 * np.log10(flux / 1000) + 8.9


def filter_matrix(matrix, matrix_id, common_ids):
    """Function to filter matrix based on common IDs"""
    sorted_indices = np.argsort(matrix_id)
    ordered_matrix = matrix[sorted_indices]
    ordered_ids = matrix_id[sorted_indices]
    mask = np.isin(ordered_ids, common_ids)
    return ordered_matrix[mask]


def pickle_imputed_data(survey):
    """Main function used to import, preprocess, and pickle the desired data"""



    ######################## IMPORT THE DATA ########################

    LOG.info('Importing data!')

    in_keys = ALL_SURVEY_KEYS[survey]
    in_err_keys = [f'ferr{in_k[1:]}' for in_k in in_keys[:8]] + [f'magerr{in_k[3:]}' for in_k in in_keys[8:]]
    in_transforms = [None for _ in range(8)] + [ab_mag_to_flux for _ in range(13)]

    out_err_keys = ['Mstar_best_err', 'SFR_best_err', 'zphot_lowlim', 'zphot_upplim']
    out_keys = ['Mstar_best', 'SFR_best', 'redshift']
    out_transforms = [lambda x: np.log10(x), lambda x: np.log10(x), None]
    # out_transforms = [None, None]

    photo, photo_ids = get_photometry_np_from_fits(os.path.join(PATH_TO_DATA, f'photometry/{survey}_photcat.v1.fits'), in_keys, transforms=in_transforms, vector_key='Gal_ID')
    untrans_photo_err, photo_err_ids = get_photometry_np_from_fits(os.path.join(PATH_TO_DATA, f'photometry/{survey}_photcat.v1.fits'), in_err_keys, transforms=[None for _ in range(len(in_keys))], vector_key='Gal_ID')
    cat, cat_ids = get_catalog_np_from_fits(os.path.join(PATH_TO_DATA, f'sed_catalog/{survey}.v1.fits'), out_keys, transforms=out_transforms, vector_key='Gal_ID')
    untrans_cat_err, cat_err_ids = get_catalog_np_from_fits(os.path.join(PATH_TO_DATA, f'sed_catalog/{survey}.v1.fits'), out_err_keys, transforms=[None for _ in range(len(out_err_keys))], vector_key='Gal_ID')


    # Calculate the zerr using the lower and upper limits (1/2 * difference between upper and lower limits according to paper)
    z_err = 0.5 * (untrans_cat_err[:, 3] - untrans_cat_err[:, 2])
    z_err[z_err == 0] = 0.01  # spectroscopic error will be zero. Set to arbitrarily low values
    # Replace the limit columns with the error
    untrans_cat_err = np.delete(untrans_cat_err, 3, 1)
    untrans_cat_err = np.delete(untrans_cat_err, 2, 1)
    untrans_cat_err = np.hstack((untrans_cat_err, np.atleast_2d(z_err).reshape(-1, 1)))





    ######################## FILTER THE DATA AND SORT TO THE SAME ORDER ########################
    # Make ztype column filterable
    ztype_w_ids = np.array(fits.open(os.path.join(PATH_TO_DATA, f'sed_catalog/{survey}.v1.fits'))[1].data)[['ztype']]
    new_dtype = np.dtype(ztype_w_ids.dtype.descr + [('Gal_ID', int)])
    new_data = np.zeros(len(ztype_w_ids), dtype=new_dtype)
    for name in ztype_w_ids.dtype.names:
        new_data[name] = ztype_w_ids[name]
    new_data['Gal_ID'] = np.arange(len(ztype_w_ids))
    ztype_w_ids = new_data

    # Filter matrices based on common IDs
    common_ids = np.intersect1d(photo_ids, np.intersect1d(photo_err_ids, np.intersect1d(cat_ids, cat_err_ids)))
    photo = filter_matrix(photo, photo_ids, common_ids)
    untrans_photo_err = filter_matrix(untrans_photo_err, photo_err_ids, common_ids)
    cat = filter_matrix(cat, cat_ids, common_ids)
    untrans_cat_err = filter_matrix(untrans_cat_err, cat_err_ids, common_ids)
    ztype = filter_matrix(ztype_w_ids['ztype'], ztype_w_ids[['Gal_ID']], common_ids)
    tractor_ids = filter_matrix(ztype_w_ids['Gal_ID'], ztype_w_ids[['Gal_ID']], common_ids)





    ######################## TRANSFORM THE ERRORS ########################

    LOG.info('Transforming data!')

    # Photometery
    ab_magerr_to_ferr = lambda sigma_m, f: np.abs(f * np.log(10) * (sigma_m / 2.5))  # transformation on the error of a magnitude turned into flux
    photo_err_trans = [None for _ in range(8)] + [ab_magerr_to_ferr for _ in range(len(in_keys[8:]))]
    photo_err = np.copy(untrans_photo_err)
    for ind, trans in enumerate(photo_err_trans):
        if trans is not None:
            photo_err[:, ind] = trans(photo_err[:, ind], photo[:, ind])

    # Catalog attributes
    err_log_trans = lambda Xerr, X: np.abs(Xerr / (10**X * np.log(10)))  # transformation on the error of a random variable transformed by f=ln(X)
    cat_err_trans = [err_log_trans, err_log_trans, None]  # M_star, SFR, redshift
    cat_err = np.copy(untrans_cat_err)
    for ind, trans in enumerate(cat_err_trans):
        if trans is not None:
            cat_err[:, ind] = trans(cat_err[:, ind], cat[:, ind])

    LOG.info('\nShapes: \nX: \t%s \nXerr: \t%s \ny: \t%s \nyerr: \t %s', photo.shape, photo_err.shape, cat.shape, cat_err.shape)




    ######################## PLOT SOME STUFF ########################
    
    LOG.info('Plotting!')
    # Sort data for convenience
    survey_filter_wavelengths = {key: FILTER_WAVELENGTHS[key] for key in in_keys if key in FILTER_WAVELENGTHS}
    sorted_filters = sorted(survey_filter_wavelengths.keys(), key=lambda k: survey_filter_wavelengths[k])
    sorted_wavelengths = sorted(survey_filter_wavelengths.values())
    ordered_indices = [sorted_filters.index(label) for label in in_keys if label in sorted_filters]
    in_err_keys = [in_err_keys[idx] for idx in ordered_indices]
    photo = photo[:, ordered_indices]
    photo_err = photo_err[:, ordered_indices]
    plot_dirpath = f'/Users/adamboesky/Research/ay98/Weird_Galaxies/preprocessing/preprocessing_plots/{survey}'
    if not os.path.isdir(plot_dirpath):
        os.mkdir(plot_dirpath)

    # Photometry plots
    # Random SED plots
    photo_err[photo_err == -1] = 0
    plt.figure()
    for sample in (100, 200, 300):
        plt.errorbar(sorted_wavelengths, photo[sample], yerr=photo_err[sample], label=f'Sample {sample}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux [mJy]')
    plt.legend()
    plt.savefig(f'{plot_dirpath}/sed_plots.png')
    plt.close()

    # Filter histograms
    for filter, vals in zip(sorted_filters, photo.T):
        plt.figure()
        plt.hist(np.log10(vals), bins=50)
        plt.xlabel(f'log(Flux) {filter} [mJy]')
        plt.ylabel('Count')
        plt.savefig(f'{plot_dirpath}/hist_{filter}.png')
        plt.close()

    # Catalog plots
    for out_k, vals in zip(out_keys, cat.T):
        plt.figure()
        plt.hist(vals, bins=50)
        plt.ylabel('Count')
        plt.xlabel(out_k)
        plt.savefig(f'{plot_dirpath}/hist_{out_k}.png')
        plt.close()




    ######################## PICKLE DATA ########################

    LOG.info('Pickling!')
    data_dict = {
        'photometry': {
            'sorted_filters': sorted_filters,
            'err_keys': in_err_keys,
            'sorted_wavelengths': sorted_wavelengths,
            'data': photo,
            'data_err': photo_err,
            'gal_id': tractor_ids
        },
        'catalog': {
            'keys': out_keys,
            'err_keys': out_err_keys,
            'data': cat,
            'data_err': cat_err,
            'ztype': ztype,
            'gal_id': tractor_ids
        }
    }
    with open(f'/Users/adamboesky/Research/ay98/clean_data/{survey}_preprocessed2.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
        LOG.info('Data preprocessed and pickled %s', f.name)
    # with open(f'ay98/preprocessing/clean_data/testing_stuff.pkl', 'wb') as f:
    #     pickle.dump(data_dict, f)
    #     LOG.info('Data preprocessed and pickled %s', f.name)



if __name__ == '__main__':
    for SURVEY in ('es1', 'wcdfs', 'xmmlss'):
        pickle_imputed_data(survey=SURVEY)
    # pickle_imputed_data(survey='wcdfs')
