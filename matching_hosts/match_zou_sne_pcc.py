import os
import sys
import numpy as np
import pandas as pd
from astropy.table import Table
from typing import Tuple
import requests
import pickle
import shutil
from io import StringIO
import wget
import urllib
from pathlib import Path
from astropy.io import fits
from astropy import table
from astropy.wcs import WCS
from glob import glob
from astropy.visualization import simple_norm
from astropy.stats import SigmaClip
from astropy.coordinates import Angle, SkyCoord
from astropy.io import ascii
from astropy.cosmology import Planck18 as cosmo  # Using the Planck 2018 cosmology
from astropy import units as u
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from photutils.background import Background2D, MADStdBackgroundRMS
from photutils.utils import circular_footprint, calc_total_error
from match_panstarrs_sne import make_query
from match_panstarrs_sne_pcc import get_images, background_subtracted, get_host_coords
import warnings
from astropy.wcs import FITSFixedWarning

# Filter out the specific FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', module='photutils.background.background_2d')
warnings.filterwarnings('ignore', module='astropy.stats.sigma_clipping')


sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')
sys.path.append('/Users/adamboesky/Research/ay98/Weird_Galaxies')
from logger import get_clean_logger
LOG = get_clean_logger(logger_name = Path(__file__).name)  # Get my beautiful logger

# os.mkdir = 'ps1_dir'
PS1FILENAME = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
FITSCUT = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
CLUSTER = True

if CLUSTER:
    PATH_TO_STORAGE = '/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/'
else: 
    PATH_TO_STORAGE = '/Users/adamboesky/Research/ay98/clean_data'


def get_coords_for_ids(gal_ids: np.bytes_, lookup_table_path: str = '/Users/adamboesky/Research/ay98/clean_data/coord_lookup_table.pkl') -> Tuple[float, float]:
    """Function used to get the coordinates of a given tractor ID."""
    with open(lookup_table_path, 'rb') as f:
        coord_lookup_table = pickle.load(f)

    sorted_inds = np.argsort(coord_lookup_table['gal_id'])
    lup_tab_sorted = coord_lookup_table[sorted_inds]
    
    # Find the sorted indices of 'a' in 'b'
    indices_in_b_sorted = np.searchsorted(lup_tab_sorted['gal_id'], gal_ids, side='left')
    
    # Make sure that all elements of 'a' are actually present in 'b'
    indices_in_b_sorted = indices_in_b_sorted[indices_in_b_sorted < len(coord_lookup_table['gal_id'])]
    valid_mask = lup_tab_sorted['gal_id'][indices_in_b_sorted] == gal_ids
    indices_in_b_sorted = indices_in_b_sorted[valid_mask]

    # Take the sorted indices and map back to original indices in 'b'
    all_indices = lup_tab_sorted[indices_in_b_sorted]
    
    return all_indices


def filter_sne_for_Zou(sne: pd.DataFrame) -> pd.DataFrame:
    """This function cuts down the SN sample to things within the deep fields
    that Zou et al. considers, and filters for only classified sne."""

    # Get the sne coords as astropy objects
    sne_ra = []
    sne_dec = []
    sne_coords = sne[['ra', 'dec']]                     # sne coords array
    for ra, dec in sne_coords.to_numpy():
        dec_ang = Angle(f'{dec.split(",")[0]} degrees')
        ra_ang = Angle(ra.split(',')[0], unit='hourangle')

        sne_ra.append(ra_ang.wrap_at("24h").deg)
        sne_dec.append(dec_ang.deg)
    sn_sky_coords = SkyCoord(sne_ra * u.deg, sne_dec * u.deg, frame='icrs')

    # Filter the SNe to only be in the deep fields that Zou considers
    wcdfs_center = SkyCoord(Angle('03:32:09', unit='hourangle').deg * u.deg, Angle('-28:08:32 degrees').deg * u.deg, frame='icrs')
    es1_center = SkyCoord(Angle('00:37:47', unit='hourangle').deg * u.deg, Angle('-44:00:07 degrees').deg * u.deg, frame='icrs')
    xmmlss_center = SkyCoord(Angle('02:22:10', unit='hourangle').deg * u.deg, Angle('-04:45:00 degrees').deg * u.deg, frame='icrs')

    # Get the seperation of each SN from the center of the field
    field_seps = np.zeros((len(sn_sky_coords), 3))
    for i, field_center in enumerate((wcdfs_center, es1_center, xmmlss_center)):
        field_seps[:, i] = field_center.separation(sn_sky_coords).degree
    min_sep = np.min(field_seps, axis=1)

    # Only select sne that are within 4 degrees of the field, classified, and are not nans
    max_field_raidus = 4  # 1.3 is the approximate radius of the depe field, but we'll use 4 to be conservative
    in_field_mask = min_sep < max_field_raidus
    classified_mask = sne[in_field_mask]['claimedtype'] != 'Candidate'
    nan_mask = sne[in_field_mask][classified_mask]['claimedtype'].to_numpy().astype(str) == 'nan'
    LOG.info(f'There are {np.sum(in_field_mask)} supernovae within the deep fields.')
    LOG.info(f"Of those, approximate {len(sne[in_field_mask][classified_mask]['claimedtype'].dropna())} are classified.")
    LOG.info(f"Of the classified, {sne['claimedtype'][in_field_mask][classified_mask].str.contains('ia', case=False, na=False).sum() / len(sne[in_field_mask][classified_mask]['claimedtype'].dropna())} are type Ia")
    classified_field_sne = sne[in_field_mask][classified_mask][~nan_mask]

    return classified_field_sne


def get_current_data():
    """Get the current place in our data."""
    # Grab the sne data
    with open(os.path.join(PATH_TO_STORAGE, 'sn_coords_clean.csv'), 'rb') as f:
        sne = pickle.load(f)

    # Apply some Zou-specific filters
    sne = filter_sne_for_Zou(sne)

    # Create empty columns
    cols = ['raMean', 'decMean'] + [f'{filt}MeanApMag' for filt in ['g', 'r', 'i', 'z', 'y']] + [f'{filt}MeanApMagErr' for filt in ['g', 'r', 'i', 'z', 'y']]  # desired columns
    sne[cols] = np.NaN
    n = len(sne)

    # If the associate table already exists, pick up from the end of the already associated hosts
    LOG.info('Getting the index of the last host in saved table')
    if os.path.exists(os.path.join(PATH_TO_STORAGE, 'zou_hosts_pcc.ecsv')):

        # Get the index of the last already associated SN
        all_res = pd.read_csv(os.path.join(PATH_TO_STORAGE, 'zou_hosts_pcc.csv'))
        last_ra, last_dec = all_res[-1]['SN_ra'], all_res[-1]['SN_dec']
        col_types = {col: all_res[col].dtype for col in all_res.columns}
        for i, sn_ra, sn_dec in zip(range(n), sne['ra'], sne['dec']):

            # Put angles in a dictionary
            dec_deg = Angle(f'{sn_dec.split(",")[0]} degrees').deg
            ra_deg = Angle(sn_ra.split(',')[0], unit='hourangle').deg

            tol = 1E-10  # tolerance for SNe being the same
            if abs(ra_deg - last_ra) < tol and abs(dec_deg - last_dec) < tol:
                LOG.info(f'Search going to pick up from row {i} / {n}')
                last_searched_ind = i
                break

    else:
        LOG.info('No table exists, starting cone search from beginning')
        last_searched_ind = 0
        col_types = None
        all_res = None

    return sne, last_searched_ind, col_types, all_res


def get_mean_of_strs(s: str) -> float:
    """Get the mean of a string of floats."""
    if isinstance(s, float):
        return s
    else:
        arr = np.array(s.split(',')).astype(float)
        return np.nanmean(arr)


def match_host_sne():
    """Match the sne to host galaxies in the panstarrs databse and save files."""

    # Grab the current data
    sne, last_searched_ind, col_types, all_res = get_current_data()
    n = len(sne)

    # Grab Zou data
    with open(f'/Users/adamboesky/Research/ay98/clean_data/all_cat.pkl', 'rb') as f:
        final_cat = pickle.load(f)
    with open(f'/Users/adamboesky/Research/ay98/clean_data/all_photometry.pkl', 'rb') as f:
        final_photo = pickle.load(f)
    gal_coords_arr = get_coords_for_ids(final_photo['gal_id'], lookup_table_path='/Users/adamboesky/Research/ay98/clean_data/coord_lookup_table.pkl')
    gal_coords = SkyCoord(gal_coords_arr['RA'] * u.deg, gal_coords_arr['DEC'] * u.deg, frame='icrs')

    # Headers for the all_res dataframe
    photo_headers = final_photo['sorted_filters']
    photo_err_headers = [ph + '_err' for ph in photo_headers]
    prop_headers = final_cat['keys']
    prop_err_headers = [ph + '_err' for ph in prop_headers]
    all_headers = ['gal_id', 'host_ra', 'host_dec', 'SN_ra', 'SN_dec'] + photo_headers + photo_err_headers + prop_headers + prop_err_headers

    # Get the host candidate coords
    for i, sn_ra, sn_dec, sn_z, sn_class in zip(range(n)[last_searched_ind + 1:], sne['ra'][last_searched_ind + 1:], sne['dec'][last_searched_ind + 1:], sne['redshift'][last_searched_ind + 1:], sne['claimedtype'][last_searched_ind + 1:]):
        if isinstance(sn_class, str) and sn_class != 'Candidate':  # only do the search if the SN is classified!!!

            # Convert SN coords to degrees
            sn_ra_ang = Angle(sn_ra.split(',')[0], unit='hourangle').deg
            sn_dec_ang = Angle(f'{sn_dec.split(",")[0]} degrees').deg

            # Get the host coordinates
            sn_z = get_mean_of_strs(sn_z)
            host_ras, host_decs, host_P_ccs = get_host_coords(sn_ra_ang, sn_dec_ang, sn_z)
            if len(host_ras) != 0:  # if we find a host
                host_ra = host_ras[0]       # first one is the host!
                host_dec = host_decs[0]     # first one is the host!
                host_coord = SkyCoord(host_ra * u.deg, host_dec * u.deg, frame='icrs')
                LOG.info(f'SN @ {sn_ra_ang, sn_dec_ang}, z={sn_z}: \t \t {len(host_ras)} candidates found with P_cc < 0.1')

                # Only keep the host if the host is in the Zou dataset
                seps_to_Zou = host_coord.separation(gal_coords).arcsec  # angular separations from Zou galaxies
                smallest_sep = np.min(seps_to_Zou)
                smallest_sep_ind = np.argmin(seps_to_Zou)
                if smallest_sep < 1:  # if the host is within 1 arcsec of a Zou galaxy, it's a Zou galaxy
                    LOG.info(f'Found a host in the Zou catalog.')

                    # Construct a new results to append to the df
                    new_res = {'gal_id': [final_photo['gal_id'][smallest_sep_ind]],
                               'host_ra': [host_ra],
                               'host_dec': [host_dec],
                               'SN_ra': [sn_ra_ang],
                               'SN_dec': [sn_dec_ang]
                               }
                    for i, wl in enumerate(photo_headers):  # photometry stuff
                        new_res[wl] = [final_photo['data'][smallest_sep_ind][i]]
                        new_res[wl+'_err'] = [final_photo['data_err'][smallest_sep_ind][i]]
                    for i, propname in enumerate(prop_headers):  # property stuff
                        new_res[propname] = [final_cat['data'][smallest_sep_ind][i]]
                        new_res[propname+'_err'] = [final_cat['data_err'][smallest_sep_ind][i]]

                    # Append to df
                    new_res = pd.DataFrame(new_res)
                    all_res = pd.concat((all_res, new_res), ignore_index = True)
 
                    # Log
                    if all_res is not None and i % 1 == 0:
                        LOG.info(f'Index = {i} / {n}, logging...')
                        all_res.to_csv(os.path.join(PATH_TO_STORAGE, 'zou_hosts_pcc.csv'))
                        # ascii.write(all_res, os.path.join(PATH_TO_STORAGE, 'zou_hosts_pcc.csv'), overwrite=True, format='ecsv')
                else:
                    LOG.info(f'Host not in the Zou catalog.')

    # Log when done
    LOG.info(f'Done with search ({i} / {n})!!! Logging...')
    all_res.to_csv(os.path.join(PATH_TO_STORAGE, 'zou_hosts_pcc.csv'))
    # ascii.write(all_res, os.path.join(PATH_TO_STORAGE, 'zou_hosts_pcc.csv'), overwrite=True, format='ecsv')


if __name__=='__main__':
    match_host_sne()
