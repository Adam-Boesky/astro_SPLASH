import os
import sys
import argparse
import numpy as np
import pickle
from astropy.coordinates import Angle
import warnings
from astropy.wcs import FITSFixedWarning
from SPLASH.pipeline2 import Splash_Pipeline


# Filter out the specific FITSFixedWarning
warnings.filterwarnings('ignore', category=FITSFixedWarning)
warnings.filterwarnings('ignore', module='photutils.background.background_2d')
warnings.filterwarnings('ignore', module='astropy.stats.sigma_clipping')


sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')
sys.path.append('/Users/adamboesky/Research/ay98/Weird_Galaxies')

PS1FILENAME = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
FITSCUT = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
CLUSTER = False

if CLUSTER:
    PATH_TO_STORAGE = '/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/'
else: 
    PATH_TO_STORAGE = '/Users/adamboesky/Research/ay98/clean_data'


def get_current_data():
    """Get the current place in our data."""
    # Grab the sne data
    with open(os.path.join(PATH_TO_STORAGE, 'sn_coords_clean.csv'), 'rb') as f:
        sne = pickle.load(f)

    return sne


def match_host_sne_chunk(ind_begin: int, ind_end: int):
    """Match the sne to host galaxies in the panstarrs databse and save files."""

    # Grab the current data
    sne = get_current_data()

    # Drop the unclassified SNe and grab the ra, dec
    classified_mask = np.array([isinstance(t, str) and t not in ('Candidate', 'candidate') for t in sne['claimedtype']])
    sne = sne[classified_mask]
    ra = Angle([ra.split(',')[0] for ra in sne['ra']], unit='hourangle').deg
    dec = Angle([f'{dec.split(",")[0]} degrees' for dec in sne['dec']]).deg

    # Make output dir if it doesn't exist
    if not os.path.exists(os.path.join(PATH_TO_STORAGE, 'panstarrs_hosts_prost')):
        os.mkdir(os.path.join(PATH_TO_STORAGE, 'panstarrs_hosts_prost'))

    # Cut out the chunk
    sne = sne[ind_begin:ind_end]
    ra = ra[ind_begin:ind_end]
    dec = dec[ind_begin:ind_end]
    
    # Associate and save
    pipeline = Splash_Pipeline()
    transient_catalog = pipeline.get_transient_catalog(
        ra,
        dec,
        names=sne['event'],
        parallel=True,
        cat_cols=True,
    )
    transient_catalog.to_csv(
        os.path.join(PATH_TO_STORAGE,
                        'panstarrs_hosts_prost',
                        f'panstarrs_hosts_prost{ind_begin}_{ind_end}.ecsv',
        )
    )


    # # Associate sne with their hosts
    # starting_ind = 0
    # chunk_len = 100
    # chunk_counter = 0
    # while starting_ind < len(sne):

    #     pipeline = Splash_Pipeline()

    #     transient_catalog = pipeline.get_transient_catalog(
    #         ra[starting_ind:starting_ind + chunk_len],
    #         dec[starting_ind:starting_ind + chunk_len],
    #         names=sne['event'][starting_ind:starting_ind + chunk_len],
    #         parallel=True,
    #         cat_cols=True,
    #     )

    #     # Save the results
    #     transient_catalog.to_csv(
    #         os.path.join(PATH_TO_STORAGE,
    #                      'panstarrs_hosts_prost',
    #                      f'panstarrs_hosts_prost{chunk_counter}.ecsv',
    #         )
    #     )

    #     # Update the info
    #     starting_ind += chunk_len
    #     chunk_counter += 1


def process_chunk():

    # Configure parser
    parser = argparse.ArgumentParser(description='Associate SNe with Prost in chunks.')
    parser.add_argument(
        '-b',
        '--beginning_ind',
        type=int,
        help='Chunk beginning index.'
    )
    parser.add_argument(
        '-e',
        '--ending_ind',
        type=int,
        help='Chunk ending index.'
    )
    args = parser.parse_args()

    # Associate the chunk
    match_host_sne_chunk(args.beginning_ind, args.ending_ind)


if __name__=='__main__':
    process_chunk()
