import os
import sys
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


def match_host_sne():
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

    # Associate sne with their hosts
    starting_ind = 0
    chunk_len = 100
    chunk_counter = 0
    while starting_ind < len(sne):
        pipeline = Splash_Pipeline()
        transient_catalog = pipeline.get_transient_catalog(
            ra[starting_ind:starting_ind + chunk_len],
            dec[starting_ind:starting_ind + chunk_len],
            names=sne['event'][starting_ind:starting_ind + chunk_len],
            parallel=False,
            cat_cols=True,
        )

        # Save the results
        transient_catalog.to_csv(
            os.path.join(PATH_TO_STORAGE,
                         'panstarrs_hosts_prost',
                         f'panstarrs_hosts_prost{chunk_counter}.ecsv',
            )
        )

        # Update the info
        starting_ind += chunk_len
        chunk_counter += 1


if __name__=='__main__':
    match_host_sne()
