import re
import os
import sys
import argparse
import numpy as np
import pickle
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord
import warnings
from typing import Optional
from astropy.wcs import FITSFixedWarning
from SPLASH.pipeline import Splash_Pipeline


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


def extract_id_regex(html_str):
    # Simple check to see if the string contains an <a> tag
    pattern = r'id="([^"]+)"'
    if '<a ' in html_str.lower():
        match = re.search(pattern, html_str)
        if match:
            return match.group(1)
    return html_str  # Return None for non-HTML or missing id


def get_current_data():
    """Get the current place in our data."""
    # Grab the sne data
    with open(os.path.join(PATH_TO_STORAGE, 'sn_coords_clean.csv'), 'rb') as f:
        sne = pickle.load(f)

    # Drop the unclassified SNe and grab the ra, dec
    classified_mask = np.array([isinstance(t, str) and t not in ('Candidate', 'candidate') for t in sne['claimedtype']])
    sne = sne[classified_mask]

    # Fix the names
    sne['event'] = [extract_id_regex(html) for html in sne['event']]

    return sne


def match_host_sne_chunk(ind_begin: Optional[int], ind_end: Optional[int]):
    """Match the sne to host galaxies in the panstarrs databse and save files."""

    # Grab the current data
    print('Grabbing data!')
    sne = get_current_data()

    bad_coord = []
    for idx, row in sne.iterrows():
        try:
            SkyCoord(row['ra'], row['dec'], unit=(u.hourangle, u.deg))
        except:
            bad_coord.append(row.event)

    sne = sne[~sne.event.isin(bad_coord)]
    sne['redshift'] = sne['redshift'].astype("str")
    sne['redshift'] = [x.split(",")[0] for x in sne['redshift']]
    sne['redshift'] = sne['redshift'].astype("float")

    # Filter out events with z<0.001
    sne = sne[sne['redshift'] > 0.001]
    print(f'Total sne length: {len(sne)}')

    # Cut out the chunk
    if ind_begin is not None and ind_end is not None:
        sne = sne[ind_begin:ind_end]

    # Associate and save
    print('Creating pipeline object!')
    pipeline = Splash_Pipeline()
    print('Getting transient catalog!')
    transient_catalog = pipeline.get_transient_catalog(
        sne['ra'],
        sne['dec'],
        redshift=sne['redshift'],
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


def process_chunk():

    # Configure parser
    parser = argparse.ArgumentParser(description='Associate SNe with Prost in chunks.')
    parser.add_argument(
        '-b',
        '--beginning_ind',
        type=int,
        default=None,
        help='Chunk beginning index.'
    )
    parser.add_argument(
        '-e',
        '--ending_ind',
        type=int,
        default=None,
        help='Chunk ending index.'
    )
    args = parser.parse_args()

    # Associate the chunk
    match_host_sne_chunk(args.beginning_ind, args.ending_ind)


if __name__=='__main__':
    process_chunk()
