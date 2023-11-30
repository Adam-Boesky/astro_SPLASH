import sys
from io import StringIO

from astro_ghost.PS1QueryFunctions import ps1search

sys.path.append('/Users/adamboesky/Research/ay98/Weird_Galaxies')

import pandas as pd

from logger import get_clean_logger

LOG = get_clean_logger('panstarrs_log')


def get_panstarrs_data():

    # Read in data
    cols = {'columns': ['raMean', 'decMean'] + [f'{filt}MeanApMag' for filt in ['g', 'r', 'i', 'z', 'y']] + [f'{filt}MeanApMagErr' for filt in ['g', 'r', 'i', 'z', 'y']]}
    panstarrs = ps1search(release='dr2', kw=cols, columns=cols['columns'])
    pstar_df = pd.read_csv(StringIO(panstarrs), dtype=float)
    print(len(pstar_df.sheet_names))

    # Get only rows without -999 (missing band data)
    pstar_df = pstar_df[(pstar_df != -999.0).all(axis=1)]
    duplicates = pstar_df.duplicated(subset=['raMean', 'decMean'], keep='first')
    LOG.info('There are %s full grizy rows in our dataset', len(pstar_df))
    LOG.info('There are %s duplicate rows in the table', len(pstar_df[duplicates]))
    LOG.info('The columns:\n%s', pstar_df.columns)

    # Save data
    LOG.info('Saving df...')
    pstar_df.to_csv('/Users/adamboesky/Research/ay98/clean_data/panstarrs_photometry.csv')


if __name__ == '__main__':
    get_panstarrs_data()
