import os
import sys
import pathlib
import sys
import pickle
import numpy as np
import pandas as pd

from astropy import table
from astropy.io import ascii
from astropy.coordinates import Angle
from mastcasjobs import MastCasJobs

from pathlib import Path
sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')
sys.path.append('/Users/adamboesky/Research/ay98/Weird_Galaxies')
from logger import get_clean_logger
LOG = get_clean_logger(logger_name = Path(__file__).name)  # Get my beautiful logger


def make_nan(catalog, replace = np.nan):
    '''
    Go through an astropy table and covert any empty values
    into a single aunified value specified by 'replace'
    '''
    for i in range(len(catalog)):
        for j in catalog[i].colnames:
            if str(catalog[i][j]) in [False, 'False', '', '-999', '-999.0', '--', 'n', '-9999.0', 'nan', b'']:
                catalog[i][j] = replace

    return catalog


def make_query(ra_deg: float, dec_deg: float, search_radius_arcmin: float, sn_ra: float = None, sn_dec: float = None):
    '''
    Adapted from FLEET.
    '''

    # Get the PS1 MAST username and password from /Users/username/3PI_key.txt
    key_location = os.path.join(pathlib.Path.home(), 'vault/mast_login.txt')
    wsid, password = np.genfromtxt(key_location, dtype = 'str')

    # 3PI query
    # Kron Magnitude, ps_score (we define galaxies as ps_score < 0.9)
    the_query = """
    SELECT o.objID,o.raStack,o.decStack,m.primaryDetection,
    m.gKronMag,m.rKronMag,m.iKronMag,m.zKronMag,m.yKronMag,m.gKronMagErr,m.rKronMagErr,
    m.iKronMagErr,m.zKronMagErr,m.yKronMagErr, psc.ps_score
    FROM fGetNearbyObjEq(%s, %s, %s) nb
    INNER JOIN ObjectThin o on o.objid=nb.objid
    INNER JOIN StackObjectThin m on o.objid=m.objid
    LEFT JOIN HLSP_PS1_PSC.pointsource_scores psc on o.objid=psc.objid
    FULL JOIN StackModelFitSer s on o.objid=s.objid
    INNER JOIN StackObjectAttributes b on o.objid=b.objid WHERE m.primaryDetection = 1 AND psc.ps_score < 0.9
    """
    la_query = the_query%(ra_deg, dec_deg, search_radius_arcmin)

    # Format Query
    jobs    = MastCasJobs(userid=wsid, password=password, context="PanSTARRS_DR2")
    results = jobs.quick(la_query, task_name="python cone search")

    # For New format
    if type(results) != str:
        catalog_3pi = table.Table(results, dtype=[str] * len(results.columns))
        if len(catalog_3pi) == 0:
            print('Found %s objects'%len(catalog_3pi))
            return None
    else:
        raise TypeError(f'Query returned type {type(results)}, not astropy.Table.')

    # Clean up 3pi's empty cells
    catalog_3pi = make_nan(catalog_3pi)

    # Append '3pi' to column name
    for i in range(len(catalog_3pi.colnames)):
        catalog_3pi[catalog_3pi.colnames[i]].name = catalog_3pi.colnames[i] + '_3pi'

    # Remove duplicates
    catalog_3pi = table.unique(catalog_3pi, keys = 'objID_3pi', keep = 'first')
    if sn_ra is None and sn_dec is None:
        catalog_3pi.add_column(ra_deg, name='SN_ra')
        catalog_3pi.add_column(dec_deg, name='SN_dec')
    else:
        catalog_3pi.add_column(sn_ra, name='SN_ra')
        catalog_3pi.add_column(sn_dec, name='SN_dec')

    print('Found %s objects \n'%len(catalog_3pi))
    return catalog_3pi


def match_sne():

    # Grab the sne data
    print('Getting SNe')
    with open('/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/sn_coords_clean.csv', 'rb') as f:
        sne = pickle.load(f)

    # Create empty columns
    cols = ['raMean', 'decMean'] + [f'{filt}MeanApMag' for filt in ['g', 'r', 'i', 'z', 'y']] + [f'{filt}MeanApMagErr' for filt in ['g', 'r', 'i', 'z', 'y']]  # desired columns
    sne[cols] = np.NaN
    n = len(sne)

    # If the associate table already exists, pick up from the end of the already associated hosts
    print('Getting the index of the last host')
    if os.path.exists('/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/panstarrs_hosts.ecsv'):

        # Get the index of the last already associated SN
        all_res = ascii.read("/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/panstarrs_hosts.ecsv", delimiter=' ', format='ecsv')
        last_ra, last_dec = all_res[-1]['SN_ra'], all_res[-1]['SN_dec']
        col_types = {col: all_res[col].dtype for col in all_res.columns}
        for i, sn_ra, sn_dec in zip(range(n), sne['ra'], sne['dec']):
            # Put angles in a dictionary
            dec_ang = Angle(f'{sn_dec.split(",")[0]} degrees')
            ra_ang = Angle(sn_ra.split(',')[0], unit='hourangle')

            if ra_ang.deg == last_ra and  dec_ang.deg == last_dec:
                print(f'Search going to pick up from row {i} / {n}')
                last_searched_ind = i
                break
    else:
        print('No table exists, starting cone search from beginning')
        last_searched_ind = 0
        col_types = None
        all_res = None


    # Run cone search of panstarrs database for each sn coord
    sr = 6/3600  # search radius [deg]
    print(f'Beginning cone search for {n} hosts')
    for i, sn_ra, sn_dec in zip(range(n)[last_searched_ind:], sne['ra'][last_searched_ind:], sne['dec'][last_searched_ind:]):

        # Print status
        if i % 1000 == 0:
            print(f'{i} / {n}')

        # Put angles in a dictionary
        dec_ang = Angle(f'{sn_dec.split(",")[0]} degrees')
        ra_ang = Angle(sn_ra.split(',')[0], unit='hourangle')

        # Cone search
        try:
            res = make_query(ra_ang.deg, dec_ang.deg, search_radius=sr)
            if all_res is None and res is not None:
                all_res = res
            elif res is not None:
                # Convert the types of the table columns
                if col_types:
                    for col, dtype in col_types.items():
                        res[col] = res[col].astype(dtype)
                all_res = table.vstack([all_res, res])
            print(res)

            if all_res is not None:
                ascii.write(all_res, '/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/panstarrs_hosts.ecsv', overwrite=True, format='ecsv')
        except:
            print('Exception encountered. Continuing.')
            continue


if __name__ == '__main__':
    print('Running match.')
    LOG.info('Running match.')
    match_sne()
