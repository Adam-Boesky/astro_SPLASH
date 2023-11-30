import os
import pathlib
import sys
from io import StringIO

from astro_ghost.PS1QueryFunctions import ps1search

sys.path.append('/Users/adamboesky/Research/ay98/Weird_Galaxies')
import pickle

import numpy as np
import pandas as pd
from astro_ghost.PS1QueryFunctions import ps1search
from astropy import table
from astropy.coordinates import Angle
from mastcasjobs import MastCasJobs


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


def make_query(ra_deg: float, dec_deg: float, search_radius: float):
    '''
    Adapted from FLEET.
    '''

    # Get the PS1 MAST username and password from /Users/username/3PI_key.txt
    key_location = os.path.join(pathlib.Path.home(), 'vault/mast_login_harvard.txt')
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
    la_query = the_query%(ra_deg, dec_deg, search_radius)

    # Format Query
    jobs    = MastCasJobs(userid=wsid, password=password, context="PanSTARRS_DR1")
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

    print('Found %s objects \n'%len(catalog_3pi))
    return catalog_3pi


def match():

    # Grab the sne data
    with open('/Users/adamboesky/Research/ay98/clean_data/sn_coords_clean.csv', 'rb') as f:
        sne = pickle.load(f)

    # Create empty columns
    cols = ['raMean', 'decMean'] + [f'{filt}MeanApMag' for filt in ['g', 'r', 'i', 'z', 'y']] + [f'{filt}MeanApMagErr' for filt in ['g', 'r', 'i', 'z', 'y']]  # desired columns
    sne[cols] = np.NaN

    # Run cone search of panstarrs database for each sn coord
    sr = 30/3600  # search radius [deg]
    dat = {
        'sr': sr, 
        'sort_by': 'distance.ASC'
    }
    n = len(sne)
    all_res = None
    print(f'Beginning cone search for {n} hosts')
    for i, host_ra, host_dec in zip(range(n), sne['ra'], sne['dec']):

        # Print status
        if i % 1000 == 0:
            print(f'{i} / {n}')

        # Put angles in a dictionary
        dec_ang = Angle(f'{host_dec.split(",")[0]} degrees')
        ra_ang = Angle(host_ra.split(',')[0], unit='hourangle')
        dat['ra'], dat['dec'] = ra_ang.deg, dec_ang.deg

        # Cone search
        # res = ps1search(
        #     release='dr2',
        #     columns=cols,
        #     sr=sr,
        #     sort_by='distance.ASC',
        #     ra=ra_ang.deg,
        #     dec=dec_ang.deg
        # )
        # if res is not None and res != '':
        #     test = res.split('\n')
        #     res_list = [float(v) for v in res.split('\n')[1].split(',')]
        #     print(res_list)
        #     sne.iloc[i, sne.columns.get_indexer(cols)] = res_list

        res = make_query(ra_ang.deg, dec_ang.deg, search_radius=sr)
        if all_res is None and res is not None:
            all_res = res
        elif res is not None:
            all_res = table.vstack([all_res, res])
        print(all_res)


if __name__=='__main__':
    match()
