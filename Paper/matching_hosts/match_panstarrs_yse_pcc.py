import os
import sys
import numpy as np
import pandas as pd
from astropy.table import Table
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
from astropy.coordinates import Angle
from astropy.io import ascii
from astropy.cosmology import Planck18 as cosmo  # Using the Planck 2018 cosmology
from astropy import units as u
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from photutils.background import Background2D, MADStdBackgroundRMS
from photutils.utils import circular_footprint, calc_total_error
from match_panstarrs_sne import make_query
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
OUTPUT_FILENAME = 'yse_hosts.ecsv'
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
    yse_sne = pd.read_csv(os.path.join(PATH_TO_STORAGE, 'parsnip_results_for_ysedr1_table_A1_full_for_online.csv'))
    yse_sne = yse_sne[~yse_sne['Spec. Class'].isin(['SN', 'Other', 'TDE', 'LBV', 'LRN', np.nan])]  # Drop non-SNe

    # Create empty columns
    cols = ['raMean', 'decMean'] + [f'{filt}MeanApMag' for filt in ['g', 'r', 'i', 'z', 'y']] + [f'{filt}MeanApMagErr' for filt in ['g', 'r', 'i', 'z', 'y']]  # desired columns
    yse_sne[cols] = np.NaN
    n = len(yse_sne)

    # If the associate table already exists, pick up from the end of the already associated hosts
    LOG.info('Getting the index of the last host in saved table')
    if os.path.exists(os.path.join(PATH_TO_STORAGE, OUTPUT_FILENAME)):

        # Get the index of the last already associated SN
        all_res = ascii.read(os.path.join(PATH_TO_STORAGE, OUTPUT_FILENAME), delimiter=' ', format='ecsv')
        last_ra, last_dec = all_res[-1]['SN_ra'], all_res[-1]['SN_dec']
        col_types = {col: all_res[col].dtype for col in all_res.columns}
        for i, sn_ra, sn_dec in zip(range(n), yse_sne['ra'], yse_sne['dec']):

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

    return yse_sne, last_searched_ind, col_types, all_res


# modified from https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service
def get_images(tra, tdec, size_arcsec=None, filters="grizy", format="fits", imagetypes="stack"):
    """Query ps1filenames.py service for multiple positions to get a list of images
    This adds a url column to the table to retrieve the cutout.

    tra, tdec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    format = data format (options are "fits", "jpg", or "png")
    imagetypes = list of any of the acceptable image types.  Default is stack;
        other common choices include warp (single-epoch images), stack.wt (weight image),
        stack.mask, stack.exp (exposure time), stack.num (number of exposures),
        warp.wt, and warp.mask.  This parameter can be a list of strings or a
        comma-separated string.

    Returns an astropy table with the results
    """

    # If there was no redshift, we default to a 60 arcsecond radius search
    if size_arcsec is None:
        size_arcsec = 120
    size = np.ceil(size_arcsec * (1 / 0.25)).astype(int)   # size in pixels
    LOG.info(f'Downloading image of size {size}.')

    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    # if imagetypes is a list, convert to a comma-separated string
    if not isinstance(imagetypes,str):
        imagetypes = ",".join(imagetypes)
    # put the positions in an in-memory file object
    cbuf = StringIO()
    cbuf.write('\n'.join(["{} {}".format(ra, dec) for (ra, dec) in zip([tra], [tdec])]))
    cbuf.seek(0)
    # use requests.post to pass in positions as a file... 3 tries in the query
    for attempt in range(3):
        try:
            r = requests.post(PS1FILENAME, data=dict(filters=filters, type=imagetypes), files=dict(file=cbuf))
            r.raise_for_status()
            tab = Table.read(r.text, format="ascii")

            tab["url"] = ["{}?red={}&format={}&x={}&y={}&size={}&wcs=1&imagename={}".format(FITSCUT,
                                                                                            filename,
                                                                                            format,
                                                                                            ra,
                                                                                            dec,
                                                                                            size,
                                                                                            'cutout_'+shortname) 
                        for (filename,ra,dec,shortname) in zip(tab["filename"],tab["ra"],tab["dec"],tab['shortname'])]
            break
        except Exception as e:
            LOG.info(f'Exception {attempt} encountered getting image: {e}. Continuing...')
            tab = None
            continue

    return tab


def background_subtracted(data):
    """Estimate background to get background subtracted data and background error in counts."""
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(data, nsigma=3, sigma_clip=sigma_clip)
    segment_img = detect_sources(data, threshold, npixels=4)
    if segment_img is None:
        return None, None
    footprint = circular_footprint(radius=3)
    mask = segment_img.make_source_mask(footprint=footprint)
    xsize = round(data.shape[1]/100)
    ysize = round(data.shape[0]/100)
    try:
        bkg = Background2D(data, (ysize,xsize), filter_size=(3, 3),mask=mask,
                        bkgrms_estimator = MADStdBackgroundRMS(sigma_clip))
    except:
        return None, None
    sub_data = data-bkg.background
    return sub_data, bkg


def get_host_coords(sn_ra: float, sn_dec: float, sn_z: float) -> (float, float):
    """Get the host coordinates for a set of sne coords.
    Args:
        sn_ras: The right acensions of the supernova.
        sn_decs: The declinations of the supernova.
    Returns:
        1. The RAs of the host candidates.
        2. The DECs of the host candidates.
        3. The probability of chance coincidence of the candidates.
        NOTE: The returned arrays are sorted in increasing P_cc
    """

    # Get the arcseconds that we want to search for. We will use within 100kpc
    if not np.isnan(sn_z):
        distance = cosmo.angular_diameter_distance(sn_z)
        angular_size_rad = (100 * u.kpc / distance).decompose() * u.rad
        angular_size_arcsec = (angular_size_rad.to(u.arcsec)).value
    else:
        angular_size_arcsec = None

    # Get the PS1 info for those positions
    table = get_images(sn_ra, sn_dec, size_arcsec=angular_size_arcsec, filters='r')

    # Arrays to put vals in
    host_ras = []
    host_decs = []
    host_P_ccs = []

    if table is not None:
        # Download the cutout to your directory
        ps1_dirpath = os.path.join(PATH_TO_STORAGE, 'ps1_dir')
        if not os.path.exists(ps1_dirpath):
            os.mkdir(ps1_dirpath)

        try:
            wget.download(table['url'][0],out=ps1_dirpath)
        except urllib.error.HTTPError as e:
            LOG.info(f'Error code on wget: {e.code}')
            LOG.info(f'Error message on wget: {e.reason}')
            LOG.info(f'URL is: {table["url"][0]}')
            return host_ras, host_decs, host_P_ccs

        ## Load the data
        sn_image = glob(os.path.join(ps1_dirpath, '*.fits'))[0]
        sn = fits.open(sn_image)
        sn_data, sn_bkg = background_subtracted(sn[0].data)

        # If we can't get the SN out
        if sn_data is None and sn_bkg is None:
            return host_ras, host_decs, host_P_ccs

        sn_header = sn[0].header
        sn_wcs = WCS(sn_header)
        sn.close()

        sn_x, sn_y = sn_wcs.all_world2pix(sn_ra, sn_dec, 1)
        thres = 3
        npix = 20
        threshold = thres*sn_bkg.background_rms
        segm_deblend = detect_sources(sn_data, threshold, npixels=npix)


        if segm_deblend is not None:  # if we detect sources

            # background error
            err = calc_total_error(sn_data, sn_bkg.background_rms, sn_header['CELL.GAIN'] * sn_header['EXPTIME'])
            cat = SourceCatalog(sn_data, segm_deblend, error=err, kron_params=(2.5,1.4))
            tbl = cat.to_table()

            ## PS1 zeropoint for r band is 24.68 (https://iopscience.iop.org/article/10.1088/0004-637X/756/2/158/pdf Table 1)
            m_app = -2.5*np.log10(cat.kron_flux) + 24.68

            ## the equation from Edo's paper
            sigma_m = (1/(0.33*np.log(10)))*10**(0.33*(m_app-24)-2.44)

            ## r50 is an array of half light radii for all detected objects in the frame
            r50 = cat.fluxfrac_radius(0.5).value * 0.25

            ## r is an array of distance from the SN location to the centroid of each detected object
            r = np.sqrt((tbl['xcentroid'].data-sn_x)**2+(tbl['ycentroid'].data-sn_y)**2)*0.25

            ## No uncertainties, so this is the effective radius for each object
            R_e = np.sqrt(r**2+4*r50**2)

            ## Probability of chance coincidence
            P_cc = 1-np.exp(-np.pi*R_e**2*sigma_m)

            # The indices of the host candidates
            host_inds = np.where(P_cc < 0.1)[0]

            if len(host_inds) > 0:  # If there are any host candidates!!!
                for host_ind in host_inds:

                    # Get host coords
                    host = cat.get_label(host_ind+1)
                    host_x, host_y = (host.xcentroid,host.ycentroid)
                    host_ra, host_dec = sn_wcs.all_pix2world(host_x, host_y, 1)

                    # Append values
                    host_ras.append(float(host_ra))
                    host_decs.append(float(host_dec))
                    host_P_ccs.append(P_cc[host_ind])

            # Sort candidates in increasing P_cc
            combined = sorted(zip(host_P_ccs, host_ras, host_decs))
            if combined:
                host_P_ccs, host_ras, host_decs = zip(*combined)
            else:
                # Handle the empty case appropriately
                host_P_ccs, host_ras, host_decs = [], [], []
            # host_P_ccs, host_ras, host_decs = zip(*combined)

            # Clean up dir
            shutil.rmtree(ps1_dirpath)

    return host_ras, host_decs, host_P_ccs


def get_mean_of_strs(s: str) -> float:
    """Get the mean of a string of floats."""
    if isinstance(s, float):
        return s
    else:
        arr = np.array(s.split(',')).astype(float)
        return np.nanmean(arr)





def match_host_sne():
    """Match the sne to host galaxies in the panstarrs databse and save files."""

    # Necessary params
    sr_arcmin = 6/60  # arcmins radius [arcminutes]

    # Grab the current data
    sne, last_searched_ind, col_types, all_res = get_current_data()
    n = len(sne)

    # Get the host candidate coords
    for i, sn_ra, sn_dec, sn_z, sn_class in zip(range(n)[last_searched_ind + 1:], sne['RA'][last_searched_ind + 1:], sne['Dec'][last_searched_ind + 1:], sne['Redshift, $z$'][last_searched_ind + 1:], sne['Spec. Class'][last_searched_ind + 1:]):
        if isinstance(sn_class, str) and sn_class != 'Candidate':  # only do the search if the SN is classified!!!

            # Convert SN coords to degrees
            sn_ra_ang = sn_ra
            sn_dec_ang = sn_dec

            # Get the host coordinates
            host_ras, host_decs, host_P_ccs = get_host_coords(sn_ra_ang, sn_dec_ang, sn_z)
            if len(host_ras) != 0:  # if we find a host
                host_ra = host_ras[0]       # first one is the host!
                host_dec = host_decs[0]     # first one is the host!
                LOG.info(f'SN @ {sn_ra_ang, sn_dec_ang}, z={sn_z}: \t \t {len(host_ras)} candidates found with P_cc < 0.1')

                # Search through all the candidates (in increasing order of probability) and get data
                res = None

                # 3 tries in the query
                for attempt in range(3):
                    try:
                        res = make_query(host_ra, host_dec, search_radius_arcmin=sr_arcmin, sn_ra=sn_ra_ang, sn_dec=sn_dec_ang)

                        if all_res is None and res is not None:

                            # Only first (closest) row
                            res = res[0:1]
                            all_res = res

                        elif res is not None:
                            # Only first (closest) row
                            res = res[0:1]

                            # Append SN type to result
                            res['sn_class'] = sn_class
                            res['sn_redshift'] = sn_z

                            # Convert the types of the table columns
                            if col_types:
                                for col, dtype in col_types.items():
                                    res[col] = res[col].astype(dtype)

                            # Append result to final table
                            all_res = table.vstack([all_res, res])
                        LOG.info(f'Best result for host at {host_ra, host_dec}: \n{res}')

                        if all_res is not None and i % 100 == 0:
                            LOG.info(f'Index = {i} / {n}, logging...')
                            ascii.write(all_res, os.path.join(PATH_TO_STORAGE, OUTPUT_FILENAME), overwrite=True, format='ecsv')
                        break
                    except Exception as e:
                        if "500" in str(e):
                            LOG.info(f'Caught a 500 Internal Server Error on attempt {attempt}: {e}. Continuing.')
                            continue
                        else:
                            LOG.error(f'Unexpected exception on attempt {attempt}: {e}.')
                            raise

    # Log when done
    LOG.info(f'Done with search ({i} / {n})!!! Logging...')
    ascii.write(all_res, os.path.join(PATH_TO_STORAGE, OUTPUT_FILENAME), overwrite=True, format='ecsv')


if __name__=='__main__':
    match_host_sne()
