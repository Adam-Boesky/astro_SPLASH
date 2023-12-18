import os
import numpy as np
from astropy.table import Table
import requests
import pickle
from io import StringIO
import wget
from astropy.io import fits
from astropy import table
from astropy.wcs import WCS
from glob import glob
from astropy.visualization import simple_norm
from astropy.stats import SigmaClip
from astropy.coordinates import Angle
from astropy.io import ascii
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
from photutils.background import Background2D, MADStdBackgroundRMS
from photutils.utils import circular_footprint, calc_total_error
from match_panstarrs_sne import make_query
 
# os.mkdir = 'ps1_dir'
PS1FILENAME = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
FITSCUT = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"


def get_current_data():
    """Get the current place in our data."""
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
    if os.path.exists('/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/panstarrs_hosts_pcc.ecsv'):

        # Get the index of the last already associated SN
        all_res = ascii.read("/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/panstarrs_hosts_pcc.ecsv", delimiter=' ', format='ecsv')
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

    return sne, last_searched_ind, col_types, all_res


# modified from https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service
def get_images(tra, tdec, size=1024, filters="grizy", format="fits", imagetypes="stack"):
    """Query ps1filenames.py service for multiple positions to get a list of images
    This adds a url column to the table to retrieve the cutout.

    tra, tdec = list of positions in degrees
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

    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    # if imagetypes is a list, convert to a comma-separated string
    if not isinstance(imagetypes,str):
        imagetypes = ",".join(imagetypes)
    # put the positions in an in-memory file object
    cbuf = StringIO()
    cbuf.write('\n'.join(["{} {}".format(ra, dec) for (ra, dec) in zip(tra,tdec)]))
    cbuf.seek(0)
    # use requests.post to pass in positions as a file
    r = requests.post(PS1FILENAME, data=dict(filters=filters, type=imagetypes),
        files=dict(file=cbuf))
    r.raise_for_status()
    tab = Table.read(r.text, format="ascii")
 
    tab["url"] = ["{}?red={}&format={}&x={}&y={}&size={}&wcs=1&imagename={}".format(FITSCUT,
                                                                                    filename,
                                                                                    format,
                                                                                    ra,dec,size,
                                                                                    'cutout_'+shortname) 
                  for (filename,ra,dec,shortname) in zip(tab["filename"],tab["ra"],tab["dec"],tab['shortname'])]
    return tab


def background_subtracted(data):
    """Estimate background to get background subtracted data and background error in counts."""
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(data, nsigma=3, sigma_clip=sigma_clip)
    segment_img = detect_sources(data, threshold, npixels=4)
    footprint = circular_footprint(radius=3)
    mask = segment_img.make_source_mask(footprint=footprint)
    xsize = round(data.shape[1]/100)
    ysize = round(data.shape[0]/100)
    bkg = Background2D(data, (ysize,xsize), filter_size=(3, 3),mask=mask,
                       bkgrms_estimator = MADStdBackgroundRMS(sigma_clip))
    sub_data = data-bkg.background
    return sub_data, bkg


def get_host_coords(sn_ra: np.ndarray, sn_dec: np.ndarray) -> (np.ndarray, np.ndarray):
    """Get the host coordinates for a set of sne coords.
    Args:
        sn_ras: The right acensions of the supernovae.
        sn_decs: The declinations of the supernovae.
    Returns:
        1. The RAs of the host candidates.
        2. The DECs of the host candidates.
        3. The probability of chance coincidence of the candidates.
        NOTE: The returned arrays are sorted in increasing P_cc
    """

    # Get the PS1 info for those positions
    table = get_images(sn_ra, sn_dec, filters='r')

    # Download the cutout to your directory
    if not os.path.exists('/n/holystore01/LABS/berger_lab/Users/aboesky/weird_galaxy_data/ps1_dir'):
        os.mkdir('/n/holystore01/LABS/berger_lab/Users/aboesky/weird_galaxy_data/ps1_dir')
    wget.download(table['url'][0],out='/n/holystore01/LABS/berger_lab/Users/aboesky/weird_galaxy_data/ps1_dir')

    ## Load the data 
    sn_image = glob('/n/holystore01/LABS/berger_lab/Users/aboesky/weird_galaxy_data/ps1_dir/*.fits')
    sn = fits.open(sn_image)
    sn_data, sn_bkg = background_subtracted(sn[0].data)
    sn_header = sn[0].header
    sn_wcs = WCS(sn_header)
    sn.close()

    sn_x, sn_y = sn_wcs.all_world2pix(sn_ra, sn_dec, 1)
    thres = 3
    npix = 20
    threshold = thres*sn_bkg.background_rms
    segm_deblend = detect_sources(sn_data, threshold, npixels=npix)

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
    host_inds = np.where(P_cc < 0.1)

    host_ras = []
    host_decs = []
    host_P_ccs = []
    if len(host_inds) > 0:  # If there is a host!!!
        for host_ind in host_inds:

            # Get host coords
            host = cat.get_label(host_ind+1)
            host_x, host_y = (host.xcentroid,host.ycentroid)
            host_ra, host_dec = sn_wcs.all_pix2world(host_x, host_y, 1)

            # Append values
            host_ras.append(host_ra)
            host_decs.append(host_dec)
            host_P_ccs.append(P_cc[host_ind])

    # Sort candidates in increasing P_cc
    combined = sorted(zip(host_P_ccs, host_ras, host_decs))
    host_P_ccs, host_ras, host_decs = zip(*combined)

    return host_ras, host_decs, host_P_ccs








def match_host_sne():
    """Match the sne to host galaxies in the panstarrs databse and save files."""

    # Necessary params
    sr = 6/3600  # search radius [deg]

    # Grab the current data
    sne, last_searched_ind, col_types, all_res = get_current_data()
    n = len(sne)

    # Get the host candidate coords
    for i, sn_ra, sn_dec in zip(range(n)[last_searched_ind:], sne['ra'][last_searched_ind:], sne['dec'][last_searched_ind:]):
        # Get the host coordinates for the sn
        host_ras, host_decs, host_P_ccs = get_host_coords(sn_ra, sn_dec)
        print(f'{len(host_ras)} candidates found with P_cc < 0.1')

        # Search through all the candidates (in increasing order of probability) and get data
        for cand_ra, cand_dec in zip(host_ras, host_decs):

            # Convert angles to degrees and make query
            ra_ang = Angle(cand_dec.split(',')[0], unit='hourangle')
            dec_ang = Angle(f'{cand_ra.split(",")[0]} degrees')
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
                print(f'Best Result: \n{res}')

                if all_res is not None:
                    ascii.write(all_res, '/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/panstarrs_hosts_pcc.ecsv', overwrite=True, format='ecsv')
            except:
                print('Exception encountered. Continuing.')
                continue


if __name__=='__main__':
    match_host_sne()
