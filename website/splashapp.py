import pickle
import pandas as pd
import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from SPLASH.pipeline import Splash_Pipeline
import os
import shutil
from astro_ghost.PS1QueryFunctions import getAllPostageStamps
from astro_ghost.TNSQueryFunctions import getTNSSpectra
from astro_ghost.NEDQueryFunctions import getNEDSpectra
from astro_ghost.ghostHelperFunctions import getTransientHosts, getGHOST
from astropy import units as u
from datetime import datetime
from astropy.time import Time
from alerce.core import Alerce

# get rid of stupid folders...HACK
def clean_stuff():
    # The substring you are looking for in the directory names
    substring = "transients"
    # Get the current directory
    current_directory = os.getcwd()
    # List all files and directories in the current directory
    for item in os.listdir(current_directory):
        # Construct full path
        item_path = os.path.join(current_directory, item)
        # Check if it is a directory and the substring is in its name
        if os.path.isdir(item_path) and substring in item:
            # Remove the directory and all its contents
            shutil.rmtree(item_path)
            print(f"Removed directory: {item}")

def return_folder():
    # The substring you are looking for in the directory names
    substring = "transients"
    # Get the current directory
    current_directory = os.getcwd()
    # List all files and directories in the current directory
    for item in os.listdir(current_directory):
        # Construct full path
        item_path = os.path.join(current_directory, item)
        # Check if it is a directory and the substring is in its name
        if os.path.isdir(item_path) and substring in item:
            # Return the directory path
            return item_path

# should go in utils..
def ab_mag_to_flux(AB_mag: np.ndarray) -> np.ndarray:
    """Convert AB magnitude to flux in mJy"""
    return np.exp((AB_mag - 8.9) / -2.5) / 1000

ab_magerr_to_ferr = lambda sigma_m, f: np.abs(f * np.log(10) * (sigma_m / 2.5))  # transformation on the error of a magnitude turned into flux

def get_features(transients):

    #create directories to store the host spectra, the transient spectra, and the postage stamps

    grizy = transients[['gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag']].to_numpy().astype(float)
    grizy_err = transients[['gKronMagErr', 'rKronMagErr', 'iKronMagErr', 'zKronMagErr', 'yKronMagErr']].to_numpy().astype(float)
    angular_seps = transients['dist'].to_numpy().astype(float)
    # Convert the grizy data to mJy
    grizy = ab_mag_to_flux(grizy)
    grizy_err = ab_magerr_to_ferr(grizy_err, grizy)
    return grizy, grizy_err, angular_seps

def run_alerce():
    client = Alerce()
    now = datetime.now()
    formatted_date = now.strftime('%Y-%m-%dT00:00:00')
    min_firstmjd = Time(formatted_date, format="isot", scale="utc").mjd - 7

    SNe = client.query_objects(classifier="stamp_classifier",
                               class_name="SN",
                               firstmjd = min_firstmjd,
                               order_by = "oid",
                               order_mode = "DESC",
                               probability = 0.8,
                               page_size=5)
    return SNe

def run_ghost(sn_db):
    # we want to include print statements so we know what the algorithm is doing
    verbose = 1
    # download the database from ghost.ncsa.illinois.edu
    # note: real=False creates an empty database, which
    # allows you to use the association methods without
    # needing to download the full database first

    getGHOST(real=False, verbose=verbose)

    # create a list of the supernova names, their skycoords, and their classes (these three are from TNS)
    snName = list(sn_db['oid'].values)
    snCoord = []
    for i in np.arange(len(snName)):
        snCoord.append(SkyCoord(sn_db['meanra'][i]*u.deg, sn_db['meandec'][i]*u.deg, frame='icrs'))

    # hosts = getTransientHosts(snName, snCoord, verbose=True)
    hosts = getTransientHosts(transientName=snName, transientCoord=snCoord, verbose='True', starcut='gentle', ascentMatch=False)
    return hosts

def run_splash(grizy, grizy_err, angular_seps):
    # Load pipeline object
    pipeline = Splash_Pipeline(pipeline_version='weighted_full_band',   # the default version of the pipeline
                               pre_transformed=False,                   # whether the given data is pre-logged and nnormalized
                               within_4sigma=True)                      # whether we only want to classify objects with properties within 4-sigma of the training set
    # Predict the classes. n_resamples is the number of boostraps for getting the median predicted host properties.
    try:
        classes = pipeline.predict_classes(grizy, angular_seps, grizy_err, n_resamples=50)
    except:
        classes = [-1]
    return classes

# Main function to create the HTML file
def create_html():
    clean_stuff()
    sn_db = run_alerce()
    hosts = run_ghost(sn_db)

    grizy, grizy_err, angular_seps = get_features(hosts)

    class_dict = {0: 'Ia', 1: 'Ib/c', 2: 'SLSN', 3: 'IIn', 4: 'IIP/L', -1: 'N/A'}
    all_classes = np.zeros(len(grizy), dtype="<U10")
    for i in np.arange(len(grizy)):
        # RESORT
        correct_j = np.where(hosts['TransientName'] == sn_db['oid'].values[i])[0][0]
        print('running on...', i, sn_db['oid'].values[i])
        classes = run_splash(grizy[correct_j:correct_j+1], grizy_err[correct_j:correct_j+1], angular_seps[correct_j:correct_j+1])
        all_classes[i] = class_dict[classes[0]]

    sn_names = list(sn_db['oid'].values)
    supernovae = list(zip(sn_names, all_classes))

    # Define the HTML template
    html_template = '''
    <html>
        <head>
            <title>SPLASH Classification Results for today...</title>
        </head>
        <body>
            <h1>Supernova Classification Results</h1>
            <table border="1">
                <tr>
                    <th>Supernova Name</th>
                    <th>Class</th>
                </tr>
                {rows}
            </table>
        </body>
    </html>
    '''

    # Create the table rows
    rows = ''
    for sn_name, class_output in supernovae:
        rows += f'<tr><td><a href="https://alerce.online/object/{sn_name}">{sn_name}</a></td><td>{class_output}</td></tr>'

    # Render the HTML content
    html_content = html_template.format(rows=rows)

    # Write the HTML content to a file
    with open('splash.html', 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    create_html()
