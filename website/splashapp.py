import os
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
#from astro_ghost.PS1QueryFunctions import geturl
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
from bs4 import BeautifulSoup

import SPLASH
from SPLASH.pipeline import Splash_Pipeline


# For now, build in a custom PS1 photo-z code
import torch
import torch.nn as nn

class PhotoZMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.z_mean = nn.Linear(hidden_dim, 1)
        self.z_std = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.net(x)
        z_mu = self.z_mean(h).squeeze(-1)
        raw_std = self.z_std(h).squeeze(-1)
        z_sigma = torch.exp(raw_std)
        z_sigma = torch.clamp(z_sigma, min=1e-2, max=10.0)
        return z_mu, z_sigma

def estimate_photoz_ps1(g, r, i, z, y):
    class PhotoZMLP(torch.nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            )
            self.z_mean = torch.nn.Linear(hidden_dim, 1)
            self.z_std = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x):
            h = self.net(x)
            z_mu = self.z_mean(h).squeeze(-1)
            raw_std = self.z_std(h).squeeze(-1)
            z_sigma = torch.exp(raw_std)
            z_sigma = torch.clamp(z_sigma, min=1e-2, max=10.0)
            return z_mu, z_sigma

    # Convert mags to float32 and mask
    mags = np.array([g, r, i, z, y], dtype=np.float32)
    mask = ~np.isnan(mags)
    mags_clean = np.nan_to_num(mags, nan=0).astype(np.float32)
    mask = mask.astype(np.float32)

    x = np.concatenate([mags_clean, mask]).reshape(1, -1)
    x_tensor = torch.tensor(x, dtype=torch.float32) # enforce the type!

    # Initialize model + convert weights to float32
    model = PhotoZMLP()
    state_dict = torch.load("photoz_ps1_model.pt", map_location="cpu")

    # Just in case the weights were saved as float64
    for k in state_dict:
        if state_dict[k].dtype == torch.float64:
            state_dict[k] = state_dict[k].float()

    model.load_state_dict(state_dict)
    model = model.float()  # <- ENFORCED TYPE
    model.eval()

    with torch.no_grad():
        z_mu, z_std = model(x_tensor)
        return z_mu.item(), z_std.item()


def geturl(ra, dec, size=240):
    """
    Construct a PS1 multi-band JPEG URL using the actual FITS file paths (g/i/r) and fitscut.cgi.

    Parameters:
    - ra, dec: Coordinates in degrees
    - size: Image size in pixels

    Returns:
    - A URL to a JPEG image showing the g/i/r composite
    """
    url = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters=grizy"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"PS1 filename query failed for RA={ra}, Dec={dec}")
        return None

    lines = response.text.strip().split('\n')
    if len(lines) <= 1:
        print(f"No data returned for RA={ra}, Dec={dec}")
        return None

    # Parse table lines
    header = lines[0].split()
    idx_filter = header.index("filter")
    idx_filename = header.index("filename")

    fits_paths = {}
    for line in lines[1:]:
        fields = line.split()
        band = fields[idx_filter]
        path = fields[idx_filename]
        if band in ['g', 'r', 'i']:
            fits_paths[band] = path

    if not all(b in fits_paths for b in ['g', 'r', 'i']):
        print(f"Missing bands for RA={ra}, Dec={dec}")
        return None

    # Build the fitscut URL using g → blue, i → green, r → red (standard PS1 composite)
    fitscut_url = (
    f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
    f"red={fits_paths['r']}"
    f"&green={fits_paths['i']}"
    f"&blue={fits_paths['g']}"
    f"&x={ra}&y={dec}"
    f"&size={size}&output_size={size}"
    f"&autoscale=99.75&asinh=True&format=jpg"
    f"&wcs=1"
)

    return fitscut_url


# get rid of folders...hacky
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

def run_alerce():
    from alerce.core import Alerce
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

def run_TNS():
    # This is a hacky function. Currently, the TNS API doesn't support
    # what we need..so this scrapes the website instead
    three_days_ago = datetime.today() - timedelta(days=10)
    three_days_ago_date = three_days_ago.strftime('%Y-%m-%d')


    url = 'https://www.wis-tns.org/search?&reporting_groupid%5B%5D=74&at_type%5B%5D=1&date_start%5Bdate%5D=' + three_days_ago_date

    #hack
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36",
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content = response.text

    soup = BeautifulSoup(content, 'html.parser')
    rows = soup.find_all('tr', class_='row-even public odd')
    db_rows = []
    for row in rows:
        if 'cell-name' not in str(row):
            continue
        name = row.find('td', class_='cell-name').get_text(strip=True).replace(" ", "")
        try:
            # If no internal name, continue (likely a Blackgem)
            internal_name = row.find('td', class_='cell-internal_name').get_text(strip=True).replace(" ", "")
        except:
            print(name+" appears to be a non-ZTF candidate; ANTARES page will not load.")
            continue
        ra = row.find('td', class_='cell-ra').get_text(strip=True)
        ra_deg = Angle(ra, unit=u.hourangle).degree

        decl = row.find('td', class_='cell-decl').get_text(strip=True)
        dec_deg = Angle(decl, unit=u.deg).degree

        db_rows.append((name, internal_name, ra_deg, dec_deg))

    SNe = pd.DataFrame(db_rows, columns=['oid', 'internal-name', 'meanra', 'meandec'])
    return SNe



def get_host_pics(ra, dec):
    if dec > -30:
        ps1_pic = geturl(ra, dec)
    else:
        ps1_pic = ""
    return ps1_pic

def run_splash(sn_db):
    n = len(sn_db)

    # Initialize outputs
    classes = np.full(n, -3, dtype=int)
    Ia_prob = np.full(n, np.nan)
    SLSN_prob = np.full(n, np.nan)
    mass = np.full(n, np.nan)
    sfr = np.full(n, np.nan)
    mass_err = np.full(n, np.nan)
    sfr_err = np.full(n, np.nan)
    redshift = np.full(n, np.nan)
    redshift_err = np.full(n, np.nan)
    ps1_photoz_flag = np.full(n, False, dtype=bool)


    for i, row in sn_db.iterrows():
        pipeline = Splash_Pipeline(
            pre_transformed=False,
            within_4sigma=True,
            nan_thresh_ratio=1.0,
        )
        print(f"\nProcessing {row['oid']} ({i+1}/{n})")

        # Step 1: Get catalog
        catalog = pipeline.get_transient_catalog(
            ra=np.array([row['meanra']]),
            dec=np.array([row['meandec']])
        )

        # Fix any zero redshifts
        if catalog.loc[0, 'host_redshift_mean'] == 0:
            catalog.loc[0, 'host_redshift_mean'] = np.nan
            catalog.loc[0, 'host_redshift_std'] = np.nan

        # Step 2: Use photo-z for PS1
        if str(catalog.loc[0, "best_cat"]).lower() == "panstarrs":
            g = catalog.get("gKronMag", np.nan)
            r = catalog.get("rKronMag", np.nan)
            i_ = catalog.get("iKronMag", np.nan)
            z = catalog.get("zKronMag", np.nan)
            y = catalog.get("yKronMag", np.nan)
            ps1_photoz_flag[i] = True

            z_phot, z_err = estimate_photoz_ps1(g, r, i_, z, y)

            # Clip unphysical values
            if z_phot < 0:
                print(f"WARNING: Negative photo-z ({z_phot:.3f}) for {row['oid']}. Replacing with NaN.")
                z_phot = np.nan
                z_err = np.nan

            catalog.loc[0, "host_redshift_mean"] = z_phot
            catalog.loc[0, "host_redshift_std"] = z_err
            print(f" Used photo-z: z = {z_phot if not np.isnan(z_phot) else 'NaN'} ± {z_err if not np.isnan(z_err) else 'NaN'}")


        z_use = catalog.loc[0, "host_redshift_mean"]
        zerr_use = catalog.loc[0, "host_redshift_std"]

        # Step 3: Build photometry arrays
        grizy = np.array([[
            catalog["gKronMag"].iloc[0],
            catalog["rKronMag"].iloc[0],
            catalog["iKronMag"].iloc[0],
            catalog["zKronMag"].iloc[0],
            catalog["yKronMag"].iloc[0],
        ]], dtype=np.float32)

        grizy_err = np.array([[
            catalog["gKronMagErr"].iloc[0],
            catalog["rKronMagErr"].iloc[0],
            catalog["iKronMagErr"].iloc[0],
            catalog["zKronMagErr"].iloc[0],
            catalog["yKronMagErr"].iloc[0],
        ]], dtype=np.float32)

        # Clean bad values
        grizy[grizy < -10] = np.nan
        grizy_err[grizy_err < 0] = np.nan

        # Step 4: Host inference
        host_props, host_props_err, _ = pipeline.infer_host_properties(
            ra=np.array([row["meanra"]]),
            dec=np.array([row["meandec"]]),
            grizy=grizy,
            grizy_err=grizy_err,
            redshift=np.array([z_use]),
            redshift_err=np.array([zerr_use]),
            n_resamples=10,
            return_no_host_mask=True
        )

        redshift[i] = z_use
        redshift_err[i] = zerr_use
        mass[i] = host_props[0, 0]
        sfr[i] = host_props[0, 1]
        mass_err[i] = host_props_err[0, 0]
        sfr_err[i] = host_props_err[0, 1]

        print(f"Host: z={z_use:.3f}, mass={mass[i]:.2f}, sfr={sfr[i]:.2f}")

        # Step 5: Classification if host info valid
        if not pd.isna(mass[i]):
            classes[i] = pipeline.infer_classes(
                ra=np.array([row['meanra']]),
                dec=np.array([row['meandec']]),
                grizy=grizy,
                grizy_err=grizy_err,
                redshift=np.array([z_use]),
                redshift_err=np.array([zerr_use]),
                n_resamples=10
            )[0]

            probs = pipeline.infer_probs(
                ra=np.array([row['meanra']]),
                dec=np.array([row['meandec']]),
                grizy=grizy,
                grizy_err=grizy_err,
                redshift=np.array([z_use]),
                redshift_err=np.array([zerr_use]),
                n_resamples=10
            )

            Ia_prob[i] = probs[0, 0]
            SLSN_prob[i] = probs[0, 2]

    return classes, Ia_prob, mass, sfr, redshift, mass_err, sfr_err, redshift_err, SLSN_prob, ps1_photoz_flag

# Main function to create the HTML file
def create_html():
    clean_stuff()
    sn_db = run_TNS()
    sn_db = sn_db[-3:].reset_index(drop=True)

    # sn_db = run_alerce() # Removed following suggestion from Alerce

    class_dict = {0: 'Ia', 1: 'Ib/c', 2: 'SLSN', 3: 'IIn', 4: 'IIP/L', -1: 'Out of range', -2: 'Hostless? (beta)', -3:"Outside of range!"}
    classes, Ia_prob, mass, sfr, redshift, mass_err, sfr_err, redshift_err, SLSN_prob, ps1_flag = run_splash(sn_db)

    # Get current date and SPLASH version
    current_date = datetime.now().strftime("%Y-%m-%d")
    splash_version = SPLASH.__version__  # Replace this with the actual SPLASH version number if dynamic retrieval is possible
    github_link = 'https://github.com/Adam-Boesky/astro_SPLASH'

    html_template = '''
        <html>
            <head>
                <title>SPLASH Classification Results for {date}</title>
                <style>
                    body {{
                        background-color: #fdf6f0;
                        color: #2e1a1a;
                        font-family: 'Georgia', serif;
                        margin: 0;
                        padding: 20px;
                    }}
                    h1 {{
                        text-align: center;
                        color: #8a100b;
                        font-size: 2.8em;
                        margin-bottom: 30px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                        box-shadow: 0 0 10px rgba(138, 16, 11, 0.2);
                    }}
                    th, td {{
                        padding: 14px;
                        text-align: center;
                    }}
                    th {{
                        background-color: #8a100b;
                        color: #fffaf5;
                        font-size: 1.1em;
                        text-transform: uppercase;
                        border-bottom: 2px solid #660000;
                    }}
                    td {{
                        background-color: #fffaf5;
                        border-bottom: 1px solid #ddd;
                    }}
                    tr:hover {{
                        background-color: #fbe9e7;
                        transition: background-color 0.3s;
                    }}
                    a {{
                        color: #8a100b;
                        text-decoration: none;
                        font-weight: bold;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    img {{
                        border-radius: 8px;
                        transition: transform 0.3s;
                    }}
                    img:hover {{
                        transform: scale(1.05);
                    }}
                    footer {{
                        margin-top: 40px;
                        text-align: center;
                        font-size: 0.9em;
                        color: #555;
                    }}
                </style>
            </head>
            <body>
                <h1>Supernova Classification Results</h1>
                <table>
                    <tr>
                        <th>Supernova Name</th>
                        <th>SPLASH Class Prediction</th>
                        <th>Ia Probability</th>
                        <th>SLSN Probability</th>
                        <th>Mass</th>
                        <th>SFR</th>
                        <th>Redshift</th>
                        <th>Host Galaxy Image</th>
                    </tr>
                    {rows}
                </table>
                <footer>
                    <p>* Redshifts marked with an asterisk were inferred using a beta photometric redshift model trained on PS1 data.</p>
                    <p>SPLASH Version: {version} &mdash; <a href="{github}">SPLASH GitHub</a></p>
                </footer>
            </body>
        </html>
        '''


    # Create the table rows
    rows = ''
    for i, _ in enumerate(classes):
        sn_name = sn_db['oid'].iloc[i]
        internal_name = sn_db['internal-name'].iloc[i]
        class_output = classes[i]
        ia_prob = Ia_prob[i]
        slsn_prob = SLSN_prob[i]
        ps1_pic = get_host_pics(sn_db['meanra'].iloc[i], sn_db['meandec'].iloc[i])
        img_html = f'<img src="{ps1_pic}" width="100" height="100">' if ps1_pic else "N/A"

        # Redshift display with asterisk if from PS1 photo-z
        z_label = f'{redshift[i]:.2f} ± {redshift_err[i]:.2f}'
        if ps1_flag[i]:
            z_label += '*'  # Add asterisk to redshift

        rows += f'<tr><td><a href="https://alerce.online/object/{internal_name}">{sn_name}</a></td>'
        rows += f'<td>{class_dict.get(class_output, "Unknown")}</td>'
        rows += f'<td>{ia_prob:.3f}</td><td>{slsn_prob:.3f}</td>'
        rows += f'<td>{mass[i]:.2f} ± {mass_err[i]:.2f}</td>'
        rows += f'<td>{sfr[i]:.2f} ± {sfr_err[i]:.2f}</td>'
        rows += f'<td>{z_label}</td>'
        rows += f'<td>{img_html}</td></tr>'

    # Render the HTML content
    html_content = html_template.format(rows=rows, date=current_date, version=splash_version, github=github_link)

    # Write the HTML content to a file
    with open('splash.html', 'w') as f:
        f.write(html_content)



if __name__ == '__main__':
    create_html()
