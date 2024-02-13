"""Grid search to tune hyperparameters"""
import ast
import os
import shutil
import sys
from itertools import product
from subprocess import PIPE, Popen

import numpy as np

sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')

from pathlib import Path

from logger import get_clean_logger

LOG = get_clean_logger(logger_name = Path(__file__).name)

# Make parameter grids
weight_grid = ['balanced']
w_vals = np.logspace(start=0, stop=3, base=2, num=3)
for v1 in w_vals:
    for v2 in w_vals:
        for v3 in w_vals:
            for v4 in w_vals:
                if ~(v1 == v2 and v2==v3 and v3==v4):
                    weight_grid.append({1: v1,
                                        2: v2,
                                        3: v3,
                                        4: v4})
thresh_grid = np.arange(0.1, 0.6, 0.1)


def tune_parameters():
    """Tun the parameters with a grid search!"""

    # Set up resutls directory
    grid_search_results_dirpath = '/n/home04/aboesky/berger/Weird_Galaxies/domain_transfer_grid_search/results'
    if os.path.exists(grid_search_results_dirpath) and os.path.isdir(grid_search_results_dirpath):
        shutil.rmtree(grid_search_results_dirpath)
    os.mkdir(grid_search_results_dirpath)

    # Conduct grid search
    ps = []
    for i, (thresh, weights) in enumerate(product(
        thresh_grid,
        weight_grid)):

        LOG.info('Submitting agent %i', i)

        # Open a pipe to the sbatch command.
        os.environ['AGENT_I'] = str(i)
        os.environ['THRESH'] = str(thresh)
        os.environ['WEIGHTS'] = str(weights)

        sbatch_command = f'sbatch --wait run_agent.sh'
        proc = Popen(sbatch_command, shell=True)
        ps.append(proc)

    exit_codes = [p.wait() for p in ps]  # wait for processes to finish
    return exit_codes 


if __name__ == '__main__':
    tune_parameters()
