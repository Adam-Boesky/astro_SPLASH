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
weight_grid = ['balanced',
               {0: 1,
                1: 1,
                2: 1,
                3: 1,
                4: 1}]
w_vals = np.logspace(start=0, stop=3, base=2, num=4)
for v1 in w_vals:
    for v2 in w_vals:
        for v3 in w_vals:
            for v4 in w_vals:
                for v5 in w_vals:
                    if ~(v1 == v2 and v2==v3 and v3==v4 and v4 == v5): # don't look for equally weighted cases
                        weight_grid.append({0: v1,
                                            1: v2,
                                            2: v3,
                                            3: v4,
                                            4: v5})


def tune_parameters():
    """Tun the parameters with a grid search!"""

    # Set up resutls directory
    grid_search_results_dirpath = '/n/home04/aboesky/berger/Weird_Galaxies/flat_rf_grid_search/results'
    if os.path.exists(grid_search_results_dirpath) and os.path.isdir(grid_search_results_dirpath):
        shutil.rmtree(grid_search_results_dirpath)
    os.mkdir(grid_search_results_dirpath)

    # Conduct grid search
    ps = []
    for i, weights in enumerate(weight_grid):

        LOG.info('Submitting agent %i', i)

        # Open a pipe to the sbatch command.
        os.environ['AGENT_I'] = str(i)
        os.environ['WEIGHTS'] = str(weights)

        sbatch_command = f'sbatch --wait run_agent.sh'
        proc = Popen(sbatch_command, shell=True)
        ps.append(proc)

    exit_codes = [p.wait() for p in ps]  # wait for processes to finish
    return exit_codes


if __name__ == '__main__':
    tune_parameters()
