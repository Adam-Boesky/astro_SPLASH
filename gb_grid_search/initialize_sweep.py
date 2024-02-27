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
GRID_CONFIG = {
    'max_bin': np.linspace(200, 1500, num=3),           # ***  [200 ---> 15000]
    'num_iterations': np.linspace(100, 1000, num=4),    # ***  [100 ---> 1000]
    'learning_rate': np.logspace(-4, -1, num=4),        # ***  [0.1  ---> 0.0001]
    'num_leaves': np.linspace(30, 150, num=5)           # ***  [30 ---> 150]
}


def tune_parameters():
    """Tun the parameters with a grid search!"""

    # Set up resutls directory
    if os.path.exists('/n/home04/aboesky/berger/Weird_Galaxies/gb_grid_search/results') and os.path.isdir('/n/home04/aboesky/berger/Weird_Galaxies/grid_search/results'):
        shutil.rmtree('/n/home04/aboesky/berger/Weird_Galaxies/gb_grid_search/results')
    os.mkdir('/n/home04/aboesky/berger/Weird_Galaxies/gb_grid_search/results')

    # Conduct grid search
    ps = []
    for i, (max_bin, num_iter, lr, num_leaf) in enumerate(product(
        GRID_CONFIG['max_bin'], 
        GRID_CONFIG['num_iterations'],
        GRID_CONFIG['learning_rate'], 
        GRID_CONFIG['num_leaves'])):

        LOG.info('Submitting agent %i', i)

        # Open a pipe to the sbatch command.
        os.environ['AGENT_I'] = str(i)
        os.environ['MAX_BINS'] = str(max_bin)
        os.environ['NUM_ITERS'] = str(num_iter)
        os.environ['LEARNING_RATE'] = str(lr)
        os.environ['NUM_LEAVES'] = str(num_leaf)

        sbatch_command = f'sbatch --wait run_agent.sh'
        proc = Popen(sbatch_command, shell=True)
        ps.append(proc)

    exit_codes = [p.wait() for p in ps]  # wait for processes to finish
    return exit_codes 


if __name__ == '__main__':
    tune_parameters()
