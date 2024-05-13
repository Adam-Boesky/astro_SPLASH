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
    'batch_size': [32, 64, 128, 256, 512, 1024, 2048, 4096],
    'nodes_per_layer': [
        [7, 10, 13], 
        [7, 9, 11, 13], 
        [6, 8, 9, 11, 13]
    ],
    'num_linear_output_layers': [1, 2, 3],
    'learning_rate': np.logspace(-3, 0, num=6)
}


def tune_parameters():
    """Tun the parameters with a grid search!"""

    # Set up resutls directory
    grid_search_results_dirpath = '/n/home04/aboesky/berger/Weird_Galaxies/domain_transfer_grid_search/results'
    if os.path.exists(grid_search_results_dirpath) and os.path.isdir(grid_search_results_dirpath):
        shutil.rmtree(grid_search_results_dirpath)
    os.mkdir(grid_search_results_dirpath)

    # Conduct grid search
    ps = []
    for i, (batch_size, nodes_per_layer, num_linear_output_layers, learning_rate) in enumerate(product(
        GRID_CONFIG['batch_size'], 
        GRID_CONFIG['nodes_per_layer'],
        GRID_CONFIG['num_linear_output_layers'], 
        GRID_CONFIG['learning_rate'])):

        LOG.info('Submitting agent %i', i)

        # Open a pipe to the sbatch command.
        os.environ['AGENT_I'] = str(i)
        os.environ['BATCH_SIZE'] = str(batch_size)
        os.environ['NODES_PER_LAYER'] = str(nodes_per_layer)
        os.environ['NUM_LINEAR_OUTPUT_LAYERS'] = str(num_linear_output_layers)
        os.environ['LEARNING_RATE'] = str(learning_rate)

        sbatch_command = f'sbatch --wait run_agent.sh'
        proc = Popen(sbatch_command, shell=True)
        ps.append(proc)

    exit_codes = [p.wait() for p in ps]  # wait for processes to finish
    return exit_codes 


if __name__ == '__main__':
    tune_parameters()
