"""Grid search to tune hyperparameters"""
import sys
import time
from subprocess import Popen, PIPE
import click

sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')

import wandb
import yaml

from pathlib import Path
from logger import get_clean_logger
from neural_net import load_and_preprocess, get_model, get_tensor_batch, checkpoint, resume, CustomLoss

LOG = get_clean_logger(logger_name = Path(__file__).name)
PROJECT = 'Astronomy 98'
# with open('/n/home04/aboesky/berger/Weird_Galaxies/sweep_config.yaml', 'r') as f:
#     SWEEP_CONFIG = yaml.safe_load(f)
# SWEEP_BASH_SCRIPT = """#!/bin/bash

# #SBATCH -c 48                                       # Number of cores (-c)
# #SBATCH --job-name={job_name}                       # This is the name of your job
# #SBATCH --mem=184G                                  # Memory pool for all cores (see also --mem-per-cpu)
# #SBATCH -t 0-12:00                                  # Runtime in D-HH:MM, minimum of 10 minutes

# # Paths to STDOUT or STDERR files should be absolute or relative to current working directory
# #SBATCH -o cluster_logs/myoutput_\%j.out                          # File to which STDOUT will be written, %j inserts jobid
# #SBATCH -e cluster_logs/myerrors_\%j.err                          # File to which STDERR will be written, %j inserts jobid
# #SBATCH --mail-user=aboesky@college.harvard.edu     # Send email to user

# #SBATCH -p shared

# # Remember:
# # The variable $TMPDIR points to the local hard disks in the computing nodes.
# # The variable $HOME points to your home directory.
# # The variable $SLURM_JOBID stores the ID number of your job.


# # Load modules
# #################################
# module load python/3.10.12-fasrc01
# conda activate ay98

# # Commands
# #############################
# export WANDB_API_KEY=6ecd8ea5ceb5a64219d98bc34ce67af0904f2be8
# wandb agent {sweep_id}
# # """
SWEEP_BASH_SCRIPT = """#!/bin/bash

#SBATCH -c 12                                       # Number of cores (-c)
#SBATCH --job-name={job_name}                       # This is the name of your job
#SBATCH --mem=56G                                   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-00:10                                  # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o cluster_logs/myoutput_\%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cluster_logs/myerrors_\%j.err                          # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=aboesky@college.harvard.edu     # Send email to user

#SBATCH -p test

# Remember:
# The variable $TMPDIR points to the local hard disks in the computing nodes.
# The variable $HOME points to your home directory.
# The variable $SLURM_JOBID stores the ID number of your job.

# Export environment variables
#################################
export WANDB_API_KEY=6ecd8ea5ceb5a64219d98bc34ce67af0904f2be8

# Load modules
#################################


# Commands
#############################
wandb login
wandb agent {sweep_id} --project "{project}"
"""

@click.command()
@click.argument("config_yaml")
@click.argument("train_file")
@click.argument("project_name")
def tune_parameters(config_yaml, train_file, project_name):
    """Use grid search to tune hyperparameters."""
    # # Set api key
    # os.environ["WANDB_API_KEY"] = '6ecd8ea5ceb5a64219d98bc34ce67af0904f2be8'

    # # Make sweep and set sweep ID environment variable
    # wandb.init(project=PROJECT)
    # sweep_id = wandb.sweep(SWEEP_CONFIG, project=PROJECT)

    # # Submit a number of agents to complete the sweep
    # ps = []
    # for i in range(2):
    #     job_name = f'{sweep_id}_agent_{i}'
    #     sbatchFile = open('submit_agent.sh', 'w')
    #     LOG.info('Submitting agent %i', i)
    #     sbatchFile.write(SWEEP_BASH_SCRIPT.format(job_name=job_name, sweep_id=sweep_id))
    #     sbatchFile.close()

    #     # Open a pipe to the sbatch command.
    #     sbatch_command = f'sbatch --wait submit_agent.sh'
    #     proc = Popen(sbatch_command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
    #     ps.append(proc)

    #     # Send job_string to sbatch
    #     if (sys.version_info > (3, 0)):
    #         proc.stdin.write(sbatch_command.encode('utf-8'))
    #     else:
    #         proc.stdin.write(sbatch_command)

    #     LOG.info('\tsbatch command: %s', sbatch_command)
    #     out, err = proc.communicate()
    #     LOG.info("\tout = %s", out)
    #     job_id = out.split()
    #     LOG.info("\tjob_id: %s", job_id)
    #     LOG.info("\terror: %s", err)

    # outputs = [p.wait() for p in ps]
    # print(outputs)
    wandb.init(project=project_name)
    
    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict['program'] = train_file

    sweep_id = wandb.sweep(config_dict, project=project_name)

    time.sleep(10)
    
    ps = []
    for i in range(2):
        LOG.info('Submitting agent %i', i)
        job_name = f'{sweep_id}_agent_{i}'
        sbatchFile = open('submit_agent.sh', 'w')
        sbatchFile.write(SWEEP_BASH_SCRIPT.format(job_name=job_name, sweep_id=sweep_id, project=project_name))
        sbatchFile.close()

        # Open a pipe to the sbatch command.
        sbatch_command = f'sbatch --wait submit_agent.sh'
        proc = Popen(sbatch_command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)
        ps.append(proc)

    exit_codes = [p.wait() for p in ps]  # wait for processes to finish
    return exit_codes 


if __name__ == '__main__':
    tune_parameters()
