#!/bin/bash

#SBATCH -c 10                               # Number of cores (-c)
#SBATCH --job-name=Test                     # This is the name of your job
#SBATCH --mem=50G                          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-00:10                          # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o myoutput_%j.out                  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err                  # File to which STDERR will be written, %j inserts jobid

#SBATCH -p test

# Remember:
# The variable $TMPDIR points to the local hard disks in the computing nodes.
# The variable $HOME points to your home directory.
# The variable $SLURM_JOBID stores the ID number of your job.


export WANDB_API_KEY=6ecd8ea5ceb5a64219d98bc34ce67af0904f2be8

# # Load modules
# #################################
module load python/3.10.12-fasrc01
source activate wandb_env

# # Commands
# #############################
# wandb login
# cd /n/home04/aboesky/berger/Weird_Galaxies
# python initialize_sweep.py
cd /n/home04/aboesky/berger/Weird_Galaxies/grid_search

config_yaml='/n/home04/aboesky/berger/Weird_Galaxies/grid_search/sweep_config.yaml'
echo 'config:' $config_yaml

train_file='/n/home04/aboesky/berger/Weird_Galaxies/grid_search/agent_model.py'
echo 'train_file:' $train_file

project_name='Astronomy_98'
echo 'project_name:' $project_name

echo 'running script'
python initialize_sweep.py $config_yaml $train_file $project_name