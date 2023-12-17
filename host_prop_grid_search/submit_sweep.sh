#!/bin/bash
#
#SBATCH -p shared
#SBATCH -n 1
#SBATCH --mem-per-cpu=5G
#SBATCH -t 1-00:00                          # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o /n/home04/aboesky/berger/Weird_Galaxies/grid_search/cluster_logs/myoutput_%j.out                  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home04/aboesky/berger/Weird_Galaxies/grid_search/cluster_logs/myerrors_%j.err                  # File to which STDERR will be written, %j inserts jobid

# Remember:
# The variable $TMPDIR points to the local hard disks in the computing nodes.
# The variable $HOME points to your home directory.
# The variable $SLURM_JOBID stores the ID number of your job.

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

echo 'running script'
python3 initialize_sweep.py