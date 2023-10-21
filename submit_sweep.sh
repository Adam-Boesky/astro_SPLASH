#!/bin/bash

#SBATCH -c 48                               # Number of cores (-c)
#SBATCH --job-name=Test                     # This is the name of your job
#SBATCH --mem=184G                          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-00:10                          # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o myoutput_%j.out                  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err                  # File to which STDERR will be written, %j inserts jobid

#SBATCH -p test

# Remember:
# The variable $TMPDIR points to the local hard disks in the computing nodes.
# The variable $HOME points to your home directory.
# The variable $SLURM_JOBID stores the ID number of your job.


# Load modules
#################################
module load python/3.10.12-fasrc01
conda activate ay98

# Commands
#############################
python model_grid_search.py
