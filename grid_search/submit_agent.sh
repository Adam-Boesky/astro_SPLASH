#!/bin/bash

#SBATCH -c 12                                       # Number of cores (-c)
#SBATCH --job-name=i33uxph0_agent_1                       # This is the name of your job
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
wandb agent i33uxph0 --project "Astronomy 98"
