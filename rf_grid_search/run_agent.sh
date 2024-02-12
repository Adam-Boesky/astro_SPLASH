#!/bin/bash

#SBATCH -p shared
#SBATCH -c 12                                       # Number of cores (-c)
#SBATCH --mem=56G                                   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-16:00                          # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o cluster_logs/myoutput_\%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cluster_logs/myerrors_\%j.err                          # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=aboesky@college.harvard.edu     # Send email to user

# Remember:
# The variable $TMPDIR points to the local hard disks in the computing nodes.
# The variable $HOME points to your home directory.
# The variable $SLURM_JOBID stores the ID number of your job.

python3 /n/home04/aboesky/berger/Weird_Galaxies/rf_grid_search/run_agent.py $AGENT_I $BATCH_SIZE "$NODES_PER_LAYER" $NUM_LINEAR_OUTPUT_LAYERS $LEARNING_RATE