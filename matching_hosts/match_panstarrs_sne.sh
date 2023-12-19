#!/bin/bash

#SBATCH -p test
#SBATCH -c 12                                       # Number of cores (-c)
#SBATCH --mem=12G                                   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-01:00                          # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o /n/home04/aboesky/berger/Weird_Galaxies/matching_logs/myoutput_\%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home04/aboesky/berger/Weird_Galaxies/matching_logs/myerrors_\%j.err                          # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=aboesky@college.harvard.edu     # Send email to user

# Remember:
# The variable $TMPDIR points to the local hard disks in the computing nodes.
# The variable $HOME points to your home directory.
# The variable $SLURM_JOBID stores the ID number of your job.

# # Load modules
# #################################
echo "Loading environment"
module load python/3.10.12-fasrc01
source activate ay98

echo "Runnning script"
python /n/home04/aboesky/berger/Weird_Galaxies/matching_hosts/match_panstarrs_sne_pcc.py
