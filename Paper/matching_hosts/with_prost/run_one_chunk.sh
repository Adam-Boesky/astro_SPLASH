#!/bin/bash

#SBATCH -p shared
#SBATCH -c 12                                       # Number of cores (-c)
#SBATCH --mem=56G                                   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-01:00                                  # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o /n/home04/aboesky/berger/Weird_Galaxies/Paper/matching_hosts/with_prost/logs/myoutput_\%j.out                          # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home04/aboesky/berger/Weird_Galaxies/Paper/matching_hosts/with_prost/logs/myoutput_\%j.err                          # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=aboesky@college.harvard.edu     # Send email to user

python3 /n/home04/aboesky/berger/Weird_Galaxies/Paper/matching_hosts/with_prost/match_panstarrs_sne_prost.py -b=$STARTING_IND -e=$ENDING_IND
