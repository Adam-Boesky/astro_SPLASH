import os

from subprocess import Popen
from match_panstarrs_sne_prost import get_current_data


def deploy_associaters():
    sne = get_current_data()

    # Chunk and deploy
    starting_ind = 0
    chunk_len = 100
    chunk_counter = 0
    ps = []
    while starting_ind < len(sne):

        ending_ind = starting_ind + chunk_len
        print(f'Submitting chunk from {starting_ind} to {ending_ind}')

        # Make sure that the chunk hasn't already been associated
        out_fname = f'/Users/adamboesky/Research/ay98/clean_data/panstarrs_hosts_prost/panstarrs_hosts_prost{starting_ind}_{ending_ind}.ecsv'
        if os.path.exists(out_fname):
            print(f'Chunk from {starting_ind} to {ending_ind} already associated. Skipping.')
        else:
            # Associate chunk
            sbatch_command = f'python3 match_panstarrs_sne_prost.py -b={starting_ind} -e={ending_ind}'
            proc = Popen(
                sbatch_command,
                preexec_fn=os.setsid,  # Detach the process
                shell=True)
            exit_code = proc.wait()  # Wait for the current process to finish
            print(f'Chunk from {starting_ind} to {ending_ind} associated and stored to {out_fname} with exit code {exit_code}!')

        # Update the info
        starting_ind += chunk_len
        chunk_counter += 1


if __name__=='__main__':
    deploy_associaters()
