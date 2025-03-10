import os

from subprocess import Popen
from match_panstarrs_sne_prost import get_current_data


def deploy_associaters():
    sne = get_current_data()

    # Chunk and deploy
    starting_ind = 0
    chunk_len = 1000
    chunk_counter = 0
    ps = []
    while starting_ind < len(sne):

        ending_ind = starting_ind + chunk_len
        print(f'Submitting chunk from {starting_ind} to {ending_ind}')

        # Open a pipe to the sbatch command.
        os.environ['STARTING_IND'] = str(starting_ind)
        os.environ['ENDING_IND'] = str(ending_ind)

        sbatch_command = f'sbatch --wait /n/home04/aboesky/berger/Weird_Galaxies/Paper/matching_hosts/with_prost/run_one_chunk.sh'
        proc = Popen(sbatch_command, shell=True)
        ps.append(proc)

        # Update the info
        starting_ind += chunk_len
        chunk_counter += 1
    exit_codes = [p.wait() for p in ps]  # wait for processes to finish

    return exit_codes 


if __name__=='__main__':
    deploy_associaters()
