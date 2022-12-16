#!/bin/bash -l

#SBATCH --job-name=RCB_sim
#SBATCH --output=%x-%j.out

#SBATCH --mem=1G
#SBATCH --time=1-000000   

# default
# #SBATCH -N 1 #nodes
# #SBATCH -n 2 #tasks
# #SBATCH --cpus-per-task=2
# #SBATCH --gres=gpu2

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vgraf@cs.princeton.edu

tag=RCB_sim;
OUT_DIRECTORY='output.'${tag}
# mkdir ${OUT_DIRECTORY};

source ~/.bashrc
srun python3 RCB_simulation.py 

exit 0