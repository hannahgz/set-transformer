#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gpus 1
#SBATCH -t 0-12:00
#SBATCH --mem 150G
#SBATCH -o job_seed_%A.out
#SBATCH -e job_seed_%A.err
#SBATCH -J set_seed

# Get the seed value from command line
SEED=$1

# Load your environment
eval "$(/n/sw/Mambaforge-23.3.1-1/bin/conda shell.bash hook)"
mamba activate initial

# Change to your working directory
cd /n/home08/hannahgz/thesis/set-transformer/new_datasets

# Run the script with seed argument
python batch_script_main.py $SEED
