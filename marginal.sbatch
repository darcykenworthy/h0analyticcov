#!/bin/bash

#SBATCH --account=pi-rkessler
#SBATCH --job-name=pecvelmcmc
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=01:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --mem=2G

# Print the task id.
source activate pystan
python marginal.py "${@:1}"

