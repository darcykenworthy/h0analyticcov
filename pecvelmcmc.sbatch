#!/bin/bash

#SBATCH --account=pi-rkessler
#SBATCH --job-name=pecvelmcmc
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --array=0-16
#SBATCH --partition=broadwl
#SBATCH --ntasks=4
#SBATCH --mem=4G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
source activate pystan
python pecvelmcmc.py "$1" "$2" "$SLURM_ARRAY_TASK_ID" --niter "$3" --nchains 4 "${@:4}"

