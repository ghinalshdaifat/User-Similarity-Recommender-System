#!/bin/bash
#SBATCH --job-name=movie_twins
#SBATCH --output=movie_twins.out
#SBATCH --error=movie_twins.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

module purge
module load anaconda3/2024.02
export CONDA_PKGS_DIRS=/scratch/gha2009/conda-pkgs
source activate /scratch/gha2009/conda-envs/sparkenv

python /home/gha2009/capstone-bdcs-10/capstone-bdcs-10/scripts/customer_segmentation_final.py
