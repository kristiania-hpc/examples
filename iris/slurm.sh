#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=LocalQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCh --cpus-per-task=1
#SBATCH --time=00:11:00
#SBATCH --output=output/slurm%j.out
#SBATCH --error=output/slurm%j.err

conda activate irisenv

python --version
which python

python3 -m source.predict_supervised
