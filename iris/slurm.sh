#!/bin/bash
#SBATCH --job-name=iris
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCh --cpus-per-task=1
#SBATCH --time=00:11:00
#SBATCH --output=output/slurm%j.out
#SBATCH --error=output/slurm%j.err

source /cluster/storage/anaconda3/etc/profile.d/conda.sh
which conda
conda --version
conda activate iris

python --version
which python

python3 -m source.predict_supervised
