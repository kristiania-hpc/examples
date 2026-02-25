#!/bin/bash
#SBATCH --job-name=irisCPU                    # Job name
#SBATCH --partition=CPUQ                     # GPU partition
# SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks-per-node=1                  # Tasks per node (1 for single-GPU job)
#SBATCH --cpus-per-task=4                    # CPUs allocated per task
#SBATCH --mem=50GB                           # Memory per node
#SBATCH --time=00-01:00:00                   # Walltime (DD-HH:MM:SS)
#SBATCH --output=output/slurm%j.out         # Output file (%j = job ID) in output dir
#SBATCH --error=output/slurm%j.err          # Error file

source /cluster/apps/conda/etc/profile.d/conda.sh
conda --version
conda activate iris

python --version
which python

python3 -m source.dnn-torch