#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCh --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=output/slurm%j.out
#SBATCH --error=output/slurm%j.err

echo "Hello world"
source /cluster/storage/anaconda3/etc/profile.d/conda.sh
which conda 
conda --version
conda activate iris
python3 --version
which python3

python3 test.py
