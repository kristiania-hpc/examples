# Example projects for HPC experimentation

This repository provides several practical examples for running data analysis and machine learning jobs on our Kristiania-HPC cluster using SLURM.

## Example 1: Hello World

A minimal example to test your SLURM job submission setup. Create and activate conda environment if you want [optional] before submitting slurm job. 

#### Create a conda virtual environment

```code
conda create -n testenv python==3.10
```

#### Activate the conda virtual environment

```code
conda activate testenv
```

#### To submit the job `slurm.sh` to the HPC (`helloworld/slurm.sh`)

```code
sbatch slurm.sh
```

You will get a generated output file inside `output` directory as specified in `#SBATCH --output=output/slurm%j.out`

---

## Example 2: Iris data analysis

This project demonstrates data analysis of the iris dataset using **Python**. We will be doing classification by supervised machine learning algorithm - Logistic Regression and unsupervised machine learning algorithm - Kmeans Clustering. This dataset consists of the physical parameters of three species of flower - Versicolor, Setosa, Virginica. The numeric parameters which the dataset contains are Sepal width, Sepal Length, Petal width and Petal length. In this data we will be predicting the classes of the flowers based on these parameters.

---

### Environment setup

As a good practice for any python projects, we will create a virtual environment to exercise full control via a stable, reproducible, and portable environment.

#### Create a conda virtual environment

```code
conda create -n irisenv python==3.10
```

#### Activate the conda virtual environment

```code
conda activate irisenv
```

#### Install required libraries

```code
pip install -r requirements.txt
```


#### Job submission

Submit a SLURM job script to run it in `CPUQ` partition: 

```code
sbatch slurm.sh
```

Inside the script (slurm.sh), you can specify the Python module or script to execute, such as:

```code
python3 -m source.predict_supervised
```

If your workload requires GPU resources, submit the job to the `HGXQ` partition using:

```code
sbatch slurm-gpu.sh
```

Be sure that slurm-gpu.sh includes appropriate directives for GPU access (e.g., `#SBATCH --partition=HGXQ` and `#SBATCH --gres=gpu:1`).