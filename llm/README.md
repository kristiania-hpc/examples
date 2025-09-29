# Inference and fine-tuning of a tiny LLM:

This document provides the steps to set up the environment and run inference or fine-tuning of a LLM model.

---

## 1. Create a Virtual Environment
```bash
python3 -m venv venv
```

---

## 2. Activate the Environment
```bash
source venv/bin/activate
```

---

## 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Submit the SLURM Job
To run fine-tuning or inference, submit the SLURM script:
```bash
sbatch slurm-gpu.sh
```

---

## 5. [optional] You can now view the job status

```bash
squeue -u <username>
```

---

## 6. [optional] You can view the job output

```bash
cat output/slurm%j.out
```

or view error - 

```bash
cat output/slurm%j.err
```

---