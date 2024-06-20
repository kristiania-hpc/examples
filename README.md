# Examples projects for HPC experimentation

---

## Example 1: Hello World

#### Create A Conda Virtual Environment

```code
conda create -n testenv python==3.10
```

#### Activate The Conda Virtual Environment

```code
conda activate testenv
```

#### To submit the job `slurm.sh` to the HPC

```code
sbatch slurm.sh
```

You can generated output files inside `output` directory as specified in `#SBATCH --output=output/slurm%j.out`

---

## Example 2: Iris_Dataset_Analysis

This project demonstrates advanced data analysis of the iris dataset using **Python**. We will be doing classification by supervised machine learning algorithm - Logistic Regression and unsupervised machine learning algorithm - Kmeans Clustering.

### Problem Statement

This dataset consists of the physical parameters of three species of flower - Versicolor, Setosa, Virginica. The numeric parameters which the dataset contains are Sepal width, Sepal Length, Petal width and Petal length. In this data we will be predicting the classes of the flowers based on these parameters.

---

### Environment Setup

As a good practice for any python projects, we will create a virtual environment to exercise full control via a stable, reproducible, and portable environment.

#### Create A Conda Virtual Environment

```code
conda create -n irisenv python==3.10
```

#### Activate The Conda Virtual Environment

```code
conda activate irisenv
```

#### Install Required Libraries

```code
pip install -r requirements.txt
```

### Advanced Data Analysis

In this data exploration, we retrieved crucial information such as skewness, correlation and even feature importances by XGBClassifier to help us made informed choices. In the end, we decided to keep only PetalLengthCm for our prediction.

#### To Run Analysis

```code
python3 -m source.analysis
```

---

### Prediction (Supervised)

As mentioned above, we will be using **Logistic Regression** for the demonstration of classifying the iris dataset. We will also be using some common machine learning metrics that will help us gauge the performance of our model. In the end, we achieved a model accuracy of 96.67% in 0.0497s runtime with only PetalLengthCm and PetalWidthCm. Using cross-validation, we also noted that our mean accuracy is 95.83% (4.17% standard deviation).

#### To Run Prediction

```code
python3 -m source.predict_supervised
```

---

### Prediction (Unsupervised)

As mentioned above, we will be using **Kmeans Clustering** for the demonstration of classifying the iris dataset. We will also be using some common machine learning metrics that will help us gauge the performance of our model. In the end, we achieved a model accuracy of 96.67% in 0.117s runtime with only PetalLengthCm and PetalWidthCm. Note here that cross-validation is not suitable for this application.

#### To Run Prediction

```code
python3 -m source.predict_unsupervised
```
