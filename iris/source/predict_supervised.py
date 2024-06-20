import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score


def evaluation_report(Y_test, Y_predict):
    # Use metrics.accuracy_score to measure the score
    print("Classification Report:\n",
          metrics.classification_report(Y_test, Y_predict))
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_predict)
    print("Confusion Matrix:\n", confusion_matrix)
    return confusion_matrix


def logistic_regression(X_train, X_test, Y_train, Y_test):
    lr = LogisticRegression(
        C=1, solver='saga', penalty='elasticnet', l1_ratio=0.5, random_state=0)

    # fit the model
    lr.fit(X_train, Y_train)

    # create predictions
    Y_predict = lr.predict(X_test)

    print("\nLogisticRegression Accuracy: %.2f percent" %
          (metrics.accuracy_score(Y_test, Y_predict) * 100))
    confusion_mat = evaluation_report(Y_test, Y_predict)

    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
    lr_cv = cross_val_score(lr, X_train, Y_train, cv=kfolds)
    print("\nAccuracy: %.2f%% (%.2f%%)" % (lr_cv.mean()*100, lr_cv.std()*100))

    return confusion_mat


if __name__ == '__main__':

    start_time = time.time()
    print('='*30)
    print("Start Time:", time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    print('='*30)
    # load dataset
    df = pd.read_csv('dataset/iris.csv')
    df = df.drop(columns=['Id'])

    # encode categorical variables into numeric
    encoder = LabelEncoder()
    classes = df['Species'].unique()
    encoder.fit(classes)
    encoded_Species = encoder.transform(np.ravel(df['Species']))
    df[df.columns[4]] = encoded_Species

    # drop features SepalLengthCm and SepalWidthCm
    df = df.drop(columns=['SepalLengthCm', 'SepalWidthCm'])

    features = df.iloc[:, [0, 1]]
    groundtruth = np.ravel(df.iloc[:, [2]])

    # scale features
    sc = StandardScaler()
    sc.fit(features)
    features_std = pd.DataFrame(sc.transform(features))

    # split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(
        features_std, groundtruth, test_size=0.2, random_state=0, stratify=groundtruth)

    # predict
    confusion_mat = logistic_regression(X_train, X_test, Y_train, Y_test)

    end_time = time.time()
    # execution run time
    runtime = round(end_time - start_time, 6)
    print("\nRuntime: ", runtime, "seconds")
    print('='*30)
    print("End Time:", time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    print('='*30)

    # Create the directory if it does not exist
    Path("figure").mkdir(parents=True, exist_ok=True)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('figure/confusion_matrix.png')
