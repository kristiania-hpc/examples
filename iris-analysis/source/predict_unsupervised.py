import time
import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def evaluation_report(Y_test, Y_predict):
	# Use metrics.accuracy_score to measure the score
	print("Classification Report:\n", metrics.classification_report(Y_test, Y_predict))
	print("Confusion Matrix:\n", metrics.confusion_matrix(Y_test, Y_predict))


def k_means(X_train, X_test, Y_train, Y_test):
	km = KMeans(n_clusters=3, random_state=0)
	
	# fit the model
	km.fit(X_train, Y_train)

	# create predictions
	Y_predict = km.predict(X_test)

	print("\nKMeans Clustering Accuracy: %.2f percent" % (metrics.accuracy_score(Y_test, Y_predict) * 100))
	evaluation_report(Y_test, Y_predict)


if __name__ == '__main__':

	start_time = time.time()
	print('='*30)
	print("Start Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
	print('='*30)
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
	X_train, X_test, Y_train, Y_test = train_test_split(features_std, groundtruth, test_size=0.2, random_state=32, stratify=groundtruth)

	# predict
	k_means(X_train, X_test, Y_train, Y_test)

	end_time = time.time()

	# execution run time
	runtime = round(end_time - start_time, 6)
	print("\nRuntime: ", runtime, "seconds")
	print('='*30)
	print("End Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
	print('='*30)