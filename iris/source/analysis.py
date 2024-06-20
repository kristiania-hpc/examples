import os
import time
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


if __name__ == '__main__':

	start_time = time.time()

	# load dataset
	#load_dotenv(verbose=True)
	#iris_dataset = os.getenv("IRIS_DATASET")

	df = pd.read_csv('dataset/iris.csv')
	df = df.drop(columns=['Id'])

	print("\nFirst 5 Rows of Dataset:")
	print(df.head())

	# check for NaN values and data types
	print("\nDataset Information:")
	print(df.info(memory_usage=False))

	# encode categorical variables into numeric
	encoder = LabelEncoder()
	classes = df['Species'].unique()
	encoder.fit(classes)
	encoded_Species = encoder.transform(np.ravel(df['Species']))
	df[df.columns[4]] = encoded_Species

	# check features correlation
	print("\nFeatures Correlation:")
	print(df.corr())

	# distinguish features and target/groundtruth
	features = df.iloc[:, [0, 1, 2, 3]]
	groundtruth = df.iloc[:, [4]]

	# scale features
	sc = StandardScaler()
	sc.fit(features)
	features_std = pd.DataFrame(sc.transform(features))

	# check features skewness
	print("\nFeatures Skewness:")
	print(features.skew())

	# fit model
	model = XGBClassifier()
	model.fit(features_std, groundtruth)

	# check feature importances
	importance = list(zip(features.columns.values, model.feature_importances_))
	print("\nFeatures Importance:")
	print(sorted(importance, key = lambda x: x[1]))

	end_time = time.time()

	# execution run time
	runtime = round(end_time - start_time, 6)
	print("\nRuntime: ", runtime, "seconds")
