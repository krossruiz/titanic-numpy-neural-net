import math
import pandas as pd
import csv
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib

training_data_path = './train.csv'
test_data_path = './test.csv'

raw_training_data = pd.read_csv(training_data_path);
raw_test_data = pd.read_csv(test_data_path);

def formatTitanicData(raw_data):

	encoder = LabelBinarizer();
	minMaxScaler = MinMaxScaler();

	#Initial Columns
	#['PassengerId', 'Survived', 'Pclass', 'Sex', 'Name', 'Ticket', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
	formatted_training_data = raw_data;
	formatted_training_data = formatted_training_data.drop(columns=["Name","Ticket"]);

	formatted_training_data["Age"] = formatted_training_data.loc[:,"Age"].replace(np.nan, 0, regex=True);
	age_normed = minMaxScaler.fit_transform(formatted_training_data["Age"].values.reshape(-1,1));
	formatted_training_data["Age"] = pd.DataFrame(age_normed);

	dummySibSp = pd.get_dummies(formatted_training_data["SibSp"]);
	for x in dummySibSp:
		name = "SibSp" + str(x);
		formatted_training_data[name] = dummySibSp[x];
	formatted_training_data = formatted_training_data.drop(columns=["SibSp"]);

	formatted_training_data["Sex"] = encoder.fit_transform(formatted_training_data.loc[:,"Sex"]);
	# print(training_data["Sex"]);
	#Binarized sex, where
	#0: female
	#1: male

	parchDummy = pd.get_dummies(formatted_training_data["Parch"]);
	for x in parchDummy:
		name = "Parch" + str(x);
		formatted_training_data[name] = parchDummy[x];
	formatted_training_data = formatted_training_data.drop(columns=["Parch"]);

	dummyPclass = pd.get_dummies(formatted_training_data["Pclass"]);
	for x in dummyPclass:
		name = "Pclass" + str(x);
		formatted_training_data[name] = dummyPclass[x];
	formatted_training_data = formatted_training_data.drop(columns=["Pclass"]);
	#One-hot-encoded Pclass into three new columns:
	#PclassOne
	#PclassTwo
	#PclassThree
	# print(training_data.loc[:,"Age"]);
	formatted_training_data["Fare"] = formatted_training_data.loc[:,"Fare"].replace(np.nan, 0, regex=True);
	normFare = minMaxScaler.fit_transform(formatted_training_data["Fare"].values.reshape(-1,1));
	formatted_training_data["Fare"] = pd.DataFrame(normFare);
	#Fare normalized

	formatted_training_data = formatted_training_data.drop(columns=["Cabin"]);

	embarkedDummy = pd.get_dummies(formatted_training_data["Embarked"]);
	for x in embarkedDummy:
		name = "Embarked" + str(x);
		formatted_training_data[name] = embarkedDummy[x];
	formatted_training_data = formatted_training_data.drop(columns=["Embarked"]);

	formatted_training_data = formatted_training_data.drop(columns=["PassengerId"]);
	return formatted_training_data;

formatted_training_data = formatTitanicData(raw_training_data);
formatted_test_data = formatTitanicData(raw_test_data);

# formatted_training_data.to_csv("cleaned_titanic.csv", sep='\t');
#
# training_data_transpose = training_data.transpose();
# training_data_rows, training_data_columns = training_data_transpose.shape;

# test_ratio = 0.2;
# shuffled_indices = np.random.permutation(training_data_columns);
# test_set_size = int(training_data_columns * test_ratio);
# training_set = training_data.iloc[shuffled_indices[:test_set_size]];
# test_set = training_data.iloc[shuffled_indices[test_set_size:]];

def runNetwork(inputSet):

	inputSet = inputSet.transpose();
	inputSetRows, inputSetColumns = inputSet.shape;
	first_training_instance = inputSet.iloc[:,0].values.reshape(-1,1);

	input_size = inputSetRows;
	n1_size = 3;
	n2_size = 3;
	output_size = 1;

	w1_size = input_size, n1_size;
	w2_size = n1_size, n2_size;
	w3_size = n2_size, output_size;

	# w1 = np.random.rand(w1_size[0], w1_size[1]);
	# w2 = np.random.rand(w2_size[0], w2_size[1]);
	# w3 = np.random.rand(w3_size[0], w3_size[1]);
	#
	# n1_bias = np.random.rand(1, n2_size);
	# n2_bias = np.random.rand(1, n2_size);

	w1 = np.ones((w1_size[0], w1_size[1]));
	w2 = np.ones((w2_size[0], w2_size[1]));
	w3 = np.ones((w3_size[0], w3_size[1]));

	n1_bias = np.ones((1, n2_size));
	n2_bias = np.ones((1, n2_size));

	n1_pre_bias = np.dot(first_training_instance.transpose(), w1);
	n1_post_bias = np.add(n1_pre_bias, n1_bias);
	n1_activated = n1_post_bias;
	n1_activated[n1_activated<0] = 0;

	n2_pre_bias = np.dot(n1_activated, w2);
	n2_post_bias = np.add(n2_pre_bias, n2_bias);
	n2_activated = n2_post_bias;
	n2_activated[n2_activated<0] = 0;

	o = np.dot(n2_activated, w3);
	print(o);
	o = 1 / ( 1 + math.exp( -o[0,0] ) );
	print(o);

	return;

print("TRAINING SET ANSWER");
runNetwork(formatted_training_data);
print("TEST SET ANSWER");
runNetwork(formatted_test_data);


#25-----3------3------1
#w1 shape = 25 * 3 = 75
#w2 shape = 3 * 3 = 9
#w3 shape = 3 * 1 = 3

#----b-------b-----------
#i--w1--n1--w2--n2--w3--o
#i--w1--n1--w2--n2--w3---
#i--w1--n1--w2--n2--w3---
#.--w1------w2------w3---
#.--w1------w2------w3---
#.--w1------w2------w3---
#i--w1------w2------w3---
