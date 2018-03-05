import numpy as np
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

dataset = pd.read_csv("cleaned_titanic.csv", sep="\t");
testSet = pd.read_csv("cleaned_titanic_test.csv", sep="\t")

dataset = dataset.iloc[:,1:]

testPassengerIds = testSet.loc[:,"PassengerId"]
testSet = testSet.iloc[:,2:]
testSet = testSet.drop(['Parch9'], axis=1)

survived = dataset.iloc[:,0]
features = dataset.iloc[:,1:]
survived = survived.values
features = features.values
survived = survived.reshape(-1,1)
survived = to_categorical(survived)
features = features.reshape(-1,23)

testFeatures = testSet.values.reshape(-1,23)

model = Sequential()

model.add(
	Dense(10, activation="relu", input_dim=23)
)
model.add(
	Dense(2, activation="softmax")
)

model.compile(
	loss='categorical_crossentropy',
	optimizer='sgd',
	metrics=['accuracy']
)

X = features
Y = survived

model.fit(X, Y, epochs=3000)
predictions = model.predict(testFeatures)
formatted_predictions = [] 
xPredictionShape, yPredictionShape = predictions.shape
for x in range(xPredictionShape):
	if(predictions[x,0] > 0.5):
		formatted_predictions.append(0)
	else:
		formatted_predictions.append(1)
formatted_predictions = pd.DataFrame(formatted_predictions, columns=['Survived'])
formatted_predictions.insert(loc=0, column='PassengerId', value=testPassengerIds.values)
formatted_predictions.to_csv("formatted_predictions.csv", index=False)
