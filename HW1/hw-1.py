from multiprocessing import process
from re import T
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, pairwise_distances, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
       

headers = [
    "age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours",
    "country", "target"
]
data = pd.read_csv('income.train.txt.5k', names=headers)
target_data = data["target"]
target_column = pd.DataFrame({'target': target_data})

# Part 1 - Q1
'''earning = data["target"]
count = 0
for i in range(len(earning)):
  if str(earning[i]) == ' >50K':
    count = count + 1
trainingPercent = count / len(earning)
print("% of training data with positive label: ", trainingPercent * 100)'''

dataDev = pd.read_csv('income.dev.txt', names=headers)
target_data_dev = dataDev["target"]
target_column = pd.DataFrame({'target': target_data})
'''earningD = dataDev["target"]
countD = 0
for i in range(len(earningD)):
  if str(earningD[i]) == ' >50K':
    countD = countD + 1
devPercent = countD / len(dataDev)
print("% of training data with positive label in dev: ", devPercent * 100)

#Part 1 - Q2
ages = data["age"]
print("Minimum age: ", min(ages))
print("Maximum age: ", max(ages))

hoursWorked = data["hours"]
print("Least hours worked: ", min(hoursWorked))
print("Most hours worked: ", max(hoursWorked))

# Part 1 - Q5
features = 0
for i in range(len(headers)):
  uniqueVals = data[headers[i]].unique()
  if headers[i] == "age" or headers[i] == "hours":
    numVals = 1
  else:
    numVals = len(uniqueVals)
  features = features + numVals
features = features - 2
print("Unique features: ", features)'''

# Part 2 - Q1
dataT = pd.read_csv('toy.txt', sep=", ", names=["age", "sector"])
encoded_data = pd.get_dummies(dataT, columns=["age", "sector"])
#fit to toy dataset
#encoder.fit(dataT) # you only fit the encoder once (on training)
#binary_data = encoder.transform(dataT)  # but use it to transform training, dev, and test sets
#binarizing full dataset
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
data = data.drop(columns=['target'])
encoder.fit(data)
binary_data = encoder.transform(data)
dataDev=dataDev.drop(columns=['target'])
binary_data_dev = encoder.transform(dataDev)
print("Number of dimensions: ", len(binary_data[0]))

# Part 2 - Q4
y_train = target_data
y_dev = target_data_dev
'''for k in range(1,101,2): #handles odd numbers from 1-100
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(binary_data, y_train)
  pred_train = knn.predict(binary_data)
  pred_dev = knn.predict(binary_data_dev)
  train_err = (1-accuracy_score(y_train, pred_train))*100
  dev_err = (1-accuracy_score(y_dev, pred_dev))*100
  cm_train = confusion_matrix(y_train, pred_train)
  cm_dev = confusion_matrix(y_dev, pred_dev)
  train_positive_rate = cm_train[1, 1] / (cm_train[1, 0] + cm_train[1, 1]) * 100
  dev_positive_rate = cm_dev[1, 1] / (cm_dev[1, 0] + cm_dev[1, 1]) * 100
  print(f"k={k}  train_err {round(train_err, 2)}% (+: {round(train_positive_rate, 2)}%)  dev_err {round(dev_err, 2)}% (+: {round(dev_positive_rate, 2)}%)")'''


# Part 3 - Q1
'''num_processor = 'passthrough' # i.e., no transformation
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')
preprocessor = ColumnTransformer([('num', num_processor, ['age','hours']),('cat', cat_processor, ['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])])
preprocessor.fit(data)
processed_data = preprocessor.transform(data)
processed_data_dev = preprocessor.transform(dataDev)
for k in range(1,101,2):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(processed_data, y_train)
  pred_train = knn.predict(processed_data)
  pred_dev = knn.predict(processed_data_dev)
  train_err = (1 - accuracy_score(y_train, pred_train)) * 100
  dev_err = (1 - accuracy_score(y_dev, pred_dev)) * 100
  cm_train = confusion_matrix(y_train, pred_train)
  cm_dev = confusion_matrix(y_dev, pred_dev)
  train_positive_rate = cm_train[1, 1] / (cm_train[1, 0] + cm_train[1, 1]) * 100
  dev_positive_rate = cm_dev[1, 1] / (cm_dev[1, 0] + cm_dev[1, 1]) * 100
  print(f"k={k}  train_err {round(train_err, 2)}% (+: {round(train_positive_rate, 2)}%)  dev_err {round(dev_err, 2)}% (+: {round(dev_positive_rate, 2)}%)")'''

# Part 3 - Q2
num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')
preprocessor = ColumnTransformer([('num', num_processor, ['age', 'hours']),('cat', cat_processor, ['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])])
preprocessor.fit(data)
processed_data = preprocessor.transform(data)
processed_data_dev = preprocessor.transform(dataDev)
for k in range(1,101,2):
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(processed_data, y_train)
  train_pred = knn.predict(processed_data)
  dev_pred = knn.predict(processed_data_dev)
  train_err = (1 - accuracy_score(y_train, train_pred)) * 100
  dev_err = (1 - accuracy_score(y_dev, dev_pred)) * 100
  cm_train = confusion_matrix(y_train, train_pred)
  cm_dev = confusion_matrix(y_dev, dev_pred)
  train_positive_rate = cm_train[1, 1] / (cm_train[1, 0] + cm_train[1, 1]) * 100
  dev_positive_rate = cm_dev[1, 1] / (cm_dev[1, 0] + cm_dev[1, 1]) * 100
  print(f"k={k}  train_err {round(train_err, 2)}% (+: {round(train_positive_rate, 2)}%)  dev_err {round(dev_err, 2)}% (+: {round(dev_positive_rate, 2)}%)")

# Part 4 - Q1
dev_person = processed_data_dev[0]

# My calculation for distances 
euclidean_distances_custom = np.linalg.norm(processed_data - dev_person, axis=1)
manhattan_distances_custom = np.linalg.norm(processed_data - dev_person, ord =1, axis=1)

# Using KNN
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(processed_data, y_train)
top3_indices = knn.kneighbors([dev_person], n_neighbors=3, return_distance=False)[0]
euclidean_distances= pairwise_distances(processed_data, [dev_person], metric='euclidean')
manhattan_distances= pairwise_distances(processed_data, [dev_person], metric='manhattan')

print("Results using my own implementation:")
print("Euclidean Distances:", euclidean_distances_custom[top3_indices])
print("Manhattan Distances:", manhattan_distances_custom[top3_indices])
print("Results using scikit-learn's KNeighborsClassifier:")
print("Euclidean Distances:",euclidean_distances[euclidean_distances.argsort(axis=0)[:3]].flatten())
print("Manhattan Distances:", manhattan_distances[manhattan_distances.argsort(axis=0)[:3]].flatten())
