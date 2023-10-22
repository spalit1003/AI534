import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import time

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1) # Euclidean Distance
        #distances = np.linalg.norm(self.X_train - x, ord=1, axis=1) # Manhattan Distance
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels)
        return np.argmax(most_common)

if __name__ == "__main__":
    headers = ["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country", "target"]
    data = pd.read_csv('income.train.txt.5k', names=headers)
    target_data = data["target"]
    data = data.drop(columns=['target'])

    dataDev = pd.read_csv('income.dev.txt', names=headers)
    target_data_dev = dataDev["target"]
    dataDev = dataDev.drop(columns=['target'])

    # Convert labels to integers
    y_train = (target_data == ' >50K').astype(int)
    y_dev = (target_data_dev == ' >50K').astype(int)

    # Get the preprocessed training and test data
    num_processor = MinMaxScaler(feature_range=(0, 2))
    cat_processor = OneHotEncoder(sparse=False, handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        [('num', num_processor, ['age', 'hours']),
         ('cat', cat_processor, ['sector', 'edu', 'marriage', 'occupation', 'race', 'sex', 'country'])]
    )
    preprocessor.fit(data)
    processed_data = preprocessor.transform(data)
    processed_data_dev = preprocessor.transform(dataDev)

    # Test for different values of k
    for k in range(1, 101, 2):
        start_time = time.process_time()
        custom_knn = CustomKNN(k=k)
        custom_knn.fit(processed_data, y_train)
        
        # Calculate train error
        train_predictions = custom_knn.predict(processed_data)
        train_error_rate = 100 * (1 - accuracy_score(y_train, train_predictions))
        cm_train = confusion_matrix(y_train, train_predictions)
        train_positive_rate = cm_train[1, 1] / (cm_train[1, 0] + cm_train[1, 1]) * 100

        # Calculate dev error
        dev_predictions = custom_knn.predict(processed_data_dev)
        dev_error_rate = 100 * (1 - accuracy_score(y_dev, dev_predictions))
        cm_dev = confusion_matrix(y_dev, dev_predictions)
        dev_positive_rate = cm_dev[1, 1] / (cm_dev[1, 0] + cm_dev[1, 1]) * 100
        
        end_time = time.process_time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        
        print(f"k={k}  train_err {round(train_error_rate, 2)}% (+: {round(train_positive_rate, 2)}%)  dev_err {round(dev_error_rate, 2)}% (+: {round(dev_positive_rate, 2)}%) time {round(elapsed_time,2)}")
