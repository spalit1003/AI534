import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, pairwise_distances
import time

class CustomKNN:
    def __init__(self, k=3, distance_measure='euclidean'):
        self.k = k
        self.distance_measure = distance_measure

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        if self.distance_measure == 'euclidean':
            distances = np.linalg.norm(self.X_train - x, axis=1)  # Euclidean Distance
        elif self.distance_measure == 'manhattan':
            distances = np.linalg.norm(self.X_train - x, ord=1, axis=1)  # Manhattan Distance
        else:
            raise ValueError("Invalid distance measure")
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
    best_k = 1
    best_dev_error = float('inf')
    best_distance_measure = None

    distance_measures = ['euclidean', 'manhattan']

    for distance_measure in distance_measures:  # Loop through distance measures
        for k in range(1, 101, 2):
            start_time = time.process_time()
            custom_knn = CustomKNN(k=k, distance_measure=distance_measure)
            custom_knn.fit(processed_data, y_train)
        
            # Calculate dev error
            dev_predictions = custom_knn.predict(processed_data_dev)
            dev_error_rate = 100 * (1 - accuracy_score(y_dev, dev_predictions))
            num_positive_dev = np.sum(dev_predictions)
            num_total_dev = len(dev_predictions)
            dev_positive_rate = (num_positive_dev / num_total_dev) * 100
        
            end_time = time.process_time()  
            elapsed_time = end_time - start_time  
        
            # Update the best k if a lower dev error is found
            if dev_error_rate < best_dev_error:
                best_dev_error = dev_error_rate
                best_k = k
                best_distance_measure = distance_measure  # Set the best distance measure

        print(f"Best k with {distance_measure} distance:", best_k, "Best Dev Error:", round(best_dev_error, 2), "Positive Ratio:", round(dev_positive_rate, 2))

    # Print the overall best distance measure
    print("Best Distance Measure:", best_distance_measure)

    # Train the model on the full training dataset with the best k
    custom_knn = CustomKNN(k=best_k)
    custom_knn.fit(np.vstack((processed_data, processed_data_dev)), np.hstack((y_train, y_dev)))

    # Run the trained model on the semi-blind test data
    dataTest = pd.read_csv('income.test.blind', names=headers)
    processed_data_test = preprocessor.transform(dataTest)
    test_predictions = custom_knn.predict(processed_data_test)

    # Create the "income.test.predicted" file with the correct format
    test_data = dataTest.copy()
    test_data['target'] = [' >50K' if pred == 1 else ' <=50K' for pred in test_predictions]
    test_data['hours'] = ' ' + test_data['hours'].astype(str)
    test_data.to_csv('income.test.predicted', index=False, header=False)
