import pandas as pd

# Load the train and test datasets
new_train_data = pd.read_csv("train.csv")
new_test_data = pd.read_csv("test.csv")

# Check basic structure of the datasets
print("Train Dataset Overview:\n")
print(new_train_data.info())
print("\nTest Dataset Overview:\n")
print(new_test_data.info())

# Display summary statistics for numerical columns
print("\nTrain Dataset Numerical Summary:\n", new_train_data.describe())
print("\nTest Dataset Numerical Summary:\n", new_test_data.describe())

# Display first few rows of the datasets
print("\nTrain Dataset Sample Rows:\n", new_train_data.head())
print("\nTest Dataset Sample Rows:\n", new_test_data.head())
