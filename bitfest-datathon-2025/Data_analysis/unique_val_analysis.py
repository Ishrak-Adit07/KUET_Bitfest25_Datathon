# Display unique values in each column
import pandas as pd

# Load the train and test datasets
new_train_data = pd.read_csv("train.csv")
new_test_data = pd.read_csv("test.csv")
for col in new_train_data.columns:
    print(f"Column: {col}")
    print(f"Unique Values: {new_train_data[col].nunique()}")
    print("-" * 50)
