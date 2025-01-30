import pandas as pd

# Load the train and test datasets
new_train_data = pd.read_csv("train.csv")
new_test_data = pd.read_csv("test.csv")
# Calculate missing value percentages
train_missing = new_train_data.isnull().mean() * 100
test_missing = new_test_data.isnull().mean() * 100

print("Train Dataset Missing Values (%):\n", train_missing[train_missing > 0].sort_values(ascending=False))
print("\nTest Dataset Missing Values (%):\n", test_missing[test_missing > 0].sort_values(ascending=False))
