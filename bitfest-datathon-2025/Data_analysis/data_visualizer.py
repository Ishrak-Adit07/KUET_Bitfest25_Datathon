import seaborn as sns
import matplotlib.pyplot as plt

# Visualizing the target variable distribution
import pandas as pd

# Load the train and test datasets
new_train_data = pd.read_csv("train.csv")
new_test_data = pd.read_csv("test.csv")
plt.figure(figsize=(8, 5))
sns.histplot(new_train_data['matched_score'], kde=True, color='blue')
plt.title('Distribution of Matched Score')
plt.xlabel('Matched Score')
plt.ylabel('Frequency')
plt.show()
