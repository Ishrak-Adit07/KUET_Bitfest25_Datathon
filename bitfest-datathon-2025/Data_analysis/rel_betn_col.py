# Analyzing the impact of skills_required on matched_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the train and test datasets
new_train_data = pd.read_csv("train.csv")
new_test_data = pd.read_csv("test.csv")
plt.figure(figsize=(10, 6))
sns.boxplot(x='skills_required', y='matched_score', data=new_train_data)
plt.title('Skills Required vs Matched Score')
plt.xticks(rotation=45)
plt.show()
