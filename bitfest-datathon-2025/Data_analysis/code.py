import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import pandas as pd

# Load the train and test datasets
new_train_data = pd.read_csv("train.csv")
new_test_data = pd.read_csv("test.csv")
# Split the data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the Neural Network model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.2),  # Dropout for regularization
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # Adjust epochs as needed
    batch_size=32,
    verbose=1
)

# Predict on validation data
y_pred_val = model.predict(X_val).flatten()

# Calculate Mean Squared Error on validation set
mse_val = mean_squared_error(y_val, y_pred_val)
print(f"Validation MSE: {mse_val}")
# Ensure test data columns match training data
X_test = new_test_data.drop(columns=['address'])  # Replace 'address' with irrelevant columns
X_test = X_test[X_train.columns]  # Ensure the same feature columns as the training set

# Predict on test data
test_predictions = model.predict(X_test).flatten()
# Prepare submission dataframe
submission = pd.DataFrame({
    "ID": new_test_data["ID"],  # Replace "ID" with the unique identifier column in your test data
    "Predicted_Score": test_predictions
})

# Save to CSV
submission.to_csv("submission_nn.csv", index=False)
print("Submission file 'submission_nn.csv' created successfully!")
import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
