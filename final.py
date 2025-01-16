import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Create the dataset using the Width data
width_data = [67.3, 70.3, 70.6, 71.4, 68.2, 76.1, 74.0, 68.4, 68.5, 70.9, 72.7, 72.7, 74.7, 73.5, 74.5, 75.0, 75.5,
              70.3, 77.0, 67.9, 69.4, 72.5, 72.7, 74.1, 73.6, 66.7, 73.0, 69.7, 69.2, 74.4, 71.0, 74.4, 74.4, 74.4, 69.1,
              71.0, 74.7, 75.7, 79.3, 78.8, 78.8, 71.5, 71.7, 76.8, 67.0, 73.1, 69.1, 73.0, 66.9, 78.2, 70.2, 76.6, 78.7,
              79.1, 67.1, 70.3, 68.9, 70.4, 75.6, 65.7, 66.9, 71.6, 70.2, 71.6, 66.7, 69.4, 72.3, 70.5, 70.9, 70.9, 72.0,
              76.4, 71.5, 73.6, 78.2, 79.9, 66.5, 68.9, 68.5, 70.3, 72.4, 69.9, 66.7, 69.1, 69.6, 73.0, 78.2, 70.2, 74.9,
              67.7, 70.8, 73.1, 71.3, 67.5, 67.5, 67.8, 73.1, 72.2, 67.3, 69.1, 70.3, 74.9, 71.7, 70.4, 66.5, 69.4, 73.6,
              70.1, 74.4, 67.8, 72.2, 74.4, 71.0, 76.8, 76.3, 68.4, 70.4, 74.5, 72.7, 72.6, 72.7, 70.1, 69.5, 69.5, 70.6,
              67.4, 66.4, 66.4, 66.4, 69.0, 69.0, 67.5, 68.3, 66.7, 70.1, 71.7, 68.3, 66.5, 73.4, 66.7, 66.5, 76.4, 68.3,
              68.3, 68.5, 66.7, 68.3, 67.9, 67.6, 67.6, 69.3, 69.3, 71.5, 72.1]

# Create a DataFrame
df = pd.DataFrame({'Width': width_data})

# Step 2: Create the target variable (Bought)
median_width = df['Width'].median()
df['HighWidth'] = np.where(df['Width'] > median_width, 1, 0)

# Step 3: Split the data into features and target
X = df[['Width']]  # Feature (Width)
y = df['HighWidth']  # Target variable (HighWidth)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
