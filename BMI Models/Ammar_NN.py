import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the data
df = pd.read_csv('final_dataset.csv')

# Remove 'Exercise Recommendation Plan' column if it exists
if 'Exercise Recommendation Plan' in df.columns:
    df = df.drop('Exercise Recommendation Plan', axis=1)

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['BMIcase'] = le.fit_transform(df['BMIcase'])

# Prepare the data
X = df.drop(['BMIcase', 'BMI'], axis=1)  # Removing BMI as it's directly related to BMIcase
y = df['BMIcase']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
nn_model = MLPClassifier(max_iter=1000)

# Train the model on the full training set
nn_model.fit(X_train_scaled, y_train)

# Evaluate on the test set
y_pred = nn_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

# Output the result as a percentage
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Accuracy Score Plot
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy'], [test_accuracy * 100], color='lightblue')
plt.ylim(0, 100)
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.show()

# Feature Importance (if applicable)
# This requires the model to be able to provide feature importances.
# MLPClassifier doesn't directly provide this, but you can use permutation importance as an alternative.

from sklearn.inspection import permutation_importance

result = permutation_importance(nn_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], yerr=result.importances_std[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx])
plt.title('Feature Importance')
plt.xlabel('Mean Accuracy Decrease')
plt.show()
