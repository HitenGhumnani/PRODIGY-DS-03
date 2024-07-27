import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'data.csv'
data = pd.read_csv(file_path, sep=';')

# Explore the dataset
print("First few rows of the dataset:")
print(data.head(), "\n")

print("Summary statistics of the dataset:")
print(data.describe(), "\n")

print("Information about the dataset:")
print(data.info(), "\n")

print("Checking for missing values in the dataset:")
print(data.isnull().sum(), "\n")

# Encode categorical variables
print("Encoding categorical variables using one-hot encoding:")
data = pd.get_dummies(data, drop_first=True)

# Define feature variables (X) and target variable (y)
X = data.drop('y_yes', axis=1)
y = data['y_yes']

# Split the data into training and testing sets
print("Splitting the data into training and testing sets:")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Decision Tree Classifier with limited depth
print("Training the Decision Tree Classifier with max depth of 3:")
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)

# Make predictions on the test data
print("Making predictions on the test data:")
y_pred = clf.predict(X_test)

# Evaluate the model
print("Evaluating the model:")
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}\n')

conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}\n')

class_report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{class_report}\n')

# Visualize the decision tree
print("Visualizing the decision tree:")
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['no', 'yes'])
plt.show()
