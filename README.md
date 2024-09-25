# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the employee dataset and preprocess categorical variables using label encoding.
2.Split the preprocessed data into features (X) and target variable (y), and then divide it into training and testing sets.
3.Initialize and train a Decision Tree Classifier on the training data.
4.Make predictions on the test set using the trained model.
5.Calculate and print the accuracy of the model.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MOPURI ANKITHA
RegisterNumber: 212223040117
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Employee.csv')

# Fix column name with extra space
data.rename(columns={'Departments ': 'Departments'}, inplace=True)

# Preprocess categorical data (Departments and salary)
label_encoder_dept = LabelEncoder()
label_encoder_salary = LabelEncoder()

data['Departments'] = label_encoder_dept.fit_transform(data['Departments'])
data['salary'] = label_encoder_salary.fit_transform(data['salary'])

# Display first 5 rows of the preprocessed dataset
first_five_rows = data.head()

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 3))  # Adjust the figure size for higher quality

# Hide axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create the table
table = ax.table(cellText=first_five_rows.values, colLabels=first_five_rows.columns, cellLoc='center', loc='center')

# Set font size and scaling
table.set_fontsize(14)
table.scale(1.5, 1.5)  # Adjust table size for better readability

# Save the table as a high-resolution image (increase dpi for better quality)
plt.savefig('preprocessed_data_high_quality.png', bbox_inches='tight', dpi=600)  # Increase dpi

# Show the table
plt.show()

# Features and target variable
X = data.drop(columns=['left'])  # Features
y = data['left']  # Target

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy of the model
print(f"\nDecision Tree Classifier Accuracy: {accuracy * 100:.2f}%")
```

## Output:
![image](https://github.com/user-attachments/assets/502080fe-3f95-40f9-9c0b-46a7557d423e)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
