import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

# -- CHECK FOR NULL VALUES
# Load your dataset
df = pd.read_csv('/content/Student_academic.csv')

# Check for null values
null_values = df.isnull().sum()

# Display columns with null values and the count of null values in each column
print("Columns with Null Values:")
for col, count in null_values.items():
    if count > 0:
        print(f"{col}: {count} null values")

# Alternatively, you can check if there are any null values in the entire dataset
if df.isnull().values.any():
    print("There are null values in the dataset.")
else:
    print("No null values found in the dataset.")

# -- result: 
# Columns with Null Values:
# No null values found in the dataset.

# -- DISTRIBUTION OF STUDENTS BY TARGET AND INTERNATIONAL STATUS
# Load your dataset
df = pd.read_csv('/content/Student_academic.csv')

# Create a bar chart to examine the distribution
plt.figure(figsize=(8, 6))
international_students = df[df['International'] == 1]
domestic_students = df[df['International'] == 0]

target_labels = ['Graduate', 'Dropout', 'Enrolled']
target_counts_international = international_students['Target'].value_counts().reindex(target_labels, fill_value=0)
target_counts_domestic = domestic_students['Target'].value_counts().reindex(target_labels, fill_value=0)

bar_width = 0.35
x = range(len(target_labels))
plt.bar(x, target_counts_international, width=bar_width, label='International Students', color='skyblue')
plt.bar([i + bar_width for i in x], target_counts_domestic, width=bar_width, label='Domestic Students', color='lightcoral')

plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Distribution of Students by Target and International Status')
plt.xticks([i + bar_width / 2 for i in x], target_labels)
plt.legend()
plt.show()

# -- DISTRIBUTION OF STUDENTS BY GENDER AND TARGET STATUS
# Load your dataset
df = pd.read_csv('/content/Student_academic.csv')

# Map gender values to 'Male' and 'Female'
df['Gender'] = df['Gender'].map({0: 'Male', 1: 'Female'})

# Create a bar chart to examine the distribution
plt.figure(figsize=(8, 6))

# Group the data by 'Gender' and 'Target' and calculate the counts
gender_target_counts = df.groupby(['Gender', 'Target']).size().unstack().fillna(0)

# Plot the bar chart
gender_target_counts.plot(kind='bar', stacked=True, color=['lightcoral', 'skyblue', 'lightgreen'])
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Students by Gender and Target Status')
plt.xticks(rotation=0)
plt.legend(title='Target', labels=gender_target_counts.columns)
plt.show()

# -- CORRELATION HEATMAP FOR ALL COLUMNS
# Load your dataset
df = pd.read_csv('/content/Student_academic.csv')

# Replace values in the 'Target' column
df['Target'] = df['Target'].replace({'Dropout': 0, 'Enrolled': 1, 'Graduate': 2})

# Calculate the correlation matrix for all columns
correlation_matrix = df.corr(numeric_only=True)

# Create a heatmap to visualize the correlations
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for All Columns')
plt.show()

# Sort correlation values for colomn 'Target'
correlation_matrix['Target'].sort_values()

# -- OUTLIERS DETECTION AND HANDLING
# Load your dataset
df = pd.read_csv('/content/Student_academic.csv')

# Step 1: Visual Inspection (Box Plot) and Outlier Handling
columns_to_check = ['Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)']

for col in columns_to_check:
    # Visual Inspection (Box Plot) Before Outlier Handling
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot for {col} (Before Handling)')
    
    # IQR Method for Outlier Detection
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Imputation - Replace Outliers with Mean Value
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    mean_value = df[col].mean()
    
    # Create a copy of the column for modification
    df_copy = df.copy()
    df_copy[col].loc[outliers.index] = mean_value
    
    # Update the original DataFrame with the modified column
    df[col] = df_copy[col]
    
    # Visual Inspection (Box Plot) After Outlier Handling
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot for {col} (After Handling)')
    
    plt.tight_layout()
    plt.show()


# -- LOGISTIC REGRESSION

# Define the columns you want to use for X and y
X_columns = ['Tuition fees up to date', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (approved)']
y_column = 'Target'

# Assign data values
X = df[X_columns]
y = df[y_column]

# Split the data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression model
model = LogisticRegression(random_state=0)

# Fit the model to your training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# For precision, recall, and F1-score in multiclass classification, specify the average parameter
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["True Negative", "True Positive"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# -- DECISION TREE

# Data already assigned and split
# Create and fit the Decision Tree model
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# For precision, recall, and F1-score in multiclass classification, specify the average parameter
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["True Negative", "True Positive"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# -- RANDOM FOREST CLASSIFIER

# Data already assigned and split
# Create and fit the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=10, criterion='entropy')
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# For precision, recall, and F1-score in multiclass classification, specify the average parameter
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Define class labels
class_labels = ['Dropout', 'Enrolled', 'Graduate']

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# -- NAIVE BAYES CLASSIFIER (Gaussian)

# Data already assigned and split
# Create and fit the Naive Bayes Classifier model, Gaussian type in this case
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# For precision, recall, and F1-score in multiclass classification, specify the average parameter
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Define class labels
class_labels = ['Dropout', 'Enrolled', 'Graduate']

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# -- SUPPORT VECTOR MACHINE

# Data already assigned and split
# Create and fit the SVM model
model = SVC(kernel='poly',probability=True, random_state=0)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# For precision, recall, and F1-score in multiclass classification, specify the average parameter
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Define class labels
class_labels = ['Dropout', 'Enrolled', 'Graduate']

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()