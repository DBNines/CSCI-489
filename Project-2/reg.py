import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score, classification_report

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 'income']

# Data Processing 
df = pd.read_csv('adult.data.csv', names=columns, skipinitialspace=True)
df_test = pd.read_csv('adult.test.csv', names=columns, skipinitialspace=True)

#Make income categorical
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
df_test['income'] = df_test['income'].apply(lambda x: 1 if x.strip('.') == '>50K' else 0)

#Split features
X_train_raw = df.drop(columns=['income', 'fnlwgt', 'education'])
y_train = df['income']

X_test_raw = df_test.drop(columns=['income', 'fnlwgt', 'education'])
y_test = df_test['income']

#Make categorical probs
X_train = pd.get_dummies(X_train_raw)
X_test = pd.get_dummies(X_test_raw)

# Ensure both train and test have the exact same columns after dummy encoding
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7. Prediction & Thresholding
# Linear Regression predicts a continuous 'score'
raw_scores = model.predict(X_test_scaled)

# If score >= 0.5, predict 1 (>50K), else 0 (<=50K)
binary_predictions = (raw_scores >= 0.5).astype(int)

# 8. Evaluate
print(f"Accuracy: {accuracy_score(y_test, binary_predictions)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, binary_predictions))
print("\nClassification Report:")
print(classification_report(y_test, binary_predictions))

# Get coefficients
importances = model.coef_
feature_names = X_train.columns

# Create DF
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Add a column for Absolute Importance for "impact" ranking
feature_importance_df['Abs_Importance'] = feature_importance_df['Importance'].abs()

# Sort by Absolute Importance to see the most influential features overall
feature_importance_df = feature_importance_df.sort_values(by='Abs_Importance', ascending=False)

print("\nTop 10 Most Influential Features (Overall):")
print(feature_importance_df.head(10))
