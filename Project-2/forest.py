import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score, classification_report

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 'income']

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
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

regressor = RandomForestClassifier(n_estimators=200,max_depth=20,min_samples_leaf=2,max_features='sqrt',class_weight={0: 1, 1: 2},random_state=9)

regressor.fit(X_train, y_train)

# 7. Prediction & Thresholding
# Linear Regression predicts a continuous 'score'
predictions = regressor.predict(X_test)

# 8. Evaluate
print(f"Accuracy: {accuracy_score(y_test, predictions)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

import matplotlib.pyplot as plt

# Get feature importances
importances = regressor.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the top 10
print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))