import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32), 
    #hidden_layer_sizes=(256, 128, 64), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=9,
    early_stopping=True,
    alpha=0.001,               # L2 regularization to prevent overfitting
    batch_size=64,             # Smaller batches can help find better local minima
    validation_fraction=0.1,
    learning_rate_init=0.001
)
model.fit(X_train_scaled, y_train)

# 7. Prediction & Thresholding
# Get probabilities for class 1 (>50K)
probs = model.predict_proba(X_test_scaled)[:, 1]

# Manually threshold at 0.35 to help with the class imbalance
binary_predictions = (probs >= 0.35).astype(int)

# 8. Evaluate
print(f"Accuracy: {accuracy_score(y_test, binary_predictions)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, binary_predictions))
print("\nClassification Report:")
print(classification_report(y_test, binary_predictions))

#Precision, Recall, and F!
precision = precision_score(y_test, binary_predictions)
print(f"Precision: {precision*100 :.2f}%")
recall = recall_score(y_test, binary_predictions)
print(f"Recall: {recall*100 :.2f}%")
f1 = f1_score(y_test, binary_predictions)
print(f"F1: {f1*100 :.2f}%")

from sklearn.inspection import permutation_importance

# Calculate Permutation Importance
# This takes a minute because it re-evaluates the model multiple times
result = permutation_importance(model, X_test_scaled, y_test, n_repeats=5, random_state=9, n_jobs=-1)

# Organize into a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': result.importances_mean
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Features (DNN Permutation Importance):")
print(feature_importance_df.head(10))