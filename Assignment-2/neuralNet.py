import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight

# Data Processing and Splitting
df = pd.read_csv('bank-full.csv', sep=';')
df = pd.get_dummies(df, drop_first=True)
inputData = df.drop('y_yes', axis = 1)
resultData = df['y_yes']

input_train, input_test, result_train, result_test = train_test_split(inputData, resultData, test_size=0.2, random_state=9)

scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test = scaler.transform(input_test)

#Train model
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=9)
mlp.fit(input_train, result_train)

#Test model, Accuracy
result_prediction = mlp.predict(input_test)
accuracy = accuracy_score(result_test, result_prediction)
accuracy_count = accuracy_score(result_test, result_prediction, normalize=False)
print(f"Accuracy: {accuracy*100 :.2f}%")
print(f"Accuracy Count: {accuracy_count :.0f} out of {len(input_test)}")
#Confusion Matrix
cm = confusion_matrix(result_test, result_prediction)
print(cm)
#Precision, Recall, and F!
precision = precision_score(result_test, result_prediction)
print(f"Precision: {precision*100 :.2f}%")
recall = recall_score(result_test, result_prediction)
print(f"Recall: {recall*100 :.2f}%")
f1 = f1_score(result_test, result_prediction)
print(f"F1: {f1*100 :.2f}%")

print("############### BALANCED")
#Loss curve
plt.plot(mlp.loss_curve_, label="Unbalanced")
plt.title("Neural Network Training Loss")
plt.xlabel("Epochs (Training Runs)")
plt.ylabel("Loss (Error Rate)")
plt.savefig('neuralNetLoss.png', dpi=300)

#balanced model ########
sample_weights = compute_sample_weight(class_weight='balanced', y=result_train)
mlp.fit(input_train, result_train, sample_weight=sample_weights)

#Test model, Accuracy
result_prediction = mlp.predict(input_test)
accuracy = accuracy_score(result_test, result_prediction)
accuracy_count = accuracy_score(result_test, result_prediction, normalize=False)
print(f"Accuracy: {accuracy*100 :.2f}%")
print(f"Accuracy Count: {accuracy_count :.0f} out of {len(input_test)}")
#Confusion Matrix
cm = confusion_matrix(result_test, result_prediction)
print(cm)
#Precision, Recall, and F!
precision = precision_score(result_test, result_prediction)
print(f"Precision: {precision*100 :.2f}%")
recall = recall_score(result_test, result_prediction)
print(f"Recall: {recall*100 :.2f}%")
f1 = f1_score(result_test, result_prediction)
print(f"F1: {f1*100 :.2f}%")

#Loss curve
plt.plot(mlp.loss_curve_, label="Balanced")
plt.title("Neural Network Training Loss")
plt.xlabel("Epochs (Training Runs)")
plt.ylabel("Loss (Error Rate)")
plt.legend()
plt.savefig('neuralNetLossBalanced.png', dpi=300)