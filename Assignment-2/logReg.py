import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score

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
logReg = LogisticRegression(max_iter=1000, class_weight='balanced')
logReg.fit(input_train,result_train)

#Train unbalanced model
logRegUnbalanced = LogisticRegression(max_iter=1000)
logRegUnbalanced.fit(input_train,result_train)

#Test model, Accuracy
result_prediction = logReg.predict(input_test)
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

#Unbalanced model
print("##### Unbalanced Model!")
result_prediction = logRegUnbalanced.predict(input_test)
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