import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

def main():
    # 1. Load the dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv('application_train.csv')
    except FileNotFoundError:
        print("Error: Dataset not found. Please ensure 'application_train.csv' is in the directory.")
        return

    # Sample exactly 30,757 rows as requested
    if len(df) > 30757:
        df = df.sample(n=30757, random_state=42)

    # 2. Separate Features (X) and Target (y)
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Extract protected attribute (Gender) BEFORE preprocessing
    gender_test = X_test['CODE_GENDER'].copy()

    # 4. Preprocessing
    print("Preprocessing data...")
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    if len(categorical_cols) > 0:
        X_train_encoded[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded[categorical_cols] = encoder.transform(X_test[categorical_cols])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_encoded)
    X_test_imputed = imputer.transform(X_test_encoded)

    # Normalize with Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # 5. Build and Train Models
    print("Training Decision Forest (Random Forest)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf_model.fit(X_train_scaled, y_train)

    print("Training Neural Network (MLP)...")
    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
    nn_model.fit(X_train_scaled, y_train)

    # 6. Make Predictions
    print("Making predictions on the test set...")
    rf_preds = rf_model.predict(X_test_scaled)
    nn_preds = nn_model.predict(X_test_scaled)

    # 7. Bias Detection: Evaluate metrics on protected sub-groups
    print("\nCalculating metrics by gender...")
    
    mask_female = (gender_test == 'F')
    mask_male = (gender_test == 'M')

    # --- Metrics for Decision Forest ---
    rf_acc_f = accuracy_score(y_test[mask_female], rf_preds[mask_female])
    rf_acc_m = accuracy_score(y_test[mask_male], rf_preds[mask_male])
    
    # zero_division=0 prevents warnings if a model predicts 0 positives for a group
    rf_f1_f = f1_score(y_test[mask_female], rf_preds[mask_female], zero_division=0)
    rf_f1_m = f1_score(y_test[mask_male], rf_preds[mask_male], zero_division=0)
    
    rf_rec_f = recall_score(y_test[mask_female], rf_preds[mask_female], zero_division=0)
    rf_rec_m = recall_score(y_test[mask_male], rf_preds[mask_male], zero_division=0)

    # --- Metrics for Neural Network ---
    nn_acc_f = accuracy_score(y_test[mask_female], nn_preds[mask_female])
    nn_acc_m = accuracy_score(y_test[mask_male], nn_preds[mask_male])
    
    nn_f1_f = f1_score(y_test[mask_female], nn_preds[mask_female], zero_division=0)
    nn_f1_m = f1_score(y_test[mask_male], nn_preds[mask_male], zero_division=0)
    
    nn_rec_f = recall_score(y_test[mask_female], nn_preds[mask_female], zero_division=0)
    nn_rec_m = recall_score(y_test[mask_male], nn_preds[mask_male], zero_division=0)

    # --- Print Results ---
    print("\n" + "="*45)
    print(" "*10 + "EVALUATION METRICS BY GENDER")
    print("="*45)
    
    print("\n--- Decision Forest ---")
    print(f"         Accuracy |    F1    |  Recall")
    print(f"Female :  {rf_acc_f:.4f}  |  {rf_f1_f:.4f}  |  {rf_rec_f:.4f}")
    print(f"Male   :  {rf_acc_m:.4f}  |  {rf_f1_m:.4f}  |  {rf_rec_m:.4f}")

    print("\n--- Neural Network ---")
    print(f"         Accuracy |    F1    |  Recall")
    print(f"Female :  {nn_acc_f:.4f}  |  {nn_f1_f:.4f}  |  {nn_rec_f:.4f}")
    print(f"Male   :  {nn_acc_m:.4f}  |  {nn_f1_m:.4f}  |  {nn_rec_m:.4f}")
    print("="*45 + "\n")

    # 8. Generate Bar Figures
    print("Generating comprehensive bar figures...")
    
    labels = ['Female', 'Male']
    x = np.arange(len(labels))
    width = 0.35

    # Create a figure with 3 subplots side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_data = [
        ('Accuracy', [rf_acc_f, rf_acc_m], [nn_acc_f, nn_acc_m]),
        ('F1 Score', [rf_f1_f, rf_f1_m], [nn_f1_f, nn_f1_m]),
        ('Recall', [rf_rec_f, rf_rec_m], [nn_rec_f, nn_rec_m])
    ]

    for i, (metric_name, rf_scores, nn_scores) in enumerate(metrics_data):
        ax = axes[i]
        rects1 = ax.bar(x - width/2, rf_scores, width, label='Decision Forest', color='skyblue')
        rects2 = ax.bar(x + width/2, nn_scores, width, label='Neural Network', color='salmon')

        ax.set_ylabel(metric_name)
        ax.set_title(f'Model {metric_name} by Gender')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
        # Scale y-axis dynamically based on the max score in that specific subplot to ensure visibility
        max_score = max(max(rf_scores), max(nn_scores))
        ax.set_ylim([0, max_score + (max_score * 0.2) + 0.05])
        
        if i == 0: # Only add legend to the first plot to avoid clutter
            ax.legend()

        ax.bar_label(rects1, padding=3, fmt='%.4f')
        ax.bar_label(rects2, padding=3, fmt='%.4f')

    fig.tight_layout()
    plt.savefig('comprehensive_bias_results.png')
    plt.show()

if __name__ == "__main__":
    main()