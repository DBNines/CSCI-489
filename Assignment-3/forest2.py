import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier

# ── 1. LOAD & SAMPLE ──────────────────────────────────────────────────────────
df = pd.read_csv('application_train.csv')
df = df.sample(n=30757, random_state=9).reset_index(drop=True)

# ── 2. PROTECTED FEATURE ──────────────────────────────────────────────────────
gender_series = df['CODE_GENDER'].copy()

# ── 3. SEPARATE TARGET ────────────────────────────────────────────────────────
target = df['TARGET']
features = df.drop(columns=['TARGET', 'SK_ID_CURR'])

# ── 4. PRE-PROCESSING ─────────────────────────────────────────────────────────
cat_cols = features.select_dtypes(include='str').columns.tolist()
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
features[cat_cols] = oe.fit_transform(features[cat_cols])

(input_train, input_test,
 result_train, result_test,
 gender_train, gender_test) = train_test_split(
    features, target, gender_series,
    test_size=0.25, random_state=9, stratify=target
)

scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test  = scaler.transform(input_test)

imputer = SimpleImputer(strategy='mean')
input_train = imputer.fit_transform(input_train)
input_test  = imputer.transform(input_test)

# ── 5. HELPER: print full metrics ─────────────────────────────────────────────
def print_metrics(label, y_true, y_pred):
    acc   = accuracy_score(y_true, y_pred)
    acc_n = accuracy_score(y_true, y_pred, normalize=False)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc*100:.2f}%  ({acc_n} / {len(y_true)})")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1 Score : {f1*100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")

# ── 6. THRESHOLD HELPER ───────────────────────────────────────────────────────
# Instead of predict() which uses 0.5 cutoff, we use predict_proba() and
# lower the threshold so the model flags more potential defaulters
THRESHOLD_BALANCED   = 0.35  # balanced_subsample is aggressive, needs higher cutoff
THRESHOLD_UNBALANCED = 0.15  # unbalanced forest needs lower cutoff to catch defaulters

def predict_with_threshold(model, X, threshold):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)

# ── 7. BALANCED RANDOM FOREST ─────────────────────────────────────────────────
print(f"\n##### Balanced Decision Forest (class_weight='balanced_subsample', threshold={THRESHOLD_BALANCED})")
decForest = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced_subsample',  # applies balanced weight per tree
    max_depth=10,                        # prevents all-zero predictions
    random_state=9,
    n_jobs=-1
)
decForest.fit(input_train, result_train)

pred_balanced = predict_with_threshold(decForest, input_test, THRESHOLD_BALANCED)
print_metrics("Balanced Decision Forest — Full Test Set", result_test, pred_balanced)

# ── 8. UNBALANCED RANDOM FOREST ───────────────────────────────────────────────
print(f"\n##### Unbalanced Decision Forest (no class weight, threshold={THRESHOLD_UNBALANCED})")
decForestUnbalanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=9,
    n_jobs=-1
)
decForestUnbalanced.fit(input_train, result_train)

pred_unbalanced = predict_with_threshold(decForestUnbalanced, input_test, THRESHOLD_UNBALANCED)
print_metrics("Unbalanced Decision Forest — Full Test Set", result_test, pred_unbalanced)

# ── 9. BIAS ANALYSIS BY GENDER ────────────────────────────────────────────────
gender_test  = gender_test.reset_index(drop=True)
result_test  = result_test.reset_index(drop=True)
pred_bal_s   = pd.Series(pred_balanced)
pred_unbal_s = pd.Series(pred_unbalanced)

groups = {
    'Female': gender_test == 'F',
    'Male':   gender_test == 'M',
}

print("\n##### Gender-Stratified Accuracies")
acc_results = {'Balanced': {}, 'Unbalanced': {}}

for group_name, mask in groups.items():
    y_true = result_test[mask]
    for model_label, preds in [('Balanced', pred_bal_s), ('Unbalanced', pred_unbal_s)]:
        y_pred = preds[mask]
        acc = accuracy_score(y_true, y_pred)
        acc_results[model_label][group_name] = acc
        print(f"  {model_label:12s} | {group_name:6s}: {acc*100:.2f}%  (n={mask.sum()})")

# ── 10. BAR CHARTS ────────────────────────────────────────────────────────────
genders = ['Female', 'Male']
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
fig.suptitle('Decision Forest — Prediction Accuracy by Gender (Bias Analysis)',
             fontsize=14, fontweight='bold')

colors = {'Female': '#E07B9A', 'Male': '#5B8DB8'}

for ax, (model_label, accs) in zip(axes, acc_results.items()):
    vals = [accs[g] * 100 for g in genders]
    bars = ax.bar(genders, vals,
                  color=[colors[g] for g in genders],
                  width=0.5, edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title(f'{model_label} Decision Forest', fontsize=12)
    ax.set_xlabel('Gender Group', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('decision_forest_gender_bias.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nBar chart saved as 'decision_forest_gender_bias.png'")