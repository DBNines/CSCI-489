import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

# ── 1. LOAD & SAMPLE ──────────────────────────────────────────────────────────
df = pd.read_csv('application_train.csv')
df = df.sample(n=30757, random_state=9).reset_index(drop=True)

# ── 2. PROTECTED FEATURE ──────────────────────────────────────────────────────
gender_series = df['CODE_GENDER'].copy()

# ── 3. SEPARATE TARGET ────────────────────────────────────────────────────────
target   = df['TARGET']
features = df.drop(columns=['TARGET', 'SK_ID_CURR'])

# ── 4. PRE-PROCESSING ─────────────────────────────────────────────────────────
# 4a. Ordinal-encode categorical columns
cat_cols = features.select_dtypes(include='object').columns.tolist()
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
features[cat_cols] = oe.fit_transform(features[cat_cols])

# 4b. Train / test split (75 / 25)
(input_train, input_test,
 result_train, result_test,
 gender_train, gender_test) = train_test_split(
    features, target, gender_series,
    test_size=0.25, random_state=9, stratify=target
)

# 4c. Standard-scale (fit on train only)
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test  = scaler.transform(input_test)

# 4d. Impute missing values with mean (fit on train only)
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

# ── 6. UNBALANCED MLP ─────────────────────────────────────────────────────────
print("\n##### Unbalanced Neural Network (no sample weights)")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=9)
mlp.fit(input_train, result_train)

pred_unbalanced = mlp.predict(input_test)
print_metrics("Unbalanced MLP — Full Test Set", result_test, pred_unbalanced)

# Loss curve — unbalanced
fig_loss, ax_loss = plt.subplots(figsize=(9, 5))
ax_loss.plot(mlp.loss_curve_, label='Unbalanced', color='steelblue')

# ── 7. BALANCED MLP (sample weights) ─────────────────────────────────────────
print("\n##### Balanced Neural Network (sample_weight='balanced')")
sample_weights = compute_sample_weight(class_weight='balanced', y=result_train)
mlp.fit(input_train, result_train, sample_weight=sample_weights)

pred_balanced = mlp.predict(input_test)
print_metrics("Balanced MLP — Full Test Set", result_test, pred_balanced)

# Add balanced loss curve and save
ax_loss.plot(mlp.loss_curve_, label='Balanced', color='darkorange')
ax_loss.set_title('Neural Network Training Loss', fontsize=13, fontweight='bold')
ax_loss.set_xlabel('Epochs (Training Runs)', fontsize=11)
ax_loss.set_ylabel('Loss (Error Rate)', fontsize=11)
ax_loss.legend(fontsize=11)
ax_loss.grid(linestyle='--', alpha=0.5)
fig_loss.tight_layout()
fig_loss.savefig('neural_net_loss_curves.png', dpi=150, bbox_inches='tight')
print("\nLoss curve saved as 'neural_net_loss_curves.png'")

# ── 8. BIAS ANALYSIS BY GENDER ────────────────────────────────────────────────
gender_test  = gender_test.reset_index(drop=True)
result_test  = result_test.reset_index(drop=True)
pred_bal_s   = pd.Series(pred_balanced)
pred_unbal_s = pd.Series(pred_unbalanced)

groups = {
    'Female': gender_test == 'F',
    'Male':   gender_test == 'M',
}

print("\n##### Gender-Stratified Accuracies")
acc_results = {'Unbalanced': {}, 'Balanced': {}}

for group_name, mask in groups.items():
    y_true = result_test[mask]
    for model_label, preds in [('Unbalanced', pred_unbal_s), ('Balanced', pred_bal_s)]:
        y_pred = preds[mask]
        acc = accuracy_score(y_true, y_pred)
        acc_results[model_label][group_name] = acc
        print(f"  {model_label:12s} | {group_name:6s}: {acc*100:.2f}%  (n={mask.sum()})")

# ── 9. BAR CHARTS ─────────────────────────────────────────────────────────────
genders = ['Female', 'Male']
colors  = {'Female': '#E07B9A', 'Male': '#5B8DB8'}

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
fig.suptitle('Neural Network — Prediction Accuracy by Gender (Bias Analysis)',
             fontsize=14, fontweight='bold')

for ax, (model_label, accs) in zip(axes, acc_results.items()):
    vals = [accs[g] * 100 for g in genders]
    bars = ax.bar(genders, vals,
                  color=[colors[g] for g in genders],
                  width=0.5, edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title(f'{model_label} Neural Network', fontsize=12)
    ax.set_xlabel('Gender Group', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    gap = abs(vals[0] - vals[1])
    '''ax.annotate(f'Gap: {gap:.2f}%',
                xy=(0.5, 0.93), xycoords='axes fraction',
                ha='center', fontsize=10, fontweight='bold',
                color='red' if gap > 2 else 'green')'''

plt.tight_layout()
plt.savefig('neural_net_gender_bias.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nBar chart saved as 'neural_net_gender_bias.png'")