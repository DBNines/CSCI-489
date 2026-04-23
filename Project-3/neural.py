import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
from sklearn.neural_network import MLPClassifier

# ── 1. LOAD & SAMPLE ──────────────────────────────────────────────────────────
df = pd.read_csv('diabetic_data.csv')
df = df.sample(n=30757, random_state=9).reset_index(drop=True)

# ── 2. BINARIZE TARGET ────────────────────────────────────────────────────────
# 1 = readmitted within 30 days, 0 = all other cases (>30 days or not at all)
df['readmitted'] = (df['readmitted'] == '<30').astype(int)

# ── 3. PROTECTED FEATURE (RACE) ───────────────────────────────────────────────
# Drop rows where race is unknown ('?')
df = df[df['race'] != '?'].reset_index(drop=True)
race_series = df['race'].copy()
RACE_GROUPS = sorted(race_series.unique().tolist())
print(f"Race groups found: {RACE_GROUPS}")

# ── 4. SEPARATE TARGET ────────────────────────────────────────────────────────
target   = df['readmitted']
features = df.drop(columns=['readmitted', 'encounter_id', 'patient_nbr'])

# ── 5. PRE-PROCESSING ─────────────────────────────────────────────────────────
# 5a. Replace '?' with NaN so imputer can handle them
features.replace('?', np.nan, inplace=True)

# 5b. Ordinal-encode all categorical columns
cat_cols = features.select_dtypes(include='str').columns.tolist()
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
features[cat_cols] = oe.fit_transform(features[cat_cols].astype(str))

# 5c. Train / test split (75 / 25) — stratified to preserve class balance
(input_train, input_test,
 result_train, result_test,
 race_train, race_test) = train_test_split(
    features, target, race_series,
    test_size=0.25, random_state=9, stratify=target
)

# 5d. Standard-scale (fit on train only)
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)
input_test  = scaler.transform(input_test)

# 5e. Impute missing values with mean (fit on train only)
imputer = SimpleImputer(strategy='mean')
input_train = imputer.fit_transform(input_train)
input_test  = imputer.transform(input_test)

# ── 6. HELPERS ────────────────────────────────────────────────────────────────
THRESHOLD = 0.15

def predict_with_threshold(model, X):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= THRESHOLD).astype(int)

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

def race_accuracies(y_true, y_pred, race_idx, label):
    """Returns dict of {race: accuracy} across all race groups."""
    results = {}
    race_idx = race_idx.reset_index(drop=True)
    y_true   = y_true.reset_index(drop=True)
    y_pred_s = pd.Series(y_pred)
    for r in RACE_GROUPS:
        mask = race_idx == r
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_true[mask], y_pred_s[mask])
        results[r] = acc
        print(f"  {label:25s} | {r:20s}: {acc*100:.2f}%  (n={mask.sum()})")
    return results

# ── 7. TRAIN ORIGINAL MODEL ───────────────────────────────────────────────────
print(f"\n##### Unbalanced Neural Network (threshold={THRESHOLD})")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=9)
mlp.fit(input_train, result_train)

pred_before = predict_with_threshold(mlp, input_test)
print_metrics("Unbalanced Neural Network — Full Test Set", result_test, pred_before)

# Loss curve
fig_loss, ax_loss = plt.subplots(figsize=(9, 5))
ax_loss.plot(mlp.loss_curve_, label='Before Mitigation', color='steelblue')

# ── 8. INITIAL RACE BIAS ANALYSIS ─────────────────────────────────────────────
print("\n##### Race-Stratified Accuracies (Before Mitigation)")
acc_before = race_accuracies(result_test, pred_before, race_test, "Unbalanced")

# ── 9. BIAS MITIGATION — DIFFERENTIAL SAMPLING ───────────────────────────────
lower_group = min(acc_before, key=acc_before.get)
lower_acc   = acc_before[lower_group]
print(f"\n>>> Lowest accuracy group: {lower_group} ({lower_acc*100:.2f}%)")
print(f">>> Oversampling '{lower_group}' to 40% of training set to mitigate bias...")

train_df = pd.DataFrame(input_train)
train_df['target'] = result_train.values
train_df['race']   = race_train.values

lower_df = train_df[train_df['race'] == lower_group]
other_df = train_df[train_df['race'] != lower_group]

n_total = len(train_df)
n_lower = int(n_total * 0.40)
n_other = n_total - n_lower

lower_resampled = lower_df.sample(n=n_lower, replace=True,  random_state=9)
other_resampled = other_df.sample(n=n_other, replace=False, random_state=9)

mitigated_df     = pd.concat([lower_resampled, other_resampled]).sample(frac=1, random_state=9)
feature_cols     = [c for c in mitigated_df.columns if c not in ['target', 'race']]
input_train_mit  = mitigated_df[feature_cols].values
result_train_mit = mitigated_df['target'].values

# ── 10. TRAIN MITIGATED MODEL ─────────────────────────────────────────────────
print(f"\n##### Unbalanced Neural Network (After Mitigation, threshold={THRESHOLD})")
mlp.fit(input_train_mit, result_train_mit)

pred_after = predict_with_threshold(mlp, input_test)
print_metrics("Unbalanced Neural Network Mitigated — Full Test Set", result_test, pred_after)

# Add mitigated loss curve and save
ax_loss.plot(mlp.loss_curve_, label='After Mitigation', color='darkorange')
ax_loss.set_title('Neural Network Training Loss', fontsize=13, fontweight='bold')
ax_loss.set_xlabel('Epochs (Training Runs)', fontsize=11)
ax_loss.set_ylabel('Loss (Error Rate)', fontsize=11)
ax_loss.legend(fontsize=11)
ax_loss.grid(linestyle='--', alpha=0.5)
fig_loss.tight_layout()
fig_loss.savefig('neural_net_loss_curves.png', dpi=150, bbox_inches='tight')
print("\nLoss curve saved as 'neural_net_loss_curves.png'")

# ── 11. POST-MITIGATION RACE ACCURACY ─────────────────────────────────────────
print("\n##### Race-Stratified Accuracies (After Mitigation)")
acc_after = race_accuracies(result_test, pred_after, race_test, "Unbalanced Mitigated")

# ── 12. BAR CHART ─────────────────────────────────────────────────────────────
groups = [r for r in RACE_GROUPS if r in acc_before and r in acc_after]
x      = np.arange(len(groups))
width  = 0.35

palette = ['#5B8DB8', '#E07B9A', '#6DBF8A', '#F0A500', '#9B59B6']
colors  = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle('Neural Network — Prediction Accuracy by Race (Bias Analysis & Mitigation)',
             fontsize=13, fontweight='bold')

vals_before = [acc_before[g] * 100 for g in groups]
vals_after  = [acc_after[g]  * 100 for g in groups]

bars1 = ax.bar(x - width/2, vals_before, width, label='Before Mitigation',
               color=[colors[g] for g in groups], edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, vals_after,  width, label='After Mitigation',
               color=[colors[g] for g in groups], edgecolor='black', linewidth=0.8,
               alpha=0.5, hatch='//')

for bar, val in zip(bars1, vals_before):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, vals_after):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(groups, rotation=0, ha='center')
ax.set_xlabel('Race Group', fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('neural_net_race_bias.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nBar chart saved as 'neural_net_race_bias.png'")