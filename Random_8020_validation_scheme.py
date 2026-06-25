"""
KRAS Random 80/20 Split 
====================================================
  - Scaling and correlation-based feature selection performed inside each run
    on training data only
  - Hyperparameter tuning (GridSearchCV) performed inside each run on
    training data only
  - Feature importance extracted inside each run from trained models
    (LR: coef_, RF: feature_importances_, SVM: permutation importance)
  - 10 runs with different random seeds for stability estimation
  - 7 model configurations across 3 methods

Model configurations:
  Logistic Regression:
    LR_L2_default : C=1.0, solver=lbfgs (default)
    LR_L1_optimized : GridSearchCV -> best C, solver=saga
    LR_EN_optimized : GridSearchCV -> best C + l1_ratio, solver=saga

  Random Forest:
    RF_default  : n_estimators=100, max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, max_features='sqrt'
    RF_optimized : GridSearchCV -> best params

  Support Vector Machine:
    SVM_default : C=1.0, gamma='scale', kernel='rbf'
    SVM_optimized : GridSearchCV -> best params

Preprocessing: StandardScaler + correlation filter (<0.75) inside each run.
Encoding: 0 = Sensitive, 1 = Resistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Colorblind-safe palette globally
sns.set_palette("colorblind")
sns.set_style("whitegrid")
cb_palette = sns.color_palette("colorblind")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================

data = pd.read_excel('KRAS_MD_molecular_descriptors.xlsx')

def extract_system(sample_name):
    return str(sample_name).split(' ')[0]

data['System'] = data['Sample'].apply(extract_system)

exclude_columns = ['Sample', 'Resistance', 'Trajectory frames percentage', 'System']
feature_cols = [c for c in data.columns if c not in exclude_columns]

for c in feature_cols:
    if data[c].dtype == object:
        data[c] = (data[c].astype(str)
                   .str.replace('\xa0', '', regex=False)
                   .str.strip())
        data[c] = pd.to_numeric(data[c], errors='coerce')

X_full = data[feature_cols].values
y_full = data['Resistance'].values
print(f"Dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features")
print(f"Class distribution: sensitive={(y_full==0).sum()}, resistant={(y_full==1).sum()}\n")

# =============================================================================
# 2. PREPROCESSING FUNCTION (inside each fold)
# =============================================================================

def preprocess(X_train_raw, X_test_raw, feature_names):
    """Scale and apply correlation filter on training data only.
    Returns processed arrays and list of selected feature indices."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    train_df    = pd.DataFrame(X_train_scaled, columns=feature_names)
    corr_matrix = train_df.corr().abs()

    selected_idx = []
    for idx, feat in enumerate(feature_names):
        if all(corr_matrix.loc[feat, feature_names[s]] < 0.75
               for s in selected_idx):
            selected_idx.append(idx)

    return (X_train_scaled[:, selected_idx],
            X_test_scaled[:, selected_idx],
            selected_idx)

# =============================================================================
# 3. METRICS FUNCTION
# =============================================================================

def compute_metrics(y_true, y_pred, y_proba):
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='binary', zero_division=0)
    auc       = roc_auc_score(y_true, y_proba) \
                if len(np.unique(y_true)) > 1 else np.nan
    cm        = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        specificity = np.nan
    return dict(accuracy=accuracy, precision=precision, recall=recall,
                f1=f1, auc=auc, specificity=specificity)

# =============================================================================
# 4. FEATURE IMPORTANCE EXTRACTION (inside each fold)
# =============================================================================

def extract_importance(model_name, model, X_train, X_test, y_test,
                       selected_idx, feature_names, seed):
    """
    Extract feature importances inside the fold.
      LR  : absolute value of coefficients (coef_)
      RF  : built-in feature_importances_
      SVM : permutation importance on TEST set (non-linear kernel has no coef_)

    Returns list of dicts {Feature, Importance} for selected features only.
    """
    sel_names = [feature_names[i] for i in selected_idx]

    if model_name.startswith('LR'):
        importances = np.abs(model.coef_[0])

    elif model_name.startswith('RF'):
        importances = model.feature_importances_

    elif model_name.startswith('SVM'):
        # Permutation importance on test set; n_repeats=10 for stability
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=seed, scoring='roc_auc')
        importances = perm.importances_mean
        # Clip negatives to 0 (negligible features may score slightly negative)
        importances = np.clip(importances, 0, None)

    return [{'Feature': name, 'Importance': imp}
            for name, imp in zip(sel_names, importances)]

# =============================================================================
# 5. GRID SEARCH PARAMETER GRIDS
# =============================================================================

lr_l1_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
lr_en_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
              'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}
rf_grid    = {'n_estimators': [50, 100, 200],
              'max_depth': [None, 5, 10],
              'max_features': ['sqrt', 'log2'],
              'min_samples_split': [2, 5],
              'min_samples_leaf': [1, 2]}
svm_grid   = {'C': [0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 'scale'],
              'kernel': ['rbf']}

# =============================================================================
# 6. MODEL
# =============================================================================

def get_models(X_train, y_train, seed):
    """Fit all 7 models. Grid search for optimized variants — on X_train only."""

    lr_l2 = LogisticRegression(C=1.0, solver='lbfgs', penalty='l2',
                                max_iter=1000, random_state=seed)
    lr_l2.fit(X_train, y_train)

    gs_l1 = GridSearchCV(
        LogisticRegression(penalty='l1', solver='saga', max_iter=1000,
                           random_state=seed),
        lr_l1_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    gs_l1.fit(X_train, y_train)

    gs_en = GridSearchCV(
        LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000,
                           random_state=seed),
        lr_en_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    gs_en.fit(X_train, y_train)

    rf_def = RandomForestClassifier(n_estimators=100, max_depth=None,
                                     min_samples_split=2, min_samples_leaf=1,
                                     max_features='sqrt', random_state=seed)
    rf_def.fit(X_train, y_train)

    gs_rf = GridSearchCV(
        RandomForestClassifier(random_state=seed),
        rf_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    gs_rf.fit(X_train, y_train)

    svm_def = SVC(C=1.0, gamma='scale', kernel='rbf',
                  probability=True, random_state=seed)
    svm_def.fit(X_train, y_train)

    gs_svm = GridSearchCV(
        SVC(probability=True, random_state=seed),
        svm_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    gs_svm.fit(X_train, y_train)

    models = {
        'LR_L2_default':   lr_l2,
        'LR_L1_optimized': gs_l1.best_estimator_,
        'LR_EN_optimized': gs_en.best_estimator_,
        'RF_default':      rf_def,
        'RF_optimized':    gs_rf.best_estimator_,
        'SVM_default':     svm_def,
        'SVM_optimized':   gs_svm.best_estimator_,
    }
    best_params = {
        'LR_L1_optimized': gs_l1.best_params_,
        'LR_EN_optimized': gs_en.best_params_,
        'RF_optimized':    gs_rf.best_params_,
        'SVM_optimized':   gs_svm.best_params_,
    }
    return models, best_params

# =============================================================================
# 7. MAIN LOOP — 10 RUNS
# =============================================================================

N_RUNS = 10
model_names = ['LR_L2_default', 'LR_L1_optimized', 'LR_EN_optimized',
               'RF_default', 'RF_optimized', 'SVM_default', 'SVM_optimized']

all_results        = []
importance_records = {m: [] for m in model_names}
# best_params_records: {model_name: [dict of params per run]}
best_params_records = {m: [] for m in
                       ['LR_L1_optimized', 'LR_EN_optimized',
                        'RF_optimized', 'SVM_optimized']}

roc_colors = sns.color_palette("colorblind", N_RUNS)

# Store ROC data per model per run for individual saving
roc_data = {m: [] for m in model_names}

for run in range(N_RUNS):
    seed = run
    print(f"Run {run+1}/{N_RUNS} ...", end=' ', flush=True)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=seed, stratify=y_full)

    X_train, X_test, sel_idx = preprocess(X_train_raw, X_test_raw, feature_cols)
    models, best_params = get_models(X_train, y_train, seed)

    for m, params in best_params.items():
        best_params_records[m].append({'run': run + 1, **params})

    for model_name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        all_results.append({
            'run': run + 1,
            'model': model_name,
            'n_features_selected': len(sel_idx),
            **metrics
        })

        imp_list = extract_importance(
            model_name, model, X_train, X_test, y_test,
            sel_idx, feature_cols, seed)
        for entry in imp_list:
            importance_records[model_name].append(
                {'run': run + 1, **entry})

        # Store ROC data for later individual plotting
        if not np.isnan(metrics['auc']):
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[model_name].append({
                'run': run + 1,
                'fpr': fpr,
                'tpr': tpr,
                'auc': metrics['auc']
            })

    print("done")

# --- Individual ROC figure per model ---
print("\nSaving individual ROC figures...")
for model_name in model_names:
    fig, ax = plt.subplots(figsize=(6, 5))
    for entry in roc_data[model_name]:
        ax.plot(entry['fpr'], entry['tpr'],
                color=roc_colors[entry['run'] - 1],
                alpha=0.75, linewidth=1.5,
                label=f"Run {entry['run']} (AUC={entry['auc']:.2f})")
    ax.plot([0,1],[0,1],'--', color='gray', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate', fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(f'ROC Curves — {model_name}\n(Random 80/20, 10 runs)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right', framealpha=0.85)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fname = f'KRAS_random8020_ROC_{model_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# =============================================================================
# 8. PERFORMANCE SUMMARY TABLE
# =============================================================================

results_df  = pd.DataFrame(all_results)
metric_cols = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']

results_df.to_excel('KRAS_random8020_all_runs.xlsx', index=False)

summary_rows = []
for mname in model_names:
    sub = results_df[results_df['model'] == mname][metric_cols]
    row = {'model': mname}
    for col in metric_cols:
        row[f'{col}_mean']    = sub[col].mean()
        row[f'{col}_sd']      = sub[col].std()
        row[f'{col}_mean_sd'] = f"{sub[col].mean():.3f} ± {sub[col].std():.3f}"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel('KRAS_random8020_summary.xlsx', index=False)

print("\n" + "="*80)
print("PERFORMANCE SUMMARY — Mean ± SD across 10 runs")
print("="*80)
display_cols = ['model'] + [f'{c}_mean_sd' for c in metric_cols]
print(summary_df[display_cols].to_string(index=False))

# =============================================================================
# 9. PERFORMANCE BAR CHART
# =============================================================================

fig_bar, ax_bar = plt.subplots(figsize=(14, 6))
x           = np.arange(len(model_names))
width       = 0.13
metric_plot = ['accuracy', 'precision', 'recall', 'specificity', 'auc']
bar_colors  = [cb_palette[i] for i in range(len(metric_plot))]

for i, (metric, color) in enumerate(zip(metric_plot, bar_colors)):
    means = [summary_df[summary_df['model']==m][f'{metric}_mean'].values[0]
             for m in model_names]
    sds   = [summary_df[summary_df['model']==m][f'{metric}_sd'].values[0]
             for m in model_names]
    ax_bar.bar(x + i*width, means, width, yerr=sds, label=metric,
               color=color, alpha=0.85, capsize=3, edgecolor='black',
               linewidth=0.5)

ax_bar.set_xticks(x + width*2)
ax_bar.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
ax_bar.set_ylabel('Score', fontsize=13)
ax_bar.set_ylim(0, 1.15)
ax_bar.set_title('Random 80/20 — Mean ± SD across 10 runs\n(corrected preprocessing)',
                  fontsize=13)
ax_bar.legend(fontsize=10)
ax_bar.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('KRAS_random8020_summary_bar.png', dpi=150)
plt.show()

# =============================================================================
# 10. FEATURE IMPORTANCE — aggregated across 10 runs per model
#     Only features selected in a given run contribute to that run's average.
#     FoldCount shows in how many of 10 runs the feature was selected.
# =============================================================================

importance_summary = {}

for model_name in model_names:
    records = importance_records[model_name]
    if not records:
        continue
    imp_df = pd.DataFrame(records)
    agg = (imp_df.groupby('Feature')['Importance']
                 .agg(MeanImportance='mean',
                      SDImportance='std',
                      RunCount='count')
                 .reset_index()
                 .sort_values('MeanImportance', ascending=False))
    importance_summary[model_name] = agg

# Save all importance tables to one Excel file with one sheet per model
with pd.ExcelWriter('KRAS_random8020_feature_importance.xlsx') as writer:
    for model_name, agg in importance_summary.items():
        agg.to_excel(writer, sheet_name=model_name[:31], index=False)

print("\nFeature importance saved to KRAS_random8020_feature_importance.xlsx")

print("\nSaving individual feature importance figures...")
for m_idx, model_name in enumerate(model_names):
    if model_name not in importance_summary:
        continue
    top15 = importance_summary[model_name].head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(len(top15)), top15['MeanImportance'],
                   xerr=top15['SDImportance'],
                   color=cb_palette[m_idx % len(cb_palette)],
                   alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15['Feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Importance ± SD', fontsize=12)
    ax.set_title(f'Top-15 Feature Importance — {model_name}\n'
                 f'(Random 80/20, averaged across {N_RUNS} runs)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    for bar, (_, row) in zip(bars, top15.iterrows()):
        ax.text(bar.get_width() + (top15['MeanImportance'].max() * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"n={int(row['RunCount'])}/{N_RUNS}",
                va='center', fontsize=8, color='dimgray')
    plt.tight_layout()
    fname = f'KRAS_random8020_importance_{model_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# =============================================================================
# 11. BEST HYPERPARAMETER SUMMARY — range across 10 runs
# =============================================================================

print("\n" + "="*80)
print("BEST HYPERPARAMETER RANGES ACROSS 10 RUNS (for table footnotes)")
print("="*80)

params_summary_rows = []

for model_name, records in best_params_records.items():
    if not records:
        continue
    params_df = pd.DataFrame(records)
    print(f"\n{model_name}:")
    row = {'model': model_name}
    for col in params_df.columns:
        if col == 'run':
            continue
        unique_vals = sorted(params_df[col].astype(str).unique())
        range_str   = ', '.join(unique_vals)
        print(f"  {col}: {range_str}")
        row[f'{col}_values']  = range_str
        row[f'{col}_mode']    = params_df[col].astype(str).mode()[0]
    params_summary_rows.append(row)

params_summary_df = pd.DataFrame(params_summary_rows)
params_summary_df.to_excel('KRAS_random8020_best_params.xlsx', index=False)
print("\nBest params summary saved to KRAS_random8020_best_params.xlsx")
print("(Use '_values' columns for range reporting, '_mode' for most frequent value)")

print("\nDone. Output files:")
print("  KRAS_random8020_all_runs.xlsx")
print("  KRAS_random8020_summary.xlsx")
print("  KRAS_random8020_feature_importance.xlsx")
print("  KRAS_random8020_best_params.xlsx")
print("  KRAS_random8020_summary_bar.png")
for m in model_names:
    print(f"  KRAS_random8020_ROC_{m}.png")
    print(f"  KRAS_random8020_importance_{m}.png")
