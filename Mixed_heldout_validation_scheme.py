"""
KRAS Mixed Held-Out Validation — All Methods, 10 Runs
=======================================================
3 ML methods with both default and optimized hyperparameters.

Design:
  - All 9 combinations of (1 sensitive PDB system) × (1 resistant mutant)
    held out as test sets. Training uses remaining 4 systems.
  - Each of the 9 folds run 10 times with different random seeds.
  - Grid search (5-fold CV) re-run inside each fold/run on training data only.
  - All preprocessing (scaling + correlation filter) inside each fold/run.
  - Feature importance extracted inside each fold/run:
      LR  : |coef_|
      RF  : feature_importances_
      SVM : permutation importance on test set

Model configurations (7 total):
  LR_L2_default, LR_L1_optimized, LR_EN_optimized
  RF_default, RF_optimized
  SVM_default, SVM_optimized

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

from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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

X_full  = data[feature_cols].values
y_full  = data['Resistance'].values
systems = data['System'].values

sensitive_systems = ['G12C(6oim)', 'G12C(6ut0)', 'G12C(6mbt)']
resistant_systems = ['G12C/Y96C',  'G12C/Y96D',  'G12C/Y96S']

print(f"Dataset: {X_full.shape[0]} samples, {X_full.shape[1]} features")
print(f"Class: sensitive={(y_full==0).sum()}, resistant={(y_full==1).sum()}\n")

# =============================================================================
# 2. PREPROCESSING FUNCTION (inside each fold/run)
# =============================================================================

def preprocess(X_train_raw, X_test_raw, feature_names):
    """Scale and apply correlation filter on training data only."""
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
    if len(np.unique(y_true)) < 2:
        return dict(accuracy=accuracy_score(y_true, y_pred),
                    precision=np.nan, recall=np.nan, f1=np.nan,
                    auc=np.nan, specificity=np.nan)
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1        = f1_score(y_true, y_pred, average='binary', zero_division=0)
    auc       = roc_auc_score(y_true, y_proba)
    cm        = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    else:
        specificity = np.nan
    return dict(accuracy=accuracy, precision=precision, recall=recall,
                f1=f1, auc=auc, specificity=specificity)

# =============================================================================
# 4. FEATURE IMPORTANCE EXTRACTION (inside each fold/run)
# =============================================================================

def extract_importance(model_name, model, X_train, X_test, y_test,
                       selected_idx, feature_names, seed):
    """
    LR  : absolute value of model coefficients (coef_)
    RF  : built-in feature_importances_
    SVM : permutation importance on test set (RBF kernel has no coef_)
    Returns list of {Feature, Importance} for selected features only.
    """
    sel_names = [feature_names[i] for i in selected_idx]

    if model_name.startswith('LR'):
        importances = np.abs(model.coef_[0])

    elif model_name.startswith('RF'):
        importances = model.feature_importances_

    elif model_name.startswith('SVM'):
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=seed, scoring='roc_auc')
        importances = np.clip(perm.importances_mean, 0, None)

    return [{'Feature': name, 'Importance': imp}
            for name, imp in zip(sel_names, importances)]

# =============================================================================
# 5. PARAMETER GRIDS
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

model_names = ['LR_L2_default', 'LR_L1_optimized', 'LR_EN_optimized',
               'RF_default', 'RF_optimized', 'SVM_default', 'SVM_optimized']

# =============================================================================
# 6. MODEL
# =============================================================================

def get_models_fitted(X_train, y_train, seed):
    """Fit all 7 models. Grid search runs on X_train only — no leakage."""

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
# 7. MAIN LOOP — 9 FOLDS × 10 RUNS × 7 MODELS
# =============================================================================

N_RUNS      = 10
metric_cols = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']
all_results = []
importance_records  = {m: [] for m in model_names}
best_params_records = {m: [] for m in
                       ['LR_L1_optimized', 'LR_EN_optimized',
                        'RF_optimized', 'SVM_optimized']}

fold_combos = list(product(sensitive_systems, resistant_systems))
print(f"Running {len(fold_combos)} folds × {N_RUNS} runs × {len(model_names)} models "
      f"= {len(fold_combos)*N_RUNS*len(model_names)} total fits\n")

for fold_num, (sens_sys, res_sys) in enumerate(fold_combos):

    test_mask  = np.isin(systems, [sens_sys, res_sys])
    train_mask = ~test_mask

    X_train_raw = X_full[train_mask]
    X_test_raw  = X_full[test_mask]
    y_train_raw = y_full[train_mask]
    y_test      = y_full[test_mask]

    n_sens_test = (y_test == 0).sum()
    n_res_test  = (y_test == 1).sum()

    print(f"Fold {fold_num+1}/9: hold out [{sens_sys}] + [{res_sys}] "
          f"| test: sens={n_sens_test}, res={n_res_test}")

    for run in range(N_RUNS):
        seed = fold_num * 100 + run

        # Preprocessing INSIDE this fold/run on training data only
        X_train, X_test, sel_idx = preprocess(
            X_train_raw, X_test_raw, feature_cols)

        # Fit all 7 models
        models, best_params = get_models_fitted(X_train, y_train_raw, seed)

        # Record best params for optimized models this fold/run
        for m, params in best_params.items():
            best_params_records[m].append({
                'fold': fold_num + 1,
                'held_out_sensitive': sens_sys,
                'held_out_resistant': res_sys,
                'run': run + 1,
                **params
            })

        for model_name, model in models.items():
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_proba)

            all_results.append({
                'fold':               fold_num + 1,
                'held_out_sensitive': sens_sys,
                'held_out_resistant': res_sys,
                'run':                run + 1,
                'model':              model_name,
                'n_features':         len(sel_idx),
                'n_sens_test':        n_sens_test,
                'n_res_test':         n_res_test,
                **metrics
            })

            # Feature importance INSIDE this fold/run
            imp_list = extract_importance(
                model_name, model, X_train, X_test, y_test,
                sel_idx, feature_cols, seed)
            for entry in imp_list:
                importance_records[model_name].append({
                    'fold': fold_num + 1,
                    'held_out_sensitive': sens_sys,
                    'held_out_resistant': res_sys,
                    'run': run + 1,
                    **entry
                })

        print(f"  Run {run+1}/{N_RUNS} done", end='\r')

    print(f"  All {N_RUNS} runs complete.          ")

# =============================================================================
# 8. PERFORMANCE RESULTS TABLES
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df.to_excel('KRAS_mixed_heldout_all_runs.xlsx', index=False)
print("\nAll runs saved to KRAS_mixed_heldout_all_runs.xlsx")

# Per-fold summary (mean ± SD across 10 runs) per model
fold_summary_rows = []
for model_name in model_names:
    for fold_num, (sens_sys, res_sys) in enumerate(fold_combos):
        sub = results_df[
            (results_df['model'] == model_name) &
            (results_df['fold']  == fold_num + 1)
        ][metric_cols]
        row = {'model': model_name, 'fold': fold_num + 1,
               'held_out_sensitive': sens_sys, 'held_out_resistant': res_sys}
        for col in metric_cols:
            row[f'{col}_mean']    = sub[col].mean()
            row[f'{col}_sd']      = sub[col].std()
            row[f'{col}_mean_sd'] = f"{sub[col].mean():.3f} ± {sub[col].std():.3f}"
        fold_summary_rows.append(row)

fold_summary_df = pd.DataFrame(fold_summary_rows)
fold_summary_df.to_excel('KRAS_mixed_heldout_fold_summary.xlsx', index=False)

# Overall summary (mean ± SD across all 90 fold-runs) per model
overall_rows = []
for model_name in model_names:
    sub = results_df[results_df['model'] == model_name][metric_cols]
    row = {'model': model_name}
    for col in metric_cols:
        row[f'{col}_mean']    = sub[col].mean()
        row[f'{col}_sd']      = sub[col].std()
        row[f'{col}_mean_sd'] = f"{sub[col].mean():.3f} ± {sub[col].std():.3f}"
    overall_rows.append(row)

overall_df = pd.DataFrame(overall_rows)
overall_df.to_excel('KRAS_mixed_heldout_overall_summary.xlsx', index=False)

print("\n" + "="*80)
print("OVERALL SUMMARY — Mean ± SD across all 9 folds × 10 runs")
print("="*80)
print(overall_df[['model'] + [f'{c}_mean_sd' for c in metric_cols]].to_string(index=False))

# =============================================================================
# 9. AUC HEATMAPS (mean across 10 runs per fold, per model)
# =============================================================================

# --- Individual AUC heatmap per model ---
print("\nSaving individual AUC heatmap figures...")
res_labels_short = [s.replace('G12C/', '') for s in resistant_systems]
sens_labels_short = [s.replace('G12C', 'G12C') for s in sensitive_systems]

for model_name in model_names:
    auc_matrix = np.full((len(sensitive_systems), len(resistant_systems)), np.nan)
    for i, sens_sys in enumerate(sensitive_systems):
        for j, res_sys in enumerate(resistant_systems):
            sub = results_df[
                (results_df['model'] == model_name) &
                (results_df['held_out_sensitive'] == sens_sys) &
                (results_df['held_out_resistant'] == res_sys)
            ]['auc']
            auc_matrix[i, j] = sub.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(auc_matrix, cmap='RdBu', vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Mean AUC')

    # Major ticks at cell centers (for axis labels only, no gridlines here)
    ax.set_xticks(range(len(resistant_systems)))
    ax.set_yticks(range(len(sensitive_systems)))
    ax.set_xticklabels(res_labels_short, fontsize=13)
    ax.set_yticklabels(sensitive_systems, fontsize=10)
    ax.grid(False)  # explicitly disable any default/inherited major grid

    # Minor ticks at cell BOUNDARIES (half-integer offsets) — gridlines drawn here
    ax.set_xticks(np.arange(-0.5, len(resistant_systems), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sensitive_systems), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)  # hide minor tick marks

    ax.set_title(f'AUC Heatmap — {model_name}\n(mean across 10 runs per fold)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Held-out Resistant', fontsize=11)
    ax.set_ylabel('Held-out Sensitive', fontsize=11)
    for i in range(len(sensitive_systems)):
        for j in range(len(resistant_systems)):
            val = auc_matrix[i, j]
            text_color = 'white' if (val < 0.25 or val > 0.85) else 'black'
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold', color=text_color)
    plt.tight_layout()
    fname = f'KRAS_mixed_heldout_AUC_heatmap_{model_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# =============================================================================
# 10. OVERALL AUC BAR CHART
# =============================================================================

fig_bar, ax_bar = plt.subplots(figsize=(13, 6))
x      = np.arange(len(model_names))
means  = [overall_df[overall_df['model']==m]['auc_mean'].values[0] for m in model_names]
sds    = [overall_df[overall_df['model']==m]['auc_sd'].values[0]   for m in model_names]
bcolors = [cb_palette[0]]*3 + [cb_palette[1]]*2 + [cb_palette[2]]*2

bars = ax_bar.bar(x, means, yerr=sds, color=bcolors, alpha=0.85,
                  capsize=5, edgecolor='black', linewidth=0.8)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
ax_bar.set_ylabel('Mean AUC ± SD', fontsize=13)
ax_bar.set_ylim(0, 1.15)
ax_bar.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
ax_bar.set_title('Overall Mean AUC — Mixed Held-Out (9 folds × 10 runs)', fontsize=13)
ax_bar.legend(fontsize=10)
ax_bar.grid(True, axis='y', alpha=0.3)
for bar, mean, sd in zip(bars, means, sds):
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + sd + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('KRAS_mixed_heldout_AUC_summary.png', dpi=150)
plt.show()

# =============================================================================
# 11. FEATURE IMPORTANCE — aggregated across all 9 folds × 10 runs per model
# =============================================================================

importance_agg = {}

for model_name in model_names:
    records = importance_records[model_name]
    if not records:
        continue
    imp_df = pd.DataFrame(records)
    agg = (imp_df.groupby('Feature')['Importance']
                 .agg(MeanImportance='mean',
                      SDImportance='std',
                      FoldRunCount='count')
                 .reset_index()
                 .sort_values('MeanImportance', ascending=False))
    importance_agg[model_name] = agg

with pd.ExcelWriter('KRAS_mixed_heldout_feature_importance.xlsx') as writer:
    for model_name, agg in importance_agg.items():
        agg.to_excel(writer, sheet_name=model_name[:31], index=False)

print("\nFeature importance saved to KRAS_mixed_heldout_feature_importance.xlsx")

# --- Individual feature importance figure per model ---
print("\nSaving individual feature importance figures...")
total_runs = len(fold_combos) * N_RUNS  # 90

for m_idx, model_name in enumerate(model_names):
    if model_name not in importance_agg:
        continue
    top15 = importance_agg[model_name].head(15)
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
                 f'(Mixed Held-Out, averaged across 9 folds × {N_RUNS} runs)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    for bar, (_, row) in zip(bars, top15.iterrows()):
        ax.text(bar.get_width() + (top15['MeanImportance'].max() * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"n={int(row['FoldRunCount'])}/{total_runs}",
                va='center', fontsize=8, color='dimgray')
    plt.tight_layout()
    fname = f'KRAS_mixed_heldout_importance_{model_name}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

# =============================================================================
# 12. BEST HYPERPARAMETER SUMMARY — range across all 9 folds × 10 runs
# =============================================================================

print("\n" + "="*80)
print("BEST HYPERPARAMETER RANGES ACROSS ALL FOLDS × RUNS (for table footnotes)")
print("="*80)

params_summary_rows = []

for model_name, records in best_params_records.items():
    if not records:
        continue
    params_df = pd.DataFrame(records)
    print(f"\n{model_name}:")
    row = {'model': model_name}
    param_cols = [c for c in params_df.columns
                  if c not in ['fold','held_out_sensitive',
                                'held_out_resistant','run']]
    for col in param_cols:
        unique_vals = sorted(params_df[col].astype(str).unique())
        range_str   = ', '.join(unique_vals)
        print(f"  {col}: {range_str}")
        row[f'{col}_values'] = range_str
        row[f'{col}_mode']   = params_df[col].astype(str).mode()[0]
    params_summary_rows.append(row)

    # Also save per-fold breakdown
    fold_param_summary = (params_df.groupby('held_out_sensitive')[param_cols]
                          .agg(lambda x: ', '.join(sorted(x.astype(str).unique())))
                          .reset_index())
    print(f"  Per-sensitive-system breakdown:")
    print(fold_param_summary.to_string(index=False))

params_summary_df = pd.DataFrame(params_summary_rows)

with pd.ExcelWriter('KRAS_mixed_heldout_best_params.xlsx') as writer:
    params_summary_df.to_excel(writer, sheet_name='Overall_range', index=False)
    for model_name, records in best_params_records.items():
        if records:
            pd.DataFrame(records).to_excel(
                writer, sheet_name=model_name[:31], index=False)

print("\nBest params saved to KRAS_mixed_heldout_best_params.xlsx")
print("  Sheet 'Overall_range': range of params across all folds/runs")
print("  Per-model sheets: every individual fold/run selection")

print("\nDone. Output files:")
print("  KRAS_mixed_heldout_all_runs.xlsx")
print("  KRAS_mixed_heldout_fold_summary.xlsx")
print("  KRAS_mixed_heldout_overall_summary.xlsx")
print("  KRAS_mixed_heldout_feature_importance.xlsx")
print("  KRAS_mixed_heldout_best_params.xlsx")
print("  KRAS_mixed_heldout_AUC_summary.png")
for m in model_names:
    print(f"  KRAS_mixed_heldout_AUC_heatmap_{m}.png")
    print(f"  KRAS_mixed_heldout_importance_{m}.png")
