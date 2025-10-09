#1. Data import and scaling

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_excel('KRAS_MD_molecular_descriptors.xlsx')
original_colnames = data.columns

# Define columns to exclude
exclude_columns = ['Sample', 'Resistance', 'Trajectory frames percentage']
data_to_scale = data.drop(columns=exclude_columns)
original_colnames2 = data_to_scale.columns

# Apply Z-score normalization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_to_scale)
data_df = pd.DataFrame(data_scaled)

#2. Check for inter-variable correlation

import numpy as np

# Calculate correlation matrix
correlation_matrix = data_df.corr().abs()

# Initialize an empty list to store selected features (inter-variable correlation less than 75%)
selected_features075 = []

# Iterate over each feature in the DataFrame and add these that match the criteria
for feature in correlation_matrix.columns:
    # Check if this feature is < 0.75 correlated with all features already in the selected list
    if all(correlation_matrix.loc[feature, selected] < 0.75 for selected in selected_features075):
        selected_features075.append(feature)  

# Save the selected features in a new DataFrame
print("Selected features with pairwise correlations < 0.75:")
colnames_selected075=original_colnames2[selected_features075]
df_selected075=data_df[selected_features075]
print(colnames_selected075)

#3. Random Forest model with default settings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, log_loss, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score, fbeta_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Define the feature matrix (X) and target variable (y)

X = df_selected075 
y = data['Resistance']

# Run the process 10 times, initialize a list to store results
results = []
feature_importances = np.zeros(X.shape[1])  

# Initialize plot for all ROC curves
plt.figure(figsize=(10, 8))  

for i in range(10):
    print(f"\n*** Iteration {i+1} ***")

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize and train the Random Forest model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')

    # Calculate confusion matrix and specificity
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()  # Extract True Negatives, False Positives, False Negatives, True Positives
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Avoid division by zero

    # Store results
    results.append({
        "Iteration": i + 1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc,
        "Specificity": specificity
    })

    # Update cumulative feature importance
    feature_importances += rf.feature_importances_

    # Plot ROC Curve for the current iteration
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"Iteration {i+1}")

# Finalize ROC Curve plot
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)
plt.title("Random Forest ROC-AUC", fontsize=24)
plt.legend(loc="lower right", fontsize=16 )
plt.xticks(fontsize=16)  # Set font size for X-axis tick labels
plt.yticks(fontsize=16)  # Set font size for Y-axis tick labels
plt.grid()
plt.show()

# Calculate average feature importance
avg_feature_importance = feature_importances / 10
feature_importance_df = pd.DataFrame({
    "Feature": original_colnames2[df_selected075.columns],
    "Importance": avg_feature_importance
}).sort_values(by="Importance", ascending=False)

# Print descriptive statistics for selected metrics
results_df = pd.DataFrame(results)
print(results_df[['Accuracy', 'Precision', 'Recall', 'ROC-AUC', 'Specificity']].describe())

# The top 15 features by importance
top_features = feature_importance_df.sort_values(by="Importance", ascending=False).head(15)

# Plot the top 15 features
plt.figure(figsize=(12, 8))
plt.bar(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.ylabel('Average Feature Importance Over 10 Iterations', fontsize=20)
plt.xlabel('Feature', fontsize=20)
plt.title('Top 15 Feature Importance in Random Forest', fontsize=24)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
