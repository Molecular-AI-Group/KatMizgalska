#1. Data import and scaling

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_excel('KRAS_MD_molecular_descriptors.xlsx')
original_colnames = data.columns

# Define columns to exclude
exclude_columns = ['sample', 'resistance', 'frames percentage']
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

#3. Logistic regression model with default settings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the feature matrix (X) and target variable (y)
X = df_selected075  
y = data['resistance']      

# Initialize a list to store results
results = []

# Create a figure for ROC-AUC curves
plt.figure(figsize=(10, 8))

for i in range(10):
    print(f"Run {i+1}:")

    # Split the data (changing random_state ensures different splits)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Initialize and fit the logistic regression model
    log_reg = LogisticRegression(max_iter=100, random_state=i)
    log_reg.fit(X_train, y_train)

    # Make predictions
    y_pred = log_reg.predict(X_test)
    y_pred_proba = log_reg.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_pred_proba) 

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()  

    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Append results to the list
    results.append({
        'run': i+1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'specificity': specificity,
        'confusion_matrix': cm
    })

    # Plot ROC curve for this run
    plt.plot(fpr, tpr, label=f'Iteration {i+1}')

# Finalize ROC Curve plot
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate", fontsize=20)
plt.ylabel("True Positive Rate", fontsize=20)
plt.title("Logistic Regression ROC-AUC", fontsize=24)
plt.legend(loc="lower right", fontsize=16 )
plt.xticks(fontsize=16)  # Set font size for X-axis tick labels
plt.yticks(fontsize=16)  # Set font size for Y-axis tick labels
plt.grid()
plt.show()

# Print the model evaluation results
results_df075 = pd.DataFrame(results)
print(results_df075[['accuracy', 'precision', 'recall', 'auc', 'specificity']].describe())

# Feature importance extraction
coefficients = log_reg.coef_[0]  
features = original_colnames2[df_selected075.columns] 

# Create a DataFrame for better readability
importance_df075 = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients,
    'Importance': np.abs(coefficients)  
})

# Sort by importance
importance_df075 = importance_df075.sort_values(by='Importance', ascending=False)

# Get the top 15 features by importance
top_features = importance_df075.sort_values(by="Importance", ascending=False).head(15)

# Plot the top 15 features
plt.figure(figsize=(12, 8))
plt.bar(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.ylabel('Absolute Coefficient Value', fontsize=20)
plt.xlabel('Feature', fontsize=20)
plt.title('Top 15 Feature Importance in Logistic Regression', fontsize=24)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
