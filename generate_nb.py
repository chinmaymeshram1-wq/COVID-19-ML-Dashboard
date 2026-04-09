import json
import os

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

# --- NOTEBOOK CONTENT ---

add_md("# COVID-19 Comprehensive Machine Learning Analysis\n\nThis notebook demonstrates 5 key systems:\n1. **COVID Trend Analysis Dashboard** (EDA, Visualizations, PCA)\n2. **COVID-19 Severity Prediction System** (Classification with Logistic Regression, Decision Tree, Random Forest)\n3. **Death Rate Prediction System** (Linear Regression)\n4. **WHO Region Classification System** (Classification)\n5. **Model Comparison System** (Performance comparison)")

add_md("## 1. Setup and Data Loading\nImporting all necessary Python libraries for data manipulation, visualization, and machine learning.")
add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# IMPORTANT: Ensure 'country_wise_latest.csv' is uploaded to the Colab environment.
df = pd.read_csv('country_wise_latest.csv')
display(df.head())""")

add_md("## 2. System 4: COVID Trend Analysis Dashboard (EDA)\nComprehensive Exploratory Data Analysis including summary statistics, missing values check, histograms, boxplots, correlation heatmaps, and PCA.")
add_code("""print("Data Shape:", df.shape)

print("\\n--- Statistical Summary ---")
display(df.describe())

print("\\n--- Missing Values ---")
display(df.isnull().sum())

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap of COVID-19 Features")
plt.show()

# 2. Histograms (Distributions)
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
sns.histplot(df['Confirmed'], bins=30, kde=True, color='blue')
plt.title("Distribution of Confirmed Cases")

plt.subplot(1, 3, 2)
sns.histplot(df['Deaths'], bins=30, kde=True, color='red')
plt.title("Distribution of Deaths")

plt.subplot(1, 3, 3)
sns.histplot(df['Recovered'], bins=30, kde=True, color='green')
plt.title("Distribution of Recovered")
plt.tight_layout()
plt.show()

# 3. Boxplot for Outliers Identification
plt.figure(figsize=(12, 6))
sns.boxplot(x='WHO Region', y='Deaths / 100 Cases', data=df, palette='Set2')
plt.xticks(rotation=45)
plt.title("Death Rate Across WHO Regions")
plt.show()

# 4. Scatter Plot (Feature relationships)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Confirmed', y='Deaths', hue='WHO Region', data=df, s=100, alpha=0.7)
plt.title("Confirmed Cases vs Deaths by WHO Region")
plt.show()

# 5. PCA Visualization (Dimensionality Reduction)
# Standardize features before applying PCA
features_pca = ['Confirmed', 'Deaths', 'Recovered', 'Active']
X_pca = df[features_pca]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=df['WHO Region'], s=100, alpha=0.8)
plt.title("PCA 2D Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()""")


add_md("## 3. System 2: Death Rate Prediction System (Regression)\nPredicting the `Deaths / 100 Cases` using Linear Regression based on Confirmed Cases, Active Cases, and WHO Region.")
add_code("""# Preparing data for Regression
df_reg = df.copy()

# Encoding 'WHO Region' for the regression model
le_region = LabelEncoder()
df_reg['WHO Region Encoded'] = le_region.fit_transform(df_reg['WHO Region'])

# Features and Target
X_reg = df_reg[['Confirmed', 'Active', 'WHO Region Encoded']]
y_reg = df_reg['Deaths / 100 Cases']

# Train-Test Split (80% train, 20% test)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Standardize the features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# Model Training
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)

# Predictions
y_pred_reg = lr_model.predict(X_test_reg_scaled)

# Evaluation
print("--- Linear Regression Evaluation ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
print(f"R-squared Score (R2): {r2_score(y_test_reg, y_pred_reg):.4f}")

# Visualization of Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, color='purple')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], color='red', lw=2, linestyle='--')
plt.xlabel("Actual Death Rate")
plt.ylabel("Predicted Death Rate")
plt.title("Linear Regression: Actual vs Predicted Death Rate")
plt.show()""")


add_md("## 4. System 1: COVID-19 Severity Prediction System (Classification)\nClassifying severity into 'Low', 'Medium', and 'High' Risk based on Active cases, using Logistic Regression, Decision Tree, and Random Forest.")
add_code("""# 1. Create a New Column categorizing Severity Risk
# Logic: If Active cases > 50,000 (High Risk), > 10,000 (Medium Risk), else (Low Risk)
def classify_severity(active_cases):
    if active_cases > 50000:
        return 'High Risk'
    elif active_cases > 10000:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df['Risk_Level'] = df['Active'].apply(classify_severity)

print("Distribution of Risk Levels:")
print(df['Risk_Level'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Risk_Level', data=df, order=['Low Risk', 'Medium Risk', 'High Risk'], palette='viridis')
plt.title("Risk Level Distribution (Based on Active Cases)")
plt.show()

# 2. Features and Target
X_sev = df[['Confirmed', 'Deaths', 'Recovered', 'New cases', 'New deaths']]
y_sev = df['Risk_Level']

# 3. Data Preprocessing
le_risk = LabelEncoder()
y_sev_encoded = le_risk.fit_transform(y_sev) # Encodes Low, Medium, High into 0, 1, 2

X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X_sev, y_sev_encoded, test_size=0.2, random_state=42)

scaler_sev = StandardScaler()
X_train_sev_scaled = scaler_sev.fit_transform(X_train_sev)
X_test_sev_scaled = scaler_sev.transform(X_test_sev)

# --- MODEL TRAINING ---

# Model A: Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_sev_scaled, y_train_sev)
pred_log_sev = log_model.predict(X_test_sev_scaled)
acc_log_sev = accuracy_score(y_test_sev, pred_log_sev)

# Model B: Decision Tree
dt_model_sev = DecisionTreeClassifier(random_state=42)
dt_model_sev.fit(X_train_sev_scaled, y_train_sev)
pred_dt_sev = dt_model_sev.predict(X_test_sev_scaled)
acc_dt_sev = accuracy_score(y_test_sev, pred_dt_sev)

# Model C: Random Forest
rf_model_sev = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_sev.fit(X_train_sev_scaled, y_train_sev)
pred_rf_sev = rf_model_sev.predict(X_test_sev_scaled)
acc_rf_sev = accuracy_score(y_test_sev, pred_rf_sev)

print("\\n--- Classification Accuracies ---")
print(f"Logistic Regression: {acc_log_sev:.4f}")
print(f"Decision Tree:       {acc_dt_sev:.4f}")
print(f"Random Forest:       {acc_rf_sev:.4f}")

# Confusion Matrix for the Best Performing Model (usually Random Forest)
plt.figure(figsize=(7, 5))
cm_rf = confusion_matrix(y_test_sev, pred_rf_sev)
sns.heatmap(cm_rf, annot=True, cmap='Blues', fmt='d', xticklabels=le_risk.classes_, yticklabels=le_risk.classes_)
plt.title("Confusion Matrix: Random Forest (Severity Prediction)")
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")
plt.show()""")


add_md("## 5. System 3: WHO Region Classification System\nPredicting the WHO Region based on the primary pandemic indicators (Confirmed, Deaths, Recovered, Active).")
add_code("""# Note: Some regions have very few records making this a challenging task for basic models, but great for learning!
X_who = df[['Confirmed', 'Deaths', 'Recovered', 'Active']]
y_who = df['WHO Region']

X_train_who, X_test_who, y_train_who, y_test_who = train_test_split(X_who, y_who, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_who = DecisionTreeClassifier(random_state=42)
dt_who.fit(X_train_who, y_train_who)
pred_dt_who = dt_who.predict(X_test_who)
acc_dt_who = accuracy_score(y_test_who, pred_dt_who)

# Random Forest Classifier
rf_who = RandomForestClassifier(n_estimators=100, random_state=42)
rf_who.fit(X_train_who, y_train_who)
pred_rf_who = rf_who.predict(X_test_who)
acc_rf_who = accuracy_score(y_test_who, pred_rf_who)

print(f"Decision Tree Accuracy (WHO Region): {acc_dt_who:.4f}")
print(f"Random Forest Accuracy (WHO Region): {acc_rf_who:.4f}")

print("\\nClassification Report for Random Forest:")
print(classification_report(y_test_who, pred_rf_who, zero_division=0))""")


add_md("## 6. System 5: Model Comparison System\nVisualizing the performance of our Classification Models from the Severity Prediction task (System 1) to determine the best algorithm.")
add_code("""models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [acc_log_sev, acc_dt_sev, acc_rf_sev]

plt.figure(figsize=(9, 6))
ax = sns.barplot(x=models, y=accuracies, palette='magma')
plt.ylim(0, 1.1)  # Accuracies go from 0 to 1
plt.title("Model Accuracy Comparison (Severity Prediction Task)", fontsize=14)
plt.ylabel("Accuracy Score", fontsize=12)
plt.xlabel("Machine Learning Model", fontsize=12)

# Displaying text values on top of the bars
for i, v in enumerate(accuracies):
    ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=12)

plt.show()

# Dynamic Output of the Best Model
best_accuracy = max(accuracies)
best_model_idx = accuracies.index(best_accuracy)
best_model_name = models[best_model_idx]

print("=" * 40)
print(f"🏆 BEST MODEL IDENTIFIED 🏆")
print("=" * 40)
print(f"The best performing model for Severity Prediction is:")
print(f"*** {best_model_name} ***")
print(f"With an Accuracy of: {best_accuracy * 100:.2f}%")
print("=" * 40)""")

# Write output to ipynb file in the correct directory
output_dir = r"c:\Users\CHINMAY\OneDrive\Desktop\davl"
output_path = os.path.join(output_dir, 'covid_ml_analysis.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully generated {output_path}")
