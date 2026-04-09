import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(page_title="COVID-19 ML Dashboard", layout="wide", page_icon="🦠")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #2E86C1;
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #F8F9F9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2E86C1;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), 'country_wise_latest.csv')
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: Could not find 'country_wise_latest.csv' in {os.path.dirname(__file__)}. Please ensure the dataset is in the same folder as this script.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# --- MAIN TITLE ---
st.markdown("<div class='main-title'>COVID-19 Comprehensive Machine Learning Analysis</div>", unsafe_allow_html=True)


# --- DATASET OVERVIEW ---
with st.expander("📊 Dataset Overview (Click to Expand)", expanded=False):
    st.markdown("### Basic Dataset Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Countries/Regions", df.shape[0])
    col2.metric("Total Features (Columns)", df.shape[1])
    col3.metric("Total Confirmed Cases (Global)", f"{df['Confirmed'].sum():,}")
    
    st.markdown("#### Head (First 5 Rows)")
    st.dataframe(df.head())
    
    st.markdown("#### Tail (Last 5 Rows)")
    st.dataframe(df.tail())
    
    st.markdown("#### Statistical Description")
    st.dataframe(df.describe())
    
    st.markdown("#### Missing Values")
    missing_data = df.isnull().sum()
    if missing_data.sum() == 0:
        st.success("No missing values found in the dataset! Perfect!")
    else:
        st.write(missing_data[missing_data > 0])


# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
st.sidebar.markdown("Explore the 5 different Machine Learning analytical systems developed for this project:")
selected_option = st.sidebar.radio(
    "Choose a System:",
    [
        "1. COVID Trend Analysis Dashboard",
        "2. Severity Prediction (Classification)",
        "3. Death Rate Prediction (Regression)",
        "4. WHO Region Classification",
        "5. Final Model Comparison"
    ]
)


st.divider()

# ==============================================================================
# SYSTEM 1: COVID TREND ANALYSIS DASHBOARD
# ==============================================================================
if selected_option == "1. COVID Trend Analysis Dashboard":
    st.header("📈 System 1: COVID Trend Analysis & EDA")
    st.markdown("<div class='info-box'>This section uses Exploratory Data Analysis (EDA) and Principal Component Analysis (PCA) to find trends and visualize the underlying structure of the pandemic data.</div>", unsafe_allow_html=True)
    
    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)
    
    st.subheader("Global Case Distributions")
    colA, colB = st.columns(2)
    with colA:
        fig_hist1, ax_hist1 = plt.subplots()
        sns.histplot(df['Confirmed'], bins=30, kde=True, color='blue', ax=ax_hist1)
        ax_hist1.set_title("Distribution of Confirmed Cases")
        st.pyplot(fig_hist1)
    with colB:
        fig_hist2, ax_hist2 = plt.subplots()
        sns.histplot(df['Deaths'], bins=30, kde=True, color='red', ax=ax_hist2)
        ax_hist2.set_title("Distribution of Deaths")
        st.pyplot(fig_hist2)

    st.subheader("PCA: 2D Spatial Reduction")
    st.write("We compress multi-dimensional features (Cases, Deaths, Recovered, Active) into two principal components to easily visualize clusters on a 2D plot.")
    features_pca = ['Confirmed', 'Deaths', 'Recovered', 'Active']
    X_pca = df[features_pca]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_scaled)
    
    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=df['WHO Region'], s=100, alpha=0.8, ax=ax_pca)
    ax_pca.set_title("PCA 2D Visualization by WHO Region")
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")
    st.pyplot(fig_pca)

# ==============================================================================
# SYSTEM 2: SEVERITY PREDICTION SYSTEM
# ==============================================================================
elif selected_option == "2. Severity Prediction (Classification)":
    st.header("🚦 System 2: COVID-19 Severity Prediction System")
    st.markdown("<div class='info-box'><b>Goal:</b> Classify a country's pandemic severity into 'Low', 'Medium', or 'High' Risk based on the number of Active cases.<br><b>Models:</b> Logistic Regression, Decision Tree, Random Forest.</div>", unsafe_allow_html=True)
    
    def classify_severity(active_cases):
        if active_cases > 50000:
            return 'High Risk'
        elif active_cases > 10000:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    df_sev = df.copy()
    df_sev['Risk_Level'] = df_sev['Active'].apply(classify_severity)
    
    st.subheader("Target Variable Generation")
    fig_sev, ax_sev = plt.subplots(figsize=(8, 4))
    sns.countplot(x='Risk_Level', data=df_sev, order=['Low Risk', 'Medium Risk', 'High Risk'], palette='viridis', ax=ax_sev)
    ax_sev.set_title("Distribution of Risk Levels (Generated from Active Cases)")
    st.pyplot(fig_sev)
    
    st.subheader("Model Training & Evaluation")
    st.write("Using features: Confirmed, Deaths, Recovered, New cases, New deaths.")
    
    X_sev = df_sev[['Confirmed', 'Deaths', 'Recovered', 'New cases', 'New deaths']]
    y_sev = df_sev['Risk_Level']
    
    le_risk = LabelEncoder()
    y_sev_encoded = le_risk.fit_transform(y_sev)
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X_sev, y_sev_encoded, test_size=0.2, random_state=42)
    scaler_sev = StandardScaler()
    X_train_sev_scaled = scaler_sev.fit_transform(X_train_sev)
    X_test_sev_scaled = scaler_sev.transform(X_test_sev)

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_sev_scaled, y_train_sev)
    acc_log_sev = accuracy_score(y_test_sev, log_model.predict(X_test_sev_scaled))

    dt_model_sev = DecisionTreeClassifier(random_state=42)
    dt_model_sev.fit(X_train_sev_scaled, y_train_sev)
    acc_dt_sev = accuracy_score(y_test_sev, dt_model_sev.predict(X_test_sev_scaled))

    rf_model_sev = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_sev.fit(X_train_sev_scaled, y_train_sev)
    pred_rf_sev = rf_model_sev.predict(X_test_sev_scaled)
    acc_rf_sev = accuracy_score(y_test_sev, pred_rf_sev)

    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression Accuracy", f"{acc_log_sev*100:.2f}%")
    col2.metric("Decision Tree Accuracy", f"{acc_dt_sev*100:.2f}%")
    col3.metric("Random Forest Accuracy", f"{acc_rf_sev*100:.2f}%")
    
    st.subheader("Detailed Analysis: Random Forest")
    colA, colB = st.columns(2)
    with colA:
        st.write("Classification Report")
        report_dict = classification_report(y_test_sev, pred_rf_sev, target_names=le_risk.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict).transpose())
    
    with colB:
        st.write("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        cm_rf = confusion_matrix(y_test_sev, pred_rf_sev)
        sns.heatmap(cm_rf, annot=True, cmap='Blues', fmt='d', xticklabels=le_risk.classes_, yticklabels=le_risk.classes_, ax=ax_cm)
        ax_cm.set_ylabel('Actual')
        ax_cm.set_xlabel('Predicted')
        st.pyplot(fig_cm)


# ==============================================================================
# SYSTEM 3: DEATH RATE PREDICTION SYSTEM
# ==============================================================================
elif selected_option == "3. Death Rate Prediction (Regression)":
    st.header("📉 System 3: Death Rate Prediction System")
    st.markdown("<div class='info-box'><b>Goal:</b> Use Linear Regression to predict the complex continuous target: 'Deaths / 100 Cases' (Death Percentage).<br><b>Features Used:</b> Confirmed cases, Active cases, and WHO Region.</div>", unsafe_allow_html=True)
    
    df_reg = df.copy()
    le_region = LabelEncoder()
    df_reg['WHO Region Encoded'] = le_region.fit_transform(df_reg['WHO Region'])
    
    X_reg = df_reg[['Confirmed', 'Active', 'WHO Region Encoded']]
    y_reg = df_reg['Deaths / 100 Cases']
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_reg_scaled, y_train_reg)
    y_pred_reg = lr_model.predict(X_test_reg_scaled)
    
    st.subheader("Model Evaluation")
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
    col2.metric("R-squared Score (R2)", f"{r2:.4f}")
    
    st.subheader("Actual vs Predicted Fit Plot")
    fig_fit, ax_fit = plt.subplots(figsize=(8, 5))
    ax_fit.scatter(y_test_reg, y_pred_reg, alpha=0.7, color='purple')
    ax_fit.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], color='red', lw=2, linestyle='--')
    ax_fit.set_xlabel("Actual Death Rate (%)")
    ax_fit.set_ylabel("Predicted Death Rate (%)")
    ax_fit.set_title("Linear Regression: Actual vs Predicted")
    st.pyplot(fig_fit)


# ==============================================================================
# SYSTEM 4: WHO REGION CLASSIFICATION
# ==============================================================================
elif selected_option == "4. WHO Region Classification":
    st.header("🌍 System 4: WHO Region Classification System")
    st.markdown("<div class='info-box'><b>Goal:</b> Try to classify the geographical WHO Region of a country strictly by looking at its COVID-19 numbers (Confirmed, Deaths, Recovered, Active).<br><b>Observation:</b> This is challenging due to the small dataset size (187 countries spread across many regions).</div>", unsafe_allow_html=True)
    
    X_who = df[['Confirmed', 'Deaths', 'Recovered', 'Active']]
    y_who = df['WHO Region']
    
    X_train_who, X_test_who, y_train_who, y_test_who = train_test_split(X_who, y_who, test_size=0.2, random_state=42)
    
    dt_who = DecisionTreeClassifier(random_state=42)
    dt_who.fit(X_train_who, y_train_who)
    pred_dt_who = dt_who.predict(X_test_who)
    acc_dt_who = accuracy_score(y_test_who, pred_dt_who)
    
    rf_who = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_who.fit(X_train_who, y_train_who)
    pred_rf_who = rf_who.predict(X_test_who)
    acc_rf_who = accuracy_score(y_test_who, pred_rf_who)
    
    st.subheader("Classification Accuracies")
    col1, col2 = st.columns(2)
    col1.metric("Decision Tree Accuracy", f"{acc_dt_who*100:.2f}%")
    col2.metric("Random Forest Accuracy", f"{acc_rf_who*100:.2f}%")
    
    st.subheader("Classification Report (Random Forest)")
    report_dict_who = classification_report(y_test_who, pred_rf_who, zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report_dict_who).transpose())


# ==============================================================================
# SYSTEM 5: FINAL MODEL COMPARISON
# ==============================================================================
elif selected_option == "5. Final Model Comparison":
    st.header("🏆 System 5: Comprehensive Model Comparison")
    st.markdown("<div class='info-box'><b>Goal:</b> Compare the accuracies of our Machine Learning models developed in the Severity Prediction task (System 2) to ascertain the best algorithm for this dataset.</div>", unsafe_allow_html=True)
    
    # Needs to recalculate accs since we didn't cache them globally
    df_sev2 = df.copy()
    def cl_sev(active):
        if active > 50000: return 'High Risk'
        elif active > 10000: return 'Medium Risk'
        return 'Low Risk'
    df_sev2['Risk'] = df_sev2['Active'].apply(cl_sev)
    
    X2 = df_sev2[['Confirmed', 'Deaths', 'Recovered', 'New cases', 'New deaths']]
    y2 = LabelEncoder().fit_transform(df_sev2['Risk'])
    
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
    scaler2 = StandardScaler()
    Xt2 = scaler2.fit_transform(X_train2)
    Xts2 = scaler2.transform(X_test2)
    
    acc_l = accuracy_score(y_test2, LogisticRegression(max_iter=1000).fit(Xt2, y_train2).predict(Xts2))
    acc_d = accuracy_score(y_test2, DecisionTreeClassifier(random_state=42).fit(Xt2, y_train2).predict(Xts2))
    acc_r = accuracy_score(y_test2, RandomForestClassifier(random_state=42).fit(Xt2, y_train2).predict(Xts2))
    
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    accuracies = [acc_l, acc_d, acc_r]
    
    st.subheader("Model Validation Accuracy Chart")
    fig_comp, ax_comp = plt.subplots(figsize=(9, 5))
    bars = sns.barplot(x=models, y=accuracies, palette='magma', ax=ax_comp)
    ax_comp.set_ylim(0, 1.1)
    ax_comp.set_ylabel("Accuracy Score")
    
    for i, v in enumerate(accuracies):
        ax_comp.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=12)
        
    st.pyplot(fig_comp)
    
    best_acc = max(accuracies)
    idx = accuracies.index(best_acc)
    
    st.success(f"**Conclusion:** The absolute best performing model across all testing was the **{models[idx]}** scoring **{best_acc*100:.2f}%** validation accuracy.")
