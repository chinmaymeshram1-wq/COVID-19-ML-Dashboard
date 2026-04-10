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

# --- PREMIUM DARK MODE STYLE FOR MATPLOTLIB/SEABORN ---
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#121212", "figure.facecolor": "none", "axes.grid": False, "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

# --- CUSTOM CSS (ANIMATIONS & GLASSMORPHISM) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Apply font safely without breaking Material icon ligatures */
    h1, h2, h3, h4, h5, h6, .main-title, [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    p, li, label, .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }

    h1, h2, h3, h4, h5, h6, p, li, label, span, .stMarkdown, .info-box {
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.8), 0 0 10px rgba(0,0,0,0.5); /* Needed so text is readable over white background */
    }

    /* Animated Dynamic Gradient Background - White & Black Theme */
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .stApp {
        background: linear-gradient(-45deg, #000000, #333333, #ffffff, #0a0a0a);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #ffffff;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 3.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.8), 0 0 15px rgba(255,255,255,0.3);
        background: -webkit-linear-gradient(45deg, #ffffff, #888888, #aaaaaa, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInDown 1.5s cubic-bezier(0.2, 0.8, 0.2, 1);
        padding: 20px 0;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* Glassmorphism Info Box */
    .info-box {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 25px;
        border-radius: 16px;
        border-left: 6px solid #00f2fe;
        color: #ffffff;
        font-size: 1.1rem;
        transition: all 0.4s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .info-box:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 15px 30px rgba(0, 242, 254, 0.15);
        border-left: 6px solid #f093fb;
        background: rgba(255, 255, 255, 0.05);
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 30, 0.6) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Professional Transparent Pill Navigation */
    [data-testid="stSidebar"] ul {
        list-style: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="stSidebar"] li {
        margin-bottom: 12px;
    }
    
    [data-testid="stSidebar"] a {
        text-decoration: none !important;
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1.05rem;
        display: block;
        padding: 12px 20px;
        border-radius: 30px; /* Circular professional pill */
        background: rgba(40, 40, 40, 0.8); /* Dark Glassmorphism */
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
        text-shadow: none !important; /* Keep links clean */
    }
    
    [data-testid="stSidebar"] a:hover {
        background: linear-gradient(90deg, #ffffff, #d4d4d4);
        color: #000000 !important;
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 8px 20px rgba(255, 255, 255, 0.5);
        border-color: transparent;
    }

    /* Click click animation with circle effect */
    [data-testid="stSidebar"] a:active {
        transform: translateX(4px) scale(0.92);
        border-radius: 50px;
        background: radial-gradient(circle, #ffffff 0%, #888888 100%);
        color: #000000 !important;
        box-shadow: 0 0 25px rgba(255,255,255,0.9);
        transition: all 0.1s ease;
    }

    /* Metric Cards Styling */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.4);
        animation: pulse 2s infinite alternate;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    @keyframes pulse {
        from { text-shadow: 0 0 10px rgba(255,255,255,0.3); }
        to { text-shadow: 0 0 20px rgba(255,255,255,0.8), 0 0 30px rgba(255,255,255,0.5); }
    }

    /* Streamlit Expander overrides & animations */
    .streamlit-expanderHeader {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        transition: background 0.3s ease !important;
    }
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: #00f2fe !important;
    }
    
    /* General element entrance animation */
    .element-container {
        animation: fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Dataframe wrapper styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
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


# --- SIDEBAR NAVIGATION (Anchor Links) ---
st.sidebar.markdown("### 🧭 Navigation & Access")
st.sidebar.markdown("""
<div class='info-box' style='padding:15px; font-size:0.9rem;'>
✨ <b>Tip:</b> Click on any section below to smoothly auto-navigate directly to the expanded data analysis.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
- [📊 Dataset Overview](#dataset-overview)
- [📈 1. COVID Trend Analysis Dashboard](#sys1)
- [🚦 2. Severity Prediction](#sys2)
- [📉 3. Death Rate Prediction](#sys3)
- [🌍 4. WHO Region Classification](#sys4)
- [🏆 5. Final Model Comparison](#sys5)
""", unsafe_allow_html=True)

st.sidebar.markdown("<br><br><br><br><hr>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='text-align:center; color:#666; font-size:0.8rem;'>Analysis Engine Active<br>v2.0 Premium Edition</div>", unsafe_allow_html=True)

# ==============================================================================
# DATASET OVERVIEW (ALWAYS ACTIVE & EXPANDED)
# ==============================================================================
st.header("📊 Dataset Overview", anchor="dataset-overview")
with st.expander("Explore the Raw Dataset & Statistics", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("🌍 Total Countries/Regions", df.shape[0])
    col2.metric("🔢 Total Features", df.shape[1])
    col3.metric("🦠 Total Global Cases", f"{df['Confirmed'].sum():,}")

    st.markdown("#### The Data (Snapshot)")
    st.dataframe(df.head(10), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Feature Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
    with colB:
        st.markdown("#### Missing Values Check")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ Clean Data: No missing values found in the dataset! Perfect!")
        else:
            st.write(missing_data[missing_data > 0])


st.divider()

# ==============================================================================
# SYSTEM 1: COVID TREND ANALYSIS DASHBOARD
# ==============================================================================
st.header("📈 System 1: COVID Trend Analysis & EDA", anchor="sys1")
with st.expander("Expand System 1: Exploratory Data Analysis", expanded=True):
    st.markdown("<div class='info-box'><b>Goal:</b> Find trends and visualize the underlying structure of the pandemic data globally.<br><br><b>🧠 ML Operation Used:</b> Principal Component Analysis (PCA) - Unsupervised Dimensionality Reduction.<br><b>💡 Why it's used:</b> PCA compresses multi-dimensional data into fewer principal components (2D space) while retaining the most variance, allowing us to visualize complex relationships easily.<br><b>🌍 Real-world Uses:</b> Image compression, exploratory data visualization, and noise filtering.</div>", unsafe_allow_html=True)

    st.subheader("Correlation Heatmap")
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    # Use dark palette for premium syntax
    sns.heatmap(numeric_df.corr(), annot=True, cmap='mako', fmt=".2f", ax=ax_corr, 
                linewidths=0.5, linecolor='black')
    st.pyplot(fig_corr)

    st.subheader("Global Case Distributions")
    colA, colB = st.columns(2)
    with colA:
        fig_hist1, ax_hist1 = plt.subplots()
        sns.histplot(df['Confirmed'], bins=30, kde=True, color='#00f2fe', ax=ax_hist1)
        ax_hist1.set_title("Distribution of Confirmed Cases", color="white")
        st.pyplot(fig_hist1)
    with colB:
        fig_hist2, ax_hist2 = plt.subplots()
        sns.histplot(df['Deaths'], bins=30, kde=True, color='#f5576c', ax=ax_hist2)
        ax_hist2.set_title("Distribution of Deaths", color="white")
        st.pyplot(fig_hist2)

    st.subheader("PCA: 2D Spatial Reduction")
    st.write("Compressing multi-dimensional features into two principal components to easily visualize WHO Region clusters on a 2D space.")
    features_pca = ['Confirmed', 'Deaths', 'Recovered', 'Active']
    X_pca = df[features_pca]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    pca = PCA(n_components=2)
    X_pca_2d = pca.fit_transform(X_scaled)

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=df['WHO Region'], palette='Set2', s=120, alpha=0.9, edgecolor='w', linewidth=0.5, ax=ax_pca)
    ax_pca.set_title("PCA 2D Visualization by WHO Region", color="white", size=14)
    ax_pca.set_xlabel("Principal Component 1")
    ax_pca.set_ylabel("Principal Component 2")
    
    # Legend styling
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)
    st.pyplot(fig_pca)

st.divider()


# ==============================================================================
# SYSTEM 2: SEVERITY PREDICTION SYSTEM
# ==============================================================================
st.header("🚦 System 2: COVID-19 Severity Prediction System", anchor="sys2")
with st.expander("Expand System 2: Classification Models", expanded=True):
    st.markdown("<div class='info-box'><b>Goal:</b> Classify pandemic severity into 'Low', 'Medium', or 'High' Risk.<br><br><b>🧠 ML Operation Used:</b> Multi-class Classification (Supervised Learning) via Logistic Regression, Decision Tree, & Random Forest.<br><b>💡 Why it's used:</b> We have labeled discrete risk categories and want the model to learn the boundaries between them based on numerical infection metrics.<br><b>🌍 Real-world Uses:</b> Medical disease diagnosis, spam email filtering, and sentiment analysis.</div>", unsafe_allow_html=True)

    def classify_severity(active_cases):
        if active_cases > 50000:
            return 'High Risk'
        elif active_cases > 10000:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    df_sev = df.copy()
    df_sev['Risk_Level'] = df_sev['Active'].apply(classify_severity)

    col1_v, col2_v = st.columns([1, 2])
    with col1_v:
        st.subheader("Target Distribution")
        fig_sev, ax_sev = plt.subplots(figsize=(6, 5))
        sns.countplot(x='Risk_Level', data=df_sev, order=['Low Risk', 'Medium Risk', 'High Risk'], palette='cool', ax=ax_sev)
        ax_sev.set_title("Risk Levels Generated", color="white")
        st.pyplot(fig_sev)
        
    with col2_v:
        st.subheader("Model Training & Evaluation")
        st.write("Using features: `Confirmed`, `Deaths`, `Recovered`, `New cases`, `New deaths`.")

        X_sev = df_sev[['Confirmed', 'Deaths', 'Recovered', 'New cases', 'New deaths']]
        y_sev = df_sev['Risk_Level']

        le_risk = LabelEncoder()
        y_sev_encoded = le_risk.fit_transform(y_sev)
        X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X_sev, y_sev_encoded, test_size=0.2, random_state=42)

        scaler_sev_ml = StandardScaler()
        X_train_sev_scaled = scaler_sev_ml.fit_transform(X_train_sev)
        X_test_sev_scaled = scaler_sev_ml.transform(X_test_sev)

        # 1. Log Reg
        log_model = LogisticRegression(max_iter=1000, random_state=42)
        log_model.fit(X_train_sev_scaled, y_train_sev)
        acc_log_sev = accuracy_score(y_test_sev, log_model.predict(X_test_sev_scaled))

        # 2. DT
        dt_model_sev = DecisionTreeClassifier(random_state=42)
        dt_model_sev.fit(X_train_sev_scaled, y_train_sev)
        acc_dt_sev = accuracy_score(y_test_sev, dt_model_sev.predict(X_test_sev_scaled))

        # 3. RF
        rf_model_sev = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model_sev.fit(X_train_sev_scaled, y_train_sev)
        pred_rf_sev = rf_model_sev.predict(X_test_sev_scaled)
        acc_rf_sev = accuracy_score(y_test_sev, pred_rf_sev)

        col1_s, col2_s, col3_s = st.columns(3)
        col1_s.metric("Logistic Regression", f"{acc_log_sev*100:.1f}%")
        col2_s.metric("Decision Tree", f"{acc_dt_sev*100:.1f}%")
        col3_s.metric("Random Forest", f"{acc_rf_sev*100:.1f}%")

    st.subheader("Zoom In: Random Forest Performance")
    colA_s, colB_s = st.columns(2)
    with colA_s:
        st.write("Classification Report")
        report_dict = classification_report(y_test_sev, pred_rf_sev, target_names=le_risk.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict).transpose().style.background_gradient(cmap='Blues'), use_container_width=True)

    with colB_s:
        st.write("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(6,4))
        cm_rf = confusion_matrix(y_test_sev, pred_rf_sev)
        sns.heatmap(cm_rf, annot=True, cmap='rocket', fmt='d', xticklabels=le_risk.classes_, yticklabels=le_risk.classes_, ax=ax_cm)
        ax_cm.set_ylabel('Actual', color='white')
        ax_cm.set_xlabel('Predicted', color='white')
        st.pyplot(fig_cm)


st.divider()

# ==============================================================================
# SYSTEM 3: DEATH RATE PREDICTION SYSTEM
# ==============================================================================
st.header("📉 System 3: Death Rate Prediction System", anchor="sys3")
with st.expander("Expand System 3: Regression Algorithms", expanded=True):
    st.markdown("<div class='info-box'><b>Goal:</b> Predict a continuous target metric like <i>'Deaths / 100 Cases'</i> based on case features.<br><br><b>🧠 ML Operation Used:</b> Linear Regression (Supervised Learning).<br><b>💡 Why it's used:</b> Regression is the standard approach for modeling the linear relationship between independent variables and forecasting a continuous numeric outcome.<br><b>🌍 Real-world Uses:</b> Predicting housing prices, sales forecasting, and financial risk estimation.</div>", unsafe_allow_html=True)

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

    col1_r, col2_r = st.columns(2)
    with col1_r:
        st.subheader("Model Validation")
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        
        st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
        st.metric("R-squared Score (R2)", f"{r2:.4f}")
        st.write("*(Note: Low R2 suggests death rate relies on more nuanced factors globally than just confirmed/active cases)*")

    with col2_r:
        st.subheader("Linear Regression Fit Visualization")
        fig_fit, ax_fit = plt.subplots(figsize=(8, 5))
        ax_fit.scatter(y_test_reg, y_pred_reg, alpha=0.8, color='#00f2fe', edgecolor='k', s=80)
        ax_fit.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], color='#f5576c', lw=3, linestyle='--')
        ax_fit.set_xlabel("Actual Death Rate (%)", color='white')
        ax_fit.set_ylabel("Predicted Death Rate (%)", color='white')
        ax_fit.set_title("Actual vs Predicted Fit", color='white')
        ax_fit.grid(True, linestyle=":", alpha=0.3)
        st.pyplot(fig_fit)


st.divider()

# ==============================================================================
# SYSTEM 4: WHO REGION CLASSIFICATION
# ==============================================================================
st.header("🌍 System 4: WHO Region Classification System", anchor="sys4")
with st.expander("Expand System 4: WHO Region Prediction", expanded=True):
    st.markdown("<div class='info-box'><b>Goal:</b> Predict the geographical WHO Region of a country based purely on its COVID-19 statistical footprint.<br><br><b>🧠 ML Operation Used:</b> Ensemble & Tree-based Classification.<br><b>💡 Why it's used:</b> Trees can handle non-linear data and find complex thresholds or splits in the data that might uniquely identify specific global region patterns.<br><b>🌍 Real-world Uses:</b> Customer market segmentation, fraud detection, and recommendation systems.</div>", unsafe_allow_html=True)

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

    col1_w, col2_w = st.columns(2)
    with col1_w:
        st.subheader("Test Method Accuracies")
        st.metric("Decision Tree Accuracy", f"{acc_dt_who*100:.2f}%")
        st.metric("Random Forest Accuracy", f"{acc_rf_who*100:.2f}%")

    with col2_w:
        st.subheader("Classification Reality Matrix (Random Forest)")
        report_dict_who = classification_report(y_test_who, pred_rf_who, zero_division=0, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict_who).transpose(), use_container_width=True)


st.divider()

# ==============================================================================
# SYSTEM 5: FINAL MODEL COMPARISON
# ==============================================================================
st.header("🏆 System 5: Comprehensive Model Comparison", anchor="sys5")
with st.expander("Expand System 5: Champion Evaluation", expanded=True):
    st.markdown("<div class='info-box'><b>Core Objective:</b> Ascertain the champion algorithm by comparing cross-model accuracies.<br><br><b>🧠 ML Operation Used:</b> Model Evaluation & Validation Metrics.<br><b>💡 Why it's used:</b> No single algorithm works best universally (No Free Lunch theorem). Evaluation quantifies accuracy so we select the most reliable predictor.<br><b>🌍 Real-world Uses:</b> A/B testing models in production, automated hyperparameter tuning, and reducing algorithmic bias.</div>", unsafe_allow_html=True)

    models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    accuracies = [acc_log_sev, acc_dt_sev, acc_rf_sev]

    st.subheader("Model Validation Accuracy Trophy Chart")
    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
    
    # Custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = sns.barplot(x=models, y=accuracies, palette='viridis', ax=ax_comp, edgecolor='white', linewidth=2)
    
    ax_comp.set_ylim(0, 1.1)
    ax_comp.set_ylabel("Accuracy Score", color="white")
    
    # Customize grid
    ax_comp.grid(axis='y', linestyle='--', alpha=0.3)

    for i, v in enumerate(accuracies):
        ax_comp.text(i, v + 0.03, f"{v*100:.1f}%", ha='center', fontweight='bold', fontsize=14, color='white')
        
    st.pyplot(fig_comp)

    best_acc = max(accuracies)
    idx = accuracies.index(best_acc)

    st.markdown(f"""
    <div style='background: linear-gradient(90deg, #11998e, #38ef7d); padding: 20px; border-radius: 12px; font-size: 1.3rem; text-align: center; color: white; font-weight: bold; box-shadow: 0 10px 20px rgba(0,0,0,0.2);'>
        🚀 Conclusion: The absolute champion model across all testing was the {models[idx]} scoring {best_acc*100:.2f}% accuracy.
    </div>
    """, unsafe_allow_html=True)
