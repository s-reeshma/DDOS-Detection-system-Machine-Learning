import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------- SETUP ----------------
st.set_page_config(
    page_title="DDoS Shield AI",
    layout="wide"
)
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f8fbff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: grey !important;
}

/* Titles */
h1 {
    color: #1565c0 !important;
    font-weight: 700;
}

h2, h3 {
    color: #1976d2 !important;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background-color: white;
    border: 2px solid #2196f3;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(33,150,243,0.15);
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #bbdefb;
}

/* Success Box */
.stSuccess {
    background-color: #e3f2fd !important;
    color: #0d47a1 !important;
    border-left: 5px solid #2196f3;
}

/* Buttons */
.stButton > button {
    background-color: #1976d2;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #0d47a1;
    color: white;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #2196f3;
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)
# ---------------- LOAD MODEL ----------------
model = joblib.load("ddos_model.pkl")
features = joblib.load("features.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title(" Control Panel")

mode = st.sidebar.selectbox(
    "Mode",
    ["Demo Dataset", "Upload CSV"]
)

st.sidebar.info(
    " DDoS Detection System"
)

# ---------------- HEADER ----------------
st.title(" DDoS Shield AI Dashboard")
st.caption(
    "Machine Learning-Based Network Attack Detection"
)

# ---------------- DATA SOURCE ----------------
df = None

if mode == "Demo Dataset":

    # Better: create a smaller demo_data.csv later
    df = pd.read_csv("data/trafic.csv")

else:

    file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"]
    )

    if file:
        df = pd.read_csv(file)

if df is None:
    st.info("Select Demo Dataset or Upload CSV")
    st.stop()

# ---------------- PREVIEW ----------------
st.subheader(" Raw Data Preview")
st.dataframe(df.head())

# ---------------- CLEANING ----------------
df.columns = df.columns.str.strip()

df.replace(
    [np.inf, -np.inf],
    0,
    inplace=True
)

df.fillna(0, inplace=True)

# ---------------- FEATURES ----------------
X_input = df.drop(
    columns=["Label"],
    errors="ignore"
)

X_input = X_input.reindex(
    columns=features,
    fill_value=0
)

# ---------------- PREDICTIONS ----------------
pred = model.predict(X_input)

df["Prediction"] = pred

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Records",
    len(df)
)

col2.metric(
    "Attacks Detected",
    int(sum(pred))
)

col3.metric(
    "Benign Traffic",
    int(len(pred) - sum(pred))
)

# ---------------- TRAFFIC PIE CHART ----------------
st.subheader(" Traffic Distribution")

pie_df = pd.DataFrame({
    "Class": df["Prediction"].map({
        0: "BENIGN",
        1: "ATTACK"
    })
})

fig = px.pie(
    pie_df,
    names="Class",
    title="Network Traffic Breakdown",
    color="Class",
    color_discrete_map={
        "BENIGN": "#42a5f5",
        "ATTACK": "#0d47a1"
    }
)

st.plotly_chart(
    fig,
    width="stretch"
)

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("Top 10 Important Features")

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig2 = px.bar(
    importance_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Most Important Features",
    color="Importance",
    color_continuous_scale="Blues"
)

st.plotly_chart(
    fig2,
    width="stretch"
)

# ---------------- PREDICTION TABLE ----------------
st.subheader("Prediction Results")

display_df = df.copy()

display_df["Prediction"] = display_df["Prediction"].map({
    0: "BENIGN",
    1: "ATTACK"
})

st.dataframe(
    display_df.head(100),
    width="stretch"
)

# ---------------- SUMMARY ----------------
st.subheader(" Model Summary")

st.success(
    """
    Random Forest Classifier

    1 Trained on CICIDS DDoS Traffic Dataset

    2 Detects Benign vs Attack Traffic

    3 Uses Network Flow Features

    4 Explainability via Feature Importance Analysis
    """
)