# =========================================================
# IMPORT
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# =========================================================
# LOAD MODEL & DATA
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessing.pkl")
    return model, preprocessor

model, preprocessor = load_model()

@st.cache_data
def load_data():
    return pd.read_csv("data/external/telco_customer_churn.csv")

df = load_data()

# =========================================================
# PREPARE DATA (GLOBAL)
# =========================================================
X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Dashboard EDA",
        "Prediksi Churn",
        "Evaluasi Model",
        "Dokumentasi"
    ]
)

# =========================================================
# DASHBOARD EDA
# =========================================================
if page == "Dashboard EDA":
    st.title("Dashboard Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        churn_dist = df["Churn"].value_counts().reset_index()
        churn_dist.columns = ["Churn", "Count"]
        fig = px.bar(
            churn_dist,
            x="Churn",
            y="Count",
            title="Distribusi Customer Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            title="Monthly Charges vs Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn berdasarkan Jenis Kontrak")
    contract_churn = (
        df.groupby(["Contract", "Churn"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        contract_churn,
        x="Contract",
        y="Count",
        color="Churn",
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# PREDIKSI CHURN
# =========================================================
elif page == "Prediksi Churn":
    st.title("🔮 Prediksi Customer Churn")
    st.write("Masukkan data pelanggan:")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (bulan)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    if st.button("Prediksi"):
    # ==================================
    # AMBIL TEMPLATE DATA ASLI 
    # ==================================
        input_df = X.sample(1, random_state=42).copy()

    # ==================================
    # TIMPA DENGAN INPUT USER
    # ==================================
        input_df["tenure"] = tenure
        input_df["MonthlyCharges"] = monthly_charges
        input_df["Contract"] = contract
        input_df["InternetService"] = internet_service
        input_df["PaymentMethod"] = payment_method
    # ==================================
    # PREDIKSI (PIPELINE)
    # ==================================
        elif page == "Prediksi Churn":
    st.title("🔮 Prediksi Customer Churn")

    st.write("Masukkan data pelanggan di bawah ini:")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (bulan)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # ================================
    # TOMBOL PREDIKSI 
    # ================================
    if st.button("Prediksi"):
        # AMBIL 1 DATA ASLI SEBAGAI TEMPLATE
        input_df = X.sample(1, random_state=42).copy()

        # TIMPA DENGAN INPUT USER
        input_df["tenure"] = tenure
        input_df["MonthlyCharges"] = monthly_charges
        input_df["Contract"] = contract
        input_df["InternetService"] = internet_service
        input_df["PaymentMethod"] = payment_method

        # PREDIKSI (MODEL PIPELINE)
        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.subheader("📌 Hasil Prediksi")

        if prediction == 1:
            st.error(f"⚠️ Pelanggan diprediksi **CHURN** (Probabilitas: {prob:.2f})")
        else:
            st.success(f"✅ Pelanggan diprediksi **TIDAK CHURN** (Probabilitas: {prob:.2f})")

    # ==================================
    # OUTPUT
    # ==================================
    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error(f"Pelanggan diprediksi **CHURN** (Probabilitas: {prob:.2f})")
    else:
        st.success(f"Pelanggan diprediksi **TIDAK CHURN** (Probabilitas: {prob:.2f})")

# =========================================================
# EVALUASI MODEL
# =========================================================
elif page == "Evaluasi Model":
    st.title("Evaluasi Model")

    # LANGSUNG KE MODEL (pipeline)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y, y_pred):.3f}")

    st.metric("F1-Score", f"{f1_score(y, y_pred):.3f}")
    st.metric("ROC-AUC", f"{roc_auc_score(y, y_prob):.3f}")

    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    st.subheader("Confusion Matrix")
    st.dataframe(cm_df)


# =========================================================
# DOKUMENTASI
# =========================================================
elif page == "Dokumentasi":
    st.title("Dokumentasi Proyek")

    st.markdown("""
    ### Deskripsi
    Aplikasi ini memprediksi kemungkinan customer churn
    menggunakan Machine Learning.

    ### Dataset
    - Telco Customer Churn (Kaggle)
    - 7.043 data pelanggan

    ### Model
    - Logistic Regression
    - Random Forest
    - **XGBoost (Model Terbaik)**

    ### Fitur
    - Dashboard EDA
    - Prediksi Churn
    - Evaluasi Model
    - Confusion Matrix
    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Capstone Project Data Mining | Naufal Ferdiansyah")
