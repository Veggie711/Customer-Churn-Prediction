import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("📊 Customer Churn Prediction App")

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

rf_model = load_model("rf_model.pkl")
xgb_model = load_model("xgb_model.pkl")
log_model = load_model("log_model.pkl")

models = {
    "Random Forest": rf_model,
    "Logistic Regression": log_model,
    "XGBoost": xgb_model
}

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction"])
data_mode = st.sidebar.radio("📥 Data Input Mode", ["Use Uploaded Data", "Manual Input"])
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
selected_model_name = st.sidebar.selectbox("🤖 Choose Model", list(models.keys()))

# -------------------------
# Data Loading and Preprocessing
# -------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df

def preprocess(df):
    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.drop(columns=['customerID'], errors='ignore', inplace=True)

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].replace({'Yes': 1, 'No': 0})

    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})

    df = pd.get_dummies(df, drop_first=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    return df

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.subheader("📘 Project Overview")
    st.markdown("""
    Welcome to the Customer Churn Prediction App. Here's what you can do:
    - Upload customer data
    - Choose from pre-trained ML models
    - Predict which customers are likely to churn

    **Tech Stack:** Streamlit, Scikit-learn, XGBoost  
    **Models Included:** Logistic Regression, Random Forest, XGBoost
    """)

# -------------------------
# Prediction Page
# -------------------------
elif page == "Prediction":
    st.subheader("📊 Churn Prediction")

    model = models[selected_model_name]

    if data_mode == "Use Uploaded Data":
        if uploaded_file:
            df = load_data(uploaded_file)
        else:
            df = pd.read_csv("final.csv")

        df_processed = preprocess(df)
        X = df_processed.drop(columns=['Churn'], errors='ignore')
        y = df_processed['Churn'] if 'Churn' in df_processed.columns else None

        if selected_model_name == "Logistic Regression":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        if st.sidebar.button("🔍 Predict"):
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)[:, 1]

            results = X.copy()
            results["Prediction"] = y_pred
            results["Probability"] = y_proba
            st.dataframe(results.head(10))

            if y is not None:
                st.text("📌 Classification Report:")
                st.text(classification_report(y, y_pred))

                st.write("📌 Confusion Matrix:")
                st.write(confusion_matrix(y, y_pred))

                st.write(f"📌 ROC AUC Score: {roc_auc_score(y, y_proba):.2f}")
                fpr, tpr, _ = roc_curve(y, y_proba)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label="AUC")
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title("ROC Curve")
                st.pyplot(fig)

    else:
        st.sidebar.header("🧾 Enter Customer Details")
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        senior = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
        dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
        tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.sidebar.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=0.5)
        total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, step=1.0)

        manual_input = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == "Yes" else 0,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        df_manual = pd.DataFrame([manual_input])
        df_manual = pd.get_dummies(df_manual)

        # Align columns to match model input
        df_full = pd.read_csv("final.csv")  # Fallback for column structure
        df_full = preprocess(df_full)
        all_features = df_full.drop(columns=['Churn'], errors='ignore').columns
        for col in all_features:
            if col not in df_manual.columns:
                df_manual[col] = 0
        df_manual = df_manual[all_features]

        if selected_model_name == "Logistic Regression":
            scaler = StandardScaler()
            df_manual = scaler.fit_transform(df_manual)

        if st.sidebar.button("🧮 Manual Predict"):
            result = model.predict(df_manual)[0]
            st.success("Prediction: **{}**".format("Churn" if result == 1 else "Not Churn"))
