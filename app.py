import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# 1️⃣ App Title & Description
# ---------------------------------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="wide")

st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    """
    This system uses a **Stacking Ensemble Machine Learning model**
    to predict whether a loan will be **approved or rejected** by combining
    multiple ML models for better decision making.
    """
)

st.divider()

# ---------------------------------------------------
# Load & Prepare Dataset
# ---------------------------------------------------
df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")

# Target encoding
y = df["Loan_Status"].map({"Y": 1, "N": 0})

# Drop unused columns
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)

# Handle Dependents
X["Dependents"] = X["Dependents"].fillna(X["Dependents"].mode()[0])
X["Dependents"] = X["Dependents"].replace("3+", 3).astype(int)

# Fill missing values
for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    X[col] = X[col].fillna(X[col].median())

for col in ["Gender", "Married", "Self_Employed"]:
    X[col] = X[col].fillna(X[col].mode()[0])

# Encoding
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# Base Models
# ---------------------------------------------------
lr = LogisticRegression(max_iter=500)
dt = DecisionTreeClassifier(max_depth=3)
rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)

# Meta-features (NO DATA LEAKAGE)
meta_train = np.column_stack([
    cross_val_predict(lr, X_train, y_train, cv=5),
    cross_val_predict(dt, X_train, y_train, cv=5),
    cross_val_predict(rf, X_train, y_train, cv=5)
])

meta_test = np.column_stack([
    lr.fit(X_train, y_train).predict(X_test),
    dt.fit(X_train, y_train).predict(X_test),
    rf.fit(X_train, y_train).predict(X_test)
])

# Meta-model
meta_model = LogisticRegression()
meta_model.fit(meta_train, y_train)

# ---------------------------------------------------
# 2️⃣ Input Section (Sidebar)
# ---------------------------------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
self_employed = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

credit_history_val = 1 if credit_history == "Yes" else 0
self_emp_val = 1 if self_employed == "Self-Employed" else 0

# ---------------------------------------------------
# 3️⃣ Model Architecture Display
# ---------------------------------------------------
st.subheader("🧠 Model Architecture (Stacking)")

st.markdown("""
**Base Models Used:**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model Used:**
- Logistic Regression  

*The meta-model learns how to combine base model predictions optimally.*
""")

st.divider()

# ---------------------------------------------------
# 4️⃣ Prediction Button
# ---------------------------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    input_data = np.array([
        app_income,
        coapp_income,
        loan_amount,
        loan_term,
        credit_history_val,
        self_emp_val
    ]).reshape(1, -1)

    input_scaled = scaler.transform(
        np.hstack([input_data, np.zeros((1, X_train.shape[1] - input_data.shape[1]))])
    )

    # Base predictions
    lr_pred = lr.predict(input_scaled)[0]
    dt_pred = dt.predict(input_scaled)[0]
    rf_pred = rf.predict(input_scaled)[0]

    meta_input = np.array([[lr_pred, dt_pred, rf_pred]])
    final_pred = meta_model.predict(meta_input)[0]
    confidence = meta_model.predict_proba(meta_input)[0][final_pred] * 100

    # ---------------------------------------------------
    # 5️⃣ Output Section
    # ---------------------------------------------------
    st.subheader("📊 Prediction Results")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 🔍 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred else 'Rejected'}")

    st.markdown(f"### 🧠 Final Stacking Decision: **{'Approved' if final_pred else 'Rejected'}**")
    st.markdown(f"### 📈 Confidence Score: **{confidence:.2f}%**")

    # ---------------------------------------------------
    # 6️⃣ Business Explanation
    # ---------------------------------------------------
    st.info(
        f"""
        Based on the applicant's income, credit history, employment status,
        and combined predictions from multiple machine learning models,
        the system predicts that the applicant is
        **{'likely' if final_pred else 'unlikely'} to repay the loan**.
        Therefore, the stacking model recommends **loan {'approval' if final_pred else 'rejection'}**.
        """
    )
