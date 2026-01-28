import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# PAGE CONFIG (IMPORTANT: FIRST LINE)
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# SOFT, EYE-PLEASING THEME (NO DARK / NO FINTECH)
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f9fafb;
}

section[data-testid="stSidebar"] {
    background-color: #eef2f7;
}

h1, h2, h3, h4 {
    color: #1f2937;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    margin-top: 16px;
}

.success {
    background-color: #ecfdf5;
    border-left: 6px solid #22c55e;
}

.error {
    background-color: #fef2f2;
    border-left: 6px solid #ef4444;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE & DESCRIPTION
# --------------------------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This application predicts **loan approval or rejection** using a "
    "**Stacking Ensemble Machine Learning model** that combines multiple models "
    "for better decision making."
)

# --------------------------------------------------
# LOAD DATA (FROM REPO)
# --------------------------------------------------
df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")

y = df["Loan_Status"].map({"Y": 1, "N": 0})
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)

# Preprocessing
X["Dependents"] = X["Dependents"].fillna(X["Dependents"].mode()[0])
X["Dependents"] = X["Dependents"].replace("3+", 3).astype(int)

for col in ["LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    X[col] = X[col].fillna(X[col].median())

for col in ["Gender", "Married", "Self_Employed"]:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# MODELS
# --------------------------------------------------
lr = LogisticRegression(max_iter=500)
dt = DecisionTreeClassifier(max_depth=3)
rf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)

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

meta_model = LogisticRegression()
meta_model.fit(meta_train, y_train)

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

credit_val = 1 if credit == "Yes" else 0
self_emp_val = 1 if employment == "Self-Employed" else 0

# --------------------------------------------------
# MODEL ARCHITECTURE (CLEAN)
# --------------------------------------------------
st.subheader("Model Architecture")
st.write("""
**Base Models**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model**
- Logistic Regression
""")

# --------------------------------------------------
# PREDICTION BUTTON & OUTPUT
# --------------------------------------------------
if st.button("Check Loan Eligibility (Stacking Model)"):

    user_input = np.array([
        app_income,
        co_income,
        loan_amount,
        loan_term,
        credit_val,
        self_emp_val
    ]).reshape(1, -1)

    padded = np.hstack([
        user_input,
        np.zeros((1, X_train.shape[1] - user_input.shape[1]))
    ])

    user_scaled = scaler.transform(padded)

    lr_pred = lr.predict(user_scaled)[0]
    dt_pred = dt.predict(user_scaled)[0]
    rf_pred = rf.predict(user_scaled)[0]

    meta_input = np.array([[lr_pred, dt_pred, rf_pred]])
    final_pred = meta_model.predict(meta_input)[0]
    confidence = meta_model.predict_proba(meta_input)[0][final_pred] * 100

    # ---------------- RESULT ----------------
    if final_pred == 1:
        st.markdown(
            "<div class='result-card success'><h3>✅ Loan Approved</h3></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-card error'><h3>❌ Loan Rejected</h3></div>",
            unsafe_allow_html=True
        )

    # ---------------- DETAILS ----------------
    st.subheader("Base Model Predictions")
    st.write(f"Logistic Regression: {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"Decision Tree: {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"Random Forest: {'Approved' if rf_pred else 'Rejected'}")

    st.subheader("Confidence Score")
    st.progress(int(confidence))
    st.write(f"{confidence:.2f}% confidence")

    st.subheader("Business Explanation")
    st.info(
        f"The applicant is **{'likely' if final_pred else 'unlikely'}** to repay the loan "
        f"based on income, credit history, and combined predictions from multiple models. "
        f"The stacking model therefore recommends **{'approval' if final_pred else 'rejection'}**."
    )
