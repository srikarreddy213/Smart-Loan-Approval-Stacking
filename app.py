import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# 🌸 SOFT, EYE-PLEASING THEME (PASTEL)
# --------------------------------------------------
st.set_page_config(page_title="Smart Loan Approval", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #f8fafc;
}

h1, h2, h3 {
    color: #334155;
}

section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
}

.stButton>button {
    background-color: #6366f1;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    background-color: #4f46e5;
}

.card {
    padding: 18px;
    border-radius: 14px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 1️⃣ TITLE & DESCRIPTION
# --------------------------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict "
    "whether a loan will be **approved or rejected** by combining multiple models "
    "for better decision making."
)

st.divider()

# --------------------------------------------------
# LOAD DATA (FROM REPO)
# --------------------------------------------------
df = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")

y = df["Loan_Status"].map({"Y": 1, "N": 0})
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)

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
# 2️⃣ INPUT SECTION (SIDEBAR)
# --------------------------------------------------
st.sidebar.header("📋 Applicant Details")

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
# 3️⃣ MODEL ARCHITECTURE DISPLAY
# --------------------------------------------------
st.subheader("🧠 Stacking Architecture")
st.markdown("""
- **Base Models**
  - Logistic Regression
  - Decision Tree
  - Random Forest  
- **Meta Model**
  - Logistic Regression
""")

st.divider()

# --------------------------------------------------
# 4️⃣ PREDICTION
# --------------------------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # ---------- Prediction logic (unchanged) ----------
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

    st.divider()

    # ---------- MAIN RESULT ----------
    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # ---------- BASE MODEL RESULTS ----------
    st.subheader("📊 Base Model Predictions")
    st.write(f"Logistic Regression → {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"Decision Tree → {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"Random Forest → {'Approved' if rf_pred else 'Rejected'}")

    # ---------- CONFIDENCE ----------
    st.subheader("📈 Confidence Score")
    st.progress(int(confidence))
    st.write(f"Model Confidence: **{confidence:.2f}%**")

    # ---------- BUSINESS EXPLANATION ----------
    st.subheader("🧠 Business Explanation")
    st.info(
        f"""
        Based on the applicant's income, credit history, employment status,
        and combined predictions from multiple models,
        the applicant is **{'likely' if final_pred else 'unlikely'}**
        to repay the loan.

        Therefore, the stacking model recommends
        **loan {'approval' if final_pred else 'rejection'}**.
        """
    )


    # --------------------------------------------------
    # 5️⃣ OUTPUT SECTION (DYNAMIC COLORS)
    # --------------------------------------------------
    if final_pred == 1:
        bg = "#ecfdf5"
        border = "#22c55e"
        text = "✅ Loan Approved"
        color = "#15803d"
    else:
        bg = "#fef2f2"
        border = "#ef4444"
        text = "❌ Loan Rejected"
        color = "#b91c1c"

    st.markdown(
        f"""
        <div class="card" style="background:{bg}; border-left:6px solid {border};">
            <h2 style="color:{color};">{text}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📊 Base Model Predictions")
    st.write(f"• Logistic Regression → {'Approved' if lr_pred else 'Rejected'}")
    st.write(f"• Decision Tree → {'Approved' if dt_pred else 'Rejected'}")
    st.write(f"• Random Forest → {'Approved' if rf_pred else 'Rejected'}")

    st.subheader("📈 Confidence")
    st.progress(int(confidence))
    st.write(f"Model Confidence: **{confidence:.2f}%**")

    # --------------------------------------------------
    # 6️⃣ BUSINESS EXPLANATION
    # --------------------------------------------------
    st.markdown(
        f"""
        <div class="card" style="background:#f1f5f9; border-left:5px solid #6366f1;">
        Based on applicant income, credit history, employment status, and
        combined predictions from multiple models, the applicant is
        <b>{"likely" if final_pred else "unlikely"}</b> to repay the loan.
        <br><br>
        Therefore, the stacking model recommends
        <b>{"loan approval" if final_pred else "loan rejection"}</b>.
        </div>
        """,
        unsafe_allow_html=True
    )
