import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.title("Alzheimer Disease Prediction")

uploaded = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded:

    df = pd.read_csv(uploaded)

    df = df.dropna()

    drop_cols = []

    if "PatientID" in df.columns:
        drop_cols.append("PatientID")

    if "DoctorInCharge" in df.columns:
        drop_cols.append("DoctorInCharge")

    df = df.drop(columns=drop_cols)

    X = df.drop(columns=["Diagnosis"])
    y = df["Diagnosis"]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    st.success("Model trained successfully!")

    st.write("Rows:", len(df))
