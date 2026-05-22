import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Page title
st.title("🧠 Alzheimer's Disease Prediction")

# Load dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Clean data
df = df.dropna()

# Remove unnecessary columns
drop_cols = []

if "PatientID" in df.columns:
    drop_cols.append("PatientID")

if "DoctorInCharge" in df.columns:
    drop_cols.append("DoctorInCharge")

df = df.drop(columns=drop_cols)

# Features and target
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

# Convert text columns
X = pd.get_dummies(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Scale
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

# Output
st.success("Model trained successfully!")

st.subheader("Model Accuracy")

st.write(f"{round(accuracy*100,2)} %")

# Show dataset
st.subheader("Dataset Preview")

st.dataframe(df.head())

# Prediction count
st.subheader("Prediction Summary")

st.write("Positive Predictions:", sum(pred))

st.write("Total Tested:", len(pred))
