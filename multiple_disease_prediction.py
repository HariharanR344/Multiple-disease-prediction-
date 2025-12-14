import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

st.set_page_config(page_title="Multiple Disease Prediction", layout="centered")

st.title("Multiple Disease Prediction System")

# Load Models

with open("parkinsons_model.pkl", "rb") as f:
    parkinsons_model = pickle.load(f)

with open("kidney_model.pkl", "rb") as f:
    kidney_model = pickle.load(f)

with open("liver_model.pkl", "rb") as f:
    liver_model = pickle.load(f)

# Parkinson’s Scaler

parkinsons_df = pd.read_csv(
    "https://raw.githubusercontent.com/HariharanR344/Multiple-disease-prediction-/refs/heads/main/parkinsons%20-%20parkinsons.csv"
)

parkinsons_features = [
    'PPE','spread1','spread2','D2','RPDE','DFA','HNR',
    'MDVP:Jitter(%)','MDVP:Shimmer','NHR'
]

X = parkinsons_df[parkinsons_features]
y = parkinsons_df['status']

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y
)

rus = RandomUnderSampler()
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

parkinsons_scaler = MinMaxScaler()
parkinsons_scaler.fit(X_train_rus)

# Kidney Scaler

kidney_df = pd.read_csv(
    "https://raw.githubusercontent.com/HariharanR344/Multiple-disease-prediction-/refs/heads/main/kidney_disease%20-%20kidney_disease.csv"
)

cols_to_numeric = ['pcv', 'wc', 'rc']
for col in cols_to_numeric:
    kidney_df[col] = kidney_df[col].replace('?', np.nan)
    kidney_df[col] = pd.to_numeric(kidney_df[col], errors='coerce')

num_cols = kidney_df.select_dtypes(include=['int64','float64']).columns
kidney_df[num_cols] = kidney_df[num_cols].fillna(kidney_df[num_cols].median())

cat_cols = kidney_df.select_dtypes(include='object').columns
kidney_df[cat_cols] = kidney_df[cat_cols].fillna(kidney_df[cat_cols].mode().iloc[0])

le = LabelEncoder()
kidney_df['rbc_encoded'] = le.fit_transform(kidney_df['rbc'])
kidney_df['htn_encoded'] = le.fit_transform(kidney_df['htn'])
kidney_df['dm_encoded'] = le.fit_transform(kidney_df['dm'])
kidney_df['classification_encoded'] = le.fit_transform(kidney_df['classification'])

kidney_features = [
    'sc','bu','al','hemo','pcv','sg',
    'rbc_encoded','htn_encoded','dm_encoded','bgr'
]

A = kidney_df[kidney_features]
B = kidney_df['classification_encoded']

A_train, _, B_train, _ = train_test_split(
    A, B, test_size=0.2, random_state=4, stratify=B
)

rus = RandomUnderSampler()
A_train_rus, B_train_rus = rus.fit_resample(A_train, B_train)

kidney_scaler = MinMaxScaler()
kidney_scaler.fit(A_train_rus)

# Liver Scaler

with open("scaler (1).pkl", "rb") as f:
    liver_scaler = pickle.load(f)

# Sidebar Navigation

menu = st.sidebar.selectbox(
    "Select Disease",
    ["Parkinson's Disease", "Kidney Disease", "Liver Disease"]
)

# Parkinson’s Prediction

if menu == "Parkinson's Disease":
    st.subheader("Parkinson’s Disease Prediction")

    inputs = [st.number_input(f, format="%.6f") for f in parkinsons_features]

    if st.button("Predict Parkinson’s"):
        features = np.array(inputs).reshape(1, -1)
        features_scaled = parkinsons_scaler.transform(features)
        prediction = parkinsons_model.predict(features_scaled)[0]

        st.success(
            "Parkinson's Disease Detected" if prediction == 1 else "No Parkinson's Disease"
        )

# Kidney Prediction

elif menu == "Kidney Disease":
    st.subheader("Kidney Disease Prediction")

    sc = st.number_input("Serum Creatinine(sc)")
    bu = st.number_input("Blood Urea(bu)")
    al = st.number_input("Albumin(al)")
    hemo = st.number_input("Hemoglobin(hemo)")
    pcv = st.number_input("Packed Cell Volume(pcv)")
    sg = st.number_input("Specific Gravity(sg)")
    rbc = st.radio("Red Blood Cells(rbc)", ["normal", "abnormal"])
    htn = st.radio("Hypertension(htn)", ["yes", "no"])
    dm = st.radio("Diabetes Mellitus(dm)", ["yes", "no"])
    bgr = st.number_input("Blood Glucose Random(bgr)")

    if st.button("Predict Kidney Disease"):
        rbc_encoded = 1 if rbc == "normal" else 0
        htn_encoded = 1 if htn == "yes" else 0
        dm_encoded = 1 if dm == "yes" else 0

        features = np.array([
            sc, bu, al, hemo, pcv, sg,
            rbc_encoded, htn_encoded, dm_encoded, bgr
        ]).reshape(1, -1)

        features_scaled = kidney_scaler.transform(features)
        prediction = kidney_model.predict(features_scaled)[0]

        st.success(
            "Kidney Disease Detected" if prediction == 0 else "No Kidney Disease"
        )

# Liver Prediction

else:
    st.subheader("Liver Disease Prediction")

    age = st.number_input("Age")
    gender = st.radio("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin")
    db = st.number_input("Direct Bilirubin")
    ap = st.number_input("Alkaline Phosphotase")
    alt = st.number_input("Alamine Aminotransferase")
    ast = st.number_input("Aspartate Aminotransferase")
    tp = st.number_input("Total Proteins")
    alb = st.number_input("Albumin")
    agr = st.number_input("Albumin and Globulin Ratio")

    if st.button("Predict Liver Disease"):
        gender_encoded = 1 if gender == "Male" else 0

        features = np.array([
            age, gender_encoded, tb, db, ap,
            alt, ast, tp, alb, agr
        ]).reshape(1, -1)

        features_scaled = liver_scaler.transform(features)
        prediction = liver_model.predict(features_scaled)[0]

        st.success(
            "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
        )
