import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Rekomendasi Tanaman", layout="centered")
st.title("🌾 Sistem Rekomendasi Tanaman")

MODEL_PATH = "crop_recommendation_model.pkl"

# Load atau latih model jika file tidak ditemukan
def train_model():
    df = pd.read_csv("Crop_recommendation.csv")
    X = df.drop("label", axis=1)
    y = df["label"]

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.warning("Model belum ditemukan. Melatih model baru...")
    model = train_model()

# Input user
st.subheader("Masukkan Nilai Parameter Tanah & Iklim:")
N = st.number_input("Nitrogen (N)", 0, 140)
P = st.number_input("Fosfor (P)", 0, 140)
K = st.number_input("Kalium (K)", 0, 200)
temperature = st.number_input("Temperatur (°C)", 0.0, 50.0)
humidity = st.number_input("Kelembaban (%)", 0.0, 100.0)
ph = st.number_input("pH Tanah", 0.0, 14.0)
rainfall = st.number_input("Curah Hujan (mm)", 0.0, 400.0)

# Prediksi
if st.button("🔍 Prediksi Tanaman yang Direkomendasikan"):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_features)[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_features)
        confidence = np.max(probabilities)
        st.success(f"🌱 Tanaman yang direkomendasikan: **{prediction.capitalize()}**")
        st.info(f"✅ Tingkat keyakinan model: {confidence:.2f} (maksimum = 1.0)")
        if confidence < 0.6:
            st.warning("⚠️ Model kurang yakin terhadap hasil ini. Coba masukkan data yang lebih akurat.")
    else:
        st.success(f"🌱 Tanaman yang direkomendasikan: **{prediction.capitalize()}**")
