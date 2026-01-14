import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ola Bike Ride Demand Forecast", layout="centered")

# ---------- Load Model & Scaler ----------

def load_artifacts():
    model = pickle.load(open("ola_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_artifacts()

# ---------- UI ----------
st.title("Ola Bike Ride Demand Forecasting System")
st.markdown("Predicts **future ride demand** using time and weather conditions.")

st.sidebar.header("Enter Ride & Weather Details")

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
month = st.sidebar.slider("Month", 1, 12, 6)
year = st.sidebar.slider("Year", 2018, 2026, 2024)

temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
windspeed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 60.0, 10.0)

holiday = st.sidebar.selectbox("Holiday", [0, 1])
workingday = st.sidebar.selectbox("Working Day", [0, 1])

# Rain proxy logic
rain = 1 if (humidity > 75 and windspeed > 20) else 0

# ---------- Prepare Input ----------
input_data = pd.DataFrame([[hour, day, month, year, temp, humidity, windspeed, holiday, workingday, rain]],
columns=['hour','day','month','year','temp','humidity','windspeed','holiday','workingday','rain'])

scaled_input = scaler.transform(input_data)

# ---------- Prediction ----------
if st.button("Predict Ride Demand"):
    try:
        prediction = model.predict(scaled_input)
        st.success(f"Expected Number of Ride Requests: **{int(prediction[0])}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------- Visualization ----------
st.markdown("###  Historical Ride Demand Pattern")

data = pd.read_csv("ola.csv")

# Detect datetime column
datetime_col = None
for col in data.columns:
    if "date" in col.lower() or "time" in col.lower():
        datetime_col = col
        break

if datetime_col is None:
    st.error("No datetime column found in dataset.")
elif "count" not in data.columns:
    st.error("Dataset must contain a 'count' column.")
else:
    data['datetime'] = pd.to_datetime(data[datetime_col], errors='coerce')
    data['hour'] = data['datetime'].dt.hour

    hourly_avg = data.groupby("hour")["count"].mean().reset_index()

    fig, ax = plt.subplots()
    ax.plot(hourly_avg["hour"], hourly_avg["count"])
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Ride Requests")
    st.pyplot(fig)

st.info("This system helps Ola optimize fleet allocation and reduce passenger wait time.")