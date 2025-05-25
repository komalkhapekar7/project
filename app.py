
import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Live Reliance Stock Forecast")

@st.cache_data
def load_data():
    df = yf.download("RELIANCE.NS", start="2020-01-01")
    df = df.reset_index()
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

df = load_data()

st.write("Latest data sample")
st.write(df.tail())

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig, ax = plt.subplots()
ax.plot(df['ds'], df['y'], label='Actual')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
ax.legend()
st.pyplot(fig)
