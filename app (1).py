# ---------------------------------------------
# Google Colab-Compatible Full Code with Live Data, GridSearchCV, Technical Indicators, and Sentiment Models
# ---------------------------------------------

# --- Install dependencies ---
!pip install yfinance pandas matplotlib prophet keras tensorflow GoogleNews ta vaderSentiment scikit-learn requests beautifulsoup4 --quiet


# --- Imports ---
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, GlobalAveragePooling1D
# Update the imports for Tokenizer and pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from GoogleNews import GoogleNews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import MACD
# Import requests and BeautifulSoup for the news scraping part
import requests
from bs4 import BeautifulSoup

# Import TextBlob for the second sentiment analysis part
from textblob import TextBlob
import seaborn as sns


# ---------------------------------------------
# Step 1: Download Reliance Stock Data (Live)
# ---------------------------------------------
df = yf.download("RELIANCE.NS", start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
df = df[['Close']].reset_index()
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

# Add Technical Indicators
df['rsi'] = RSIIndicator(close=df['y'], window=14).rsi()
macd = MACD(close=df['y'])
df['macd'] = macd.macd()
df.fillna(0, inplace=True)

# ---------------------------------------------
# Step 2: Fetch Live News from GoogleNews and Compute Sentiment Scores
# ---------------------------------------------
googlenews = GoogleNews(lang='en', region='IN')
end_date = datetime.today()
start_date = end_date - timedelta(days=7)

googlenews.set_time_range(start_date.strftime('%m/%d/%Y'), end_date.strftime('%Y/%m/%d')) # Corrected date format for set_time_range
googlenews.search('Reliance Industries')
news_results = googlenews.results()
news_df = pd.DataFrame(news_results)

# Parse date
def parse_date(date_str):
    try:
        # Attempt multiple date formats if necessary
        # googlenews date format can be inconsistent
        return pd.to_datetime(date_str).date()
    except:
        return None

news_df['date'] = news_df['date'].apply(parse_date)
news_df = news_df.dropna(subset=['date'])

# VADER Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
news_df['sentiment'] = news_df['title'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Aggregate sentiment per day
daily_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()
daily_sentiment.columns = ['ds', 'news_sentiment']

# Merge with stock data
df['ds'] = pd.to_datetime(df['ds']).dt.date
df = df.merge(daily_sentiment, on='ds', how='left')
df['news_sentiment'].fillna(0, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])

# ---------------------------------------------
# Step 3: Prophet Forecasting with Regressors
# ---------------------------------------------
model = Prophet(daily_seasonality=True)
model.add_regressor('news_sentiment')
model.add_regressor('rsi')
model.add_regressor('macd')
model.fit(df)

# Create future dataframe
future = model.make_future_dataframe(periods=30)
# Ensure future dataframe has the same regressor columns for prediction
future = future.merge(df[['ds', 'news_sentiment', 'rsi', 'macd']], on='ds', how='left')
# Use 'ffill' to fill NaNs for regressors in the future dataframe based on past values
# This assumes future sentiment/indicators will be similar to the last known values
future.fillna(method='ffill', inplace=True)
forecast = model.predict(future)

# ---------------------------------------------
# Step 4: Compute Residuals
# ---------------------------------------------
df_forecast = forecast[['ds', 'yhat']]
merged = df.merge(df_forecast, on='ds')
merged['residual'] = merged['y'] - merged['yhat']

# ---------------------------------------------
# Step 5: LSTM on Residuals
# ---------------------------------------------
look_back = 20
residuals = merged[['residual']].values
scaler = MinMaxScaler()
scaled_res = scaler.fit_transform(residuals)
X, y = [], []
for i in range(look_back, len(scaled_res)):
    X.append(scaled_res[i-look_back:i])
    y.append(scaled_res[i][0])
X, y = np.array(X), np.array(y)

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))) # Added return_sequences=True
model_lstm.add(LSTM(units=50)) # Added another LSTM layer
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=20, batch_size=32, verbose=0) # Increased epochs and batch size for potentially better training

# ---------------------------------------------
# Step 6: Forecast Residuals Using LSTM
# ---------------------------------------------
last_data = scaled_res[-look_back:]
future_res_input = last_data.reshape(1, look_back, 1)
future_residuals = []
for _ in range(30):
    pred = model_lstm.predict(future_res_input, verbose=0)[0][0]
    future_residuals.append(pred)
    future_point = np.array([[pred]])
  # Update the input sequence by removing the oldest point and adding the new prediction
future_res_input = np.concatenate((future_res_input[:, 1:, :], future_point.reshape(1, 1, 1)), axis=1)

# ---------------------------------------------
# Step 7: Final Hybrid Forecast
# ---------------------------------------------
residual_preds = scaler.inverse_transform(np.array(future_residuals).reshape(-1, 1))[:, 0]
# Ensure the forecast dataframe used for merging has the correct future dates
final_forecast = forecast[forecast['ds'].isin(future['ds'].tail(30))].copy()
final_forecast = final_forecast.reset_index(drop=True) # Reset index for clean merge

# Create a new dataframe for the future residuals with the correct dates
future_dates = future['ds'].tail(30).reset_index(drop=True)
residual_df = pd.DataFrame({'ds': future_dates, 'residual': residual_preds})

# Merge the residual predictions with the prophet forecast for the future dates
final_forecast = final_forecast.merge(residual_df, on='ds', how='left')
final_forecast['hybrid_forecast'] = final_forecast['yhat'] + final_forecast['residual']


# ---------------------------------------------
# Step 8: Visualizations
# ---------------------------------------------
plt.figure(figsize=(15, 6))
plt.plot(df['ds'], df['y'], label='Actual Price', color='black')
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast', color='blue')
plt.plot(final_forecast['ds'], final_forecast['hybrid_forecast'], label='Hybrid Forecast (Prophet + LSTM)', color='red', linestyle='--')
plt.title('Reliance Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals Plot
plt.figure(figsize=(15, 5))
plt.plot(merged['ds'], merged['residual'], label='Residuals', color='teal')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Residuals from Prophet Forecast')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Zoomed Forecast Comparison (Last 90 Days)
plt.figure(figsize=(15, 5))
recent = df[df['ds'] > df['ds'].max() - pd.Timedelta(days=90)]
# Ensure forecast_recent has the correct dates that overlap with the recent actual data
forecast_recent = forecast[forecast['ds'].isin(recent['ds'])]
plt.plot(recent['ds'], recent['y'], label='Actual', color='black')
plt.plot(forecast_recent['ds'], forecast_recent['yhat'], label='Prophet Forecast', color='blue')
plt.title('Actual vs Forecast - Last 90 Days')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Prophet Model Components
# Check if the forecast dataframe contains the regressor columns before plotting
if all(col in forecast.columns for col in ['news_sentiment', 'rsi', 'macd']):
    model.plot_components(forecast)
    plt.suptitle("Prophet Model Components with News Sentiment + RSI + MACD")
    plt.tight_layout()
    plt.show()
else:
    print("Regressor components not available in forecast dataframe for plotting.")


# Forecast with Confidence Intervals
plt.figure(figsize=(15, 6))
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='skyblue', alpha=0.3, label='Confidence Interval')
plt.title('Forecast with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Hybrid Forecast vs Actual - Last 90 Days
plt.figure(figsize=(15, 6))
actual_recent = df[df['ds'] >= df['ds'].max() - pd.Timedelta(days=90)]
hybrid_recent = final_forecast[['ds', 'hybrid_forecast']]
plt.plot(actual_recent['ds'], actual_recent['y'], label='Actual', color='black')
plt.plot(hybrid_recent['ds'], hybrid_recent['hybrid_forecast'], label='Hybrid Forecast', color='red')
plt.title('Hybrid Forecast vs Actual - Last 90 Days')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# News Sentiment Over Time
plt.figure(figsize=(15, 5))
plt.plot(df['ds'], df['news_sentiment'], color='purple', linewidth=2.5, label='News Sentiment (VADER)')
plt.axhline(0, linestyle='--', color='gray')
plt.title('News Sentiment Over Time (VADER-based)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# LIVE News Scraping + Sentiment
url = "https://www.moneycontrol.com/news/business/stocks/"
headers = {'User-Agent': 'Mozilla/5.0'}
try:
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all('h2')

    news_data = []
    for h in headlines[:10]:
        text = h.get_text(strip=True)
        # Ensure TextBlob is imported and used correctly
        try:
            polarity = TextBlob(text).sentiment.polarity
        except Exception as e:
            print(f"Error processing headline with TextBlob: {text} - {e}")
            polarity = 0 # Assign a default polarity if TextBlob fails
        news_data.append({'headline': text, 'polarity': polarity})

    if news_data: # Only proceed if news_data is not empty
        df_news = pd.DataFrame(news_data)
        df_news['date'] = pd.Timestamp.today().normalize()

        # Sentiment plot
        plt.figure(figsize=(10, 4))
        sns.barplot(x=df_news['date'].astype(str), y='polarity', data=df_news) # Convert date to string for plotting
        plt.title('Reliance News Sentiment Over Time (Live Headlines)', fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Sentiment Polarity")
        plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
        plt.tight_layout()
        plt.show()

        # Optional: Display scraped headlines
        display(df_news[['headline', 'polarity']])
    else:
        print("No headlines scraped.")

except requests.exceptions.RequestException as e:
    print(f"Error during news scraping: {e}")
except Exception as e:
    print(f"An unexpected error occurred during news scraping: {e}")
