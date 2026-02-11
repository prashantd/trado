import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def calculate_ema50(series):
    return series.ewm(span=50, adjust=False).mean()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

symbol = 'INFY.NS'
start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
df = yf.download(symbol, start=start, interval='1d', progress=False, auto_adjust=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]
df = df.reset_index(drop=True)
df['EMA50'] = calculate_ema50(df['Close'])
df['RSI'] = calculate_rsi(df['Close'], window=14)
print(df[['Close','EMA50','RSI']].tail(3))
