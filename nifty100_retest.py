import yfinance as yf
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta

# You need a list of Nifty 100 stock symbols. Here is a static list (as of 2024, update as needed):
nifty100_symbols = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ITC.NS', 'LT.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'BAJFINANCE.NS',
    'ASIANPAINT.NS', 'HCLTECH.NS', 'AXISBANK.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
    'TITAN.NS', 'ULTRACEMCO.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS',
    'AMBUJACEM.NS', 'APOLLOHOSP.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS',
    'BAJAJHLDNG.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BEL.NS', 'BERGEPAINT.NS',
    'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'COLPAL.NS', 'DABUR.NS',
    'DIVISLAB.NS', 'DLF.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GAIL.NS', 'GODREJCP.NS',
    'GRASIM.NS', 'HAVELLS.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDPETRO.NS',
    'ICICIGI.NS', 'ICICIPRULI.NS', 'IDFCFIRSTB.NS', 'INDIGO.NS', 'INDUSINDBK.NS',
    'JSWSTEEL.NS', 'JUBLFOOD.NS', 'LUPIN.NS', 'M&M.NS', 'MUTHOOTFIN.NS',
    'NAUKRI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'PETRONET.NS',
    'PIIND.NS', 'PIDILITIND.NS', 'PNB.NS', 'POWERGRID.NS', 'RECLTD.NS', 'SAIL.NS',
    'SBICARD.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SRF.NS', 'TATACHEM.NS',
    'TATACONSUM.NS', 'TMPV.NS', 'TATAPOWER.NS', 'TATASTEEL.NS', 'TORNTPHARM.NS',
    'TRENT.NS', 'TVSMOTOR.NS', 'UBL.NS', 'UPL.NS', 'VEDL.NS', 'VOLTAS.NS', 'WIPRO.NS',
    'ZEEL.NS'
]


def calculate_retest(df, window=20, tolerance=0.5):
    import traceback
    try:
        df = df.copy()
        df['rolling_max'] = df['High'].rolling(window).max()
        df['rolling_min'] = df['Low'].rolling(window).min()
        df['breakout'] = (df['Close'] > df['rolling_max'].shift(1))
        df['breakdown'] = (df['Close'] < df['rolling_min'].shift(1))
        n = len(df)
        def arr(s):
            a = np.asarray(s)
            if a.ndim > 1:
                a = a.ravel()
            if len(a) != n:
                a = np.resize(a, n)
            return a
        # Use .to_numpy(dtype=bool, na_value=False) to avoid FutureWarning
        breakout_prev = arr(df['breakout'].shift(1).to_numpy(dtype=bool, na_value=False))
        breakdown_prev = arr(df['breakdown'].shift(1).to_numpy(dtype=bool, na_value=False))
        low_arr = arr(df['Low'])
        max_arr = arr(df['rolling_max'].shift(1))
        high_arr = arr(df['High'])
        min_arr = arr(df['rolling_min'].shift(1))
        retest_up = np.full(n, False)
        retest_down = np.full(n, False)
        valid_up = (~np.isnan(low_arr)) & (~np.isnan(max_arr))
        valid_down = (~np.isnan(high_arr)) & (~np.isnan(min_arr))
        retest_up[valid_up] = breakout_prev[valid_up] & (np.abs(low_arr[valid_up] - max_arr[valid_up]) <= tolerance)
        retest_down[valid_down] = breakdown_prev[valid_down] & (np.abs(high_arr[valid_down] - min_arr[valid_down]) <= tolerance)
        df['retest_up'] = pd.Series(retest_up, index=df.index)
        df['retest_down'] = pd.Series(retest_down, index=df.index)
        return df
    except Exception as e:
        print(f"EXCEPTION in calculate_retest: {e}")
        traceback.print_exc()
        print(f"DEBUG DataFrame columns: {df.columns}")
        print(f"DEBUG DataFrame shape: {df.shape}")
        print(df.head())
        raise

def find_retests(symbols, start=None, end=None):
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    results = []
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
            if df.empty or len(df) < 21:
                continue
            # Fix for MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten columns and select only this symbol
                df.columns = [col[0] for col in df.columns]
            df = df.reset_index(drop=True)
            df = calculate_retest(df)
            if df['retest_up'].iloc[-1]:
                results.append({'symbol': symbol, 'type': 'UPWARD', 'date': df.index[-1]})
            if df['retest_down'].iloc[-1]:
                results.append({'symbol': symbol, 'type': 'DOWNWARD', 'date': df.index[-1]})
        except Exception as e:
            print(f"Error for {symbol}: {e}")
    return pd.DataFrame(results)

def calculate_ema(series, span=20):
    return series.ewm(span=span, adjust=False).mean()

def calculate_ema50(series):
    return series.ewm(span=50, adjust=False).mean()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(df):
    # VWAP = sum(price * volume) / sum(volume) intraday
    pv = (df['Close'] * df['Volume']).cumsum()
    vol = df['Volume'].cumsum()
    vwap = pv / vol
    return vwap

def check_bullish_stocks_daily(symbols, start=None, end=None):
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    results = []
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, end=end, interval='1d', progress=False, auto_adjust=False)
            if df.empty or len(df) < 21:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.reset_index(drop=True)
            df['EMA20'] = calculate_ema(df['Close'], span=20)
            df['EMA50'] = calculate_ema50(df['Close'])
            df['RSI'] = calculate_rsi(df['Close'], window=14)
            df['VWAP'] = calculate_vwap(df)
            # Last two daily candles
            if (
                df['Close'].iloc[-1] > df['Close'].iloc[-2]
                and df['RSI'].iloc[-1] > 70
                and df['Close'].iloc[-1] > df['EMA20'].iloc[-1]
                and df['Close'].iloc[-1] > df['VWAP'].iloc[-1]
                and df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]
            ):
                results.append({
                    'symbol': symbol,
                    'date': df.index[-1],
                    'close': df['Close'].iloc[-1],
                    'prev_close': df['Close'].iloc[-2],
                    'RSI': df['RSI'].iloc[-1],
                    'EMA20': df['EMA20'].iloc[-1],
                    'EMA50': df['EMA50'].iloc[-1],
                    'VWAP': df['VWAP'].iloc[-1]
                })
        except Exception as e:
            print(f"Error for {symbol} (daily bullish filter): {e}")
    return pd.DataFrame(results)

def check_bearish_stocks_daily(symbols, start=None, end=None):
    if start is None:
        start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    results = []
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start, end=end, interval='1d', progress=False, auto_adjust=False)
            if df.empty or len(df) < 21:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            df = df.reset_index(drop=True)
            df['EMA20'] = calculate_ema(df['Close'], span=20)
            df['EMA50'] = calculate_ema50(df['Close'])
            df['RSI'] = calculate_rsi(df['Close'], window=14)
            # df['VWAP'] = calculate_vwap(df)
            # Last two daily candles
            if (
                df['Close'].iloc[-1] < df['Close'].iloc[-2]
                and df['RSI'].iloc[-1] < 30
                and df['Close'].iloc[-1] < df['EMA50'].iloc[-1]
                and df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]
                # and df['Close'].iloc[-1] < df['VWAP'].iloc[-1]
            ):
                results.append({
                    'symbol': symbol,
                    'date': df.index[-1],
                    'close': df['Close'].iloc[-1],
                    'prev_close': df['Close'].iloc[-2],
                    'RSI': df['RSI'].iloc[-1],
                    'EMA20': df['EMA20'].iloc[-1],
                    'EMA50': df['EMA50'].iloc[-1],
                    # 'VWAP': df['VWAP'].iloc[-1]
                })
        except Exception as e:
            print(f"Error for {symbol} (daily bearish filter): {e}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Stocks with successful retest today:")
    retest_df = find_retests(nifty100_symbols)
    print(retest_df)
    print("\nStocks with bullish daily close, RSI>70, and price>EMA20 (last month):")
    bullish_daily_df = check_bullish_stocks_daily(nifty100_symbols)
    print(bullish_daily_df)
    print("\nStocks with bearish daily close, RSI<30, and price<EMA50 (last month):")
    bearish_daily_df = check_bearish_stocks_daily(nifty100_symbols)
    print(bearish_daily_df)
