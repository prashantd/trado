import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# --------------------
# 1. Load Nifty Data
# --------------------
def load_nifty_data(start=None, end=None):
    from datetime import datetime, timedelta
    
    # Calculate dates programmatically if not provided
    if end is None:
        end = datetime.today()
    if start is None:
        start = end - timedelta(days=5*365)  # 5 years back
    
    # Convert to string format if datetime objects
    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d")
    
    print(f"Loading Nifty data from {start} to {end}")
    df = yf.download("^NSEI", start=start, end=end)
    
    # Flatten MultiIndex columns to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df.to_csv("nifty50.csv")
    return df

nifty = load_nifty_data()

# --------------------
# 2. Add Technical Features
# --------------------
def calculate_rsi(prices, window=14):
    """
    Calculate RSI (Relative Strength Index)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

nifty["Return"] = nifty["Close"].pct_change()
nifty["MA9"] = nifty["Close"].rolling(9).mean()
nifty["MA20"] = nifty["Close"].rolling(20).mean()
nifty["MA50"] = nifty["Close"].rolling(50).mean()
nifty["Volatility"] = nifty["Return"].rolling(5).std()
nifty["RSI"] = calculate_rsi(nifty["Close"])

# Binary target: 1 if next-day close > today, else 0
nifty["Target"] = (nifty["Close"].shift(-1) > nifty["Close"]).astype(int)

# Regression targets: percentage changes instead of absolute prices
nifty["Next_Close_Return"] = ((nifty["Close"].shift(-1) - nifty["Close"]) / nifty["Close"]) * 100
nifty["Next_High_Return"] = ((nifty["High"].shift(-1) - nifty["Close"]) / nifty["Close"]) * 100  
nifty["Next_Low_Return"] = ((nifty["Low"].shift(-1) - nifty["Close"]) / nifty["Close"]) * 100
nifty["Next_Range"] = ((nifty["High"].shift(-1) - nifty["Low"].shift(-1)) / nifty["Close"]) * 100

# Store the latest row for prediction (before dropna removes it)
latest_row = nifty.iloc[-1:].copy()

# Remove rows with NaN target values for training
nifty_for_training = nifty.dropna()

# Add the latest row back for prediction (it will have NaN targets but we don't need them for prediction)
nifty = pd.concat([nifty_for_training, latest_row], ignore_index=False).drop_duplicates(keep='last')

# --------------------
# 3. Fetch Live News + Sentiment
# --------------------
NEWS_API_KEY = "621aa252bdf1430c9726fe92705252a8"  # replace with your NewsAPI key

def fetch_news(days_back=7):  # Reduced from 30 to 7 days for more focused recent news
    """
    Fetch news from multiple relevant sources that affect Nifty 50:
    1. Indian market news (Nifty, BSE, NSE)
    2. US Fed and monetary policy news  
    3. Global markets (Dow, NASDAQ)
    4. Indian economic policy and RBI news
    """
    from datetime import datetime, timedelta
    to_date = datetime.today()
    from_date = to_date - timedelta(days=days_back)
    
    # Define multiple focused queries for better coverage
    queries = [
        "Nifty 50 OR India stock market OR NSE OR BSE",  # Indian markets
        "Federal Reserve OR Fed rate OR FOMC OR Jerome Powell",  # Fed policy
        "Dow Jones OR NASDAQ OR Wall Street OR US markets",  # US markets
        "RBI OR Reserve Bank India OR Finance Ministry OR India GDP",  # Indian economy
        "FII OR foreign investment OR emerging markets India"  # Foreign investment
    ]
    
    all_news = []
    
    for query in queries:
        url = (
            f"https://newsapi.org/v2/everything?q={query}"
            f"&from={from_date.date()}&to={to_date.date()}"
            f"&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            articles = data.get("articles", [])
            
            for article in articles:
                all_news.append({
                    "Date": article["publishedAt"][:10],   # YYYY-MM-DD
                    "Headline": article["title"],
                    "Query": query  # Track which query found this news
                })
        except Exception as e:
            print(f"Error fetching news for query '{query}': {e}")
            continue
    
    # Remove duplicates based on headline
    news_df = pd.DataFrame(all_news)
    if not news_df.empty:
        news_df = news_df.drop_duplicates(subset=['Headline'], keep='first')
    
    return news_df

# Fetch live news
print("📰 Fetching comprehensive news from multiple sources...")
news_df = fetch_news()

if not news_df.empty:
    news_df["Date"] = pd.to_datetime(news_df["Date"])
    
    # Show news source breakdown
    print(f"📊 News Coverage Summary:")
    query_counts = news_df['Query'].value_counts()
    for query, count in query_counts.items():
        source_type = {
            "Nifty 50 OR India stock market OR NSE OR BSE": "🇮🇳 Indian Markets",
            "Federal Reserve OR Fed rate OR FOMC OR Jerome Powell": "🏛️ Fed Policy", 
            "Dow Jones OR NASDAQ OR Wall Street OR US markets": "🇺🇸 US Markets",
            "RBI OR Reserve Bank India OR Finance Ministry OR India GDP": "🏦 Indian Economy",
            "FII OR foreign investment OR emerging markets India": "💰 Foreign Investment"
        }
        print(f"   {source_type.get(query, query)}: {count} articles")
    
    print(f"📈 Total articles collected: {len(news_df)} (after removing duplicates)")
    
    # Sentiment Scoring
    analyzer = SentimentIntensityAnalyzer()
    news_df["Sentiment"] = news_df["Headline"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    
    # Show recent impactful headlines
    recent_news = news_df.sort_values('Date', ascending=False).head(5)
    print(f"🔍 Recent High-Impact Headlines:")
    for _, row in recent_news.iterrows():
        sentiment_emoji = "🟢" if row['Sentiment'] > 0.1 else "🔴" if row['Sentiment'] < -0.1 else "🟡"
        print(f"   {sentiment_emoji} {row['Sentiment']:.3f} | {row['Headline'][:80]}...")
    
    # Aggregate Daily Sentiment (weighted by absolute sentiment strength)
    news_df['SentimentWeight'] = abs(news_df['Sentiment'])
    sentiment_daily = news_df.groupby("Date").apply(
        lambda x: np.average(x['Sentiment'], weights=x['SentimentWeight']) if sum(x['SentimentWeight']) > 0 else x['Sentiment'].mean()
    ).reset_index()
    sentiment_daily.columns = ["Date", "DailySentiment"]
    
    print(f"📊 Latest Daily Sentiment: {sentiment_daily['DailySentiment'].iloc[-1]:.4f}")
else:
    print("⚠️ No news data retrieved - using neutral sentiment")
    sentiment_daily = pd.DataFrame({
        "Date": [pd.Timestamp.today().date()],
        "DailySentiment": [0.0]
    })

# --------------------
# 4. Merge Features  
# --------------------
nifty_features = nifty.reset_index()

# Create enhanced sentiment handling
if not sentiment_daily.empty and len(sentiment_daily) > 0:
    # Forward fill recent sentiment for missing days (instead of zero-filling)
    nifty_features = pd.merge(nifty_features, sentiment_daily, left_on="Date", right_on="Date", how="left")
    
    # Use rolling sentiment approach: fill missing values with recent sentiment trends
    # This prevents the 99% zero problem that makes models ignore sentiment
    nifty_features["DailySentiment"] = nifty_features["DailySentiment"].fillna(method='ffill')  # Forward fill
    nifty_features["DailySentiment"] = nifty_features["DailySentiment"].fillna(method='bfill')  # Backward fill
    nifty_features["DailySentiment"] = nifty_features["DailySentiment"].fillna(0)  # Only zero-fill if absolutely no sentiment data
else:
    print("⚠️ No sentiment data available - using neutral sentiment")
    nifty_features["DailySentiment"] = 0.0

# Enhance sentiment signal by amplifying its impact (multiply by factor for stronger signal)
SENTIMENT_AMPLIFICATION = 30.0
nifty_features["DailySentiment"] = nifty_features["DailySentiment"] * SENTIMENT_AMPLIFICATION

# Debug sentiment data quality
print(f"\n🔍 ENHANCED SENTIMENT DATA ANALYSIS:")
print(f"   Total sentiment records: {len(sentiment_daily) if not sentiment_daily.empty else 0}")
if not sentiment_daily.empty:
    print(f"   Sentiment date range: {sentiment_daily['Date'].min()} to {sentiment_daily['Date'].max()}")
print(f"   Nifty date range: {nifty_features['Date'].min()} to {nifty_features['Date'].max()}")
print(f"   Enhanced sentiment variance: {nifty_features['DailySentiment'].var():.6f}")
print(f"   Enhanced sentiment range: {nifty_features['DailySentiment'].min():.4f} to {nifty_features['DailySentiment'].max():.4f}")
print(f"   Non-zero sentiment days: {(nifty_features['DailySentiment'] != 0).sum()}/{len(nifty_features)}")
print(f"   Sentiment amplification factor: {SENTIMENT_AMPLIFICATION}x")

# Show sample sentiment values
recent_sentiment = nifty_features[['Date', 'DailySentiment']].tail(10)
print(f"   Recent enhanced sentiment samples:")
for _, row in recent_sentiment.iterrows():
    sentiment_direction = "📈 BULLISH" if row['DailySentiment'] > 0.05 else "📉 BEARISH" if row['DailySentiment'] < -0.05 else "📊 NEUTRAL"
    print(f"      {row['Date'].date()}: {row['DailySentiment']:.4f} {sentiment_direction}")

# --------------------
# 5. Train/Test Split
# --------------------
# Only use rows with valid target values for training (exclude the latest row with NaN targets)
valid_rows = (~nifty_features["Target"].isna() & 
              ~nifty_features["Next_Close_Return"].isna() & 
              ~nifty_features["Next_High_Return"].isna() & 
              ~nifty_features["Next_Low_Return"].isna() & 
              ~nifty_features["Next_Range"].isna())

X_valid = nifty_features[valid_rows][["MA9", "MA20", "MA50", "Volatility", "RSI", "DailySentiment"]]
y_direction_valid = nifty_features[valid_rows]["Target"]
y_close_return_valid = nifty_features[valid_rows]["Next_Close_Return"]
y_high_return_valid = nifty_features[valid_rows]["Next_High_Return"]
y_low_return_valid = nifty_features[valid_rows]["Next_Low_Return"]
y_range_valid = nifty_features[valid_rows]["Next_Range"]

X_train, X_test, y_dir_train, y_dir_test = train_test_split(X_valid, y_direction_valid, test_size=0.2, shuffle=False)
_, _, y_close_ret_train, y_close_ret_test = train_test_split(X_valid, y_close_return_valid, test_size=0.2, shuffle=False)
_, _, y_high_ret_train, y_high_ret_test = train_test_split(X_valid, y_high_return_valid, test_size=0.2, shuffle=False)
_, _, y_low_ret_train, y_low_ret_test = train_test_split(X_valid, y_low_return_valid, test_size=0.2, shuffle=False)
_, _, y_range_train, y_range_test = train_test_split(X_valid, y_range_valid, test_size=0.2, shuffle=False)

# --------------------
# 6. Train Multiple Models with Feature Scaling
# --------------------
print("Training improved models with feature scaling...")

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Direction prediction - try multiple models and pick the best
print("\nTesting different classification models...")

# Option 1: Improved LogisticRegression with regularization
lr_model = LogisticRegression(
    C=1.0,                    # Regularization strength
    max_iter=1000,            # More iterations
    random_state=42,
    class_weight='balanced'   # Handle class imbalance
)
lr_model.fit(X_train_scaled, y_dir_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_dir_test, lr_pred)

# Option 2: RandomForest Classifier
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)
rf_classifier.fit(X_train, y_dir_train)  # RF doesn't need scaling
rf_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_dir_test, rf_pred)

print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")

# Test 3: Sentiment-Aware Model (manual sentiment override)
def sentiment_aware_prediction(model, features_scaled, features_raw, sentiment_threshold=0.15):
    """Apply sentiment-based adjustments to model predictions"""
    base_pred = model.predict(features_scaled)
    base_proba = model.predict_proba(features_scaled)
    
    # Get sentiment values (last column after scaling)
    sentiment_values = features_raw[:, -1]  # DailySentiment is last feature
    
    adjusted_predictions = []
    adjustments = 0
    
    for i, (pred, proba, sentiment) in enumerate(zip(base_pred, base_proba, sentiment_values)):
        # Strong sentiment override
        if sentiment > sentiment_threshold:  # Strong positive sentiment -> BULLISH
            if proba[0] > 0.55:  # Model says DOWN but sentiment says UP
                adjusted_predictions.append(1)  # Override to UP
                adjustments += 1
            else:
                adjusted_predictions.append(pred)  # Keep original
        elif sentiment < -sentiment_threshold:  # Strong negative sentiment -> BEARISH
            if proba[1] > 0.55:  # Model says UP but sentiment says DOWN
                adjusted_predictions.append(0)  # Override to DOWN
                adjustments += 1
            else:
                adjusted_predictions.append(pred)  # Keep original
        else:
            adjusted_predictions.append(pred)  # Keep original for weak sentiment
    
    print(f"   Sentiment overrides applied: {adjustments}/{len(base_pred)}")
    return np.array(adjusted_predictions)

# Test sentiment-aware predictions on LogisticRegression
sentiment_pred = sentiment_aware_prediction(lr_model, X_test_scaled, X_test.values, sentiment_threshold=0.15)
sentiment_accuracy = accuracy_score(y_dir_test, sentiment_pred)
print(f"Sentiment-Aware LR Accuracy: {sentiment_accuracy:.3f}")

# Select the best model
if sentiment_accuracy > max(lr_accuracy, rf_accuracy):
    direction_model = lr_model
    use_scaling = True
    use_sentiment_aware = True
    best_pred = sentiment_pred
    best_accuracy = sentiment_accuracy
    print("✅ Selected: Sentiment-Aware Logistic Regression")
elif lr_accuracy >= rf_accuracy:
    direction_model = lr_model
    use_scaling = True
    use_sentiment_aware = False
    best_pred = lr_pred
    best_accuracy = lr_accuracy
    print("✅ Selected: Improved Logistic Regression")
else:
    direction_model = rf_classifier
    use_scaling = False
    use_sentiment_aware = False
    best_pred = rf_pred
    best_accuracy = rf_accuracy
    print("✅ Selected: Random Forest Classifier")

# Store the scaler for later use
feature_scaler = scaler

# Regression models for price predictions
close_return_model = RandomForestRegressor(n_estimators=100, random_state=42)
close_return_model.fit(X_train, y_close_ret_train)

high_return_model = RandomForestRegressor(n_estimators=100, random_state=42)
high_return_model.fit(X_train, y_high_ret_train)

low_return_model = RandomForestRegressor(n_estimators=100, random_state=42)
low_return_model.fit(X_train, y_low_ret_train)

range_model = RandomForestRegressor(n_estimators=100, random_state=42)
range_model.fit(X_train, y_range_train)

# --------------------
# 7. Evaluate Models
# --------------------
print(f"\n🎯 BEST DIRECTION MODEL PERFORMANCE:")
print("="*50)
print(f"Model: {type(direction_model).__name__}")
print(f"Accuracy: {best_accuracy:.3f}")
print(f"Detailed Classification Report:")
print(classification_report(y_dir_test, best_pred, target_names=['DOWN', 'UP']))

# Check feature importance if available
if hasattr(direction_model, 'feature_importances_'):
    feature_names = ["MA9", "MA20", "MA50", "Volatility", "RSI", "DailySentiment"]
    importances = direction_model.feature_importances_
    print(f"\n📊 FEATURE IMPORTANCE:")
    for name, importance in zip(feature_names, importances):
        print(f"   {name}: {importance:.3f}")
elif hasattr(direction_model, 'coef_'):
    feature_names = ["MA9", "MA20", "MA50", "Volatility", "RSI", "DailySentiment"]
    coefficients = direction_model.coef_[0]
    print(f"\n📊 FEATURE COEFFICIENTS (Logistic Regression):")
    for name, coef in zip(feature_names, coefficients):
        print(f"   {name}: {coef:.3f}")

# Regression model evaluation
close_ret_preds = close_return_model.predict(X_test)
high_ret_preds = high_return_model.predict(X_test)
low_ret_preds = low_return_model.predict(X_test)
range_preds = range_model.predict(X_test)

print(f"\n📈 PRICE PREDICTION PERFORMANCE:")
print("="*50)

print(f"\nClose Return Prediction - MAE: {mean_absolute_error(y_close_ret_test, close_ret_preds):.3f}%, R²: {r2_score(y_close_ret_test, close_ret_preds):.3f}")
print(f"High Return Prediction - MAE: {mean_absolute_error(y_high_ret_test, high_ret_preds):.3f}%, R²: {r2_score(y_high_ret_test, high_ret_preds):.3f}")
print(f"Low Return Prediction - MAE: {mean_absolute_error(y_low_ret_test, low_ret_preds):.3f}%, R²: {r2_score(y_low_ret_test, low_ret_preds):.3f}")
print(f"Range Prediction - MAE: {mean_absolute_error(y_range_test, range_preds):.3f}%, R²: {r2_score(y_range_test, range_preds):.3f}")

# --------------------
# 8. Predict Next Day's Direction and Range
# --------------------
def predict_next_day():
    # Get the most recent data point (latest available features)
    latest_data = nifty_features.iloc[-1]
    
    # Extract features for prediction
    latest_features = [
        latest_data["MA9"],
        latest_data["MA20"],
        latest_data["MA50"], 
        latest_data["Volatility"],
        latest_data["RSI"],
        latest_data["DailySentiment"]
    ]
    
    # Reshape for prediction (model expects 2D array)
    latest_features = np.array(latest_features).reshape(1, -1)
    
    # Apply scaling if the selected model needs it
    if use_scaling:
        latest_features_scaled = feature_scaler.transform(latest_features)
        direction_pred = direction_model.predict(latest_features_scaled)[0]
        direction_proba = direction_model.predict_proba(latest_features_scaled)[0]
        
        # Apply sentiment-aware override if using that model
        if 'use_sentiment_aware' in locals() and use_sentiment_aware:
            sentiment_value = latest_features[0, -1]  # Last feature is DailySentiment
            print(f"🔍 Sentiment Analysis: {sentiment_value:.4f} ({'📉 BEARISH' if sentiment_value < -0.05 else '📈 BULLISH' if sentiment_value > 0.05 else '📊 NEUTRAL'})")
            
            if sentiment_value > 0.15 and direction_proba[0] > 0.51:  # Lower threshold: Strong positive sentiment but model says DOWN
                print(f"🔄 SENTIMENT OVERRIDE: Strong positive sentiment ({sentiment_value:.4f}) overrides bearish prediction")
                direction_pred = 1
                direction_proba = np.array([0.4, 0.6])  # Adjust probabilities
            elif sentiment_value < -0.15 and direction_proba[1] > 0.51:  # Lower threshold: Strong negative sentiment but model says UP
                print(f"🔄 SENTIMENT OVERRIDE: Strong negative sentiment ({sentiment_value:.4f}) overrides bullish prediction")
                direction_pred = 0
                direction_proba = np.array([0.6, 0.4])  # Adjust probabilities
        else:
            # Even if not using sentiment-aware model, show sentiment analysis
            sentiment_value = latest_features[0, -1]
            print(f"🔍 Sentiment Analysis: {sentiment_value:.4f} ({'📉 BEARISH' if sentiment_value < -0.05 else '📈 BULLISH' if sentiment_value > 0.05 else '📊 NEUTRAL'})")
            print(f"💡 Note: Current model ignores sentiment (coefficient: {direction_model.coef_[0][-1] if hasattr(direction_model, 'coef_') else 'N/A'})")
            
            # Manual override for very strong contradictory sentiment
            if sentiment_value < -0.25 and direction_pred == 1:
                print(f"⚠️  WARNING: Very negative sentiment ({sentiment_value:.4f}) contradicts BULLISH prediction!")
                print(f"⚠️  Consider bearish bias in trading decisions")
            elif sentiment_value > 0.25 and direction_pred == 0:
                print(f"⚠️  WARNING: Very positive sentiment ({sentiment_value:.4f}) contradicts BEARISH prediction!")  
                print(f"⚠️  Consider bullish bias in trading decisions")
    else:
        direction_pred = direction_model.predict(latest_features)[0]
        direction_proba = direction_model.predict_proba(latest_features)[0]
    
    # Get latest price and date info
    latest_close = latest_data["Close"]
    latest_date = latest_data["Date"]
    
    # Predict percentage returns and convert to actual prices
    close_return_pred = close_return_model.predict(latest_features)[0]
    high_return_pred = high_return_model.predict(latest_features)[0]
    low_return_pred = low_return_model.predict(latest_features)[0]
    range_pred = range_model.predict(latest_features)[0]
    
    # Apply realistic constraints based on historical data
    close_return_pred = np.clip(close_return_pred, -2.5, 2.5)  # Max ±2.5% move
    high_return_pred = np.clip(high_return_pred, 0, 1.5)       # Max 1.5% above close
    low_return_pred = np.clip(low_return_pred, -1.5, 0)       # Max 1.5% below close
    range_pred = np.clip(range_pred, 0.3, 2.0)                # Range between 0.3-2%
    
    # Convert percentage returns to actual prices
    predicted_close = latest_close * (1 + close_return_pred/100)
    predicted_high = latest_close * (1 + high_return_pred/100)
    predicted_low = latest_close * (1 + low_return_pred/100)
    
    # Calculate range metrics
    range_width = predicted_high - predicted_low
    midpoint = (predicted_high + predicted_low) / 2
    
    print("\n" + "="*30)
    print("NEXT DAY PREDICTION & RANGE FORECAST")
    print("="*30)
    print(f"Latest Date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"Current Close: ₹{latest_close:.2f}")
    print(f"MA9: ₹{latest_data['MA9']:.2f}")
    print(f"MA20: ₹{latest_data['MA20']:.2f}")
    print(f"MA50: ₹{latest_data['MA50']:.2f}")
    print(f"Volatility: {latest_data['Volatility']:.4f}")
    print(f"RSI: {latest_data['RSI']:.2f}")
    print(f"Daily Sentiment: {latest_data['DailySentiment']:.4f}")
    print("-" * 30)
    
    # Direction prediction
    if direction_pred == 1:
        print("🟢 DIRECTION: BULLISH (Price likely to go UP)")
        print(f"   Confidence: {direction_proba[1]:.2%}")
    else:
        print("🔴 DIRECTION: BEARISH (Price likely to go DOWN)")
        print(f"   Confidence: {direction_proba[0]:.2%}")

    print("-" * 30)
    print("📊 PREDICTED PRICE RANGE:")
    print(f"   Expected High:  ₹{predicted_high:.2f} ({high_return_pred:+.2f}%)")
    print(f"   Expected Low:   ₹{predicted_low:.2f} ({low_return_pred:+.2f}%)")
    print(f"   Expected Close: ₹{predicted_close:.2f} ({close_return_pred:+.2f}%)")
    print(f"   Range Width:    ₹{range_width:.2f} ({range_pred:.2f}%)")
    print(f"   Midpoint:       ₹{midpoint:.2f}")
    print(f"   Predicted Returns: High{high_return_pred:+.2f}%, Low{low_return_pred:+.2f}%, Close{close_return_pred:+.2f}%")
    
    # Trading levels
    print("-" * 30)
    print("🎯 TRADING LEVELS:")
    print(f"   Resistance:     ₹{predicted_high:.2f}")
    print(f"   Support:        ₹{predicted_low:.2f}")
    if predicted_close > latest_close:
        print(f"   Target:         ₹{predicted_close:.2f} (BUY)")
    else:
        print(f"   Target:         ₹{predicted_close:.2f} (SELL)")

    print("="*30)

    return {
        'direction': 'BULLISH' if direction_pred == 1 else 'BEARISH',
        'direction_confidence': direction_proba[1] if direction_pred == 1 else direction_proba[0],
        'predicted_high': predicted_high,
        'predicted_low': predicted_low,
        'predicted_close': predicted_close,
        'current_price': latest_close,
        'range_width': range_width,
        'latest_date': latest_date
    }

# Make prediction for next day
next_day_prediction = predict_next_day()

# --------------------
# 9. Custom Prediction Function
# --------------------
def predict_custom(ma9, ma20, ma50, volatility, rsi, daily_sentiment):
    """
    Make a prediction with custom feature values
    
    Args:
        ma9: 9-day moving average
        ma20: 20-day moving average
        ma50: 50-day moving average  
        volatility: 5-day rolling volatility
        rsi: RSI (Relative Strength Index)
        daily_sentiment: Daily sentiment score (-1 to 1)
    
    Returns:
        dict: Prediction result with direction, confidence, and price range
    """
    features = np.array([ma9, ma20, ma50, volatility, rsi, daily_sentiment]).reshape(1, -1)
    
    # Apply scaling if the direction model needs it (LogisticRegression)
    if hasattr(direction_model, 'C'):  # LogisticRegression has 'C' parameter
        features_scaled = scaler.transform(features)
        direction_features = features_scaled
    else:
        direction_features = features
    
    # Direction prediction using scaled features if needed
    direction_pred = direction_model.predict(direction_features)[0]
    direction_proba = direction_model.predict_proba(direction_features)[0]
    
    # Calculate reference price as average of latest Nifty high and low
    latest_data = nifty_features.iloc[-1]
    reference_price = (latest_data["High"] + latest_data["Low"]) / 2
    
    # Predict percentage returns
    close_return_pred = close_return_model.predict(features)[0]
    high_return_pred = high_return_model.predict(features)[0]
    low_return_pred = low_return_model.predict(features)[0]
    
    # Apply realistic constraints
    close_return_pred = np.clip(close_return_pred, -2.5, 2.5)
    high_return_pred = np.clip(high_return_pred, 0, 1.5)
    low_return_pred = np.clip(low_return_pred, -1.5, 0)
    
    # Convert to actual prices
    predicted_close = reference_price * (1 + close_return_pred/100)
    predicted_high = reference_price * (1 + high_return_pred/100)
    predicted_low = reference_price * (1 + low_return_pred/100)
    
    result = {
        'direction': 'BULLISH' if direction_pred == 1 else 'BEARISH',
        'direction_confidence': direction_proba[1] if direction_pred == 1 else direction_proba[0],
        'predicted_close_return': close_return_pred,
        'predicted_high_return': high_return_pred,
        'predicted_low_return': low_return_pred,
        'predicted_high': predicted_high,
        'predicted_low': predicted_low,
        'predicted_close': predicted_close,
        'range_width': predicted_high - predicted_low,
        'features': {
            'MA9': ma9,
            'MA20': ma20,
            'MA50': ma50,
            'Volatility': volatility,
            'RSI': rsi,
            'DailySentiment': daily_sentiment
        }
    }
    
    return result

# Example: Test with current market conditions
print("\n" + "="*60)
print("EXAMPLE: Custom Prediction with Range")
print("="*60)

# Get current values as example
current_ma9 = nifty_features["MA9"].iloc[-1]
current_ma20 = nifty_features["MA20"].iloc[-1]
current_ma50 = nifty_features["MA50"].iloc[-1]
current_vol = nifty_features["Volatility"].iloc[-1]
current_rsi = nifty_features["RSI"].iloc[-1]
current_sentiment = nifty_features["DailySentiment"].iloc[-1]

custom_pred = predict_custom(current_ma9, current_ma20, current_ma50, current_vol, current_rsi, current_sentiment)
print(f"Direction: {custom_pred['direction']} (Confidence: {custom_pred['direction_confidence']:.2%})")
print(f"Expected Returns - Close: {custom_pred['predicted_close_return']:+.2f}%, High: {custom_pred['predicted_high_return']:+.2f}%, Low: {custom_pred['predicted_low_return']:+.2f}%")
print(f"Expected Prices - High: ₹{custom_pred['predicted_high']:.2f}, Low: ₹{custom_pred['predicted_low']:.2f}, Close: ₹{custom_pred['predicted_close']:.2f}")
print(f"Range Width: ₹{custom_pred['range_width']:.2f}")
print("="*60)
