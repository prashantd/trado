from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.conf import settings
import yfinance as yf
import pandas as pd
import json
import requests
import hashlib
from datetime import date
from .models import RetestStock, BullishStock, BearishStock, StockUpdateLog, Instrument
import requests

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'stocks/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

def kite_login(request):
    """Redirect to Kite Connect login page"""
    kite_login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={settings.KITE_API_KEY}"
    return redirect(kite_login_url)

def kite_callback(request):
    """Handle callback from Kite Connect and exchange request_token for access_token"""
    request_token = request.GET.get('request_token')

    if not request_token:
        messages.error(request, 'No request token received from Kite.')
        return redirect('login')

    try:
        # Create checksum for token exchange
        checksum_string = f"{settings.KITE_API_KEY}{request_token}{settings.KITE_API_SECRET}"
        checksum = hashlib.sha256(checksum_string.encode()).hexdigest()

        # Exchange request_token for access_token
        token_url = "https://api.kite.trade/session/token"
        headers = {"X-Kite-Version": "3"}
        data = {
            "api_key": settings.KITE_API_KEY,
            "request_token": request_token,
            "checksum": checksum
        }

        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()

        token_data = response.json()

        if token_data.get('status') == 'success':
            access_token = token_data['data']['access_token']
            user_name = token_data['data']['user_name']

            # Store access_token in session
            request.session['kite_access_token'] = access_token
            request.session['kite_user_name'] = user_name
            request.session['kite_api_key'] = settings.KITE_API_KEY

            messages.success(request, f'Successfully logged in as {user_name}')
            return redirect('dashboard')
        else:
            messages.error(request, 'Failed to obtain access token from Kite.')
            return redirect('login')

    except requests.exceptions.RequestException as e:
        messages.error(request, f'Error communicating with Kite API: {str(e)}')
        return redirect('login')
    except KeyError as e:
        messages.error(request, f'Invalid response from Kite API: {str(e)}')
        return redirect('login')

@login_required
def dashboard(request):
    # Get all stocks from different categories
    retest_stocks = RetestStock.objects.all()
    bullish_stocks = BullishStock.objects.all()
    bearish_stocks = BearishStock.objects.all()

    # Combine all unique stocks
    all_stocks = set()
    for stock in retest_stocks:
        all_stocks.add(stock.symbol)
    for stock in bullish_stocks:
        all_stocks.add(stock.symbol)
    for stock in bearish_stocks:
        all_stocks.add(stock.symbol)

    stocks_data = []
    for symbol in all_stocks:
        stock_info = {'symbol': symbol, 'categories': []}
        if RetestStock.objects.filter(symbol=symbol).exists():
            retest = RetestStock.objects.filter(symbol=symbol).first()
            stock_info['categories'].append(f"Retest ({retest.retest_type})")
        if BullishStock.objects.filter(symbol=symbol).exists():
            stock_info['categories'].append("Bullish")
        if BearishStock.objects.filter(symbol=symbol).exists():
            stock_info['categories'].append("Bearish")
        stocks_data.append(stock_info)

    # Get current prices and percentage changes
    stock_prices = {}
    if all_stocks:
        try:
            # Fetch current data for all stocks at once
            tickers = yf.Tickers(' '.join(all_stocks))
            for symbol in all_stocks:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    previous_close = info.get('previousClose')

                    if current_price and previous_close:
                        percent_change = ((current_price - previous_close) / previous_close) * 100
                        stock_prices[symbol] = {
                            'price': current_price,
                            'change': percent_change
                        }
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    continue
        except Exception as e:
            print(f"Error fetching stock prices: {e}")

    # Add price data to stocks_data
    for stock in stocks_data:
        symbol = stock['symbol']
        if symbol in stock_prices:
            stock['current_price'] = stock_prices[symbol]['price']
            stock['percent_change'] = stock_prices[symbol]['change']

        # Add flag for Indian stocks
        stock['is_indian_stock'] = symbol.endswith('.NS')

    # Add ATM options for bullish and bearish stocks
    for stock in stocks_data:
        symbol = stock['symbol']
        underlying = symbol.replace('.NS', '')  # Remove .NS for Indian stocks
        current_price = stock.get('current_price')

        if not current_price:
            continue

        atm_option = None
        if "Bullish" in stock['categories']:
            # Find ATM CE
            options = Instrument.objects.filter(
                tradingsymbol__startswith=underlying,
                instrument_type='CE',
                segment='NFO-OPT',
                expiry__gte=date.today()
            ).order_by('expiry')
            if options.exists():
                nearest_expiry = options.first().expiry
                expiry_options = options.filter(expiry=nearest_expiry)
                # Find strike closest to current_price
                closest_option = min(expiry_options, key=lambda x: abs(x.strike - current_price) if x.strike else float('inf'))
                atm_option = closest_option.tradingsymbol
        elif "Bearish" in stock['categories']:
            # Find ATM PE
            options = Instrument.objects.filter(
                tradingsymbol__startswith=underlying,
                instrument_type='PE',
                segment='NFO-OPT',
                expiry__gte=date.today()
            ).order_by('expiry')
            if options.exists():
                nearest_expiry = options.first().expiry
                expiry_options = options.filter(expiry=nearest_expiry)
                # Find strike closest to current_price
                closest_option = min(expiry_options, key=lambda x: abs(x.strike - current_price) if x.strike else float('inf'))
                atm_option = closest_option.tradingsymbol

        stock['atm_option'] = atm_option

    # Get last update time
    last_update = None
    if StockUpdateLog.objects.exists():
        last_update = StockUpdateLog.objects.first().last_updated

    return render(request, 'stocks/dashboard.html', {
        'stocks': stocks_data,
        'last_update': last_update
    })

@login_required
def get_stock_news(request, symbol):
    # Get latest news for the stock
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news

        # Process news items
        news_items = []
        for item in news[:10]:  # Limit to 10 news items
            content = item.get('content', {})
            news_item = {
                'title': content.get('title', 'No title'),
                'summary': content.get('summary', ''),
                'publisher': content.get('provider', {}).get('displayName', 'Unknown'),
                'url': content.get('clickThroughUrl', {}).get('url', ''),
                'published_at': content.get('pubDate', ''),
            }
            news_items.append(news_item)

        return JsonResponse({'news': news_items})
    except Exception as e:
        return JsonResponse({'error': str(e)})

@login_required
def get_stock_chart(request, symbol):
    # Get chart data for the stock
    try:
        ticker = yf.Ticker(symbol)
        # Get 1 month of daily data for candlestick chart
        data = ticker.history(period="1mo", interval="1d")
        
        # Calculate EMA 20
        def calculate_ema(prices, period=20):
            if len(prices) < period:
                return None
            ema = []
            multiplier = 2 / (period + 1)
            
            # First EMA value is SMA
            sma = sum(prices[:period]) / period
            ema.append(sma)
            
            # Calculate subsequent EMA values
            for price in prices[period:]:
                ema_value = (price * multiplier) + (ema[-1] * (1 - multiplier))
                ema.append(ema_value)
            
            # Pad with None for the first (period-1) values
            return [None] * (period - 1) + ema
        
        close_prices = data['Close'].tolist()
        ema_20 = calculate_ema(close_prices, 20)
        
        # Calculate pivot levels based on the most recent complete trading day
        if len(data) >= 2:
            # Use the second-to-last day for pivot calculation (previous complete day)
            prev_day = data.iloc[-2]
            high = prev_day['High']
            low = prev_day['Low']
            close = prev_day['Close']
            
            # Calculate pivot levels
            pp = (high + low + close) / 3
            r1 = (2 * pp) - low
            r2 = pp + (high - low)
            s1 = (2 * pp) - high
            s2 = pp - (high - low)
            
            pivot_levels = {
                'pp': round(pp, 2),
                'r1': round(r1, 2),
                'r2': round(r2, 2),
                's1': round(s1, 2),
                's2': round(s2, 2)
            }
        else:
            pivot_levels = None
        
        # Convert to format expected by frontend (Plotly candlestick)
        chart_data = {
            'x': data.index.strftime('%Y-%m-%d').tolist(),
            'open': data['Open'].tolist(),
            'high': data['High'].tolist(),
            'low': data['Low'].tolist(),
            'close': data['Close'].tolist(),
            'ema_20': ema_20,
            'pivot_levels': pivot_levels
        }
        
        return JsonResponse(chart_data)
    except Exception as e:
        return JsonResponse({'error': str(e)})

@login_required
def get_stock_list(request):
    """API endpoint to get the current stock list data"""
    # Get all stocks from different categories
    retest_stocks = RetestStock.objects.all()
    bullish_stocks = BullishStock.objects.all()
    bearish_stocks = BearishStock.objects.all()

    # Combine all unique stocks
    all_stocks = set()
    for stock in retest_stocks:
        all_stocks.add(stock.symbol)
    for stock in bullish_stocks:
        all_stocks.add(stock.symbol)
    for stock in bearish_stocks:
        all_stocks.add(stock.symbol)

    stocks_data = []
    for symbol in all_stocks:
        stock_info = {'symbol': symbol, 'categories': []}
        if RetestStock.objects.filter(symbol=symbol).exists():
            retest = RetestStock.objects.filter(symbol=symbol).first()
            stock_info['categories'].append(f"Retest ({retest.retest_type})")
        if BullishStock.objects.filter(symbol=symbol).exists():
            stock_info['categories'].append("Bullish")
        if BearishStock.objects.filter(symbol=symbol).exists():
            stock_info['categories'].append("Bearish")
        stocks_data.append(stock_info)

    # Get current prices and percentage changes
    stock_prices = {}
    if all_stocks:
        try:
            # Fetch current data for all stocks at once
            tickers = yf.Tickers(' '.join(all_stocks))
            for symbol in all_stocks:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                    previous_close = info.get('previousClose')

                    if current_price and previous_close:
                        percent_change = ((current_price - previous_close) / previous_close) * 100
                        stock_prices[symbol] = {
                            'price': current_price,
                            'change': percent_change
                        }
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    continue
        except Exception as e:
            print(f"Error fetching stock prices: {e}")

    # Add price data to stocks_data
    for stock in stocks_data:
        symbol = stock['symbol']
        if symbol in stock_prices:
            stock['current_price'] = stock_prices[symbol]['price']
            stock['percent_change'] = stock_prices[symbol]['change']

        # Add flag for Indian stocks
        stock['is_indian_stock'] = symbol.endswith('.NS')

    # Add ATM options for bullish and bearish stocks
    for stock in stocks_data:
        symbol = stock['symbol']
        underlying = symbol.replace('.NS', '')  # Remove .NS for Indian stocks
        current_price = stock.get('current_price')

        if not current_price:
            continue

        atm_option = None
        if "Bullish" in stock['categories']:
            # Find ATM CE
            options = Instrument.objects.filter(
                tradingsymbol__startswith=underlying,
                instrument_type='CE',
                segment='NFO-OPT',
                expiry__gte=date.today()
            ).order_by('expiry')
            if options.exists():
                nearest_expiry = options.first().expiry
                expiry_options = options.filter(expiry=nearest_expiry)
                # Find strike closest to current_price
                closest_option = min(expiry_options, key=lambda x: abs(x.strike - current_price) if x.strike else float('inf'))
                atm_option = closest_option.tradingsymbol
        elif "Bearish" in stock['categories']:
            # Find ATM PE
            options = Instrument.objects.filter(
                tradingsymbol__startswith=underlying,
                instrument_type='PE',
                segment='NFO-OPT',
                expiry__gte=date.today()
            ).order_by('expiry')
            if options.exists():
                nearest_expiry = options.first().expiry
                expiry_options = options.filter(expiry=nearest_expiry)
                # Find strike closest to current_price
                closest_option = min(expiry_options, key=lambda x: abs(x.strike - current_price) if x.strike else float('inf'))
                atm_option = closest_option.tradingsymbol

        stock['atm_option'] = atm_option

    # Get last update time
    last_update = None
    if StockUpdateLog.objects.exists():
        last_update = StockUpdateLog.objects.first().last_updated.isoformat()

    return JsonResponse({
        'stocks': stocks_data,
        'last_update': last_update
    })

@login_required
def positions(request):
    """Display current positions from Kite API"""
    try:
        # Get access token from session (set during Kite OAuth login)
        access_token = request.session.get('kite_access_token')
        api_key = request.session.get('kite_api_key')

        positions_data = {'data': [], 'net': []}  # Default empty data

        if access_token and api_key:
            # Make the API request
            url = "https://api.kite.trade/portfolio/positions"
            headers = {
                "X-Kite-Version": "3",
                "Authorization": f"token {api_key}:{access_token}"
            }

            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                api_data = response.json()
                positions_data = api_data
                messages.success(request, f'Successfully loaded {len(api_data.get("data", []))} positions')
            except requests.exceptions.Timeout:
                messages.error(request, 'API request timed out. Please try again.')
            except requests.exceptions.ConnectionError:
                messages.error(request, 'Unable to connect to Kite API. Check your internet connection.')
            except requests.exceptions.HTTPError as e:
                if response.status_code == 401:
                    messages.error(request, 'Authentication failed. Please log in with Kite again.')
                elif response.status_code == 403:
                    messages.error(request, 'Access forbidden. Your API key may not have the required permissions.')
                else:
                    messages.error(request, f'API error: {response.status_code} - {response.text}')
            except json.JSONDecodeError:
                messages.error(request, 'Invalid response from positions API.')
            except Exception as e:
                messages.error(request, f'Unexpected error: {str(e)}')
        else:
            messages.warning(request, 'Please log in with Kite to view your positions.')

        return render(request, 'stocks/positions.html', {
            'positions': positions_data.get('data', {}).get('net', []),
            'net': positions_data.get('data', {}).get('net', [])
        })
    except Exception as e:
        # Catch any unexpected errors
        messages.error(request, f'An unexpected error occurred: {str(e)}')
        return render(request, 'stocks/positions.html', {
            'positions': [],
            'net': []
        })
