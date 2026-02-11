from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
import yfinance as yf
import pandas as pd
import json
from .models import RetestStock, BullishStock, BearishStock

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

    return render(request, 'stocks/dashboard.html', {'stocks': stocks_data})

@login_required
def get_stock_chart(request, symbol):
    # Get 1 month daily chart data
    try:
        data = yf.download(symbol, period='1mo', interval='1d')
        if data.empty:
            return JsonResponse({'error': 'No data available'})

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten columns and select only this symbol
            data.columns = [col[0] for col in data.columns]

        # Convert to format suitable for candlestick chart
        chart_data = []
        for idx, row in data.iterrows():
            chart_data.append({
                'x': str(idx.date()),  # Use date only for cleaner labels
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close'])
            })

        return JsonResponse({'data': chart_data})
    except Exception as e:
        return JsonResponse({'error': str(e)})
