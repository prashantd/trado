from django.core.management.base import BaseCommand
import sys
import os
from datetime import datetime
# Add the trader directory to the path
sys.path.append('/Users/pdeshkar/git/trader')
from nifty100_retest import find_retests, check_bullish_stocks_daily, check_bearish_stocks_daily, nifty100_symbols
from stocks.models import RetestStock, BullishStock, BearishStock, StockUpdateLog

class Command(BaseCommand):
    help = 'Update stock data from nifty100_retest.py'

    def handle(self, *args, **options):
        # Clear existing data
        RetestStock.objects.all().delete()
        BullishStock.objects.all().delete()
        BearishStock.objects.all().delete()

        today = datetime.now().date()

        # Get retest stocks
        retest_df = find_retests(nifty100_symbols)
        for _, row in retest_df.iterrows():
            RetestStock.objects.create(
                symbol=row['symbol'],
                retest_type=row['type'],
                date=today
            )

        # Get bullish stocks
        bullish_df = check_bullish_stocks_daily(nifty100_symbols)
        for _, row in bullish_df.iterrows():
            BullishStock.objects.create(
                symbol=row['symbol'],
                date=today,
                close=row['close'],
                prev_close=row['prev_close'],
                rsi=row['RSI'],
                ema20=row['EMA20'],
                ema50=row['EMA50'],
                vwap=row['VWAP']
            )

        # Get bearish stocks
        bearish_df = check_bearish_stocks_daily(nifty100_symbols)
        for _, row in bearish_df.iterrows():
            BearishStock.objects.create(
                symbol=row['symbol'],
                date=today,
                close=row['close'],
                prev_close=row['prev_close'],
                rsi=row['RSI'],
                ema20=row['EMA20'],
                ema50=row['EMA50']
            )

        # Record the update time
        StockUpdateLog.objects.all().delete()  # Keep only one record
        StockUpdateLog.objects.create()

        self.stdout.write(self.style.SUCCESS('Successfully updated stock data'))