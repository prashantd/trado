from django.db import models

class RetestStock(models.Model):
    symbol = models.CharField(max_length=20)
    retest_type = models.CharField(max_length=10)  # UPWARD or DOWNWARD
    date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.symbol} - {self.retest_type}"

class BullishStock(models.Model):
    symbol = models.CharField(max_length=20)
    date = models.DateField()
    close = models.FloatField()
    prev_close = models.FloatField()
    rsi = models.FloatField()
    ema20 = models.FloatField()
    ema50 = models.FloatField()
    vwap = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.symbol} - Bullish"

class BearishStock(models.Model):
    symbol = models.CharField(max_length=20)
    date = models.DateField()
    close = models.FloatField()
    prev_close = models.FloatField()
    rsi = models.FloatField()
    ema20 = models.FloatField()
    ema50 = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.symbol} - Bearish"
