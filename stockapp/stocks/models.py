from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid

class User(AbstractUser):
    name = models.CharField(max_length=100, blank=True)
    api_key = models.CharField(max_length=100, unique=True, blank=True, null=True)

    def save(self, *args, **kwargs):
        if not self.api_key:
            self.api_key = str(uuid.uuid4())
        super().save(*args, **kwargs)

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

class StockUpdateLog(models.Model):
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Stock Update Log"
        verbose_name_plural = "Stock Update Logs"

    def __str__(self):
        return f"Last updated: {self.last_updated}"

class Instrument(models.Model):
    instrument_token = models.BigIntegerField(unique=True)
    exchange_token = models.IntegerField()
    tradingsymbol = models.CharField(max_length=50)
    name = models.CharField(max_length=100, blank=True, null=True)
    last_price = models.FloatField(blank=True, null=True)
    expiry = models.DateField(blank=True, null=True)
    strike = models.FloatField(blank=True, null=True)
    tick_size = models.FloatField()
    lot_size = models.IntegerField()
    instrument_type = models.CharField(max_length=10)
    segment = models.CharField(max_length=20)
    exchange = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.tradingsymbol} - {self.exchange}"
