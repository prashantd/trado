from django.core.management.base import BaseCommand
import requests
import csv
import io
from datetime import datetime
from stocks.models import Instrument

class Command(BaseCommand):
    help = 'Fetch instruments from Kite API and store in database'

    def add_arguments(self, parser):
        parser.add_argument('--api_key', type=str, required=True, help='Kite API key')

    def handle(self, *args, **options):
        api_key = options['api_key']

        # Read access token from api_key.txt
        try:
            with open('/Users/pdeshkar/git/trader/api_key.txt', 'r') as f:
                access_token = f.read().strip()
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR('api_key.txt file not found'))
            return

        # Make the API request
        url = "https://api.kite.trade/instruments"
        headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token {api_key}:{access_token}"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            self.stdout.write(self.style.ERROR(f'Failed to fetch data: {e}'))
            return

        # Parse CSV
        csv_data = io.StringIO(response.text)
        reader = csv.DictReader(csv_data)

        # Clear existing instruments
        Instrument.objects.all().delete()

        count = 0
        for row in reader:
            try:
                expiry = None
                if row['expiry']:
                    expiry = datetime.strptime(row['expiry'], '%Y-%m-%d').date()

                Instrument.objects.create(
                    instrument_token=int(row['instrument_token']),
                    exchange_token=int(row['exchange_token']),
                    tradingsymbol=row['tradingsymbol'],
                    name=row.get('name', ''),
                    last_price=float(row['last_price']) if row['last_price'] else None,
                    expiry=expiry,
                    strike=float(row['strike']) if row['strike'] else None,
                    tick_size=float(row['tick_size']),
                    lot_size=int(row['lot_size']),
                    instrument_type=row['instrument_type'],
                    segment=row['segment'],
                    exchange=row['exchange']
                )
                count += 1
            except (ValueError, KeyError) as e:
                self.stdout.write(self.style.WARNING(f'Skipping row: {e}'))

        self.stdout.write(self.style.SUCCESS(f'Successfully imported {count} instruments'))