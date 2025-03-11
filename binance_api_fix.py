import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the original binance_api
from data.binance_api import BinanceAPI

# Create a patch function
def patch_binance_api():
    # Store the original method
    original_get_historical_open_interest = BinanceAPI.get_historical_open_interest
    original_get_historical_long_short_ratio = BinanceAPI.get_historical_long_short_ratio
    original_get_historical_taker_buy_sell_ratio = BinanceAPI.get_historical_taker_buy_sell_ratio
    
    # Create patched methods
    def patched_get_historical_open_interest(self, symbol, interval="1d", start_time=None, end_time=None, limit=500):
        """Patched version that handles Binance's requirements for perpetual futures"""
        # Convert symbol to USDT-margined perpetual contract format if needed
        if "USDT" in symbol:
            # Get only the base asset - remove USDT
            base_asset = symbol.replace("USDT", "")
            # For major pairs, Binance uses e.g., "BTCUSDT" for spot but just "BTC" for some futures endpoints
            symbol_to_use = base_asset
        else:
            symbol_to_use = symbol
            
        try:
            return original_get_historical_open_interest(self, symbol_to_use, interval, start_time, end_time, limit)
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch open interest data for {symbol}: {e}")
            print("Using alternative method...")
            try:
                # Try using the original symbol
                return original_get_historical_open_interest(self, symbol, interval, start_time, end_time, limit)
            except Exception as e2:
                print(f"⚠️  Warning: Alternative method failed: {e2}")
                print("Returning empty DataFrame for open interest")
                import pandas as pd
                return pd.DataFrame()
    
    # Similar patches for other methods
    def patched_get_historical_long_short_ratio(self, symbol, interval="1d", start_time=None, end_time=None, limit=500):
        """Patched version that handles Binance's requirements for perpetual futures"""
        # Convert symbol to USDT-margined perpetual contract format if needed
        if "USDT" in symbol:
            base_asset = symbol.replace("USDT", "")
            symbol_to_use = base_asset
        else:
            symbol_to_use = symbol
            
        try:
            return original_get_historical_long_short_ratio(self, symbol_to_use, interval, start_time, end_time, limit)
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch long/short ratio data for {symbol}: {e}")
            print("Using alternative method...")
            try:
                return original_get_historical_long_short_ratio(self, symbol, interval, start_time, end_time, limit)
            except Exception as e2:
                print(f"⚠️  Warning: Alternative method failed: {e2}")
                print("Returning empty DataFrame for long/short ratio")
                import pandas as pd
                return pd.DataFrame()
    
    def patched_get_historical_taker_buy_sell_ratio(self, symbol, interval="1d", start_time=None, end_time=None, limit=500):
        """Patched version that handles Binance's requirements for perpetual futures"""
        # Convert symbol to USDT-margined perpetual contract format if needed
        if "USDT" in symbol:
            base_asset = symbol.replace("USDT", "")
            symbol_to_use = base_asset
        else:
            symbol_to_use = symbol
            
        try:
            return original_get_historical_taker_buy_sell_ratio(self, symbol_to_use, interval, start_time, end_time, limit)
        except Exception as e:
            print(f"⚠️  Warning: Could not fetch taker buy/sell ratio data for {symbol}: {e}")
            print("Using alternative method...")
            try:
                return original_get_historical_taker_buy_sell_ratio(self, symbol, interval, start_time, end_time, limit)
            except Exception as e2:
                print(f"⚠️  Warning: Alternative method failed: {e2}")
                print("Returning empty DataFrame for taker buy/sell ratio")
                import pandas as pd
                return pd.DataFrame()
    
    # Apply the patches
    BinanceAPI.get_historical_open_interest = patched_get_historical_open_interest
    BinanceAPI.get_historical_long_short_ratio = patched_get_historical_long_short_ratio
    BinanceAPI.get_historical_taker_buy_sell_ratio = patched_get_historical_taker_buy_sell_ratio
    
    print("✓ Binance API patched for better futures data handling")

# Patch the API immediately when imported
patch_binance_api()