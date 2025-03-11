import asyncio
import json
import websockets
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Any, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('binance_api')


class BinanceAPI:
    """
    Enhanced interface to the Binance API for fetching cryptocurrency data.
    Supports both REST API for historical data and WebSocket for real-time data.
    Includes methods for fetching open interest, funding rates, and other futures market data.
    """
    
    # API URLs
    REST_API_BASE_URL = "https://api.binance.com/api/v3"
    FUTURES_API_BASE_URL = "https://fapi.binance.com/fapi/v1"
    WS_API_BASE_URL = "wss://stream.binance.com:9443/ws"
    FUTURES_WS_API_BASE_URL = "wss://fstream.binance.com/ws"
    
    def __init__(self, rate_limit_pause: float = 0.5, max_workers: int = 5):
        """
        Initialize the BinanceAPI client.
        
        Args:
            rate_limit_pause: Time to pause between API calls to avoid rate limiting (in seconds)
            max_workers: Maximum number of threads for parallel API requests
        """
        self.ws = None
        self.futures_ws = None
        self.rate_limit_pause = rate_limit_pause
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Cache for available symbols
        self._symbols_cache = None
        self._symbols_cache_time = None
        self._symbols_cache_duration = 3600  # Cache duration in seconds (1 hour)
    
    async def connect_websocket(self, futures: bool = False):
        """
        Connect to Binance WebSocket API.
        
        Args:
            futures: Whether to connect to futures WebSocket API
        """
        if futures:
            if self.futures_ws is None or self.futures_ws.closed:
                self.futures_ws = await websockets.connect(self.FUTURES_WS_API_BASE_URL)
                logger.info("Connected to Binance Futures WebSocket API")
        else:
            if self.ws is None or self.ws.closed:
                self.ws = await websockets.connect(self.WS_API_BASE_URL)
                logger.info("Connected to Binance Spot WebSocket API")
    
    async def close_websocket(self, futures: bool = False):
        """
        Close the WebSocket connection.
        
        Args:
            futures: Whether to close futures WebSocket connection
        """
        if futures:
            if self.futures_ws and not self.futures_ws.closed:
                await self.futures_ws.close()
                logger.info("Closed Binance Futures WebSocket connection")
        else:
            if self.ws and not self.ws.closed:
                await self.ws.close()
                logger.info("Closed Binance Spot WebSocket connection")
    
    async def subscribe_kline_stream(self, symbol: str, interval: str = "1d", futures: bool = False):
        """
        Subscribe to kline/candlestick stream for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1m", "1h", "1d")
            futures: Whether to use futures market data
            
        Returns:
            Subscription response
        """
        await self.connect_websocket(futures)
        
        # Create subscription message
        stream_name = f"{symbol.lower()}@kline_{interval}"
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": 1
        }
        
        # Send subscription message
        if futures:
            await self.futures_ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to futures {stream_name}")
            response = await self.futures_ws.recv()
        else:
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to spot {stream_name}")
            response = await self.ws.recv()
            
        return json.loads(response)
    
    async def subscribe_multi_stream(self, symbols: List[str], interval: str = "1d", futures: bool = False):
        """
        Subscribe to multiple kline streams for different symbols.
        
        Args:
            symbols: List of trading pair symbols
            interval: Candlestick interval
            futures: Whether to use futures market data
            
        Returns:
            Subscription response
        """
        await self.connect_websocket(futures)
        
        # Create stream names
        stream_names = [f"{symbol.lower()}@kline_{interval}" for symbol in symbols]
        
        # Create subscription message
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": stream_names,
            "id": 1
        }
        
        # Send subscription message
        if futures:
            await self.futures_ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to multiple futures streams: {stream_names}")
            response = await self.futures_ws.recv()
        else:
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to multiple spot streams: {stream_names}")
            response = await self.ws.recv()
            
        return json.loads(response)
    
    async def subscribe_open_interest_stream(self, symbol: str, interval: str = "1h"):
        """
        Subscribe to open interest updates for a futures contract.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Update interval (e.g., "5m", "1h", "1d")
            
        Returns:
            Subscription response
        """
        await self.connect_websocket(True)  # Use futures WebSocket
        
        # Create subscription message
        stream_name = f"{symbol.lower()}@openInterest_{interval}"
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": 1
        }
        
        # Send subscription message
        await self.futures_ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to open interest stream: {stream_name}")
        
        # Wait for confirmation
        response = await self.futures_ws.recv()
        return json.loads(response)
    
    async def subscribe_funding_rate_stream(self, symbol: str):
        """
        Subscribe to funding rate updates for a futures contract.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            
        Returns:
            Subscription response
        """
        await self.connect_websocket(True)  # Use futures WebSocket
        
        # Create subscription message
        stream_name = f"{symbol.lower()}@markPrice@1s"  # Includes funding rate info
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": 1
        }
        
        # Send subscription message
        await self.futures_ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to funding rate stream: {stream_name}")
        
        # Wait for confirmation
        response = await self.futures_ws.recv()
        return json.loads(response)
    
    async def subscribe_book_ticker_stream(self, symbol: str, futures: bool = False):
        """
        Subscribe to book ticker updates (best bid/ask prices).
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            futures: Whether to use futures market data
            
        Returns:
            Subscription response
        """
        await self.connect_websocket(futures)
        
        # Create subscription message
        stream_name = f"{symbol.lower()}@bookTicker"
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": 1
        }
        
        # Send subscription message
        if futures:
            await self.futures_ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to futures book ticker stream: {stream_name}")
            response = await self.futures_ws.recv()
        else:
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to spot book ticker stream: {stream_name}")
            response = await self.ws.recv()
            
        return json.loads(response)
    
    async def receive_kline_updates(self, futures: bool = False):
        """
        Receive kline updates from the WebSocket.
        
        Args:
            futures: Whether to use futures WebSocket
        
        Returns:
            Parsed JSON response from the WebSocket
        """
        if futures:
            if self.futures_ws and not self.futures_ws.closed:
                response = await self.futures_ws.recv()
                return json.loads(response)
        else:
            if self.ws and not self.ws.closed:
                response = await self.ws.recv()
                return json.loads(response)
        return None
    
    async def collect_real_time_data(self, symbols: List[str], interval: str = "1d", futures: bool = False, duration_seconds: int = 300):
        """
        Collect real-time data for multiple symbols over a specified duration.
        
        Args:
            symbols: List of trading pair symbols
            interval: Candlestick interval
            futures: Whether to use futures market data
            duration_seconds: Duration to collect data in seconds
            
        Returns:
            Dictionary mapping symbols to collected data
        """
        # Subscribe to streams
        await self.subscribe_multi_stream(symbols, interval, futures)
        
        # Prepare data collection
        collected_data = {symbol: [] for symbol in symbols}
        start_time = time.time()
        
        # Collect data for the specified duration
        while time.time() - start_time < duration_seconds:
            update = await self.receive_kline_updates(futures)
            
            # Process kline data
            if 'k' in update:
                kline = update['k']
                symbol = update['s']
                
                if symbol in collected_data:
                    collected_data[symbol].append({
                        'open_time': kline['t'],
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'close_time': kline['T'],
                        'is_final': kline['x']
                    })
        
        # Close connection
        await self.close_websocket(futures)
        
        # Convert to DataFrames
        result = {}
        for symbol, data in collected_data.items():
            if data:
                df = pd.DataFrame(data)
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                result[symbol] = df
        
        return result
    
    def get_historical_klines(self, symbol: str, interval: str = "1d", 
                             start_time: Optional[int] = None, end_time: Optional[int] = None, 
                             limit: int = 500, futures: bool = False) -> pd.DataFrame:
        """
        Get historical klines (candlesticks) from Binance API.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1m", "1h", "1d")
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of candlesticks to return (max 1000)
            futures: Whether to use futures market data
        
        Returns:
            DataFrame containing historical price data
        """
        # Select appropriate API endpoint
        base_url = self.FUTURES_API_BASE_URL if futures else self.REST_API_BASE_URL
        endpoint = f"{base_url}/klines" if not futures else f"{base_url}/klines"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            # Make the request
            response = requests.get(endpoint, params=params)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the response
            klines = response.json()
            
            # Convert to DataFrame
            columns = [
                "Open time", "Open", "High", "Low", "Close", "Volume",
                "Close time", "Quote asset volume", "Number of trades",
                "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
            ]
            df = pd.DataFrame(klines, columns=columns)
            
            # Convert numeric columns
            numeric_columns = ["Open", "High", "Low", "Close", "Volume",
                            "Quote asset volume", "Taker buy base asset volume", 
                            "Taker buy quote asset volume"]
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            # Convert timestamp columns
            df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
            df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit_pause)
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_klines_chunked(self, symbol: str, interval: str = "1d", 
                                    start_time: Optional[datetime] = None, 
                                    end_time: Optional[datetime] = None, 
                                    chunk_size: int = 1000, 
                                    futures: bool = False) -> pd.DataFrame:
        """
        Get historical klines in chunks to handle large date ranges.
        
        Args:
            symbol: Trading pair symbol
            interval: Candlestick interval
            start_time: Start time as datetime
            end_time: End time as datetime
            chunk_size: Size of each chunk (max 1000)
            futures: Whether to use futures market data
            
        Returns:
            DataFrame containing historical price data
        """
        # Default start and end times
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        # Convert to timestamps in milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Initialize an empty list to store chunks
        chunks = []
        
        # Define mapping of interval to milliseconds
        interval_ms = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "8h": 8 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "3d": 3 * 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
            "1M": 30 * 24 * 60 * 60 * 1000
        }
        
        # Calculate chunk interval
        if interval in interval_ms:
            chunk_interval = interval_ms[interval] * chunk_size
        else:
            chunk_interval = 24 * 60 * 60 * 1000 * chunk_size  # Default to daily
        
        # Fetch data in chunks
        current_start = start_ms
        while current_start < end_ms:
            current_end = min(current_start + chunk_interval, end_ms)
            
            chunk = self.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end,
                limit=chunk_size,
                futures=futures
            )
            
            if not chunk.empty:
                chunks.append(chunk)
            
            # Update the start time for the next chunk
            current_start = current_end
        
        # Combine all chunks
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def get_historical_open_interest(self, symbol: str, interval: str = "1d",
                                   start_time: Optional[int] = None, 
                                   end_time: Optional[int] = None,
                                   limit: int = 500) -> pd.DataFrame:
        """
        Get historical open interest data for a futures contract.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Data interval (e.g., "5m", "1h", "1d")
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing historical open interest data
        """
        endpoint = f"{self.FUTURES_API_BASE_URL}/openInterest/hist"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "period": interval,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            # Make the request
            response = requests.get(endpoint, params=params)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            if not df.empty:
                df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'])
                df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'])
                
                # Convert timestamp column
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit_pause)
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical open interest for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_funding_rates(self, symbol: str, 
                                   start_time: Optional[int] = None,
                                   end_time: Optional[int] = None,
                                   limit: int = 500) -> pd.DataFrame:
        """
        Get historical funding rates for a futures contract.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing historical funding rate data
        """
        endpoint = f"{self.FUTURES_API_BASE_URL}/fundingRate"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            # Make the request
            response = requests.get(endpoint, params=params)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            if not df.empty:
                df['fundingRate'] = pd.to_numeric(df['fundingRate'])
                
                # Convert timestamp column
                df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit_pause)
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical funding rates for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_long_short_ratio(self, symbol: str, interval: str = "1d",
                                      start_time: Optional[int] = None,
                                      end_time: Optional[int] = None,
                                      limit: int = 500) -> pd.DataFrame:
        """
        Get historical long/short ratio for a futures contract.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Data interval (e.g., "5m", "1h", "1d")
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing historical long/short ratio data
        """
        endpoint = f"{self.FUTURES_API_BASE_URL}/globalLongShortAccountRatio"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "period": interval,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            # Make the request
            response = requests.get(endpoint, params=params)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            if not df.empty:
                df['longShortRatio'] = pd.to_numeric(df['longShortRatio'])
                df['longAccount'] = pd.to_numeric(df['longAccount'])
                df['shortAccount'] = pd.to_numeric(df['shortAccount'])
                
                # Convert timestamp column
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit_pause)
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical long/short ratio for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_historical_taker_buy_sell_ratio(self, symbol: str, interval: str = "1d",
                                          start_time: Optional[int] = None,
                                          end_time: Optional[int] = None,
                                          limit: int = 500) -> pd.DataFrame:
        """
        Get historical taker buy/sell volume ratio for a futures contract.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Data interval (e.g., "5m", "1h", "1d")
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing historical taker buy/sell ratio data
        """
        endpoint = f"{self.FUTURES_API_BASE_URL}/takerlongshortRatio"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "period": interval,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            # Make the request
            response = requests.get(endpoint, params=params)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            if not df.empty:
                df['buySellRatio'] = pd.to_numeric(df['buySellRatio'])
                df['buyVol'] = pd.to_numeric(df['buyVol'])
                df['sellVol'] = pd.to_numeric(df['sellVol'])
                
                # Convert timestamp column
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit_pause)
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical taker buy/sell ratio for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str, limit: int = 100, futures: bool = False) -> Dict[str, Any]:
        """
        Get current order book data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            limit: Depth of order book (max 1000)
            futures: Whether to use futures market data
            
        Returns:
            Dictionary containing order book data
        """
        # Select appropriate API endpoint
        base_url = self.FUTURES_API_BASE_URL if futures else self.REST_API_BASE_URL
        endpoint = f"{base_url}/depth"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        try:
            # Make the request
            response = requests.get(endpoint, params=params)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Process data into more usable format
            result = {
                'bids': [],
                'asks': []
            }
            
            for bid in data['bids']:
                result['bids'].append({
                    'price': float(bid[0]),
                    'quantity': float(bid[1])
                })
            
            for ask in data['asks']:
                result['asks'].append({
                    'price': float(ask[0]),
                    'quantity': float(ask[1])
                })
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit_pause)
            
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}
    
    def get_liquidations(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get recent liquidation orders for a futures contract.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing liquidation data
        """
        endpoint = f"{self.FUTURES_API_BASE_URL}/allForceOrders"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        try:
            # Make the request
            response = requests.get(endpoint, params=params)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            if not df.empty:
                df['price'] = pd.to_numeric(df['price'])
                df['origQty'] = pd.to_numeric(df['origQty'])
                df['executedQty'] = pd.to_numeric(df['executedQty'])
                df['averagePrice'] = pd.to_numeric(df['averagePrice'])
                
                # Convert timestamp column
                df['time'] = pd.to_datetime(df['time'], unit='ms')
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit_pause)
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching liquidations for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_aggregated_trades(self, symbol: str, start_time: Optional[int] = None,
                            end_time: Optional[int] = None, limit: int = 500) -> pd.DataFrame:
        """
        Get aggregated trade data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of records to return
            
        Returns:
            DataFrame containing aggregated trade data
        """
        endpoint = f"{self.REST_API_BASE_URL}/aggTrades"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            # Make the request
            response = requests.get(endpoint, params=params)
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'a', 'p', 'q', 'f', 'l', 'T', 'm', 'M'
            ])
            
            # Rename columns
            df.columns = [
                'agg_trade_id', 'price', 'quantity', 'first_trade_id',
                'last_trade_id', 'timestamp', 'buyer_maker', 'best_match'
            ]
            
            # Convert numeric columns
            if not df.empty:
                df['price'] = pd.to_numeric(df['price'])
                df['quantity'] = pd.to_numeric(df['quantity'])
                
                # Convert timestamp column
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Pause to respect rate limits
            time.sleep(self.rate_limit_pause)
            
            return df
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching aggregated trades for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_training_data(self, symbol: str, days: int = 365, interval: str = "1d", include_futures: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive historical data for training the model, including both spot and futures data if available.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            days: Number of days of historical data to fetch
            interval: Candlestick interval (e.g., "1m", "1h", "1d")
            include_futures: Whether to include futures market data
        
        Returns:
            Dictionary containing different types of historical data for the symbol
        """
        # Calculate start time and end time
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Initialize result dictionary
        result = {}
        
        # Fetch basic OHLCV data
        logger.info(f"Fetching OHLCV data for {symbol}...")
        df_spot = self.get_historical_klines_chunked(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            futures=False
        )
        
        if not df_spot.empty:
            result['spot_ohlcv'] = df_spot
        
        # If futures data is requested and the symbol supports futures
        if include_futures:
            # Fetch futures OHLCV data
            try:
                logger.info(f"Fetching futures OHLCV data for {symbol}...")
                df_futures = self.get_historical_klines_chunked(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    futures=True
                )
                
                if not df_futures.empty:
                    result['futures_ohlcv'] = df_futures
                
                # Fetch open interest data
                logger.info(f"Fetching open interest data for {symbol}...")
                df_open_interest = self.get_historical_open_interest(
                    symbol=symbol,
                    interval=interval,
                    start_time=int(start_time.timestamp() * 1000),
                    end_time=int(end_time.timestamp() * 1000)
                )
                
                if not df_open_interest.empty:
                    result['open_interest'] = df_open_interest
                
                # Fetch funding rate data
                logger.info(f"Fetching funding rate data for {symbol}...")
                df_funding = self.get_historical_funding_rates(
                    symbol=symbol,
                    start_time=int(start_time.timestamp() * 1000),
                    end_time=int(end_time.timestamp() * 1000)
                )
                
                if not df_funding.empty:
                    result['funding_rates'] = df_funding
                
                # Fetch long/short ratio data
                logger.info(f"Fetching long/short ratio data for {symbol}...")
                df_ls_ratio = self.get_historical_long_short_ratio(
                    symbol=symbol,
                    interval=interval,
                    start_time=int(start_time.timestamp() * 1000),
                    end_time=int(end_time.timestamp() * 1000)
                )
                
                if not df_ls_ratio.empty:
                    result['long_short_ratio'] = df_ls_ratio
                
                # Fetch taker buy/sell ratio data
                logger.info(f"Fetching taker buy/sell ratio data for {symbol}...")
                df_taker_ratio = self.get_historical_taker_buy_sell_ratio(
                    symbol=symbol,
                    interval=interval,
                    start_time=int(start_time.timestamp() * 1000),
                    end_time=int(end_time.timestamp() * 1000)
                )
                
                if not df_taker_ratio.empty:
                    result['taker_buy_sell_ratio'] = df_taker_ratio
                
            except Exception as e:
                logger.warning(f"Error fetching futures data for {symbol}: {e}")
                logger.warning(f"Continuing with spot data only...")
        
        return result
    
    def get_multi_crypto_data(self, symbols: List[str], days: int = 365, interval: str = "1d", include_futures: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get historical data for multiple cryptocurrencies in parallel.
        
        Args:
            symbols: List of trading pair symbols
            days: Number of days of historical data to fetch
            interval: Candlestick interval
            include_futures: Whether to include futures market data
            
        Returns:
            Nested dictionary mapping symbols to their data dictionaries
        """
        result = {}
        
        # Use thread pool to fetch data in parallel
        def fetch_data_for_symbol(symbol):
            try:
                data = self.get_training_data(symbol, days, interval, include_futures)
                return symbol, data
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                return symbol, {}
        
        # Submit tasks to thread pool
        futures = [self.thread_pool.submit(fetch_data_for_symbol, symbol) for symbol in symbols]
        
        # Collect results
        for future in futures:
            try:
                symbol, data = future.result()
                if data:  # Only add if we got some data
                    result[symbol] = data
            except Exception as e:
                logger.error(f"Error processing thread result: {e}")
        
        return result
    
    def merge_data_frames(self, data_dict: Dict[str, Dict[str, pd.DataFrame]], target_symbol: str) -> pd.DataFrame:
        """
        Merge different data frames for a target symbol into a single DataFrame with aligned timestamps.
        
        Args:
            data_dict: Nested dictionary of data frames from get_multi_crypto_data
            target_symbol: The main symbol to focus on
            
        Returns:
            DataFrame with all data sources merged
        """
        if target_symbol not in data_dict:
            raise ValueError(f"Target symbol {target_symbol} not found in data dictionary")
        
        target_data = data_dict[target_symbol]
        
        # Start with spot OHLCV as the base
        if 'spot_ohlcv' not in target_data:
            raise ValueError(f"Spot OHLCV data not found for {target_symbol}")
        
        base_df = target_data['spot_ohlcv'].copy()
        
        # Ensure we have a proper index
        base_df.set_index('Open time', inplace=True)
        
        # Add suffix to avoid column name conflicts
        base_df = base_df.add_suffix('_spot')
        
        # Merge futures OHLCV if available
        if 'futures_ohlcv' in target_data:
            futures_df = target_data['futures_ohlcv'].copy()
            futures_df.set_index('Open time', inplace=True)
            futures_df = futures_df.add_suffix('_futures')
            base_df = base_df.join(futures_df, how='outer')
        
        # Merge open interest if available
        if 'open_interest' in target_data:
            oi_df = target_data['open_interest'].copy()
            oi_df.set_index('timestamp', inplace=True)
            oi_df = oi_df.add_suffix('_oi')
            base_df = base_df.join(oi_df, how='outer')
        
        # Merge funding rates if available
        if 'funding_rates' in target_data:
            funding_df = target_data['funding_rates'].copy()
            funding_df.set_index('fundingTime', inplace=True)
            funding_df = funding_df.add_suffix('_funding')
            base_df = base_df.join(funding_df, how='outer')
        
        # Merge long/short ratio if available
        if 'long_short_ratio' in target_data:
            ls_df = target_data['long_short_ratio'].copy()
            ls_df.set_index('timestamp', inplace=True)
            ls_df = ls_df.add_suffix('_ls')
            base_df = base_df.join(ls_df, how='outer')
        
        # Merge taker buy/sell ratio if available
        if 'taker_buy_sell_ratio' in target_data:
            taker_df = target_data['taker_buy_sell_ratio'].copy()
            taker_df.set_index('timestamp', inplace=True)
            taker_df = taker_df.add_suffix('_taker')
            base_df = base_df.join(taker_df, how='outer')
        
        # Fill missing values by forward fill, then backward fill
        base_df = base_df.ffill().bfill()
        
        # Reset index to have the timestamp as a column
        base_df.reset_index(inplace=True)
        base_df.rename(columns={'index': 'timestamp'}, inplace=True)
        
        return base_df
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading pairs on Binance.
        
        Returns:
            List of available trading pair symbols
        """
        # Check if we have a recent cache
        current_time = time.time()
        if (self._symbols_cache is not None and 
            self._symbols_cache_time is not None and 
            current_time - self._symbols_cache_time < self._symbols_cache_duration):
            return self._symbols_cache
        
        # Fetch from API
        endpoint = f"{self.REST_API_BASE_URL}/exchangeInfo"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            
            exchange_info = response.json()
            symbols = [symbol["symbol"] for symbol in exchange_info["symbols"]]
            
            # Update cache
            self._symbols_cache = symbols
            self._symbols_cache_time = current_time
            
            return symbols
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching available symbols: {e}")
            
            # Return cached values if available, otherwise empty list
            return self._symbols_cache if self._symbols_cache is not None else []
    
    def get_futures_symbols(self) -> List[str]:
        """
        Get list of available futures contract symbols.
        
        Returns:
            List of available futures contract symbols
        """
        endpoint = f"{self.FUTURES_API_BASE_URL}/exchangeInfo"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            
            exchange_info = response.json()
            symbols = [symbol["symbol"] for symbol in exchange_info["symbols"]]
            
            return symbols
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching available futures symbols: {e}")
            return []
    
    def get_popular_symbols(self, quote_asset: str = "USDT", limit: int = 20) -> List[str]:
        """
        Get list of popular trading pairs based on volume.
        
        Args:
            quote_asset: Quote asset symbol (e.g., "USDT", "BTC")
            limit: Maximum number of symbols to return
            
        Returns:
            List of popular trading pair symbols
        """
        # Get ticker prices with 24h volume
        endpoint = f"{self.REST_API_BASE_URL}/ticker/24hr"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            
            tickers = response.json()
            
            # Filter by quote asset
            filtered_tickers = [t for t in tickers if t["symbol"].endswith(quote_asset)]
            
            # Sort by volume
            sorted_tickers = sorted(filtered_tickers, key=lambda x: float(x["volume"]), reverse=True)
            
            # Extract symbols and limit
            symbols = [t["symbol"] for t in sorted_tickers[:limit]]
            
            return symbols
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching popular symbols: {e}")
            return []
    
    def get_correlation_matrix(self, symbols: List[str], days: int = 90, interval: str = "1d") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Calculate correlation matrix for a list of cryptocurrencies.
        
        Args:
            symbols: List of trading pair symbols
            days: Number of days of historical data to use
            interval: Candlestick interval
            
        Returns:
            Tuple of (correlation_matrix, price_data)
        """
        # Get historical data for all symbols
        data_dict = self.get_multi_crypto_data(symbols, days, interval, include_futures=False)
        
        # Extract close prices
        close_prices = {}
        for symbol, data in data_dict.items():
            if 'spot_ohlcv' in data and not data['spot_ohlcv'].empty:
                close_prices[symbol] = data['spot_ohlcv'].set_index("Open time")["Close"]
        
        # Create a DataFrame with all close prices
        price_df = pd.DataFrame(close_prices)
        
        # Calculate correlation matrix
        corr_matrix = price_df.corr()
        
        return corr_matrix, data_dict