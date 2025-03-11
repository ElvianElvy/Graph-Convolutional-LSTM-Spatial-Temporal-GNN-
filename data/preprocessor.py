import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch
from typing import Tuple, List, Dict, Any, Optional
import networkx as nx
import logging
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preprocessor')


class CryptoDataPreprocessor:
    """
    Enhanced preprocessor for cryptocurrency price data.
    Prepares data for training and prediction with the Graph Convolutional LSTM model.
    Handles graph structures, multiple cryptocurrencies, and advanced features like
    open interest, funding rates, and other futures market data.
    """
    
    def __init__(self, sequence_length: int = 30, num_cryptos: int = 5, 
                 prediction_length: int = 182, corr_threshold: float = 0.5,
                 scaler_type: str = "minmax", use_all_features: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            sequence_length: Number of timesteps to use for each input sequence
            num_cryptos: Maximum number of cryptocurrencies to include in the graph
            prediction_length: Number of days to predict (6 months = ~182 days)
            corr_threshold: Correlation threshold for creating edges in the graph
            scaler_type: Type of scaler to use ('minmax', 'standard', or 'robust')
            use_all_features: Whether to use all available features including futures data
        """
        self.sequence_length = sequence_length
        self.num_cryptos = num_cryptos
        self.prediction_length = prediction_length
        self.corr_threshold = corr_threshold
        self.scaler_type = scaler_type
        self.use_all_features = use_all_features
        
        # Initialize scalers based on type
        if scaler_type == "standard":
            self.create_scalers = lambda: StandardScaler()
        elif scaler_type == "robust":
            self.create_scalers = lambda: RobustScaler()
        else:  # default to minmax
            self.create_scalers = lambda: MinMaxScaler()
        
        # Scalers for different data types and cryptocurrencies
        self.price_scalers = [self.create_scalers() for _ in range(num_cryptos)]
        self.volume_scalers = [self.create_scalers() for _ in range(num_cryptos)]
        self.feature_scalers = [self.create_scalers() for _ in range(num_cryptos)]
        self.time_scaler = MinMaxScaler()
        
        # Store crypto symbols and features
        self.crypto_symbols = []
        self.feature_names = []
        self.available_features = set()
        
        # Graph structure
        self.static_adj_matrix = None
        self.dynamic_adj_matrices = None
        
        # Feature sets
        self.ohlcv_features = ["Open", "High", "Low", "Close", "Volume"]
        self.technical_features = [
            "MA7", "MA14", "MA30", "RSI", "MACD", "MACD_signal",
            "BB_middle", "BB_std", "BB_upper", "BB_lower", "daily_return", 
            "volatility", "log_return", "momentum_1d", "momentum_3d", "momentum_7d"
        ]
        self.futures_features = [
            "open_interest", "funding_rate", "long_short_ratio", 
            "taker_buy_sell_ratio", "price_premium", "basis",
            "liquidation_volume", "funding_premium", "open_interest_value"
        ]
        
        # Load configuration for dynamic graph settings
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    config = json.load(f)
                    self.use_dynamic_graph = config.get('model', {}).get('use_dynamic_graph', False)
                    self.dynamic_window_size = config.get('model', {}).get('dynamic_window_size', sequence_length)
                    self.dynamic_graph_method = config.get('model', {}).get('dynamic_graph_method', 'correlation')
            else:
                self.use_dynamic_graph = False
                self.dynamic_window_size = sequence_length
                self.dynamic_graph_method = 'correlation'
        except Exception as e:
            logger.warning(f"Error loading configuration: {e}. Using default settings.")
            self.use_dynamic_graph = False
            self.dynamic_window_size = sequence_length
            self.dynamic_graph_method = 'correlation'
    
    def process_raw_data(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Process raw data from Binance API for a single cryptocurrency.
        
        Args:
            df: Raw DataFrame from Binance API
            symbol: Cryptocurrency symbol
            
        Returns:
            Processed DataFrame with selected features
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Add symbol if provided
        if symbol is not None:
            df['symbol'] = symbol
            
            # Store symbol if not already in list
            if symbol not in self.crypto_symbols and len(self.crypto_symbols) < self.num_cryptos:
                self.crypto_symbols.append(symbol)
        
        # Select relevant columns
        if "Open time" in df.columns:
            df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]].copy()
            
            # Add timestamp column
            df.loc[:, "timestamp"] = df["Open time"].apply(lambda x: x.timestamp())
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Handle missing values
        df = df.dropna()
        
        return df
    
    def process_comprehensive_data(self, data_dict: Dict[str, Dict[str, pd.DataFrame]], symbol: str) -> pd.DataFrame:
        """
        Process comprehensive data including futures data for a symbol.
        
        Args:
            data_dict: Nested dictionary from BinanceAPI.get_training_data
            symbol: Main cryptocurrency symbol
            
        Returns:
            Processed DataFrame with all available features
        """
        if symbol not in data_dict:
            raise ValueError(f"Symbol {symbol} not found in data dictionary")
        
        symbol_data = data_dict[symbol]
        
        # Start with spot OHLCV
        if 'spot_ohlcv' not in symbol_data:
            raise ValueError(f"Spot OHLCV data not found for {symbol}")
        
        # Process basic OHLCV data
        df = self.process_raw_data(symbol_data['spot_ohlcv'], symbol)
        
        # Record the features we have available
        self.available_features.update(df.columns)
        
        if self.use_all_features:
            # Add futures market data if available
            df = self._safely_add_futures_data(df, symbol_data)
            
            # Add advanced features that combine different metrics
            df = self._add_advanced_features(df)
        
        return df
    
    def _safely_add_futures_data(self, df: pd.DataFrame, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add futures market data to the DataFrame with better error handling.
        
        Args:
            df: Processed DataFrame with OHLCV and technical indicators
            symbol_data: Dictionary containing different data types for a symbol
            
        Returns:
            DataFrame with added futures market features
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Add placeholders for futures features we expect
        futures_features = [
            'open_interest', 'open_interest_value',
            'funding_rate', 'funding_premium',
            'long_short_ratio', 'taker_buy_sell_ratio',
            'price_premium', 'basis',
            'liquidation_volume'
        ]
        
        for feature in futures_features:
            if feature not in df.columns:
                df[feature] = 0.0
                self.available_features.add(feature)
        
        # Temporarily set timestamp as index for joins
        date_column = 'Open time' if 'Open time' in df.columns else 'timestamp'
        if date_column in df.columns:
            df.set_index(date_column, inplace=True)
        
        try:
            # Add open interest if available
            if 'open_interest' in symbol_data and not symbol_data['open_interest'].empty:
                oi_df = symbol_data['open_interest'].copy()
                if 'timestamp' in oi_df.columns:
                    oi_df.set_index('timestamp', inplace=True)
                    
                    # Select and rename columns
                    if 'sumOpenInterest' in oi_df.columns:
                        oi_df = oi_df[['sumOpenInterest', 'sumOpenInterestValue']]
                        oi_df.columns = ['open_interest', 'open_interest_value']
                        
                        # Join with main DataFrame
                        df = df.join(oi_df, how='left')
                        
                        # Record the features we have available
                        self.available_features.update(['open_interest', 'open_interest_value'])
        except Exception as e:
            logger.warning(f"Error adding open interest data: {e}")
        
        try:
            # Add funding rates if available
            if 'funding_rates' in symbol_data and not symbol_data['funding_rates'].empty:
                funding_df = symbol_data['funding_rates'].copy()
                if 'fundingTime' in funding_df.columns:
                    funding_df.set_index('fundingTime', inplace=True)
                    
                    # Select and rename columns
                    if 'fundingRate' in funding_df.columns:
                        funding_df = funding_df[['fundingRate']]
                        funding_df.columns = ['funding_rate']
                        
                        # Join with main DataFrame
                        df = df.join(funding_df, how='left')
                        
                        # Record the features we have available
                        self.available_features.add('funding_rate')
                        
                        # Calculate funding premium (exponentially weighted moving average)
                        df['funding_premium'] = df['funding_rate'].ewm(span=24).mean()
                        self.available_features.add('funding_premium')
        except Exception as e:
            logger.warning(f"Error adding funding rate data: {e}")
        
        try:
            # Add long/short ratio if available
            if 'long_short_ratio' in symbol_data and not symbol_data['long_short_ratio'].empty:
                ls_df = symbol_data['long_short_ratio'].copy()
                if 'timestamp' in ls_df.columns:
                    ls_df.set_index('timestamp', inplace=True)
                    
                    # Select and rename columns
                    if 'longShortRatio' in ls_df.columns:
                        ls_df = ls_df[['longShortRatio', 'longAccount', 'shortAccount']]
                        ls_df.columns = ['long_short_ratio', 'long_account', 'short_account']
                        
                        # Join with main DataFrame
                        df = df.join(ls_df, how='left')
                        
                        # Record the features we have available
                        self.available_features.update(['long_short_ratio', 'long_account', 'short_account'])
        except Exception as e:
            logger.warning(f"Error adding long/short ratio data: {e}")
        
        try:
            # Add taker buy/sell ratio if available
            if 'taker_buy_sell_ratio' in symbol_data and not symbol_data['taker_buy_sell_ratio'].empty:
                taker_df = symbol_data['taker_buy_sell_ratio'].copy()
                if 'timestamp' in taker_df.columns:
                    taker_df.set_index('timestamp', inplace=True)
                    
                    # Select and rename columns
                    if 'buySellRatio' in taker_df.columns:
                        taker_df = taker_df[['buySellRatio', 'buyVol', 'sellVol']]
                        taker_df.columns = ['taker_buy_sell_ratio', 'buy_volume', 'sell_volume']
                        
                        # Join with main DataFrame
                        df = df.join(taker_df, how='left')
                        
                        # Record the features we have available
                        self.available_features.update(['taker_buy_sell_ratio', 'buy_volume', 'sell_volume'])
        except Exception as e:
            logger.warning(f"Error adding taker buy/sell ratio data: {e}")
        
        try:
            # Add futures price premium if futures OHLCV is available
            if 'futures_ohlcv' in symbol_data and not symbol_data['futures_ohlcv'].empty:
                futures_df = symbol_data['futures_ohlcv'].copy()
                if 'Open time' in futures_df.columns:
                    futures_df.set_index('Open time', inplace=True)
                    
                    # Select close price and rename
                    futures_df = futures_df[['Close']]
                    futures_df.columns = ['futures_close']
                    
                    # Join with main DataFrame
                    df = df.join(futures_df, how='left')
                    
                    # Record the features we have available
                    self.available_features.add('futures_close')
                    
                    # Calculate price premium (futures price - spot price) / spot price
                    if 'futures_close' in df.columns and 'Close' in df.columns:
                        df['price_premium'] = (df['futures_close'] - df['Close']) / df['Close'] * 100
                        self.available_features.add('price_premium')
                        
                        # Calculate basis (difference between spot and futures, absolute)
                        df['basis'] = df['futures_close'] - df['Close']
                        self.available_features.add('basis')
        except Exception as e:
            logger.warning(f"Error adding futures price premium data: {e}")
        
        # Reset index
        if df.index.name:
            df.reset_index(inplace=True)
        
        # Forward fill missing values, then backward fill
        df = df.ffill().bfill()
        
        return df
    
    def _add_futures_data(self, df: pd.DataFrame, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add futures market data to the DataFrame.
        
        Args:
            df: Processed DataFrame with OHLCV and technical indicators
            symbol_data: Dictionary containing different data types for a symbol
            
        Returns:
            DataFrame with added futures market features
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Temporarily set timestamp as index for joins
        date_column = 'Open time' if 'Open time' in df.columns else 'timestamp'
        if date_column in df.columns:
            df.set_index(date_column, inplace=True)
        
        # Add open interest if available
        if 'open_interest' in symbol_data and not symbol_data['open_interest'].empty:
            oi_df = symbol_data['open_interest'].copy()
            if 'timestamp' in oi_df.columns:
                oi_df.set_index('timestamp', inplace=True)
                
                # Select and rename columns
                if 'sumOpenInterest' in oi_df.columns:
                    oi_df = oi_df[['sumOpenInterest', 'sumOpenInterestValue']]
                    oi_df.columns = ['open_interest', 'open_interest_value']
                    
                    # Join with main DataFrame
                    df = df.join(oi_df, how='left')
                    
                    # Record the features we have available
                    self.available_features.update(['open_interest', 'open_interest_value'])
        
        # Add funding rates if available
        if 'funding_rates' in symbol_data and not symbol_data['funding_rates'].empty:
            funding_df = symbol_data['funding_rates'].copy()
            if 'fundingTime' in funding_df.columns:
                funding_df.set_index('fundingTime', inplace=True)
                
                # Select and rename columns
                if 'fundingRate' in funding_df.columns:
                    funding_df = funding_df[['fundingRate']]
                    funding_df.columns = ['funding_rate']
                    
                    # Join with main DataFrame
                    df = df.join(funding_df, how='left')
                    
                    # Record the features we have available
                    self.available_features.add('funding_rate')
                    
                    # Calculate funding premium (exponentially weighted moving average)
                    df['funding_premium'] = df['funding_rate'].ewm(span=24).mean()
                    self.available_features.add('funding_premium')
        
        # Add long/short ratio if available
        if 'long_short_ratio' in symbol_data and not symbol_data['long_short_ratio'].empty:
            ls_df = symbol_data['long_short_ratio'].copy()
            if 'timestamp' in ls_df.columns:
                ls_df.set_index('timestamp', inplace=True)
                
                # Select and rename columns
                if 'longShortRatio' in ls_df.columns:
                    ls_df = ls_df[['longShortRatio', 'longAccount', 'shortAccount']]
                    ls_df.columns = ['long_short_ratio', 'long_account', 'short_account']
                    
                    # Join with main DataFrame
                    df = df.join(ls_df, how='left')
                    
                    # Record the features we have available
                    self.available_features.update(['long_short_ratio', 'long_account', 'short_account'])
        
        # Add taker buy/sell ratio if available
        if 'taker_buy_sell_ratio' in symbol_data and not symbol_data['taker_buy_sell_ratio'].empty:
            taker_df = symbol_data['taker_buy_sell_ratio'].copy()
            if 'timestamp' in taker_df.columns:
                taker_df.set_index('timestamp', inplace=True)
                
                # Select and rename columns
                if 'buySellRatio' in taker_df.columns:
                    taker_df = taker_df[['buySellRatio', 'buyVol', 'sellVol']]
                    taker_df.columns = ['taker_buy_sell_ratio', 'buy_volume', 'sell_volume']
                    
                    # Join with main DataFrame
                    df = df.join(taker_df, how='left')
                    
                    # Record the features we have available
                    self.available_features.update(['taker_buy_sell_ratio', 'buy_volume', 'sell_volume'])
        
        # Add futures price premium if futures OHLCV is available
        if 'futures_ohlcv' in symbol_data and not symbol_data['futures_ohlcv'].empty:
            futures_df = symbol_data['futures_ohlcv'].copy()
            if 'Open time' in futures_df.columns:
                futures_df.set_index('Open time', inplace=True)
                
                # Select close price and rename
                futures_df = futures_df[['Close']]
                futures_df.columns = ['futures_close']
                
                # Join with main DataFrame
                df = df.join(futures_df, how='left')
                
                # Record the features we have available
                self.available_features.add('futures_close')
                
                # Calculate price premium (futures price - spot price) / spot price
                if 'futures_close' in df.columns and 'Close' in df.columns:
                    df['price_premium'] = (df['futures_close'] - df['Close']) / df['Close'] * 100
                    self.available_features.add('price_premium')
                    
                    # Calculate basis (difference between spot and futures, absolute)
                    df['basis'] = df['futures_close'] - df['Close']
                    self.available_features.add('basis')
        
        # Add liquidation volume if available (this is more complex, would need additional API request)
        # For now, we'll add a placeholder
        df['liquidation_volume'] = 0
        self.available_features.add('liquidation_volume')
        
        # Reset index
        if df.index.name:
            df.reset_index(inplace=True)
        
        # Forward fill missing values, then backward fill
        df = df.ffill().bfill()
        
        return df
    
    def process_multi_crypto_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process raw data for multiple cryptocurrencies
        
        Args:
            data_dict: Dictionary mapping crypto symbols to DataFrames
            
        Returns:
            Dictionary of processed DataFrames
        """
        processed_data = {}
        
        # Process each cryptocurrency's data
        for symbol, df in data_dict.items():
            processed_data[symbol] = self.process_raw_data(df, symbol)
        
        return processed_data
    
    def process_comprehensive_multi_crypto_data(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Process comprehensive data for multiple cryptocurrencies.
        
        Args:
            data_dict: Nested dictionary from BinanceAPI.get_multi_crypto_data
            
        Returns:
            Dictionary of processed DataFrames with all available features
        """
        processed_data = {}
        
        # Process each cryptocurrency's data
        for symbol, symbol_data in data_dict.items():
            try:
                processed_data[symbol] = self.process_comprehensive_data(data_dict, symbol)
            except Exception as e:
                logger.error(f"Error processing comprehensive data for {symbol}: {e}")
        
        return processed_data
    
    def align_multi_crypto_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align data from multiple cryptocurrencies to the same timestamps
        
        Args:
            data_dict: Dictionary of processed DataFrames
            
        Returns:
            Dictionary of aligned DataFrames
        """
        # Get common timestamps
        all_timestamps = set()
        date_column = "Open time" if "Open time" in next(iter(data_dict.values())).columns else "timestamp"
        
        for df in data_dict.values():
            all_timestamps.update(df[date_column].tolist())
        
        common_timestamps = sorted(list(all_timestamps))
        
        # Align data to common timestamps
        aligned_data = {}
        for symbol, df in data_dict.items():
            # Create a new DataFrame with all timestamps
            aligned_df = pd.DataFrame({date_column: common_timestamps})
            
            # Merge with original data
            aligned_df = pd.merge(aligned_df, df, on=date_column, how="left")
            
            # Forward fill missing values
            aligned_df = aligned_df.ffill()
            
            # Backward fill any remaining missing values at the beginning
            aligned_df = aligned_df.bfill()
            
            aligned_data[symbol] = aligned_df
        
        return aligned_data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with added technical indicators
        """
        # Copy the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Check if we have the necessary columns
        if 'Close' not in df.columns or 'Open' not in df.columns:
            return df
        
        # Calculate moving averages
        df.loc[:, "MA7"] = df["Close"].rolling(window=7).mean()
        df.loc[:, "MA14"] = df["Close"].rolling(window=14).mean()
        df.loc[:, "MA30"] = df["Close"].rolling(window=30).mean()
        
        # Calculate relative strength index (RSI)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.000001)  # Avoid division by zero
        df.loc[:, "RSI"] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df.loc[:, "MACD"] = ema12 - ema26
        df.loc[:, "MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df.loc[:, "BB_middle"] = df["Close"].rolling(window=20).mean()
        df.loc[:, "BB_std"] = df["Close"].rolling(window=20).std()
        df.loc[:, "BB_upper"] = df["BB_middle"] + (df["BB_std"] * 2)
        df.loc[:, "BB_lower"] = df["BB_middle"] - (df["BB_std"] * 2)
        
        # Daily returns
        df.loc[:, "daily_return"] = df["Close"].pct_change()
        
        # Volatility (standard deviation of returns)
        df.loc[:, "volatility"] = df["daily_return"].rolling(window=7).std()
        
        # Calculate logarithmic returns for better stationarity
        df.loc[:, "log_return"] = np.log(df["Close"]).diff()
        
        # Calculate price momentum
        df.loc[:, "momentum_1d"] = df["Close"].pct_change(1)
        df.loc[:, "momentum_3d"] = df["Close"].pct_change(3)
        df.loc[:, "momentum_7d"] = df["Close"].pct_change(7)
        
        # Record the features we added
        self.available_features.update([
            "MA7", "MA14", "MA30", "RSI", "MACD", "MACD_signal",
            "BB_middle", "BB_std", "BB_upper", "BB_lower", 
            "daily_return", "volatility", "log_return",
            "momentum_1d", "momentum_3d", "momentum_7d"
        ])
        
        return df
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced features that combine different metrics.
        
        Args:
            df: DataFrame with basic features
        
        Returns:
            DataFrame with added advanced features
        """
        df = df.copy()
        
        # Relative Price Strength (RPS) - The ratio of current price to the MA30
        if "Close" in df.columns and "MA30" in df.columns:
            df.loc[:, "RPS"] = df["Close"] / df["MA30"]
            self.available_features.add("RPS")
        
        # Volatility Ratio - Current volatility compared to 30-day average volatility
        if "volatility" in df.columns:
            df.loc[:, "volatility_MA30"] = df["volatility"].rolling(window=30).mean()
            df.loc[:, "volatility_ratio"] = df["volatility"] / df["volatility_MA30"].replace(0, 0.000001)
            self.available_features.update(["volatility_MA30", "volatility_ratio"])
        
        # MACD Histogram - The difference between MACD and Signal line
        if "MACD" in df.columns and "MACD_signal" in df.columns:
            df.loc[:, "MACD_histogram"] = df["MACD"] - df["MACD_signal"]
            self.available_features.add("MACD_histogram")
        
        # RSI Divergence - Difference between RSI and price trend
        if "RSI" in df.columns and "Close" in df.columns:
            df.loc[:, "RSI_MA5"] = df["RSI"].rolling(window=5).mean()
            df.loc[:, "price_trend"] = df["Close"].pct_change(5)
            df.loc[:, "RSI_divergence"] = (df["RSI_MA5"].pct_change(5) - df["price_trend"]) * 100
            self.available_features.update(["RSI_MA5", "price_trend", "RSI_divergence"])
        
        # If we have futures data, we can create more advanced features
        if "open_interest" in df.columns and "Close" in df.columns:
            # Open Interest to Price Ratio
            df.loc[:, "OI_price_ratio"] = df["open_interest"] / df["Close"]
            self.available_features.add("OI_price_ratio")
            
            # Open Interest Momentum
            df.loc[:, "OI_momentum"] = df["open_interest"].pct_change(3)
            self.available_features.add("OI_momentum")
        
        if "funding_rate" in df.columns:
            # Funding Rate Momentum
            df.loc[:, "funding_momentum"] = df["funding_rate"].diff(3)
            self.available_features.add("funding_momentum")
            
            # Funding Rate Extreme Indicator (1 if extreme, 0 otherwise)
            mean = df["funding_rate"].mean()
            std = df["funding_rate"].std()
            df.loc[:, "funding_extreme"] = ((df["funding_rate"] > mean + 2*std) | 
                                          (df["funding_rate"] < mean - 2*std)).astype(float)
            self.available_features.add("funding_extreme")
        
        if "long_short_ratio" in df.columns:
            # Long-Short Ratio Momentum
            df.loc[:, "ls_ratio_momentum"] = df["long_short_ratio"].pct_change(3)
            self.available_features.add("ls_ratio_momentum")
        
        if "taker_buy_sell_ratio" in df.columns:
            # Taker Buy-Sell Ratio Moving Average
            df.loc[:, "taker_ratio_MA7"] = df["taker_buy_sell_ratio"].rolling(window=7).mean()
            self.available_features.add("taker_ratio_MA7")
        
        # Market Regime Indicator - combines volatility, trend, and momentum
        if "volatility" in df.columns and "momentum_7d" in df.columns and "RSI" in df.columns:
            # Get the min and max to normalize between 0 and 1
            vol_min, vol_max = df["volatility"].min(), df["volatility"].max()
            mom_min, mom_max = df["momentum_7d"].min(), df["momentum_7d"].max()
            
            # Add small constant to avoid division by zero
            vol_range = vol_max - vol_min + 1e-8
            mom_range = mom_max - mom_min + 1e-8
            
            # Normalize components between 0 and 1
            vol_norm = (df["volatility"] - vol_min) / vol_range
            mom_norm = (df["momentum_7d"] - mom_min) / mom_range
            rsi_norm = df["RSI"] / 100
            
            # Create market regime indicator (higher values indicate bullish conditions)
            df.loc[:, "market_regime"] = mom_norm * (1 - vol_norm) * rsi_norm
            self.available_features.add("market_regime")
        
        return df
    
    def create_adjacency_matrix(self, data_dict: Dict[str, pd.DataFrame], method: str = "correlation") -> np.ndarray:
        """
        Create adjacency matrix for the cryptocurrency graph.
        
        Args:
            data_dict: Dictionary of aligned DataFrames
            method: Method for creating the adjacency matrix ('correlation', 'distance', 'feature_similarity', etc.)
            
        Returns:
            Adjacency matrix as numpy array
        """
        symbols = list(data_dict.keys())
        n = len(symbols)
        
        # Store symbols
        self.crypto_symbols = symbols
        
        if method == "correlation":
            # Extract close prices for correlation calculation
            price_data = []
            for symbol in symbols:
                price_data.append(data_dict[symbol]["Close"].values)
            
            price_data = np.array(price_data)
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(price_data)
            
            # Apply threshold to create binary adjacency matrix
            adj_matrix = np.zeros_like(corr_matrix)
            adj_matrix[np.abs(corr_matrix) > self.corr_threshold] = 1
            
            # Ensure self-loops (diagonal elements = 1)
            np.fill_diagonal(adj_matrix, 1)
            
        elif method == "distance":
            # Create adjacency based on price distance
            adj_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        adj_matrix[i, j] = 1  # Self-loop
                    else:
                        # Calculate average price distance
                        price_i = data_dict[symbols[i]]["Close"].values
                        price_j = data_dict[symbols[j]]["Close"].values
                        
                        # Normalize prices to [0, 1] range for fair comparison
                        price_i_min, price_i_max = np.min(price_i), np.max(price_i)
                        price_j_min, price_j_max = np.min(price_j), np.max(price_j)
                        
                        # Add small constant to avoid division by zero
                        price_i_range = price_i_max - price_i_min + 1e-8
                        price_j_range = price_j_max - price_j_min + 1e-8
                        
                        norm_price_i = (price_i - price_i_min) / price_i_range
                        norm_price_j = (price_j - price_j_min) / price_j_range
                        
                        # Calculate distance
                        distance = np.mean(np.abs(norm_price_i - norm_price_j))
                        
                        # Convert distance to similarity (closer = higher similarity)
                        similarity = np.exp(-distance)
                        
                        # Apply threshold
                        if similarity > self.corr_threshold:
                            adj_matrix[i, j] = 1
            
        elif method == "market_cap_weighted":
            # Create adjacency based on market cap relationship
            adj_matrix = np.ones((n, n))  # Fully connected by default
            
            # Get market caps (this would need to be provided or fetched)
            # For now we'll use a placeholder based on position in the list
            # Typically BTC has highest market cap, followed by ETH, etc.
            market_caps = np.array([(n - i) / n for i in range(n)])
            
            # Weight the edges by relative market cap
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # Higher market cap assets influence lower ones more
                        adj_matrix[i, j] = market_caps[i] / (market_caps[i] + market_caps[j])
                        
            # Binarize using threshold
            adj_matrix = (adj_matrix > self.corr_threshold).astype(float)
            np.fill_diagonal(adj_matrix, 1)  # Self-loops
            
        elif method == "feature_similarity":
            # Create adjacency based on similarity of multiple features
            adj_matrix = np.zeros((n, n))
            
            # Features to consider for similarity
            feature_list = ["daily_return", "volatility", "momentum_7d", "RSI"]
            available_features = list(set(feature_list) & self.available_features)
            
            if not available_features:
                # Fall back to correlation method if no features are available
                return self.create_adjacency_matrix(data_dict, "correlation")
            
            # Calculate similarity across multiple features
            for i in range(n):
                for j in range(n):
                    if i == j:
                        adj_matrix[i, j] = 1  # Self-loop
                    else:
                        similarity = 0
                        valid_features = 0
                        
                        for feature in available_features:
                            if feature in data_dict[symbols[i]].columns and feature in data_dict[symbols[j]].columns:
                                feat_i = data_dict[symbols[i]][feature].values
                                feat_j = data_dict[symbols[j]][feature].values
                                
                                # Handle NaN values
                                mask = ~(np.isnan(feat_i) | np.isnan(feat_j))
                                if np.sum(mask) > 0:
                                    # Compute correlation for this feature
                                    feat_corr = np.corrcoef(feat_i[mask], feat_j[mask])[0, 1]
                                    if not np.isnan(feat_corr):
                                        similarity += np.abs(feat_corr)
                                        valid_features += 1
                        
                        # Average similarity across features
                        if valid_features > 0:
                            similarity /= valid_features
                            
                            # Apply threshold
                            if similarity > self.corr_threshold:
                                adj_matrix[i, j] = 1
            
        elif method == "manually":
            # For small number of cryptocurrencies, we can manually define relationships
            # This could be based on domain knowledge (e.g., BTC influences all others)
            adj_matrix = np.ones((n, n))  # Fully connected by default
            
            # If we have BTC, ETH, and others, we can create a custom adjacency matrix
            btc_idx = -1
            eth_idx = -1
            
            for i, symbol in enumerate(symbols):
                if "BTC" in symbol:
                    btc_idx = i
                elif "ETH" in symbol:
                    eth_idx = i
            
            if btc_idx >= 0:
                # BTC influences all others but is less influenced by them
                for i in range(n):
                    if i != btc_idx:
                        adj_matrix[i, btc_idx] = 0.5  # Reduced influence on BTC
            
            if eth_idx >= 0 and btc_idx >= 0:
                # ETH has strong connection with BTC
                adj_matrix[eth_idx, btc_idx] = 1
                adj_matrix[btc_idx, eth_idx] = 1
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store the static adjacency matrix
        self.static_adj_matrix = adj_matrix
        
        return adj_matrix
    
    def create_dynamic_adjacency_matrix(self, data_dict: Dict[str, pd.DataFrame], window_size: int = 30) -> np.ndarray:
        """
        Create a dynamic adjacency matrix that changes over time based on rolling correlations.
        
        Args:
            data_dict: Dictionary of aligned DataFrames
            window_size: Size of rolling window for correlation calculation
            
        Returns:
            3D array of adjacency matrices (time, num_nodes, num_nodes)
        """
        symbols = list(data_dict.keys())
        n = len(symbols)
        
        # Store symbols
        self.crypto_symbols = symbols
        
        # Determine how many time steps we'll have
        sample_df = next(iter(data_dict.values()))
        num_timesteps = max(0, len(sample_df) - window_size + 1)
        
        if num_timesteps <= 0:
            logger.warning(f"Not enough data points for dynamic graph with window_size={window_size}. Falling back to static graph.")
            return np.expand_dims(self.create_adjacency_matrix(data_dict), axis=0)
        
        # Initialize the dynamic adjacency matrices
        dynamic_adj = np.zeros((num_timesteps, n, n))
        
        # For each time window, calculate correlation-based adjacency
        for t in range(num_timesteps):
            # Extract close prices for the current window
            window_prices = np.zeros((n, window_size))
            
            for i, symbol in enumerate(symbols):
                window_prices[i] = data_dict[symbol]["Close"].values[t:t+window_size]
            
            # Calculate correlation matrix for this window
            corr_matrix = np.corrcoef(window_prices)
            
            # Apply threshold to create binary adjacency matrix
            adj_matrix = np.zeros_like(corr_matrix)
            adj_matrix[np.abs(corr_matrix) > self.corr_threshold] = 1
            
            # Ensure self-loops (diagonal elements = 1)
            np.fill_diagonal(adj_matrix, 1)
            
            # Store this adjacency matrix
            dynamic_adj[t] = adj_matrix
        
        # Store the dynamic adjacency matrices
        self.dynamic_adj_matrices = dynamic_adj
        
        return dynamic_adj
    
    def select_features(self, feature_categories: List[str] = None) -> List[str]:
        """
        Select features based on categories and availability.
        
        Args:
            feature_categories: List of feature categories to include ('ohlcv', 'technical', 'futures')
                               If None, all available features will be used
        
        Returns:
            List of selected feature names
        """
        if feature_categories is None:
            feature_categories = ['ohlcv', 'technical']
            if self.use_all_features:
                feature_categories.append('futures')
        
        selected_features = []
        
        if 'ohlcv' in feature_categories:
            selected_features.extend([f for f in self.ohlcv_features if f in self.available_features])
        
        if 'technical' in feature_categories:
            selected_features.extend([f for f in self.technical_features if f in self.available_features])
        
        if 'futures' in feature_categories and self.use_all_features:
            selected_features.extend([f for f in self.futures_features if f in self.available_features])
        
        # Make sure we have a consistent set of features
        self.feature_names = selected_features
        
        return selected_features
    
    def prepare_graph_data(self, data_dict: Dict[str, pd.DataFrame], 
                         adj_matrix: Optional[np.ndarray] = None,
                         feature_categories: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for graph-based model training or prediction.
        
        Args:
            data_dict: Dictionary of aligned DataFrames
            adj_matrix: Adjacency matrix (optional, will use stored one if None)
            feature_categories: List of feature categories to include
            
        Returns:
            Tuple of (X, adj, y) tensors
            X: Feature tensor of shape (batch_size, seq_len, num_nodes, feat_dim)
            adj: Adjacency matrix of shape (batch_size, num_nodes, num_nodes) or (batch_size, seq_len, num_nodes, num_nodes)
            y: Target values of shape (batch_size, num_forecasts)
        """
        # Check if dynamic graphs are enabled
        use_dynamic_graph = self.use_dynamic_graph
        
        if use_dynamic_graph:
            # Create dynamic adjacency matrices
            if self.dynamic_adj_matrices is None or adj_matrix is None:
                dynamic_adj_matrix = self.create_dynamic_adjacency_matrix(
                    data_dict, 
                    window_size=self.dynamic_window_size
                )
                adj_matrix = dynamic_adj_matrix  # This will be a 3D array
        else:
            # Use static adjacency matrix
            if adj_matrix is None and self.static_adj_matrix is None:
                # Create adjacency matrix if not provided
                adj_matrix = self.create_adjacency_matrix(data_dict)
            elif adj_matrix is None:
                adj_matrix = self.static_adj_matrix
        
        symbols = list(data_dict.keys())
        n = len(symbols)
        
        # Select features for model input
        selected_features = self.select_features(feature_categories)
        
        # If no features were selected, raise an error
        if not selected_features:
            raise ValueError("No features selected. Check feature availability and categories.")
        
        # Prepare feature data
        all_features = []
        all_timestamps = []
        all_targets = []
        
        # Extract and scale features for each cryptocurrency
        for i, symbol in enumerate(symbols):
            df = data_dict[symbol]
            
            # Check if all selected features are available
            missing_features = [f for f in selected_features if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features for {symbol}: {missing_features}")
                # Skip this cryptocurrency if critical features are missing
                if any(f in self.ohlcv_features for f in missing_features):
                    continue
            
            # Use only available features from the selected ones
            available_selected = [f for f in selected_features if f in df.columns]
            
            # Extract features
            feature_data = df[available_selected].values
            timestamps = df["timestamp"].values.reshape(-1, 1) if "timestamp" in df.columns else None
            
            # Initialize feature scaler if needed
            if not hasattr(self.feature_scalers[i], "data_min_") or not hasattr(self.feature_scalers[i], "data_max_"):
                self.feature_scalers[i].fit(feature_data)
            
            # Scale features
            feature_data_scaled = self.feature_scalers[i].transform(feature_data)
            
            # Scale timestamps if available
            if timestamps is not None:
                if not hasattr(self.time_scaler, "data_min_") or not hasattr(self.time_scaler, "data_max_"):
                    self.time_scaler.fit(timestamps)
                timestamps_scaled = self.time_scaler.transform(timestamps)
                all_timestamps.append(timestamps_scaled)
            
            all_features.append(feature_data_scaled)
            
            # Create target array (next 6 months of daily open and close prices)
            if i == 0:  # Only need targets for the main cryptocurrency
                target_data = []
                for j in range(len(df) - self.sequence_length - self.prediction_length):
                    target_sequence = []
                    for k in range(1, self.prediction_length + 1):
                        idx = j + self.sequence_length + k - 1
                        if idx < len(df) and "Open" in df.columns and "Close" in df.columns:
                            target_sequence.extend([df["Open"].iloc[idx], df["Close"].iloc[idx]])
                        else:
                            # Pad with last known values if we reach the end of data
                            last_open = df["Open"].iloc[-1] if "Open" in df.columns else 0
                            last_close = df["Close"].iloc[-1] if "Close" in df.columns else 0
                            target_sequence.extend([last_open, last_close])
                    
                    target_data.append(target_sequence)
                
                if target_data:
                    all_targets = np.array(target_data)
        
        # Create sequences
        X, adj, y = self._create_graph_sequences(all_features, all_targets, adj_matrix, use_dynamic_graph)
        
        return X, adj, y
    
    def _create_graph_sequences(self, all_features: List[np.ndarray], all_targets: np.ndarray, 
                               adj_matrix: np.ndarray, use_dynamic_graph: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create sequences for graph-based model.
        
        Args:
            all_features: List of feature arrays for each cryptocurrency
            all_targets: Array of target values for the main cryptocurrency
            adj_matrix: Either static or dynamic adjacency matrix
            use_dynamic_graph: Whether to use dynamic graph structure
            
        Returns:
            Tuple of (X, adj, y) tensors
        """
        # Extract dimensions
        n = len(all_features)  # Number of cryptocurrencies
        
        if n == 0:
            raise ValueError("No valid feature data available")
        
        # Find minimum length among all feature arrays
        min_length = min(feat.shape[0] for feat in all_features)
        
        # Truncate all features to the same length
        features_truncated = [feat[:min_length] for feat in all_features]
        
        T = features_truncated[0].shape[0]  # Number of timestamps
        d = features_truncated[0].shape[1]  # Feature dimension
        
        # Create sequences
        X, y = [], []
        
        # For each possible sequence
        for i in range(T - self.sequence_length - self.prediction_length):
            # Input sequence for each cryptocurrency
            x_sequence = []
            
            for j in range(n):
                x_sequence.append(features_truncated[j][i:i+self.sequence_length])
            
            # Stack to create graph node features
            x_graph = np.stack(x_sequence, axis=1)  # (seq_len, num_nodes, feat_dim)
            
            X.append(x_graph)
            
            # Target is the main cryptocurrency's future prices
            if len(all_targets) > 0 and i < len(all_targets):
                y.append(all_targets[i])
        
        # Convert to tensors
        X = torch.tensor(np.array(X), dtype=torch.float32)  # (batch_size, seq_len, num_nodes, feat_dim)
        
        # Handle adjacency matrix based on whether we're using dynamic graphs
        if use_dynamic_graph:
            # For dynamic graph, we need adjacency matrix for each time step in each batch
            batch_size = len(X)
            seq_len = X.shape[1]
            
            # If adj_matrix is already 3D (time, num_nodes, num_nodes)
            if len(adj_matrix.shape) == 3:
                # Extract the relevant time steps for each batch
                dynamic_adj = []
                for i in range(batch_size):
                    start_idx = i
                    end_idx = start_idx + seq_len
                    if end_idx <= len(adj_matrix):
                        dynamic_adj.append(adj_matrix[start_idx:end_idx])
                    else:
                        # If we don't have enough time steps, use the last ones
                        pad_length = end_idx - len(adj_matrix)
                        padded_adj = np.concatenate([
                            adj_matrix[start_idx:],
                            np.repeat(adj_matrix[-1:], pad_length, axis=0)
                        ])
                        dynamic_adj.append(padded_adj)
                
                adj = torch.tensor(np.array(dynamic_adj), dtype=torch.float32)  # (batch_size, seq_len, num_nodes, num_nodes)
            else:
                # If adj_matrix is still 2D, just repeat it for each time step
                adj = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                adj = adj.repeat(batch_size, seq_len, 1, 1)  # (batch_size, seq_len, num_nodes, num_nodes)
        else:
            # For static graph, just repeat the adjacency matrix for each batch
            adj = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0).repeat(len(X), 1, 1)  # (batch_size, num_nodes, num_nodes)
        
        if len(all_targets) > 0:
            y = torch.tensor(np.array(y), dtype=torch.float32)  # (batch_size, num_forecasts)
        else:
            y = torch.tensor([])
        
        return X, adj, y
    
    def prepare_single_prediction(self, data_dict: Dict[str, pd.DataFrame], 
                                feature_categories: List[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for a single prediction.
        
        Args:
            data_dict: Dictionary of aligned DataFrames
            feature_categories: List of feature categories to include
            
        Returns:
            Tuple of (X, adj) tensors for prediction
        """
        # Check if we have enough data
        main_symbol = list(data_dict.keys())[0]  # First symbol is the main one
        if len(data_dict[main_symbol]) < self.sequence_length:
            raise ValueError(f"Not enough data for prediction. Need at least {self.sequence_length} data points, but got {len(data_dict[main_symbol])}.")
        
        # Create adjacency matrix if needed
        use_dynamic_graph = self.use_dynamic_graph
        
        if use_dynamic_graph:
            if self.dynamic_adj_matrices is None:
                adj_matrix = self.create_dynamic_adjacency_matrix(
                    data_dict, 
                    window_size=self.dynamic_window_size
                )
            else:
                adj_matrix = self.dynamic_adj_matrices
        else:
            if self.static_adj_matrix is None:
                adj_matrix = self.create_adjacency_matrix(data_dict)
            else:
                adj_matrix = self.static_adj_matrix
        
        # Get the most recent data for the sequence
        recent_data = {}
        for symbol, df in data_dict.items():
            recent_data[symbol] = df.iloc[-self.sequence_length:].copy()
        
        # Prepare graph data without targets
        X, adj, _ = self.prepare_graph_data(recent_data, adj_matrix, feature_categories)
        
        return X, adj
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform the scaled predictions back to original scale.
        
        Args:
            predictions: Predicted values (batch_size, num_forecasts)
                        representing open and close prices for next 6 months
        
        Returns:
            Original scale predictions
        """
        # Reshape predictions for inverse transform
        batch_size = predictions.shape[0]
        num_days = self.prediction_length
        
        # If we don't have a fitted price scaler, return the predictions as-is
        if not hasattr(self.price_scalers[0], "data_min_") or not hasattr(self.price_scalers[0], "data_max_"):
            logger.warning("Price scaler not fitted. Returning unscaled predictions.")
            return predictions
        
        # Check if we're doing single variable prediction (price only)
        if predictions.shape[1] == num_days:
            # Simple case: direct inverse transform
            reshaped_preds = predictions.reshape(-1, 1)
            original_values = self.price_scalers[0].inverse_transform(reshaped_preds)
            return original_values.reshape(batch_size, num_days)
        
        # Handle open/close pairs
        if predictions.shape[1] == num_days * 2:
            # Create temporary arrays for open, high, low, close
            dummy_array = np.zeros((batch_size * num_days, 4))
            
            for i in range(num_days):
                # Extract open and close from predictions
                open_idx = i * 2  # Even indices for open prices
                close_idx = i * 2 + 1  # Odd indices for close prices
                
                day_idx = np.arange(batch_size) + i * batch_size
                
                # Assign open and close
                dummy_array[day_idx, 0] = predictions[:, open_idx]  # Open
                dummy_array[day_idx, 3] = predictions[:, close_idx]  # Close
                
                # For high and low, use max and min of open/close
                dummy_array[day_idx, 1] = np.maximum(predictions[:, open_idx], predictions[:, close_idx])  # High
                dummy_array[day_idx, 2] = np.minimum(predictions[:, open_idx], predictions[:, close_idx])  # Low
            
            # Inverse transform using the first cryptocurrency's price scaler
            original_values = self.price_scalers[0].inverse_transform(dummy_array)
            
            # Extract just open and close and reshape
            result = np.zeros((batch_size, num_days * 2))
            for i in range(num_days):
                day_idx = np.arange(batch_size) + i * batch_size
                result[:, i*2] = original_values[day_idx, 0]      # Open
                result[:, i*2+1] = original_values[day_idx, 3]    # Close
            
            return result
        
        # Other cases - just reshape and inverse transform directly
        logger.warning(f"Unexpected prediction shape: {predictions.shape}. Attempting direct inverse transform.")
        try:
            reshaped_preds = predictions.reshape(-1, predictions.shape[-1] // num_days)
            original_values = self.price_scalers[0].inverse_transform(reshaped_preds)
            return original_values.reshape(batch_size, -1)
        except Exception as e:
            logger.error(f"Error in inverse transform: {e}")
            return predictions
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get the stored adjacency matrix"""
        if self.static_adj_matrix is None and self.dynamic_adj_matrices is None:
            raise ValueError("Adjacency matrix has not been created yet")
        
        if self.use_dynamic_graph and self.dynamic_adj_matrices is not None:
            return self.dynamic_adj_matrices
        else:
            return self.static_adj_matrix
    
    def get_crypto_symbols(self) -> List[str]:
        """Get the list of cryptocurrency symbols"""
        return self.crypto_symbols
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names"""
        return self.feature_names
    
    def get_available_features(self) -> List[str]:
        """Get the list of available features"""
        return list(self.available_features)