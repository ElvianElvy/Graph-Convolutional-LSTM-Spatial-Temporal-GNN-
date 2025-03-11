import os
import json
import argparse
from typing import Dict, Any, List, Optional

# Default configuration
DEFAULT_CONFIG = {
    "paths": {
        "model_dir": "saved_models",
        "output_dir": "predictions",
        "log_dir": "logs",
        "data_cache_dir": "data_cache"  # For caching API responses
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_size": 256,
        "num_layers": 3,
        "sequence_length": 30,
        "train_days": 365,
        "validation_split": 0.2,
        "reference_symbols": ["ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
    },
    "prediction": {
        "create_visualization": True,
        "prediction_days": 182,  # ~6 months
        "advanced_visualization": True,
        "confidence_intervals": True,
        "export_formats": ["png", "html", "csv", "json"]
    },
    "model": {
        "dropout": 0.3,
        "l2_reg": 1e-5,
        "use_attention": True,
        "graph_construction": "correlation",  # "correlation", "distance", "feature_similarity", or "manually"
        "correlation_threshold": 0.5,
        "graph_convolution_type": "standard",  # "standard" or "cheb"
        "feature_categories": ["ohlcv", "technical", "futures"],  # Which categories of features to use
        "use_dynamic_graph": False,  # Whether to use a dynamic graph that changes over time
        "scaler_type": "minmax"  # "minmax", "standard", or "robust"
    },
    "features": {
        "use_futures_data": True,  # Whether to use futures market data
        "use_open_interest": True,  # Whether to include open interest data
        "use_funding_rates": True,  # Whether to include funding rate data
        "use_long_short_ratio": True,  # Whether to include long/short ratio data
        "use_taker_buy_sell_ratio": True,  # Whether to include taker buy/sell volume ratio data
        "use_order_book": False,  # Whether to include order book data (more intensive)
        "use_advanced_features": True,  # Whether to calculate and use additional derived features
        "technicals": {
            "use_all": True,  # Whether to use all technical indicators
            "moving_averages": True,  # Moving averages: MA7, MA14, MA30
            "oscillators": True,  # RSI, MACD, etc.
            "volatility": True,  # Bollinger Bands, volatility measures
            "momentum": True  # Momentum indicators
        }
    },
    "binance_api": {
        "rate_limit_pause": 0.5,  # Seconds to pause between API calls
        "default_interval": "1d",  # Default candle interval
        "use_websockets": True,  # Whether to use WebSockets for real-time data
        "max_workers": 5,  # Maximum number of threads for parallel API requests
        "use_cache": True,  # Whether to cache API responses
        "cache_expiry": 3600  # Cache expiry time in seconds
    },
    "advanced": {
        "early_stopping": True,  # Whether to use early stopping
        "patience": 20,  # Number of epochs to wait for improvement before stopping
        "use_tensorboard": True,  # Whether to log metrics to TensorBoard
        "save_checkpoints": True,  # Whether to save checkpoints during training
        "save_best_only": True,  # Whether to save only the best model
        "mixed_precision": False,  # Whether to use mixed precision training (if available)
        "gradient_clipping": 1.0,  # Maximum gradient norm for clipping
        "use_ensemble": False,  # Whether to use an ensemble of models
        "ensemble_size": 3,  # Number of models in the ensemble
        "monte_carlo_dropout": False,  # Whether to use Monte Carlo dropout for uncertainty estimation
        "monte_carlo_samples": 100  # Number of Monte Carlo samples
    }
}

CONFIG_FILE = "config.json"


def create_default_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Create a default configuration file if it doesn't exist.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)) if os.path.dirname(config_path) else '.', exist_ok=True)
    
    # Write the default config to the file
    with open(config_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    
    print(f"Created default configuration file: {config_path}")
    return DEFAULT_CONFIG


def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Load configuration from file or create default if it doesn't exist.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    # Check if file exists and has content
    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        return create_default_config(config_path)
    
    # Try to load the config file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError:
        # If there's an error decoding JSON, create a default config
        print(f"Error parsing {config_path}. Creating default configuration.")
        return create_default_config(config_path)


def update_config(updates: Dict[str, Any], config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of updates
        config_path: Path to the configuration file
    
    Returns:
        Updated configuration dictionary
    """
    # Load current config
    config = load_config(config_path)
    
    # Recursive function to update nested dictionaries
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                update_dict(d[k], v)
            else:
                d[k] = v
    
    # Update config
    update_dict(config, updates)
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return config


def optimize_config(symbol: str, optimize_for: str = "accuracy") -> Dict[str, Any]:
    """
    Optimize configuration for a specific cryptocurrency.
    
    Args:
        symbol: Trading pair symbol
        optimize_for: Optimization target ("accuracy", "speed", or "balanced")
    
    Returns:
        Optimized configuration dictionary
    """
    # Load default config
    config = load_config()
    
    # Optimize based on target
    if optimize_for == "accuracy":
        config["training"]["epochs"] = 150
        config["training"]["batch_size"] = 32
        config["training"]["learning_rate"] = 0.0005
        config["training"]["hidden_size"] = 384
        config["training"]["num_layers"] = 3
        config["training"]["sequence_length"] = 45
        config["training"]["train_days"] = 730  # 2 years
        config["model"]["dropout"] = 0.3
        config["model"]["l2_reg"] = 1e-5
        config["model"]["correlation_threshold"] = 0.3  # More connections in the graph
        config["model"]["graph_convolution_type"] = "cheb"  # Chebyshev convolutions (higher order)
        config["model"]["scaler_type"] = "robust"  # More robust to outliers
        config["features"]["use_advanced_features"] = True
        config["advanced"]["early_stopping"] = True
        config["advanced"]["patience"] = 25
        config["advanced"]["gradient_clipping"] = 1.0
        config["advanced"]["use_ensemble"] = True
        config["advanced"]["ensemble_size"] = 3
    
    elif optimize_for == "speed":
        config["training"]["epochs"] = 50
        config["training"]["batch_size"] = 64
        config["training"]["learning_rate"] = 0.001
        config["training"]["hidden_size"] = 128
        config["training"]["num_layers"] = 2
        config["training"]["sequence_length"] = 20
        config["training"]["train_days"] = 365  # 1 year
        config["model"]["dropout"] = 0.2
        config["model"]["l2_reg"] = 1e-6
        config["model"]["correlation_threshold"] = 0.7  # Fewer connections in the graph
        config["model"]["graph_convolution_type"] = "standard"  # Simpler convolutions
        config["model"]["feature_categories"] = ["ohlcv"]  # Only use basic price features
        config["features"]["use_futures_data"] = False
        config["features"]["use_advanced_features"] = False
        config["features"]["technicals"]["use_all"] = False
        config["advanced"]["early_stopping"] = True
        config["advanced"]["patience"] = 10
        config["advanced"]["use_ensemble"] = False
    
    elif optimize_for == "balanced":
        config["training"]["epochs"] = 100
        config["training"]["batch_size"] = 48
        config["training"]["learning_rate"] = 0.0008
        config["training"]["hidden_size"] = 256
        config["training"]["num_layers"] = 2
        config["training"]["sequence_length"] = 30
        config["training"]["train_days"] = 548  # 1.5 years
        config["model"]["dropout"] = 0.25
        config["model"]["l2_reg"] = 1e-5
        config["model"]["correlation_threshold"] = 0.5  # Balanced graph connectivity
        config["model"]["feature_categories"] = ["ohlcv", "technical"]  # No futures data for balanced approach
        config["features"]["use_futures_data"] = True
        config["features"]["use_advanced_features"] = True
        config["features"]["technicals"]["use_all"] = True
        config["advanced"]["early_stopping"] = True
        config["advanced"]["patience"] = 15
        config["advanced"]["use_ensemble"] = False
    
    # Symbol-specific optimizations
    if symbol == "BTCUSDT" or symbol == "ETHUSDT":
        # Major cryptocurrencies
        config["training"]["reference_symbols"] = ["ETHUSDT" if symbol == "BTCUSDT" else "BTCUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT", "DOTUSDT"]
        config["features"]["use_futures_data"] = True  # Major cryptos have good futures data
    elif symbol.startswith("BNB"):
        # BNB and related tokens
        config["training"]["reference_symbols"] = ["BTCUSDT", "ETHUSDT", "BUSDUSDT", "CAKEUSDT", "XVSUSDT"]
    elif symbol.endswith("BTC"):
        # Altcoins traded against BTC
        ref_symbols = ["ETHBTC", "BNBBTC", "ADABTC", "XRPBTC"]
        if symbol not in ref_symbols:
            ref_symbols.append(symbol)
        config["training"]["reference_symbols"] = ref_symbols
        # Adjust feature use for BTC-paired tokens
        config["features"]["use_futures_data"] = False  # Less futures data available for BTC pairs
    
    # Check if we're likely to have futures data for this symbol
    if not (symbol.endswith("USDT") or symbol.endswith("BUSD") or symbol.endswith("USDC")):
        # Non-stablecoin quote pairs likely won't have futures data
        config["features"]["use_futures_data"] = False
        config["features"]["use_open_interest"] = False
        config["features"]["use_funding_rates"] = False
        config["features"]["use_long_short_ratio"] = False
        config["features"]["use_taker_buy_sell_ratio"] = False
        # Update feature categories accordingly
        if "futures" in config["model"]["feature_categories"]:
            config["model"]["feature_categories"].remove("futures")
    
    return config


def get_recommended_symbols(quote_asset: str = "USDT") -> List[Dict[str, Any]]:
    """
    Get a list of recommended cryptocurrency symbols with metadata.
    
    Args:
        quote_asset: Quote asset to filter by
    
    Returns:
        List of dictionaries with symbol metadata
    """
    # Define popular cryptocurrencies with metadata
    recommended = [
        {
            "symbol": "BTCUSDT",
            "name": "Bitcoin",
            "description": "The original cryptocurrency and largest by market cap",
            "category": "Major",
            "liquidity": "Very High",
            "has_futures": True
        },
        {
            "symbol": "ETHUSDT",
            "name": "Ethereum",
            "description": "Smart contract platform and second largest cryptocurrency",
            "category": "Major",
            "liquidity": "Very High",
            "has_futures": True
        },
        {
            "symbol": "BNBUSDT",
            "name": "Binance Coin",
            "description": "Native token of the Binance exchange and ecosystem",
            "category": "Exchange",
            "liquidity": "High",
            "has_futures": True
        },
        {
            "symbol": "XRPUSDT",
            "name": "Ripple",
            "description": "Digital payment protocol and cryptocurrency",
            "category": "Payment",
            "liquidity": "High",
            "has_futures": True
        },
        {
            "symbol": "ADAUSDT",
            "name": "Cardano",
            "description": "Proof-of-stake blockchain platform",
            "category": "Platform",
            "liquidity": "High",
            "has_futures": True
        },
        {
            "symbol": "SOLUSDT",
            "name": "Solana",
            "description": "High-performance blockchain supporting smart contracts",
            "category": "Platform",
            "liquidity": "High",
            "has_futures": True
        },
        {
            "symbol": "DOGEUSDT",
            "name": "Dogecoin",
            "description": "Meme-based cryptocurrency",
            "category": "Meme",
            "liquidity": "High",
            "has_futures": True
        },
        {
            "symbol": "DOTUSDT",
            "name": "Polkadot",
            "description": "Multi-chain network enabling interoperability",
            "category": "Infrastructure",
            "liquidity": "Medium",
            "has_futures": True
        },
        {
            "symbol": "LINKUSDT",
            "name": "Chainlink",
            "description": "Decentralized oracle network",
            "category": "Oracle",
            "liquidity": "Medium",
            "has_futures": True
        },
        {
            "symbol": "UNIUSDT",
            "name": "Uniswap",
            "description": "Decentralized exchange protocol",
            "category": "DeFi",
            "liquidity": "Medium",
            "has_futures": True
        },
        {
            "symbol": "AVAXUSDT",
            "name": "Avalanche",
            "description": "Layer 1 blockchain with smart contract functionality",
            "category": "Platform",
            "liquidity": "Medium",
            "has_futures": True
        },
        {
            "symbol": "MATICUSDT",
            "name": "Polygon",
            "description": "Layer 2 scaling solution for Ethereum",
            "category": "Scaling",
            "liquidity": "Medium",
            "has_futures": True
        },
        {
            "symbol": "AAVEUSDT",
            "name": "Aave",
            "description": "Decentralized lending protocol",
            "category": "DeFi",
            "liquidity": "Medium",
            "has_futures": True
        },
        {
            "symbol": "SHIBUSDT",
            "name": "Shiba Inu",
            "description": "Meme-based cryptocurrency with ecosystem aspirations",
            "category": "Meme",
            "liquidity": "Medium",
            "has_futures": True
        },
        {
            "symbol": "INJUSDT",
            "name": "Injective Protocol",
            "description": "Decentralized derivatives exchange protocol",
            "category": "DeFi",
            "liquidity": "Medium",
            "has_futures": True
        }
    ]
    
    # Filter by quote asset if needed
    if quote_asset != "USDT":
        for item in recommended:
            item["symbol"] = item["symbol"].replace("USDT", quote_asset)
            # Adjust has_futures based on quote asset
            if quote_asset not in ["USDT", "BUSD", "USDC"]:
                item["has_futures"] = False
    
    return recommended


def get_feature_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available features and their descriptions.
    
    Returns:
        Dictionary of feature information
    """
    feature_info = {
        "ohlcv": {
            "Open": "Opening price of the time interval",
            "High": "Highest price reached during time interval",
            "Low": "Lowest price reached during time interval",
            "Close": "Closing price of the time interval",
            "Volume": "Trading volume during the time interval"
        },
        "technical": {
            "MA7": "7-day moving average of close price",
            "MA14": "14-day moving average of close price",
            "MA30": "30-day moving average of close price",
            "RSI": "Relative Strength Index (momentum oscillator)",
            "MACD": "Moving Average Convergence Divergence (trend indicator)",
            "MACD_signal": "Signal line for MACD",
            "BB_middle": "Bollinger Band middle line (20-day MA)",
            "BB_upper": "Bollinger Band upper line (middle + 2*std)",
            "BB_lower": "Bollinger Band lower line (middle - 2*std)",
            "daily_return": "Daily percentage return",
            "volatility": "7-day rolling standard deviation of returns",
            "log_return": "Logarithmic daily return",
            "momentum_1d": "1-day price momentum",
            "momentum_3d": "3-day price momentum",
            "momentum_7d": "7-day price momentum"
        },
        "futures": {
            "open_interest": "Total number of outstanding futures contracts",
            "open_interest_value": "USD value of open interest",
            "funding_rate": "Periodic payment between long/short positions",
            "funding_premium": "Exponentially weighted average of funding rate",
            "long_short_ratio": "Ratio of long vs short positions",
            "taker_buy_sell_ratio": "Ratio of buyer vs seller initiated trades",
            "price_premium": "Percentage premium of futures price over spot",
            "basis": "Absolute difference between futures and spot price",
            "liquidation_volume": "Volume of liquidated positions"
        },
        "advanced": {
            "RPS": "Relative Price Strength (ratio of price to MA30)",
            "volatility_ratio": "Current volatility compared to 30-day average",
            "MACD_histogram": "Difference between MACD and signal line",
            "RSI_divergence": "Divergence between RSI and price trend",
            "OI_price_ratio": "Ratio of open interest to price",
            "OI_momentum": "3-day change in open interest",
            "funding_momentum": "3-day change in funding rate",
            "ls_ratio_momentum": "3-day change in long/short ratio",
            "market_regime": "Composite indicator of market conditions"
        }
    }
    
    return feature_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction Configuration")
    parser.add_argument("--create", action="store_true", help="Create default configuration file")
    parser.add_argument("--view", action="store_true", help="View current configuration")
    parser.add_argument("--optimize", type=str, help="Optimize configuration for symbol")
    parser.add_argument("--target", type=str, choices=["accuracy", "speed", "balanced"], default="balanced",
                        help="Optimization target")
    parser.add_argument("--recommended", action="store_true", help="Show recommended cryptocurrency symbols")
    parser.add_argument("--features", action="store_true", help="Show information about available features")
    
    args = parser.parse_args()
    
    if args.create:
        create_default_config()
        print("Default configuration created.")
    
    if args.view:
        config = load_config()
        print(json.dumps(config, indent=4))
    
    if args.optimize:
        config = optimize_config(args.optimize, args.target)
        print(f"Optimized configuration for {args.optimize} (target: {args.target}):")
        print(json.dumps(config, indent=4))
    
    if args.recommended:
        symbols = get_recommended_symbols()
        print("Recommended cryptocurrency symbols:")
        for item in symbols:
            futures_status = "with futures data" if item["has_futures"] else "no futures data"
            print(f"- {item['symbol']} ({item['name']}): {item['description']} ({futures_status})")
    
    if args.features:
        feature_info = get_feature_info()
        print("Available features by category:")
        for category, features in feature_info.items():
            print(f"\n{category.upper()} FEATURES:")
            for feature, description in features.items():
                print(f"  {feature}: {description}")