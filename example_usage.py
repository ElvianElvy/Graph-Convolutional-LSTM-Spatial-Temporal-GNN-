import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import pickle
import asyncio
from datetime import datetime, timedelta

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.graph_conv_lstm import CryptoGraphConvLSTM
from train import train_model
from predict import predict_crypto_prices
from config import load_config, update_config


def example_basic_usage():
    """
    Example of basic usage of the system with the enhanced Binance data.
    """
    print("=== Basic Usage Example ===")
    
    # Load configuration
    config = load_config()
    
    # Update configuration to use futures data
    config["features"]["use_futures_data"] = True
    config["features"]["use_open_interest"] = True
    config["features"]["use_funding_rates"] = True
    config["model"]["feature_categories"] = ["ohlcv", "technical", "futures"]
    update_config(config)
    
    # Main cryptocurrency to predict
    symbol = "BTCUSDT"
    
    # Reference cryptocurrencies for the graph
    reference_symbols = ["ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
    
    print(f"Training model for {symbol} with reference cryptocurrencies: {reference_symbols}")
    
    # Train the model
    model_path, preprocessor_path = train_model(
        symbol=symbol,
        reference_symbols=reference_symbols,
        epochs=5,  # Just a few epochs for this example
        batch_size=32,
        learning_rate=0.001,
        hidden_size=128,
        num_layers=2,
        sequence_length=30,
        train_days=90,  # Just 90 days for this example
        save_dir="saved_models"
    )
    
    # Make predictions
    df_pred, summary = predict_crypto_prices(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        output_dir="predictions",
        advanced_viz=True
    )
    
    # Print prediction summary
    print("\nPrediction Summary:")
    print(f"Symbol: {summary['symbol']}")
    print(f"Prediction Period: {summary['prediction_period']['start']} to {summary['prediction_period']['end']}")
    print(f"Overall Trend: {summary['overall']['trend']}")
    print(f"Price Change: {summary['overall']['price_change']:.2f} ({summary['overall']['price_change_pct']:.2f}%)")
    
    print("\nKey Milestones:")
    for period, info in summary['milestones'].items():
        if info['data']:
            print(f"  {period.replace('_', ' ').title()}: {info['data']['close']:.2f} " + 
                  f"({'+' if info['data']['change_from_start'] >= 0 else ''}{info['data']['change_from_start']:.2f}%)")


async def example_advanced_data_fetching():
    """
    Example of advanced data fetching with the enhanced Binance API.
    Shows how to fetch and visualize various types of futures market data.
    """
    print("=== Advanced Data Fetching Example ===")
    
    # Initialize Binance API
    api = BinanceAPI()
    
    # Main cryptocurrency to analyze
    symbol = "BTCUSDT"
    
    # Fetch comprehensive data including futures market data
    start_time = datetime.now() - timedelta(days=30)  # Last 30 days
    end_time = datetime.now()
    
    print(f"Fetching comprehensive data for {symbol}...")
    
    # Fetch open interest
    open_interest_df = api.get_historical_open_interest(
        symbol=symbol,
        interval="1d",
        start_time=int(start_time.timestamp() * 1000),
        end_time=int(end_time.timestamp() * 1000)
    )
    
    # Fetch funding rates
    funding_df = api.get_historical_funding_rates(
        symbol=symbol,
        start_time=int(start_time.timestamp() * 1000),
        end_time=int(end_time.timestamp() * 1000)
    )
    
    # Fetch long/short ratio
    ls_ratio_df = api.get_historical_long_short_ratio(
        symbol=symbol,
        interval="1d",
        start_time=int(start_time.timestamp() * 1000),
        end_time=int(end_time.timestamp() * 1000)
    )
    
    # Fetch taker buy/sell ratio
    taker_ratio_df = api.get_historical_taker_buy_sell_ratio(
        symbol=symbol,
        interval="1d",
        start_time=int(start_time.timestamp() * 1000),
        end_time=int(end_time.timestamp() * 1000)
    )
    
    # Fetch price data
    price_df = api.get_historical_klines(
        symbol=symbol,
        interval="1d",
        start_time=int(start_time.timestamp() * 1000),
        end_time=int(end_time.timestamp() * 1000)
    )
    
    print("Data fetched successfully!")
    
    # Visualize the data
    plt.figure(figsize=(14, 20))
    
    # Plot 1: Price
    plt.subplot(5, 1, 1)
    plt.plot(price_df["Open time"], price_df["Close"], label="Close Price")
    plt.title(f"{symbol} Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USDT)")
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Open Interest
    plt.subplot(5, 1, 2)
    if not open_interest_df.empty:
        plt.plot(open_interest_df["timestamp"], open_interest_df["sumOpenInterest"], label="Open Interest")
        plt.title(f"{symbol} Open Interest")
        plt.xlabel("Date")
        plt.ylabel("Open Interest")
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Open Interest data available", horizontalalignment='center', verticalalignment='center')
    
    # Plot 3: Funding Rate
    plt.subplot(5, 1, 3)
    if not funding_df.empty:
        plt.plot(funding_df["fundingTime"], funding_df["fundingRate"] * 100, label="Funding Rate %")
        plt.title(f"{symbol} Funding Rate")
        plt.xlabel("Date")
        plt.ylabel("Funding Rate (%)")
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Funding Rate data available", horizontalalignment='center', verticalalignment='center')
    
    # Plot 4: Long/Short Ratio
    plt.subplot(5, 1, 4)
    if not ls_ratio_df.empty:
        plt.plot(ls_ratio_df["timestamp"], ls_ratio_df["longShortRatio"], label="Long/Short Ratio")
        plt.title(f"{symbol} Long/Short Ratio")
        plt.xlabel("Date")
        plt.ylabel("Ratio")
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Long/Short Ratio data available", horizontalalignment='center', verticalalignment='center')
    
    # Plot 5: Taker Buy/Sell Ratio
    plt.subplot(5, 1, 5)
    if not taker_ratio_df.empty:
        plt.plot(taker_ratio_df["timestamp"], taker_ratio_df["buySellRatio"], label="Taker Buy/Sell Ratio")
        plt.title(f"{symbol} Taker Buy/Sell Ratio")
        plt.xlabel("Date")
        plt.ylabel("Ratio")
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Taker Buy/Sell Ratio data available", horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs("analysis", exist_ok=True)
    plt.savefig(f"analysis/{symbol}_futures_data_analysis.png")
    plt.close()
    
    print(f"Visualization saved to analysis/{symbol}_futures_data_analysis.png")


def example_feature_exploration():
    """
    Example of exploring available features and their impact on prediction.
    """
    print("=== Feature Exploration Example ===")
    
    # Initialize components
    api = BinanceAPI()
    
    # Main cryptocurrency to explore
    symbol = "BTCUSDT"
    
    # Fetch training data with all available features
    data_dict = api.get_training_data(symbol=symbol, days=60, include_futures=True)
    
    # Initialize preprocessor
    preprocessor = CryptoDataPreprocessor(use_all_features=True)
    
    # Process data
    processed_df = preprocessor.process_comprehensive_data({"BTCUSDT": data_dict}, symbol)
    
    # Get available features
    available_features = preprocessor.get_available_features()
    
    print(f"Available features for {symbol}:")
    for feature in sorted(available_features):
        print(f"  - {feature}")
    
    # Visualize key features
    plt.figure(figsize=(14, 14))
    
    # Group features by category
    ohlcv_features = [f for f in preprocessor.ohlcv_features if f in available_features]
    technical_features = [f for f in preprocessor.technical_features if f in available_features]
    futures_features = [f for f in preprocessor.futures_features if f in available_features]
    
    # Plot OHLCV features
    subplot_idx = 1
    
    # Plot price data
    plt.subplot(4, 1, subplot_idx)
    plt.plot(processed_df["Open time"], processed_df["Open"], label="Open")
    plt.plot(processed_df["Open time"], processed_df["High"], label="High")
    plt.plot(processed_df["Open time"], processed_df["Low"], label="Low")
    plt.plot(processed_df["Open time"], processed_df["Close"], label="Close")
    plt.title(f"{symbol} OHLC Data")
    plt.xlabel("Date")
    plt.ylabel("Price (USDT)")
    plt.grid(True)
    plt.legend()
    
    subplot_idx += 1
    
    # Plot key technical indicators
    plt.subplot(4, 1, subplot_idx)
    plt.plot(processed_df["Open time"], processed_df["RSI"], label="RSI")
    if "MACD" in available_features:
        plt.plot(processed_df["Open time"], processed_df["MACD"], label="MACD")
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
    plt.title(f"{symbol} Technical Indicators")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    
    subplot_idx += 1
    
    # Plot key futures data if available
    if futures_features:
        plt.subplot(4, 1, subplot_idx)
        
        # Plot funding rate if available
        if "funding_rate" in available_features:
            plt.plot(processed_df["Open time"], processed_df["funding_rate"], label="Funding Rate")
        
        # Plot long/short ratio if available
        if "long_short_ratio" in available_features:
            plt.plot(processed_df["Open time"], processed_df["long_short_ratio"], label="Long/Short Ratio")
        
        # Plot price premium if available
        if "price_premium" in available_features:
            plt.plot(processed_df["Open time"], processed_df["price_premium"], label="Price Premium %")
        
        plt.title(f"{symbol} Futures Market Data")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        
        subplot_idx += 1
    
    # Plot correlation matrix for selected features
    plt.subplot(4, 1, subplot_idx)
    
    # Select key features
    key_features = ["Close", "Volume", "RSI", "volatility"]
    
    # Add futures features if available
    if "funding_rate" in available_features:
        key_features.append("funding_rate")
    if "open_interest" in available_features:
        key_features.append("open_interest")
    if "long_short_ratio" in available_features:
        key_features.append("long_short_ratio")
    
    # Calculate correlation matrix
    correlation_matrix = processed_df[key_features].corr()
    
    # Create heatmap
    im = plt.imshow(correlation_matrix, cmap="coolwarm")
    plt.colorbar(im)
    plt.title(f"{symbol} Feature Correlation Matrix")
    
    # Add feature labels
    plt.xticks(range(len(key_features)), key_features, rotation=45)
    plt.yticks(range(len(key_features)), key_features)
    
    # Add correlation values in the cells
    for i in range(len(key_features)):
        for j in range(len(key_features)):
            text = plt.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black")
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs("analysis", exist_ok=True)
    plt.savefig(f"analysis/{symbol}_feature_exploration.png")
    plt.close()
    
    print(f"Feature exploration visualization saved to analysis/{symbol}_feature_exploration.png")


def example_model_comparison():
    """
    Example of comparing model performance with and without futures data.
    """
    print("=== Model Comparison Example ===")
    
    # Load configuration
    config = load_config()
    
    # Main cryptocurrency to predict
    symbol = "BTCUSDT"
    
    # Reference cryptocurrencies for the graph
    reference_symbols = ["ETHUSDT", "BNBUSDT"]
    
    # Training parameters
    epochs = 5  # Just a few epochs for this example
    batch_size = 32
    hidden_size = 128
    num_layers = 2
    sequence_length = 30
    train_days = 60  # Just 60 days for this example
    
    # First model: Without futures data
    print("\n1. Training model WITHOUT futures data...")
    config["features"]["use_futures_data"] = False
    config["model"]["feature_categories"] = ["ohlcv", "technical"]
    update_config(config)
    
    # Train the model
    model_path_1, preprocessor_path_1 = train_model(
        symbol=symbol,
        reference_symbols=reference_symbols,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        sequence_length=sequence_length,
        train_days=train_days,
        save_dir="saved_models"
    )
    
    # Second model: With futures data
    print("\n2. Training model WITH futures data...")
    config["features"]["use_futures_data"] = True
    config["features"]["use_open_interest"] = True
    config["features"]["use_funding_rates"] = True
    config["model"]["feature_categories"] = ["ohlcv", "technical", "futures"]
    update_config(config)
    
    # Train the model
    model_path_2, preprocessor_path_2 = train_model(
        symbol=symbol,
        reference_symbols=reference_symbols,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        sequence_length=sequence_length,
        train_days=train_days,
        save_dir="saved_models"
    )
    
    # Make predictions with both models
    print("\nMaking predictions with both models...")
    
    # Predictions without futures data
    df_pred_1, summary_1 = predict_crypto_prices(
        model_path=model_path_1,
        preprocessor_path=preprocessor_path_1,
        output_dir="predictions/without_futures",
        advanced_viz=True
    )
    
    # Predictions with futures data
    df_pred_2, summary_2 = predict_crypto_prices(
        model_path=model_path_2,
        preprocessor_path=preprocessor_path_2,
        output_dir="predictions/with_futures",
        advanced_viz=True
    )
    
    # Compare predictions
    print("\nComparison of model predictions:")
    print("\nModel WITHOUT futures data:")
    print(f"Overall Trend: {summary_1['overall']['trend']}")
    print(f"Price Change: {summary_1['overall']['price_change']:.2f} ({summary_1['overall']['price_change_pct']:.2f}%)")
    
    print("\nModel WITH futures data:")
    print(f"Overall Trend: {summary_2['overall']['trend']}")
    print(f"Price Change: {summary_2['overall']['price_change']:.2f} ({summary_2['overall']['price_change_pct']:.2f}%)")
    
    # Calculate difference
    price_diff = summary_2['overall']['price_change'] - summary_1['overall']['price_change']
    pct_diff = summary_2['overall']['price_change_pct'] - summary_1['overall']['price_change_pct']
    
    print("\nDifference (futures - no futures):")
    print(f"Price Change Difference: {price_diff:.2f} ({pct_diff:.2f}%)")
    
    # Create comparison visualization
    plt.figure(figsize=(14, 10))
    
    # Plot price predictions
    plt.subplot(2, 1, 1)
    plt.plot(df_pred_1["Date"], df_pred_1["Predicted Close"], label="Without Futures Data", color="blue")
    plt.plot(df_pred_2["Date"], df_pred_2["Predicted Close"], label="With Futures Data", color="red")
    plt.title(f"{symbol} Price Prediction Comparison")
    plt.xlabel("Date")
    plt.ylabel("Price (USDT)")
    plt.grid(True)
    plt.legend()
    
    # Plot difference
    plt.subplot(2, 1, 2)
    diff = df_pred_2["Predicted Close"] - df_pred_1["Predicted Close"]
    plt.plot(df_pred_1["Date"], diff, label="Difference (With - Without)", color="green")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title(f"Prediction Difference (With Futures - Without Futures)")
    plt.xlabel("Date")
    plt.ylabel("Price Difference (USDT)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs("analysis", exist_ok=True)
    plt.savefig(f"analysis/{symbol}_model_comparison.png")
    plt.close()
    
    print(f"Model comparison visualization saved to analysis/{symbol}_model_comparison.png")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    
    # Basic usage example
    example_basic_usage()
    
    # Feature exploration example
    example_feature_exploration()
    
    # Model comparison example
    example_model_comparison()
    
    # Advanced data fetching example (requires async)
    asyncio.run(example_advanced_data_fetching())