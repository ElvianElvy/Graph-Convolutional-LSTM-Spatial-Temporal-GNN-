import os
import argparse
import json
import pickle
import time
import logging
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

import torch
import pandas as pd
import numpy as np

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.graph_conv_lstm import CryptoGraphConvLSTM
from train import train_model
from predict import predict_crypto_prices
from config import load_config, update_config, optimize_config, get_recommended_symbols

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')


def list_available_cryptos():
    """List available cryptocurrencies on Binance."""
    print("Fetching available cryptocurrencies from Binance...")
    api = BinanceAPI()
    symbols = api.get_available_symbols()
    
    # Filter for common quote assets
    quote_assets = ["USDT", "BUSD", "BTC", "ETH"]
    filtered_symbols = []
    
    for symbol in symbols:
        for quote in quote_assets:
            if symbol.endswith(quote) and not symbol.startswith("USDT"):
                filtered_symbols.append(symbol)
                break
    
    # Group by quote asset
    grouped = {}
    for symbol in filtered_symbols:
        for quote in quote_assets:
            if symbol.endswith(quote):
                if quote not in grouped:
                    grouped[quote] = []
                grouped[quote].append(symbol)
                break
    
    # Print in organized format
    print(f"\nAvailable cryptocurrencies on Binance ({len(filtered_symbols)} pairs):")
    for quote, symbols in grouped.items():
        print(f"\n{quote} pairs ({len(symbols)}):")
        # Print in multiple columns
        col_width = 12
        cols = 6
        symbols_sorted = sorted(symbols)
        for i in range(0, len(symbols_sorted), cols):
            row = symbols_sorted[i:i+cols]
            print("  ".join(symbol.ljust(col_width) for symbol in row))


def list_top_cryptocurrencies():
    """List top cryptocurrencies by trading volume on Binance."""
    print("Fetching top cryptocurrencies by volume...")
    api = BinanceAPI()
    
    # Get popular symbols for different quote assets
    usdt_symbols = api.get_popular_symbols(quote_asset="USDT", limit=20)
    btc_symbols = api.get_popular_symbols(quote_asset="BTC", limit=10)
    eth_symbols = api.get_popular_symbols(quote_asset="ETH", limit=10)
    
    print("\nTop USDT pairs by volume:")
    for i, symbol in enumerate(usdt_symbols, 1):
        print(f"{i:2d}. {symbol}")
    
    print("\nTop BTC pairs by volume:")
    for i, symbol in enumerate(btc_symbols, 1):
        print(f"{i:2d}. {symbol}")
    
    print("\nTop ETH pairs by volume:")
    for i, symbol in enumerate(eth_symbols, 1):
        print(f"{i:2d}. {symbol}")
    
    # Print recommended cryptocurrencies
    print("\nRecommended cryptocurrencies for prediction:")
    recommended = get_recommended_symbols()
    for item in recommended:
        print(f"- {item['symbol']} ({item['name']}): {item['description']}")


def visualize_crypto_correlations():
    """Visualize correlations between popular cryptocurrencies."""
    print("Analyzing correlations between cryptocurrencies...")
    api = BinanceAPI()
    
    # Get popular symbols
    symbols = api.get_popular_symbols(limit=10)
    
    # Calculate correlation matrix
    corr_matrix, data_dict = api.get_correlation_matrix(symbols, days=90)
    
    # Create a graph from the correlation matrix
    threshold = 0.5
    G = nx.Graph()
    
    # Add nodes
    for symbol in symbols:
        G.add_node(symbol)
    
    # Add edges for correlations above threshold
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i < j:  # Only process each pair once
                correlation = corr_matrix.loc[symbol1, symbol2]
                if abs(correlation) >= threshold:
                    G.add_edge(symbol1, symbol2, weight=correlation)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    
    # Draw edges with varying thickness based on correlation
    edge_widths = [abs(G[u][v]['weight']) * 5 for u, v in G.edges()]
    edge_colors = ['green' if G[u][v]['weight'] > 0 else 'red' for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    # Add edge labels (correlation values)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(f"Cryptocurrency Correlation Network (Threshold: {threshold})")
    plt.axis('off')
    
    # Save and show
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("analysis", exist_ok=True)
    plt.savefig(f"analysis/crypto_correlation_network_{timestamp}.png")
    
    print(f"Correlation network saved to analysis/crypto_correlation_network_{timestamp}.png")
    plt.show()


def find_latest_model(symbol, model_dir="saved_models"):
    """Find the latest trained model for a symbol."""
    if not os.path.exists(model_dir):
        return None, None
    
    # List all files in the directory
    files = os.listdir(model_dir)
    
    # Filter for model files matching the symbol
    model_files = [f for f in files if f.startswith(f"{symbol}_graph_conv_lstm_") and f.endswith(".pt")]
    preprocessor_files = [f for f in files if f.startswith(f"{symbol}_preprocessor_") and f.endswith(".pkl")]
    
    if not model_files or not preprocessor_files:
        return None, None
    
    # Sort by timestamp in filename
    model_files.sort(reverse=True)
    preprocessor_files.sort(reverse=True)
    
    # Return paths
    model_path = os.path.join(model_dir, model_files[0])
    preprocessor_path = os.path.join(model_dir, preprocessor_files[0])
    
    return model_path, preprocessor_path


def train_new_model(args, config):
    """Train a new model with the specified configuration."""
    print(f"\n=== Training new model for {args.symbol} ===\n")
    
    # Extract training parameters from config
    train_params = config["training"]
    
    # Train the model
    model, history, preprocessor = train_model(
        symbol=args.symbol,
        reference_symbols=train_params.get("reference_symbols"),
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        learning_rate=train_params["learning_rate"],
        hidden_size=train_params["hidden_size"],
        num_layers=train_params["num_layers"],
        sequence_length=train_params["sequence_length"],
        train_days=train_params["train_days"],
        validation_split=train_params["validation_split"],
        save_dir=args.model_dir
    )
    
    print("\nTraining completed.")
    
    # Find paths to the saved model and preprocessor
    model_path, preprocessor_path = find_latest_model(args.symbol, args.model_dir)
    
    if model_path and preprocessor_path:
        return model_path, preprocessor_path
    else:
        raise FileNotFoundError("Could not find the trained model files.")


def predict_prices(args, config, model_path=None, preprocessor_path=None):
    """Predict prices for the specified cryptocurrency."""
    print(f"\n=== Predicting prices for {args.symbol} ===\n")
    
    # If paths are not provided, find the latest model
    if not model_path or not preprocessor_path:
        model_path, preprocessor_path = find_latest_model(args.symbol, args.model_dir)
    
    # Check if model exists
    if not model_path or not preprocessor_path:
        print(f"No trained model found for {args.symbol}. Please train a model first.")
        return
    
    # Extract prediction parameters from config
    pred_params = config["prediction"]
    
    # Make predictions
    df_pred, summary = predict_crypto_prices(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        output_dir=args.output_dir,
        advanced_viz=pred_params["advanced_visualization"]
    )
    
    return df_pred, summary


def analyze_model(args, config):
    """Analyze a trained model and its predictions."""
    print(f"\n=== Analyzing model for {args.symbol} ===\n")
    
    # Find latest model
    model_path, preprocessor_path = find_latest_model(args.symbol, args.model_dir)
    
    # Check if model exists
    if not model_path or not preprocessor_path:
        print(f"No trained model found for {args.symbol}. Please train a model first.")
        return
    
    # Load model and preprocessor
    try:
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Extract model configuration and adjacency matrix
        config = checkpoint['config']
        adjacency_matrix = checkpoint.get('adjacency_matrix')
        symbols = config.get('symbols', [args.symbol])
        
        print(f"Model information:")
        print(f"  Model type: Graph Convolutional LSTM")
        print(f"  Hidden size: {config['hidden_size']}")
        print(f"  Number of layers: {config['num_layers']}")
        print(f"  Sequence length: {config['sequence_length']}")
        print(f"  Cryptocurrencies in graph: {', '.join(symbols)}")
        
        # Visualize the graph structure
        if adjacency_matrix is not None and len(symbols) > 0:
            plt.figure(figsize=(10, 8))
            G = nx.from_numpy_array(adjacency_matrix)
            
            # Set node labels to cryptocurrency symbols
            labels = {i: symbol for i, symbol in enumerate(symbols)}
            
            # Set node sizes proportional to their degree
            node_size = [3000 * (G.degree(i) / len(symbols)) for i in G.nodes()]
            
            # Set edge weights proportional to adjacency values
            edge_weights = [adjacency_matrix[u][v] * 3 for u, v in G.edges()]
            
            # Use a spring layout for visualization
            pos = nx.spring_layout(G, seed=42)
            
            # Draw the graph
            nx.draw(G, pos, with_labels=True, 
                    node_color='skyblue', 
                    node_size=node_size, 
                    font_size=15,
                    width=edge_weights,
                    edge_color='gray',
                    labels=labels)
            
            plt.title(f"Cryptocurrency Relationship Network for {args.symbol} Model")
            
            # Save the graph visualization
            os.makedirs("analysis", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_path = os.path.join("analysis", f"{args.symbol}_graph_structure_{timestamp}.png")
            plt.savefig(graph_path)
            
            print(f"Graph structure visualization saved to {graph_path}")
        
        # Make a prediction to analyze model performance
        df_pred, summary = predict_crypto_prices(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            output_dir=args.output_dir,
            advanced_viz=True
        )
        
        print("\nModel Summary:")
        print(f"  Overall trend prediction: {summary['overall']['trend']}")
        print(f"  Price change after 6 months: {summary['overall']['price_change_pct']:.2f}%")
        
        # Print milestone predictions
        print("\nPrice Milestones:")
        for period, info in summary['milestones'].items():
            if info['data']:
                print(f"  {period.replace('_', ' ').title()}: {info['data']['close']:.2f} ({'+' if info['data']['change_from_start'] >= 0 else ''}{info['data']['change_from_start']:.2f}%)")
        
    except Exception as e:
        print(f"Error analyzing model: {str(e)}")


def interactive_mode():
    """Run the application in interactive mode."""
    # Load configuration
    config = load_config()
    
    # Initialize Binance API
    api = BinanceAPI()
    
    print("=== Crypto Price Prediction with Graph Convolutional LSTM ===\n")
    
    while True:
        print("\nOptions:")
        print("1. List available cryptocurrencies")
        print("2. List top cryptocurrencies by volume")
        print("3. Visualize cryptocurrency correlations")
        print("4. Train a new model")
        print("5. Predict prices")
        print("6. Train and predict")
        print("7. Analyze model")
        print("8. Optimize configuration")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ")
        
        if choice == "1":
            list_available_cryptos()
        
        elif choice == "2":
            list_top_cryptocurrencies()
        
        elif choice == "3":
            visualize_crypto_correlations()
        
        elif choice in ["4", "5", "6", "7", "8"]:
            # Get symbol
            symbol = input("\nEnter cryptocurrency symbol (e.g., BTCUSDT): ").upper()
            
            # Create parser for argument handling
            parser = argparse.ArgumentParser()
            parser.add_argument("--symbol", type=str, default=symbol)
            parser.add_argument("--model_dir", type=str, default=config["paths"]["model_dir"])
            parser.add_argument("--output_dir", type=str, default=config["paths"]["output_dir"])
            args = parser.parse_args([])
            
            if choice == "4":
                # Train a new model
                try:
                    model_path, preprocessor_path = train_new_model(args, config)
                    print(f"\nModel trained and saved to {model_path}")
                except Exception as e:
                    print(f"Error training model: {str(e)}")
            
            elif choice == "5":
                # Predict prices
                try:
                    df_pred, summary = predict_prices(args, config)
                except Exception as e:
                    print(f"Error predicting prices: {str(e)}")
            
            elif choice == "6":
                # Train and predict
                try:
                    model_path, preprocessor_path = train_new_model(args, config)
                    df_pred, summary = predict_prices(args, config, model_path, preprocessor_path)
                except Exception as e:
                    print(f"Error: {str(e)}")
            
            elif choice == "7":
                # Analyze model
                try:
                    analyze_model(args, config)
                except Exception as e:
                    print(f"Error analyzing model: {str(e)}")
            
            elif choice == "8":
                # Optimize configuration
                print("\nOptimization targets:")
                print("1. Accuracy (slower training, potentially better results)")
                print("2. Speed (faster training, potentially less accurate)")
                print("3. Balanced (compromise between speed and accuracy)")
                
                target_choice = input("\nSelect optimization target (1-3): ")
                
                if target_choice == "1":
                    target = "accuracy"
                elif target_choice == "2":
                    target = "speed"
                else:
                    target = "balanced"
                
                try:
                    optimized_config = optimize_config(symbol, target)
                    update_config(optimized_config)
                    print(f"\nConfiguration optimized for {symbol} (target: {target}).")
                    print("The new configuration will be used for future training sessions.")
                except Exception as e:
                    print(f"Error optimizing configuration: {str(e)}")
        
        elif choice == "9":
            print("\nExiting application. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number from 1 to 9.")


def main():
    """Main entry point for the application."""
    # Load configuration
    config = load_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction with Graph Convolutional LSTM")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list", action="store_true", help="List available cryptocurrencies")
    parser.add_argument("--top", action="store_true", help="List top cryptocurrencies by volume")
    parser.add_argument("--correlations", action="store_true", help="Visualize cryptocurrency correlations")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--predict", action="store_true", help="Predict prices")
    parser.add_argument("--analyze", action="store_true", help="Analyze trained model")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Cryptocurrency symbol")
    parser.add_argument("--model_dir", type=str, default=config["paths"]["model_dir"], help="Directory for saved models")
    parser.add_argument("--output_dir", type=str, default=config["paths"]["output_dir"], help="Directory for prediction outputs")
    parser.add_argument("--optimize", type=str, choices=["accuracy", "speed", "balanced"], help="Optimize configuration")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run in interactive mode if requested
    if args.interactive:
        interactive_mode()
        return
    
    # List available cryptocurrencies if requested
    if args.list:
        list_available_cryptos()
        return
    
    # List top cryptocurrencies if requested
    if args.top:
        list_top_cryptocurrencies()
        return
    
    # Visualize cryptocurrency correlations if requested
    if args.correlations:
        visualize_crypto_correlations()
        return
    
    # Optimize configuration if requested
    if args.optimize:
        optimized_config = optimize_config(args.symbol, args.optimize)
        update_config(optimized_config)
        print(f"Configuration optimized for {args.symbol} (target: {args.optimize}).")
        return
    
    # Train a new model if requested
    if args.train:
        try:
            model_path, preprocessor_path = train_new_model(args, config)
            print(f"\nModel trained and saved to {model_path}")
            
            # If predict is also requested, use the newly trained model
            if args.predict:
                df_pred, summary = predict_prices(args, config, model_path, preprocessor_path)
        except Exception as e:
            print(f"Error training model: {str(e)}")
    
    # Predict prices if requested (and not already done after training)
    elif args.predict:
        try:
            df_pred, summary = predict_prices(args, config)
        except Exception as e:
            print(f"Error predicting prices: {str(e)}")
    
    # Analyze model if requested
    if args.analyze:
        try:
            analyze_model(args, config)
        except Exception as e:
            print(f"Error analyzing model: {str(e)}")


if __name__ == "__main__":
    main()