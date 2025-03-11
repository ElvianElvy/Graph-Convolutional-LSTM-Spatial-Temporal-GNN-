import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm
import networkx as nx
from collections import defaultdict

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.graph_conv_lstm import CryptoGraphConvLSTM
from utils.metrics import calculate_metrics


def train_model(symbol: str, reference_symbols: list = None, epochs: int = 100, batch_size: int = 32, 
               learning_rate: float = 0.001, hidden_size: int = 128, num_layers: int = 2, 
               sequence_length: int = 30, train_days: int = 365, validation_split: float = 0.2, 
               save_dir: str = "saved_models"):
    """
    Train the Graph Convolutional LSTM model on cryptocurrency data.
    
    Args:
        symbol: Main trading pair symbol to predict (e.g., "BTCUSDT")
        reference_symbols: List of related trading pairs to include in the graph (e.g., ["ETHUSDT", "BNBUSDT"])
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_size: Hidden size of the Graph LSTM
        num_layers: Number of Graph LSTM layers
        sequence_length: Number of timesteps in each input sequence
        train_days: Number of days of historical data to use for training
        validation_split: Fraction of data to use for validation
        save_dir: Directory to save the trained model
    
    Returns:
        Trained model and training history
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # If no reference symbols provided, use some defaults
    if reference_symbols is None:
        if symbol == "BTCUSDT":
            reference_symbols = ["ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
        elif symbol == "ETHUSDT":
            reference_symbols = ["BTCUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
        else:
            reference_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]
    
    # Ensure main symbol is first in the list
    all_symbols = [symbol] + [s for s in reference_symbols if s != symbol]
    num_cryptos = len(all_symbols)
    
    print(f"Training model for {symbol} with reference cryptocurrencies: {reference_symbols}")
    print(f"Total cryptocurrencies in the graph: {num_cryptos}")
    
    # Initialize components
    api = BinanceAPI()
    preprocessor = CryptoDataPreprocessor(
        sequence_length=sequence_length,
        num_cryptos=num_cryptos,
        prediction_length=182  # ~6 months of daily data
    )
    
    print(f"Fetching historical data for all cryptocurrencies...")
    data_dict = {}
    
    for sym in all_symbols:
        print(f"Fetching data for {sym}...")
        df = api.get_training_data(symbol=sym, days=train_days)
        data_dict[sym] = preprocessor.process_raw_data(df, sym)
    
    print(f"Aligning data across cryptocurrencies...")
    aligned_data = preprocessor.align_multi_crypto_data(data_dict)
    
    print(f"Creating adjacency matrix...")
    adjacency_matrix = preprocessor.create_adjacency_matrix(aligned_data, method="correlation")
    
    # Visualize the graph structure
    plt.figure(figsize=(10, 8))
    G = nx.from_numpy_array(adjacency_matrix)
    pos = nx.spring_layout(G)
    
    edge_weights = [adjacency_matrix[u][v] * 3 for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, 
            node_color='skyblue', 
            node_size=1500, 
            font_size=15,
            width=edge_weights,
            edge_color='gray',
            labels={i: symbol for i, symbol in enumerate(all_symbols)})
    
    plt.title("Cryptocurrency Relationship Graph")
    graph_path = os.path.join(save_dir, f"{symbol}_graph_structure.png")
    plt.savefig(graph_path)
    plt.close()
    
    print(f"Graph visualization saved to {graph_path}")
    
    print(f"Preparing graph data for training...")
    X, adj, y = preprocessor.prepare_graph_data(aligned_data, adjacency_matrix)
    
    print(f"X shape: {X.shape}, adj shape: {adj.shape}, y shape: {y.shape}")
    
    # Split into training and validation sets
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    adj_train, adj_val = adj[:split_idx], adj[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, adj_train, y_train)
    val_dataset = TensorDataset(X_val, adj_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = X.shape[3]  # Number of features
    num_nodes = X.shape[2]  # Number of cryptocurrencies
    model = CryptoGraphConvLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_nodes=num_nodes,
        num_layers=num_layers
    ).to(device)
    
    # Print model architecture
    print(f"Model architecture:\n{model}")
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    early_stop_patience = 20
    early_stop_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, adj_batch, y_batch in progress_bar:
            # Move tensors to device
            X_batch, adj_batch, y_batch = X_batch.to(device), adj_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch, adj_batch)
            loss = criterion(outputs, y_batch)
            
            # Add L2 regularization if implemented
            if hasattr(model, 'get_l2_regularization_loss'):
                reg_loss = model.get_l2_regularization_loss()
                loss = loss + reg_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, adj_batch, y_batch in val_loader:
                # Move tensors to device
                X_batch, adj_batch, y_batch = X_batch.to(device), adj_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch, adj_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Calculate validation metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Convert predictions back to original scale
        original_predictions = preprocessor.inverse_transform_predictions(all_predictions)
        original_targets = preprocessor.inverse_transform_predictions(all_targets)
        
        # Calculate metrics for the first 7 days (14 values for open and close)
        short_term_metrics = calculate_metrics(original_predictions[:, :14], original_targets[:, :14])
        
        # Calculate metrics for the entire prediction horizon
        long_term_metrics = calculate_metrics(original_predictions, original_targets)
        
        metrics = {
            'short_term': short_term_metrics,
            'long_term': long_term_metrics
        }
        
        history['val_metrics'].append(metrics)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Short-term (7 days) - MAE: {short_term_metrics['mae']:.2f}, RMSE: {short_term_metrics['rmse']:.2f}, MAPE: {short_term_metrics['mape']:.2f}%")
        print(f"Long-term (6 months) - MAE: {long_term_metrics['mae']:.2f}, RMSE: {long_term_metrics['rmse']:.2f}, MAPE: {long_term_metrics['mape']:.2f}%")
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # Save the best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f"{symbol}_graph_conv_lstm_{timestamp}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'sequence_length': sequence_length,
                    'input_size': input_size,
                    'num_nodes': num_nodes,
                    'symbols': all_symbols
                },
                'adjacency_matrix': adjacency_matrix
            }, model_path)
            print(f"Model saved to {model_path}")
            
            # Save preprocessor
            preprocessor_path = os.path.join(save_dir, f"{symbol}_preprocessor_{timestamp}.pkl")
            import pickle
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            print(f"Preprocessor saved to {preprocessor_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Plot training history
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot([m['short_term']['mae'] for m in history['val_metrics']], label='Short-term MAE (7 days)')
    plt.plot([m['long_term']['mae'] for m in history['val_metrics']], label='Long-term MAE (6 months)')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot([m['short_term']['rmse'] for m in history['val_metrics']], label='Short-term RMSE (7 days)')
    plt.plot([m['long_term']['rmse'] for m in history['val_metrics']], label='Long-term RMSE (6 months)')
    plt.title('Root Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot([m['short_term']['dir_acc'] for m in history['val_metrics']], label='Short-term Directional Accuracy (7 days)')
    plt.plot([m['long_term']['dir_acc'] for m in history['val_metrics']], label='Long-term Directional Accuracy (6 months)')
    plt.title('Directional Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f"{symbol}_training_history_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Training history plot saved to {plot_path}")
    
    # Create some example predictions on validation data
    with torch.no_grad():
        model.eval()
        X_example = X_val[:1].to(device)
        adj_example = adj_val[:1].to(device)
        y_example = y_val[:1].cpu().numpy()
        
        pred_example = model(X_example, adj_example).cpu().numpy()
        
        # Convert to original scale
        pred_example_orig = preprocessor.inverse_transform_predictions(pred_example)
        y_example_orig = preprocessor.inverse_transform_predictions(y_example)
        
        # Plot example prediction
        plt.figure(figsize=(15, 8))
        
        # Extract days, open and close prices
        days = np.arange(pred_example_orig.shape[1] // 2)
        pred_open = pred_example_orig[0, ::2]
        pred_close = pred_example_orig[0, 1::2]
        actual_open = y_example_orig[0, ::2]
        actual_close = y_example_orig[0, 1::2]
        
        # Plot open prices
        plt.subplot(2, 1, 1)
        plt.plot(days[:30], actual_open[:30], 'b-', label='Actual Open')
        plt.plot(days[:30], pred_open[:30], 'r--', label='Predicted Open')
        plt.title(f'{symbol} - Open Price Prediction (First 30 Days)')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot close prices
        plt.subplot(2, 1, 2)
        plt.plot(days[:30], actual_close[:30], 'g-', label='Actual Close')
        plt.plot(days[:30], pred_close[:30], 'm--', label='Predicted Close')
        plt.title(f'{symbol} - Close Price Prediction (First 30 Days)')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the example prediction plot
        example_path = os.path.join(save_dir, f"{symbol}_example_prediction_{timestamp}.png")
        plt.savefig(example_path)
        plt.close()
        
        print(f"Example prediction plot saved to {example_path}")
    
    return model, history, preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cryptocurrency price prediction model with Graph Conv LSTM')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Main trading pair symbol to predict')
    parser.add_argument('--reference_symbols', type=str, nargs='+', help='Reference trading pairs for the graph')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of Graph LSTM')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of Graph LSTM layers')
    parser.add_argument('--seq_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--train_days', type=int, default=365, help='Days of historical data')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    
    args = parser.parse_args()
    
    train_model(
        symbol=args.symbol,
        reference_symbols=args.reference_symbols,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sequence_length=args.seq_length,
        train_days=args.train_days,
        save_dir=args.save_dir
    )