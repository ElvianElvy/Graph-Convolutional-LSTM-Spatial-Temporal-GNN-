import os
import torch
import numpy as np
import pandas as pd
import argparse
import pickle
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import networkx as nx
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.graph_conv_lstm import CryptoGraphConvLSTM


class CryptoPredictor:
    """
    Class for making cryptocurrency price predictions using trained Graph Convolutional LSTM models.
    """
    
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the predictor with a trained model and preprocessor.
        
        Args:
            model_path: Path to the saved model checkpoint
            preprocessor_path: Path to the saved preprocessor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the preprocessor
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        self.config = checkpoint['config']
        self.adjacency_matrix = checkpoint.get('adjacency_matrix')
        self.symbols = self.config.get('symbols', [])
        
        # Initialize model
        self.model = CryptoGraphConvLSTM(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_nodes=self.config['num_nodes'],
            num_layers=self.config['num_layers']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize Binance API
        self.api = BinanceAPI()
    
    def predict(self, days=182):
        """
        Make predictions for the cryptocurrencies.
        
        Args:
            days: Number of days to predict (default: 182 for 6 months)
        
        Returns:
            DataFrame with predictions
        """
        # Get main symbol (first in the list)
        main_symbol = self.symbols[0]
        
        # Fetch recent data for all symbols
        data_dict = {}
        
        for symbol in self.symbols:
            # Fetch more data than needed to ensure we have enough after processing
            df = self.api.get_historical_klines(
                symbol=symbol,
                interval="1d",
                limit=self.preprocessor.sequence_length + 30
            )
            
            # Process the data
            data_dict[symbol] = self.preprocessor.process_raw_data(df, symbol)
        
        # Align data across cryptocurrencies
        aligned_data = self.preprocessor.align_multi_crypto_data(data_dict)
        
        # Ensure we have enough data after processing
        for symbol, df in aligned_data.items():
            if len(df) < self.preprocessor.sequence_length:
                raise ValueError(f"Not enough data for {symbol} after preprocessing. Got {len(df)} valid data points, need {self.preprocessor.sequence_length}.")
        
        # Prepare data for prediction
        X, adj = self.preprocessor.prepare_single_prediction(aligned_data)
        
        # If adjacency_matrix was saved with the model, use it
        if self.adjacency_matrix is not None:
            adj_tensor = torch.tensor(self.adjacency_matrix, dtype=torch.float32).unsqueeze(0)
            # Make sure it has the right shape
            if adj_tensor.shape == adj.shape:
                adj = adj_tensor
        
        # Verify tensor shapes
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Invalid tensor shape: X shape is {X.shape}. Check data preprocessing.")
        
        # Move tensors to device
        X, adj = X.to(self.device), adj.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            predictions = self.model(X, adj)
        
        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        
        # Scale back to original values
        original_predictions = self.preprocessor.inverse_transform_predictions(predictions_np)
        
        # Reshape to [open, close] format for 6 months (182 days)
        original_predictions = original_predictions.reshape(-1, days, 2)
        
        # Create a DataFrame with predictions
        last_date = aligned_data[main_symbol]["Open time"].iloc[-1]
        dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        result = []
        for i, date in enumerate(dates):
            result.append({
                "Date": date,
                "Predicted Open": original_predictions[0, i, 0],
                "Predicted Close": original_predictions[0, i, 1]
            })
        
        return pd.DataFrame(result)
    
    def create_visualization(self, df_pred, output_dir="predictions", advanced=True):
        """
        Create enhanced visualization of the predictions.
        
        Args:
            df_pred: DataFrame with predictions
            output_dir: Directory to save the visualization
            advanced: Whether to use advanced visualization techniques
            
        Returns:
            Path to the saved visualization
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get main symbol (first in the list)
        main_symbol = self.symbols[0]
        
        # Fetch recent historical data for context (last 90 days)
        df_hist = self.api.get_historical_klines(
            symbol=main_symbol,
            interval="1d",
            limit=90
        )
        
        # Select relevant columns and convert to DataFrame format
        df_hist = df_hist[["Open time", "Open", "High", "Low", "Close", "Volume"]].copy()
        df_hist.loc[:, "Open"] = pd.to_numeric(df_hist["Open"])
        df_hist.loc[:, "High"] = pd.to_numeric(df_hist["High"])
        df_hist.loc[:, "Low"] = pd.to_numeric(df_hist["Low"])
        df_hist.loc[:, "Close"] = pd.to_numeric(df_hist["Close"])
        df_hist.loc[:, "Volume"] = pd.to_numeric(df_hist["Volume"])
        df_hist.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        
        # Calculate additional metrics for historical data
        df_hist['MA7'] = df_hist['Close'].rolling(window=7).mean()
        df_hist['MA30'] = df_hist['Close'].rolling(window=30).mean()
        df_hist['Daily_Change'] = df_hist['Close'].pct_change() * 100
        df_hist['Volatility'] = df_hist['Daily_Change'].rolling(window=7).std()
        
        # Calculate additional metrics for prediction data
        df_pred['Daily_Change'] = 0.0
        df_pred['MA7'] = 0.0
        df_pred['MA30'] = 0.0
        df_pred['Volatility'] = 0.0
        
        # Calculate percentage change from last close to predicted values
        last_close = df_hist["Close"].iloc[-1]
        df_pred.loc[:, "Change_From_Last"] = ((df_pred["Predicted Close"] - last_close) / last_close * 100)
        
        # Calculate daily changes for predictions
        for i in range(1, len(df_pred)):
            df_pred.loc[df_pred.index[i], 'Daily_Change'] = (
                (df_pred['Predicted Close'].iloc[i] - df_pred['Predicted Close'].iloc[i-1]) / 
                df_pred['Predicted Close'].iloc[i-1] * 100
            )
        
        # Set first day's daily change
        df_pred.loc[df_pred.index[0], 'Daily_Change'] = (
            (df_pred['Predicted Close'].iloc[0] - last_close) / last_close * 100
        )
        
        # Calculate moving averages for predictions
        for i in range(len(df_pred)):
            # 7-day MA
            if i < 7:
                lookback = min(i+1, 7)
                ma7_values = list(df_pred['Predicted Close'].iloc[:i+1])
                if lookback < 7:
                    # Include some historical data
                    ma7_values = list(df_hist['Close'].iloc[-(7-lookback):]) + ma7_values
                df_pred.loc[df_pred.index[i], 'MA7'] = sum(ma7_values) / 7
            else:
                df_pred.loc[df_pred.index[i], 'MA7'] = df_pred['Predicted Close'].iloc[i-7:i+1].mean()
            
            # 30-day MA
            if i < 30:
                lookback = min(i+1, 30)
                ma30_values = list(df_pred['Predicted Close'].iloc[:i+1])
                if lookback < 30:
                    # Include some historical data
                    ma30_values = list(df_hist['Close'].iloc[-(30-lookback):]) + ma30_values
                df_pred.loc[df_pred.index[i], 'MA30'] = sum(ma30_values) / 30
            else:
                df_pred.loc[df_pred.index[i], 'MA30'] = df_pred['Predicted Close'].iloc[i-30:i+1].mean()
        
        # Calculate volatility (7-day rolling std of daily changes)
        for i in range(len(df_pred)):
            if i < 7:
                lookback = min(i+1, 7)
                vol_values = list(df_pred['Daily_Change'].iloc[:i+1])
                if lookback < 7:
                    # Include some historical data
                    vol_values = list(df_hist['Daily_Change'].iloc[-(7-lookback):]) + vol_values
                df_pred.loc[df_pred.index[i], 'Volatility'] = np.std(vol_values)
            else:
                df_pred.loc[df_pred.index[i], 'Volatility'] = df_pred['Daily_Change'].iloc[i-7:i+1].std()
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if advanced:
            # Create interactive Plotly visualization
            return self._create_advanced_visualization(df_hist, df_pred, main_symbol, output_dir, timestamp)
        else:
            # Create standard Matplotlib visualization
            return self._create_standard_visualization(df_hist, df_pred, main_symbol, output_dir, timestamp)
    
    def _create_standard_visualization(self, df_hist, df_pred, symbol, output_dir, timestamp):
        """Create standard visualization with Matplotlib"""
        # Set seaborn style
        sns.set(style="whitegrid")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Main price chart
        ax1 = fig.add_subplot(gs[0])
        
        # Plot historical prices as candlesticks
        for i in range(len(df_hist)):
            # Determine if it's a bullish or bearish candle
            if df_hist['Close'].iloc[i] >= df_hist['Open'].iloc[i]:
                color = 'green'
                body_bottom = df_hist['Open'].iloc[i]
                body_top = df_hist['Close'].iloc[i]
            else:
                color = 'red'
                body_bottom = df_hist['Close'].iloc[i]
                body_top = df_hist['Open'].iloc[i]
            
            # Plot the candle body
            date = i
            rect = plt.Rectangle((date-0.4, body_bottom), 0.8, body_top-body_bottom, 
                                 color=color, alpha=0.7, zorder=2)
            ax1.add_patch(rect)
            
            # Plot the high/low wicks
            ax1.plot([date, date], [df_hist['Low'].iloc[i], df_hist['High'].iloc[i]], 
                     color='black', linewidth=1, zorder=1)
        
        # Plot moving averages for historical data
        hist_x = range(len(df_hist))
        ax1.plot(hist_x, df_hist['MA7'].values, color='blue', linewidth=1.5, label='7-day MA')
        ax1.plot(hist_x, df_hist['MA30'].values, color='orange', linewidth=1.5, label='30-day MA')
        
        # Add prediction zone shading
        hist_len = len(df_hist)
        ax1.axvspan(hist_len-1, hist_len+len(df_pred)-1, color='lightgray', alpha=0.3, label='Prediction Zone')
        
        # Draw vertical line to separate historical data and predictions
        ax1.axvline(x=hist_len-1, color="black", linestyle="--", linewidth=1.5)
        
        # Plot predicted open/close as candlesticks
        for i in range(len(df_pred)):
            idx = hist_len + i
            # Determine if it's a bullish or bearish candle
            if df_pred['Predicted Close'].iloc[i] >= df_pred['Predicted Open'].iloc[i]:
                color = 'green'
                body_bottom = df_pred['Predicted Open'].iloc[i]
                body_top = df_pred['Predicted Close'].iloc[i]
            else:
                color = 'red'
                body_bottom = df_pred['Predicted Close'].iloc[i]
                body_top = df_pred['Predicted Open'].iloc[i]
            
            # Plot the candle body with hatch pattern for predictions
            rect = plt.Rectangle((idx-0.4, body_bottom), 0.8, body_top-body_bottom, 
                                color=color, alpha=0.5, hatch='///', zorder=2)
            ax1.add_patch(rect)
            
            # Connect predicted candles with lines
            if i > 0:
                prev_idx = hist_len + i - 1
                ax1.plot([prev_idx, idx], 
                         [df_pred['Predicted Close'].iloc[i-1], df_pred['Predicted Open'].iloc[i]],
                         color='blue', linestyle=':', linewidth=1)
        
        # Plot moving averages for predictions
        pred_x = range(hist_len, hist_len + len(df_pred))
        ax1.plot(pred_x, df_pred['MA7'].values, color='blue', linewidth=1.5, linestyle='--')
        ax1.plot(pred_x, df_pred['MA30'].values, color='orange', linewidth=1.5, linestyle='--')
        
        # Set y-axis limits with padding
        all_prices = [
            df_hist['Low'].min(), df_hist['High'].max(),
            df_pred['Predicted Open'].min(), df_pred['Predicted Open'].max(),
            df_pred['Predicted Close'].min(), df_pred['Predicted Close'].max()
        ]
        y_min, y_max = min(all_prices), max(all_prices)
        y_range = y_max - y_min
        ax1.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
        
        # Add price labels on the right for predicted values
        label_interval = max(1, len(df_pred) // 20)  # Show at most 20 labels
        for i in range(0, len(df_pred), label_interval):
            idx = hist_len + i
            price = df_pred['Predicted Close'].iloc[i]
            color = 'green' if df_pred['Predicted Close'].iloc[i] >= df_pred['Predicted Open'].iloc[i] else 'red'
            change = df_pred['Change_From_Last'].iloc[i]
            change_sign = '+' if change >= 0 else ''
            ax1.annotate(f"{price:.2f} ({change_sign}{change:.2f}%)", 
                        xy=(idx, price),
                        xytext=(idx + 0.1, price),
                        fontsize=9,
                        color=color)
        
        # Set x-axis limits
        ax1.set_xlim(-0.5, hist_len + len(df_pred) - 0.5)
        
        # Format x-axis with dates
        all_dates = list(df_hist['Date']) + list(df_pred['Date'])
        x_ticks = range(len(all_dates))
        
        # Only show a selection of dates to avoid crowding
        date_step = max(1, len(all_dates) // 10)  # Show at most 10 dates
        x_tick_positions = list(range(0, len(all_dates), date_step))
        x_tick_labels = [all_dates[i].strftime('%Y-%m-%d') for i in x_tick_positions]
        
        # Always include the last historical date and some prediction dates
        if hist_len - 1 not in x_tick_positions:
            x_tick_positions.append(hist_len - 1)
            x_tick_labels.append(all_dates[hist_len - 1].strftime('%Y-%m-%d'))
            
        # Add markers for 1 month, 3 months, and 6 months into the future
        future_markers = [30, 90, 180]
        for days in future_markers:
            if hist_len + days - 1 < len(all_dates):
                if hist_len + days - 1 not in x_tick_positions:
                    x_tick_positions.append(hist_len + days - 1)
                    x_tick_labels.append(all_dates[hist_len + days - 1].strftime('%Y-%m-%d'))
        
        # Sort positions and labels
        x_tick_positions, x_tick_labels = zip(*sorted(zip(x_tick_positions, x_tick_labels)))
        
        ax1.set_xticks(x_tick_positions)
        ax1.set_xticklabels(x_tick_labels, rotation=45, ha='right')
        
        # Volume subplot
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        # Plot historical volume
        for i in range(len(df_hist)):
            color = 'green' if df_hist['Close'].iloc[i] >= df_hist['Open'].iloc[i] else 'red'
            ax2.bar(i, df_hist['Volume'].iloc[i], color=color, alpha=0.5, width=0.8)
        
        # Add volume title
        ax2.set_ylabel('Volume')
        ax2.tick_params(axis='x', labelbottom=False)
        
        # Daily change subplot
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        # Plot historical daily changes
        for i in range(len(df_hist)):
            change = df_hist['Daily_Change'].iloc[i]
            color = 'green' if change >= 0 else 'red'
            ax3.bar(i, change, color=color, alpha=0.7, width=0.8)
        
        # Plot predicted daily changes
        for i in range(len(df_pred)):
            idx = hist_len + i
            change = df_pred['Daily_Change'].iloc[i]
            color = 'green' if change >= 0 else 'red'
            ax3.bar(idx, change, color=color, alpha=0.5, width=0.8, hatch='///')
            
            # Add percentage labels at regular intervals
            if i % label_interval == 0:
                ax3.annotate(f"{change:.2f}%", 
                            xy=(idx, change),
                            xytext=(idx, change + (0.5 if change >= 0 else -0.5)),
                            fontsize=8,
                            ha='center',
                            va='bottom' if change >= 0 else 'top',
                            color=color)
        
        # Volatility subplot
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        
        # Plot historical volatility
        ax4.plot(range(len(df_hist)), df_hist['Volatility'], color='purple', alpha=0.7, linewidth=2)
        
        # Plot predicted volatility
        ax4.plot(range(hist_len, hist_len + len(df_pred)), df_pred['Volatility'], 
                color='purple', alpha=0.7, linewidth=2, linestyle='--')
        
        # Set y-axis limits
        max_vol = max(df_hist['Volatility'].max(), df_pred['Volatility'].max())
        ax4.set_ylim(0, max_vol * 1.2)
        
        # Add volatility title
        ax4.set_ylabel('Volatility (7d)')
        ax4.set_xlabel('Date')
        
        # Add labels and title
        ax1.set_ylabel('Price')
        ax3.set_ylabel('Daily Change %')
        ax1.set_title(f"{symbol} Price Prediction for Next 6 Months", fontsize=16)
        ax1.legend()
        
        # Add annotations
        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        month1_change = df_pred.iloc[29]['Change_From_Last'] if len(df_pred) > 29 else 0
        month3_change = df_pred.iloc[89]['Change_From_Last'] if len(df_pred) > 89 else 0
        month6_change = df_pred.iloc[-1]['Change_From_Last'] if len(df_pred) > 0 else 0
        
        change_text = (
            f"Predicted changes from current price:\n"
            f"1 Month: {'+' if month1_change >= 0 else ''}{month1_change:.2f}%\n"
            f"3 Months: {'+' if month3_change >= 0 else ''}{month3_change:.2f}%\n"
            f"6 Months: {'+' if month6_change >= 0 else ''}{month6_change:.2f}%"
        )
        
        fig.text(0.02, 0.02, f"Prediction made on: {prediction_date}\n{change_text}", 
                fontsize=10, color="gray")
        
        # Add prediction summary
        avg_pred = df_pred["Predicted Close"].mean()
        min_pred = df_pred["Predicted Close"].min()
        max_pred = df_pred["Predicted Close"].max()
        trend = "Bullish" if month6_change > 0 else "Bearish"
        
        summary_text = (
            f"Prediction Summary:\n"
            f"Current price: {df_hist['Close'].iloc[-1]:.2f}\n"
            f"Average predicted: {avg_pred:.2f}\n"
            f"Range: {min_pred:.2f} - {max_pred:.2f}\n"
            f"6-month outlook: {trend}"
        )
        
        # Add text box for summary
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Add markers for 1 month, 3 months, and 6 months in the future
        for days, label in zip(future_markers, ['1 Month', '3 Months', '6 Months']):
            if hist_len + days - 1 < len(all_dates):
                marker_x = hist_len + days - 1
                marker_y = df_pred.iloc[days-1]['Predicted Close'] if days - 1 < len(df_pred) else df_pred.iloc[-1]['Predicted Close']
                ax1.axvline(x=marker_x, color="blue", linestyle="-.", linewidth=1.0, alpha=0.6)
                ax1.text(marker_x, y_min, label, rotation=90, verticalalignment='bottom', alpha=0.8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f"{symbol}_prediction_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Standard visualization saved to {plot_path}")
        
        return plot_path
    
    def _create_advanced_visualization(self, df_hist, df_pred, symbol, output_dir, timestamp):
        """Create advanced interactive visualization with Plotly"""
        # Create a comprehensive Plotly figure
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{symbol} Price Prediction for Next 6 Months",
                "Trading Volume",
                "Daily Price Change (%)",
                "Volatility (7-day)",
                "Prediction Confidence"
            ),
            row_heights=[0.5, 0.1, 0.15, 0.1, 0.15]
        )
        
        # Prepare data
        hist_dates = df_hist['Date']
        pred_dates = df_pred['Date']
        all_dates = list(hist_dates) + list(pred_dates)
        
        # Calculate price range for confidence intervals
        # Simulate confidence intervals based on volatility
        df_pred['Volatility_Factor'] = df_pred['Volatility'] / df_pred['Volatility'].iloc[0]
        base_uncertainty = 0.05  # 5% base uncertainty
        
        df_pred['Upper_95'] = df_pred['Predicted Close'] * (1 + base_uncertainty * df_pred['Volatility_Factor'] * 1.96)
        df_pred['Lower_95'] = df_pred['Predicted Close'] * (1 - base_uncertainty * df_pred['Volatility_Factor'] * 1.96)
        df_pred['Upper_68'] = df_pred['Predicted Close'] * (1 + base_uncertainty * df_pred['Volatility_Factor'])
        df_pred['Lower_68'] = df_pred['Predicted Close'] * (1 - base_uncertainty * df_pred['Volatility_Factor'])
        
        # Add confidence intervals to 1st subplot
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=df_pred['Upper_95'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=df_pred['Lower_95'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(231, 234, 241, 0.3)',
                name='95% Confidence'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=df_pred['Upper_68'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=df_pred['Lower_68'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(231, 234, 241, 0.5)',
                name='68% Confidence'
            ),
            row=1, col=1
        )
        
        # Add historical candlestick to 1st subplot
        fig.add_trace(
            go.Candlestick(
                x=hist_dates,
                open=df_hist['Open'],
                high=df_hist['High'],
                low=df_hist['Low'],
                close=df_hist['Close'],
                name='Historical Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add predicted candlestick to 1st subplot
        fig.add_trace(
            go.Candlestick(
                x=pred_dates,
                open=df_pred['Predicted Open'],
                high=df_pred['Predicted Open'].combine(df_pred['Predicted Close'], max),  # Use max of open/close for high
                low=df_pred['Predicted Open'].combine(df_pred['Predicted Close'], min),   # Use min of open/close for low
                close=df_pred['Predicted Close'],
                name='Predicted Price',
                increasing_line_color='green',
                decreasing_line_color='red',
                increasing_line_width=1,
                decreasing_line_width=1,
                increasing_fillcolor='rgba(0, 255, 0, 0.3)',
                decreasing_fillcolor='rgba(255, 0, 0, 0.3)'
            ),
            row=1, col=1
        )
        
        # Add moving averages to 1st subplot
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=df_hist['MA7'],
                mode='lines',
                line=dict(color='blue', width=2),
                name='7-day MA (Historical)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=df_hist['MA30'],
                mode='lines',
                line=dict(color='orange', width=2),
                name='30-day MA (Historical)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=df_pred['MA7'],
                mode='lines',
                line=dict(color='blue', width=2, dash='dash'),
                name='7-day MA (Predicted)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=df_pred['MA30'],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name='30-day MA (Predicted)'
            ),
            row=1, col=1
        )
        
        # Add volume to 2nd subplot
        colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in df_hist.iterrows()]
        fig.add_trace(
            go.Bar(
                x=hist_dates,
                y=df_hist['Volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Add daily change to 3rd subplot
        colors_hist = ['green' if x >= 0 else 'red' for x in df_hist['Daily_Change']]
        colors_pred = ['rgba(0, 128, 0, 0.5)' if x >= 0 else 'rgba(255, 0, 0, 0.5)' for x in df_pred['Daily_Change']]
        
        fig.add_trace(
            go.Bar(
                x=hist_dates,
                y=df_hist['Daily_Change'],
                marker_color=colors_hist,
                name='Historical Daily Change'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=pred_dates,
                y=df_pred['Daily_Change'],
                marker_color=colors_pred,
                name='Predicted Daily Change'
            ),
            row=3, col=1
        )
        
        # Add zero line for daily change
        fig.add_shape(
            type="line",
            x0=all_dates[0],
            y0=0,
            x1=all_dates[-1],
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=3, col=1
        )
        
        # Add volatility to 4th subplot
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=df_hist['Volatility'],
                mode='lines',
                line=dict(color='purple', width=2),
                name='Historical Volatility'
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=df_pred['Volatility'],
                mode='lines',
                line=dict(color='purple', width=2, dash='dash'),
                name='Predicted Volatility'
            ),
            row=4, col=1
        )
        
        # Add prediction confidence to 5th subplot
        # Create a declining confidence metric based on forecast horizon
        days = np.arange(len(df_pred))
        confidence = 100 * np.exp(-0.01 * days)  # Exponential decay
        confidence_adjusted = confidence * (1 - 0.3 * (df_pred['Volatility'] / df_pred['Volatility'].max()))
        
        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=confidence_adjusted,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Prediction Confidence'
            ),
            row=5, col=1
        )
        
        # Add confidence levels
        fig.add_shape(
            type="line",
            x0=pred_dates[0],
            y0=90,
            x1=pred_dates[-1],
            y1=90,
            line=dict(color="green", width=1, dash="dot"),
            row=5, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=pred_dates[0],
            y0=70,
            x1=pred_dates[-1],
            y1=70,
            line=dict(color="orange", width=1, dash="dot"),
            row=5, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=pred_dates[0],
            y0=50,
            x1=pred_dates[-1],
            y1=50,
            line=dict(color="red", width=1, dash="dot"),
            row=5, col=1
        )
        
        # Add markers for 1, 3, and 6 months
        future_markers = [30, 90, 180]
        marker_labels = ['1 Month', '3 Months', '6 Months']
        
        for days, label in zip(future_markers, marker_labels):
            if days < len(df_pred):
                # Add vertical lines
                fig.add_vline(
                    x=pred_dates[days-1],
                    line_width=1,
                    line_dash="dashdot",
                    line_color="blue",
                    annotation_text=label,
                    annotation_position="bottom"
                )
                
                # Add price annotations
                month_price = df_pred['Predicted Close'].iloc[days-1]
                month_change = df_pred['Change_From_Last'].iloc[days-1]
                sign = '+' if month_change >= 0 else ''
                
                fig.add_annotation(
                    x=pred_dates[days-1],
                    y=month_price,
                    text=f"{month_price:.2f} ({sign}{month_change:.2f}%)",
                    showarrow=True,
                    arrowhead=1,
                    row=1, col=1
                )
        
        # Add separator between historical and predicted data
        last_hist_date = hist_dates.iloc[-1]
        fig.add_vline(
            x=last_hist_date,
            line_width=2,
            line_dash="dash",
            line_color="black",
            annotation_text="Prediction Start",
            annotation_position="top"
        )
        
        # Update layout
        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        month1_change = df_pred.iloc[29]['Change_From_Last'] if len(df_pred) > 29 else 0
        month3_change = df_pred.iloc[89]['Change_From_Last'] if len(df_pred) > 89 else 0
        month6_change = df_pred.iloc[-1]['Change_From_Last'] if len(df_pred) > 0 else 0
        
        fig.update_layout(
            title=f"{symbol} Price Prediction - Generated on {prediction_date}",
            xaxis_title="Date",
            yaxis_title="Price",
            height=1200,
            width=1200,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=[
                dict(
                    x=0.01,
                    y=0.01,
                    xref="paper",
                    yref="paper",
                    text=(
                        f"Predicted changes from current price:<br>"
                        f"1 Month: {'+' if month1_change >= 0 else ''}{month1_change:.2f}%<br>"
                        f"3 Months: {'+' if month3_change >= 0 else ''}{month3_change:.2f}%<br>"
                        f"6 Months: {'+' if month6_change >= 0 else ''}{month6_change:.2f}%"
                    ),
                    showarrow=False,
                    bgcolor="white",
                    opacity=0.8,
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )
        
        # Set y-axis range for confidence subplot
        fig.update_yaxes(range=[0, 100], row=5, col=1)
        
        # Save the plot as HTML and PNG
        html_path = os.path.join(output_dir, f"{symbol}_prediction_interactive_{timestamp}.html")
        png_path = os.path.join(output_dir, f"{symbol}_prediction_{timestamp}.png")
        
        fig.write_html(html_path)
        fig.write_image(png_path, scale=2)
        
        print(f"Interactive visualization saved to {html_path}")
        print(f"Advanced visualization image saved to {png_path}")
        
        # Also save graph visualization
        if self.adjacency_matrix is not None and self.symbols:
            # Create a network graph visualization
            plt.figure(figsize=(10, 8))
            G = nx.from_numpy_array(self.adjacency_matrix)
            
            # Set node labels to cryptocurrency symbols
            labels = {i: symbol for i, symbol in enumerate(self.symbols)}
            
            # Set node sizes proportional to their degree
            node_size = [3000 * (G.degree(i) / len(self.symbols)) for i in G.nodes()]
            
            # Set edge weights proportional to adjacency values
            edge_weights = [self.adjacency_matrix[u][v] * 3 for u, v in G.edges()]
            
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
            
            plt.title(f"Cryptocurrency Relationship Network for {symbol} Prediction")
            
            # Save the graph visualization
            graph_path = os.path.join(output_dir, f"{symbol}_graph_structure_{timestamp}.png")
            plt.savefig(graph_path)
            plt.close()
            
            print(f"Graph structure visualization saved to {graph_path}")
        
        return png_path
    
    def get_prediction_summary(self, df_pred):
        """
        Generate a summary of the predictions.
        
        Args:
            df_pred: DataFrame with predictions
            
        Returns:
            Dictionary with prediction summary
        """
        # Get main symbol (first in the list)
        main_symbol = self.symbols[0]
        
        # Calculate price changes
        first_open = df_pred["Predicted Open"].iloc[0]
        last_close = df_pred["Predicted Close"].iloc[-1]
        price_change = last_close - first_open
        price_change_pct = (price_change / first_open) * 100
        
        # Calculate key milestone predictions
        milestones = {
            "1_week": {"idx": 6, "data": {}},
            "1_month": {"idx": 29, "data": {}},
            "3_months": {"idx": 89, "data": {}},
            "6_months": {"idx": -1, "data": {}}  # Last day
        }
        
        for period, info in milestones.items():
            idx = info["idx"]
            if idx < len(df_pred) or (idx == -1 and len(df_pred) > 0):
                idx_to_use = idx if idx >= 0 else len(df_pred) - 1
                milestones[period]["data"] = {
                    "date": df_pred["Date"].iloc[idx_to_use].strftime("%Y-%m-%d"),
                    "open": float(df_pred["Predicted Open"].iloc[idx_to_use]),
                    "close": float(df_pred["Predicted Close"].iloc[idx_to_use]),
                    "change_from_start": float(df_pred["Change_From_Last"].iloc[idx_to_use])
                }
        
        # Calculate daily changes
        daily_predictions = []
        for i in range(len(df_pred)):
            daily_predictions.append({
                "date": df_pred["Date"].iloc[i].strftime("%Y-%m-%d"),
                "day_of_week": df_pred["Date"].iloc[i].strftime("%A"),
                "open": float(df_pred["Predicted Open"].iloc[i]),
                "close": float(df_pred["Predicted Close"].iloc[i]),
                "change_pct": float(df_pred["Daily_Change"].iloc[i]),
                "change_from_start": float(df_pred["Change_From_Last"].iloc[i])
            })
        
        # Check if trend is bullish, bearish, or sideways
        if price_change_pct > 10:
            trend = "Strongly Bullish"
        elif price_change_pct > 3:
            trend = "Bullish"
        elif price_change_pct < -10:
            trend = "Strongly Bearish"
        elif price_change_pct < -3:
            trend = "Bearish"
        else:
            trend = "Sideways"
        
        # Calculate volatility stats
        avg_volatility = df_pred["Volatility"].mean()
        max_volatility = df_pred["Volatility"].max()
        min_volatility = df_pred["Volatility"].min()
        
        # Create summary
        summary = {
            "symbol": main_symbol,
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_period": {
                "start": df_pred["Date"].iloc[0].strftime("%Y-%m-%d"),
                "end": df_pred["Date"].iloc[-1].strftime("%Y-%m-%d"),
                "num_days": len(df_pred)
            },
            "overall": {
                "start_price": float(first_open),
                "end_price": float(last_close),
                "price_change": float(price_change),
                "price_change_pct": float(price_change_pct),
                "trend": trend,
                "min_price": float(df_pred[["Predicted Open", "Predicted Close"]].min().min()),
                "max_price": float(df_pred[["Predicted Open", "Predicted Close"]].max().max()),
                "avg_price": float(df_pred[["Predicted Open", "Predicted Close"]].stack().mean())
            },
            "milestones": milestones,
            "volatility": {
                "average": float(avg_volatility),
                "maximum": float(max_volatility),
                "minimum": float(min_volatility)
            },
            "daily_predictions": daily_predictions,
            "graph_structure": {
                "nodes": len(self.symbols),
                "cryptocurrency_symbols": self.symbols
            }
        }
        
        return summary


def predict_crypto_prices(model_path, preprocessor_path, output_dir="predictions", advanced_viz=True):
    """
    Predict cryptocurrency prices using a trained model.
    
    Args:
        model_path: Path to the saved model checkpoint
        preprocessor_path: Path to the saved preprocessor
        output_dir: Directory to save the results
        advanced_viz: Whether to use advanced visualization
    """
    # Create predictor
    predictor = CryptoPredictor(model_path, preprocessor_path)
    
    # Get main symbol (first in the list)
    main_symbol = predictor.symbols[0]
    
    print(f"\nPredicting prices for {main_symbol} (and related cryptocurrencies in the graph)")
    
    # Make predictions
    df_pred = predictor.predict()
    
    # Print predictions
    print("\nPredictions for", main_symbol)
    print("\nShort-term predictions (next 7 days):")
    print(df_pred.head(7).to_string(index=False))
    
    print("\nLong-term milestones:")
    milestones = [29, 89, 179]  # 1 month, 3 months, 6 months
    milestone_labels = ["1 Month", "3 Months", "6 Months"]
    
    milestone_df = pd.DataFrame()
    for i, (idx, label) in enumerate(zip(milestones, milestone_labels)):
        if idx < len(df_pred):
            milestone_df = pd.concat([milestone_df, df_pred.iloc[[idx]]])
    
    if not milestone_df.empty:
        print(milestone_df.to_string(index=False))
    
    # Create visualization
    print("\nCreating visualization...")
    plot_path = predictor.create_visualization(df_pred, output_dir, advanced_viz)
    
    # Generate summary
    summary = predictor.get_prediction_summary(df_pred)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Symbol: {summary['symbol']}")
    print(f"Prediction Period: {summary['prediction_period']['start']} to {summary['prediction_period']['end']}")
    print(f"Overall Trend: {summary['overall']['trend']}")
    print(f"Price Change: {summary['overall']['price_change']:.2f} ({summary['overall']['price_change_pct']:.2f}%)")
    
    print("\nPrice Milestones:")
    for period, info in summary['milestones'].items():
        if info['data']:
            print(f"  {period.replace('_', ' ').title()}: {info['data']['close']:.2f} ({'+' if info['data']['change_from_start'] >= 0 else ''}{info['data']['change_from_start']:.2f}%)")
    
    # Save summary to JSON
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"{main_symbol}_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSummary saved to {summary_path}")
    print(f"Visualization saved to {plot_path}")
    
    return df_pred, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict cryptocurrency prices with Graph Conv LSTM')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--preprocessor', type=str, required=True, help='Path to the preprocessor')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    parser.add_argument('--advanced', action='store_true', help='Use advanced visualization')
    
    args = parser.parse_args()
    
    predict_crypto_prices(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        output_dir=args.output_dir,
        advanced_viz=args.advanced
    )