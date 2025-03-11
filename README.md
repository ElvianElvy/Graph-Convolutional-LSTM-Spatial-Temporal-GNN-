# Enhanced Cryptocurrency Prediction with Graph Convolutional LSTM

A comprehensive PyTorch-based system for predicting cryptocurrency prices using an advanced Graph Convolutional LSTM (Spatial-Temporal Graph Neural Network) model with expanded Binance API data integration. This application fetches historical and real-time data from Binance, including Open Interest, Funding Rates, and other futures market metrics, to model complex cryptocurrency relationships and generate accurate price forecasts for the next 6 months with enhanced visualization techniques.

## Key Features

- **Advanced Graph Convolutional LSTM Architecture**: Implementation of a state-of-the-art Spatial-Temporal Graph Neural Network that captures both the temporal price patterns and spatial relationships between cryptocurrencies
- **Comprehensive Binance Data Integration**: Incorporates a wide range of data sources including:
  - OHLCV (Open, High, Low, Close, Volume) price data
  - Open Interest (number of outstanding futures contracts)
  - Funding Rates (periodic payments between long/short positions)
  - Long/Short Ratio (proportion of long vs short positions)
  - Taker Buy/Sell Volume Ratio (ratio of aggressive buys vs sells)
  - Order Book Depth (market depth information)
- **Market Insight Analysis**: Models cryptocurrencies as nodes in a graph with edges representing their price correlations, allowing the model to capture market-wide trends and inter-cryptocurrency influences
- **Real-time Data Integration**: Fetches data from Binance WebSocket API for the most up-to-date information
- **Advanced Visualization**: Generates interactive and static visualizations of predictions with confidence intervals, trend analysis, and multi-dimensional market insights
- **Flexible Feature Selection**: Easily configure which data sources and features to use in your model
- **Interactive Mode**: User-friendly command-line interface for exploring available cryptocurrencies, training models, and generating predictions
- **Modular Architecture**: Clean, maintainable codebase designed for a single engineer to understand and extend

## Key Advantages of Enhanced Data Approach

This implementation provides several advantages over the standard approach:

1. **Futures Market Insight**: Captures valuable signals from futures markets that are often leading indicators of price movements
2. **Market Sentiment Capture**: Uses long/short ratios and funding rates to gauge market sentiment and positioning
3. **Liquidity Understanding**: Open interest and order book data provide insights into market liquidity and depth
4. **Improved Market Regime Detection**: Combination of spot and futures data helps identify different market regimes (trending, ranging, etc.)
5. **Cross-Market Information Flow**: Graph structure allows information to flow between different cryptocurrencies, improving prediction accuracy
6. **Enhanced Confidence Measures**: Multiple data sources enable better uncertainty quantification in predictions

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Internet connection for accessing Binance API

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto-graph-conv-lstm.git
   cd crypto-graph-conv-lstm
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
crypto-graph-conv-lstm/
├── config.py               # Configuration management
├── main.py                 # Main application entry point
├── predict.py              # Prediction functionality
├── train.py                # Model training functionality
├── example_usage.py        # Examples of using the system
├── requirements.txt        # Project dependencies
├── config.json             # Configuration file
├── data/
│   ├── binance_api.py      # Enhanced Binance API interface with futures data
│   └── preprocessor.py     # Advanced data preprocessing and graph construction
├── models/
│   └── graph_conv_lstm.py  # Graph Convolutional LSTM implementation
├── utils/
│   └── metrics.py          # Evaluation metrics
├── saved_models/           # Directory for saved models
├── predictions/            # Directory for prediction outputs
└── analysis/               # Directory for analysis outputs
```

## Data Sources

This enhanced version includes multiple data sources from Binance:

### Spot Market Data
- **OHLCV**: Traditional price and volume data
- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc.

### Futures Market Data
- **Open Interest**: Total number of outstanding futures contracts
- **Funding Rates**: Periodic payments between long and short positions
- **Long/Short Ratio**: Proportion of long vs short positions
- **Taker Buy/Sell Ratio**: Ratio of aggressive buys vs sells
- **Price Premium**: Difference between futures and spot price
- **Liquidation Data**: Information about forced position closures

### Derived Advanced Features
- **Volatility Indicators**: Multiple measures of market volatility
- **Momentum Metrics**: Price momentum across different timeframes
- **Market Regime Indicators**: Composite indicators of market conditions
- **Cross-market Metrics**: Relationships between different cryptocurrencies

## Usage

### Interactive Mode

Run the application in interactive mode for a user-friendly experience:

```bash
python main.py --interactive
```

This will present a menu with options to:
1. List available cryptocurrencies
2. List top cryptocurrencies by volume
3. Visualize cryptocurrency correlations
4. Train a new model
5. Predict prices
6. Train and predict
7. Analyze model
8. Optimize configuration
9. Exit

### Command Line Options

Train a new model with enhanced futures data:

```bash
python main.py --train --symbol BTCUSDT
```

Make predictions with an existing model:

```bash
python main.py --predict --symbol BTCUSDT
```

List available cryptocurrencies:

```bash
python main.py --list
```

List top cryptocurrencies by volume:

```bash
python main.py --top
```

Visualize cryptocurrency correlations:

```bash
python main.py --correlations
```

Analyze a trained model:

```bash
python main.py --analyze --symbol BTCUSDT
```

Optimize configuration for a specific cryptocurrency:

```bash
python main.py --optimize balanced --symbol BTCUSDT
```

### Example Usage

The repository includes an `example_usage.py` file that demonstrates different ways to use the system:

```bash
python example_usage.py
```

This will:
1. Train a model with basic settings
2. Explore available features for a cryptocurrency
3. Compare model performance with and without futures data
4. Fetch and visualize different types of futures market data

### Configuration

The default configuration is stored in `config.json`. You can modify this file to change parameters like:
- Training epochs, batch size, learning rate
- Model architecture parameters
- Feature selection options
- Visualization settings
- Binance API parameters

You can optimize the configuration for specific cryptocurrencies using the `--optimize` flag with different targets:
- `accuracy`: Prioritizes prediction accuracy (more complex model, longer training time)
- `speed`: Prioritizes fast training and inference (simpler model)
- `balanced`: Balances training speed and prediction accuracy

## Model Architecture

The Graph Convolutional LSTM architecture combines spatial and temporal modeling for cryptocurrency price prediction:

### Key Components:

1. **Graph Construction**:
   - Cryptocurrencies are represented as nodes in a graph
   - Edges are formed based on price correlations or other relationship metrics
   - Edge weights represent the strength of relationships between cryptocurrencies

2. **Spatial Processing (Graph Convolution)**:
   - Graph Convolutional layers capture the spatial relationships between cryptocurrencies
   - Chebyshev graph convolutions for higher-order graph relationships
   - Spatial attention mechanism to focus on the most important node connections

3. **Temporal Processing (LSTM)**:
   - Graph-augmented LSTM layers process time series data for each cryptocurrency
   - Temporal attention mechanism to focus on the most important time steps
   - Residual connections to improve gradient flow during training

4. **Feature Integration**:
   - Integrates spot and futures market data for comprehensive market insight
   - Multi-layer feature extraction with advanced regularization
   - Adaptive learning components to handle market regime changes

5. **Advanced Visualization**:
   - Interactive Plotly-based visualizations
   - Confidence interval representation for prediction uncertainty
   - Historical comparison and trend analysis

The model takes in multiple features including:
- Price data (Open, High, Low, Close)
- Volume data
- Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
- Futures market data (Open Interest, Funding Rates, etc.)
- Market-wide signals from related cryptocurrencies

## Performance Optimization

The model includes several optimizations to prevent underfitting and overfitting:

1. **Early Stopping**: Prevents overfitting by stopping training when validation loss stops improving
2. **Learning Rate Scheduling**: Cosine annealing with warm restarts for better convergence
3. **Gradient Clipping**: Prevents exploding gradients
4. **L2 Regularization**: Penalizes large weights to prevent overfitting
5. **Dropout**: Applied at multiple layers with different rates
6. **Attention Mechanisms**: Helps the model focus on relevant parts of the input sequence and graph
7. **Graph Structure Regularization**: Uses graph structure to constrain the model and reduce overfitting

## Evaluation Metrics

The system evaluates predictions using multiple metrics:

1. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual prices
2. **Root Mean Squared Error (RMSE)**: Square root of the average squared differences
3. **Mean Absolute Percentage Error (MAPE)**: Percentage difference between predicted and actual prices
4. **Directional Accuracy**: Percentage of correct predictions of price movement direction
5. **Rolling Window Metrics**: Evaluation across different time horizons (1 week, 1 month, 3 months, 6 months)

## Example Output

When making predictions, the system generates:

1. A DataFrame with predicted open and close prices for the next 6 months
2. Interactive and static visualizations showing:
   - Historical prices and future predictions
   - Confidence intervals for predictions
   - Moving averages and technical indicators
   - Price volatility analysis
   - Cryptocurrency relationship graph
   - Futures market indicators
3. A JSON summary with:
   - Overall price trend analysis
   - Milestone predictions (1 week, 1 month, 3 months, 6 months)
   - Prediction confidence metrics
   - Detailed daily forecasts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Research on Graph Neural Networks for time series prediction
- PyTorch team for the excellent deep learning framework
- Binance for providing the WebSocket API and futures market data