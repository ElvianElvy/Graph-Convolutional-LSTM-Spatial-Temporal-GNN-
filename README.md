# Advanced Cryptocurrency Price Prediction System

A comprehensive PyTorch-based system for predicting cryptocurrency prices using a state-of-the-art **Spatial-Temporal Graph Neural Network** (ST-GNN) model. This advanced implementation integrates real-time and historical data from Binance, including **futures market metrics** like Open Interest and Funding Rates, to generate accurate price forecasts for up to 6 months ahead with enhanced visualization capabilities.

![Prediction Visualization](https://raw.githubusercontent.com/yourusername/crypto-prediction/main/docs/images/prediction_sample.png)

## 🚀 Features

- **Spatial-Temporal Graph Neural Network Architecture**
  - Models cryptocurrencies as nodes in a graph with edges representing market relationships
  - Combines Graph Convolutional Networks (GCN) with LSTM to capture both spatial and temporal patterns
  - Optional dynamic graph structure that evolves over time to capture changing market dynamics

- **Comprehensive Binance Data Integration**
  - OHLCV price and volume data for spot markets
  - Open Interest (total outstanding futures contracts)
  - Funding Rates (periodic payments between long/short positions)
  - Long/Short Ratio (proportion of long vs short positions)
  - Taker Buy/Sell Volume Ratio (market sentiment indicator)
  - Historical liquidation data

- **Advanced Market Analysis**
  - Cryptocurrency correlation networks with visualizations
  - Inter-market influence detection
  - Market regime identification
  - Multi-timeframe technical indicators

- **Enhanced Visualization**
  - Interactive Plotly-based dashboards
  - Candlestick charts with prediction overlays
  - Confidence interval bands
  - Comparative performance visualization
  - Market structure network graphs

- **Flexible Configuration System**
  - Feature selection (use any combination of data sources)
  - Model architecture customization
  - Training parameter optimization
  - Prediction horizon adjustment

## 📊 Performance Advantages

This system offers significant advantages over traditional forecasting methods:

1. **Cross-Market Information Flow**: Information flows between different cryptocurrencies through the graph structure
2. **Futures Market Insight**: Captures valuable signals from futures markets that are leading indicators
3. **Market Structure Understanding**: Models the complex relationships between different cryptocurrencies
4. **Adaptive Prediction**: Dynamic graphs adapt to changing market conditions
5. **Confidence Quantification**: Provides meaningful uncertainty estimates for predictions

## 🔧 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Internet connection for accessing Binance API

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crypto-prediction.git
   cd crypto-prediction
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

4. Create necessary directories:
   ```bash
   mkdir -p saved_models predictions data_cache logs analysis
   ```

## 🔍 Project Structure

```
crypto-prediction/
├── config.py               # Configuration management
├── main.py                 # Main application entry point
├── train.py                # Model training functionality
├── predict.py              # Prediction functionality
├── binance_api_fix.py      # Patches for Binance API
├── enable_dynamic_graph.py # Script to toggle dynamic graph mode
├── run_with_futures.py     # Script to run with futures data enabled
│
├── data/
│   ├── __init__.py         # Package initialization
│   ├── binance_api.py      # Enhanced Binance API interface
│   └── preprocessor.py     # Data preprocessing and graph construction
│
├── models/
│   ├── __init__.py         # Package initialization
│   └── graph_conv_lstm.py  # ST-GNN model implementation
│
├── utils/
│   ├── __init__.py         # Package initialization
│   └── metrics.py          # Evaluation metrics
│
├── saved_models/           # Directory for saved models
├── predictions/            # Directory for prediction outputs
├── data_cache/             # Directory for cached API data
├── logs/                   # Directory for training logs
└── analysis/               # Directory for analysis outputs
```

## 💻 Usage

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

Train a new model:

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

Visualize cryptocurrency correlations:

```bash
python main.py --correlations
```

Analyze a trained model:

```bash
python main.py --analyze --symbol BTCUSDT
```

### Using Futures Data

To enable futures market data (Open Interest, Funding Rates, etc.):

```bash
python run_with_futures.py --interactive
```

This script:
1. Patches the Binance API to handle futures market data requirements
2. Updates the config to enable futures data sources
3. Runs the main program with futures data enabled

### Using Dynamic Graphs

To enable dynamic graph structures that evolve over time:

```bash
python enable_dynamic_graph.py --enable
python main.py --interactive
```

To disable dynamic graphs:

```bash
python enable_dynamic_graph.py --disable
```

## ⚙️ Configuration

The system is highly configurable through the `config.json` file. Key settings include:

### Model Configuration

```json
"model": {
    "dropout": 0.3,
    "l2_reg": 1e-5,
    "use_attention": true,
    "graph_construction": "correlation",
    "correlation_threshold": 0.5,
    "graph_convolution_type": "standard",
    "feature_categories": ["ohlcv", "technical", "futures"],
    "use_dynamic_graph": true,
    "scaler_type": "minmax"
}
```

### Features Configuration

```json
"features": {
    "use_futures_data": true,
    "use_open_interest": true,
    "use_funding_rates": true,
    "use_long_short_ratio": true,
    "use_taker_buy_sell_ratio": true,
    "use_order_book": false,
    "use_advanced_features": true,
    "technicals": {
        "use_all": true,
        "moving_averages": true,
        "oscillators": true,
        "volatility": true,
        "momentum": true
    }
}
```

### Prediction Configuration

```json
"prediction": {
    "create_visualization": true,
    "prediction_days": 182,
    "advanced_visualization": true,
    "confidence_intervals": true,
    "export_formats": ["png", "html", "csv", "json"]
}
```

## 🧪 Model Architecture

The Spatial-Temporal Graph Neural Network combines graph-based spatial modeling with sequential temporal modeling:

### Graph Construction

- Cryptocurrencies are modeled as nodes in a graph
- Edges represent relationships (correlations or other metrics)
- Dynamic graphs update edge weights over time to capture evolving relationships

### Spatial-Temporal Processing

1. **Graph Convolutional Layers**
   - Capture market structure and inter-cryptocurrency effects
   - Apply attention mechanisms to focus on important relationships
   - Support both standard and Chebyshev graph convolutions

2. **LSTM Temporal Processing**
   - Process the time dimension of cryptocurrency data
   - Maintain long-term memory of market conditions
   - Graph-augmented cell states incorporate structural information

3. **Multi-Layer Feature Integration**
   - Combines outputs from spatial and temporal branches
   - Applies residual connections for better gradient flow
   - Regularization through dropout and L2 penalty

## 📈 Example Outputs

The system produces several outputs for each prediction:

### Price Predictions

A DataFrame containing open and close price predictions for the next 6 months:

```
                   Date  Predicted Open  Predicted Close  Change_From_Last
0  2025-03-12 00:00:00       3531.45        3568.29           2.45
1  2025-03-13 00:00:00       3565.78        3602.13           3.45
...
180 2025-09-08 00:00:00      5123.67        5245.89          50.67
181 2025-09-09 00:00:00      5267.34        5389.56          54.77
```

### Visualization

Interactive charts showing:
- Historical and predicted prices
- Confidence intervals
- Moving averages
- Key support/resistance levels
- Market volatility
- Prediction milestones (1 month, 3 months, 6 months)

### Analysis

- Cryptocurrency relationship network visualization
- Feature importance analysis
- Prediction confidence metrics
- Market regime identification

## 🔗 Integration with Trading Systems

The prediction outputs can be integrated with trading systems through:

1. **JSON Output Files**: Structured prediction data for algorithmic trading
2. **CSV Exports**: Tabular data for spreadsheet-based analysis
3. **Webhook Support**: Send predictions to external systems
4. **Visualization Exports**: HTML interactive dashboards for manual review

## 📚 Requirements

Key dependencies include:
- PyTorch 1.9+
- Networkx 2.6+
- Pandas 1.3+
- Numpy 1.20+
- Matplotlib 3.4+
- Plotly 5.5+
- Seaborn 0.11+
- Scikit-learn 0.24+
- Requests 2.26+
- Websockets 10.0+

Full requirements are listed in `requirements.txt`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for their excellent deep learning framework
- Binance for providing the API and market data
- Research in Graph Neural Networks for time series forecasting