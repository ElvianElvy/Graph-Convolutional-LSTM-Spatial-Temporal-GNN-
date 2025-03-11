#!/usr/bin/env python3
import os
import json
import argparse
from datetime import datetime

# First import our patch to fix Binance API
import binance_api_fix

# Now import everything else
from main import main

# Update configuration to use futures data
def enable_futures_data():
    # Load config
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Enable futures data
    config['features']['use_futures_data'] = True
    config['features']['use_open_interest'] = True
    config['features']['use_funding_rates'] = True
    config['features']['use_long_short_ratio'] = True
    config['features']['use_taker_buy_sell_ratio'] = True
    config['features']['handle_missing_data'] = True
    
    # Add futures to feature categories if not already there
    if 'futures' not in config['model']['feature_categories']:
        config['model']['feature_categories'].append('futures')
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print("✓ Configuration updated to use futures data")

if __name__ == "__main__":
    # Enable futures data
    enable_futures_data()
    
    # Run the main program
    main()