#!/usr/bin/env python3
import json

def toggle_dynamic_graph(enable=True):
    """Enable or disable dynamic graph in configuration"""
    config_path = 'config.json'
    
    # Load current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update dynamic graph setting
    if 'model' not in config:
        config['model'] = {}
    
    config['model']['use_dynamic_graph'] = enable
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    status = "enabled" if enable else "disabled"
    print(f"Dynamic graph {status} in configuration")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Toggle dynamic graph functionality")
    parser.add_argument("--enable", action="store_true", help="Enable dynamic graph")
    parser.add_argument("--disable", action="store_true", help="Disable dynamic graph")
    
    args = parser.parse_args()
    
    if args.disable:
        toggle_dynamic_graph(False)
    else:
        # Default to enable
        toggle_dynamic_graph(True)