import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for cryptocurrency price predictions.
    
    Args:
        predictions: Predicted values (batch_size, num_forecasts)
                    where num_forecasts is the number of days * 2 (open and close)
        targets: Target values (batch_size, num_forecasts)
    
    Returns:
        Dictionary of metrics including MAE, RMSE, MAPE, and directional accuracy
    """
    # Ensure predictions and targets have the same shape
    if predictions.shape != targets.shape:
        raise ValueError(f"Shapes of predictions {predictions.shape} and targets {targets.shape} do not match")
    
    # Mean Absolute Error
    mae = mean_absolute_error(targets, predictions)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    # Mean Absolute Percentage Error with protection against division by zero
    # Create a mask for non-zero targets
    mask = targets != 0
    mape = 0.0
    if np.any(mask):
        mape = 100 * mean_absolute_percentage_error(targets[mask], predictions[mask])
    
    # Directional Accuracy (for close prices only)
    # Extract close prices (odd indices)
    close_indices = np.arange(1, targets.shape[1], 2)
    target_close = targets[:, close_indices]
    pred_close = predictions[:, close_indices]
    
    direction_correct = 0
    total_directions = 0
    
    # For each instance in the batch
    for i in range(predictions.shape[0]):
        # For each day in the prediction horizon
        for j in range(1, len(close_indices)):
            # Calculate day-over-day directional movement
            true_dir = target_close[i, j] - target_close[i, j-1]
            pred_dir = pred_close[i, j] - pred_close[i, j-1]
            
            # Check if directions match (both positive or both negative)
            if (true_dir * pred_dir) > 0:
                direction_correct += 1
            
            total_directions += 1
    
    # Calculate directional accuracy percentage
    dir_acc = 0.0
    if total_directions > 0:
        dir_acc = (direction_correct / total_directions) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'dir_acc': dir_acc
    }


def calculate_rolling_metrics(predictions: np.ndarray, targets: np.ndarray, window_sizes: list = [7, 30, 90]) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for different time horizons (rolling windows).
    
    Args:
        predictions: Predicted values (batch_size, num_forecasts)
        targets: Target values (batch_size, num_forecasts)
        window_sizes: List of window sizes in days
    
    Returns:
        Dictionary of metrics for each window size
    """
    metrics_by_window = {}
    
    for window in window_sizes:
        # Convert window size from days to indices (accounting for open and close prices)
        window_indices = window * 2
        
        # If prediction horizon is shorter than window, use the entire horizon
        if window_indices > predictions.shape[1]:
            window_indices = predictions.shape[1]
        
        # Calculate metrics for this window
        window_metrics = calculate_metrics(
            predictions[:, :window_indices],
            targets[:, :window_indices]
        )
        
        metrics_by_window[f"{window}_days"] = window_metrics
    
    return metrics_by_window


def evaluate_trend_accuracy(predictions: np.ndarray, targets: np.ndarray, thresholds: list = [0.0, 0.03, 0.05, 0.1]) -> Dict[str, float]:
    """
    Evaluate the model's accuracy in predicting price trends.
    
    Args:
        predictions: Predicted values (batch_size, num_forecasts)
        targets: Target values (batch_size, num_forecasts)
        thresholds: List of price change thresholds to evaluate
        
    Returns:
        Dictionary of trend accuracy metrics
    """
    results = {}
    
    # Extract first and last close prices for each sequence
    first_price_idx = 1  # First close price (odd index)
    
    # Use multiple prediction horizons
    horizons = [7, 30, 90, 180]  # 1 week, 1 month, 3 months, 6 months
    
    for horizon in horizons:
        # Calculate corresponding index for this horizon's close price
        last_close_idx = min(horizon * 2 - 1, predictions.shape[1] - 1)
        
        # Skip if we don't have enough data for this horizon
        if last_close_idx >= predictions.shape[1]:
            continue
        
        # Calculate price changes
        target_changes = (targets[:, last_close_idx] - targets[:, first_price_idx]) / targets[:, first_price_idx]
        pred_changes = (predictions[:, last_close_idx] - predictions[:, first_price_idx]) / predictions[:, first_price_idx]
        
        # Evaluate accuracy for different thresholds
        for threshold in thresholds:
            # True trend is considered bullish if change > threshold
            target_bullish = target_changes > threshold
            target_bearish = target_changes < -threshold
            target_sideways = ~(target_bullish | target_bearish)
            
            pred_bullish = pred_changes > threshold
            pred_bearish = pred_changes < -threshold
            pred_sideways = ~(pred_bullish | pred_bearish)
            
            # Calculate accuracy for each trend type
            bullish_correct = np.sum(target_bullish & pred_bullish)
            bearish_correct = np.sum(target_bearish & pred_bearish)
            sideways_correct = np.sum(target_sideways & pred_sideways)
            
            total_correct = bullish_correct + bearish_correct + sideways_correct
            total_samples = len(target_changes)
            
            accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
            
            # Calculate precision and recall for bullish predictions
            bullish_precision = 0
            if np.sum(pred_bullish) > 0:
                bullish_precision = bullish_correct / np.sum(pred_bullish) * 100
                
            bullish_recall = 0
            if np.sum(target_bullish) > 0:
                bullish_recall = bullish_correct / np.sum(target_bullish) * 100
            
            # Calculate precision and recall for bearish predictions
            bearish_precision = 0
            if np.sum(pred_bearish) > 0:
                bearish_precision = bearish_correct / np.sum(pred_bearish) * 100
                
            bearish_recall = 0
            if np.sum(target_bearish) > 0:
                bearish_recall = bearish_correct / np.sum(target_bearish) * 100
            
            result_key = f"d{horizon}_t{threshold:.2f}"
            results[result_key] = {
                "accuracy": accuracy,
                "bullish_precision": bullish_precision,
                "bullish_recall": bullish_recall,
                "bearish_precision": bearish_precision,
                "bearish_recall": bearish_recall
            }
    
    return results