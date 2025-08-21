"""
Core Strategy Components - Single Source of Truth
Consolidated functions to avoid duplication
"""

import pandas as pd
import numpy as np
import pytz
from datetime import time
import warnings
warnings.filterwarnings('ignore')


class IndicatorCalculator:
    """TradingView-compatible indicator calculations"""
    
    @staticmethod
    def calculate_rma(values, period):
        """
        Pine Script's ta.rma (Relative Moving Average) implementation
        Used internally by RSI calculation
        """
        rma = np.full(len(values), np.nan)
        alpha = 1.0 / period
        
        # Find first valid index
        first_valid_idx = -1
        for i in range(len(values)):
            if not np.isnan(values[i]):
                first_valid_idx = i
                break
        
        if first_valid_idx == -1 or first_valid_idx + period > len(values):
            return rma
        
        # Initial RMA is SMA of first 'period' values
        initial_sum = 0
        for i in range(first_valid_idx, first_valid_idx + period):
            initial_sum += values[i] if not np.isnan(values[i]) else 0
        
        rma[first_valid_idx + period] = initial_sum / period
        
        # Calculate subsequent RMA values
        for i in range(first_valid_idx + period + 1, len(values)):
            if not np.isnan(rma[i - 1]):
                rma[i] = alpha * (values[i] if not np.isnan(values[i]) else 0) + (1 - alpha) * rma[i - 1]
        
        return rma
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """
        RSI calculation exactly matching TradingView/Pine Script
        This is the correct implementation that matches the HTML app
        """
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        # Calculate price changes
        changes = np.zeros(len(prices))
        changes[0] = 0  # First bar has no change
        for i in range(1, len(prices)):
            changes[i] = prices[i] - prices[i - 1]
        
        # Separate gains and losses
        gains = np.maximum(changes, 0)
        losses = np.maximum(-changes, 0)
        
        # Calculate RMA of gains and losses
        avg_gains = IndicatorCalculator.calculate_rma(gains, period)
        avg_losses = IndicatorCalculator.calculate_rma(losses, period)
        
        # Calculate RSI with Pine Script's edge case handling
        rsi = np.full(len(prices), np.nan)
        for i in range(len(prices)):
            if not np.isnan(avg_gains[i]) and not np.isnan(avg_losses[i]):
                if avg_losses[i] == 0:
                    rsi[i] = 100
                elif avg_gains[i] == 0:
                    rsi[i] = 0
                else:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
        """
        Calculate Bollinger Bands
        Returns dict with 'upper', 'middle', 'lower' as numpy arrays
        """
        middle = pd.Series(prices).rolling(window=period).mean().values
        rolling_std = pd.Series(prices).rolling(window=period).std(ddof=1).values
        upper = middle + (rolling_std * std_dev)
        lower = middle - (rolling_std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }


class DataLoader:
    """Standardized data loading and preparation"""
    
    @staticmethod
    def load_and_prepare_data(file_path, timezone='America/Chicago'):
        """
        Load and prepare GLBX MDP3 data for backtesting
        
        Args:
            file_path: Path to CSV file
            timezone: Target timezone for datetime conversion
        
        Returns:
            DataFrame with standardized columns
        """
        df = pd.read_csv(file_path)
        
        # Filter for MESU5
        df = df[df['symbol'] == 'MESU5'].copy()
        
        # Rename columns to standard format
        df = df.rename(columns={
            'ts_event': 'datetime_raw',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        # Parse datetime - keep raw for reference
        df['datetime_utc'] = pd.to_datetime(df['datetime_raw'].str.replace('Z', ''))
        
        # Convert to specified timezone
        df['datetime'] = pd.to_datetime(df['datetime_raw'].str.replace('Z', '+00:00'))
        df['datetime'] = df['datetime'].dt.tz_convert(timezone)
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any NaN rows
        df = df.dropna()
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df


class MarketHours:
    """Market hours utilities"""
    
    @staticmethod
    def is_market_hours(dt_et):
        """
        Check if datetime is within US market hours
        
        Args:
            dt_et: datetime in Eastern Time
        
        Returns:
            bool: True if within market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
        """
        market_open = time(9, 30)   # 9:30 AM
        market_close = time(16, 0)   # 4:00 PM
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if dt_et.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if within market hours
        current_time = dt_et.time()
        return market_open <= current_time < market_close
    
    @staticmethod
    def add_market_hours_column(df, timezone='America/New_York'):
        """
        Add market hours column to dataframe
        
        Args:
            df: DataFrame with datetime column
            timezone: Market timezone (default Eastern)
        
        Returns:
            DataFrame with 'in_market_hours' column added
        """
        tz = pytz.timezone(timezone)
        
        if 'datetime_utc' in df.columns:
            df['datetime_et'] = df['datetime_utc'].dt.tz_localize('UTC').dt.tz_convert(tz)
        else:
            # Assume datetime is in local timezone, convert to ET
            df['datetime_et'] = pd.to_datetime(df['datetime']).dt.tz_localize(None).dt.tz_localize(tz)
        
        df['in_market_hours'] = df['datetime_et'].apply(MarketHours.is_market_hours)
        df['hour_et'] = df['datetime_et'].dt.hour
        df['minute_et'] = df['datetime_et'].dt.minute
        
        return df


class BacktestEngine:
    """Core backtesting logic"""
    
    def __init__(self, data):
        """
        Initialize backtester with data
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data
        self.indicators = IndicatorCalculator()
        
    def add_indicators(self, bb_period=22, bb_std=2.6, rsi_period=14):
        """Add technical indicators to data"""
        close_prices = self.data['Close'].values
        
        # Calculate Bollinger Bands
        bb = self.indicators.calculate_bollinger_bands(close_prices, bb_period, bb_std)
        self.data['bb_upper'] = bb['upper']
        self.data['bb_middle'] = bb['middle']
        self.data['bb_lower'] = bb['lower']
        
        # Calculate RSI
        self.data['rsi'] = self.indicators.calculate_rsi(close_prices, rsi_period)
        
        return self.data
    
    def run_backtest(self, params):
        """
        Run backtest with given parameters
        
        Args:
            params: dict with strategy parameters
                - bb_period: Bollinger Band period
                - bb_std: Bollinger Band standard deviations
                - rsi_threshold: RSI threshold for entry
                - stop_loss: Stop loss in points
                - take_profit: Take profit in points
                - use_market_hours: bool, whether to filter by market hours
                - start_hour: Start hour for time filter
                - end_hour: End hour for time filter
        
        Returns:
            list of trade dictionaries
        """
        # Add indicators
        self.add_indicators(params['bb_period'], params['bb_std'])
        
        # Prepare time filter
        if params.get('use_market_hours', False):
            self.data = MarketHours.add_market_hours_column(self.data)
        
        trades = []
        in_position = False
        entry_price = None
        entry_time = None
        entry_index = None
        
        # Start after indicators are ready
        start_index = max(100, params['bb_period'])
        
        for i in range(start_index, len(self.data)):
            row = self.data.iloc[i]
            
            if in_position:
                # Check exit conditions
                if row['High'] >= entry_price + params['stop_loss']:
                    # Stop loss hit
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': row['datetime'],
                        'entry_price': entry_price,
                        'exit_price': entry_price + params['stop_loss'],
                        'pnl_points': -params['stop_loss'],
                        'result': 'STOP_LOSS'
                    })
                    in_position = False
                    
                elif row['Low'] <= entry_price - params['take_profit']:
                    # Take profit hit
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': row['datetime'],
                        'entry_price': entry_price,
                        'exit_price': entry_price - params['take_profit'],
                        'pnl_points': params['take_profit'],
                        'result': 'TAKE_PROFIT'
                    })
                    in_position = False
                    
            else:
                # Check entry conditions
                bb_signal = row['Close'] > row['bb_upper']
                rsi_signal = row['rsi'] > params['rsi_threshold']
                
                # Time filter
                if params.get('use_market_hours', False):
                    time_signal = row['in_market_hours']
                elif 'start_hour' in params and 'end_hour' in params:
                    hour = row['datetime'].hour if 'datetime' in self.data.columns else i % 24
                    time_signal = params['start_hour'] <= hour <= params['end_hour']
                else:
                    time_signal = True
                
                if bb_signal and rsi_signal and time_signal:
                    entry_price = row['Close']
                    entry_time = row['datetime']
                    entry_index = i
                    in_position = True
        
        return trades
    
    @staticmethod
    def calculate_metrics(trades):
        """
        Calculate performance metrics from trades
        
        Args:
            trades: list of trade dictionaries
        
        Returns:
            dict with performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'sharpe': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winners = trades_df[trades_df['pnl_points'] > 0]
        losers = trades_df[trades_df['pnl_points'] < 0]
        
        win_rate = len(winners) / total_trades
        total_pnl = trades_df['pnl_points'].sum()
        avg_pnl = trades_df['pnl_points'].mean()
        
        # Sharpe ratio
        if trades_df['pnl_points'].std() > 0:
            sharpe = trades_df['pnl_points'].mean() / trades_df['pnl_points'].std()
        else:
            sharpe = 0
        
        # Profit factor
        total_wins = winners['pnl_points'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['pnl_points'].sum()) if len(losers) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Max drawdown
        cumulative_pnl = trades_df['pnl_points'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'sharpe': sharpe,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'winners': len(winners),
            'losers': len(losers)
        }


# Preset strategy configurations
STRATEGY_CONFIGS = {
    'robust': {
        'bb_period': 22,
        'bb_std': 2.6,
        'rsi_threshold': 70,
        'stop_loss': 5.0,
        'take_profit': 5.0,
        'description': 'Most consistent across all market conditions'
    },
    'original': {
        'bb_period': 20,
        'bb_std': 2.5,
        'rsi_threshold': 75,
        'stop_loss': 5.0,
        'take_profit': 5.0,
        'description': 'Original parameters from FINAL_BACKTEST.py'
    },
    'simple': {
        'bb_period': 20,
        'bb_std': 2.0,
        'rsi_threshold': 70,
        'stop_loss': 5.0,
        'take_profit': 5.0,
        'description': 'Simplified parameters'
    }
}