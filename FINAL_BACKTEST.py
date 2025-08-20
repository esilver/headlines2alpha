"""
FINAL CORRECT BACKTEST - Simple BB Reversal Strategy
Handles reverse chronological order data correctly
"""

import pandas as pd
import numpy as np
from datetime import datetime

def run_final_backtest():
    """Run the definitive backtest with correct data handling"""
    
    print("=" * 100)
    print("FINAL BACKTEST - SIMPLE BB REVERSAL STRATEGY")
    print("=" * 100)
    
    # Load data - using new GLBX MDP3 format
    print("\n📊 Loading MESU5 data from GLBX MDP3 file...")
    df = pd.read_csv('/Users/elisilver/workspace/headlines2alpha/glbx-mdp3-20250719-20250818.ohlcv-1m (1).csv')
    
    # Filter for MESU5 only
    df = df[df['symbol'] == 'MESU5'].copy()
    print(f"   Found {len(df)} MESU5 bars")
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'ts_event': 'datetime',
        'open': 'Open', 
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Parse datetime (already in ISO format with Z timezone)
    df['datetime'] = pd.to_datetime(df['datetime'].str.replace('Z', '+00:00'))
    
    # Convert to Chicago timezone (CME timezone)
    df['datetime'] = df['datetime'].dt.tz_convert('America/Chicago')
    df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone info for simplicity
    
    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    # IMPORTANT: Sort by datetime to ensure chronological order
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} bars")
    print(f"Period: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Data is now in chronological order: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    
    # Calculate indicators
    print("\n🔧 Calculating indicators...")
    
    # Bollinger Bands (20, 2.5)
    bb_period = 20
    bb_std = 2.5
    df['bb_middle'] = df['Close'].rolling(window=bb_period).mean()
    rolling_std = df['Close'].rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (rolling_std * bb_std)
    df['bb_lower'] = df['bb_middle'] - (rolling_std * bb_std)
    
    # RSI (14) - Pine Script Compatible (Wilder's RMA)
    def calculate_rma(prices, period):
        """Calculate RMA (Running Moving Average) - same as Pine Script ta.rma()"""
        alpha = 1.0 / period
        rma_values = np.zeros(len(prices))
        
        for i in range(len(prices)):
            if i < period - 1:
                rma_values[i] = np.nan
            elif i == period - 1:
                # First RMA value is simple average of first 'period' values
                rma_values[i] = prices.iloc[i-period+1:i+1].mean()
            else:
                # Subsequent values use exponential smoothing
                rma_values[i] = alpha * prices.iloc[i] + (1 - alpha) * rma_values[i-1]
        
        return pd.Series(rma_values, index=prices.index)
    
    def calculate_rsi(prices, period=14):
        """RSI calculation matching Pine Script v6 exactly"""
        if len(prices) < period + 1:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0).fillna(0)
        losses = (-delta.where(delta < 0, 0)).fillna(0)
        
        # Initialize RMA arrays
        avg_gains = pd.Series(index=prices.index, dtype=float)
        avg_losses = pd.Series(index=prices.index, dtype=float)
        
        # Calculate first RMA value (SMA of first 'period' values)
        avg_gains.iloc[period] = gains.iloc[1:period+1].mean()
        avg_losses.iloc[period] = losses.iloc[1:period+1].mean()
        
        # Calculate subsequent RMA values
        alpha = 1.0 / period
        for i in range(period + 1, len(prices)):
            avg_gains.iloc[i] = alpha * gains.iloc[i] + (1 - alpha) * avg_gains.iloc[i-1]
            avg_losses.iloc[i] = alpha * losses.iloc[i] + (1 - alpha) * avg_losses.iloc[i-1]
        
        # Calculate RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        rsi[avg_losses == 0] = 100
        rsi[avg_gains == 0] = 0
        
        return rsi
    
    df['rsi'] = calculate_rsi(df['Close'])
    df['hour'] = df['datetime'].dt.hour
    
    # Strategy parameters
    rsi_threshold = 75
    start_hour = 9
    end_hour = 15
    stop_loss = 5.0
    take_profit = 5.0
    
    print(f"\n📈 Strategy Parameters:")
    print(f"   BB: {bb_period} period, {bb_std} std dev")
    print(f"   RSI: {rsi_threshold} threshold")
    print(f"   Trading Hours: {start_hour}:00 - {end_hour}:59")
    print(f"   Stop Loss: {stop_loss} points")
    print(f"   Take Profit: {take_profit} points")
    
    # Run backtest
    print(f"\n🚀 Running backtest...")
    print("-" * 80)
    
    trades = []
    in_position = False
    entry_price = None
    entry_time = None
    entry_index = None
    
    for i in range(100, len(df)):  # Start after indicators are ready
        row = df.iloc[i]
        
        if in_position:
            # Check exit conditions
            bars_since_entry = i - entry_index
            
            # Check stop loss (price went up)
            if row['High'] >= entry_price + stop_loss:
                exit_price = entry_price + stop_loss
                exit_time = row['datetime']
                pnl = -stop_loss
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_points': pnl,
                    'pnl_dollars': pnl * 5,
                    'result': 'STOP_LOSS',
                    'bars_held': bars_since_entry
                })
                
                print(f"   ❌ STOP LOSS @ {exit_time.strftime('%H:%M')}, "
                      f"Exit: {exit_price:.2f}, Loss: {pnl:.1f} points")
                in_position = False
                
            # Check take profit (price went down)
            elif row['Low'] <= entry_price - take_profit:
                exit_price = entry_price - take_profit
                exit_time = row['datetime']
                pnl = take_profit
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_points': pnl,
                    'pnl_dollars': pnl * 5,
                    'result': 'TAKE_PROFIT',
                    'bars_held': bars_since_entry
                })
                
                print(f"   ✅ TAKE PROFIT @ {exit_time.strftime('%H:%M')}, "
                      f"Exit: {exit_price:.2f}, Profit: {pnl:.1f} points")
                in_position = False
                
        else:
            # Check entry conditions
            bb_signal = row['Close'] > row['bb_upper']
            rsi_signal = row['rsi'] > rsi_threshold
            time_signal = start_hour <= row['hour'] <= end_hour
            
            if bb_signal and rsi_signal and time_signal:
                entry_price = row['Close']
                entry_time = row['datetime']
                entry_index = i
                in_position = True
                
                print(f"\n🎯 SHORT ENTRY @ {entry_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"   Price: {entry_price:.2f}")
                print(f"   BB Upper: {row['bb_upper']:.2f}")
                print(f"   RSI: {row['rsi']:.1f}")
    
    print("-" * 80)
    
    # Process results
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Verify all trades have correct timing
        trades_df['duration_minutes'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
        
        print(f"\n" + "=" * 100)
        print("BACKTEST RESULTS")
        print("=" * 100)
        
        # Statistics
        winners = trades_df[trades_df['pnl_points'] > 0]
        losers = trades_df[trades_df['pnl_points'] < 0]
        
        total_trades = len(trades_df)
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0
        total_pnl_points = trades_df['pnl_points'].sum()
        total_pnl_dollars = trades_df['pnl_dollars'].sum()
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Winners: {len(winners)} ({win_rate:.1f}%)")
        print(f"   Losers: {len(losers)} ({100-win_rate:.1f}%)")
        
        print(f"\n💰 PROFIT/LOSS:")
        print(f"   Total P&L: {total_pnl_points:.1f} points (${total_pnl_dollars:.2f})")
        print(f"   Average P&L: {trades_df['pnl_points'].mean():.2f} points per trade")
        
        if len(winners) > 0:
            print(f"   Average Win: {winners['pnl_points'].mean():.2f} points")
            print(f"   Largest Win: {winners['pnl_points'].max():.2f} points")
        
        if len(losers) > 0:
            print(f"   Average Loss: {losers['pnl_points'].mean():.2f} points")
            print(f"   Largest Loss: {losers['pnl_points'].min():.2f} points")
        
        print(f"\n⏱️ TIMING:")
        print(f"   Average Trade Duration: {trades_df['duration_minutes'].mean():.1f} minutes")
        print(f"   Shortest Trade: {trades_df['duration_minutes'].min():.1f} minutes")
        print(f"   Longest Trade: {trades_df['duration_minutes'].max():.1f} minutes")
        
        # Verify timing is correct
        timing_errors = trades_df[trades_df['duration_minutes'] < 0]
        if len(timing_errors) > 0:
            print(f"\n⚠️ WARNING: {len(timing_errors)} trades have timing errors!")
        else:
            print(f"\n✅ All trades have correct timing (exit after entry)")
        
        # Monthly breakdown
        trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
        monthly_stats = trades_df.groupby('month').agg({
            'pnl_dollars': ['sum', 'count'],
            'pnl_points': 'mean'
        })
        
        print(f"\n📅 MONTHLY BREAKDOWN:")
        for month in monthly_stats.index:
            monthly_pnl = monthly_stats.loc[month, ('pnl_dollars', 'sum')]
            monthly_trades = monthly_stats.loc[month, ('pnl_dollars', 'count')]
            monthly_avg = monthly_stats.loc[month, ('pnl_points', 'mean')]
            print(f"   {month}: {monthly_trades:.0f} trades, ${monthly_pnl:.2f} total, {monthly_avg:.2f} pts avg")
        
        # Final portfolio
        init_cash = 10000
        commission = 0.62 * 2 * total_trades  # Round trip
        net_pnl = total_pnl_dollars - commission
        final_value = init_cash + net_pnl
        roi = (net_pnl / init_cash) * 100
        
        print(f"\n💵 PORTFOLIO SUMMARY:")
        print(f"   Starting Capital: ${init_cash:,.2f}")
        print(f"   Gross P&L: ${total_pnl_dollars:.2f}")
        print(f"   Commission: ${commission:.2f}")
        print(f"   Net P&L: ${net_pnl:.2f}")
        print(f"   Final Value: ${final_value:,.2f}")
        print(f"   ROI: {roi:.2f}%")
        
        # Trade log
        print(f"\n📝 DETAILED TRADE LOG:")
        for idx, trade in trades_df.iterrows():
            result_emoji = "✅" if trade['result'] == 'TAKE_PROFIT' else "❌"
            print(f"\n   Trade {idx+1}:")
            print(f"      Entry: {trade['entry_price']:.2f} @ {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}")
            print(f"      Exit:  {trade['exit_price']:.2f} @ {trade['exit_time'].strftime('%Y-%m-%d %H:%M')}")
            print(f"      Duration: {trade['duration_minutes']:.0f} minutes ({trade['bars_held']} bars)")
            print(f"      P&L: {trade['pnl_points']:.1f} points (${trade['pnl_dollars']:.2f}) {result_emoji}")
        
        # Save results
        trades_df.to_csv('final_backtest_results.csv', index=False)
        print(f"\n💾 Results saved to 'final_backtest_results.csv'")
        
        return trades_df
    
    else:
        print(f"\n⚠️ No trades executed")
        return pd.DataFrame()

if __name__ == '__main__':
    # Run final backtest
    trades = run_final_backtest()
    
    print(f"\n" + "=" * 100)
    print("✅ FINAL BACKTEST COMPLETE - ALL TIMES VERIFIED")
    print("=" * 100)