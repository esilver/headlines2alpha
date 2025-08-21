import pandas as pd
import numpy as np
from datetime import datetime

# Load the data
file_path = '/Users/elisilver/workspace/headlines2alpha/glbx-mdp3-20250719-20250818.ohlcv-1m (1).csv'
df = pd.read_csv(file_path)

# Display basic information
print("="*60)
print("DATA STRUCTURE ANALYSIS")
print("="*60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nLast 5 rows:\n{df.tail()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Convert timestamp to datetime if needed
if 'ts_event' in df.columns:
    df['timestamp'] = pd.to_datetime(df['ts_event'])
    df = df.sort_values('timestamp')
    print(f"\nDate range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['date'] = df['timestamp'].dt.date

# Statistical analysis
print("\n" + "="*60)
print("STATISTICAL SUMMARY")
print("="*60)
print(df[['open', 'high', 'low', 'close', 'volume']].describe())

# Calculate returns and volatility
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
df['price_range'] = df['high'] - df['low']

# Calculate true range properly
high_prev_close = (df['high'] - df['close'].shift(1)).abs()
low_prev_close = (df['low'] - df['close'].shift(1)).abs()
df['true_range'] = pd.concat([df['price_range'], high_prev_close, low_prev_close], axis=1).max(axis=1)

print("\n" + "="*60)
print("RETURNS ANALYSIS")
print("="*60)
print(f"Average return: {df['returns'].mean():.6f}")
print(f"Std deviation: {df['returns'].std():.6f}")
print(f"Sharpe ratio (annualized): {(df['returns'].mean() / df['returns'].std()) * np.sqrt(252 * 390):.4f}")
print(f"Skewness: {df['returns'].skew():.4f}")
print(f"Kurtosis: {df['returns'].kurtosis():.4f}")

# Volume analysis
print("\n" + "="*60)
print("VOLUME ANALYSIS")
print("="*60)
print(f"Average volume: {df['volume'].mean():.0f}")
print(f"Median volume: {df['volume'].median():.0f}")
print(f"Volume std: {df['volume'].std():.0f}")
print(f"Max volume: {df['volume'].max():.0f}")
print(f"Min volume: {df['volume'].min():.0f}")

# Correlation analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)
df['volume_log'] = np.log1p(df['volume'])
df['abs_returns'] = df['returns'].abs()

correlations = df[['returns', 'abs_returns', 'volume', 'volume_log', 'price_range']].corr()
print(correlations)

# Time-based patterns
if 'hour' in df.columns:
    print("\n" + "="*60)
    print("HOURLY PATTERNS")
    print("="*60)
    hourly_stats = df.groupby('hour').agg({
        'returns': ['mean', 'std'],
        'volume': 'mean',
        'price_range': 'mean'
    }).round(6)
    print(hourly_stats)
    
    print("\n" + "="*60)
    print("DAY OF WEEK PATTERNS")
    print("="*60)
    daily_stats = df.groupby('day_of_week').agg({
        'returns': ['mean', 'std'],
        'volume': 'mean',
        'price_range': 'mean'
    }).round(6)
    print(daily_stats)

# Identify potential patterns
print("\n" + "="*60)
print("PATTERN IDENTIFICATION")
print("="*60)

# Mean reversion opportunities
df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
extreme_moves = df[df['z_score'].abs() > 2]
print(f"Extreme moves (|z-score| > 2): {len(extreme_moves)} ({len(extreme_moves)/len(df)*100:.2f}%)")

# Momentum patterns
df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

print(f"\nMomentum correlation with future returns:")
for period in [5, 10, 20]:
    corr = df[f'momentum_{period}'].corr(df['returns'].shift(-1))
    print(f"  {period}-period momentum: {corr:.4f}")

# Volume spikes
df['volume_ma'] = df['volume'].rolling(20).mean()
df['volume_spike'] = df['volume'] / df['volume_ma']
print(f"\nVolume spikes (>2x avg): {(df['volume_spike'] > 2).sum()} occurrences")

# Gap analysis
df['gap'] = df['open'] - df['close'].shift(1)
df['gap_pct'] = df['gap'] / df['close'].shift(1)
significant_gaps = df[df['gap_pct'].abs() > 0.001]
print(f"Significant gaps (>0.1%): {len(significant_gaps)} occurrences")

print("\n" + "="*60)
print("STRATEGY IDEAS BASED ON ANALYSIS")
print("="*60)

ideas = []

# Check for mean reversion
if df['z_score'].abs().mean() > 0.5:
    ideas.append("1. MEAN REVERSION: Strong mean reversion opportunities detected")
    ideas.append("   - Use Bollinger Bands or z-score based entries")
    ideas.append("   - Enter when price deviates 2+ std from mean")
    ideas.append("   - Exit at mean or opposite band")

# Check for momentum
momentum_corr = df['momentum_10'].corr(df['returns'].shift(-10))
if abs(momentum_corr) > 0.05:
    if momentum_corr > 0:
        ideas.append("\n2. MOMENTUM CONTINUATION: Positive momentum correlation found")
        ideas.append("   - Buy on strength with momentum indicators")
        ideas.append("   - Use moving average crossovers")
    else:
        ideas.append("\n2. MOMENTUM REVERSAL: Negative momentum correlation found")
        ideas.append("   - Fade strong moves")
        ideas.append("   - Use RSI divergences")

# Check for volume patterns
volume_return_corr = df['volume_spike'].corr(df['returns'].abs().shift(-1))
if abs(volume_return_corr) > 0.1:
    ideas.append("\n3. VOLUME-BASED: Volume predicts volatility")
    ideas.append("   - Trade breakouts on high volume")
    ideas.append("   - Use volume-weighted indicators")

# Check for time patterns
if 'hour' in df.columns:
    hourly_returns = df.groupby('hour')['returns'].mean()
    if hourly_returns.std() > hourly_returns.mean() * 2:
        ideas.append("\n4. TIME-BASED: Strong intraday patterns detected")
        ideas.append("   - Trade specific hours with higher edge")
        ideas.append(f"   - Best hours: {hourly_returns.nlargest(3).index.tolist()}")
        ideas.append(f"   - Worst hours: {hourly_returns.nsmallest(3).index.tolist()}")

# Check for gap trading
if len(significant_gaps) > len(df) * 0.01:
    gap_fill_rate = (df['gap'] * df['returns'].shift(-1) < 0).mean()
    ideas.append(f"\n5. GAP TRADING: {gap_fill_rate:.1%} gap fill rate")
    ideas.append("   - Trade gap fills in opening session")
    ideas.append("   - Fade opening gaps > 0.2%")

for idea in ideas:
    print(idea)

# Save processed data
df.to_csv('/Users/elisilver/workspace/headlines2alpha/processed_data.csv', index=False)
print("\n" + "="*60)
print("Processed data saved to processed_data.csv")
print("="*60)