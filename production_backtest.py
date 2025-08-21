"""
Production Backtest System - Clean, Single Entry Point
Uses strategy_core module for all calculations
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Import core components
from strategy_core import (
    DataLoader, 
    BacktestEngine, 
    STRATEGY_CONFIGS,
    MarketHours
)


def run_single_backtest(data_path, strategy='robust', use_market_hours=True, verbose=True):
    """
    Run a single backtest
    
    Args:
        data_path: Path to CSV data file
        strategy: Strategy name ('robust', 'original', 'simple')
        use_market_hours: Whether to use market hours filter
        verbose: Print detailed output
    
    Returns:
        dict with results
    """
    # Load data
    df = DataLoader.load_and_prepare_data(data_path)
    
    if verbose:
        print(f"\n📊 Data loaded: {len(df)} bars")
        print(f"   Period: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Get strategy config
    params = STRATEGY_CONFIGS[strategy].copy()
    params['use_market_hours'] = use_market_hours
    
    # Run backtest
    engine = BacktestEngine(df)
    trades = engine.run_backtest(params)
    
    # Calculate metrics
    metrics = BacktestEngine.calculate_metrics(trades)
    
    if verbose:
        print(f"\n📈 Results for {strategy.upper()} strategy:")
        print(f"   Trades: {metrics['total_trades']} ({metrics['winners']}W / {metrics['losers']}L)")
        print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"   Total P&L: {metrics['total_pnl']:.1f} pts")
        print(f"   Sharpe: {metrics['sharpe']:.3f}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.1f} pts")
    
    return {
        'strategy': strategy,
        'data_file': Path(data_path).name,
        'trades': trades,
        'metrics': metrics
    }


def run_multi_period_analysis(strategy='robust', use_market_hours=True):
    """
    Run analysis across all available data files
    
    Args:
        strategy: Strategy name
        use_market_hours: Whether to use market hours filter
    
    Returns:
        Summary results
    """
    # Define data files
    data_files = [
        ('April 2025', 'data/glbx-mdp3-20250401-20250429.ohlcv-1m.csv'),
        ('May 2025', 'data/glbx-mdp3-20250501-20250530.ohlcv-1m.csv'),
        ('June 2025', 'data/glbx-mdp3-20250601-20250629.ohlcv-1m.csv'),
        ('July-Aug 2025', 'data/glbx-mdp3-20250719-20250818.ohlcv-1m.csv')
    ]
    
    print("=" * 100)
    print(f"MULTI-PERIOD ANALYSIS - {strategy.upper()} STRATEGY")
    if use_market_hours:
        print("Using US Market Hours (9:30 AM - 4:00 PM ET)")
    print("=" * 100)
    
    all_results = []
    total_pnl = 0
    all_win_rates = []
    all_sharpes = []
    
    for period_name, filename in data_files:
        file_path = f"/Users/elisilver/workspace/headlines2alpha/{filename}"
        
        if not Path(file_path).exists():
            print(f"\n⚠️ Skipping {period_name} - file not found")
            continue
        
        print(f"\n{'='*50}")
        print(f"Testing {period_name}")
        print("="*50)
        
        result = run_single_backtest(file_path, strategy, use_market_hours, verbose=True)
        all_results.append(result)
        
        metrics = result['metrics']
        total_pnl += metrics['total_pnl']
        if metrics['total_trades'] > 0:
            all_win_rates.append(metrics['win_rate'])
            all_sharpes.append(metrics['sharpe'])
    
    # Calculate summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    if all_win_rates:
        avg_win_rate = np.mean(all_win_rates)
        std_win_rate = np.std(all_win_rates)
        avg_sharpe = np.mean(all_sharpes)
        std_sharpe = np.std(all_sharpes)
        consistency_score = 1 / (1 + std_win_rate + abs(std_sharpe))
        
        print(f"\n📊 Overall Performance:")
        print(f"   Total P&L: {total_pnl:.1f} pts")
        print(f"   Average Win Rate: {avg_win_rate*100:.1f}% (±{std_win_rate*100:.1f}%)")
        print(f"   Average Sharpe: {avg_sharpe:.3f} (±{std_sharpe:.3f})")
        print(f"   Consistency Score: {consistency_score:.3f}")
        
        print(f"\n🎯 Strategy: {STRATEGY_CONFIGS[strategy]['description']}")
    
    return all_results


def compare_strategies(use_market_hours=True):
    """Compare all strategies across all periods"""
    
    print("=" * 100)
    print("STRATEGY COMPARISON")
    if use_market_hours:
        print("Using US Market Hours (9:30 AM - 4:00 PM ET)")
    print("=" * 100)
    
    comparison_results = {}
    
    for strategy_name in STRATEGY_CONFIGS.keys():
        print(f"\n{'='*50}")
        print(f"Testing {strategy_name.upper()} Strategy")
        print(f"{STRATEGY_CONFIGS[strategy_name]['description']}")
        print("="*50)
        
        results = run_multi_period_analysis(strategy_name, use_market_hours)
        comparison_results[strategy_name] = results
    
    # Final comparison table
    print("\n" + "="*100)
    print("FINAL COMPARISON")
    print("="*100)
    
    print("\n{:<15} {:>15} {:>15} {:>15} {:>15}".format(
        "Strategy", "Total Trades", "Avg Win Rate", "Total P&L", "Consistency"
    ))
    print("-" * 75)
    
    for strategy_name, results in comparison_results.items():
        total_trades = sum(r['metrics']['total_trades'] for r in results)
        total_pnl = sum(r['metrics']['total_pnl'] for r in results)
        win_rates = [r['metrics']['win_rate'] for r in results if r['metrics']['total_trades'] > 0]
        sharpes = [r['metrics']['sharpe'] for r in results if r['metrics']['total_trades'] > 0]
        
        if win_rates:
            avg_win_rate = np.mean(win_rates)
            consistency = 1 / (1 + np.std(win_rates) + abs(np.std(sharpes)))
        else:
            avg_win_rate = 0
            consistency = 0
        
        print("{:<15} {:>15} {:>15.1f}% {:>15.1f} {:>15.3f}".format(
            strategy_name.upper(),
            total_trades,
            avg_win_rate * 100,
            total_pnl,
            consistency
        ))
    
    print("\n✅ Recommendation: Use the strategy with highest consistency score")


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(description='Production Backtest System')
    parser.add_argument('--strategy', '-s', 
                       choices=['robust', 'original', 'simple', 'compare'],
                       default='robust',
                       help='Strategy to test (default: robust)')
    parser.add_argument('--data', '-d',
                       help='Path to specific data file')
    parser.add_argument('--no-market-hours', 
                       action='store_true',
                       help='Disable market hours filter')
    parser.add_argument('--multi-period', '-m',
                       action='store_true',
                       help='Run analysis across all periods')
    
    args = parser.parse_args()
    
    use_market_hours = not args.no_market_hours
    
    if args.strategy == 'compare':
        compare_strategies(use_market_hours)
    elif args.data:
        # Run single backtest on specific file
        result = run_single_backtest(args.data, args.strategy, use_market_hours)
        print(f"\n✅ Backtest complete")
    elif args.multi_period:
        # Run multi-period analysis
        run_multi_period_analysis(args.strategy, use_market_hours)
    else:
        # Default: run multi-period analysis for selected strategy
        run_multi_period_analysis(args.strategy, use_market_hours)


if __name__ == '__main__':
    main()