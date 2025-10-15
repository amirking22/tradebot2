"""Visualization tools for backtest results and trading analysis."""
from __future__ import annotations
import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from datetime import datetime

# Configure matplotlib style
plt.style.use('seaborn-darkgrid')
palette = sns.color_palette("husl", 8)

class BacktestVisualizer:
    """Visualizes backtest results with various plots and metrics."""
    
    def __init__(self, backtest_result, output_dir: str = 'reports'):
        """Initialize with backtest results."""
        self.result = backtest_result
        self.output_dir = output_dir
        self.df_trades = pd.DataFrame(backtest_result.trades)
        self.df_equity = pd.DataFrame(backtest_result.equity_curve)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_equity_curve(self, title: str = 'Equity Curve', show: bool = True) -> str:
        """Plot the equity curve over time."""
        if self.df_equity.empty:
            return "No equity curve data available."
            
        plt.figure(figsize=(14, 7))
        
        # Plot equity curve
        plt.plot(
            self.df_equity['timestamp'],
            self.df_equity['equity'],
            label='Equity',
            color=palette[0],
            linewidth=2
        )
        
        # Add drawdown areas
        max_equity = self.df_equity['equity'].cummax()
        drawdown = (self.df_equity['equity'] - max_equity) / max_equity * 100
        
        plt.fill_between(
            self.df_equity['timestamp'],
            self.df_equity['equity'],
            max_equity,
            where=(self.df_equity['equity'] < max_equity),
            color='red',
            alpha=0.3,
            label='Drawdown'
        )
        
        # Add markers for trades
        if not self.df_trades.empty:
            for _, trade in self.df_trades.iterrows():
                color = 'green' if trade['pnl_pct'] > 0 else 'red'
                marker = '^' if trade['direction'] == 'LONG' else 'v'
                plt.scatter(
                    trade['exit_time'],
                    trade['balance'],
                    color=color,
                    marker=marker,
                    s=100,
                    alpha=0.7,
                    edgecolors='black'
                )
        
        # Formatting
        plt.title(f'{title}\nFinal Balance: ${self.df_equity["equity"].iloc[-1]:,.2f} ' \
                f'({((self.df_equity["equity"].iloc[-1] / self.df_equity["equity"].iloc[0] - 1) * 100):.2f}%)',
                fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.output_dir, 'equity_curve.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filename
    
    def plot_drawdown(self, title: str = 'Drawdown', show: bool = True) -> str:
        """Plot the drawdown over time."""
        if self.df_equity.empty:
            return "No equity curve data available."
            
        # Calculate drawdown
        max_equity = self.df_equity['equity'].cummax()
        drawdown = (self.df_equity['equity'] - max_equity) / max_equity * 100
        
        plt.figure(figsize=(14, 5))
        
        # Plot drawdown
        plt.fill_between(
            self.df_equity['timestamp'],
            drawdown,
            0,
            where=(drawdown < 0),
            color='red',
            alpha=0.3
        )
        
        # Formatting
        plt.title(f'{title}\nMax Drawdown: {drawdown.min():.2f}%', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(self.output_dir, 'drawdown.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filename
    
    def plot_monthly_returns(self, title: str = 'Monthly Returns', show: bool = True) -> str:
        """Plot monthly returns as a heatmap."""
        if self.df_trades.empty:
            return "No trade data available."
            
        # Create a copy of trades with datetime index
        df = self.df_trades.copy()
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df.set_index('exit_time', inplace=True)
        
        # Resample to monthly returns
        monthly_returns = df['pnl_pct'].resample('M').sum()
        
        # Create a pivot table for the heatmap
        monthly_returns_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month])\
                                            .sum().unstack()
        
        # Create month and year labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = monthly_returns.index.year.unique()
        
        plt.figure(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(
            monthly_returns_pivot,
            annot=True,
            fmt=".1f",
            cmap='RdYlGn',
            center=0,
            linewidths=0.5,
            annot_kws={"size": 9},
            cbar_kws={"label": "Return %"}
        )
        
        # Formatting
        plt.title(f'{title}\nAverage Monthly Return: {monthly_returns.mean():.2f}%', fontsize=14)
        plt.xlabel('Month')
        plt.ylabel('Year')
        
        # Set x and y labels
        plt.xticks(np.arange(12) + 0.5, months)
        plt.yticks(np.arange(len(years)) + 0.5, years, rotation=0)
        
        # Save the figure
        filename = os.path.join(self.output_dir, 'monthly_returns.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
            
        return filename
    
    def plot_trade_analysis(self, show: bool = True) -> Dict[str, str]:
        """Generate and save all trade analysis plots."""
        if self.df_trades.empty:
            return {"error": "No trade data available."}
            
        plots = {}
        
        # 1. Win/Loss Distribution
        plt.figure(figsize=(12, 6))
        sns.histplot(
            data=self.df_trades,
            x='pnl_pct',
            hue=np.where(self.df_trades['pnl_pct'] > 0, 'Win', 'Loss'),
            bins=30,
            kde=True,
            palette={True: 'green', False: 'red'}
        )
        plt.title('Win/Loss Distribution')
        plt.xlabel('Return (%)')
        plt.axvline(0, color='black', linestyle='--')
        
        filename = os.path.join(self.output_dir, 'win_loss_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plots['win_loss_dist'] = filename
        
        if show:
            plt.show()
        else:
            plt.close()
        
        # 2. Returns by Trade Direction
        if 'direction' in self.df_trades.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=self.df_trades,
                x='direction',
                y='pnl_pct',
                palette={'LONG': 'blue', 'SHORT': 'red'}
            )
            plt.title('Returns by Trade Direction')
            plt.xlabel('Direction')
            plt.ylabel('Return (%)')
            
            filename = os.path.join(self.output_dir, 'returns_by_direction.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plots['returns_by_direction'] = filename
            
            if show:
                plt.show()
            else:
                plt.close()
        
        # 3. Profit Factor by Month
        if 'exit_time' in self.df_trades.columns:
            df = self.df_trades.copy()
            df['month'] = pd.to_datetime(df['exit_time']).dt.to_period('M')
            monthly_pf = df.groupby('month').apply(
                lambda x: x[x['pnl_pct'] > 0]['pnl_pct'].sum() / 
                         abs(x[x['pnl_pct'] < 0]['pnl_pct'].sum())
                if (x['pnl_pct'] < 0).any() else float('inf')
            )
            
            plt.figure(figsize=(14, 5))
            monthly_pf.plot(kind='bar', color=palette[2])
            plt.axhline(1, color='red', linestyle='--')
            plt.title('Monthly Profit Factor')
            plt.xlabel('Month')
            plt.ylabel('Profit Factor')
            plt.xticks(rotation=45)
            
            filename = os.path.join(self.output_dir, 'monthly_profit_factor.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plots['monthly_profit_factor'] = filename
            
            if show:
                plt.show()
            else:
                plt.close()
        
        return plots
    
    def generate_html_report(self, backtest_metrics: Dict) -> str:
        """Generate an HTML report with all visualizations and metrics."""
        # Generate all plots
        equity_plot = self.plot_equity_curve(show=False)
        drawdown_plot = self.plot_drawdown(show=False)
        monthly_returns_plot = self.plot_monthly_returns(show=False)
        trade_plots = self.plot_trade_analysis(show=False)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 15px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{ 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #2c3e50;
                    margin: 5px 0;
                }}
                .metric-label {{ 
                    font-size: 14px; 
                    color: #7f8c8d;
                }}
                .plot-container {{ 
                    margin: 30px 0; 
                    text-align: center;
                }}
                .plot-container img {{ 
                    max-width: 100%; 
                    height: auto;
                    border: 1px solid #eee;
                    border-radius: 4px;
                }}
                .plot-title {{ 
                    margin: 10px 0; 
                    font-size: 18px;
                    color: #2c3e50;
                }}
                .positive {{ color: #27ae60; }}
                .negative {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Backtest Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>Key Metrics</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value {'positive' if backtest_metrics['total_pnl'] >= 0 else 'negative'}">
                            {backtest_metrics['total_pnl']:.2f}%
                        </div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{backtest_metrics['win_rate']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{backtest_metrics['profit_factor']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{backtest_metrics['max_drawdown']:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{backtest_metrics['sharpe_ratio']:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value">{backtest_metrics['total_trades']}</div>
                    </div>
                </div>
                
                <div class="plot-container">
                    <h3>Equity Curve</h3>
                    <img src="{equity_plot}" alt="Equity Curve">
                </div>
                
                <div class="plot-container">
                    <h3>Drawdown</h3>
                    <img src="{drawdown_plot}" alt="Drawdown">
                </div>
                
                <div class="plot-container">
                    <h3>Monthly Returns</h3>
                    <img src="{monthly_returns_plot}" alt="Monthly Returns">
                </div>
                
                <div class="plot-container">
                    <h3>Trade Analysis</h3>
                    <img src="{trade_plots.get('win_loss_dist', '')}" alt="Win/Loss Distribution">
                    <img src="{trade_plots.get('returns_by_direction', '')}" alt="Returns by Direction">
                    <img src="{trade_plots.get('monthly_profit_factor', '')}" alt="Monthly Profit Factor">
                </div>
                
                <div class="trades">
                    <h2>Trade History</h2>
                    {self._get_trades_table()}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = os.path.join(self.output_dir, 'backtest_report.html')
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return report_path
    
    def _get_trades_table(self) -> str:
        """Generate HTML table of trades."""
        if self.df_trades.empty:
            return "<p>No trades were executed during the backtest period.</p>"
            
        # Create a copy and format the data
        df = self.df_trades.copy()
        
        # Format numeric columns
        for col in ['entry_price', 'exit_price', 'pnl_pct', 'balance']:
            if col in df.columns:
                if col == 'pnl_pct':
                    df[col] = df[col].apply(lambda x: f"{x:+.2f}%")
                elif col == 'balance':
                    df[col] = df[col].apply(lambda x: f"${x:,.2f}")
                else:
                    df[col] = df[col].apply(lambda x: f"{x:.8f}" if isinstance(x, (int, float)) else x)
        
        # Convert to HTML
        return df.to_html(
            index=False, 
            classes='trades-table',
            border=0,
            justify='center',
            float_format=lambda x: f"{x:.8f}"
        )


def plot_candlestick_with_signals(
    ohlcv: pd.DataFrame,
    signals: pd.DataFrame,
    title: str = 'Price with Trading Signals',
    show: bool = True,
    save_path: str = None
) -> str:
    """Plot candlestick chart with buy/sell signals.
    
    Args:
        ohlcv: DataFrame with OHLCV data
        signals: DataFrame with buy/sell signals
        title: Plot title
        show: Whether to show the plot
        save_path: Path to save the plot (optional)
        
    Returns:
        Path to the saved plot or None
    """
    import mplfinance as mpf
    
    # Create a copy of the data
    df = ohlcv.copy()
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Create a style
    mc = mpf.make_marketcolors(
        up='green',
        down='red',
        edge='inherit',
        wick='inherit',
        volume='in',
        ohlc='i'
    )
    
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=':',
        gridcolor='#e0e0e0',
        facecolor='white',
        edgecolor='black',
        figcolor='white',
        rc={
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
        }
    )
    
    # Create additional plots
    add_plot = []
    
    # Add volume
    add_plot.append(
        mpf.make_addplot(
            df['volume'],
            type='bar',
            panel=1,
            color='gray',
            alpha=0.3,
            ylabel='Volume',
            secondary_y=False
        )
    )
    
    # Add buy/sell signals if provided
    if signals is not None and not signals.empty:
        # Convert signals to the same index as ohlcv
        signals = signals.reindex(df.index)
        
        # Add buy signals
        if 'buy' in signals.columns:
            add_plot.append(
                mpf.make_addplot(
                    signals['buy'],
                    type='scatter',
                    markersize=100,
                    marker='^',
                    color='green',
                    panel=0,
                    secondary_y=False
                )
            )
        
        # Add sell signals
        if 'sell' in signals.columns:
            add_plot.append(
                mpf.make_addplot(
                    signals['sell'],
                    type='scatter',
                    markersize=100,
                    marker='v',
                    color='red',
                    panel=0,
                    secondary_y=False
                )
            )
    
    # Create the plot
    fig, axes = mpf.plot(
        df,
        type='candle',
        style=style,
        title=title,
        ylabel='Price',
        volume=add_plot[0] if add_plot else False,
        addplot=add_plot[1:] if len(add_plot) > 1 else None,
        figratio=(12, 8),
        figscale=1.2,
        returnfig=True,
        show_nontrading=False
    )
    
    # Save the plot if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return save_path if save_path else None
