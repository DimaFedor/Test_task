import numpy as np
import pandas as pd
from strategies.base import StrategyBase


class SMACrossover(StrategyBase):
    def __init__(self, price_data: pd.DataFrame, short_window: int = 20, long_window: int = 100):
        """
        Реалізація стратегії перетину ковзних середніх.
        :param price_data: Вхідні дані (ціни)
        :param short_window: Період короткої SMA
        :param long_window: Період довгої SMA
        """
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self) -> pd.DataFrame:
        """Генерує торгові сигнали на основі перетину двох SMA."""
        df = self.price_data.copy()
        df['short_sma'] = df['close'].rolling(self.short_window).mean()
        df['long_sma'] = df['close'].rolling(self.long_window).mean()
        df['signal'] = np.where(df['short_sma'] > df['long_sma'], 1, -1)
        df.dropna(inplace=True)
        self.signals = df[['close', 'short_sma', 'long_sma', 'signal']]
        return self.signals

    def run_backtest(self) -> pd.DataFrame:
        """Запускає бектест (простий варіант, поки без VectorBT)."""
        if self.signals is None:
            self.generate_signals()
        df = self.signals.copy()
        df['returns'] = df['close'].pct_change() * df['signal'].shift(1)
        df['equity'] = (1 + df['returns']).cumprod()
        self.results = df
        return df[['close', 'equity']]

    def get_metrics(self) -> dict:
        """Розраховує основні метрики стратегії."""
        if self.results is None:
            self.run_backtest()
        df = self.results
        total_return = df['equity'].iloc[-1] - 1
        sharpe_ratio = df['returns'].mean() / df['returns'].std() * np.sqrt(252) if df['returns'].std() != 0 else 0
        max_drawdown = (df['equity'] / df['equity'].cummax() - 1).min()
        metrics = {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
        return metrics
