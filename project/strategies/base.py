from abc import ABC, abstractmethod
import pandas as pd


class StrategyBase(ABC):
    def __init__(self, price_data: pd.DataFrame):
        """
        Базовий клас для торгових стратегій.
        :param price_data: Вхідні дані з цінами (pandas DataFrame)
        """
        self.price_data = price_data
        self.signals = None
        self.results = None

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Метод для генерації торгових сигналів."""
        pass

    @abstractmethod
    def run_backtest(self) -> pd.DataFrame:
        """Метод для запуску бектесту."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """Метод для обчислення ключових метрик стратегії."""
        pass
