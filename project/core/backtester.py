import vectorbt as vbt
import pandas as pd
import logging
from core.metrics import calculate_metrics

# Налаштуємо логування
logging.basicConfig(level=logging.INFO)  # Базовий рівень логування
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, strategy_class, price_data: pd.DataFrame, **strategy_kwargs):
        """
        Ініціалізує бектестер.
        :param strategy_class: Клас стратегії, що наслідує StrategyBase
        :param price_data: Дані цін
        :param strategy_kwargs: Додаткові параметри для стратегії
        """
        self.strategy = strategy_class(price_data, **strategy_kwargs)
        self.price_data = price_data
        self.results = None
        self.metrics = None

    def run(self):
        """Запускає бектестування через VectorBT."""
        # Видалити всі рядки, де 'close' є нульовим або NaN
        self.price_data = self.price_data[self.price_data['close'] > 0]

        # Генеруємо сигнали
        signals = self.strategy.generate_signals()

        # Видаляємо NaN значення в сигналах
        signals = signals.dropna(subset=['signal'])
        self.price_data = self.price_data.loc[signals.index]  # Оновлюємо дані для тих самих індексів

        entries = signals['signal'] == 1
        exits = signals['signal'] == -1

        # Створюємо портфель
        portfolio = vbt.Portfolio.from_signals(
            close=self.price_data['close'],
            entries=entries,
            exits=exits,
            size=1.0,
            fees=0.001,
            slippage=0.0005
        )

        self.results = portfolio.performance()
        self.metrics = calculate_metrics(portfolio)

        logger.info(f"Бектест успішно завершено. Загальні результати: {self.results.head()}")
        return self.results

    def save_results(self, csv_path: str, plot_path: str):
        """Зберігає результати в CSV і зображення."""
        if self.results is not None:
            self.results.to_csv(csv_path)
            self.strategy.signals['equity'] = (1 + self.strategy.signals['close'].pct_change() * self.strategy.signals['signal'].shift(1)).cumprod()
            self.strategy.signals[['close', 'equity']].plot(figsize=(12, 6)).figure.savefig(plot_path)

            logger.info(f"Результати збережено у файл: {csv_path} та графік у {plot_path}")

    def get_metrics(self):
        """Повертає метрики стратегії."""
        return self.metrics
