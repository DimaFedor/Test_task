import pandas as pd
from strategies.sma_cross import SMACrossover
from core.backtester import Backtester
from core.metrics import plot_equity_curve, plot_performance_heatmap, compare_metrics

# Завантажуємо дані
df = pd.read_parquet("core/data/btc_1m_feb25.parquet")

# Запускаємо бектест для SMA Crossover
sma_strategy = Backtester(SMACrossover, df, short_window=20, long_window=100)
sma_results = sma_strategy.run()

# Зберігаємо результати
sma_strategy.save_results("results/sma_metrics.csv", "results/screenshots/sma_equity.png")

# Будуємо графіки
plot_equity_curve(sma_strategy.results, "results/screenshots/sma_equity_curve.png")

# Якщо тестуємо кілька стратегій, можемо зберегти всі метрики для порівняння
metrics_dict = {
    "SMA_Crossover": sma_strategy.get_metrics(),
    # Додамо інші стратегії сюди пізніше
}

# Порівнюємо стратегії
compare_metrics(metrics_dict, "results/screenshots/strategy_comparison.png")
