import os
import pandas as pd
import requests
import zipfile
import io
import pyarrow.parquet as pq
import pyarrow as pa

# Налаштування
BINANCE_BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
DATA_DIR = "data"
PARQUET_FILE = os.path.join(DATA_DIR, "btc_1m_feb25.parquet")

# Створення порожнього .parquet файлу при запуску
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(PARQUET_FILE):
    pq.write_table(pa.Table.from_pandas(pd.DataFrame()), PARQUET_FILE, compression="snappy")
    print(f"[INFO] Створено порожній файл {PARQUET_FILE}")

def fetch_trading_pairs() -> list:
    """Отримує 100 найбільш ліквідних пар до BTC."""
    print("[INFO] Завантаження найбільш ліквідних пар до BTC...")
    url = "https://api.binance.com/api/v3/ticker/24hr"
    pairs = [(d["symbol"], float(d["quoteVolume"])) for d in requests.get(url).json() if d["symbol"].endswith("BTC")]
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    print(f"[INFO] Завантажено {len(sorted_pairs)} пар.")
    return [p[0] for p in sorted_pairs[:100]]

def fetch_binance_data(symbol: str, year: int, month: int, index: int) -> pd.DataFrame:
    """Завантажує 1m OHLCV-дані з Binance."""
    print(f"[INFO] [{index+1}/100] Завантаження даних для {symbol}...")
    url = f"{BINANCE_BASE_URL}/{symbol}/1m/{symbol}-1m-{year}-{month:02d}.zip"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"[ERROR] Не вдалося завантажити {symbol}: {response.status_code}")
            return pd.DataFrame()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            df = pd.read_csv(z.open(csv_name), names=["open", "high", "low", "close", "volume", "close_time"])
            df["timestamp"] = pd.to_datetime(df["close_time"] / 1e6, unit="s", errors="coerce")
            df.dropna(subset=["timestamp"], inplace=True)
            df["symbol"] = symbol
            print(f"[INFO] Завантажено {len(df)} рядків для {symbol}.")
            return df
    except Exception as e:
        print(f"[ERROR] Проблема з {symbol}: {e}")
        return pd.DataFrame()


def save_parquet(df: pd.DataFrame):
    """Зберігає DataFrame в .parquet файл."""
    pq.write_table(pa.Table.from_pandas(df), PARQUET_FILE, compression="snappy")
    print(f"[SUCCESS] Дані збережено у {PARQUET_FILE}")

def load_data(year: int = 2025, month: int = 2) -> pd.DataFrame:
    """Завантажує дані для 100 найбільш ліквідних пар та зберігає в .parquet."""
    print(f"[INFO] Завантаження даних для {year}-{month:02d}...")
    pairs = fetch_trading_pairs()
    all_data = []
    for index, pair in enumerate(pairs):
        df = fetch_binance_data(pair, year, month,index)
        if not df.empty:
            all_data.append(df)
        else:
            print(f"[INFO] [{index+1}/100] Пропущено пару {pair}, дані не доступні.")
    if not all_data:
        print("[ERROR] Не вдалося завантажити жодної пари!")
        return pd.DataFrame()
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"[INFO] Оновлення файлу...")
    save_parquet(combined_df)
    return combined_df

if __name__ == "__main__":
    load_data()
