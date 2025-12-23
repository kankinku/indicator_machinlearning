"""
Binance Data Loader for Crypto (BTC) 15-minute OHLCV Data.

Fetches historical data from Binance public API.
Supports multiple date ranges for training diverse market conditions.
"""
from __future__ import annotations

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import time

from src.shared.logger import get_logger

logger = get_logger("data.binance_loader")


class BinanceDataLoader:
    """
    Fetches OHLCV data from Binance public REST API.
    No API key required for public historical data.
    """
    
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
    ):
        """
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        """
        self.symbol = symbol
        self.interval = interval
    
    def fetch_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for a specific date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_start = start_ts
        
        logger.info(f">>> [Binance] Fetching {self.symbol} {self.interval} from {start_date} to {end_date}...")
        
        while current_start < end_ts:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 1000,  # Max per request
            }
            
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Move to next batch
                last_ts = data[-1][0]
                current_start = last_ts + 1
                
                # Rate limit protection
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Binance API Error: {e}")
                break
        
        if not all_data:
            logger.warning("No data fetched from Binance")
            return pd.DataFrame()
        
        # Parse data
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        # Keep only OHLCV
        df = df[["open", "high", "low", "close", "volume"]]
        
        logger.info(f">>> [Binance] Fetched {len(df)} candles")
        return df
    
    def fetch_multiple_ranges(self, ranges: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Fetch and combine multiple date ranges.
        
        Args:
            ranges: List of (start_date, end_date) tuples
        
        Returns:
            Combined DataFrame sorted by timestamp
        """
        dfs = []
        for start, end in ranges:
            df = self.fetch_range(start, end)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        
        logger.info(f">>> [Binance] Combined total: {len(combined)} candles")
        return combined


def fetch_btc_training_data() -> pd.DataFrame:
    """
    Fetch BTC training data for:
    - Recent 6 months with 1h candles (reduced data size for faster processing)
    
    Returns:
        OHLCV DataFrame
    """
    # [V8.3] 15분봉으로 복구 및 기간 연장 (데이터 사이즈 증대 -> GPU 효율 극대화)
    loader = BinanceDataLoader(symbol="BTCUSDT", interval="15m")
    
    today = datetime.now()
    one_year_ago = today - timedelta(days=365)
    
    ranges = [
        # COVID crash period (extreme volatility)
        ("2020-02-01", "2020-05-31"),
        # Recent 1 year
        (one_year_ago.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")),
    ]
    
    return loader.fetch_multiple_ranges(ranges)


if __name__ == "__main__":
    # Test run
    df = fetch_btc_training_data()
    print(f"Total candles: {len(df)}")
    print(df.head())
    print(df.tail())
