from __future__ import annotations

import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
from typing import List, Optional

from src.shared.logger import get_logger

logger = get_logger("data.loader")

class DataLoader:
    """
    Robust Data Loader for Financial & Macro Data.
    Merges Target Asset (e.g. QQQ) with Context Assets (SPY, VIX) and Macro (Yield Curve).
    """

    def __init__(self, target_ticker: str = "QQQ", start_date: str = "2010-01-01"):
        self.target_ticker = target_ticker
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")

    def fetch_all(self) -> pd.DataFrame:
        """
        Main pipeline to fetch and merge all data sources.
        """
        from src.config import config # Import here to avoid circular if any
        
        logger.info(f">>> [Data] Fetching Market Data for {self.target_ticker}...")
        main_df = self._fetch_ohlcv(self.target_ticker)
        
        if main_df.empty:
            logger.error(f"Failed to fetch target ticker {self.target_ticker}")
            return pd.DataFrame()

        # 1. Fetch Macro/Context from YFinance (Fast & reliable)
        logger.info(f">>> [Data] Fetching Macro & Context Assets...")
        
        # Define Context Map: {Ticker: Column_Name_Prefix}
        # We want Close price for these
        context_map = {
            "SPY": "spy",
            "TLT": "tlt",
            "GLD": "gld"
        }
        
        # Add Config Macro Tickers
        # Loop through config.MACRO_TICKERS
        for name, ticker in config.MACRO_TICKERS.items():
            context_map[ticker] = name.lower() # e.g. "^VIX" -> "vix"

        context_dfs = []
        for ticker, prefix in context_map.items():
            try:
                # We need Close, maybe Volume if useful? Only Close for macro usually.
                c_df = self._fetch_ohlcv(ticker)
                
                # Check if empty
                if c_df.empty:
                    continue
                    
                # Extract Close and rename
                c_clean = c_df[["close"]].rename(columns={"close": f"{prefix}_close"})
                context_dfs.append(c_clean)
            except Exception as e:
                logger.warning(f"Failed to fetch context {ticker}: {e}")
        
        # 2. Fetch Deep Macro from FRED (Slower, requires pandas_datareader)
        # T10Y2Y (Yield Curve), UNRATE (Unemployment), FEDFUNDS (Fed Rate)
        # Using a try-except block strictly
        macro_df = pd.DataFrame()
        try:
            logger.info(">>> [Data] Fetching Deep Macro (FRED)...")
            fred_ids = ["T10Y2Y", "UNRATE", "FEDFUNDS"]
            macro_df = self._fetch_fred(fred_ids)
            macro_df.columns = [c.lower() for c in macro_df.columns]
        except Exception as e:
            logger.warning(f"[Data] FRED fetch skipped/failed (No API Key?): {e}. Using YFinance proxies only.")

        # 3. Merge Strategy
        df = main_df.copy()
        
        # Merge YFinance Context
        for c_df in context_dfs:
            # Join left to keep main timeline
            df = df.join(c_df, how="left")
            
        # Merge FRED Macro
        if not macro_df.empty:
            # Resample FRED to daily (most are monthly/daily) -> Forward Fill
            # Reindex to match main df
            macro_df = macro_df.reindex(df.index, method="ffill")
            df = df.join(macro_df, how="left")
        
        # 4. Feature Engineering for Missing/Proxy Data
        # If FRED T10Y2Y failed, but we have US10Y and US2Y from YFinance? (We only added US10Y)
        # Let's just create 'yield_spread' if possible, or leave it.
        # But we must ensure no NaNs at the end
        
        df = df.ffill().fillna(0.0)
        
        # Standardize columns lower case
        df.columns = [c.lower() for c in df.columns]
        
        logger.info(f">>> [Data] Complete. Shape: {df.shape}. Columns: {list(df.columns)}")
        return df

    def _fetch_ohlcv(self, ticker: str) -> pd.DataFrame:
        """
        Fetch OHLCV from yfinance.
        """
        try:
            # [Fix] Explicitly set auto_adjust=True
            data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
            
            if data.empty:
                return pd.DataFrame()

            # Handle MultiIndex columns (YFinance v0.2+)
            if isinstance(data.columns, pd.MultiIndex):
                # If ticker is one, level 1 might be empty or ticker name
                # Collapse to single level
                try:
                    data.columns = data.columns.get_level_values(0) 
                except IndexError:
                    pass
                
            # Rename to standard lower case
            data.columns = [c.lower() for c in data.columns]
            
            # Filter standard columns
            wanted = ["open", "high", "low", "close", "volume"]
            # Check if all exist
            if not all(k in data.columns for k in wanted):
                # Fallback for indices that might not have Volume
                if "volume" not in data.columns:
                    data["volume"] = 0
            
            return data[[c for c in wanted if c in data.columns]]
        except Exception as e:
            logger.warning(f"YF Download Error ({ticker}): {e}")
            return pd.DataFrame()

    def _fetch_fred(self, series_ids: List[str]) -> pd.DataFrame:
        """
        Fetch series from FRED.
        """
        s_dt = datetime.strptime(self.start_date, "%Y-%m-%d")
        # Retry logic or timeout could be added here
        data = web.DataReader(series_ids, "fred", s_dt, datetime.now())
        return data

if __name__ == "__main__":
    # Test Run
    loader = DataLoader("QQQ", "2023-01-01")
    df = loader.fetch_all()
    print(df.tail())
