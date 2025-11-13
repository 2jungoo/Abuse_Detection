"""common.data_manager

Singleton data manager for loading dataset(s) once and reusing them across
detectors and dashboard code.

Features
- Loads an Excel file (default: problem_data_final.xlsx in repo root) once
  into a dict of pandas.DataFrame objects (one per sheet).
- Exposes a DuckDB connection with each DataFrame registered as a table so
  existing SQL queries (used in detectors) can run against the in-memory
  tables.
- Thread-safe-ish single-instance via simple singleton pattern.

Usage
-----
from common.data_manager import get_data_manager
dm = get_data_manager('problem_data_final.xlsx')  # first call loads file
trade_df = dm.get_sheet('Trade')
con = dm.get_connection()
con.execute('SELECT COUNT(*) FROM Trade').fetchall()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import duckdb
import threading
from datetime import datetime, timedelta


class DataManager:
    """Singleton manager that loads data once and provides accessors.

    It loads all sheets from the provided Excel file into memory (pandas
    DataFrames) and registers them as DuckDB tables for SQL-based detectors.
    """

    _instance: Optional["DataManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        # thread-safe singleton
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, filepath: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self.filepath = Path(filepath) if filepath else Path.cwd() / "problem_data_final.xlsx"
        self.sheets: Dict[str, pd.DataFrame] = {}
        self._load_called = False

        # DuckDB connection (in-memory) - created lazily when requested
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._registered: Dict[str, bool] = {}

        # track last loaded timestamp per sheet (for incremental fetch)
        self.last_loaded: Dict[str, Optional[datetime]] = {}

        # default timestamp columns for known sheets
        self.timestamp_cols: Dict[str, str] = {
            'Trade': 'ts',
            'Funding': 'ts',
            'Reward': 'ts',
            'IP': 'ts',
            'Spec': 'day'
        }

        # Immediately attempt to load if file exists; otherwise defer until
        # first get_data call (gives flexibility for tests)
        if self.filepath.exists():
            self._load()

    def _load(self):
        if self._load_called:
            return
        # read all sheets. Use ExcelFile to get sheet names and load sheets
        # in parallel to speed up large files.
        try:
            xlsx = pd.ExcelFile(self.filepath, engine='openpyxl')
            sheet_names = xlsx.sheet_names

            from concurrent.futures import ThreadPoolExecutor

            def _read_sheet(name: str):
                try:
                    return name, pd.read_excel(self.filepath, sheet_name=name, engine='openpyxl')
                except Exception:
                    return name, None

            results = {}
            max_workers = min(8, max(1, len(sheet_names)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_read_sheet, n) for n in sheet_names]
                for f in futures:
                    name, df = f.result()
                    if df is not None:
                        results[name] = df

            self.sheets = results
        except Exception:
            # Be forgiving: if file not found or cannot be read, leave empty
            # and let callers handle missing data. Still set flag so we don't
            # repeatedly attempt a failing read.
            self.sheets = {}
        self._load_called = True

    def ensure_loaded(self, filepath: Optional[str] = None):
        if filepath:
            self.filepath = Path(filepath)
        if not self._load_called:
            self._load()

    def initial_register(self, initial_days: int = 7, sheets: Optional[list] = None, timestamp_columns: Optional[Dict[str, str]] = None):
        """Register tables into DuckDB with an initial time window.

        For sheets that have a timestamp column (from timestamp_columns or
        defaults), only rows within the last `initial_days` (relative to the
        sheet's max timestamp) are registered. Other sheets are fully
        registered.
        """
        if timestamp_columns:
            self.timestamp_cols.update(timestamp_columns)

        if not self._load_called:
            self._load()

        con = self.get_connection()

        sheet_list = sheets if sheets is not None else list(self.sheets.keys())
        for name in sheet_list:
            df = self.sheets.get(name)
            if df is None:
                continue

            ts_col = self.timestamp_cols.get(name)
            if ts_col and ts_col in df.columns:
                # ensure datetime
                df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
                if df[ts_col].dropna().empty:
                    # no valid timestamps - register full
                    con.register(name, df)
                    self.last_loaded[name] = None
                    self._registered[name] = True
                    continue

                max_ts = df[ts_col].max()
                cutoff = max_ts - timedelta(days=initial_days)
                subset = df[df[ts_col] >= cutoff].copy()
                con.register(name, subset)
                self.last_loaded[name] = subset[ts_col].max() if not subset.empty else None
                self._registered[name] = True
            else:
                # sheet without timestamp: register full
                con.register(name, df)
                self.last_loaded[name] = None
                self._registered[name] = True

    def append_until(self, fetch_until: datetime, sheets: Optional[list] = None):
        """Append data from loaded DataFrames into DuckDB up to fetch_until.

        For each sheet with a known timestamp column, takes rows where
        last_loaded < ts <= fetch_until and appends them to the registered
        DuckDB table. Updates last_loaded accordingly.
        """
        if not self._load_called:
            self._load()
        if self._con is None:
            self.get_connection()

        con = self._con
        sheet_list = sheets if sheets is not None else list(self.sheets.keys())
        for name in sheet_list:
            df = self.sheets.get(name)
            if df is None:
                continue
            ts_col = self.timestamp_cols.get(name)
            if not ts_col or ts_col not in df.columns:
                continue

            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            prev = self.last_loaded.get(name)
            if prev is None:
                # nothing registered before: treat as initial append from -inf
                mask = (df[ts_col] <= fetch_until)
            else:
                mask = (df[ts_col] > prev) & (df[ts_col] <= fetch_until)

            add_df = df.loc[mask].copy()
            if add_df.empty:
                continue

            # if table already registered, fetch existing and concat
            if self._registered.get(name):
                try:
                    existing = con.execute(f"SELECT * FROM \"{name}\"").fetchdf()
                except Exception:
                    existing = pd.DataFrame()
                combined = pd.concat([existing, add_df], ignore_index=True)
                con.register(name, combined)
            else:
                con.register(name, add_df)
                self._registered[name] = True

            # update last_loaded
            max_ts = add_df[ts_col].max()
            self.last_loaded[name] = max_ts

    def get_sheet(self, name: str) -> Optional[pd.DataFrame]:
        """Return a pandas DataFrame for sheet `name` or None if missing."""
        if not self._load_called:
            self._load()
        return self.sheets.get(name)

    def get_all_sheets(self) -> Dict[str, pd.DataFrame]:
        if not self._load_called:
            self._load()
        return self.sheets

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Return a DuckDB connection with loaded DataFrames registered.

        The connection is created and tables are registered on first call.
        Subsequent calls return the same connection (tables already
        registered).
        """
        if self._con is None:
            self._con = duckdb.connect(database=':memory:')
            # register DataFrames as tables
            if not self._load_called:
                self._load()
            for name, df in self.sheets.items():
                try:
                    # duckdb requires str table names without spaces; keep as-is
                    self._con.register(name, df)
                except Exception:
                    # ignore registration errors for now
                    pass
        return self._con


_GLOBAL_MANAGER: Optional[DataManager] = None


def get_data_manager(filepath: Optional[str] = None) -> DataManager:
    """Convenience accessor for the singleton DataManager.

    If a filepath is provided on the first call, it will be used to load the
    data. Subsequent calls ignore filepath.
    """
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = DataManager(filepath)
    else:
        # if caller provided a filepath and manager hasn't loaded any data,
        # ensure we load that path
        if filepath and not _GLOBAL_MANAGER._load_called:
            _GLOBAL_MANAGER.ensure_loaded(filepath)
    return _GLOBAL_MANAGER
