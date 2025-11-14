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
from typing import List


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
        # default persistent duckdb path for sharing between main and detectors
        self.duckdb_path: Path = Path.cwd() / "data" / "ingest.duckdb"
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self._registered: Dict[str, bool] = {}

        # 워킹 테이블 자동 삭제 비활성화 - 시뮬레이션 상태 유지를 위해
        # if self.duckdb_path.exists() and self.duckdb_path.stat().st_size > 0:
        #     try:
        #         con = duckdb.connect(database=str(self.duckdb_path))
        #         all_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        #         working_tables = {t for t in all_tables if not t.startswith('full_data_')}
        #         
        #         if working_tables:
        #             print(f"--- [DataManager] ---")
        #             print(f"🔄 기존 워킹 테이블 삭제 중: {', '.join(working_tables)}")
        #             for table_name in working_tables:
        #                 try:
        #                     con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        #                 except Exception:
        #                     pass # 테이블이 없거나 잠겨있으면 무시
        #             print(f"---------------------\n")
        #         con.close()
        #     except Exception as e:
        #         print(f"--- [DataManager] ---")
        #         print(f"⚠️ 워킹 테이블 삭제 실패: {e}")
        #         print(f"---------------------\n")

        # track last loaded timestamp per sheet (for incremental fetch)
        self.last_loaded: Dict[str, Optional[datetime]] = {}

        # default timestamp columns for known sheets
        self.timestamp_cols: Dict[str, str] = {
            'Trade': 'ts',
            'Funding': 'ts',
            'Reward': 'ts',
            'Spec': 'day'
        }

        # Immediately attempt to load if file exists; otherwise defer until
        # first get_data call (gives flexibility for tests)
        if self.filepath.exists():
            self._load()

    def _load(self):
        if self._load_called:
            return
        
        # --- [수정된 블록] ---
        # 💡 (요청사항 1)
        # Excel을 읽기 전에, 영구 DuckDB에 'full_data_' 테이블이 이미 있는지 확인
        
        print(f"--- [DataManager] ---")
        print(f"💾 영구 DB 확인 중: {self.duckdb_path}")

        db_full_tables = set()
        if self.duckdb_path.exists() and self.duckdb_path.stat().st_size > 0:
            try:
                con = duckdb.connect(database=str(self.duckdb_path), read_only=True)
                db_full_tables = {row[0] for row in con.execute("SHOW TABLES").fetchall() if row[0].startswith('full_data_')}
                con.close()
            except Exception as e:
                print(f"⚠️ 영구 DB 확인 실패: {e}")

        # 1. Excel에서 시트 이름만 먼저 읽기 (빠름)
        try:
            xlsx = pd.ExcelFile(self.filepath, engine='openpyxl')
            sheet_names = xlsx.sheet_names
        except Exception as e:
            print(f"🔥 Excel 파일 읽기 실패 ({self.filepath}): {e}")
            self.sheets = {}
            self._load_called = True
            print(f"---------------------\n")
            return

        # 2. DB에 없는 시트만 Excel에서 로드할 목록 생성
        sheets_to_load_from_excel = []
        self.sheets = {} # 초기화
        
        for name in sheet_names:
            full_name = self._full_table_name(name)
            if full_name in db_full_tables:
                # 💡 DB에 이미 존재함 -> Excel 로드 건너뛰기
                print(f"✅ '{name}' (-> {full_name})은(는) DB에 존재. Excel 로드 건너뜀.")
                # self.sheets에 키가 존재해야 하므로, 빈 DataFrame을 넣어둠
                self.sheets[name] = pd.DataFrame() 
            else:
                # 💡 DB에 없음 -> Excel에서 로드
                print(f"➡️  '{name}' (-> {full_name})을(를) Excel에서 로드합니다.")
                sheets_to_load_from_excel.append(name)

        # 3. 필요한 시트만 병렬로 로드
        if sheets_to_load_from_excel:
            from concurrent.futures import ThreadPoolExecutor

            def _read_sheet(name: str):
                try:
                    return name, pd.read_excel(self.filepath, sheet_name=name, engine='openpyxl')
                except Exception:
                    return name, None

            max_workers = min(8, max(1, len(sheets_to_load_from_excel)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_read_sheet, n) for n in sheets_to_load_from_excel]
                for f in futures:
                    name, df = f.result()
                    if df is not None:
                        self.sheets[name] = df # 로드된 데이터로 채우기
        self._load_called = True

    # ----------------------------- New helpers -----------------------------
    def _normalize_name(self, name: str) -> str:
        """Normalize sheet name for use in persistent table names."""
        return name.strip().lower().replace(' ', '_')

    def _full_table_name(self, name: str) -> str:
        return f"full_data_{self._normalize_name(name)}"

    def _model_table_name(self, name: str) -> str:
        return f"{name}"

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

        # Use persistent duckdb connection for registration
        con = self.get_connection(persistent=True)

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
                # register subset into in-memory registered connection
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
                    existing = con.execute(f"SELECT * FROM \'{name}\'").fetchdf()
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

    def get_connection(self, persistent: bool = True) -> duckdb.DuckDBPyConnection:
        """Return a DuckDB connection.

        By default returns an in-memory connection (used by older callers).
        If `persistent=True` the connection is created against a file at
        `self.duckdb_path` so other processes/modules can reopen the same
        DB file and read tables.
        """
        if persistent:
            if self._con is None:
                self._con = duckdb.connect(database=str(self.duckdb_path))
            return self._con

        # in-memory connection (backwards-compatible)
        if self._con is None:
            self._con = duckdb.connect(database=':memory:')
            # register DataFrames as tables
            if not self._load_called:
                self._load()
            for name, df in self.sheets.items():
                try:
                    self._con.register(name, df)
                except Exception:
                    pass
        return self._con

    # ---------------------- Persistent full/model workflow -----------------
    def seed_full_and_model(self, year: int = 2025, month: int = 2, sheets: Optional[List[str]] = None):
        """Persist full sheet tables and create model-use tables seeded to a given month.

        - Writes each loaded sheet into a persistent table named
          `full_data_<sheetname>` in the DuckDB file.
        - Creates/Replaces a working table with the original sheet name
          that contains only rows for the requested year/month (model data).
        """
        if not self._load_called:
            self._load()

        con = self.get_connection(persistent=True)
        sheet_list = sheets if sheets is not None else list(self.sheets.keys())

        start = datetime(2025, 1, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)

        try:
            con.execute('CREATE OR REPLACE TABLE "simulaterTime" AS SELECT TIMESTAMP \'' + end.isoformat() + '\' AS current_time;')
        except Exception:
            pass

        for name in sheet_list:
            df = self.sheets.get(name)
            if df is None:
                continue

            full_name = self._full_table_name(name)
            if not df.empty:
                print(f"🛠  '{full_name}' 테이블 생성/교체 중...")
                try:
                    con.register('tmp_df', df)
                    con.execute(f'CREATE OR REPLACE TABLE "{full_name}" AS SELECT * FROM tmp_df')
                    try:
                        con.unregister('tmp_df')
                    except Exception:
                        pass
                except Exception as e:
                    print(f"🔥 '{full_name}' 테이블 생성 실패: {e}")
                    continue
            else:
                # df가 비어있다면 (즉, _load가 Excel 로드를 건너뛰었다면)
                # 이미 존재하는 'full_data_' 테이블을 덮어쓰지 않고 넘어감
                print(f"✅ '{full_name}' 테이블이 이미 존재하므로 생성 건너뜀.")

            ts_col = self.timestamp_cols.get(name)
            # 'and ts_col in df.columns' 체크를 제거합니다.
            # ts_col이 설정(config)에 존재하기만 하면 필터링을 시도합니다.
            if ts_col:
                print(f"... '{name}' 워킹 테이블 생성 중 ({year}-{month} 데이터)...")
                # ensure type in DB by casting in query
                start_s = start.isoformat()
                end_s = end.isoformat()
                try:
                    con.execute(
                        f'CREATE OR REPLACE TABLE "{name}" AS '
                        f'SELECT * FROM "{full_name}" WHERE CAST({ts_col} AS TIMESTAMP) >= TIMESTAMP \'{start_s}\' '
                        f'AND CAST({ts_col} AS TIMESTAMP) < TIMESTAMP \'{end_s}\''
                    )
                    
                    # --- [수정된 부분 2] ---
                    # df가 비어있을 수 있으므로, df를 참조하는 대신 DB('full_name')에서 직접 max 값을 가져옵니다.
                    last_row = con.execute(f'SELECT MAX(CAST({ts_col} AS TIMESTAMP)) FROM "{full_name}"').fetchone()
                    last = last_row[0] if last_row else None
                    
                    if last is not None: # pd.notna(last) 대신 last is not None 사용
                    # --- [여기까지] ---
                        
                        # set last_loaded to last timestamp within the model table if exists
                        res = con.execute(f'SELECT MAX(CAST({ts_col} AS TIMESTAMP)) AS m FROM "{name}"').fetchone()
                        self.last_loaded[name] = res[0] if res else None
                        self._registered[name] = True
                
                except Exception as e:
                    # (디버깅 print 구문은 그대로 유지)
                    print(f"--- [DEBUG] ---")
                    print(f"⚠️  Sheet '{name}'의 타임스탬프 필터링 실패. 전체 데이터를 로드합니다.")
                    print(f"🕒  Timestamp Column: {ts_col}")
                    print(f"🔥  Error: {e}")
                    print(f"-----------------\n")

                    try:
                        con.execute(f'UPDATE "simulaterTime" SET current_time = TIMESTAMP \'2025-12-20T23:59:59\'')
                    except Exception:
                        pass
                    
                    # if filtering fails, fallback to copying full
                    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "{full_name}"')
                    self.last_loaded[name] = None
                    self._registered[name] = True

            else:
                # no timestamp - copy full into working table
                try:
                    con.execute(f'CREATE OR REPLACE TABLE "{name}" AS SELECT * FROM "{full_name}"')
                    self.last_loaded[name] = None
                    self.registered[name] = True
                except Exception:
                    pass

    def advance_model_by_days(self, days: int = 7, hours: int = 0, sheets: Optional[List[str]] = None):
        """Advance model tables by appending rows from full tables up to N days.

        For each sheet with a timestamp column, finds the current max timestamp
        in the working table and appends rows from the corresponding full table
        where prev < ts <= prev + days.
        """
        if not self._load_called:
            self._load()
        con = self.get_connection(persistent=True)
        sheet_list = sheets if sheets is not None else list(self.sheets.keys())

        last_time = con.execute('SELECT current_time FROM "simulaterTime"').fetchone()[0]
        new_time = pd.to_datetime(last_time) + timedelta(days=days, hours=hours)
        new_time_s = new_time.isoformat()
        try:
            con.execute(f'UPDATE "simulaterTime" SET current_time = TIMESTAMP \'{new_time_s}\'')
        except Exception:
            pass

        for name in sheet_list:
            full_name = self._full_table_name(name)
            ts_col = self.timestamp_cols.get(name)
            if ts_col is None:
                continue

            # insert rows from full table into working table
            try:
                # create working table if missing
                con.execute(f'CREATE TABLE IF NOT EXISTS "{name}" AS SELECT * FROM "{full_name}" WHERE 1=0')
                con.execute(
                    f'INSERT INTO "{name}" '
                    f'SELECT * FROM "{full_name}" WHERE CAST({ts_col} AS TIMESTAMP) > TIMESTAMP \'{last_time}\' '
                    f'AND CAST({ts_col} AS TIMESTAMP) <= TIMESTAMP \'{new_time_s}\''
                )
                # update last_loaded
                res = con.execute(f'SELECT MAX(CAST({ts_col} AS TIMESTAMP)) FROM "{name}"').fetchone()
                self.last_loaded[name] = res[0] if res else None
                self._registered[name] = True
            except Exception:
                # best-effort: continue
                continue
    
    def close_connection(self):
        """Close the persistent connection if it exists."""
        if self._con is not None:
            try:
                self._con.close()
                print("--- [DataManager] ---")
                print("Database connection closed.")
                print("---------------------\n")
            except Exception as e:
                print(f"Error closing connection: {e}")
            self._con = None

    def reopen_connection(self) -> duckdb.DuckDBPyConnection:
        """Re-establish the persistent connection."""
        if self._con is None:
            try:
                self._con = duckdb.connect(database=str(self.duckdb_path))
                print("--- [DataManager] ---")
                print("Database connection reopened.")
                print("---------------------\n")
            except Exception as e:
                print(f"Error reopening connection: {e}")
                raise e
        return self.get_connection(persistent=True) # 기존 get_connection 로직 재사용


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
