"""
Story 7.4: SQLite Database Layer for Historical Data.

Optional SQLite backend for storing forecasts, backtest results, and errors.
WAL mode for concurrent reads, proper indexing for query performance.

Usage:
    from models.quant_db import QuantDB
    db = QuantDB("src/data/quant.db")
    db.insert_forecast("SPY", "2026-03-01", 7, 1.5, 0.65)
    results = db.query_forecasts("SPY", horizon=7)
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple


class QuantDB:
    """SQLite database for quantitative data storage."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema with WAL mode."""
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Forecasts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    horizon INTEGER NOT NULL,
                    forecast_pct REAL NOT NULL,
                    confidence REAL DEFAULT 0,
                    regime TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(symbol, date, horizon)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_date "
                "ON forecasts(symbol, date)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_forecasts_date_horizon "
                "ON forecasts(date, horizon)"
            )
            
            # Backtest results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    symbol TEXT DEFAULT '',
                    sharpe REAL DEFAULT 0,
                    hit_rate REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    sortino REAL DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_backtest_run "
                "ON backtest_results(run_id)"
            )
            
            # Errors table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity INTEGER DEFAULT 0,
                    source TEXT DEFAULT '',
                    asset TEXT DEFAULT '',
                    message TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_errors_timestamp "
                "ON errors(timestamp)"
            )
            
            conn.commit()
    
    @contextmanager
    def _connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # --- Forecasts ---
    
    def insert_forecast(
        self,
        symbol: str,
        date: str,
        horizon: int,
        forecast_pct: float,
        confidence: float = 0.0,
        regime: str = "",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Insert or replace a forecast."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO forecasts "
                "(symbol, date, horizon, forecast_pct, confidence, regime, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, date, horizon, forecast_pct, confidence, regime,
                 json.dumps(metadata or {})),
            )
            conn.commit()
    
    def insert_forecasts_batch(self, rows: List[Tuple]) -> int:
        """Batch insert forecasts. Each tuple: (symbol, date, horizon, forecast_pct, confidence)."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO forecasts "
                "(symbol, date, horizon, forecast_pct, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            return len(rows)
    
    def query_forecasts(
        self,
        symbol: Optional[str] = None,
        horizon: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict]:
        """Query forecasts with optional filters."""
        clauses = []
        params = []
        
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if horizon is not None:
            clauses.append("horizon = ?")
            params.append(horizon)
        if date_from:
            clauses.append("date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("date <= ?")
            params.append(date_to)
        
        where = " AND ".join(clauses) if clauses else "1=1"
        query = f"SELECT * FROM forecasts WHERE {where} ORDER BY date DESC LIMIT ?"
        params.append(limit)
        
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    
    # --- Backtest Results ---
    
    def insert_backtest_result(
        self,
        run_id: str,
        symbol: str = "",
        sharpe: float = 0.0,
        hit_rate: float = 0.0,
        max_drawdown: float = 0.0,
        sortino: float = 0.0,
        total_pnl: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Insert a backtest result."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO backtest_results "
                "(run_id, symbol, sharpe, hit_rate, max_drawdown, sortino, total_pnl, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, symbol, sharpe, hit_rate, max_drawdown, sortino, total_pnl,
                 json.dumps(metadata or {})),
            )
            conn.commit()
    
    def query_backtest_results(self, run_id: Optional[str] = None) -> List[Dict]:
        """Query backtest results."""
        with self._connect() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT * FROM backtest_results WHERE run_id = ? ORDER BY created_at DESC",
                    (run_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 100"
                ).fetchall()
            return [dict(r) for r in rows]
    
    # --- Errors ---
    
    def insert_error(
        self,
        timestamp: str,
        severity: int,
        source: str,
        asset: str,
        message: str,
    ) -> None:
        """Insert an error record."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO errors (timestamp, severity, source, asset, message) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, severity, source, asset, message),
            )
            conn.commit()
    
    def query_errors(
        self,
        severity_min: int = 0,
        limit: int = 100,
    ) -> List[Dict]:
        """Query errors."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM errors WHERE severity >= ? ORDER BY timestamp DESC LIMIT ?",
                (severity_min, limit),
            ).fetchall()
            return [dict(r) for r in rows]
    
    def count_records(self) -> Dict[str, int]:
        """Count records in all tables."""
        with self._connect() as conn:
            result = {}
            for table in ["forecasts", "backtest_results", "errors"]:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
                result[table] = row["cnt"]
            return result
