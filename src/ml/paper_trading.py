"""
Motor de paper trading.

Flujo semanal:
  Viernes nocturno  → paper_trading.py init     (registra señales del modelo)
  Lunes mañana      → paper_trading.py status   (confirma entrada al Open)
  Viernes tarde     → paper_trading.py settle   (cierra semana con datos reales)
  Siempre           → paper_trading.py report   (scorecard acumulado)

El 'settle' descarga OHLC diario de los 5 días del trade y simula la ejecución:
  - Día a día verifica si Low <= stop → STOP_HIT
  - Si High >= T1 → T1_HIT (vende 50%, mueve stop a breakeven)
  - Si High >= T2 → T2_HIT (cierra 100%)
  - Si llega al día 5 sin tocar nada → EXPIRED (cierra al Close del día 5)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT      = Path(__file__).parent.parent.parent
DATA_DIR  = ROOT / "data"
JOURNAL   = DATA_DIR / "paper_trades.json"


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class TradeEvent:
    date:   str
    type:   str          # ENTRY | T1 | T2 | STOP | EXPIRE | SETTLE
    price:  float
    shares: int
    pnl_mxn: float = 0.0
    note:   str    = ""


@dataclass
class Trade:
    trade_id:        str
    ticker:          str
    signal_date:     str          # viernes cuando se generó la señal
    entry_date:      Optional[str] = None
    entry_price:     float = 0.0
    stop:            float = 0.0
    stop_current:    float = 0.0
    target_1:        float = 0.0
    target_2:        float = 0.0
    shares_total:    int   = 0
    shares_open:     int   = 0
    capital_invested: float = 0.0
    pct_capital:     float = 0.0
    p_up:            float = 0.0
    status:          str   = "PENDING"   # PENDING|OPEN|T1_HIT|T2_HIT|STOP_HIT|EXPIRED|CANCELLED
    pnl_mxn:         float = 0.0
    pnl_pct:         float = 0.0
    exit_price:      Optional[float] = None
    exit_date:       Optional[str]   = None
    model_correct:   Optional[bool]  = None  # ¿el precio subió en 5 días?
    events:          list = field(default_factory=list)
    reason:          str  = ""


# ─── Serialización ────────────────────────────────────────────────────────────

def _to_dict(t: Trade) -> dict:
    d = asdict(t)
    d["events"] = [asdict(e) if isinstance(e, TradeEvent) else e for e in t.events]
    return d


def _from_dict(d: dict) -> Trade:
    events = [TradeEvent(**e) if isinstance(e, dict) else e for e in d.pop("events", [])]
    t = Trade(**d)
    t.events = events
    return t


# ─── Journal (persistencia) ───────────────────────────────────────────────────

class Journal:
    def __init__(self, path: Path = JOURNAL):
        self.path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            raw["trades"] = [_from_dict(t) for t in raw.get("trades", [])]
            return raw
        return {
            "trades":       [],
            "equity_curve": [],
            "initial_capital": 100_000.0,
        }

    def save(self) -> None:
        DATA_DIR.mkdir(exist_ok=True)
        out = {
            "trades":          [_to_dict(t) for t in self._data["trades"]],
            "equity_curve":    self._data["equity_curve"],
            "initial_capital": self._data["initial_capital"],
        }
        self.path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    @property
    def trades(self) -> list[Trade]:
        return self._data["trades"]

    @property
    def initial_capital(self) -> float:
        return self._data["initial_capital"]

    @initial_capital.setter
    def initial_capital(self, v: float):
        self._data["initial_capital"] = v

    @property
    def equity_curve(self) -> list[dict]:
        return self._data["equity_curve"]

    def get_open(self) -> list[Trade]:
        return [t for t in self.trades if t.status in ("OPEN","T1_HIT","PENDING")]

    def get_closed(self) -> list[Trade]:
        return [t for t in self.trades if t.status in ("T2_HIT","STOP_HIT","EXPIRED","CANCELLED")]

    def add_trade(self, t: Trade) -> None:
        # Evita duplicados ticker+signal_date
        existing = {(tr.ticker, tr.signal_date) for tr in self.trades}
        if (t.ticker, t.signal_date) not in existing:
            self._data["trades"].append(t)

    def update_equity(self, snapshot_date: str) -> None:
        closed_pnl  = sum(t.pnl_mxn for t in self.get_closed())
        open_pnl    = sum(t.pnl_mxn for t in self.get_open() if t.status != "PENDING")
        cash_invested = sum(t.capital_invested for t in self.get_open() if t.status != "PENDING")
        equity = self.initial_capital + closed_pnl + open_pnl
        self.equity_curve.append({
            "date":     snapshot_date,
            "equity":   round(equity, 2),
            "cash":     round(self.initial_capital + closed_pnl - cash_invested, 2),
            "invested": round(cash_invested, 2),
            "pnl":      round(closed_pnl + open_pnl, 2),
        })


# ─── Carga desde signals_today.json ──────────────────────────────────────────

def load_signals_as_trades(signals_file: Path | None = None) -> list[Trade]:
    if signals_file is None:
        signals_file = DATA_DIR / "signals_today.json"
    data = json.loads(signals_file.read_text())
    signal_date = data["meta"]["date"][:10]
    trades = []
    for s in data.get("signals", []):
        t = Trade(
            trade_id         = str(uuid.uuid4())[:8],
            ticker           = s["ticker"],
            signal_date      = signal_date,
            entry_price      = s["entry"],
            stop             = s["stop"],
            stop_current     = s["stop"],
            target_1         = s["target_1"],
            target_2         = s["target_2"],
            shares_total     = s["shares"],
            shares_open      = s["shares"],
            capital_invested = s["capital_mxn"],
            pct_capital      = s["pct_capital"],
            p_up             = s["p_up"],
            status           = "PENDING",
            reason           = s.get("reason",""),
        )
        trades.append(t)
    return trades


# ─── Settle — simulación con datos OHLC reales ───────────────────────────────

def settle_trade(trade: Trade, ohlc: pd.DataFrame) -> Trade:
    """
    Simula la ejecución de un trade con datos OHLC diarios.

    ohlc: DataFrame con columnas Open,High,Low,Close indexado por fecha.
          Debe contener los 5 días del holding period comenzando en entry_date.
    """
    import yfinance as yf

    t = trade
    if t.status not in ("OPEN", "T1_HIT", "PENDING"):
        return t

    entry  = t.entry_price
    stop   = t.stop_current
    t1     = t.target_1
    t2     = t.target_2
    shares = t.shares_total
    half   = shares // 2

    pnl    = 0.0
    state  = "OPEN"

    for i, (idx, row) in enumerate(ohlc.iterrows()):
        day_str = str(idx.date())
        lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])

        if i == 0 and t.status == "PENDING":
            # Entrada al Open del primer día
            actual_entry = float(row["Open"])
            t.entry_price = actual_entry
            t.entry_date  = day_str
            t.events.append(TradeEvent(day_str,"ENTRY", actual_entry, shares, 0.0,
                                        f"Open del lunes"))
            state  = "OPEN"

        # Evaluar stop primero (prioridad)
        if lo <= stop and state in ("OPEN","T1_HIT"):
            remaining = half if state == "T1_HIT" else shares
            pnl_stop  = (stop - t.entry_price) * remaining
            pnl      += pnl_stop
            t.events.append(TradeEvent(day_str, "STOP", stop, -remaining,
                                        pnl_stop, "Stop activado"))
            state = "STOP_HIT"
            break

        # T1
        if hi >= t1 and state == "OPEN":
            pnl_t1 = (t1 - t.entry_price) * half
            pnl   += pnl_t1
            t.events.append(TradeEvent(day_str, "T1", t1, -half, pnl_t1,
                                        "T1 alcanzado — vende 50%"))
            state = "T1_HIT"
            stop  = t.entry_price  # breakeven

        # T2
        if hi >= t2 and state == "T1_HIT":
            remaining = shares - half
            pnl_t2    = (t2 - t.entry_price) * remaining
            pnl      += pnl_t2
            t.events.append(TradeEvent(day_str, "T2", t2, -remaining, pnl_t2,
                                        "T2 alcanzado — cierra 100%"))
            state = "T2_HIT"
            t.exit_price = t2
            t.exit_date  = day_str
            break

        # Último día: cierre forzado
        if i == len(ohlc) - 1 and state in ("OPEN","T1_HIT"):
            remaining = (shares - half) if state == "T1_HIT" else shares
            pnl_exp   = (cl - t.entry_price) * remaining
            pnl      += pnl_exp
            t.events.append(TradeEvent(day_str,"EXPIRE", cl, -remaining,
                                        pnl_exp, "Expiración día 5"))
            state = "EXPIRED"
            t.exit_price = cl
            t.exit_date  = day_str

    # Actualizar trade
    t.status       = state
    t.pnl_mxn      = round(pnl, 2)
    t.pnl_pct      = round(pnl / t.capital_invested * 100, 3) if t.capital_invested else 0.0
    t.model_correct = t.pnl_mxn > 0

    if state == "STOP_HIT":
        t.exit_price = stop
        t.exit_date  = t.events[-1].date

    return t


# ─── Estadísticas ─────────────────────────────────────────────────────────────

def compute_stats(trades: list[Trade], initial_capital: float) -> dict:
    closed = [t for t in trades if t.status not in ("PENDING","OPEN","T1_HIT","CANCELLED")]
    if not closed:
        return {"n_closed": 0}

    wins     = [t for t in closed if t.pnl_mxn > 0]
    losses   = [t for t in closed if t.pnl_mxn <= 0]
    t2_hits  = [t for t in closed if t.status == "T2_HIT"]
    stops    = [t for t in closed if t.status == "STOP_HIT"]
    expired  = [t for t in closed if t.status == "EXPIRED"]

    total_pnl = sum(t.pnl_mxn for t in closed)
    pnl_rets  = [t.pnl_pct for t in closed]
    avg_win   = np.mean([t.pnl_mxn for t in wins])   if wins   else 0.0
    avg_loss  = np.mean([t.pnl_mxn for t in losses]) if losses else 0.0
    pf        = abs(avg_win * len(wins)) / (abs(avg_loss) * len(losses) + 1e-8) if losses else float("inf")
    sharpe    = (np.mean(pnl_rets) / (np.std(pnl_rets) + 1e-8)) * np.sqrt(52)  # semanal → anual

    # Drawdown en curva de equity
    equity = initial_capital
    peak   = initial_capital
    max_dd = 0.0
    for t in sorted(closed, key=lambda x: x.exit_date or ""):
        equity += t.pnl_mxn
        peak    = max(peak, equity)
        max_dd  = max(max_dd, (peak - equity) / peak * 100)

    return {
        "n_closed":    len(closed),
        "n_wins":      len(wins),
        "n_losses":    len(losses),
        "n_t2_hits":   len(t2_hits),
        "n_stops":     len(stops),
        "n_expired":   len(expired),
        "win_rate":    round(len(wins)/len(closed)*100, 1),
        "total_pnl":   round(total_pnl, 2),
        "total_ret":   round(total_pnl / initial_capital * 100, 3),
        "avg_win":     round(float(avg_win), 2),
        "avg_loss":    round(float(avg_loss), 2),
        "profit_factor": round(float(pf), 3),
        "sharpe":      round(float(sharpe), 3),
        "max_drawdown":round(float(max_dd), 2),
        "equity_final":round(initial_capital + total_pnl, 2),
    }


# ─── Regla congelada y canasta contrafactual (M13; detrás de flag) ───────────

PAPER_LOG_ENABLED = False          # apagado: nada existente lo llama
PAPER_LOG_DIR     = DATA_DIR / "paper_log"


def freeze_rule(cfg: dict) -> str:
    """Hash de la regla (modelo, K, umbrales...) para congelarla antes de la 1ª señal."""
    import hashlib
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def log_counterfactual(signal_date: str, universe: list[str], picks: list[str],
                       rule_hash: str, *, seed: int = 0,
                       enabled: bool | None = None, out_dir: Path | None = None) -> Path | None:
    """Guarda en data/paper_log/<fecha>.json: picks del modelo, equiponderado y aleatorio (semilla)."""
    if not (PAPER_LOG_ENABLED if enabled is None else enabled):
        return None
    rng = np.random.default_rng(seed)
    rand = sorted(rng.choice(sorted(universe), size=len(picks), replace=False).tolist())
    d = out_dir or PAPER_LOG_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{signal_date}.json"
    path.write_text(json.dumps({
        "signal_date": signal_date, "rule_hash": rule_hash, "seed": seed,
        "model": sorted(picks), "equal_weight": sorted(universe), "random": rand,
    }, indent=2))
    return path
