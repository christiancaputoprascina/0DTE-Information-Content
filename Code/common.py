"""
common.py: percorsi, calendario del 2025 e funzioni usate da piu' script.

Ogni script Code/NN_Nome.py scrive in CBOE/Data/NN_nome/ e legge gli output degli
script precedenti e CBOE/Data/00_inputs/. Un'altra cartella dei dati si indica con
la variabile d'ambiente ODTE_DATA_DIR.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# RIPRODUCIBILITA'
# =====================================================================
# Le ultime cifre delle stime pyfixest dipendono dal seme di hash di Python: ogni script
# si rilancia con PYTHONHASHSEED=0, cosi' due esecuzioni danno output identici.
def _fix_hash_seed():
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file and os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, main_file, *sys.argv[1:]])


_fix_hash_seed()

# =====================================================================
# PERCORSI
# =====================================================================
CODE_DIR = Path(__file__).absolute().parent   # .../Code
ROOT = CODE_DIR.parent                          # cartella del progetto, con Code/ e CBOE/
DATA_DIR = Path(os.environ["ODTE_DATA_DIR"]) if os.environ.get("ODTE_DATA_DIR") else ROOT / "CBOE" / "Data"
INPUT_DIR = DATA_DIR / "00_inputs"


def output_dir(name: str) -> Path:
    """Cartella di output CBOE/Data/<name>/, creata se manca."""
    folder = DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def path(pair) -> Path:
    """Percorso di un file dato come (cartella, nome) dentro CBOE/Data."""
    return DATA_DIR / pair[0] / pair[1]


def require(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"File richiesto non trovato:\n{p}")


# =====================================================================
# CALENDARIO DEL 2025
# =====================================================================
YEAR = 2025

# Chiusura del mercato (ora di New York). Nel 2025 il NYSE chiude alle 13:00
# in tre sessioni; le opzioni SPXW in scadenza quel giorno scadono alle 13:00.
REGULAR_CLOSE = "16:00:00"
EARLY_CLOSE = "13:00:00"
EARLY_CLOSE_DATES = ("2025-07-03", "2025-11-28", "2025-12-24")

# Le stesse informazioni nei formati usati dagli script.
EARLY_CLOSE_TIMES = {d: EARLY_CLOSE for d in EARLY_CLOSE_DATES}        # "13:00:00"
EARLY_CLOSE_TIMES_HHMM = {d: EARLY_CLOSE[:5] for d in EARLY_CLOSE_DATES}  # "13:00"
EARLY_CLOSE_SECONDS = {d: 13 * 3600 for d in EARLY_CLOSE_DATES}
REGULAR_CLOSE_SECONDS = 16 * 3600


def close_time(quote_date) -> str:
    """Ora di chiusura della sessione, "HH:MM:SS"."""
    return EARLY_CLOSE_TIMES.get(str(quote_date)[:10], REGULAR_CLOSE)


def close_seconds(quote_date) -> int:
    """Ora di chiusura della sessione in secondi dalla mezzanotte."""
    return EARLY_CLOSE_SECONDS.get(str(quote_date)[:10], REGULAR_CLOSE_SECONDS)


# Annunci FOMC e pubblicazioni CPI del 2025.
FOMC_DATES = ("2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
              "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10")
CPI_DATES = ("2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10",
             "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
             "2025-09-11", "2025-10-24", "2025-12-18")

# Sessione con la varianza realizzata piu' alta del 2025 (episodio dei dazi).
STRESS_DAY = "2025-04-07"
N_APRIL_SESSIONS = 21


# =====================================================================
# FUNZIONI DI SUPPORTO
# =====================================================================
def clean_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype(str).str[:10], errors="coerce"
    ).dt.strftime("%Y-%m-%d")


def to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def to_numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    return to_numeric_series(df[col])


def safe_write_csv(df: pd.DataFrame, path: Path, overwrite: bool = True):
    if path.exists() and overwrite:
        path.unlink()

    df.to_csv(path, index=False)


def nw_lag_length(T: int) -> int:
    """Lag di Newey-West: floor(4 (T/100)^(2/9))."""
    return int(np.floor(4.0 * ((float(T) / 100.0) ** (2.0 / 9.0))))


def make_log_column(df: pd.DataFrame, source_col: str, log_col: str):
    x = pd.to_numeric(df[source_col], errors="coerce")
    df[log_col] = np.where(x > 0, np.log(x), np.nan)
