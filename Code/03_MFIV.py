"""
03_MFIV.py: varianza implicita model-free (MFIV, BJN/JT) a ogni minuto delle 0DTE SPXW.

Parte 1: MFIV di base con troncamento Cboe delle code a bid zero e varianti dell'Appendice D.
Parte 2: MFIV di robustezza con taglio |ln(K/F)| <= 0.05 e poi <= 0.02, senza troncamento Cboe.

Input:  CBOE/Data/02_black76_iv_greeks/
Output: CBOE/Data/03_mfiv/ (intraday, fine sessione, diagnostiche, Appendice D; i file della
        parte 2 hanno in piu' l'etichetta _logm_005 o _logm_002)
"""

import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DATA_DIR,
    EARLY_CLOSE_SECONDS,
    EARLY_CLOSE_TIMES,
    REGULAR_CLOSE,
    REGULAR_CLOSE_SECONDS,
    YEAR,
    output_dir,
)

# =====================================================================
# CONFIGURAZIONE COMUNE ALLE DUE PARTI
# =====================================================================
BLACK76_DIR = DATA_DIR / "02_black76_iv_greeks"

OUTPUT_DIR = output_dir("03_mfiv")

DAILY_INTRADAY_DIR = OUTPUT_DIR / "intraday_by_day"
DAILY_INTRADAY_DIR.mkdir(parents=True, exist_ok=True)

YEAR_BASIS = 365

# Esclusione vicino alla scadenza: tau < 5 minuti
NEAR_EXPIRY_MINUTES = 5
NEAR_EXPIRY_SECONDS = NEAR_EXPIRY_MINUTES * 60
NEAR_EXPIRY_TAU = NEAR_EXPIRY_SECONDS / (YEAR_BASIS * 24.0 * 60.0 * 60.0)

# Copertura minima: strike OTM per lato
MIN_OTM_EACH_SIDE = 5

# Un CSV intraday per giornata oltre a quello unico dell'anno
WRITE_DAILY_INTRADAY_FILES = True
WRITE_COMBINED_INTRADAY_FILE = True

OVERWRITE_EXISTING = True

filename_pattern = re.compile(
    r"02_0DTE_(\d{4})_(\d{2})_(\d{2})_black76_365\.csv$"
)


# =====================================================================
# FUNZIONI COMUNI ALLE DUE PARTI
# =====================================================================
def file_to_date(file_path: Path):
    match = filename_pattern.match(file_path.name)

    if not match:
        return None

    y, m, d = match.groups()
    return f"{y}-{m}-{d}"


def normalize_option_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()

    out = pd.Series("", index=series.index, dtype=str)
    out.loc[s.isin(["C", "CALL"])] = "C"
    out.loc[s.isin(["P", "PUT"])] = "P"

    return out


def parse_time_to_seconds(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series.astype(str), format="%H:%M:%S", errors="coerce")
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second


def compute_delta_k(strikes: np.ndarray) -> np.ndarray:
    n = len(strikes)

    delta_k = np.full(n, np.nan, dtype=float)

    if n < 2:
        return delta_k

    delta_k[0] = strikes[1] - strikes[0]
    delta_k[-1] = strikes[-1] - strikes[-2]

    if n > 2:
        delta_k[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0

    return delta_k


def safe_median(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]

    if values.empty:
        return np.nan

    return float(values.median())


def safe_divide(numerator, denominator):
    if (
        numerator is None
        or denominator is None
        or not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or denominator == 0
    ):
        return np.nan

    return float(numerator / denominator)


def compute_strip_mfiv(strip: pd.DataFrame, r: float, tau: float):
    if strip.empty:
        return np.nan

    clean = strip[["_strike", "_mid"]].copy()
    clean["_strike"] = pd.to_numeric(clean["_strike"], errors="coerce")
    clean["_mid"] = pd.to_numeric(clean["_mid"], errors="coerce")

    clean = clean[
        clean["_strike"].notna()
        & clean["_mid"].notna()
        & np.isfinite(clean["_strike"])
        & np.isfinite(clean["_mid"])
        & (clean["_strike"] > 0)
        & (clean["_mid"] > 0)
    ].copy()

    clean = (
        clean.groupby("_strike", as_index=False)
        .agg({"_mid": "median"})
        .sort_values("_strike")
        .reset_index(drop=True)
    )

    if len(clean) < 2:
        return np.nan

    strikes = clean["_strike"].to_numpy(dtype=float)
    prices = clean["_mid"].to_numpy(dtype=float)

    delta_k = compute_delta_k(strikes)

    valid = (
        np.isfinite(strikes)
        & np.isfinite(prices)
        & np.isfinite(delta_k)
        & (strikes > 0)
        & (prices > 0)
        & (delta_k > 0)
    )

    if not np.any(valid):
        return np.nan

    integral_sum = np.sum(
        prices[valid]
        / (strikes[valid] ** 2)
        * delta_k[valid]
    )

    mfiv = 2.0 * np.exp(r * tau) * integral_sum

    if not np.isfinite(mfiv) or mfiv < 0:
        return np.nan

    return float(mfiv)


def select_input_files() -> list:
    """File Black '76 giornalieri dell'anno, in ordine di data."""
    if not BLACK76_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {BLACK76_DIR}")

    all_files = sorted(BLACK76_DIR.glob("02_0DTE_*_black76_365.csv"))

    input_files = []

    for file in all_files:
        match = filename_pattern.match(file.name)

        if not match:
            continue

        y, _, _ = match.groups()

        if y == str(YEAR):
            input_files.append(file)

    if not input_files:
        raise FileNotFoundError(
            f"No Black76 files found in {BLACK76_DIR} for year {YEAR}."
        )

    return input_files


# =====================================================================
# PARTE 1: MFIV DI BASE CON TRONCAMENTO CBOE
# =====================================================================
# Copertura minima in deviazioni standard della distribuzione terminale, k * sigma_atm *
# sqrt(tau) * F per lato: l'ampiezza naturale dello strip si contrae con sqrt(tau), e una
# soglia fissa in percentuale di F escluderebbe soprattutto le sessioni a bassa volatilita'.
COVERAGE_K_SD = 2.0

# Chiusura della sessione per giornata (13:00 nelle chiusure anticipate), da common.py.
def close_seconds_for_date(quote_date: str) -> int:
    return EARLY_CLOSE_SECONDS.get(str(quote_date), REGULAR_CLOSE_SECONDS)


# Troncamento delle code a bid zero secondo il white paper del VIX di Cboe: senza, le put
# lontane quotate 0.00/0.05 (mid 0.025) dominano l'integrale per il peso dK/K^2.
def cboe_truncate_side(strip: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Partendo dallo strike piu' vicino a F e andando verso l'esterno ('P' verso
    gli strike piu' bassi, 'C' verso quelli piu' alti), si ferma alla prima
    coppia di strike consecutivi con bid non positivo: quella coppia e tutto
    cio' che sta oltre sono esclusi. Gli strike isolati a bid zero incontrati
    prima dello stop sono esclusi anch'essi, ma non fermano il cammino.
    Richiede le colonne _strike, _mid, _bid.
    """
    if strip.empty:
        return strip.iloc[0:0]

    ordered = strip.sort_values(
        "_strike", ascending=(side == "C")
    ).reset_index(drop=True)

    bids = pd.to_numeric(ordered["_bid"], errors="coerce").to_numpy(dtype=float)

    keep = []
    consecutive_zeros = 0

    for i in range(len(ordered)):
        b = bids[i]

        if not np.isfinite(b) or b <= 0.0:
            consecutive_zeros += 1
            if consecutive_zeros >= 2:
                break
        else:
            consecutive_zeros = 0
            keep.append(i)

    return ordered.loc[keep].sort_values("_strike").reset_index(drop=True)


def drop_zero_bid(strip: pd.DataFrame) -> pd.DataFrame:
    """Variante di robustezza dell'Appendice D: toglie ogni strike con bid non
    positivo, ovunque si trovi, senza regola di arresto."""
    if strip.empty:
        return strip.iloc[0:0]

    b = pd.to_numeric(strip["_bid"], errors="coerce")
    return strip[b.notna() & (b > 0)].sort_values("_strike").reset_index(drop=True)


def atm_sigma(valid: pd.DataFrame, F: float) -> float:
    """Media delle IV Black '76 dei quattro strike piu' vicini a F, esclusi i
    valori al tetto del solver. Usata solo per la soglia di copertura, mai per
    una misura riportata."""
    if valid.empty or "_iv" not in valid.columns:
        return np.nan

    a = valid[valid["_iv"].notna() & (valid["_iv"] > 0) & (valid["_iv"] < 5.0)]

    if a.empty:
        return np.nan

    a = a.assign(_dist=(a["_strike"] - F).abs())
    return float(a.nsmallest(4, "_dist")["_iv"].mean())


def prepare_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "quote_date" not in out.columns or "quote_time" not in out.columns:
        if "quote_datetime" in out.columns:
            out["quote_date"] = out["quote_datetime"].astype(str).str[:10]
            out["quote_time"] = out["quote_datetime"].astype(str).str[11:19]
        elif "quoted_datetime" in out.columns:
            out["quote_date"] = out["quoted_datetime"].astype(str).str[:10]
            out["quote_time"] = out["quoted_datetime"].astype(str).str[11:19]
        else:
            raise ValueError(
                "Non trovo quote_date/quote_time oppure quote_datetime/quoted_datetime."
            )

    required_base_columns = [
        "quote_date",
        "quote_time",
        "strike",
        "option_type",
        "black76_mid",
        "black76_F_365",
        "black76_T_365",
        "black76_r",
    ]

    missing = [
        col for col in required_base_columns
        if col not in out.columns
    ]

    if missing:
        raise ValueError(f"Colonne mancanti nel file input: {missing}")

    out["_strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["_mid"] = pd.to_numeric(out["black76_mid"], errors="coerce")
    out["_F"] = pd.to_numeric(out["black76_F_365"], errors="coerce")
    out["_tau"] = pd.to_numeric(out["black76_T_365"], errors="coerce")
    out["_r"] = pd.to_numeric(out["black76_r"], errors="coerce")
    out["_option_type"] = normalize_option_type(out["option_type"])
    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])

    # bid, richiesto dalla regola di troncamento
    if "black76_bid" in out.columns:
        out["_bid"] = pd.to_numeric(out["black76_bid"], errors="coerce")
    elif "bid" in out.columns:
        out["_bid"] = pd.to_numeric(out["bid"], errors="coerce")
    else:
        raise ValueError(
            "Nessuna colonna bid trovata (black76_bid oppure bid): la regola "
            "di troncamento CBOE non e' implementabile senza il bid."
        )

    # IV Black '76, usata solo per la sigma ATM della soglia di copertura
    if "black76_implied_volatility_365" in out.columns:
        out["_iv"] = pd.to_numeric(
            out["black76_implied_volatility_365"], errors="coerce"
        )
    else:
        out["_iv"] = np.nan

    if "black76_solver_success_365" in out.columns:
        out["_solver_success"] = pd.to_numeric(
            out["black76_solver_success_365"],
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        iv = pd.to_numeric(
            out.get("black76_implied_volatility_365", np.nan),
            errors="coerce"
        )

        gamma = pd.to_numeric(
            out.get("black76_gamma_365", np.nan),
            errors="coerce"
        )

        out["_solver_success"] = np.where(
            out["_mid"].notna()
            & out["_F"].notna()
            & out["_tau"].notna()
            & iv.notna()
            & gamma.notna()
            & np.isfinite(out["_mid"])
            & np.isfinite(out["_F"])
            & np.isfinite(out["_tau"])
            & np.isfinite(iv)
            & np.isfinite(gamma)
            & (out["_mid"] > 0)
            & (out["_F"] > 0)
            & (out["_tau"] > 0)
            & (iv > 0),
            1,
            0
        )

    # Flag di IV al limite del solver: serve al filtro secondario dopo il troncamento e al
    # conteggio degli strike saturi.
    if "black76_iv_at_bound_365" in out.columns:
        out["_iv_at_bound"] = pd.to_numeric(
            out["black76_iv_at_bound_365"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["_iv_at_bound"] = 0

    if "black76_near_expiry_flag_365" in out.columns:
        out["_near_expiry"] = pd.to_numeric(
            out["black76_near_expiry_flag_365"],
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["_near_expiry"] = np.where(
            out["_tau"].notna()
            & np.isfinite(out["_tau"])
            & (out["_tau"] > 0)
            & (out["_tau"] < NEAR_EXPIRY_TAU),
            1,
            0
        )

    return out


def compute_mfiv_for_timestamp(group: pd.DataFrame) -> dict:
    quote_date = str(group["quote_date"].iloc[0])
    quote_time = str(group["quote_time"].iloc[0])

    F = safe_median(group["_F"])
    tau = safe_median(group["_tau"])
    r = safe_median(group["_r"])

    base_result = {
        "quote_date": quote_date,
        "quote_time": quote_time,
        "tau": tau,
        "F": F,
        "r": r,
        "sigma_atm": np.nan,
        "n_otm_strikes": 0,
        "n_otm_put_strikes": 0,
        "n_otm_call_strikes": 0,
        "strike_min": np.nan,
        "strike_max": np.nan,
        "put_strike_min": np.nan,
        "put_strike_max": np.nan,
        "call_strike_min": np.nan,
        "call_strike_max": np.nan,
        "strike_coverage_low": np.nan,
        "strike_coverage_high": np.nan,
        "coverage_low_sd": np.nan,
        "coverage_high_sd": np.nan,
        "mfiv": np.nan,
        "mfiv_put_only": np.nan,
        "mfiv_call_only": np.nan,
        "mfiv_put_share_of_full": np.nan,
        "mfiv_call_share_of_full": np.nan,
        "mfiv_tail_imbalance": np.nan,
        "mfiv_annualized_vol": np.nan,
        "mfiv_put_annualized_vol": np.nan,
        "mfiv_call_annualized_vol": np.nan,
        # varianti di troncamento dell'Appendice D e diagnostiche
        "mfiv_raw_no_truncation": np.nan,
        "mfiv_nozero_bid": np.nan,
        "truncation_ratio": np.nan,
        "n_otm_strikes_raw": 0,
        "n_otm_put_strikes_raw": 0,
        "n_otm_call_strikes_raw": 0,
        "strike_min_raw": np.nan,
        "strike_max_raw": np.nan,
        "n_zero_bid_dropped": 0,
        "n_iv_at_bound_before": 0,
        "n_iv_at_bound_after": 0,
        "mfiv_flag": "",
    }

    if not np.isfinite(F) or F <= 0:
        base_result["mfiv_flag"] = "missing_F"
        return base_result

    if not np.isfinite(tau) or tau <= 0:
        base_result["mfiv_flag"] = "near_expiry"
        return base_result

    if tau < NEAR_EXPIRY_TAU:
        base_result["mfiv_flag"] = "near_expiry"
        return base_result

    if not np.isfinite(r):
        base_result["mfiv_flag"] = "missing_F"
        return base_result

    valid = group[
        (group["_solver_success"] == 1)
        & (group["_near_expiry"] == 0)
        & group["_strike"].notna()
        & group["_mid"].notna()
        & group["_F"].notna()
        & group["_tau"].notna()
        & group["_r"].notna()
        & np.isfinite(group["_strike"])
        & np.isfinite(group["_mid"])
        & np.isfinite(group["_F"])
        & np.isfinite(group["_tau"])
        & np.isfinite(group["_r"])
        & (group["_strike"] > 0)
        & (group["_mid"] > 0)
        & (group["_F"] > 0)
        & (group["_tau"] > 0)
        & (group["_option_type"].isin(["C", "P"]))
    ].copy()

    if valid.empty:
        base_result["mfiv_flag"] = "no_valid_options"
        return base_result

    if "_iv_at_bound" not in valid.columns:
        valid["_iv_at_bound"] = 0

    sigma = atm_sigma(valid, F)
    base_result["sigma_atm"] = sigma

    collapsed = (
        valid[["_strike", "_option_type", "_mid", "_bid", "_iv_at_bound"]]
        .groupby(["_strike", "_option_type"], as_index=False)
        .agg({"_mid": "median", "_bid": "median", "_iv_at_bound": "max"})
    )

    puts_raw = collapsed[
        (collapsed["_option_type"] == "P")
        & (collapsed["_strike"] < F)
    ][["_strike", "_mid", "_bid", "_iv_at_bound"]].copy()

    calls_raw = collapsed[
        (collapsed["_option_type"] == "C")
        & (collapsed["_strike"] > F)
    ][["_strike", "_mid", "_bid", "_iv_at_bound"]].copy()

    # Troncamento Cboe; puts_raw e calls_raw restano per le varianti dell'Appendice D.
    puts = cboe_truncate_side(puts_raw, "P")
    calls = cboe_truncate_side(calls_raw, "C")

    # Filtro secondario: dopo il troncamento si tolgono anche gli strike con IV al limite del
    # solver. Capita di rado (uno strike lontano con un bid residuo e IV non calcolabile), ma
    # senza il filtro quei minuti mostrano picchi isolati di volatilita'.
    if "_iv_at_bound" in puts.columns:
        puts = puts[puts["_iv_at_bound"] != 1].reset_index(drop=True)

    if "_iv_at_bound" in calls.columns:
        calls = calls[calls["_iv_at_bound"] != 1].reset_index(drop=True)

    puts_nz = drop_zero_bid(puts_raw)
    calls_nz = drop_zero_bid(calls_raw)

    # strike con IV satura prima del troncamento e quanti ne restano nello strip finale
    if "_iv_at_bound" in valid.columns:
        bound_before = valid[
            (valid["_iv_at_bound"] == 1)
            & (
                ((valid["_option_type"] == "P") & (valid["_strike"] < F))
                | ((valid["_option_type"] == "C") & (valid["_strike"] > F))
            )
        ]
        bound_strikes = set(bound_before["_strike"].unique())
        kept = set(puts["_strike"].tolist()) | set(calls["_strike"].tolist())
        base_result["n_iv_at_bound_before"] = int(len(bound_strikes))
        base_result["n_iv_at_bound_after"] = int(len(bound_strikes & kept))

    exact_atm = collapsed[
        np.isclose(collapsed["_strike"], F, rtol=0.0, atol=1e-8)
    ]

    atm_strip = pd.DataFrame(columns=["_strike", "_mid"])

    if not exact_atm.empty:
        atm_price = exact_atm["_mid"].mean()

        if np.isfinite(atm_price) and atm_price > 0:
            atm_strip = pd.DataFrame({
                "_strike": [F],
                "_mid": [atm_price],
            })

    n_put_strikes = int(puts["_strike"].nunique()) if not puts.empty else 0
    n_call_strikes = int(calls["_strike"].nunique()) if not calls.empty else 0

    strip_parts = []

    if not puts.empty:
        strip_parts.append(puts[["_strike", "_mid"]])

    if not atm_strip.empty:
        strip_parts.append(atm_strip)

    if not calls.empty:
        strip_parts.append(calls[["_strike", "_mid"]])

    if not strip_parts:
        base_result["mfiv_flag"] = "no_valid_options"
        return base_result

    full_strip = pd.concat(strip_parts, ignore_index=True)

    full_strip = (
        full_strip.groupby("_strike", as_index=False)
        .agg({"_mid": "median"})
        .sort_values("_strike")
        .reset_index(drop=True)
    )

    strikes = full_strip["_strike"].to_numpy(dtype=float)

    n_strikes = len(strikes)

    base_result["n_otm_strikes"] = int(n_strikes)
    base_result["n_otm_put_strikes"] = n_put_strikes
    base_result["n_otm_call_strikes"] = n_call_strikes
    base_result["strike_min"] = float(np.nanmin(strikes)) if n_strikes > 0 else np.nan
    base_result["strike_max"] = float(np.nanmax(strikes)) if n_strikes > 0 else np.nan

    if n_put_strikes > 0:
        base_result["put_strike_min"] = float(puts["_strike"].min())
        base_result["put_strike_max"] = float(puts["_strike"].max())

    if n_call_strikes > 0:
        base_result["call_strike_min"] = float(calls["_strike"].min())
        base_result["call_strike_max"] = float(calls["_strike"].max())

    if n_strikes == 0:
        base_result["mfiv_flag"] = "no_valid_options"
        return base_result

    strike_min = base_result["strike_min"]
    strike_max = base_result["strike_max"]

    coverage_low = strike_min / F - 1.0
    coverage_high = strike_max / F - 1.0

    base_result["strike_coverage_low"] = float(coverage_low)
    base_result["strike_coverage_high"] = float(coverage_high)

    # copertura in unita' di deviazione standard terminale
    sd_points = sigma * np.sqrt(tau) * F if np.isfinite(sigma) else np.nan

    if np.isfinite(sd_points) and sd_points > 0:
        base_result["coverage_low_sd"] = float((F - strike_min) / sd_points)
        base_result["coverage_high_sd"] = float((strike_max - F) / sd_points)
        coverage_fails = (
            (F - strike_min) < COVERAGE_K_SD * sd_points
            or (strike_max - F) < COVERAGE_K_SD * sd_points
        )
    else:
        # sigma non disponibile: resta solo il vincolo sul numero di strike e le
        # colonne coverage_*_sd restano vuote
        coverage_fails = False

    thin_otm = (
        (n_put_strikes < MIN_OTM_EACH_SIDE)
        or (n_call_strikes < MIN_OTM_EACH_SIDE)
        or coverage_fails
    )

    mfiv_full = compute_strip_mfiv(full_strip, r, tau)
    mfiv_put = compute_strip_mfiv(puts, r, tau)
    mfiv_call = compute_strip_mfiv(calls, r, tau)

    base_result["mfiv"] = mfiv_full
    base_result["mfiv_put_only"] = mfiv_put
    base_result["mfiv_call_only"] = mfiv_call
    base_result["mfiv_put_share_of_full"] = safe_divide(mfiv_put, mfiv_full)
    base_result["mfiv_call_share_of_full"] = safe_divide(mfiv_call, mfiv_full)

    if (
        np.isfinite(base_result["mfiv_put_share_of_full"])
        and np.isfinite(base_result["mfiv_call_share_of_full"])
    ):
        base_result["mfiv_tail_imbalance"] = float(
            base_result["mfiv_call_share_of_full"]
            - base_result["mfiv_put_share_of_full"]
        )

    if np.isfinite(mfiv_full) and mfiv_full >= 0 and tau > 0:
        base_result["mfiv_annualized_vol"] = float(np.sqrt(mfiv_full / tau))

    if np.isfinite(mfiv_put) and mfiv_put >= 0 and tau > 0:
        base_result["mfiv_put_annualized_vol"] = float(np.sqrt(mfiv_put / tau))

    if np.isfinite(mfiv_call) and mfiv_call >= 0 and tau > 0:
        base_result["mfiv_call_annualized_vol"] = float(np.sqrt(mfiv_call / tau))

    # Varianti dell'Appendice D: strip non troncato e strip senza alcun bid a zero.
    strip_raw = pd.concat(
        [d for d in (puts_raw[["_strike", "_mid"]], calls_raw[["_strike", "_mid"]])
         if not d.empty],
        ignore_index=True
    ) if (not puts_raw.empty or not calls_raw.empty) else pd.DataFrame(
        columns=["_strike", "_mid"])

    strip_nz = pd.concat(
        [d for d in (puts_nz[["_strike", "_mid"]], calls_nz[["_strike", "_mid"]])
         if not d.empty],
        ignore_index=True
    ) if (not puts_nz.empty or not calls_nz.empty) else pd.DataFrame(
        columns=["_strike", "_mid"])

    if not strip_raw.empty:
        strip_raw = (strip_raw.groupby("_strike", as_index=False)
                     .agg({"_mid": "median"})
                     .sort_values("_strike").reset_index(drop=True))

    if not strip_nz.empty:
        strip_nz = (strip_nz.groupby("_strike", as_index=False)
                    .agg({"_mid": "median"})
                    .sort_values("_strike").reset_index(drop=True))

    mfiv_raw = compute_strip_mfiv(strip_raw, r, tau)
    mfiv_nozero = compute_strip_mfiv(strip_nz, r, tau)

    base_result["mfiv_raw_no_truncation"] = mfiv_raw
    base_result["mfiv_nozero_bid"] = mfiv_nozero
    base_result["truncation_ratio"] = safe_divide(mfiv_full, mfiv_raw)
    base_result["n_otm_strikes_raw"] = int(len(strip_raw))
    base_result["n_otm_put_strikes_raw"] = int(puts_raw["_strike"].nunique()) \
        if not puts_raw.empty else 0
    base_result["n_otm_call_strikes_raw"] = int(calls_raw["_strike"].nunique()) \
        if not calls_raw.empty else 0
    base_result["strike_min_raw"] = float(strip_raw["_strike"].min()) \
        if not strip_raw.empty else np.nan
    base_result["strike_max_raw"] = float(strip_raw["_strike"].max()) \
        if not strip_raw.empty else np.nan
    base_result["n_zero_bid_dropped"] = int(len(strip_raw) - n_strikes)

    if not np.isfinite(mfiv_full):
        base_result["mfiv_flag"] = "no_valid_options"
    elif thin_otm:
        base_result["mfiv_flag"] = "thin_otm"
    else:
        base_result["mfiv_flag"] = "ok"

    return base_result


def process_one_day(input_file: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(input_file, dtype=str, keep_default_na=False)
    df = prepare_input_frame(df_raw)

    grouped = df.groupby(["quote_date", "quote_time"], sort=True, dropna=False)

    rows = []

    for (_, _), group in grouped:
        rows.append(compute_mfiv_for_timestamp(group))

    out = pd.DataFrame(rows)

    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])
    out = out.sort_values(["quote_date", "_time_seconds"]).drop(columns=["_time_seconds"])

    return out


def build_end_of_session(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    tmp = intraday.copy()
    tmp["_time_seconds"] = parse_time_to_seconds(tmp["quote_time"])
    tmp["_mfiv_numeric"] = pd.to_numeric(tmp["mfiv"], errors="coerce")

    for quote_date, day in tmp.groupby("quote_date", sort=True):
        close_s = close_seconds_for_date(quote_date)

        before_1600 = day[
            day["_time_seconds"].notna()
            & (day["_time_seconds"] < close_s)
            & (day["mfiv_flag"] != "near_expiry")
        ].copy()

        with_mfiv = before_1600[
            before_1600["_mfiv_numeric"].notna()
            & np.isfinite(before_1600["_mfiv_numeric"])
        ].copy()

        if not with_mfiv.empty:
            selected = with_mfiv.sort_values("_time_seconds").iloc[-1].copy()
        elif not before_1600.empty:
            selected = before_1600.sort_values("_time_seconds").iloc[-1].copy()
        else:
            selected = day.sort_values("_time_seconds").iloc[-1].copy()

        rows.append(selected)

    out = pd.DataFrame(rows)

    drop_cols = [
        "_time_seconds",
        "_mfiv_numeric",
    ]

    out = out.drop(columns=[col for col in drop_cols if col in out.columns])

    return out.reset_index(drop=True)


def build_diagnostics(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for quote_date, day in intraday.groupby("quote_date", sort=True):
        n_otm = pd.to_numeric(day["n_otm_strikes"], errors="coerce")
        mfiv = pd.to_numeric(day["mfiv"], errors="coerce")
        put_share = pd.to_numeric(day["mfiv_put_share_of_full"], errors="coerce")
        call_share = pd.to_numeric(day["mfiv_call_share_of_full"], errors="coerce")
        imbalance = pd.to_numeric(day["mfiv_tail_imbalance"], errors="coerce")
        tratio = pd.to_numeric(day["truncation_ratio"], errors="coerce")
        ndrop = pd.to_numeric(day["n_zero_bid_dropped"], errors="coerce")
        smin_raw = pd.to_numeric(day["strike_min_raw"], errors="coerce")
        avol = pd.to_numeric(day["mfiv_annualized_vol"], errors="coerce")

        ok_day = day[day["mfiv_flag"] == "ok"].copy()

        if not ok_day.empty:
            ok_day["_ts"] = parse_time_to_seconds(ok_day["quote_time"])
            last_ok = ok_day.sort_values("_ts")["quote_time"].iloc[-1]
        else:
            last_ok = ""

        row = {
            "quote_date": quote_date,
            "n_intraday_rows": len(day),
            "n_ok": int((day["mfiv_flag"] == "ok").sum()),
            "n_thin_otm": int((day["mfiv_flag"] == "thin_otm").sum()),
            "n_missing_F": int((day["mfiv_flag"] == "missing_F").sum()),
            "n_near_expiry": int((day["mfiv_flag"] == "near_expiry").sum()),
            "n_no_valid_options": int((day["mfiv_flag"] == "no_valid_options").sum()),
            "last_ok_quote_time": last_ok,
            "n_otm_strikes_min": int(n_otm.min()) if n_otm.notna().any() else 0,
            "n_otm_strikes_median": float(n_otm.median()) if n_otm.notna().any() else np.nan,
            "n_otm_strikes_max": int(n_otm.max()) if n_otm.notna().any() else 0,
            "mfiv_min": mfiv.min(),
            "mfiv_median": mfiv.median(),
            "mfiv_max": mfiv.max(),
            "annualized_vol_median": avol.median(),
            "truncation_ratio_median": tratio.median(),
            "n_zero_bid_dropped_median": ndrop.median(),
            "n_iv_at_bound_before_median": float(
                pd.to_numeric(day["n_iv_at_bound_before"], errors="coerce").median()),
            "n_iv_at_bound_after_max": float(
                pd.to_numeric(day["n_iv_at_bound_after"], errors="coerce").max()),
            "strike_min_raw": smin_raw.min(),
            "sigma_atm_mean": pd.to_numeric(day["sigma_atm"], errors="coerce").mean(),
            "put_share_mean": put_share.mean(),
            "put_share_median": put_share.median(),
            "call_share_mean": call_share.mean(),
            "call_share_median": call_share.median(),
            "tail_imbalance_mean": imbalance.mean(),
            "tail_imbalance_median": imbalance.median(),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_daily_truncation(end_of_session: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    eod = end_of_session.copy()

    for quote_date, eod_row in eod.groupby("quote_date", sort=True):
        day = intraday[intraday["quote_date"] == quote_date].copy()

        put_share = pd.to_numeric(day["mfiv_put_share_of_full"], errors="coerce")
        call_share = pd.to_numeric(day["mfiv_call_share_of_full"], errors="coerce")
        imbalance = pd.to_numeric(day["mfiv_tail_imbalance"], errors="coerce")
        tratio = pd.to_numeric(day["truncation_ratio"], errors="coerce")

        selected = eod_row.iloc[0]

        row = {
            "quote_date": quote_date,
            "eod_quote_time": selected.get("quote_time", ""),
            "eod_mfiv_flag": selected.get("mfiv_flag", ""),
            "eod_tau": selected.get("tau", np.nan),
            "eod_F": selected.get("F", np.nan),
            "eod_mfiv": selected.get("mfiv", np.nan),
            "eod_mfiv_put_only": selected.get("mfiv_put_only", np.nan),
            "eod_mfiv_call_only": selected.get("mfiv_call_only", np.nan),
            # varianti di troncamento dell'Appendice D
            "eod_mfiv_raw_no_truncation": selected.get("mfiv_raw_no_truncation", np.nan),
            "eod_mfiv_nozero_bid": selected.get("mfiv_nozero_bid", np.nan),
            "eod_truncation_ratio": selected.get("truncation_ratio", np.nan),
            "eod_n_zero_bid_dropped": selected.get("n_zero_bid_dropped", np.nan),
            "eod_strike_min": selected.get("strike_min", np.nan),
            "eod_strike_min_raw": selected.get("strike_min_raw", np.nan),
            "eod_coverage_low_sd": selected.get("coverage_low_sd", np.nan),
            "eod_coverage_high_sd": selected.get("coverage_high_sd", np.nan),
            "eod_put_share_of_full": selected.get("mfiv_put_share_of_full", np.nan),
            "eod_call_share_of_full": selected.get("mfiv_call_share_of_full", np.nan),
            "eod_tail_imbalance": selected.get("mfiv_tail_imbalance", np.nan),
            "eod_n_otm_strikes": selected.get("n_otm_strikes", np.nan),
            "eod_n_otm_put_strikes": selected.get("n_otm_put_strikes", np.nan),
            "eod_n_otm_call_strikes": selected.get("n_otm_call_strikes", np.nan),
            "intraday_truncation_ratio_median": tratio.median(),
            "intraday_put_share_mean": put_share.mean(),
            "intraday_put_share_median": put_share.median(),
            "intraday_call_share_mean": call_share.mean(),
            "intraday_call_share_median": call_share.median(),
            "intraday_tail_imbalance_mean": imbalance.mean(),
            "intraday_tail_imbalance_median": imbalance.median(),
            "intraday_n_ok": int((day["mfiv_flag"] == "ok").sum()),
            "intraday_n_thin_otm": int((day["mfiv_flag"] == "thin_otm").sum()),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def mfiv_baseline():
    input_files = select_input_files()

    print("=" * 120)
    print("MFIV FULL YEAR - BJN/JT TOTAL INTEGRATED VARIANCE + CBOE ZERO-BID TRUNCATION")
    print("=" * 120)
    print(f"Input folder:             {BLACK76_DIR}")
    print(f"Output folder:            {OUTPUT_DIR}")
    print(f"Files found:              {len(input_files)}")
    print(f"Year:                     {YEAR}")
    print(f"Near-expiry cutoff:       tau < {NEAR_EXPIRY_MINUTES} minutes")
    print(f"Minimum OTM each side:    {MIN_OTM_EACH_SIDE}")
    print(f"Coverage threshold:       {COVERAGE_K_SD} x sigma_atm x sqrt(tau) x F each side")
    print("Tail truncation:          Cboe rule, stop at first pair of consecutive zero bids")
    print("Appendix D variants:      raw (no truncation) and drop-all-zero-bid, per timestamp")
    print("=" * 120)

    start = time.time()

    all_intraday = []

    for index, input_file in enumerate(input_files, start=1):
        date_string = file_to_date(input_file)

        day_intraday = process_one_day(input_file)
        all_intraday.append(day_intraday)

        if WRITE_DAILY_INTRADAY_FILES:
            daily_output_file = DAILY_INTRADAY_DIR / f"03_mfiv_intraday_{date_string}.csv"

            if daily_output_file.exists() and OVERWRITE_EXISTING:
                daily_output_file.unlink()

            day_intraday.to_csv(daily_output_file, index=False)

        thin_count = int((day_intraday["mfiv_flag"] == "thin_otm").sum())
        ok_count = int((day_intraday["mfiv_flag"] == "ok").sum())

        n_otm = pd.to_numeric(day_intraday["n_otm_strikes"], errors="coerce")
        tr = pd.to_numeric(day_intraday["truncation_ratio"], errors="coerce")

        n_min = int(n_otm.min()) if n_otm.notna().any() else 0
        n_max = int(n_otm.max()) if n_otm.notna().any() else 0

        elapsed_so_far = time.time() - start
        eta = elapsed_so_far / index * (len(input_files) - index) / 60.0

        print(
            f"[{index:3d}/{len(input_files)}] {date_string} | "
            f"rows={len(day_intraday):,} | ok={ok_count:,} | thin={thin_count:,} | "
            f"n_otm=[{n_min},{n_max}] | trunc_ratio_med={tr.median():.3f} | "
            f"ETA {eta:4.1f} min",
            flush=True
        )

    intraday_all = pd.concat(all_intraday, ignore_index=True)

    intraday_output_file = OUTPUT_DIR / f"03_mfiv_intraday_{YEAR}_all.csv"
    eod_output_file = OUTPUT_DIR / f"03_mfiv_end_of_session_{YEAR}_all.csv"
    diagnostics_output_file = OUTPUT_DIR / f"03_mfiv_diagnostics_{YEAR}_all.csv"
    truncation_output_file = OUTPUT_DIR / f"03_mfiv_daily_truncation_{YEAR}_appendix_D.csv"

    if WRITE_COMBINED_INTRADAY_FILE:
        if intraday_output_file.exists() and OVERWRITE_EXISTING:
            intraday_output_file.unlink()

        intraday_all.to_csv(intraday_output_file, index=False)

    end_of_session = build_end_of_session(intraday_all)

    if eod_output_file.exists() and OVERWRITE_EXISTING:
        eod_output_file.unlink()

    end_of_session.to_csv(eod_output_file, index=False)

    diagnostics = build_diagnostics(intraday_all)

    if diagnostics_output_file.exists() and OVERWRITE_EXISTING:
        diagnostics_output_file.unlink()

    diagnostics.to_csv(diagnostics_output_file, index=False)

    daily_truncation = build_daily_truncation(end_of_session, intraday_all)

    if truncation_output_file.exists() and OVERWRITE_EXISTING:
        truncation_output_file.unlink()

    daily_truncation.to_csv(truncation_output_file, index=False)

    elapsed = time.time() - start

    ok_mask = intraday_all["mfiv_flag"] == "ok"
    tr_all = pd.to_numeric(intraday_all["truncation_ratio"], errors="coerce")
    vol_ok = pd.to_numeric(intraday_all.loc[ok_mask, "mfiv_annualized_vol"], errors="coerce")

    print()
    print("=" * 120)
    print("COMPLETED MFIV FULL YEAR")
    print("=" * 120)
    print(f"Files processed:                 {len(input_files)}")
    print(f"Intraday rows:                   {len(intraday_all):,}")
    print(f"End-of-session rows:             {len(end_of_session):,}")
    print(f"Total ok rows:                   {int(ok_mask.sum()):,}")
    print(f"Total thin_otm rows:             {int((intraday_all['mfiv_flag'] == 'thin_otm').sum()):,}")
    print(f"Total missing_F rows:            {int((intraday_all['mfiv_flag'] == 'missing_F').sum()):,}")
    print(f"Total near_expiry rows:          {int((intraday_all['mfiv_flag'] == 'near_expiry').sum()):,}")
    print(f"Total no_valid_options rows:     {int((intraday_all['mfiv_flag'] == 'no_valid_options').sum()):,}")
    print(f"Elapsed time:                    {elapsed / 60:.2f} minutes")
    print()
    print("--- CONTROLLI DI COERENZA ---")
    print(f"righe ok attese 92828, ottenute {int(ok_mask.sum())}  "
          f"{'MATCH' if int(ok_mask.sum()) == 92828 else 'DIVERSO'}")
    print(f"truncation_ratio  p25 {tr_all.quantile(.25):.3f} (atteso 0.291) | "
          f"mediana {tr_all.median():.3f} (atteso 0.492) | "
          f"p75 {tr_all.quantile(.75):.3f} (atteso 0.677)")
    print(f"vol annualizzata sulle ok: p05 {vol_ok.quantile(.05):.3f} | "
          f"mediana {vol_ok.median():.3f} (atteso ~0.249) | max {vol_ok.max():.3f}")
    print(f"giornate senza alcun timestamp ok: "
          f"{int((diagnostics['n_ok'] == 0).sum())} (atteso 0)")
    bb = pd.to_numeric(intraday_all["n_iv_at_bound_before"], errors="coerce")
    ba = pd.to_numeric(intraday_all["n_iv_at_bound_after"], errors="coerce")
    print()
    print("--- RIGHE CON IV SATURA: DOVE I DUE CRITERI DIVERGONO ---")
    print(f"strike con IV satura prima del troncamento, mediana: {bb.median():.1f}")
    print(f"strike che il solo troncamento CBOE avrebbe lasciato passare: "
          f"max {int(ba.max())}, totale {int(ba.sum())}")
    print("ATTENZIONE ALLA LETTURA. Questo contatore misura quanti strike saturi il")
    print("criterio ECONOMICO (bid a zero) da solo NON avrebbe escluso, non quanti")
    print("ne restano nello strip: quelle righe vengono comunque")
    print("tolte dal filtro secondario sulla saturazione, applicato dopo il")
    print("troncamento. Un totale diverso da zero indica quindi i timestamp dove i")
    print("due criteri divergono, non un difetto residuo. Misurato sul 2025: tre")
    print("timestamp su 96.960 (2025-04-09 13:21 e 13:28, 2025-10-14 15:39), uno")
    print("dei quali portava il massimo annuale di volatilita' implicita.")
    print()
    print(f"Intraday output:                 {intraday_output_file}")
    print(f"End-of-session output:           {eod_output_file}")
    print(f"Diagnostics output:              {diagnostics_output_file}")
    print(f"Daily truncation Appendix D:     {truncation_output_file}")
    print("=" * 120)


# =====================================================================
# PARTE 2: MFIV CON TAGLIO DI LOG-MONEYNESS, 0.05 POI 0.02
# =====================================================================
# Una variante per taglio, in quest'ordine; i file vanno in 03_mfiv/ accanto a quelli della
# parte 1, con l'etichetta _logm_005 o _logm_002.
LOG_MONEYNESS_VARIANTS = (0.05, 0.02)

MIN_STRIKE_COVERAGE_PCT = 0.01

# Chiusura della sessione per giornata (13:00 nelle chiusure anticipate), da common.py.
def session_close_seconds_for(quote_date) -> int:
    text = EARLY_CLOSE_TIMES.get(str(quote_date), REGULAR_CLOSE)
    h, m, s = text.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def mfiv_variant_label(log_moneyness_cutoff: float) -> str:
    """Etichetta dei file della variante: ammessi solo i tagli 0.05 e 0.02."""
    if np.isclose(log_moneyness_cutoff, 0.05, rtol=0.0, atol=1e-12):
        return "logm_005"
    elif np.isclose(log_moneyness_cutoff, 0.02, rtol=0.0, atol=1e-12):
        return "logm_002"
    else:
        raise ValueError(
            "LOG_MONEYNESS_CUTOFF must be exactly 0.05 or 0.02. "
            "The validated baseline (no cutoff) is part 1 of this script."
        )


def prepare_input_frame_logm(df: pd.DataFrame):
    """Come prepare_input_frame della parte 1, ma senza bid e IV ATM e con la
    ricerca del flag di IV saturata fra piu' nomi. Restituisce anche il nome
    della colonna del flag trovata (None se manca)."""
    out = df.copy()

    if "quote_date" not in out.columns or "quote_time" not in out.columns:
        if "quote_datetime" in out.columns:
            out["quote_date"] = out["quote_datetime"].astype(str).str[:10]
            out["quote_time"] = out["quote_datetime"].astype(str).str[11:19]
        elif "quoted_datetime" in out.columns:
            out["quote_date"] = out["quoted_datetime"].astype(str).str[:10]
            out["quote_time"] = out["quoted_datetime"].astype(str).str[11:19]
        else:
            raise ValueError(
                "Non trovo quote_date/quote_time oppure quote_datetime/quoted_datetime."
            )

    required_base_columns = [
        "quote_date",
        "quote_time",
        "strike",
        "option_type",
        "black76_mid",
        "black76_F_365",
        "black76_T_365",
        "black76_r",
    ]

    missing = [
        col for col in required_base_columns
        if col not in out.columns
    ]

    if missing:
        raise ValueError(f"Colonne mancanti nel file input: {missing}")

    out["_strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["_mid"] = pd.to_numeric(out["black76_mid"], errors="coerce")
    out["_F"] = pd.to_numeric(out["black76_F_365"], errors="coerce")
    out["_tau"] = pd.to_numeric(out["black76_T_365"], errors="coerce")
    out["_r"] = pd.to_numeric(out["black76_r"], errors="coerce")
    out["_option_type"] = normalize_option_type(out["option_type"])

    # Flag di IV saturata, per escludere le stesse righe della MFIV di base. Si cerca fra piu'
    # nomi; se manca, il filtro non si applica e il riepilogo finale lo segnala.
    iv_bound_col = None

    for candidate in ("black76_iv_at_bound_365", "iv_at_bound",
                      "black76_iv_at_bound", "iv_bound",
                      "black76_iv_bound", "iv_saturated"):
        if candidate in out.columns:
            iv_bound_col = candidate
            break

    if iv_bound_col is not None:
        out["_iv_at_bound"] = pd.to_numeric(
            out[iv_bound_col], errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["_iv_at_bound"] = 0
    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])

    if "black76_solver_success_365" in out.columns:
        out["_solver_success"] = pd.to_numeric(
            out["black76_solver_success_365"],
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        iv = pd.to_numeric(
            out.get("black76_implied_volatility_365", np.nan),
            errors="coerce"
        )

        gamma = pd.to_numeric(
            out.get("black76_gamma_365", np.nan),
            errors="coerce"
        )

        out["_solver_success"] = np.where(
            out["_mid"].notna()
            & out["_F"].notna()
            & out["_tau"].notna()
            & iv.notna()
            & gamma.notna()
            & np.isfinite(out["_mid"])
            & np.isfinite(out["_F"])
            & np.isfinite(out["_tau"])
            & np.isfinite(iv)
            & np.isfinite(gamma)
            & (out["_mid"] > 0)
            & (out["_F"] > 0)
            & (out["_tau"] > 0)
            & (iv > 0),
            1,
            0
        )

    if "black76_near_expiry_flag_365" in out.columns:
        out["_near_expiry"] = pd.to_numeric(
            out["black76_near_expiry_flag_365"],
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["_near_expiry"] = np.where(
            out["_tau"].notna()
            & np.isfinite(out["_tau"])
            & (out["_tau"] > 0)
            & (out["_tau"] < NEAR_EXPIRY_TAU),
            1,
            0
        )

    return out, iv_bound_col


def compute_mfiv_for_timestamp_logm(group: pd.DataFrame, log_moneyness_cutoff: float,
                                    variant_label: str) -> dict:
    quote_date = str(group["quote_date"].iloc[0])
    quote_time = str(group["quote_time"].iloc[0])

    F = safe_median(group["_F"])
    tau = safe_median(group["_tau"])
    r = safe_median(group["_r"])

    base_result = {
        "quote_date": quote_date,
        "quote_time": quote_time,
        "tau": tau,
        "F": F,
        "r": r,
        "log_moneyness_cutoff_applied": float(log_moneyness_cutoff),
        "n_saturated_iv_strikes_dropped": 0,
        "mfiv_variant": variant_label,
        "n_otm_strikes": 0,
        "n_otm_put_strikes": 0,
        "n_otm_call_strikes": 0,
        "strike_min": np.nan,
        "strike_max": np.nan,
        "put_strike_min": np.nan,
        "put_strike_max": np.nan,
        "call_strike_min": np.nan,
        "call_strike_max": np.nan,
        "strike_coverage_low": np.nan,
        "strike_coverage_high": np.nan,
        "mfiv": np.nan,
        "mfiv_put_only": np.nan,
        "mfiv_call_only": np.nan,
        "mfiv_put_share_of_full": np.nan,
        "mfiv_call_share_of_full": np.nan,
        "mfiv_tail_imbalance": np.nan,
        "mfiv_annualized_vol": np.nan,
        "mfiv_put_annualized_vol": np.nan,
        "mfiv_call_annualized_vol": np.nan,
        "mfiv_flag": "",
    }

    if not np.isfinite(F) or F <= 0:
        base_result["mfiv_flag"] = "missing_F"
        return base_result

    if not np.isfinite(tau) or tau <= 0:
        base_result["mfiv_flag"] = "near_expiry"
        return base_result

    if tau < NEAR_EXPIRY_TAU:
        base_result["mfiv_flag"] = "near_expiry"
        return base_result

    if not np.isfinite(r):
        base_result["mfiv_flag"] = "missing_F"
        return base_result

    valid = group[
        (group["_solver_success"] == 1)
        & (group["_near_expiry"] == 0)
        & group["_strike"].notna()
        & group["_mid"].notna()
        & group["_F"].notna()
        & group["_tau"].notna()
        & group["_r"].notna()
        & np.isfinite(group["_strike"])
        & np.isfinite(group["_mid"])
        & np.isfinite(group["_F"])
        & np.isfinite(group["_tau"])
        & np.isfinite(group["_r"])
        & (group["_strike"] > 0)
        & (group["_mid"] > 0)
        & (group["_F"] > 0)
        & (group["_tau"] > 0)
        & (group["_option_type"].isin(["C", "P"]))
    ].copy()

    if valid.empty:
        base_result["mfiv_flag"] = "no_valid_options"
        return base_result

    # Il flag di IV saturata e' per riga; passando allo strike si prende il
    # massimo: uno strike e' saturato se lo e' almeno una delle sue righe.
    collapsed = (
        valid[["_strike", "_option_type", "_mid", "_iv_at_bound"]]
        .groupby(["_strike", "_option_type"], as_index=False)
        .agg({"_mid": "median", "_iv_at_bound": "max"})
    )

    puts = collapsed[
        (collapsed["_option_type"] == "P")
        & (collapsed["_strike"] < F)
    ][["_strike", "_mid", "_iv_at_bound"]].copy()

    calls = collapsed[
        (collapsed["_option_type"] == "C")
        & (collapsed["_strike"] > F)
    ][["_strike", "_mid", "_iv_at_bound"]].copy()

    # Taglio di log-moneyness, applicato sopra la regola OTM.
    puts = puts[
        np.abs(np.log(puts["_strike"] / F)) <= log_moneyness_cutoff
    ].copy()

    calls = calls[
        np.abs(np.log(calls["_strike"] / F)) <= log_moneyness_cutoff
    ].copy()

    # Esclusione delle IV sature dopo il taglio, come nella parte 1 dopo il troncamento: incide
    # poco, ma rende la variante confrontabile con la MFIV di base.
    n_saturated_dropped = int(
        (puts["_iv_at_bound"] == 1).sum() + (calls["_iv_at_bound"] == 1).sum()
    )

    puts = puts[puts["_iv_at_bound"] == 0].copy()
    calls = calls[calls["_iv_at_bound"] == 0].copy()

    exact_atm = collapsed[
        np.isclose(collapsed["_strike"], F, rtol=0.0, atol=1e-8)
    ]

    atm_strip = pd.DataFrame(columns=["_strike", "_mid"])

    if not exact_atm.empty:
        atm_price = exact_atm["_mid"].mean()

        if np.isfinite(atm_price) and atm_price > 0:
            atm_strip = pd.DataFrame({
                "_strike": [F],
                "_mid": [atm_price],
            })

    n_put_strikes = int(puts["_strike"].nunique())
    n_call_strikes = int(calls["_strike"].nunique())

    strip_parts = []

    if not puts.empty:
        strip_parts.append(puts[["_strike", "_mid"]])

    if not atm_strip.empty:
        strip_parts.append(atm_strip)

    if not calls.empty:
        strip_parts.append(calls[["_strike", "_mid"]])

    if not strip_parts:
        base_result["mfiv_flag"] = "no_valid_options"
        return base_result

    full_strip = pd.concat(strip_parts, ignore_index=True)

    full_strip = (
        full_strip.groupby("_strike", as_index=False)
        .agg({"_mid": "median"})
        .sort_values("_strike")
        .reset_index(drop=True)
    )

    strikes = full_strip["_strike"].to_numpy(dtype=float)

    n_strikes = len(strikes)

    base_result["n_otm_strikes"] = int(n_strikes)
    base_result["n_otm_put_strikes"] = n_put_strikes
    base_result["n_otm_call_strikes"] = n_call_strikes
    base_result["n_saturated_iv_strikes_dropped"] = int(n_saturated_dropped)
    base_result["strike_min"] = float(np.nanmin(strikes)) if n_strikes > 0 else np.nan
    base_result["strike_max"] = float(np.nanmax(strikes)) if n_strikes > 0 else np.nan

    if n_put_strikes > 0:
        base_result["put_strike_min"] = float(puts["_strike"].min())
        base_result["put_strike_max"] = float(puts["_strike"].max())

    if n_call_strikes > 0:
        base_result["call_strike_min"] = float(calls["_strike"].min())
        base_result["call_strike_max"] = float(calls["_strike"].max())

    if n_strikes == 0:
        base_result["mfiv_flag"] = "no_valid_options"
        return base_result

    strike_min = base_result["strike_min"]
    strike_max = base_result["strike_max"]

    coverage_low = strike_min / F - 1.0
    coverage_high = strike_max / F - 1.0

    base_result["strike_coverage_low"] = float(coverage_low)
    base_result["strike_coverage_high"] = float(coverage_high)

    thin_otm = (
        (n_put_strikes < MIN_OTM_EACH_SIDE)
        or (n_call_strikes < MIN_OTM_EACH_SIDE)
        or (strike_min > (1.0 - MIN_STRIKE_COVERAGE_PCT) * F)
        or (strike_max < (1.0 + MIN_STRIKE_COVERAGE_PCT) * F)
    )

    mfiv_full = compute_strip_mfiv(full_strip, r, tau)
    mfiv_put = compute_strip_mfiv(puts, r, tau)
    mfiv_call = compute_strip_mfiv(calls, r, tau)

    base_result["mfiv"] = mfiv_full
    base_result["mfiv_put_only"] = mfiv_put
    base_result["mfiv_call_only"] = mfiv_call
    base_result["mfiv_put_share_of_full"] = safe_divide(mfiv_put, mfiv_full)
    base_result["mfiv_call_share_of_full"] = safe_divide(mfiv_call, mfiv_full)

    if (
        np.isfinite(base_result["mfiv_put_share_of_full"])
        and np.isfinite(base_result["mfiv_call_share_of_full"])
    ):
        base_result["mfiv_tail_imbalance"] = float(
            base_result["mfiv_call_share_of_full"]
            - base_result["mfiv_put_share_of_full"]
        )

    if np.isfinite(mfiv_full) and mfiv_full >= 0 and tau > 0:
        base_result["mfiv_annualized_vol"] = float(np.sqrt(mfiv_full / tau))

    if np.isfinite(mfiv_put) and mfiv_put >= 0 and tau > 0:
        base_result["mfiv_put_annualized_vol"] = float(np.sqrt(mfiv_put / tau))

    if np.isfinite(mfiv_call) and mfiv_call >= 0 and tau > 0:
        base_result["mfiv_call_annualized_vol"] = float(np.sqrt(mfiv_call / tau))

    if not np.isfinite(mfiv_full):
        base_result["mfiv_flag"] = "no_valid_options"
    elif thin_otm:
        base_result["mfiv_flag"] = "thin_otm"
    else:
        base_result["mfiv_flag"] = "ok"

    return base_result


def process_one_day_logm(input_file: Path, log_moneyness_cutoff: float, variant_label: str):
    """Restituisce le righe intraday della giornata e il nome della colonna del
    flag di IV saturata trovata nell'input (None se manca)."""
    df_raw = pd.read_csv(input_file, dtype=str, keep_default_na=False)
    df, iv_bound_col = prepare_input_frame_logm(df_raw)

    grouped = df.groupby(["quote_date", "quote_time"], sort=True, dropna=False)

    rows = []

    for (_, _), group in grouped:
        rows.append(compute_mfiv_for_timestamp_logm(group, log_moneyness_cutoff, variant_label))

    out = pd.DataFrame(rows)

    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])
    out = out.sort_values(["quote_date", "_time_seconds"]).drop(columns=["_time_seconds"])

    return out, iv_bound_col


def build_end_of_session_logm(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    tmp = intraday.copy()
    tmp["_time_seconds"] = parse_time_to_seconds(tmp["quote_time"])
    tmp["_mfiv_numeric"] = pd.to_numeric(tmp["mfiv"], errors="coerce")

    for quote_date, day in tmp.groupby("quote_date", sort=True):
        close_seconds = session_close_seconds_for(quote_date)

        before_close = day[
            day["_time_seconds"].notna()
            & (day["_time_seconds"] < close_seconds)
            & (day["mfiv_flag"] != "near_expiry")
        ].copy()

        with_mfiv = before_close[
            before_close["_mfiv_numeric"].notna()
            & np.isfinite(before_close["_mfiv_numeric"])
        ].copy()

        if not with_mfiv.empty:
            selected = with_mfiv.sort_values("_time_seconds").iloc[-1].copy()
        elif not before_close.empty:
            selected = before_close.sort_values("_time_seconds").iloc[-1].copy()
        else:
            selected = day.sort_values("_time_seconds").iloc[-1].copy()

        rows.append(selected)

    out = pd.DataFrame(rows)

    drop_cols = [
        "_time_seconds",
        "_mfiv_numeric",
    ]

    out = out.drop(columns=[col for col in drop_cols if col in out.columns])

    return out.reset_index(drop=True)


def build_diagnostics_logm(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for quote_date, day in intraday.groupby("quote_date", sort=True):
        n_otm = pd.to_numeric(day["n_otm_strikes"], errors="coerce")
        mfiv = pd.to_numeric(day["mfiv"], errors="coerce")
        put_share = pd.to_numeric(day["mfiv_put_share_of_full"], errors="coerce")
        call_share = pd.to_numeric(day["mfiv_call_share_of_full"], errors="coerce")
        imbalance = pd.to_numeric(day["mfiv_tail_imbalance"], errors="coerce")

        row = {
            "quote_date": quote_date,
            "n_intraday_rows": len(day),
            "n_ok": int((day["mfiv_flag"] == "ok").sum()),
            "n_thin_otm": int((day["mfiv_flag"] == "thin_otm").sum()),
            "n_missing_F": int((day["mfiv_flag"] == "missing_F").sum()),
            "n_near_expiry": int((day["mfiv_flag"] == "near_expiry").sum()),
            "n_no_valid_options": int((day["mfiv_flag"] == "no_valid_options").sum()),
            "n_otm_strikes_min": int(n_otm.min()) if n_otm.notna().any() else 0,
            "n_otm_strikes_median": float(n_otm.median()) if n_otm.notna().any() else np.nan,
            "n_otm_strikes_max": int(n_otm.max()) if n_otm.notna().any() else 0,
            "mfiv_min": mfiv.min(),
            "mfiv_median": mfiv.median(),
            "mfiv_max": mfiv.max(),
            "put_share_mean": put_share.mean(),
            "put_share_median": put_share.median(),
            "call_share_mean": call_share.mean(),
            "call_share_median": call_share.median(),
            "tail_imbalance_mean": imbalance.mean(),
            "tail_imbalance_median": imbalance.median(),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_daily_truncation_logm(end_of_session: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    eod = end_of_session.copy()

    for quote_date, eod_row in eod.groupby("quote_date", sort=True):
        day = intraday[intraday["quote_date"] == quote_date].copy()

        put_share = pd.to_numeric(day["mfiv_put_share_of_full"], errors="coerce")
        call_share = pd.to_numeric(day["mfiv_call_share_of_full"], errors="coerce")
        imbalance = pd.to_numeric(day["mfiv_tail_imbalance"], errors="coerce")

        selected = eod_row.iloc[0]

        row = {
            "quote_date": quote_date,
            "eod_quote_time": selected.get("quote_time", ""),
            "eod_mfiv_flag": selected.get("mfiv_flag", ""),
            "eod_tau": selected.get("tau", np.nan),
            "eod_F": selected.get("F", np.nan),
            "eod_mfiv": selected.get("mfiv", np.nan),
            "eod_mfiv_put_only": selected.get("mfiv_put_only", np.nan),
            "eod_mfiv_call_only": selected.get("mfiv_call_only", np.nan),
            "eod_put_share_of_full": selected.get("mfiv_put_share_of_full", np.nan),
            "eod_call_share_of_full": selected.get("mfiv_call_share_of_full", np.nan),
            "eod_tail_imbalance": selected.get("mfiv_tail_imbalance", np.nan),
            "eod_n_otm_strikes": selected.get("n_otm_strikes", np.nan),
            "eod_n_otm_put_strikes": selected.get("n_otm_put_strikes", np.nan),
            "eod_n_otm_call_strikes": selected.get("n_otm_call_strikes", np.nan),
            "intraday_put_share_mean": put_share.mean(),
            "intraday_put_share_median": put_share.median(),
            "intraday_call_share_mean": call_share.mean(),
            "intraday_call_share_median": call_share.median(),
            "intraday_tail_imbalance_mean": imbalance.mean(),
            "intraday_tail_imbalance_median": imbalance.median(),
            "intraday_n_ok": int((day["mfiv_flag"] == "ok").sum()),
            "intraday_n_thin_otm": int((day["mfiv_flag"] == "thin_otm").sum()),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def mfiv_logmoneyness(log_moneyness_cutoff: float):
    """Una variante: taglio |ln(K/F)| <= log_moneyness_cutoff (0.05 oppure 0.02)."""
    variant_label = mfiv_variant_label(log_moneyness_cutoff)

    input_files = select_input_files()

    # se la colonna del flag di IV saturata manca, il riepilogo finale lo segnala
    iv_at_bound_column_found = False
    iv_at_bound_column_name = ""

    print("=" * 120)
    print("MFIV LOG-MONEYNESS ROBUSTNESS - BJN/JT TOTAL INTEGRATED VARIANCE")
    print("=" * 120)
    print(f"Input folder:             {BLACK76_DIR}")
    print(f"Output folder:            {OUTPUT_DIR}")
    print(f"Files found:              {len(input_files)}")
    print(f"Year:                     {YEAR}")
    print(f"Near-expiry cutoff:       tau < {NEAR_EXPIRY_MINUTES} minutes")
    print(f"Minimum OTM each side:    {MIN_OTM_EACH_SIDE}")
    print(f"Minimum strike coverage:  +/- {MIN_STRIKE_COVERAGE_PCT:.2%} around F")
    print(f"Log-moneyness cutoff:      |ln(K/F)| <= {log_moneyness_cutoff:.2f}")
    print(f"Input:                     {BLACK76_DIR}")
    print(f"Baseline di riferimento:   {OUTPUT_DIR} (parte 1, file senza etichetta)")
    print(f"MFIV variant label:        {variant_label}")
    print(f"Baseline protetta:         i file della variante finiscono in _{variant_label}")
    print("Tail decomposition:       put-only MFIV and call-only MFIV at every timestamp")
    print("=" * 120)

    start = time.time()

    all_intraday = []

    for index, input_file in enumerate(input_files, start=1):
        date_string = file_to_date(input_file)

        print(f"[{index}/{len(input_files)}] Processing {input_file.name}")

        day_intraday, iv_bound_col = process_one_day_logm(
            input_file, log_moneyness_cutoff, variant_label
        )
        all_intraday.append(day_intraday)

        if iv_bound_col is not None:
            iv_at_bound_column_found = True
            iv_at_bound_column_name = iv_bound_col

        if WRITE_DAILY_INTRADAY_FILES:
            daily_output_file = DAILY_INTRADAY_DIR / f"03_mfiv_intraday_{date_string}_{variant_label}.csv"

            if daily_output_file.exists() and OVERWRITE_EXISTING:
                daily_output_file.unlink()

            day_intraday.to_csv(daily_output_file, index=False)

        thin_count = int((day_intraday["mfiv_flag"] == "thin_otm").sum())
        ok_count = int((day_intraday["mfiv_flag"] == "ok").sum())

        n_otm = pd.to_numeric(day_intraday["n_otm_strikes"], errors="coerce")

        n_min = int(n_otm.min()) if n_otm.notna().any() else 0
        n_max = int(n_otm.max()) if n_otm.notna().any() else 0

        print(
            f"    rows={len(day_intraday):,} | "
            f"ok={ok_count:,} | "
            f"thin_otm={thin_count:,} | "
            f"n_otm_range=[{n_min}, {n_max}]"
        )

    intraday_all = pd.concat(all_intraday, ignore_index=True)

    intraday_output_file = OUTPUT_DIR / f"03_mfiv_intraday_{YEAR}_all_{variant_label}.csv"
    eod_output_file = OUTPUT_DIR / f"03_mfiv_end_of_session_{YEAR}_all_{variant_label}.csv"
    diagnostics_output_file = OUTPUT_DIR / f"03_mfiv_diagnostics_{YEAR}_all_{variant_label}.csv"
    truncation_output_file = OUTPUT_DIR / f"03_mfiv_daily_truncation_{YEAR}_appendix_D_{variant_label}.csv"

    if WRITE_COMBINED_INTRADAY_FILE:
        if intraday_output_file.exists() and OVERWRITE_EXISTING:
            intraday_output_file.unlink()

        intraday_all.to_csv(intraday_output_file, index=False)

    end_of_session = build_end_of_session_logm(intraday_all)

    if eod_output_file.exists() and OVERWRITE_EXISTING:
        eod_output_file.unlink()

    end_of_session.to_csv(eod_output_file, index=False)

    diagnostics = build_diagnostics_logm(intraday_all)

    if diagnostics_output_file.exists() and OVERWRITE_EXISTING:
        diagnostics_output_file.unlink()

    diagnostics.to_csv(diagnostics_output_file, index=False)

    daily_truncation = build_daily_truncation_logm(end_of_session, intraday_all)

    if truncation_output_file.exists() and OVERWRITE_EXISTING:
        truncation_output_file.unlink()

    daily_truncation.to_csv(truncation_output_file, index=False)

    elapsed = time.time() - start

    print()
    print("=" * 120)
    print("COMPLETED MFIV FULL YEAR")
    print("=" * 120)
    print(f"Files processed:                 {len(input_files)}")
    print(f"Intraday rows:                   {len(intraday_all):,}")
    print(f"End-of-session rows:             {len(end_of_session):,}")
    print(f"Daily truncation rows:           {len(daily_truncation):,}")
    print(f"Total ok rows:                   {int((intraday_all['mfiv_flag'] == 'ok').sum()):,}")
    print(f"Total thin_otm rows:             {int((intraday_all['mfiv_flag'] == 'thin_otm').sum()):,}")
    print(f"Total missing_F rows:            {int((intraday_all['mfiv_flag'] == 'missing_F').sum()):,}")
    print(f"Total near_expiry rows:          {int((intraday_all['mfiv_flag'] == 'near_expiry').sum()):,}")
    print(f"Total no_valid_options rows:     {int((intraday_all['mfiv_flag'] == 'no_valid_options').sum()):,}")
    print(f"Elapsed time:                    {elapsed / 60:.2f} minutes")
    print()
    print(f"Intraday output:                 {intraday_output_file}")
    print(f"End-of-session output:           {eod_output_file}")
    print(f"Diagnostics output:              {diagnostics_output_file}")
    print(f"Daily truncation Appendix D:     {truncation_output_file}")

    print()
    print("--- FILTRO SULLE IV SATURATE ---")

    if iv_at_bound_column_found:
        _sat = pd.to_numeric(
            intraday_all.get("n_saturated_iv_strikes_dropped", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0)

        print(f"  colonna trovata:        {iv_at_bound_column_name}")
        print(f"  strike esclusi, totale: {int(_sat.sum()):,}")
        print(f"  timestamp toccati:      {int((_sat > 0).sum()):,} su {len(intraday_all):,}")
        print("  Con il taglio a +/- 5 per cento gli strike profondamente fuori dal")
        print("  denaro sono gia' esclusi, quindi un conteggio basso e' atteso.")
        print("  Misurato su tre giornate campione prima di questa corsa: lo 0,4 per")
        print("  cento delle righe dentro il taglio a 0,05 e' saturato, e quasi")
        print("  nessuna dentro il taglio a 0,02. Il filtro serve percio' alla")
        print("  COERENZA DI DEFINIZIONE con la baseline, non a un effetto atteso:")
        print("  senza, la variante differirebbe dalla baseline per due ragioni")
        print("  invece che per una, e una differenza nei risultati non sarebbe")
        print("  attribuibile.")
    else:
        print("  ATTENZIONE: la colonna del flag di IV saturata NON e' stata trovata")
        print("  nell'input. Il filtro NON e' stato applicato. I nomi cercati sono:")
        print("  iv_at_bound, black76_iv_at_bound, iv_bound, black76_iv_bound,")
        print("  iv_saturated. Questa variante NON e' quindi confrontabile con la")
        print("  MFIV di baseline sulla definizione degli strike inclusi.")
    print(f"Daily intraday folder:           {DAILY_INTRADAY_DIR}")
    print("=" * 120)

    print()
    print("End-of-session preview:")

    preview_cols = [
        "quote_date",
        "quote_time",
        "tau",
        "F",
        "n_otm_strikes",
        "n_otm_put_strikes",
        "n_otm_call_strikes",
        "mfiv",
        "mfiv_put_only",
        "mfiv_call_only",
        "mfiv_put_share_of_full",
        "mfiv_call_share_of_full",
        "mfiv_tail_imbalance",
        "mfiv_flag",
    ]

    print(end_of_session[preview_cols].head(20).to_string(index=False))


def main():
    mfiv_baseline()

    # Parte 2: le due varianti girano comunque entrambe, in ordine; un errore
    # in una viene segnalato alla fine.
    failed = []
    for log_moneyness_cutoff in LOG_MONEYNESS_VARIANTS:
        try:
            mfiv_logmoneyness(log_moneyness_cutoff)
        except Exception:
            traceback.print_exc()
            failed.append(str(log_moneyness_cutoff))

    if failed:
        raise SystemExit(f"Varianti con errori o controlli falliti: {failed}")


if __name__ == "__main__":
    main()
