"""
04_Gamma_Herfindahl.py: gamma aggregato e Herfindahl per strike a ogni minuto delle 0DTE SPXW.

Parte 1: con il gamma Black '76, G_OI (gamma x OI), G_VOL (gamma x volume), H_OI e H_VOL.
Parte 2: le stesse misure con il gamma del vendor e gli strike persi per il suo arrotondamento.

Input:  CBOE/Data/02_black76_iv_greeks/
Output: CBOE/Data/04_gamma_herfindahl/ (intraday, 12:00, fine sessione, diagnostiche, sintesi;
        i file della parte 2 hanno il prefisso 04_gamma_cboe_)
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    DATA_DIR,
    EARLY_CLOSE_SECONDS,
    REGULAR_CLOSE_SECONDS,
    YEAR,
    output_dir,
)

# =====================================================================
# CONFIGURAZIONE COMUNE ALLE DUE PARTI
# =====================================================================
BLACK76_DIR = DATA_DIR / "02_black76_iv_greeks"

OUTPUT_DIR = output_dir("04_gamma_herfindahl")
DAILY_INTRADAY_DIR = OUTPUT_DIR / "intraday_by_day"
DAILY_INTRADAY_DIR.mkdir(parents=True, exist_ok=True)

YEAR_BASIS = 365

NEAR_EXPIRY_MINUTES = 5
NEAR_EXPIRY_SECONDS = NEAR_EXPIRY_MINUTES * 60
NEAR_EXPIRY_TAU = NEAR_EXPIRY_SECONDS / (YEAR_BASIS * 24.0 * 60.0 * 60.0)

MIN_POSITIVE_STRIKES = 10

SNAPSHOT_TIME = "12:00:00"
SNAPSHOT_TOLERANCE_MINUTES = 5
SNAPSHOT_TOLERANCE_SECONDS = SNAPSHOT_TOLERANCE_MINUTES * 60

# =====================================================================
# UNITA' DI AGGREGAZIONE DEL HERFINDAHL (vale per le due parti)
# =====================================================================
# Herfindahl per strike: sotto la parita' put-call call e put allo stesso strike hanno lo
# stesso gamma, quindi la convessita' a quel livello di prezzo e' gamma * (OI_call + OI_put).
# La versione per contratto resta come colonna diagnostica (h_oi_contract, h_vol_contract).
HERFINDAHL_UNIT = "strike"   # "strike" oppure "contract"

# =====================================================================
# CHIUSURA DI SESSIONE, PER GIORNATA
# =====================================================================
# 16:00, oppure 13:00 nelle sessioni a chiusura anticipata (EARLY_CLOSE_SECONDS in common.py).

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


def time_string_to_seconds(time_string: str) -> int:
    h, m, s = time_string.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def close_seconds_for_date(quote_date: str) -> int:
    return EARLY_CLOSE_SECONDS.get(str(quote_date), REGULAR_CLOSE_SECONDS)


def safe_median(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]

    if values.empty:
        return np.nan

    return float(values.median())


def herfindahl_from_exposure(exposure) -> float:
    x = pd.to_numeric(pd.Series(exposure), errors="coerce")
    x = x[np.isfinite(x)]
    x = x[x > 0]

    if x.empty:
        return np.nan

    total = x.sum()

    if not np.isfinite(total) or total <= 0:
        return np.nan

    weights = x / total
    return float(np.sum(weights ** 2))


def build_end_of_session(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    tmp = intraday.copy()
    tmp["_time_seconds"] = parse_time_to_seconds(tmp["quote_time"])

    for quote_date, day in tmp.groupby("quote_date", sort=True):
        close_s = close_seconds_for_date(quote_date)

        before_close = day[
            day["_time_seconds"].notna()
            & (day["_time_seconds"] < close_s)
            & (day["gamma_flag"] != "near_expiry")
        ].copy()

        ok_rows = before_close[before_close["gamma_flag"] == "ok"].copy()

        if not ok_rows.empty:
            selected = ok_rows.sort_values("_time_seconds").iloc[-1].copy()
            selected["eod_status"] = "last_ok_before_close"
        elif not before_close.empty:
            selected = before_close.sort_values("_time_seconds").iloc[-1].copy()
            selected["eod_status"] = "last_non_near_expiry_before_close"
        else:
            selected = day.sort_values("_time_seconds").iloc[-1].copy()
            selected["eod_status"] = "fallback_last_available"

        rows.append(selected)

    out = pd.DataFrame(rows)
    out = out.drop(columns=[c for c in ["_time_seconds"] if c in out.columns])

    return out.reset_index(drop=True)


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
        raise FileNotFoundError(f"No Black76 files found in {BLACK76_DIR} for year {YEAR}.")

    return input_files


# =====================================================================
# PARTE 1: GAMMA BLACK '76, OI STOCK + VOLUME FLOW
# =====================================================================
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

    required_columns = [
        "quote_date",
        "quote_time",
        "strike",
        "option_type",
        "black76_gamma_365",
        "black76_T_365",
        "black76_r",
        "open_interest",
        "trade_volume",
    ]

    missing = [c for c in required_columns if c not in out.columns]

    if missing:
        raise ValueError(f"Colonne mancanti nel file input: {missing}")

    out["_strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["_gamma"] = pd.to_numeric(out["black76_gamma_365"], errors="coerce")
    out["_tau"] = pd.to_numeric(out["black76_T_365"], errors="coerce")
    out["_r"] = pd.to_numeric(out["black76_r"], errors="coerce")
    out["_open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce")
    out["_trade_volume"] = pd.to_numeric(out["trade_volume"], errors="coerce")
    out["_option_type"] = normalize_option_type(out["option_type"])
    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])

    if "black76_F_365" in out.columns:
        out["_F"] = pd.to_numeric(out["black76_F_365"], errors="coerce")
    else:
        out["_F"] = np.nan

    # flag di IV al limite del solver: queste righe vengono escluse dal gamma e contate
    if "black76_iv_at_bound_365" in out.columns:
        out["_iv_at_bound"] = pd.to_numeric(
            out["black76_iv_at_bound_365"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["_iv_at_bound"] = 0

    if "black76_solver_success_365" in out.columns:
        out["_solver_success"] = pd.to_numeric(
            out["black76_solver_success_365"],
            errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["_solver_success"] = np.where(
            out["_gamma"].notna()
            & out["_tau"].notna()
            & out["_r"].notna()
            & np.isfinite(out["_gamma"])
            & np.isfinite(out["_tau"])
            & np.isfinite(out["_r"])
            & (out["_gamma"] > 0)
            & (out["_tau"] > 0),
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

    return out


def compute_gamma_for_timestamp(group: pd.DataFrame) -> dict:
    quote_date = str(group["quote_date"].iloc[0])
    quote_time = str(group["quote_time"].iloc[0])

    tau = safe_median(group["_tau"])
    r = safe_median(group["_r"])
    F = safe_median(group["_F"])

    base_result = {
        "quote_date": quote_date,
        "quote_time": quote_time,
        "tau": tau,
        "F": F,
        "r": r,
        "n_strikes_gamma": 0,
        "n_rows_gamma": 0,
        "n_oi_positive_strikes": 0,
        "n_oi_positive_rows": 0,
        "n_vol_positive_strikes": 0,
        "n_vol_positive_rows": 0,
        "g_oi_raw": np.nan,
        "h_oi": np.nan,
        "h_oi_contract": np.nan,
        "g_vol_raw": np.nan,
        "h_vol": np.nan,
        "h_vol_contract": np.nan,
        "g_vol_delta": np.nan,
        "n_iv_at_bound_oi": 0,
        "share_gamma_oi_at_bound": np.nan,
        "n_iv_at_bound_dropped": 0,
        "share_gamma_oi_dropped": np.nan,
        "low_oi_flag": 0,
        "low_vol_flag": 0,
        "missing_r_flag": 0,
        "gamma_flag": "",
    }

    if not np.isfinite(tau) or tau <= 0:
        base_result["gamma_flag"] = "near_expiry"
        return base_result

    if tau < NEAR_EXPIRY_TAU:
        base_result["gamma_flag"] = "near_expiry"
        return base_result

    if not np.isfinite(r):
        base_result["gamma_flag"] = "missing_r"
        base_result["missing_r_flag"] = 1
        return base_result

    valid = group[
        (group["_solver_success"] == 1)
        & (group["_near_expiry"] == 0)
        & group["_strike"].notna()
        & group["_gamma"].notna()
        & group["_tau"].notna()
        & group["_r"].notna()
        & np.isfinite(group["_strike"])
        & np.isfinite(group["_gamma"])
        & np.isfinite(group["_tau"])
        & np.isfinite(group["_r"])
        & (group["_strike"] > 0)
        & (group["_gamma"] > 0)
        & (group["_tau"] > 0)
        & (group["_option_type"].isin(["C", "P"]))
    ].copy()

    if valid.empty:
        base_result["gamma_flag"] = "low_oi"
        base_result["low_oi_flag"] = 1
        base_result["low_vol_flag"] = 1
        return base_result

    # Esclusione delle righe con IV al limite del solver, come nella MFIV: il loro peso e'
    # piccolo, ma cresce lungo la sessione e nelle giornate volatili, quindi non e' rumore.
    # Si registrano il numero di righe escluse e la loro quota di gamma*OI.
    if "_iv_at_bound" in valid.columns:
        dropped = valid[valid["_iv_at_bound"] == 1]
        kept_oi = valid[(valid["_iv_at_bound"] != 1) & (valid["_open_interest"] > 0)]
        d_oi = dropped[dropped["_open_interest"] > 0]

        base_result["n_iv_at_bound_dropped"] = int(len(d_oi))

        num = float((d_oi["_gamma"] * d_oi["_open_interest"]).sum())
        den = num + float((kept_oi["_gamma"] * kept_oi["_open_interest"]).sum())

        base_result["share_gamma_oi_dropped"] = float(num / den) if den > 0 else 0.0

        valid = valid[valid["_iv_at_bound"] != 1].copy()

    if valid.empty:
        base_result["gamma_flag"] = "low_oi"
        base_result["low_oi_flag"] = 1
        base_result["low_vol_flag"] = 1
        return base_result

    valid_oi = valid[
        valid["_open_interest"].notna()
        & np.isfinite(valid["_open_interest"])
        & (valid["_open_interest"] > 0)
    ].copy()

    valid_vol = valid[
        valid["_trade_volume"].notna()
        & np.isfinite(valid["_trade_volume"])
        & (valid["_trade_volume"] > 0)
    ].copy()

    base_result["n_strikes_gamma"] = int(valid["_strike"].nunique())
    base_result["n_rows_gamma"] = int(len(valid))
    base_result["n_oi_positive_strikes"] = int(valid_oi["_strike"].nunique())
    base_result["n_oi_positive_rows"] = int(len(valid_oi))
    base_result["n_vol_positive_strikes"] = int(valid_vol["_strike"].nunique())
    base_result["n_vol_positive_rows"] = int(len(valid_vol))

    # Nessuno sconto aggiuntivo: il gamma di Black '76 contiene gia' il fattore exp(-r*tau).

    if not valid_oi.empty:
        valid_oi["_expo"] = valid_oi["_gamma"] * valid_oi["_open_interest"]
        expo_sum = valid_oi["_expo"].sum()

        if np.isfinite(expo_sum) and expo_sum > 0:
            base_result["g_oi_raw"] = float(expo_sum)

            by_strike = valid_oi.groupby("_strike")["_expo"].sum()

            h_strike = herfindahl_from_exposure(by_strike)
            h_contract = herfindahl_from_exposure(valid_oi["_expo"])

            base_result["h_oi"] = h_strike if HERFINDAHL_UNIT == "strike" else h_contract
            base_result["h_oi_contract"] = h_contract

            # valid non contiene piu' righe sature: queste due colonne restano a zero, le righe
            # escluse sono in n_iv_at_bound_dropped e share_gamma_oi_dropped
            base_result["n_iv_at_bound_oi"] = 0
            base_result["share_gamma_oi_at_bound"] = 0.0

    if not valid_vol.empty:
        valid_vol["_expo"] = valid_vol["_gamma"] * valid_vol["_trade_volume"]
        expo_sum_v = valid_vol["_expo"].sum()

        if np.isfinite(expo_sum_v) and expo_sum_v > 0:
            base_result["g_vol_raw"] = float(expo_sum_v)

            by_strike_v = valid_vol.groupby("_strike")["_expo"].sum()

            h_strike_v = herfindahl_from_exposure(by_strike_v)
            h_contract_v = herfindahl_from_exposure(valid_vol["_expo"])

            base_result["h_vol"] = h_strike_v if HERFINDAHL_UNIT == "strike" else h_contract_v
            base_result["h_vol_contract"] = h_contract_v

    low_oi = base_result["n_oi_positive_strikes"] < MIN_POSITIVE_STRIKES
    low_vol = base_result["n_vol_positive_strikes"] < MIN_POSITIVE_STRIKES

    base_result["low_oi_flag"] = int(low_oi)
    base_result["low_vol_flag"] = int(low_vol)

    if low_oi:
        base_result["gamma_flag"] = "low_oi"
    elif low_vol:
        base_result["gamma_flag"] = "low_vol"
    else:
        base_result["gamma_flag"] = "ok"

    return base_result


def add_volume_delta(intraday: pd.DataFrame) -> pd.DataFrame:
    """
    g_vol_delta: variazione di g_vol_raw da un minuto al successivo, per giornata.
    trade_volume nel dato CBOE e' volume per intervallo, non cumulato: g_vol_raw e'
    gia' un flusso (turnover) e g_vol_delta ne misura la variazione. I diagnostici
    sulla monotonia di g_vol sono quindi solo statistiche descrittive.
    """
    out = intraday.copy()
    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])
    out["g_vol_raw"] = pd.to_numeric(out["g_vol_raw"], errors="coerce")
    out = out.sort_values(["quote_date", "_time_seconds"]).reset_index(drop=True)
    out["g_vol_delta"] = out.groupby("quote_date")["g_vol_raw"].diff()

    first_mask = out.groupby("quote_date").cumcount() == 0
    out.loc[first_mask, "g_vol_delta"] = np.nan

    out = out.drop(columns=["_time_seconds"])
    return out


def process_one_day(input_file: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(input_file, dtype=str, keep_default_na=False)
    df = prepare_input_frame(df_raw)

    grouped = df.groupby(["quote_date", "quote_time"], sort=True, dropna=False)

    rows = [compute_gamma_for_timestamp(g) for _, g in grouped]

    out = pd.DataFrame(rows)
    out = add_volume_delta(out)

    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])
    out = out.sort_values(["quote_date", "_time_seconds"]).drop(columns=["_time_seconds"])

    return out


def build_midday_snapshot(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_seconds = time_string_to_seconds(SNAPSHOT_TIME)

    tmp = intraday.copy()
    tmp["_time_seconds"] = parse_time_to_seconds(tmp["quote_time"])
    tmp["_distance_to_snapshot"] = (tmp["_time_seconds"] - target_seconds).abs()

    for quote_date, day in tmp.groupby("quote_date", sort=True):
        exact_ok = day[
            (day["quote_time"] == SNAPSHOT_TIME)
            & (day["gamma_flag"] == "ok")
        ].copy()

        if not exact_ok.empty:
            selected = exact_ok.iloc[0].copy()
            selected["snapshot_status"] = "exact_1200_ok"
            selected["seconds_from_1200"] = 0
            rows.append(selected)
            continue

        nearby_ok = day[
            (day["gamma_flag"] == "ok")
            & day["_distance_to_snapshot"].notna()
            & (day["_distance_to_snapshot"] <= SNAPSHOT_TOLERANCE_SECONDS)
        ].copy()

        if not nearby_ok.empty:
            nearby_ok = nearby_ok.sort_values("_distance_to_snapshot")
            selected = nearby_ok.iloc[0].copy()
            selected["snapshot_status"] = "nearest_ok_within_5min"
            selected["seconds_from_1200"] = int(selected["_time_seconds"] - target_seconds)
            rows.append(selected)
            continue

        fallback = day[
            day["_distance_to_snapshot"].notna()
            & (day["_distance_to_snapshot"] <= SNAPSHOT_TOLERANCE_SECONDS)
        ].copy()

        if not fallback.empty:
            fallback = fallback.sort_values("_distance_to_snapshot")
            selected = fallback.iloc[0].copy()
            selected["snapshot_status"] = "nearest_not_ok_within_5min"
            selected["seconds_from_1200"] = int(selected["_time_seconds"] - target_seconds)
            rows.append(selected)
            continue

        selected = pd.Series({
            "quote_date": quote_date,
            "quote_time": "",
            "gamma_flag": "missing_snapshot",
            "snapshot_status": "missing_ok_within_5min",
            "seconds_from_1200": np.nan,
        })
        rows.append(selected)

    out = pd.DataFrame(rows)

    drop_cols = ["_time_seconds", "_distance_to_snapshot"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out.reset_index(drop=True)


def build_diagnostics(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for quote_date, day in intraday.groupby("quote_date", sort=True):
        h_oi = pd.to_numeric(day["h_oi"], errors="coerce")
        h_oi_c = pd.to_numeric(day["h_oi_contract"], errors="coerce")
        h_vol = pd.to_numeric(day["h_vol"], errors="coerce")
        h_vol_c = pd.to_numeric(day["h_vol_contract"], errors="coerce")
        g_oi = pd.to_numeric(day["g_oi_raw"], errors="coerce")
        g_vol = pd.to_numeric(day["g_vol_raw"], errors="coerce")
        g_vol_delta = pd.to_numeric(day["g_vol_delta"], errors="coerce")
        n_strikes = pd.to_numeric(day["n_strikes_gamma"], errors="coerce")
        n_oi_strikes = pd.to_numeric(day["n_oi_positive_strikes"], errors="coerce")
        n_oi_rows = pd.to_numeric(day["n_oi_positive_rows"], errors="coerce")
        n_vol_strikes = pd.to_numeric(day["n_vol_positive_strikes"], errors="coerce")
        n_bound = pd.to_numeric(day["n_iv_at_bound_dropped"], errors="coerce")
        share_bound = pd.to_numeric(day["share_gamma_oi_dropped"], errors="coerce")

        valid_h_oi = h_oi.dropna()
        valid_delta = g_vol_delta.dropna()
        negative_delta = valid_delta[valid_delta < -1e-12]

        if valid_h_oi.empty:
            ratio = np.nan
        else:
            med = valid_h_oi.median()
            ratio = float(valid_h_oi.iloc[-1] / med) if np.isfinite(med) and med > 0 else np.nan

        rows.append({
            "quote_date": quote_date,
            "n_intraday_rows": len(day),
            "n_ok": int((day["gamma_flag"] == "ok").sum()),
            "n_low_oi": int((day["gamma_flag"] == "low_oi").sum()),
            "n_low_vol": int((day["gamma_flag"] == "low_vol").sum()),
            "n_missing_r": int((day["gamma_flag"] == "missing_r").sum()),
            "n_near_expiry": int((day["gamma_flag"] == "near_expiry").sum()),
            "n_strikes_gamma_min": int(n_strikes.min()) if n_strikes.notna().any() else 0,
            "n_strikes_gamma_median": float(n_strikes.median()) if n_strikes.notna().any() else np.nan,
            "n_strikes_gamma_max": int(n_strikes.max()) if n_strikes.notna().any() else 0,
            "n_oi_positive_strikes_min": int(n_oi_strikes.min()) if n_oi_strikes.notna().any() else 0,
            "n_oi_positive_strikes_median": float(n_oi_strikes.median()) if n_oi_strikes.notna().any() else np.nan,
            "n_oi_positive_strikes_max": int(n_oi_strikes.max()) if n_oi_strikes.notna().any() else 0,
            # contratti per strike: quanto conta l'unita' di aggregazione del Herfindahl
            "contracts_per_strike_median": float((n_oi_rows / n_oi_strikes).median())
                if n_oi_strikes.notna().any() and (n_oi_strikes > 0).any() else np.nan,
            "n_vol_positive_strikes_min": int(n_vol_strikes.min()) if n_vol_strikes.notna().any() else 0,
            "n_vol_positive_strikes_median": float(n_vol_strikes.median()) if n_vol_strikes.notna().any() else np.nan,
            "n_vol_positive_strikes_max": int(n_vol_strikes.max()) if n_vol_strikes.notna().any() else 0,
            "g_oi_start": g_oi.dropna().iloc[0] if not g_oi.dropna().empty else np.nan,
            "g_oi_end": g_oi.dropna().iloc[-1] if not g_oi.dropna().empty else np.nan,
            "g_oi_min": g_oi.min(),
            "g_oi_median": g_oi.median(),
            "g_oi_max": g_oi.max(),
            "h_oi_min": h_oi.min(),
            "h_oi_median": h_oi.median(),
            "h_oi_max": h_oi.max(),
            # versione per contratto, per confronto
            "h_oi_contract_median": h_oi_c.median(),
            "h_oi_strike_over_contract": float(h_oi.median() / h_oi_c.median())
                if np.isfinite(h_oi_c.median()) and h_oi_c.median() > 0 else np.nan,
            "h_oi_close_to_median_ratio": ratio,
            "h_vol_min": h_vol.min(),
            "h_vol_median": h_vol.median(),
            "h_vol_max": h_vol.max(),
            "h_vol_contract_median": h_vol_c.median(),
            "g_vol_start": g_vol.dropna().iloc[0] if not g_vol.dropna().empty else np.nan,
            "g_vol_end": g_vol.dropna().iloc[-1] if not g_vol.dropna().empty else np.nan,
            "g_vol_min": g_vol.min(),
            "g_vol_median": g_vol.median(),
            "g_vol_max": g_vol.max(),
            # solo descrittive: il volume e' per intervallo, non cumulato
            "g_vol_delta_negative_count": int(len(negative_delta)),
            "g_vol_delta_min": g_vol_delta.min(),
            "g_vol_delta_median": g_vol_delta.median(),
            "g_vol_delta_max": g_vol_delta.max(),
            "g_vol_monotonic_non_decreasing": int(len(negative_delta) == 0),
            # righe con IV satura escluse e loro quota di gamma*OI
            "n_iv_at_bound_oi_median": float(n_bound.median()) if n_bound.notna().any() else np.nan,
            "share_gamma_oi_at_bound_max": float(share_bound.max()) if share_bound.notna().any() else np.nan,
        })

    return pd.DataFrame(rows)


def build_summary(intraday, midday_snapshot, end_of_session, diagnostics) -> pd.DataFrame:
    eod = end_of_session.copy()
    mid = midday_snapshot.copy()

    for col in ["h_oi", "h_vol", "g_oi_raw", "g_vol_raw", "h_oi_contract"]:
        if col in eod.columns:
            eod[col] = pd.to_numeric(eod[col], errors="coerce")
        if col in mid.columns:
            mid[col] = pd.to_numeric(mid[col], errors="coerce")

    eod_c = eod[["h_oi", "h_vol"]].dropna()
    mid_c = mid[["h_oi", "h_vol"]].dropna() if "h_oi" in mid.columns else pd.DataFrame()

    share_bound = pd.to_numeric(
        intraday["share_gamma_oi_dropped"], errors="coerce"
    ).dropna()

    summary = {
        "year": str(YEAR),
        "herfindahl_unit": HERFINDAHL_UNIT,
        "n_intraday_rows": len(intraday),
        "n_midday_rows": len(midday_snapshot),
        "n_end_of_session_rows": len(end_of_session),
        "n_days_diagnostics": len(diagnostics),
        "intraday_ok_rows": int((intraday["gamma_flag"] == "ok").sum()),
        "intraday_low_oi_rows": int((intraday["gamma_flag"] == "low_oi").sum()),
        "intraday_low_vol_rows": int((intraday["gamma_flag"] == "low_vol").sum()),
        "intraday_missing_r_rows": int((intraday["gamma_flag"] == "missing_r").sum()),
        "intraday_near_expiry_rows": int((intraday["gamma_flag"] == "near_expiry").sum()),
        "eod_ok_days": int((end_of_session["gamma_flag"] == "ok").sum()),
        "midday_ok_days": int((midday_snapshot["gamma_flag"] == "ok").sum()),
        "eod_h_oi_mean": eod["h_oi"].mean(),
        "eod_h_oi_median": eod["h_oi"].median(),
        "eod_h_oi_contract_median": eod["h_oi_contract"].median() if "h_oi_contract" in eod.columns else np.nan,
        "eod_h_vol_mean": eod["h_vol"].mean(),
        "eod_h_vol_median": eod["h_vol"].median(),
        "eod_h_oi_h_vol_corr": float(eod_c["h_oi"].corr(eod_c["h_vol"])) if len(eod_c) >= 2 else np.nan,
        "midday_h_oi_mean": mid["h_oi"].mean() if "h_oi" in mid.columns else np.nan,
        "midday_h_oi_median": mid["h_oi"].median() if "h_oi" in mid.columns else np.nan,
        "midday_h_vol_mean": mid["h_vol"].mean() if "h_vol" in mid.columns else np.nan,
        "midday_h_vol_median": mid["h_vol"].median() if "h_vol" in mid.columns else np.nan,
        "midday_h_oi_h_vol_corr": float(mid_c["h_oi"].corr(mid_c["h_vol"])) if len(mid_c) >= 2 else np.nan,
        "share_gamma_oi_at_bound_max": float(share_bound.max()) if not share_bound.empty else np.nan,
        "share_gamma_oi_at_bound_mean": float(share_bound.mean()) if not share_bound.empty else np.nan,
    }

    return pd.DataFrame([summary])


def gamma_herfindahl_black76():
    input_files = select_input_files()

    print("=" * 130)
    print("GAMMA FULL YEAR - OI STOCK + VOLUME FLOW")
    print("=" * 130)
    print(f"Input folder:              {BLACK76_DIR}")
    print(f"Output folder:             {OUTPUT_DIR}")
    print(f"Files found:               {len(input_files)}")
    print(f"Herfindahl unit:           {HERFINDAHL_UNIT}")
    print(f"Near-expiry cutoff:        tau < {NEAR_EXPIRY_MINUTES} minutes")
    print(f"Minimum positive strikes:  {MIN_POSITIVE_STRIKES}")
    print(f"Snapshot time:             {SNAPSHOT_TIME} +/- {SNAPSHOT_TOLERANCE_MINUTES} min")
    print(f"Close per day:             default {REGULAR_CLOSE_SECONDS // 3600}:00, "
          f"early close {sorted(EARLY_CLOSE_SECONDS)}")
    print("Discounting:               nessuno, il gamma di Black '76 contiene gia' exp(-r*tau)")
    print("=" * 130)

    global_start = time.time()

    all_intraday = []

    for index, input_file in enumerate(input_files, start=1):
        date_string = file_to_date(input_file)

        day_intraday = process_one_day(input_file)
        all_intraday.append(day_intraday)

        if WRITE_DAILY_INTRADAY_FILES:
            daily_output_file = DAILY_INTRADAY_DIR / f"04_gamma_intraday_{date_string}.csv"

            if daily_output_file.exists() and OVERWRITE_EXISTING:
                daily_output_file.unlink()

            day_intraday.to_csv(daily_output_file, index=False)

        h_oi = pd.to_numeric(day_intraday["h_oi"], errors="coerce")
        h_oi_c = pd.to_numeric(day_intraday["h_oi_contract"], errors="coerce")
        h_vol = pd.to_numeric(day_intraday["h_vol"], errors="coerce")
        sb = pd.to_numeric(day_intraday["share_gamma_oi_dropped"], errors="coerce")

        eta = (time.time() - global_start) / index * (len(input_files) - index) / 60.0

        print(
            f"[{index:3d}/{len(input_files)}] {date_string} | "
            f"rows={len(day_intraday):,} | "
            f"ok={(day_intraday['gamma_flag'] == 'ok').sum():,} | "
            f"H_OI={h_oi.median():.6f} | "
            f"H_OI_contr={h_oi_c.median():.6f} | "
            f"H_VOL={h_vol.median():.6f} | "
            f"bound_share_max={sb.max():.2e} | "
            f"ETA {eta:5.1f} min",
            flush=True
        )

    intraday_all = pd.concat(all_intraday, ignore_index=True)

    intraday_output_file = OUTPUT_DIR / f"04_gamma_intraday_{YEAR}_all.csv"
    midday_output_file = OUTPUT_DIR / f"04_gamma_midday_snapshot_{YEAR}_all.csv"
    eod_output_file = OUTPUT_DIR / f"04_gamma_end_of_session_{YEAR}_all.csv"
    diagnostics_output_file = OUTPUT_DIR / f"04_gamma_diagnostics_{YEAR}_all.csv"
    summary_output_file = OUTPUT_DIR / f"04_gamma_summary_{YEAR}_all.csv"

    if WRITE_COMBINED_INTRADAY_FILE:
        if intraday_output_file.exists() and OVERWRITE_EXISTING:
            intraday_output_file.unlink()
        intraday_all.to_csv(intraday_output_file, index=False)

    midday_snapshot = build_midday_snapshot(intraday_all)
    if midday_output_file.exists() and OVERWRITE_EXISTING:
        midday_output_file.unlink()
    midday_snapshot.to_csv(midday_output_file, index=False)

    end_of_session = build_end_of_session(intraday_all)
    if eod_output_file.exists() and OVERWRITE_EXISTING:
        eod_output_file.unlink()
    end_of_session.to_csv(eod_output_file, index=False)

    diagnostics = build_diagnostics(intraday_all)
    if diagnostics_output_file.exists() and OVERWRITE_EXISTING:
        diagnostics_output_file.unlink()
    diagnostics.to_csv(diagnostics_output_file, index=False)

    summary = build_summary(intraday_all, midday_snapshot, end_of_session, diagnostics)
    if summary_output_file.exists() and OVERWRITE_EXISTING:
        summary_output_file.unlink()
    summary.to_csv(summary_output_file, index=False)

    total_elapsed = time.time() - global_start

    h_oi_all = pd.to_numeric(intraday_all["h_oi"], errors="coerce")
    h_oi_c_all = pd.to_numeric(intraday_all["h_oi_contract"], errors="coerce")
    sb_all = pd.to_numeric(intraday_all["share_gamma_oi_dropped"], errors="coerce")

    print()
    print("=" * 130)
    print("COMPLETED GAMMA FULL YEAR")
    print("=" * 130)
    print(f"Files processed:                 {len(input_files)}")
    print(f"Intraday rows:                   {len(intraday_all):,}")
    print(f"Total ok rows:                   {int((intraday_all['gamma_flag'] == 'ok').sum()):,}")
    print(f"Total low_oi rows:               {int((intraday_all['gamma_flag'] == 'low_oi').sum()):,}")
    print(f"Total low_vol rows:              {int((intraday_all['gamma_flag'] == 'low_vol').sum()):,}")
    print(f"Total near_expiry rows:          {int((intraday_all['gamma_flag'] == 'near_expiry').sum()):,}")
    print()
    print("--- EFFETTO DELL'UNITA' DI AGGREGAZIONE ---")
    print(f"H_OI per strike, mediana:        {h_oi_all.median():.6f}")
    print(f"H_OI per contratto, mediana:     {h_oi_c_all.median():.6f}")
    print(f"rapporto strike/contratto:       {h_oi_all.median() / h_oi_c_all.median():.4f}")
    print(f"n efficace di strike:            {1 / h_oi_all.median():.1f} contro "
          f"{1 / h_oi_c_all.median():.1f}")
    print()
    print("--- RIGHE CON IV SATURA ESCLUSE ---")
    print(f"quota di gamma*OI esclusa: max  {sb_all.max():.3e}")
    print(f"                           media {sb_all.mean():.3e}")
    print("Prima della correzione queste righe restavano nel calcolo. La loro")
    print("incidenza cresce lungo la sessione e si concentra nelle giornate")
    print("volatili, quindi era correlata con ora e volatilita', non rumore.")
    print()
    print(f"Elapsed time:                    {total_elapsed / 60:.2f} minutes")
    print(f"Output folder:                   {OUTPUT_DIR}")
    print("=" * 130)
    print()
    print("Summary:")
    print(summary.to_string(index=False))


# =====================================================================
# PARTE 2: GAMMA DEL VENDOR, STESSE MISURE
# =====================================================================
# Stesse misure della parte 1 con il gamma del vendor (colonna "gamma") al posto del gamma
# Black '76, come seconda fonte; anche qui il Herfindahl e' per strike (HERFINDAHL_UNIT).

# =====================================================================
# UNIVERSO DI RIGHE
# =====================================================================
# Stesso filtro sul successo del solver Black '76 della parte 1, cosi' le due serie usano le
# stesse righe e sono confrontabili; il gamma resta quello del vendor. La versione senza filtro
# resta come diagnostica (h_oi_cboe_nosolver).
APPLY_SOLVER_SUCCESS_FILTER = True

# Il vendor arrotonda il gamma a quattro decimali: sotto 5e-05 diventa zero e il filtro "> 0"
# lo scarta, quindi la serie vendor perde le ali, che il gamma Black '76 conserva. Gli strike
# persi in questo modo vengono contati.
VENDOR_GAMMA_ROUNDING = 1e-4


def prepare_input_frame_vendor(df: pd.DataFrame) -> pd.DataFrame:
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

    required_columns = [
        "quote_date",
        "quote_time",
        "strike",
        "option_type",
        "gamma",
        "black76_T_365",
        "black76_r",
        "open_interest",
        "trade_volume",
    ]

    missing = [c for c in required_columns if c not in out.columns]

    if missing:
        raise ValueError(f"Colonne mancanti nel file input: {missing}")

    out["_strike"] = pd.to_numeric(out["strike"], errors="coerce")

    # gamma del vendor (colonna "gamma"), non black76_gamma_365
    out["_gamma_cboe"] = pd.to_numeric(out["gamma"], errors="coerce")

    # gamma Black '76, usato solo per contare gli strike che il vendor azzera
    # per arrotondamento
    if "black76_gamma_365" in out.columns:
        out["_gamma_own"] = pd.to_numeric(out["black76_gamma_365"], errors="coerce")
    else:
        out["_gamma_own"] = np.nan

    out["_tau"] = pd.to_numeric(out["black76_T_365"], errors="coerce")
    out["_r"] = pd.to_numeric(out["black76_r"], errors="coerce")
    out["_open_interest"] = pd.to_numeric(out["open_interest"], errors="coerce")
    out["_trade_volume"] = pd.to_numeric(out["trade_volume"], errors="coerce")
    out["_option_type"] = normalize_option_type(out["option_type"])
    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])

    if "black76_F_365" in out.columns:
        out["_F"] = pd.to_numeric(out["black76_F_365"], errors="coerce")
    else:
        out["_F"] = np.nan

    if "black76_iv_at_bound_365" in out.columns:
        out["_iv_at_bound"] = pd.to_numeric(
            out["black76_iv_at_bound_365"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["_iv_at_bound"] = 0

    if "black76_solver_success_365" in out.columns:
        out["_solver_success"] = pd.to_numeric(
            out["black76_solver_success_365"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        out["_solver_success"] = 1

    if "black76_near_expiry_flag_365" in out.columns:
        out["_near_expiry"] = pd.to_numeric(
            out["black76_near_expiry_flag_365"], errors="coerce"
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


def compute_gamma_for_timestamp_vendor(group: pd.DataFrame) -> dict:
    quote_date = str(group["quote_date"].iloc[0])
    quote_time = str(group["quote_time"].iloc[0])

    tau = safe_median(group["_tau"])
    r = safe_median(group["_r"])
    F = safe_median(group["_F"])

    base_result = {
        "quote_date": quote_date,
        "quote_time": quote_time,
        "tau": tau,
        "F": F,
        "r": r,
        "gamma_source": "CBOE_gamma",
        "n_strikes_gamma": 0,
        "n_rows_gamma": 0,
        "n_oi_positive_strikes": 0,
        "n_oi_positive_rows": 0,
        "n_vol_positive_strikes": 0,
        "n_vol_positive_rows": 0,
        "g_oi_raw_cboe": np.nan,
        "h_oi_cboe": np.nan,
        "h_oi_cboe_contract": np.nan,
        "h_oi_cboe_nosolver": np.nan,
        "g_vol_raw_cboe": np.nan,
        "h_vol_cboe": np.nan,
        "h_vol_cboe_contract": np.nan,
        "g_vol_delta_cboe": np.nan,
        # differenza di universo fra le due fonti
        "n_strikes_own_gamma_positive": 0,
        "n_strikes_lost_to_vendor_rounding": 0,
        "share_oi_lost_to_vendor_rounding": np.nan,
        "low_oi_flag": 0,
        "low_vol_flag": 0,
        "missing_r_flag": 0,
        "no_valid_cboe_gamma_flag": 0,
        "gamma_flag": "",
    }

    if not np.isfinite(tau) or tau <= 0:
        base_result["gamma_flag"] = "near_expiry"
        return base_result

    if tau < NEAR_EXPIRY_TAU:
        base_result["gamma_flag"] = "near_expiry"
        return base_result

    if not np.isfinite(r):
        base_result["gamma_flag"] = "missing_r"
        base_result["missing_r_flag"] = 1
        return base_result

    common = (
        (group["_near_expiry"] == 0)
        & group["_strike"].notna()
        & group["_tau"].notna()
        & group["_r"].notna()
        & np.isfinite(group["_strike"])
        & np.isfinite(group["_tau"])
        & np.isfinite(group["_r"])
        & (group["_strike"] > 0)
        & (group["_tau"] > 0)
        & (group["_option_type"].isin(["C", "P"]))
    )

    solver_ok = (group["_solver_success"] == 1) if APPLY_SOLVER_SUCCESS_FILTER else True

    gamma_ok = group["_gamma_cboe"].notna() & np.isfinite(group["_gamma_cboe"]) \
        & (group["_gamma_cboe"] > 0)

    # esclusione delle righe con IV al limite del solver, come nella parte 1
    at_bound_ok = (group["_iv_at_bound"] != 1) \
        if "_iv_at_bound" in group.columns else True

    valid = group[common & solver_ok & gamma_ok & at_bound_ok].copy()

    # universo senza filtro solver, solo per la diagnostica
    valid_nosolver = group[common & gamma_ok].copy()

    # Strike con gamma Black '76 positivo ma gamma vendor nullo per arrotondamento.
    own_positive = group[
        common
        & solver_ok
        & group["_gamma_own"].notna()
        & np.isfinite(group["_gamma_own"])
        & (group["_gamma_own"] > 0)
    ]

    lost = own_positive[
        own_positive["_gamma_cboe"].isna() | (own_positive["_gamma_cboe"] <= 0)
    ]

    base_result["n_strikes_own_gamma_positive"] = int(own_positive["_strike"].nunique())
    base_result["n_strikes_lost_to_vendor_rounding"] = int(lost["_strike"].nunique())

    oi_all = pd.to_numeric(own_positive["_open_interest"], errors="coerce")
    oi_lost = pd.to_numeric(lost["_open_interest"], errors="coerce")
    oi_all_sum = oi_all[oi_all > 0].sum()

    if np.isfinite(oi_all_sum) and oi_all_sum > 0:
        base_result["share_oi_lost_to_vendor_rounding"] = float(
            oi_lost[oi_lost > 0].sum() / oi_all_sum
        )

    if valid.empty:
        base_result["gamma_flag"] = "no_valid_cboe_gamma"
        base_result["no_valid_cboe_gamma_flag"] = 1
        base_result["low_oi_flag"] = 1
        base_result["low_vol_flag"] = 1
        return base_result

    valid_oi = valid[
        valid["_open_interest"].notna()
        & np.isfinite(valid["_open_interest"])
        & (valid["_open_interest"] > 0)
    ].copy()

    valid_vol = valid[
        valid["_trade_volume"].notna()
        & np.isfinite(valid["_trade_volume"])
        & (valid["_trade_volume"] > 0)
    ].copy()

    base_result["n_strikes_gamma"] = int(valid["_strike"].nunique())
    base_result["n_rows_gamma"] = int(len(valid))
    base_result["n_oi_positive_strikes"] = int(valid_oi["_strike"].nunique())
    base_result["n_oi_positive_rows"] = int(len(valid_oi))
    base_result["n_vol_positive_strikes"] = int(valid_vol["_strike"].nunique())
    base_result["n_vol_positive_rows"] = int(len(valid_vol))

    # Nessuno sconto aggiuntivo, come nella parte 1: g_oi_raw_cboe e' la somma di gamma * OI.

    if not valid_oi.empty:
        valid_oi["_expo"] = valid_oi["_gamma_cboe"] * valid_oi["_open_interest"]
        expo_sum = valid_oi["_expo"].sum()

        if np.isfinite(expo_sum) and expo_sum > 0:
            base_result["g_oi_raw_cboe"] = float(expo_sum)

            by_strike = valid_oi.groupby("_strike")["_expo"].sum()

            h_strike = herfindahl_from_exposure(by_strike)
            h_contract = herfindahl_from_exposure(valid_oi["_expo"])

            base_result["h_oi_cboe"] = h_strike if HERFINDAHL_UNIT == "strike" else h_contract
            base_result["h_oi_cboe_contract"] = h_contract

    if not valid_nosolver.empty:
        ns_oi = valid_nosolver[
            valid_nosolver["_open_interest"].notna()
            & (valid_nosolver["_open_interest"] > 0)
        ].copy()

        if not ns_oi.empty:
            ns_oi["_expo"] = ns_oi["_gamma_cboe"] * ns_oi["_open_interest"]
            base_result["h_oi_cboe_nosolver"] = herfindahl_from_exposure(
                ns_oi.groupby("_strike")["_expo"].sum()
            )

    if not valid_vol.empty:
        valid_vol["_expo"] = valid_vol["_gamma_cboe"] * valid_vol["_trade_volume"]
        expo_sum_v = valid_vol["_expo"].sum()

        if np.isfinite(expo_sum_v) and expo_sum_v > 0:
            base_result["g_vol_raw_cboe"] = float(expo_sum_v)

            by_strike_v = valid_vol.groupby("_strike")["_expo"].sum()

            h_strike_v = herfindahl_from_exposure(by_strike_v)
            h_contract_v = herfindahl_from_exposure(valid_vol["_expo"])

            base_result["h_vol_cboe"] = h_strike_v if HERFINDAHL_UNIT == "strike" else h_contract_v
            base_result["h_vol_cboe_contract"] = h_contract_v

    low_oi = base_result["n_oi_positive_strikes"] < MIN_POSITIVE_STRIKES
    low_vol = base_result["n_vol_positive_strikes"] < MIN_POSITIVE_STRIKES

    base_result["low_oi_flag"] = int(low_oi)
    base_result["low_vol_flag"] = int(low_vol)

    if low_oi:
        base_result["gamma_flag"] = "low_oi"
    elif low_vol:
        base_result["gamma_flag"] = "low_vol"
    else:
        base_result["gamma_flag"] = "ok"

    return base_result


def add_volume_delta_vendor(intraday: pd.DataFrame) -> pd.DataFrame:
    """
    Come add_volume_delta della parte 1: g_vol_delta_cboe e' la variazione di
    g_vol_raw_cboe da un minuto al successivo, per giornata.
    """
    out = intraday.copy()
    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])
    out["g_vol_raw_cboe"] = pd.to_numeric(out["g_vol_raw_cboe"], errors="coerce")
    out = out.sort_values(["quote_date", "_time_seconds"]).reset_index(drop=True)
    out["g_vol_delta_cboe"] = out.groupby("quote_date")["g_vol_raw_cboe"].diff()

    first_mask = out.groupby("quote_date").cumcount() == 0
    out.loc[first_mask, "g_vol_delta_cboe"] = np.nan

    out = out.drop(columns=["_time_seconds"])
    return out


def process_one_day_vendor(input_file: Path) -> pd.DataFrame:
    df_raw = pd.read_csv(input_file, dtype=str, keep_default_na=False)
    df = prepare_input_frame_vendor(df_raw)

    grouped = df.groupby(["quote_date", "quote_time"], sort=True, dropna=False)

    rows = [compute_gamma_for_timestamp_vendor(g) for _, g in grouped]

    out = pd.DataFrame(rows)
    out = add_volume_delta_vendor(out)

    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])
    out = out.sort_values(["quote_date", "_time_seconds"]).drop(columns=["_time_seconds"])

    return out


def build_midday_snapshot_vendor(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_seconds = time_string_to_seconds(SNAPSHOT_TIME)

    tmp = intraday.copy()
    tmp["_time_seconds"] = parse_time_to_seconds(tmp["quote_time"])
    tmp["_distance_to_snapshot"] = (tmp["_time_seconds"] - target_seconds).abs()

    for quote_date, day in tmp.groupby("quote_date", sort=True):
        exact_ok = day[
            (day["quote_time"] == SNAPSHOT_TIME)
            & (day["gamma_flag"] == "ok")
        ].copy()

        if not exact_ok.empty:
            selected = exact_ok.iloc[0].copy()
            selected["snapshot_status"] = "exact_1200_ok"
            selected["seconds_from_1200"] = 0
            rows.append(selected)
            continue

        nearby_ok = day[
            (day["gamma_flag"] == "ok")
            & day["_distance_to_snapshot"].notna()
            & (day["_distance_to_snapshot"] <= SNAPSHOT_TOLERANCE_SECONDS)
        ].copy()

        if not nearby_ok.empty:
            nearby_ok = nearby_ok.sort_values("_distance_to_snapshot")
            selected = nearby_ok.iloc[0].copy()
            selected["snapshot_status"] = "nearest_ok_within_5min"
            selected["seconds_from_1200"] = int(selected["_time_seconds"] - target_seconds)
            rows.append(selected)
            continue

        fallback = day[
            day["_distance_to_snapshot"].notna()
            & (day["_distance_to_snapshot"] <= SNAPSHOT_TOLERANCE_SECONDS)
        ].copy()

        if not fallback.empty:
            fallback = fallback.sort_values("_distance_to_snapshot")
            selected = fallback.iloc[0].copy()
            selected["snapshot_status"] = "nearest_not_ok_within_5min"
            selected["seconds_from_1200"] = int(selected["_time_seconds"] - target_seconds)
            rows.append(selected)
            continue

        rows.append(pd.Series({
            "quote_date": quote_date,
            "quote_time": "",
            "gamma_source": "CBOE_gamma",
            "gamma_flag": "missing_snapshot",
            "snapshot_status": "missing_ok_within_5min",
            "seconds_from_1200": np.nan,
        }))

    out = pd.DataFrame(rows)

    drop_cols = ["_time_seconds", "_distance_to_snapshot"]
    out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    return out.reset_index(drop=True)


def build_diagnostics_vendor(intraday: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for quote_date, day in intraday.groupby("quote_date", sort=True):
        h_oi = pd.to_numeric(day["h_oi_cboe"], errors="coerce")
        h_oi_c = pd.to_numeric(day["h_oi_cboe_contract"], errors="coerce")
        h_oi_ns = pd.to_numeric(day["h_oi_cboe_nosolver"], errors="coerce")
        h_vol = pd.to_numeric(day["h_vol_cboe"], errors="coerce")
        g_oi = pd.to_numeric(day["g_oi_raw_cboe"], errors="coerce")
        g_vol = pd.to_numeric(day["g_vol_raw_cboe"], errors="coerce")
        g_vol_delta = pd.to_numeric(day["g_vol_delta_cboe"], errors="coerce")
        n_strikes = pd.to_numeric(day["n_strikes_gamma"], errors="coerce")
        n_oi_strikes = pd.to_numeric(day["n_oi_positive_strikes"], errors="coerce")
        n_own = pd.to_numeric(day["n_strikes_own_gamma_positive"], errors="coerce")
        n_lost = pd.to_numeric(day["n_strikes_lost_to_vendor_rounding"], errors="coerce")
        sh_lost = pd.to_numeric(day["share_oi_lost_to_vendor_rounding"], errors="coerce")
        n_vol_strikes = pd.to_numeric(day["n_vol_positive_strikes"], errors="coerce")

        valid_h_oi = h_oi.dropna()
        valid_delta = g_vol_delta.dropna()
        negative_delta = valid_delta[valid_delta < -1e-12]

        if valid_h_oi.empty:
            ratio = np.nan
        else:
            med = valid_h_oi.median()
            ratio = float(valid_h_oi.iloc[-1] / med) if np.isfinite(med) and med > 0 else np.nan

        rows.append({
            "quote_date": quote_date,
            "gamma_source": "CBOE_gamma",
            "n_intraday_rows": len(day),
            "n_ok": int((day["gamma_flag"] == "ok").sum()),
            "n_low_oi": int((day["gamma_flag"] == "low_oi").sum()),
            "n_low_vol": int((day["gamma_flag"] == "low_vol").sum()),
            "n_missing_r": int((day["gamma_flag"] == "missing_r").sum()),
            "n_no_valid_cboe_gamma": int((day["gamma_flag"] == "no_valid_cboe_gamma").sum()),
            "n_near_expiry": int((day["gamma_flag"] == "near_expiry").sum()),
            "n_strikes_gamma_min": int(n_strikes.min()) if n_strikes.notna().any() else 0,
            "n_strikes_gamma_median": float(n_strikes.median()) if n_strikes.notna().any() else np.nan,
            "n_strikes_gamma_max": int(n_strikes.max()) if n_strikes.notna().any() else 0,
            "n_oi_positive_strikes_median": float(n_oi_strikes.median()) if n_oi_strikes.notna().any() else np.nan,
            "n_vol_positive_strikes_median": float(n_vol_strikes.median()) if n_vol_strikes.notna().any() else np.nan,
            # differenza di universo fra le due fonti
            "n_strikes_own_positive_median": float(n_own.median()) if n_own.notna().any() else np.nan,
            "n_strikes_lost_median": float(n_lost.median()) if n_lost.notna().any() else np.nan,
            "share_strikes_lost_median": float((n_lost / n_own).median())
                if n_own.notna().any() and (n_own > 0).any() else np.nan,
            "share_oi_lost_median": float(sh_lost.median()) if sh_lost.notna().any() else np.nan,
            "share_oi_lost_max": float(sh_lost.max()) if sh_lost.notna().any() else np.nan,
            "g_oi_cboe_median": g_oi.median(),
            "h_oi_cboe_min": h_oi.min(),
            "h_oi_cboe_median": h_oi.median(),
            "h_oi_cboe_max": h_oi.max(),
            "h_oi_cboe_contract_median": h_oi_c.median(),
            "h_oi_cboe_nosolver_median": h_oi_ns.median(),
            "h_oi_cboe_close_to_median_ratio": ratio,
            "h_vol_cboe_min": h_vol.min(),
            "h_vol_cboe_median": h_vol.median(),
            "h_vol_cboe_max": h_vol.max(),
            "g_vol_cboe_median": g_vol.median(),
            "g_vol_delta_cboe_negative_count": int(len(negative_delta)),
            "g_vol_delta_cboe_median": g_vol_delta.median(),
            "g_vol_cboe_monotonic_non_decreasing": int(len(negative_delta) == 0),
        })

    return pd.DataFrame(rows)


def build_summary_vendor(intraday, midday_snapshot, end_of_session, diagnostics) -> pd.DataFrame:
    eod = end_of_session.copy()
    mid = midday_snapshot.copy()

    for col in ["h_oi_cboe", "h_vol_cboe", "g_oi_raw_cboe", "g_vol_raw_cboe",
                "h_oi_cboe_contract", "h_oi_cboe_nosolver"]:
        if col in eod.columns:
            eod[col] = pd.to_numeric(eod[col], errors="coerce")
        if col in mid.columns:
            mid[col] = pd.to_numeric(mid[col], errors="coerce")

    eod_c = eod[["h_oi_cboe", "h_vol_cboe"]].dropna()
    mid_c = mid[["h_oi_cboe", "h_vol_cboe"]].dropna() if "h_oi_cboe" in mid.columns else pd.DataFrame()

    sh_lost = pd.to_numeric(intraday["share_oi_lost_to_vendor_rounding"], errors="coerce").dropna()

    summary = {
        "year": str(YEAR),
        "gamma_source": "CBOE_gamma",
        "herfindahl_unit": HERFINDAHL_UNIT,
        "solver_success_filter": APPLY_SOLVER_SUCCESS_FILTER,
        "n_intraday_rows": len(intraday),
        "n_end_of_session_rows": len(end_of_session),
        "intraday_ok_rows": int((intraday["gamma_flag"] == "ok").sum()),
        "intraday_low_oi_rows": int((intraday["gamma_flag"] == "low_oi").sum()),
        "intraday_low_vol_rows": int((intraday["gamma_flag"] == "low_vol").sum()),
        "intraday_no_valid_cboe_gamma_rows": int((intraday["gamma_flag"] == "no_valid_cboe_gamma").sum()),
        "intraday_near_expiry_rows": int((intraday["gamma_flag"] == "near_expiry").sum()),
        "eod_ok_days": int((end_of_session["gamma_flag"] == "ok").sum()),
        "eod_h_oi_cboe_mean": eod["h_oi_cboe"].mean(),
        "eod_h_oi_cboe_median": eod["h_oi_cboe"].median(),
        "eod_h_oi_cboe_contract_median": eod["h_oi_cboe_contract"].median()
            if "h_oi_cboe_contract" in eod.columns else np.nan,
        "eod_h_vol_cboe_mean": eod["h_vol_cboe"].mean(),
        "eod_h_vol_cboe_median": eod["h_vol_cboe"].median(),
        "eod_h_oi_h_vol_cboe_corr": float(eod_c["h_oi_cboe"].corr(eod_c["h_vol_cboe"]))
            if len(eod_c) >= 2 else np.nan,
        "midday_h_oi_h_vol_cboe_corr": float(mid_c["h_oi_cboe"].corr(mid_c["h_vol_cboe"]))
            if len(mid_c) >= 2 else np.nan,
        "share_oi_lost_to_vendor_rounding_mean": float(sh_lost.mean()) if not sh_lost.empty else np.nan,
        "share_oi_lost_to_vendor_rounding_median": float(sh_lost.median()) if not sh_lost.empty else np.nan,
        "share_oi_lost_to_vendor_rounding_max": float(sh_lost.max()) if not sh_lost.empty else np.nan,
    }

    return pd.DataFrame([summary])


def gamma_herfindahl_vendor():
    input_files = select_input_files()

    print("=" * 140)
    print("GAMMA FULL YEAR CBOE - OI STOCK + VOLUME FLOW USING VENDOR GAMMA")
    print("=" * 140)
    print(f"Input folder:              {BLACK76_DIR}")
    print(f"Output folder:             {OUTPUT_DIR}")
    print(f"Files found:               {len(input_files)}")
    print(f"Gamma source:              colonna vendor 'gamma'")
    print(f"Herfindahl unit:           {HERFINDAHL_UNIT}")
    print(f"Solver-success filter:     {APPLY_SOLVER_SUCCESS_FILTER} (stesso universo della gemella)")
    print(f"Near-expiry cutoff:        tau < {NEAR_EXPIRY_MINUTES} minutes")
    print(f"Minimum positive strikes:  {MIN_POSITIVE_STRIKES}")
    print(f"Close per day:             default {REGULAR_CLOSE_SECONDS // 3600}:00, "
          f"early close {sorted(EARLY_CLOSE_SECONDS)}")
    print("Discounting:               nessuno")
    print("=" * 140)

    global_start = time.time()

    all_intraday = []

    for index, input_file in enumerate(input_files, start=1):
        date_string = file_to_date(input_file)

        day_intraday = process_one_day_vendor(input_file)
        all_intraday.append(day_intraday)

        if WRITE_DAILY_INTRADAY_FILES:
            daily_output_file = DAILY_INTRADAY_DIR / f"04_gamma_cboe_intraday_{date_string}.csv"

            if daily_output_file.exists() and OVERWRITE_EXISTING:
                daily_output_file.unlink()

            day_intraday.to_csv(daily_output_file, index=False)

        h_oi = pd.to_numeric(day_intraday["h_oi_cboe"], errors="coerce")
        h_oi_c = pd.to_numeric(day_intraday["h_oi_cboe_contract"], errors="coerce")
        h_vol = pd.to_numeric(day_intraday["h_vol_cboe"], errors="coerce")
        sl = pd.to_numeric(day_intraday["share_oi_lost_to_vendor_rounding"], errors="coerce")

        eta = (time.time() - global_start) / index * (len(input_files) - index) / 60.0

        print(
            f"[{index:3d}/{len(input_files)}] {date_string} | "
            f"rows={len(day_intraday):,} | "
            f"ok={(day_intraday['gamma_flag'] == 'ok').sum():,} | "
            f"H_OI_cboe={h_oi.median():.6f} | "
            f"H_OI_contr={h_oi_c.median():.6f} | "
            f"H_VOL_cboe={h_vol.median():.6f} | "
            f"OI_persa={sl.median():.4f} | "
            f"ETA {eta:5.1f} min",
            flush=True
        )

    intraday_all = pd.concat(all_intraday, ignore_index=True)

    intraday_output_file = OUTPUT_DIR / f"04_gamma_cboe_intraday_{YEAR}_all.csv"
    midday_output_file = OUTPUT_DIR / f"04_gamma_cboe_midday_snapshot_{YEAR}_all.csv"
    eod_output_file = OUTPUT_DIR / f"04_gamma_cboe_end_of_session_{YEAR}_all.csv"
    diagnostics_output_file = OUTPUT_DIR / f"04_gamma_cboe_diagnostics_{YEAR}_all.csv"
    summary_output_file = OUTPUT_DIR / f"04_gamma_cboe_summary_{YEAR}_all.csv"

    if WRITE_COMBINED_INTRADAY_FILE:
        if intraday_output_file.exists() and OVERWRITE_EXISTING:
            intraday_output_file.unlink()
        intraday_all.to_csv(intraday_output_file, index=False)

    midday_snapshot = build_midday_snapshot_vendor(intraday_all)
    if midday_output_file.exists() and OVERWRITE_EXISTING:
        midday_output_file.unlink()
    midday_snapshot.to_csv(midday_output_file, index=False)

    end_of_session = build_end_of_session(intraday_all)
    if eod_output_file.exists() and OVERWRITE_EXISTING:
        eod_output_file.unlink()
    end_of_session.to_csv(eod_output_file, index=False)

    diagnostics = build_diagnostics_vendor(intraday_all)
    if diagnostics_output_file.exists() and OVERWRITE_EXISTING:
        diagnostics_output_file.unlink()
    diagnostics.to_csv(diagnostics_output_file, index=False)

    summary = build_summary_vendor(intraday_all, midday_snapshot, end_of_session, diagnostics)
    if summary_output_file.exists() and OVERWRITE_EXISTING:
        summary_output_file.unlink()
    summary.to_csv(summary_output_file, index=False)

    total_elapsed = time.time() - global_start

    h_all = pd.to_numeric(intraday_all["h_oi_cboe"], errors="coerce")
    h_c_all = pd.to_numeric(intraday_all["h_oi_cboe_contract"], errors="coerce")
    h_ns_all = pd.to_numeric(intraday_all["h_oi_cboe_nosolver"], errors="coerce")
    sl_all = pd.to_numeric(intraday_all["share_oi_lost_to_vendor_rounding"], errors="coerce")
    n_own = pd.to_numeric(intraday_all["n_strikes_own_gamma_positive"], errors="coerce")
    n_lost = pd.to_numeric(intraday_all["n_strikes_lost_to_vendor_rounding"], errors="coerce")

    print()
    print("=" * 140)
    print("COMPLETED GAMMA FULL YEAR CBOE")
    print("=" * 140)
    print(f"Files processed:                 {len(input_files)}")
    print(f"Intraday rows:                   {len(intraday_all):,}")
    print(f"Total ok rows:                   {int((intraday_all['gamma_flag'] == 'ok').sum()):,}")
    print(f"Total low_oi rows:               {int((intraday_all['gamma_flag'] == 'low_oi').sum()):,}")
    print(f"Total low_vol rows:              {int((intraday_all['gamma_flag'] == 'low_vol').sum()):,}")
    print(f"Total no_valid_cboe_gamma rows:  {int((intraday_all['gamma_flag'] == 'no_valid_cboe_gamma').sum()):,}")
    print(f"Total near_expiry rows:          {int((intraday_all['gamma_flag'] == 'near_expiry').sum()):,}")
    print()
    print("--- EFFETTO DELL'UNITA' DI AGGREGAZIONE ---")
    print(f"H_OI cboe per strike, mediana:   {h_all.median():.6f}")
    print(f"H_OI cboe per contratto:         {h_c_all.median():.6f}")
    print(f"rapporto strike/contratto:       {h_all.median() / h_c_all.median():.4f}")
    print()
    print("--- EFFETTO DEL FILTRO SOLVER (universo comune vs no) ---")
    print(f"H_OI cboe con filtro:            {h_all.median():.6f}")
    print(f"H_OI cboe senza filtro:          {h_ns_all.median():.6f}")
    print(f"scarto relativo:                 {abs(h_all.median() / h_ns_all.median() - 1) * 100:.3f}%")
    print()
    print("--- DIFFERENZA DI UNIVERSO FRA LE DUE FONTI ---")
    print("(strike con gamma proprio positivo ma gamma vendor a zero per arrotondamento)")
    print(f"strike con gamma proprio > 0, mediana:   {n_own.median():.1f}")
    print(f"strike persi dal vendor, mediana:        {n_lost.median():.1f}")
    print(f"quota di strike persi:                   {(n_lost / n_own).median() * 100:.2f}%")
    print(f"quota di OI su strike persi: mediana     {sl_all.median() * 100:.2f}%")
    print(f"                             media       {sl_all.mean() * 100:.2f}%")
    print(f"                             massimo     {sl_all.max() * 100:.2f}%")
    print()
    print(f"Elapsed time:                    {total_elapsed / 60:.2f} minutes")
    print(f"Output folder:                   {OUTPUT_DIR}")
    print("=" * 140)
    print()
    print("Summary:")
    print(summary.to_string(index=False))


def main():
    gamma_herfindahl_black76()
    gamma_herfindahl_vendor()


if __name__ == "__main__":
    main()
