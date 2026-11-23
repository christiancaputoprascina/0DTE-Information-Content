"""
02_Black76_IV_greeks.py: volatilita' implicita e greche Black '76 per ogni quotazione 0DTE.

Forward dalla parita' put-call (mediana delle 5 coppie con |C-P| minimo), prezzo mid bid/ask,
tempo a scadenza su base 365. IV per bisezione su [1e-4, 5], risolta sul lato OTM e specchiata
sullo strike gemello ITM; flag per IV al limite, non convergenza e scadenza entro 5 minuti.
Delta e gamma Black '76; confronto con IV, delta e gamma del vendor sulle sole righe OTM.

Input:  CBOE/Data/01_filter_0dte/, CBOE/Data/00_inputs/sofr_daily_2025.csv
Output: CBOE/Data/02_black76_iv_greeks/ (un file per giornata e un report annuale)
"""

from pathlib import Path
import re
import time
import csv

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from common import DATA_DIR, INPUT_DIR, YEAR, output_dir

# =====================================================
# IMPOSTAZIONI
# =====================================================

# La colonna expiration_time e' obbligatoria: nelle tre sessioni a chiusura anticipata il
# settlement e' alle 13:00, e un orario di default alle 16:00 darebbe un tau sbagliato.
REQUIRE_EXPIRATION_TIME_COLUMN = True

OVERWRITE_EXISTING = True

YEAR_BASIS = 365

PRICE_SOURCE = "mid_bid_ask"

# Forward: mediana dei forward impliciti delle N coppie call/put con |C-P| minimo. La mediana
# e' insensibile a singoli outlier, a differenza di una coppia singola (stile VIX) o di una
# regressione OLS su piu' strike.
N_CLOSEST_PAIRS_FOR_F = 5

NEAR_EXPIRY_MINUTES = 5
NEAR_EXPIRY_SECONDS = NEAR_EXPIRY_MINUTES * 60

MIN_T = 1e-10

IV_LOW = 1e-4
IV_HIGH = 5.0
IV_ITERATIONS = 70

# Convergenza misurata sul residuo di prezzo relativo, perche' la tolleranza deve scalare col
# prezzo. Sulle quotazioni 0.00/0.05 (mid 0.025, mezzo tick) il prezzo di Black '76 e' nullo
# per ogni sigma: il residuo relativo vale 1 e la riga risulta non convergente.
IV_PRICE_RESIDUAL_REL_TOL = 1e-4
IV_BOUND_TOL = 1e-6

# =====================================================
# CARTELLE
# =====================================================

FILTER_DIR = DATA_DIR / "01_filter_0dte"
OUTPUT_DIR = output_dir("02_black76_iv_greeks")
SOFR_FILE = INPUT_DIR / "sofr_daily_2025.csv"

filename_pattern = re.compile(
    r"01_0DTE_(\d{4})_(\d{2})_(\d{2})\.csv$"
)



# =====================================================
# NORMAL CDF / PDF CON SCIPY.STATS
# =====================================================

def norm_cdf(x):
    return scipy_norm.cdf(np.asarray(x, dtype=float))


def norm_pdf(x):
    return scipy_norm.pdf(np.asarray(x, dtype=float))


# =====================================================
# BLACK 76
# =====================================================

def black76_price(F, K, T, r, sigma, is_call):
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    price = np.full_like(F, np.nan, dtype=float)

    valid = (
        np.isfinite(F)
        & np.isfinite(K)
        & np.isfinite(T)
        & np.isfinite(r)
        & np.isfinite(sigma)
        & (F > 0)
        & (K > 0)
        & (T > MIN_T)
        & (sigma > 0)
    )

    if not np.any(valid):
        return price

    sqrtT = np.sqrt(T[valid])
    df = np.exp(-r[valid] * T[valid])

    d1 = (
        np.log(F[valid] / K[valid])
        + 0.5 * sigma[valid] * sigma[valid] * T[valid]
    ) / (sigma[valid] * sqrtT)

    d2 = d1 - sigma[valid] * sqrtT

    call_price = df * (
        F[valid] * norm_cdf(d1)
        - K[valid] * norm_cdf(d2)
    )

    put_price = df * (
        K[valid] * norm_cdf(-d2)
        - F[valid] * norm_cdf(-d1)
    )

    price[valid] = np.where(
        is_call[valid],
        call_price,
        put_price
    )

    return price


def black76_implied_vol(price, F, K, T, r, is_call):
    price = np.asarray(price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    iv = np.full_like(price, np.nan, dtype=float)

    valid = (
        np.isfinite(price)
        & np.isfinite(F)
        & np.isfinite(K)
        & np.isfinite(T)
        & np.isfinite(r)
        & (price > 0)
        & (F > 0)
        & (K > 0)
        & (T > MIN_T)
    )

    if not np.any(valid):
        return iv

    df = np.exp(-r[valid] * T[valid])

    intrinsic_call = df * np.maximum(F[valid] - K[valid], 0.0)
    intrinsic_put = df * np.maximum(K[valid] - F[valid], 0.0)

    intrinsic = np.where(
        is_call[valid],
        intrinsic_call,
        intrinsic_put
    )

    max_price_call = df * F[valid]
    max_price_put = df * K[valid]

    max_price = np.where(
        is_call[valid],
        max_price_call,
        max_price_put
    )

    valid_sub = (
        (price[valid] >= intrinsic - 1e-8)
        & (price[valid] <= max_price + 1e-8)
    )

    valid_indices = np.where(valid)[0]
    idx = valid_indices[valid_sub]

    if len(idx) == 0:
        return iv

    low = np.full(len(idx), IV_LOW, dtype=float)
    high = np.full(len(idx), IV_HIGH, dtype=float)

    target = price[idx]
    F_i = F[idx]
    K_i = K[idx]
    T_i = T[idx]
    r_i = r[idx]
    call_i = is_call[idx]

    for _ in range(IV_ITERATIONS):
        mid = 0.5 * (low + high)

        model = black76_price(
            F_i,
            K_i,
            T_i,
            r_i,
            mid,
            call_i
        )

        too_low = model < target

        low = np.where(too_low, mid, low)
        high = np.where(too_low, high, mid)

    iv[idx] = 0.5 * (low + high)

    return iv


def black76_delta_gamma(F, K, T, r, sigma, is_call):
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    delta = np.full_like(F, np.nan, dtype=float)
    gamma = np.full_like(F, np.nan, dtype=float)

    valid = (
        np.isfinite(F)
        & np.isfinite(K)
        & np.isfinite(T)
        & np.isfinite(r)
        & np.isfinite(sigma)
        & (F > 0)
        & (K > 0)
        & (T > MIN_T)
        & (sigma > 0)
    )

    if not np.any(valid):
        return delta, gamma

    sqrtT = np.sqrt(T[valid])
    df = np.exp(-r[valid] * T[valid])

    d1 = (
        np.log(F[valid] / K[valid])
        + 0.5 * sigma[valid] * sigma[valid] * T[valid]
    ) / (sigma[valid] * sqrtT)

    nd1 = norm_cdf(d1)
    pdf_d1 = norm_pdf(d1)

    call_delta = df * nd1
    put_delta = df * (nd1 - 1.0)

    delta[valid] = np.where(
        is_call[valid],
        call_delta,
        put_delta
    )

    gamma[valid] = (
        df
        * pdf_d1
        / (F[valid] * sigma[valid] * sqrtT)
    )

    return delta, gamma


# =====================================================
# LETTURA SOFR
# =====================================================

def load_sofr_rates(sofr_file: Path) -> pd.DataFrame:
    if not sofr_file.exists():
        raise FileNotFoundError(
            f"Non trovo il file SOFR:\n{sofr_file}\n\n"
            "Metti sofr_daily_2025.csv dentro CBOE/Data/00_inputs."
        )

    rates_raw = pd.read_csv(sofr_file)

    normalized_cols = {
        str(col).strip().lower(): col
        for col in rates_raw.columns
    }

    date_candidates = [
        "date",
        "effective date",
        "observation_date",
        "observation date",
    ]

    rate_candidates = [
        "r",
        "sofr",
        "rate",
        "rate (%)",
        "sofr (%)",
    ]

    date_col = None
    rate_col = None

    for candidate in date_candidates:
        if candidate in normalized_cols:
            date_col = normalized_cols[candidate]
            break

    for candidate in rate_candidates:
        if candidate in normalized_cols:
            rate_col = normalized_cols[candidate]
            break

    if date_col is None:
        raise ValueError(
            "Il file SOFR deve avere una colonna date, effective date oppure observation_date."
        )

    if rate_col is None:
        raise ValueError(
            "Il file SOFR deve avere una colonna r, SOFR, rate oppure rate (%)."
        )

    rates = rates_raw[[date_col, rate_col]].copy()
    rates.columns = ["date", "r"]

    rates["date"] = pd.to_datetime(rates["date"], errors="coerce").dt.normalize()
    rates["r"] = pd.to_numeric(rates["r"], errors="coerce")

    valid_r = rates["r"].dropna()

    if not valid_r.empty and valid_r.abs().median() > 1.0:
        rates["r"] = rates["r"] / 100.0

    rates = rates.dropna(subset=["date"]).sort_values("date")
    rates["r"] = rates["r"].ffill()
    rates = rates.dropna(subset=["r"])

    if rates.empty:
        raise ValueError("Il file SOFR non contiene valori validi.")

    return rates


# =====================================================
# FUNZIONI DATI
# =====================================================

def detect_quote_columns(df: pd.DataFrame):
    if "quote_date" in df.columns and "quote_time" in df.columns:
        return "quote_date", "quote_time", None

    if "quote_datetime" in df.columns:
        return None, None, "quote_datetime"

    if "quoted_datetime" in df.columns:
        return None, None, "quoted_datetime"

    raise ValueError(
        "Non trovo quote_date/quote_time oppure quote_datetime/quoted_datetime."
    )


def normalize_option_type(series: pd.Series):
    s = series.astype(str).str.strip().str.upper()

    is_call = s.isin(["C", "CALL"])
    is_put = s.isin(["P", "PUT"])

    return is_call, is_put


def choose_option_price_mid(df: pd.DataFrame) -> pd.DataFrame:
    if "bid" not in df.columns or "ask" not in df.columns:
        raise ValueError("Mancano bid e/o ask. Non posso calcolare il mid bid/ask.")

    bid = pd.to_numeric(df["bid"], errors="coerce")
    ask = pd.to_numeric(df["ask"], errors="coerce")

    mid = (bid + ask) / 2.0
    spread = ask - bid

    valid_mid = (
        bid.notna()
        & ask.notna()
        & (bid >= 0)
        & (ask > 0)
        & (ask >= bid)
        & (mid > 0)
    )

    out = df.copy()

    out["black76_price_used"] = mid.where(valid_mid)
    out["black76_price_source"] = np.where(
        valid_mid,
        "bid_ask_mid",
        "invalid_bid_ask"
    )

    out["black76_bid"] = bid
    out["black76_ask"] = ask
    out["black76_mid"] = mid.where(valid_mid)
    out["black76_spread"] = spread.where(valid_mid)

    out["black76_spread_pct_of_mid"] = np.where(
        valid_mid & (mid > 0),
        spread / mid,
        np.nan
    )

    return out


def choose_underlying_reference(df: pd.DataFrame):
    ref = pd.Series(np.nan, index=df.index, dtype=float)
    source = pd.Series("", index=df.index, dtype=str)

    def fill_from(values, source_name):
        nonlocal ref, source

        values = pd.to_numeric(values, errors="coerce")

        mask = (
            ref.isna()
            & values.notna()
            & (values > 0)
        )

        ref.loc[mask] = values.loc[mask]
        source.loc[mask] = source_name

    if "active_underlying_price" in df.columns:
        fill_from(df["active_underlying_price"], "active_underlying_price")

    if "implied_underlying_price" in df.columns:
        fill_from(df["implied_underlying_price"], "implied_underlying_price")

    if "underlying_bid" in df.columns and "underlying_ask" in df.columns:
        underlying_bid = pd.to_numeric(df["underlying_bid"], errors="coerce")
        underlying_ask = pd.to_numeric(df["underlying_ask"], errors="coerce")

        underlying_mid = (underlying_bid + underlying_ask) / 2.0

        valid_mid = (
            underlying_bid.notna()
            & underlying_ask.notna()
            & (underlying_bid > 0)
            & (underlying_ask > 0)
            & (underlying_ask >= underlying_bid)
        )

        fill_from(underlying_mid.where(valid_mid), "underlying_mid")

    return ref, source


def add_rates(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    quote_dates = pd.to_datetime(df["quote_date"], errors="coerce").dt.normalize()

    tmp = pd.DataFrame({
        "_quote_date_dt": quote_dates,
        "_row_order_for_rates": np.arange(len(df)),
    })

    valid_tmp = tmp.dropna(subset=["_quote_date_dt"]).sort_values("_quote_date_dt")

    r_values = np.full(len(df), np.nan, dtype=float)

    if not valid_tmp.empty:
        merged_valid = pd.merge_asof(
            valid_tmp,
            rates.rename(columns={"date": "_rate_date"}),
            left_on="_quote_date_dt",
            right_on="_rate_date",
            direction="backward"
        )

        valid_rows = merged_valid["_row_order_for_rates"].to_numpy(dtype=int)
        r_values[valid_rows] = merged_valid["r"].to_numpy(dtype=float)

    out = df.copy()
    out["black76_r"] = r_values

    return out


def get_timestamp_group_cols(df: pd.DataFrame):
    cols = []

    for col in [
        "underlying_symbol",
        "root",
        "quote_date",
        "quote_time",
        "expiration",
    ]:
        if col in df.columns:
            cols.append(col)

    if "quote_date" not in cols or "quote_time" not in cols or "expiration" not in cols:
        raise ValueError(
            "Per calcolare F mediano servono almeno quote_date, quote_time ed expiration."
        )

    return cols


def add_median_forward_365(df: pd.DataFrame) -> pd.DataFrame:
    F_col = "black76_F_365"
    F_count_col = "black76_F_pair_count_365"
    F_method_col = "black76_F_method_365"

    timestamp_cols = get_timestamp_group_cols(df)
    pair_cols = timestamp_cols + ["strike"]

    if "strike" not in df.columns:
        raise ValueError("Manca la colonna strike.")

    calls = df[df["_is_call"]].copy()
    puts = df[df["_is_put"]].copy()

    if calls.empty or puts.empty:
        df[F_col] = np.nan
        df[F_count_col] = 0
        df[F_method_col] = "missing_no_call_put_pairs"
        return df

    calls_grouped = (
        calls[pair_cols + ["black76_price_used"]]
        .dropna(subset=["black76_price_used"])
        .groupby(pair_cols, dropna=False, as_index=False)
        .agg({"black76_price_used": "median"})
        .rename(columns={"black76_price_used": "_call_price"})
    )

    puts_grouped = (
        puts[pair_cols + ["black76_price_used"]]
        .dropna(subset=["black76_price_used"])
        .groupby(pair_cols, dropna=False, as_index=False)
        .agg({"black76_price_used": "median"})
        .rename(columns={"black76_price_used": "_put_price"})
    )

    pairs = pd.merge(
        calls_grouped,
        puts_grouped,
        on=pair_cols,
        how="inner"
    )

    if pairs.empty:
        df[F_col] = np.nan
        df[F_count_col] = 0
        df[F_method_col] = "missing_no_matched_pairs"
        return df

    context = (
        df[pair_cols + [
            "_seconds_to_expiration",
            "black76_r",
            "black76_underlying_ref",
        ]]
        .groupby(pair_cols, dropna=False, as_index=False)
        .agg({
            "_seconds_to_expiration": "median",
            "black76_r": "median",
            "black76_underlying_ref": "median",
        })
    )

    pairs = pairs.merge(
        context,
        on=pair_cols,
        how="left"
    )

    pairs["_strike_num"] = pd.to_numeric(pairs["strike"], errors="coerce")

    pairs["_T"] = (
        pd.to_numeric(pairs["_seconds_to_expiration"], errors="coerce")
        / (YEAR_BASIS * 24.0 * 60.0 * 60.0)
    )

    pairs["_r"] = pd.to_numeric(pairs["black76_r"], errors="coerce")

    pairs["_discount_factor"] = np.exp(
        -pairs["_r"] * pairs["_T"]
    )

    pairs["_call_price"] = pd.to_numeric(pairs["_call_price"], errors="coerce")
    pairs["_put_price"] = pd.to_numeric(pairs["_put_price"], errors="coerce")

    pairs["_F_candidate"] = (
        pairs["_strike_num"]
        + (pairs["_call_price"] - pairs["_put_price"]) / pairs["_discount_factor"]
    )

    F_candidate = pd.to_numeric(pairs["_F_candidate"], errors="coerce")
    strike_num = pd.to_numeric(pairs["_strike_num"], errors="coerce")
    call_price = pd.to_numeric(pairs["_call_price"], errors="coerce")
    put_price = pd.to_numeric(pairs["_put_price"], errors="coerce")
    T = pd.to_numeric(pairs["_T"], errors="coerce")
    r = pd.to_numeric(pairs["_r"], errors="coerce")

    valid = (
        F_candidate.notna()
        & np.isfinite(F_candidate)
        & (F_candidate > 0)
        & strike_num.notna()
        & np.isfinite(strike_num)
        & (strike_num > 0)
        & call_price.notna()
        & put_price.notna()
        & np.isfinite(call_price)
        & np.isfinite(put_price)
        & (call_price > 0)
        & (put_price > 0)
        & T.notna()
        & np.isfinite(T)
        & (T > MIN_T)
        & r.notna()
        & np.isfinite(r)
    )

    pairs["_abs_c_minus_p"] = (call_price - put_price).abs()

    valid_pairs = pairs[valid].copy()

    valid_pairs["_rank_by_c_minus_p"] = (
        valid_pairs
        .groupby(timestamp_cols, dropna=False)["_abs_c_minus_p"]
        .rank(method="first")
    )

    selected = valid_pairs[
        valid_pairs["_rank_by_c_minus_p"] <= N_CLOSEST_PAIRS_FOR_F
    ]

    summary = (
        selected
        .groupby(timestamp_cols, dropna=False, as_index=False)
        .agg(
            _F_median=("_F_candidate", "median"),
            _F_count=("_F_candidate", "count"),
        )
    )

    summary[F_col] = summary["_F_median"]
    summary[F_count_col] = summary["_F_count"].fillna(0).astype(int)

    summary[F_method_col] = np.where(
        summary[F_col].notna(),
        "median_of_closest_pairs",
        "missing"
    )

    keep_cols = timestamp_cols + [
        F_col,
        F_count_col,
        F_method_col,
    ]

    df = df.merge(
        summary[keep_cols],
        on=timestamp_cols,
        how="left"
    )

    df[F_count_col] = df[F_count_col].fillna(0).astype(int)
    df[F_method_col] = df[F_method_col].fillna("missing")

    df = df.sort_values("_row_order").reset_index(drop=True)

    return df


def add_moneyness_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    F = pd.to_numeric(out["black76_F_365"], errors="coerce")
    K = pd.to_numeric(out["strike"], errors="coerce")

    valid = (
        F.notna()
        & K.notna()
        & np.isfinite(F)
        & np.isfinite(K)
        & (F > 0)
        & (K > 0)
    )

    out["black76_moneyness_365"] = np.where(
        valid,
        F / K,
        np.nan
    )

    out["black76_log_moneyness_365"] = np.where(
        valid,
        np.log(F / K),
        np.nan
    )

    return out


def compute_black76_365(df: pd.DataFrame) -> pd.DataFrame:
    T_col = "black76_T_365"
    F_col = "black76_F_365"

    price_col = "black76_theoretical_price_365"
    iv_col = "black76_implied_volatility_365"
    delta_col = "black76_delta_365"
    gamma_col = "black76_gamma_365"

    seconds_to_expiration = pd.to_numeric(
        df["_seconds_to_expiration"],
        errors="coerce"
    )

    df[T_col] = seconds_to_expiration / (YEAR_BASIS * 24.0 * 60.0 * 60.0)

    df = add_median_forward_365(df)

    df = add_moneyness_columns(df)

    price = pd.to_numeric(
        df["black76_price_used"],
        errors="coerce"
    ).to_numpy(dtype=float)

    F = pd.to_numeric(
        df[F_col],
        errors="coerce"
    ).to_numpy(dtype=float)

    K = pd.to_numeric(
        df["strike"],
        errors="coerce"
    ).to_numpy(dtype=float)

    T = pd.to_numeric(
        df[T_col],
        errors="coerce"
    ).to_numpy(dtype=float)

    r = pd.to_numeric(
        df["black76_r"],
        errors="coerce"
    ).to_numpy(dtype=float)

    call_bool = df["_is_call"].to_numpy(dtype=bool)

    # IV risolta sul lato OTM, dove il prezzo e' quasi tutto valore tempo (invertire un prezzo
    # ITM, dominato dall'intrinseco, e' instabile), e assegnata a entrambi i lati dello stesso
    # strike. Se il lato OTM manca o non si risolve, inversione indipendente riga per riga.
    is_otm = (
        (call_bool & (K > F))
        | (~call_bool & (K < F))
    )

    df["black76_is_otm_365"] = np.where(is_otm, 1, 0)

    price_otm_only = np.where(is_otm, price, np.nan)

    iv_otm_only = black76_implied_vol(
        price_otm_only,
        F,
        K,
        T,
        r,
        call_bool
    )

    pair_cols = get_timestamp_group_cols(df) + ["strike"]

    otm_iv_lookup = pd.DataFrame({col: df[col].to_numpy() for col in pair_cols})
    otm_iv_lookup["_iv_otm_only"] = iv_otm_only

    otm_iv_by_pair = (
        otm_iv_lookup
        .dropna(subset=["_iv_otm_only"])
        .groupby(pair_cols, dropna=False, as_index=False)
        .agg(_iv_mirrored=("_iv_otm_only", "first"))
    )

    df_pair_keys = df[pair_cols].merge(
        otm_iv_by_pair,
        on=pair_cols,
        how="left"
    )

    iv_mirrored = pd.to_numeric(
        df_pair_keys["_iv_mirrored"], errors="coerce"
    ).to_numpy(dtype=float)

    needs_fallback = ~np.isfinite(iv_mirrored)

    iv_fallback = black76_implied_vol(
        price,
        F,
        K,
        T,
        r,
        call_bool
    )

    iv = np.where(needs_fallback, iv_fallback, iv_mirrored)

    df["black76_iv_source_365"] = np.where(
        ~needs_fallback,
        "otm_mirrored",
        np.where(np.isfinite(iv_fallback), "independent_fallback", "missing")
    )

    theoretical_price = black76_price(
        F,
        K,
        T,
        r,
        iv,
        call_bool
    )

    delta, gamma = black76_delta_gamma(
        F,
        K,
        T,
        r,
        iv,
        call_bool
    )

    df[price_col] = theoretical_price
    df[iv_col] = iv
    df[delta_col] = delta
    df[gamma_col] = gamma

    # Controllo di convergenza: residuo di prezzo, valutato solo dove la IV e' stata risolta
    # (lato OTM o fallback; sul lato specchiato il prezzo teorico non deve riprodurre il mid),
    # e saturazione ai limiti dell'intervallo, che la bisezione da sola non segnala.
    residual = np.abs(theoretical_price - price)

    solved_here = is_otm | needs_fallback

    df["black76_price_residual_365"] = np.where(solved_here, residual, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        residual_rel = np.where(
            np.isfinite(price) & (price > 0), residual / price, np.nan
        )

    df["black76_price_residual_rel_365"] = np.where(
        solved_here, residual_rel, np.nan
    )

    df["black76_iv_at_bound_365"] = np.where(
        np.isfinite(iv)
        & (
            (iv >= IV_HIGH - IV_BOUND_TOL)
            | (iv <= IV_LOW + IV_BOUND_TOL)
        ),
        1,
        0
    )

    # convergenza: sul lato risolto serve residuo sotto tolleranza; sul lato
    # specchiato la convergenza e' quella dello strike gemello, gia' verificata.
    converged_here = np.where(
        solved_here,
        np.isfinite(residual_rel) & (residual_rel <= IV_PRICE_RESIDUAL_REL_TOL),
        True
    )

    df["black76_iv_converged_365"] = np.where(converged_here, 1, 0)

    return df


def add_quality_and_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    seconds_to_expiration = pd.to_numeric(
        out["_seconds_to_expiration"],
        errors="coerce"
    )

    price = pd.to_numeric(out["black76_price_used"], errors="coerce")
    F = pd.to_numeric(out["black76_F_365"], errors="coerce")
    K = pd.to_numeric(out["strike"], errors="coerce")
    T = pd.to_numeric(out["black76_T_365"], errors="coerce")
    r = pd.to_numeric(out["black76_r"], errors="coerce")
    iv = pd.to_numeric(out["black76_implied_volatility_365"], errors="coerce")
    delta = pd.to_numeric(out["black76_delta_365"], errors="coerce")
    gamma = pd.to_numeric(out["black76_gamma_365"], errors="coerce")

    out["black76_near_expiry_flag_365"] = np.where(
        seconds_to_expiration.notna()
        & np.isfinite(seconds_to_expiration)
        & (seconds_to_expiration >= 0)
        & (seconds_to_expiration < NEAR_EXPIRY_SECONDS),
        1,
        0
    )

    solver_success = (
        price.notna()
        & F.notna()
        & K.notna()
        & T.notna()
        & r.notna()
        & iv.notna()
        & delta.notna()
        & gamma.notna()
        & np.isfinite(price)
        & np.isfinite(F)
        & np.isfinite(K)
        & np.isfinite(T)
        & np.isfinite(r)
        & np.isfinite(iv)
        & np.isfinite(delta)
        & np.isfinite(gamma)
        & (price > 0)
        & (F > 0)
        & (K > 0)
        & (T > MIN_T)
        & (iv > 0)
        & (out["black76_F_method_365"].astype(str) != "missing")
        & (out["black76_price_source"].astype(str) == "bid_ask_mid")
    )

    # Il successo non richiede la convergenza: le righe con IV al limite restano, marcate dai
    # flag di saturazione e convergenza, e a valle le esclude il troncamento CBOE sul bid a
    # zero (criterio economico, non legato ai limiti scelti per il solver).
    out["black76_solver_success_365"] = np.where(
        solver_success,
        1,
        0
    )

    validation_usable = solver_success.copy()

    # Confronto con il vendor solo sul lato OTM: sulle righe ITM la IV e' specchiata dall'altro
    # lato dello strike, mentre il vendor inverte in modo indipendente.
    validation_usable = (
        validation_usable
        & (pd.to_numeric(out["black76_is_otm_365"], errors="coerce") == 1)
        # serve una IV convergente e non al limite
        & (pd.to_numeric(out["black76_iv_converged_365"], errors="coerce") == 1)
        & (pd.to_numeric(out["black76_iv_at_bound_365"], errors="coerce") == 0)
    )

    if "implied_volatility" in out.columns:
        cboe_iv = pd.to_numeric(out["implied_volatility"], errors="coerce")

        validation_usable = (
            validation_usable
            & cboe_iv.notna()
            & np.isfinite(cboe_iv)
            & (cboe_iv > 0)
        )
    else:
        validation_usable = validation_usable & False

    if "delta" in out.columns:
        cboe_delta = pd.to_numeric(out["delta"], errors="coerce")

        validation_usable = (
            validation_usable
            & cboe_delta.notna()
            & np.isfinite(cboe_delta)
        )
    else:
        validation_usable = validation_usable & False

    if "gamma" in out.columns:
        cboe_gamma = pd.to_numeric(out["gamma"], errors="coerce")

        validation_usable = (
            validation_usable
            & cboe_gamma.notna()
            & np.isfinite(cboe_gamma)
            & (cboe_gamma > 0)
        )
    else:
        validation_usable = validation_usable & False

    validation_usable = (
        validation_usable
        & (out["black76_near_expiry_flag_365"] == 0)
    )

    out["black76_validation_usable_365"] = np.where(
        validation_usable,
        1,
        0
    )

    return out


def add_cboe_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "implied_volatility" in out.columns:
        cboe_iv = pd.to_numeric(out["implied_volatility"], errors="coerce")
        my_iv = pd.to_numeric(out["black76_implied_volatility_365"], errors="coerce")

        out["black76_iv_error_vs_cboe_365"] = my_iv - cboe_iv
        out["black76_abs_iv_error_vs_cboe_365"] = (my_iv - cboe_iv).abs()

    if "delta" in out.columns:
        cboe_delta = pd.to_numeric(out["delta"], errors="coerce")
        my_delta = pd.to_numeric(out["black76_delta_365"], errors="coerce")

        out["black76_delta_error_vs_cboe_365"] = my_delta - cboe_delta
        out["black76_abs_delta_error_vs_cboe_365"] = (my_delta - cboe_delta).abs()

    if "gamma" in out.columns:
        cboe_gamma = pd.to_numeric(out["gamma"], errors="coerce")
        my_gamma = pd.to_numeric(out["black76_gamma_365"], errors="coerce")

        out["black76_gamma_error_vs_cboe_365"] = my_gamma - cboe_gamma
        out["black76_abs_gamma_error_vs_cboe_365"] = (my_gamma - cboe_gamma).abs()

    return out


def mean_or_nan(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def filtered_mean_or_nan(df: pd.DataFrame, value_col: str, flag_col: str):
    if value_col not in df.columns or flag_col not in df.columns:
        return np.nan

    mask = pd.to_numeric(df[flag_col], errors="coerce") == 1
    values = pd.to_numeric(df.loc[mask, value_col], errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def process_one_file(input_file: Path, output_file: Path, rates: pd.DataFrame):
    start = time.time()

    df = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    original_columns = list(df.columns)

    df["_row_order"] = np.arange(len(df))

    quote_date_col, quote_time_col, quote_datetime_col = detect_quote_columns(df)

    if quote_datetime_col is not None:
        df["quote_date"] = df[quote_datetime_col].astype(str).str.strip().str[:10]
        df["quote_time"] = df[quote_datetime_col].astype(str).str.strip().str[11:19]

        if "quote_date" not in original_columns:
            original_columns.append("quote_date")

        if "quote_time" not in original_columns:
            original_columns.append("quote_time")
    else:
        df["quote_date"] = df[quote_date_col].astype(str).str.strip()
        df["quote_time"] = df[quote_time_col].astype(str).str.strip()

    if "expiration" not in df.columns:
        raise ValueError(f"Colonna expiration mancante in {input_file.name}")

    if "expiration_time" not in df.columns:
        if REQUIRE_EXPIRATION_TIME_COLUMN:
            raise ValueError(
                f"Colonna expiration_time mancante in {input_file.name}. "
                "Non esiste un default sicuro: nelle tre sessioni early-close "
                "del 2025 (3 luglio, 28 novembre, 24 dicembre) il settlement e' "
                "alle 13:00 ET e non alle 16:00, quindi un default a 16:00:00 "
                "produrrebbe un tau sbagliato senza alcun avviso."
            )
        df["expiration_time"] = "16:00:00"

        if "expiration_time" not in original_columns:
            original_columns.append("expiration_time")

    if "strike" not in df.columns:
        raise ValueError(f"Colonna strike mancante in {input_file.name}")

    if "option_type" not in df.columns:
        raise ValueError(f"Colonna option_type mancante in {input_file.name}")

    df = choose_option_price_mid(df)

    df["black76_underlying_ref"], df["black76_underlying_ref_source"] = choose_underlying_reference(df)

    is_call, is_put = normalize_option_type(df["option_type"])

    df["_is_call"] = is_call
    df["_is_put"] = is_put

    quote_ts = pd.to_datetime(
        df["quote_date"].astype(str) + " " + df["quote_time"].astype(str),
        errors="coerce"
    )

    expiration_ts = pd.to_datetime(
        df["expiration"].astype(str).str.strip().str[:10]
        + " "
        + df["expiration_time"].astype(str).str.strip().str[:8],
        errors="coerce"
    )

    df["_seconds_to_expiration"] = (expiration_ts - quote_ts).dt.total_seconds()

    df = add_rates(df, rates)

    df = compute_black76_365(df)

    df = add_quality_and_validation_flags(df)

    df = add_cboe_validation_columns(df)

    added_columns = [
        "black76_r",
        "black76_bid",
        "black76_ask",
        "black76_mid",
        "black76_spread",
        "black76_spread_pct_of_mid",
        "black76_price_used",
        "black76_price_source",
        "black76_underlying_ref",
        "black76_underlying_ref_source",
        "black76_T_365",
        "black76_F_365",
        "black76_moneyness_365",
        "black76_log_moneyness_365",
        "black76_F_pair_count_365",
        "black76_F_method_365",
        "black76_theoretical_price_365",
        "black76_implied_volatility_365",
        "black76_iv_source_365",
        "black76_is_otm_365",
        "black76_price_residual_365",
        "black76_price_residual_rel_365",
        "black76_iv_at_bound_365",
        "black76_iv_converged_365",
        "black76_delta_365",
        "black76_gamma_365",
        "black76_near_expiry_flag_365",
        "black76_solver_success_365",
        "black76_validation_usable_365",
        "black76_iv_error_vs_cboe_365",
        "black76_abs_iv_error_vs_cboe_365",
        "black76_delta_error_vs_cboe_365",
        "black76_abs_delta_error_vs_cboe_365",
        "black76_gamma_error_vs_cboe_365",
        "black76_abs_gamma_error_vs_cboe_365",
    ]

    final_columns = original_columns + [
        col for col in added_columns
        if col in df.columns and col not in original_columns
    ]

    df_out = df[final_columns].copy()
    df_out.to_csv(output_file, index=False)

    elapsed = time.time() - start

    usable_validation_rows = int((df["black76_validation_usable_365"] == 1).sum())

    resid_solved = pd.to_numeric(
        df["black76_price_residual_365"], errors="coerce"
    ).dropna()

    # residuo relativo sulle sole righe non sature, dove la convergenza e' verificabile
    resid_rel_conv = pd.to_numeric(
        df.loc[df["black76_iv_at_bound_365"] == 0, "black76_price_residual_rel_365"],
        errors="coerce"
    ).dropna()

    result = {
        "input_file": input_file.name,
        "output_file": output_file.name,
        "rows": len(df_out),
        "missing_price_365": int(df["black76_price_used"].isna().sum()),
        "missing_F_365": int(df["black76_F_365"].isna().sum()),
        "missing_moneyness_365": int(df["black76_moneyness_365"].isna().sum()),
        "missing_iv_365": int(df["black76_implied_volatility_365"].isna().sum()),
        "missing_delta_365": int(df["black76_delta_365"].isna().sum()),
        "missing_gamma_365": int(df["black76_gamma_365"].isna().sum()),
        "near_expiry_rows_365": int((df["black76_near_expiry_flag_365"] == 1).sum()),
        "otm_rows_365": int((df["black76_is_otm_365"] == 1).sum()),
        "iv_at_bound_rows_365": int((df["black76_iv_at_bound_365"] == 1).sum()),
        "iv_not_converged_rows_365": int((df["black76_iv_converged_365"] == 0).sum()),
        "max_price_residual_365": float(resid_solved.max()) if not resid_solved.empty else np.nan,
        "median_price_residual_365": float(resid_solved.median()) if not resid_solved.empty else np.nan,
        "max_price_residual_rel_365": float(resid_rel_conv.max()) if not resid_rel_conv.empty else np.nan,
        "solver_success_rows_365": int((df["black76_solver_success_365"] == 1).sum()),
        "solver_failure_rows_365": int((df["black76_solver_success_365"] == 0).sum()),
        "validation_usable_rows_365": usable_validation_rows,
        "fallback_F_rows_365": int(
            (
                (pd.to_numeric(df["black76_F_pair_count_365"], errors="coerce") > 0)
                & (pd.to_numeric(df["black76_F_pair_count_365"], errors="coerce") < N_CLOSEST_PAIRS_FOR_F)
            ).sum()
        ),
        "missing_F_rows_365": int((df["black76_F_method_365"] == "missing").sum()),
        "mean_abs_iv_error_vs_cboe_365": filtered_mean_or_nan(
            df,
            "black76_abs_iv_error_vs_cboe_365",
            "black76_validation_usable_365"
        ),
        "mean_abs_delta_error_vs_cboe_365": filtered_mean_or_nan(
            df,
            "black76_abs_delta_error_vs_cboe_365",
            "black76_validation_usable_365"
        ),
        "mean_abs_gamma_error_vs_cboe_365": filtered_mean_or_nan(
            df,
            "black76_abs_gamma_error_vs_cboe_365",
            "black76_validation_usable_365"
        ),
        "elapsed_seconds": round(elapsed, 2),
    }

    return result


# =====================================================
# ESECUZIONE
# =====================================================

REPORT_FIELDS = [
    "input_file",
    "output_file",
    "rows",
    "missing_price_365",
    "missing_F_365",
    "missing_moneyness_365",
    "missing_iv_365",
    "missing_delta_365",
    "missing_gamma_365",
    "near_expiry_rows_365",
    "otm_rows_365",
    "iv_at_bound_rows_365",
    "iv_not_converged_rows_365",
    "max_price_residual_365",
    "median_price_residual_365",
    "max_price_residual_rel_365",
    "solver_success_rows_365",
    "solver_failure_rows_365",
    "validation_usable_rows_365",
    "fallback_F_rows_365",
    "missing_F_rows_365",
    "mean_abs_iv_error_vs_cboe_365",
    "mean_abs_delta_error_vs_cboe_365",
    "mean_abs_gamma_error_vs_cboe_365",
    "elapsed_seconds",
]


def main():
    if not FILTER_DIR.exists():
        raise FileNotFoundError(
            f"La cartella input non esiste:\n{FILTER_DIR}\n\n"
            "Lancia prima 01_Filter_0DTE.py."
        )

    if not SOFR_FILE.exists():
        raise FileNotFoundError(
            f"Non trovo il file SOFR:\n{SOFR_FILE}\n\n"
            "Metti sofr_daily_2025.csv dentro CBOE/Data/00_inputs."
        )

    input_files = []
    for file in sorted(FILTER_DIR.glob("01_0DTE_*.csv")):
        match = filename_pattern.match(file.name)
        if match and match.group(1) == str(YEAR):
            input_files.append(file)

    if not input_files:
        raise FileNotFoundError(f"Nessun file 01_0DTE_{YEAR}_*.csv in:\n{FILTER_DIR}")

    run_label = f"{YEAR}_ALL"

    rates = load_sofr_rates(SOFR_FILE)

    report_file = OUTPUT_DIR / f"02_black76_365_report_{run_label}.csv"

    print("=" * 150)
    print("BLACK 76 FINALE TESI - BASE 365 - F MEDIANO - MID BID/ASK - CONVERGENZA VERIFICATA")
    print("=" * 150)
    print(f"Cartella input:     {FILTER_DIR}")
    print(f"Cartella output:    {OUTPUT_DIR}")
    print(f"SOFR file:          {SOFR_FILE}")
    print(f"YEAR:               {YEAR}")
    print(f"File trovati:       {len(input_files)}")
    print(f"Expiry time:        dal dato (colonna expiration_time), nessun default")
    print(f"Year basis:         {YEAR_BASIS}")
    print(f"Price source:       {PRICE_SOURCE}")
    print(f"F construction:     median of {N_CLOSEST_PAIRS_FOR_F} closest pairs by |C-P|")
    print(f"IV solver:          bisezione, {IV_ITERATIONS} iterazioni, sigma in [{IV_LOW}, {IV_HIGH}]")
    print(f"Convergenza:        residuo RELATIVO <= {IV_PRICE_RESIDUAL_REL_TOL} (flag informativo)")
    print(f"Scelta B:           righe con IV al limite TENUTE e marcate, escluse a")
    print(f"                    valle dal troncamento CBOE sul bid a zero")
    print(f"Near expiry flag:   T < {NEAR_EXPIRY_MINUTES} minuti")
    print("Validation vs CBOE: SOLO righe OTM (lo specchiamento rende il")
    print("                    confronto ITM privo di significato)")
    print("=" * 150)

    global_start = time.time()

    results = []

    with open(report_file, "w", newline="", encoding="utf-8") as report_csv:
        writer = csv.DictWriter(
            report_csv,
            fieldnames=REPORT_FIELDS,
        )

        writer.writeheader()

        for index, input_file in enumerate(input_files, start=1):
            year, month, day = filename_pattern.match(input_file.name).groups()

            output_file = OUTPUT_DIR / f"02_0DTE_{year}_{month}_{day}_black76_365.csv"

            if output_file.exists():
                if OVERWRITE_EXISTING:
                    output_file.unlink()
                else:
                    print(f"[{index}/{len(input_files)}] Gia' esistente, salto: {output_file.name}")
                    continue

            result = process_one_file(input_file, output_file, rates)

            results.append(result)
            writer.writerow(result)
            report_csv.flush()

            eta = (time.time() - global_start) / index * (len(input_files) - index) / 60.0

            print(
                f"[{index:3d}/{len(input_files)}] {year}-{month}-{day} | "
                f"rows={result['rows']:,} | ok={result['solver_success_rows_365']:,} | "
                f"bound={result['iv_at_bound_rows_365']:,} | "
                f"noconv={result['iv_not_converged_rows_365']:,} | "
                f"resid_rel={result['max_price_residual_rel_365']:.2e} | "
                f"MAE_IV_OTM={result['mean_abs_iv_error_vs_cboe_365']:.5f} | "
                f"ETA {eta:5.1f} min",
                flush=True
            )

    total_elapsed = time.time() - global_start

    total_rows = sum(r["rows"] for r in results)
    total_solver_success = sum(r["solver_success_rows_365"] for r in results)
    total_solver_failure = sum(r["solver_failure_rows_365"] for r in results)
    total_bound = sum(r["iv_at_bound_rows_365"] for r in results)
    total_noconv = sum(r["iv_not_converged_rows_365"] for r in results)
    total_validation = sum(r["validation_usable_rows_365"] for r in results)
    max_resid = np.nanmax([r["max_price_residual_rel_365"] for r in results])

    mae_iv = np.nanmean([r["mean_abs_iv_error_vs_cboe_365"] for r in results])
    mae_delta = np.nanmean([r["mean_abs_delta_error_vs_cboe_365"] for r in results])
    mae_gamma = np.nanmean([r["mean_abs_gamma_error_vs_cboe_365"] for r in results])

    print("=" * 150)
    print("COMPLETATO BLACK 76 FINALE TESI")
    print("=" * 150)
    print(f"File creati:                     {len(results)}")
    print(f"Righe totali:                    {total_rows:,}")
    print(f"Solver success rows:             {total_solver_success:,}")
    print(f"Solver failure rows:             {total_solver_failure:,}")
    print(f"  di cui IV ai limiti:           {total_bound:,}")
    print(f"  di cui non convergenti:        {total_noconv:,}")
    print(f"Residuo relativo max (non sature):{max_resid:.3e}  (tolleranza {IV_PRICE_RESIDUAL_REL_TOL})")
    print(f"Validation usable rows (OTM):    {total_validation:,}")
    print(f"MAE IV vs CBOE (solo OTM):       {mae_iv:.6f}")
    print(f"MAE Delta vs CBOE (solo OTM):    {mae_delta:.6f}")
    print(f"MAE Gamma vs CBOE (solo OTM):    {mae_gamma:.3e}")
    print(f"Tempo totale:                    {total_elapsed / 60:.2f} minuti")
    print(f"Report:                          {report_file}")


if __name__ == "__main__":
    main()
