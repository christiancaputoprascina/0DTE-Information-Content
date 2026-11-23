"""
Pannello giornaliero di regressione 2025, una riga per sessione (250): varianza realizzata
dell'indice e sue finestre, MFIV media di sessione e delle 09:31, concentrazione del gamma dalle
due fonti (vendor, primaria, e Black '76) e flag di calendario.
  Parte 1: pannello di base e sua validazione.
  Parte 2: varianti logm_005 e logm_002, con il blocco MFIV calcolato sulla banda |ln(K/F)|.
Input:  CBOE/Data/03_mfiv/, CBOE/Data/04_gamma_herfindahl/, CBOE/Data/05_spx_minute_bars/
Output: CBOE/Data/07_daily_panel/
"""

import re

import numpy as np
import pandas as pd

from common import (
    DATA_DIR,
    EARLY_CLOSE_DATES,
    EARLY_CLOSE_SECONDS,
    EARLY_CLOSE_TIMES,
    REGULAR_CLOSE,
    REGULAR_CLOSE_SECONDS,
    YEAR,
    clean_date_series,
    output_dir,
)

OUTPUT_DIR = output_dir("07_daily_panel")


# =====================================================================
# IMPOSTAZIONI E FUNZIONI COMUNI ALLE DUE PARTI
# =====================================================================
USE_ONLY_OK_MFIV_ROWS = True

SURFACE_START_TIME = "09:31:00"
SNAPSHOT_TIME = "09:31:00"
SNAPSHOT_TOLERANCE_MINUTES = 2
SNAPSHOT_TOLERANCE_SECONDS = SNAPSHOT_TOLERANCE_MINUTES * 60


def time_string_to_seconds(time_string: str) -> int:
    h, m, s = str(time_string).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_time_to_seconds(series: pd.Series) -> pd.Series:
    t = pd.to_datetime(series.astype(str), format="%H:%M:%S", errors="coerce")
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second


def add_time_seconds(df: pd.DataFrame, time_col: str = "quote_time") -> pd.DataFrame:
    out = df.copy()
    out["_time_seconds"] = parse_time_to_seconds(out[time_col])
    return out


SURFACE_START_SECONDS = time_string_to_seconds(SURFACE_START_TIME)
SNAPSHOT_SECONDS = time_string_to_seconds(SNAPSHOT_TIME)


# =====================================================================
# PARTE 1: PANNELLO GIORNALIERO DI REGRESSIONE
# =====================================================================
EXPECTED_ROW_COUNT = 250

MFIV_FOLDER_NAME = "03_mfiv"
MFIV_INTRADAY_FILE_NAME = "03_mfiv_intraday_2025_all.csv"

GAMMA_CBOE_FOLDER_NAME = "04_gamma_herfindahl"
GAMMA_CBOE_INTRADAY_FILE_NAME = "04_gamma_cboe_intraday_2025_all.csv"

GAMMA_INDEP_FOLDER_NAME = "04_gamma_herfindahl"
GAMMA_INDEP_INTRADAY_FILE_NAME = "04_gamma_intraday_2025_all.csv"

# ---------------------------------------------------------------------
# FONTE DEL GAMMA PER LE MISURE PRINCIPALI
# ---------------------------------------------------------------------
# H_OI_t e H_VOL_t vengono dalla fonte scelta qui; l'altra resta nelle colonne con suffisso.
# "cboe"  = gamma del vendor (colonna gamma), indipendente dall'implementazione della tesi;
# "indep" = gamma ricalcolato sotto Black '76.
GAMMA_PRIMARY_SOURCE = "cboe"

# ---------------------------------------------------------------------
# FONTE DELLA REALIZED VARIANCE
# ---------------------------------------------------------------------
# Le barre al minuto si leggono dai CSV di 05_spx_minute_bars, gia' convertiti di fuso ed
# etichettati a fine intervallo: la realized variance ha cosi' un'unica implementazione.
SPX_FOLDER_NAME = "05_spx_minute_bars"
SPX_CSV_GLOB = "05_SPX_*_RV.csv"

# ---------------------------------------------------------------------
# STIMATORE DELLA REALIZED VARIANCE
# ---------------------------------------------------------------------
# "consecutive" = log(close_t / close_{t-1}) fra barre, standard di letteratura;
# "within_bar"  = log(close/open) dentro la barra: ignora il movimento fra barre e sottostima.
# Lo stimatore non scelto va in RV_t_alt_estimator; il confronto e' stampato a fine esecuzione.
RV_ESTIMATOR = "consecutive"

OUTPUT_PANEL_FILE_NAME = "07_daily_regression_panel_2025.csv"
OUTPUT_VALIDATION_FILE_NAME = "07_daily_regression_panel_2025_validation.csv"

USE_ONLY_OK_GAMMA_ROWS = True

SESSION_START_TIME = "09:30:00"
OPEN_WINDOW_END_TIME = "10:00:00"

# ---------------------------------------------------------------------
# CHIUSURA DI SESSIONE, PER GIORNATA
# ---------------------------------------------------------------------
# Sulle tre early close il mercato chiude alle 13:00, ma Bloomberg pubblica barre a prezzo quasi
# fermo anche dopo: la chiusura si legge quindi per giornata (date e orari in common.py).

MFIV_FILE = DATA_DIR / MFIV_FOLDER_NAME / MFIV_INTRADAY_FILE_NAME
GAMMA_CBOE_FILE = DATA_DIR / GAMMA_CBOE_FOLDER_NAME / GAMMA_CBOE_INTRADAY_FILE_NAME
GAMMA_INDEP_FILE = DATA_DIR / GAMMA_INDEP_FOLDER_NAME / GAMMA_INDEP_INTRADAY_FILE_NAME

SPX_DIR = DATA_DIR / SPX_FOLDER_NAME

OUTPUT_PANEL_FILE = OUTPUT_DIR / OUTPUT_PANEL_FILE_NAME
OUTPUT_VALIDATION_FILE = OUTPUT_DIR / OUTPUT_VALIDATION_FILE_NAME

SESSION_START_SECONDS = time_string_to_seconds(SESSION_START_TIME)
OPEN_WINDOW_END_SECONDS = time_string_to_seconds(OPEN_WINDOW_END_TIME)


def session_end_seconds_for(quote_date) -> int:
    return EARLY_CLOSE_SECONDS.get(str(quote_date), REGULAR_CLOSE_SECONDS)


def require_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato:\n{file_path}")


def require_columns(df: pd.DataFrame, required_cols, file_label: str):
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Colonne mancanti in {file_label}: {missing}")


def normalize_colname(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def detect_column(columns, candidates):
    normalized_map = {normalize_colname(c): c for c in columns}

    for candidate in candidates:
        key = normalize_colname(candidate)

        if key in normalized_map:
            return normalized_map[key]

    return None


def third_friday_flag(date_string: str) -> int:
    d = pd.to_datetime(date_string, errors="coerce")

    if pd.isna(d):
        return 0

    if d.weekday() == 4 and 15 <= d.day <= 21:
        return 1

    return 0


# ---------------------------------------------------------------------
# SPX E COSTRUZIONE DELLA RV
# ---------------------------------------------------------------------
def load_spx_from_trim_csv() -> pd.DataFrame:
    """
    Legge i CSV al minuto di 05_spx_minute_bars, gia' convertiti di fuso ed etichettati.
    Colonne attese: Underlying, Date, time, open, close, high, low, RV.
    """
    if not SPX_DIR.exists():
        raise FileNotFoundError(
            f"Cartella SPX non trovata:\n{SPX_DIR}\n\n"
            "Attesa CBOE/Data/05_spx_minute_bars, prodotta da 05_SPX_minute_bars.py."
        )

    files = sorted(SPX_DIR.glob(SPX_CSV_GLOB))

    if not files:
        raise FileNotFoundError(
            f"Nessun CSV trovato in:\n{SPX_DIR}\nCon pattern: {SPX_CSV_GLOB}\n"
            "Lancia prima 05_SPX_minute_bars.py."
        )

    frames = []

    for file in files:
        print(f"    Reading SPX csv: {file.name}")

        raw = pd.read_csv(file, dtype=str, keep_default_na=False)

        date_col = detect_column(raw.columns, ["Date", "quote_date", "date"])
        time_col = detect_column(raw.columns, ["time", "quote_time"])
        open_col = detect_column(raw.columns, ["open", "Open"])
        close_col = detect_column(raw.columns, ["close", "Close"])

        for col, label in ((date_col, "Date"), (time_col, "time"),
                           (open_col, "open"), (close_col, "close")):
            if col is None:
                raise ValueError(
                    f"Non trovo la colonna {label} nel CSV SPX:\n{file}\n\n"
                    f"Colonne disponibili:\n{list(raw.columns)}"
                )

        work = pd.DataFrame({
            "quote_date": clean_date_series(raw[date_col]),
            "quote_time": raw[time_col].astype(str).str.strip(),
            "spx_open": pd.to_numeric(raw[open_col], errors="coerce"),
            "spx_close": pd.to_numeric(raw[close_col], errors="coerce"),
            "_source_file": file.name,
        })

        # I CSV SPX riportano l'ora come HH:MM; qui serve HH:MM:SS.
        work["quote_time"] = work["quote_time"].apply(
            lambda x: x if re.fullmatch(r"\d{2}:\d{2}:\d{2}", str(x)) else
            (str(x) + ":00" if re.fullmatch(r"\d{1,2}:\d{2}", str(x)) else None)
        )

        work = work[
            work["quote_date"].notna()
            & work["quote_time"].notna()
            & work["spx_open"].notna()
            & work["spx_close"].notna()
            & np.isfinite(work["spx_open"])
            & np.isfinite(work["spx_close"])
            & (work["spx_open"] > 0)
            & (work["spx_close"] > 0)
        ].copy()

        frames.append(work)

    out = pd.concat(frames, ignore_index=True)
    out = add_time_seconds(out)
    out["_session_end_seconds"] = out["quote_date"].map(session_end_seconds_for)

    rows_before = len(out)

    out = out[
        out["quote_date"].astype(str).str.startswith(str(YEAR))
        & out["_time_seconds"].notna()
        & (out["_time_seconds"] > SESSION_START_SECONDS)
        & (out["_time_seconds"] <= out["_session_end_seconds"])
    ].copy()

    rows_after = len(out)

    out = (
        out.sort_values(["quote_date", "_time_seconds"])
        .drop_duplicates(subset=["quote_date", "quote_time"], keep="last")
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError("Dopo la lettura dei CSV SPX non resta nessuna riga valida.")

    print(f"    SPX rows loaded: {len(out):,} "
          f"(scartate fuori sessione: {rows_before - rows_after:,})")
    print(f"    SPX days loaded: {out['quote_date'].nunique():,}")
    print(f"    SPX first date:  {out['quote_date'].min()}")
    print(f"    SPX last date:   {out['quote_date'].max()}")

    return out


def build_rv_block():
    spx = load_spx_from_trim_csv()

    spx = spx.sort_values(["quote_date", "_time_seconds"]).reset_index(drop=True)

    # (a) rendimento dentro la barra
    spx["_ret_within"] = np.log(spx["spx_close"] / spx["spx_open"])

    # (b) rendimento fra barre consecutive, dentro la giornata
    spx["_ret_consecutive"] = np.log(spx["spx_close"]) - np.log(
        spx.groupby("quote_date")["spx_close"].shift(1)
    )

    spx["_sq_within"] = spx["_ret_within"] ** 2
    spx["_sq_consecutive"] = spx["_ret_consecutive"] ** 2

    primary = "_sq_within" if RV_ESTIMATOR == "within_bar" else "_sq_consecutive"

    valid = spx[spx["_time_seconds"].notna()].copy()

    def window_sum(df, lo_exclusive, hi_col_or_value, name, count_name):
        if isinstance(hi_col_or_value, str):
            sel = df[(df["_time_seconds"] > lo_exclusive)
                     & (df["_time_seconds"] <= df[hi_col_or_value])]
        else:
            sel = df[(df["_time_seconds"] > lo_exclusive)
                     & (df["_time_seconds"] <= hi_col_or_value)]

        sel = sel[sel[primary].notna() & np.isfinite(sel[primary])]

        return (
            sel.groupby("quote_date", as_index=False)
            .agg(**{name: (primary, "sum"), count_name: (primary, "count")})
        )

    rv_t = window_sum(valid, SESSION_START_SECONDS, "_session_end_seconds",
                      "RV_t", "n_returns_full_session")

    spot_open = window_sum(valid, SESSION_START_SECONDS, OPEN_WINDOW_END_SECONDS,
                           "SpotRV_open", "n_returns_open_window")

    fwd_short = window_sum(valid, OPEN_WINDOW_END_SECONDS, "_session_end_seconds",
                           "FwdRV_short", "n_returns_fwd_short")

    # Con lo stimatore a chiusure consecutive FwdRV_full coincide con RV_t per costruzione: la
    # prima barra non ha una precedente, quindi il rendimento delle 09:31 non esiste ed entrambe
    # partono da quello delle 09:32. Il confronto stretto resta la definizione corretta in generale.
    fwd_full_sel = valid[
        (valid["_time_seconds"] > SURFACE_START_SECONDS)
        & (valid["_time_seconds"] <= valid["_session_end_seconds"])
    ]
    fwd_full_sel = fwd_full_sel[fwd_full_sel[primary].notna()
                                & np.isfinite(fwd_full_sel[primary])]

    fwd_full_block = (
        fwd_full_sel.groupby("quote_date", as_index=False)
        .agg(FwdRV_full=(primary, "sum"), n_returns_fwd_full=(primary, "count"))
    )

    # confronto fra i due stimatori sulla sessione piena
    alt = "_sq_consecutive" if primary == "_sq_within" else "_sq_within"

    alt_sel = valid[
        (valid["_time_seconds"] > SESSION_START_SECONDS)
        & (valid["_time_seconds"] <= valid["_session_end_seconds"])
    ]
    alt_sel = alt_sel[alt_sel[alt].notna() & np.isfinite(alt_sel[alt])]

    rv_alt = (
        alt_sel.groupby("quote_date", as_index=False)
        .agg(RV_t_alt_estimator=(alt, "sum"))
    )

    out = rv_t.merge(spot_open, on="quote_date", how="outer")
    out = out.merge(fwd_short_block := fwd_short, on="quote_date", how="outer")
    out = out.merge(fwd_full_block, on="quote_date", how="outer")
    out = out.merge(rv_alt, on="quote_date", how="outer")

    out["_date_dt"] = pd.to_datetime(out["quote_date"], errors="coerce")
    out = out.sort_values("_date_dt").reset_index(drop=True)

    out["RV_lag1"] = out["RV_t"].shift(1)
    out["RV_roll5"] = out["RV_t"].shift(1).rolling(window=5, min_periods=5).mean()
    out["RV_roll22"] = out["RV_t"].shift(1).rolling(window=22, min_periods=22).mean()

    out = out.drop(columns=["_date_dt"])

    return out


# ---------------------------------------------------------------------
# MFIV
# ---------------------------------------------------------------------
def load_mfiv() -> pd.DataFrame:
    require_file(MFIV_FILE)

    df = pd.read_csv(MFIV_FILE, dtype=str, keep_default_na=False)
    require_columns(df, ["quote_date", "quote_time", "mfiv", "mfiv_flag", "tau"],
                    "MFIV intraday file")

    df["quote_date"] = clean_date_series(df["quote_date"])
    df["mfiv"] = pd.to_numeric(df["mfiv"], errors="coerce")
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce")

    # Varianza implicita annualizzata, mfiv / tau. mfiv e' la varianza residua fino alla scadenza:
    # la sua media di sessione vale circa meta' di RV_t sotto volatilita' costante, mentre
    # mfiv / tau stima sigma^2 senza dipendere dal tempo residuo.
    df["mfiv_ann"] = np.where(
        df["tau"].notna() & (df["tau"] > 0) & df["mfiv"].notna(),
        df["mfiv"] / df["tau"],
        np.nan,
    )
    df = add_time_seconds(df)
    df["_session_end_seconds"] = df["quote_date"].map(session_end_seconds_for)

    return df


def build_mfiv_daily_mean(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["quote_date"].notna()
        & df["_time_seconds"].notna()
        & (df["_time_seconds"] >= SURFACE_START_SECONDS)
        & (df["_time_seconds"] <= df["_session_end_seconds"])
        & df["mfiv"].notna()
        & np.isfinite(df["mfiv"])
    )

    if USE_ONLY_OK_MFIV_ROWS:
        mask = mask & (df["mfiv_flag"] == "ok")

    out = (
        df[mask].groupby("quote_date", as_index=False)
        .agg(MFIV_t=("mfiv", "mean"),
             MFIV_ann_t=("mfiv_ann", "mean"),
             tau_open=("tau", "max"),
             n_mfiv_ok_rows=("mfiv", "count"))
    )

    # MFIV_session_t = MFIV_ann_t * tau di apertura: varianza implicita alla scala della sessione,
    # con benchmark Mincer-Zarnowitz (0, 1) sia in livelli sia in log. Sulle early close il tau
    # corretto entra da solo, perche' tau_open e' letto dal dato.
    out["MFIV_session_t"] = out["MFIV_ann_t"] * out["tau_open"]

    return out


def build_mfiv_0931_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_distance_to_0931"] = (work["_time_seconds"] - SNAPSHOT_SECONDS).abs()

    rows = []

    for quote_date, day in work.groupby("quote_date", sort=True):
        valid = day[
            day["mfiv"].notna()
            & np.isfinite(day["mfiv"])
            & (day["mfiv_flag"] == "ok")
            & day["_time_seconds"].notna()
        ]

        exact = valid[valid["quote_time"] == SNAPSHOT_TIME]

        if not exact.empty:
            selected, status = exact.iloc[0], "exact_0931_ok"
        else:
            nearby = valid[valid["_distance_to_0931"] <= SNAPSHOT_TOLERANCE_SECONDS]

            if nearby.empty:
                rows.append({
                    "quote_date": quote_date, "MFIV_0931": np.nan,
                    "MFIV_ann_0931": np.nan,
                    "mfiv_0931_quote_time": "", "mfiv_0931_status": "missing_ok_within_2min",
                    "mfiv_0931_seconds_from_0931": np.nan,
                })
                continue

            selected = nearby.sort_values("_distance_to_0931").iloc[0]
            status = "nearest_ok_within_2min"

        rows.append({
            "quote_date": quote_date,
            "MFIV_0931": float(selected["mfiv"]),
            "MFIV_ann_0931": float(selected["mfiv_ann"])
                if pd.notna(selected.get("mfiv_ann", np.nan)) else np.nan,
            "mfiv_0931_quote_time": selected["quote_time"],
            "mfiv_0931_status": status,
            "mfiv_0931_seconds_from_0931": int(selected["_time_seconds"] - SNAPSHOT_SECONDS),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# GAMMA, ENTRAMBE LE FONTI
# ---------------------------------------------------------------------
def load_gamma(file_path, h_oi_col: str, h_vol_col: str,
               g_oi_col: str, g_vol_col: str, label: str) -> pd.DataFrame:
    require_file(file_path)

    df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    require_columns(
        df,
        ["quote_date", "quote_time", h_oi_col, h_vol_col, g_oi_col, g_vol_col, "gamma_flag"],
        label
    )

    df["quote_date"] = clean_date_series(df["quote_date"])

    for c in (h_oi_col, h_vol_col, g_oi_col, g_vol_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_time_seconds(df)
    df["_session_end_seconds"] = df["quote_date"].map(session_end_seconds_for)

    return df


def gamma_daily_mean(df, h_oi_col, h_vol_col, g_oi_col, g_vol_col, suffix) -> pd.DataFrame:
    mask = (
        df["quote_date"].notna()
        & df["_time_seconds"].notna()
        & (df["_time_seconds"] >= SURFACE_START_SECONDS)
        & (df["_time_seconds"] <= df["_session_end_seconds"])
    )

    if USE_ONLY_OK_GAMMA_ROWS:
        mask = mask & (df["gamma_flag"] == "ok")

    return (
        df[mask].groupby("quote_date", as_index=False)
        .agg(**{
            f"H_OI_t{suffix}": (h_oi_col, "mean"),
            f"H_VOL_t{suffix}": (h_vol_col, "mean"),
            f"G_OI_raw_t{suffix}": (g_oi_col, "mean"),
            f"G_VOL_raw_t{suffix}": (g_vol_col, "mean"),
            f"n_gamma_ok_rows{suffix}": (h_oi_col, "count"),
        })
    )


def gamma_0931_snapshot(df, h_oi_col, h_vol_col, suffix) -> pd.DataFrame:
    work = df.copy()
    work["_distance_to_0931"] = (work["_time_seconds"] - SNAPSHOT_SECONDS).abs()

    rows = []

    for quote_date, day in work.groupby("quote_date", sort=True):
        valid = day[
            day[h_oi_col].notna() & day[h_vol_col].notna()
            & np.isfinite(day[h_oi_col]) & np.isfinite(day[h_vol_col])
            & (day["gamma_flag"] == "ok")
            & day["_time_seconds"].notna()
        ]

        exact = valid[valid["quote_time"] == SNAPSHOT_TIME]

        if not exact.empty:
            selected, status = exact.iloc[0], "exact_0931_ok"
        else:
            nearby = valid[valid["_distance_to_0931"] <= SNAPSHOT_TOLERANCE_SECONDS]

            if nearby.empty:
                rows.append({
                    "quote_date": quote_date,
                    f"H_OI_0931{suffix}": np.nan, f"H_VOL_0931{suffix}": np.nan,
                    f"gamma_0931_quote_time{suffix}": "",
                    f"gamma_0931_status{suffix}": "missing_ok_within_2min",
                    f"gamma_0931_seconds_from_0931{suffix}": np.nan,
                })
                continue

            selected = nearby.sort_values("_distance_to_0931").iloc[0]
            status = "nearest_ok_within_2min"

        rows.append({
            "quote_date": quote_date,
            f"H_OI_0931{suffix}": float(selected[h_oi_col]),
            f"H_VOL_0931{suffix}": float(selected[h_vol_col]),
            f"gamma_0931_quote_time{suffix}": selected["quote_time"],
            f"gamma_0931_status{suffix}": status,
            f"gamma_0931_seconds_from_0931{suffix}": int(
                selected["_time_seconds"] - SNAPSHOT_SECONDS),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# METADATI E VALIDAZIONE
# ---------------------------------------------------------------------
def build_metadata(dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"quote_date": sorted(dates.dropna().astype(str).unique())})
    out["is_early_close"] = out["quote_date"].isin(EARLY_CLOSE_DATES).astype(int)
    out["is_opex"] = out["quote_date"].apply(third_friday_flag).astype(int)
    return out


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check, expected, actual, passed, details=""):
        rows.append({"check": check, "expected": expected, "actual": actual,
                     "passed": int(passed), "details": details})

    add("row_count_exactly_expected", EXPECTED_ROW_COUNT, len(panel),
        len(panel) == EXPECTED_ROW_COUNT)

    for col, name in (("n_mfiv_ok_rows", "mfiv"), ("n_gamma_ok_rows", "gamma"),
                      ("n_gamma_ok_rows_indep", "gamma_indep")):
        bad = panel[panel[col].isna() | (pd.to_numeric(panel[col], errors="coerce") == 0)]
        add(f"no_day_has_zero_{name}_ok_rows", 0, len(bad), len(bad) == 0,
            ";".join(bad["quote_date"].astype(str).tolist()[:20]))

    ps = panel.sort_values("quote_date").reset_index(drop=True)

    for col, k in (("RV_lag1", 1), ("RV_roll5", 5), ("RV_roll22", 22)):
        expected_nan = pd.Series(False, index=ps.index)
        expected_nan.iloc[:k] = True
        bad = ps[ps[col].isna() != expected_nan]
        add(f"{col}_NaN_only_first_{k}_trading_days", 0, len(bad), len(bad) == 0,
            ";".join(bad["quote_date"].astype(str).tolist()[:20]))

    # FwdRV_full deve coincidere con RV_t (vedi build_rv_block): se diverge, e' cambiata la
    # definizione delle finestre.
    rv = pd.to_numeric(panel["RV_t"], errors="coerce")
    fwd = pd.to_numeric(panel["FwdRV_full"], errors="coerce")
    n_ok = pd.to_numeric(panel["n_returns_full_session"], errors="coerce")
    n_fw = pd.to_numeric(panel["n_returns_fwd_full"], errors="coerce")

    diverge = panel[~np.isclose(rv, fwd, atol=1e-18, rtol=0)]
    add("FwdRV_full_identical_to_RV_t_under_consecutive_returns", 0, len(diverge),
        len(diverge) == 0,
        "attesa: identiche, perche' il rendimento delle 09:31 non esiste. "
        + ";".join(diverge["quote_date"].astype(str).tolist()[:20]))

    bad_count = panel[n_fw != n_ok]
    add("n_returns_fwd_full_equals_full_session", 0, len(bad_count),
        len(bad_count) == 0,
        ";".join(bad_count["quote_date"].astype(str).tolist()[:20]))

    # identita' di scomposizione: SpotRV_open + FwdRV_short = RV_t
    resid = (pd.to_numeric(panel["SpotRV_open"], errors="coerce")
             + pd.to_numeric(panel["FwdRV_short"], errors="coerce") - rv).abs()
    bad_dec = panel[resid > 1e-15]
    add("SpotRV_open_plus_FwdRV_short_equals_RV_t", 0, len(bad_dec),
        len(bad_dec) == 0,
        ";".join(bad_dec["quote_date"].astype(str).tolist()[:20]))

    # controllo sulle finestre delle early close
    ec = panel[panel["is_early_close"] == 1]
    n_ec = pd.to_numeric(ec["n_returns_full_session"], errors="coerce")
    add("early_close_returns_around_210", "~210",
        ";".join(n_ec.astype(str).tolist()), bool((n_ec <= 215).all()),
        "con la chiusura fissa a 16:00 erano 230")

    for label, col in (("mfiv", "mfiv_0931_status"), ("gamma", "gamma_0931_status")):
        fb = panel[panel[col].astype(str) == "nearest_ok_within_2min"]
        ms = panel[panel[col].astype(str) == "missing_ok_within_2min"]
        add(f"{label}_0931_fallback_days", "diagnostic", len(fb), True,
            ";".join(fb["quote_date"].astype(str).tolist()[:20]))
        add(f"{label}_0931_missing_days", 0, len(ms), len(ms) == 0,
            ";".join(ms["quote_date"].astype(str).tolist()[:20]))

    return pd.DataFrame(rows)


def daily_regression_panel():
    print("=" * 120)
    print("BUILD DAILY REGRESSION PANEL 2025")
    print("=" * 120)
    print(f"DATA_DIR:            {DATA_DIR}")
    print(f"MFIV file:           {MFIV_FILE}")
    print(f"Gamma CBOE file:     {GAMMA_CBOE_FILE}")
    print(f"Gamma indep file:    {GAMMA_INDEP_FILE}")
    print(f"SPX folder:          {SPX_DIR}  (CSV di 05_spx_minute_bars, non Excel)")
    print(f"Gamma PRIMARIO:      {GAMMA_PRIMARY_SOURCE}")
    print(f"Stimatore RV:        {RV_ESTIMATOR}")
    print(f"Chiusura sessione:   default {REGULAR_CLOSE}, "
          f"early close {sorted(EARLY_CLOSE_TIMES)}")
    print(f"Output panel:        {OUTPUT_PANEL_FILE}")
    print("=" * 120)

    print("Building RV block from the SPX minute CSV files...")
    rv_block = build_rv_block()
    print(f"RV block rows: {len(rv_block)}")

    print("Loading MFIV...")
    mfiv_df = load_mfiv()
    mfiv_mean = build_mfiv_daily_mean(mfiv_df)
    mfiv_0931 = build_mfiv_0931_snapshot(mfiv_df)
    print(f"MFIV mean rows: {len(mfiv_mean)} | 09:31 rows: {len(mfiv_0931)}")

    print("Loading CBOE gamma...")
    g_cboe = load_gamma(GAMMA_CBOE_FILE, "h_oi_cboe", "h_vol_cboe",
                        "g_oi_raw_cboe", "g_vol_raw_cboe", "CBOE gamma intraday file")

    print("Loading independent gamma...")
    g_indep = load_gamma(GAMMA_INDEP_FILE, "h_oi", "h_vol",
                         "g_oi_raw", "g_vol_raw", "independent gamma intraday file")

    if GAMMA_PRIMARY_SOURCE == "cboe":
        primary_mean = gamma_daily_mean(g_cboe, "h_oi_cboe", "h_vol_cboe",
                                        "g_oi_raw_cboe", "g_vol_raw_cboe", "")
        other_mean = gamma_daily_mean(g_indep, "h_oi", "h_vol",
                                      "g_oi_raw", "g_vol_raw", "_indep")
        primary_0931 = gamma_0931_snapshot(g_cboe, "h_oi_cboe", "h_vol_cboe", "")
        other_0931 = gamma_0931_snapshot(g_indep, "h_oi", "h_vol", "_indep")
    else:
        primary_mean = gamma_daily_mean(g_indep, "h_oi", "h_vol",
                                        "g_oi_raw", "g_vol_raw", "")
        other_mean = gamma_daily_mean(g_cboe, "h_oi_cboe", "h_vol_cboe",
                                      "g_oi_raw_cboe", "g_vol_raw_cboe", "_cboe")
        primary_0931 = gamma_0931_snapshot(g_indep, "h_oi", "h_vol", "")
        other_0931 = gamma_0931_snapshot(g_cboe, "h_oi_cboe", "h_vol_cboe", "_cboe")

    other_suffix = "_indep" if GAMMA_PRIMARY_SOURCE == "cboe" else "_cboe"

    print(f"Gamma primary rows: {len(primary_mean)} | other rows: {len(other_mean)}")

    metadata = build_metadata(rv_block["quote_date"])

    panel = rv_block.copy()

    for block in (mfiv_mean, primary_mean, other_mean, mfiv_0931, primary_0931, other_0931):
        panel = panel.merge(block, on="quote_date", how="outer")

    panel = panel.merge(metadata, on="quote_date", how="left")

    panel["_date_dt"] = pd.to_datetime(panel["quote_date"], errors="coerce")
    panel = panel.sort_values("_date_dt").drop(columns=["_date_dt"]).reset_index(drop=True)

    final_columns = [
        "quote_date",
        "RV_t", "RV_lag1", "RV_roll5", "RV_roll22",
        "SpotRV_open", "FwdRV_short", "FwdRV_full",
        "RV_t_alt_estimator",
        "MFIV_t", "MFIV_ann_t", "MFIV_session_t", "tau_open",
        "H_OI_t", "H_VOL_t", "G_OI_raw_t", "G_VOL_raw_t",
        f"H_OI_t{other_suffix}", f"H_VOL_t{other_suffix}",
        f"G_OI_raw_t{other_suffix}", f"G_VOL_raw_t{other_suffix}",
        "MFIV_0931", "MFIV_ann_0931", "H_OI_0931", "H_VOL_0931",
        f"H_OI_0931{other_suffix}", f"H_VOL_0931{other_suffix}",
        "is_early_close", "is_opex",
        "n_mfiv_ok_rows", "n_gamma_ok_rows", f"n_gamma_ok_rows{other_suffix}",
        "n_returns_full_session", "n_returns_open_window",
        "n_returns_fwd_short", "n_returns_fwd_full",
        "mfiv_0931_quote_time", "mfiv_0931_status", "mfiv_0931_seconds_from_0931",
        "gamma_0931_quote_time", "gamma_0931_status", "gamma_0931_seconds_from_0931",
    ]

    # n_gamma_ok_rows_indep e' letta da validate_panel
    if other_suffix == "_indep" and "n_gamma_ok_rows_indep" not in panel.columns:
        panel["n_gamma_ok_rows_indep"] = np.nan

    for col in final_columns:
        if col not in panel.columns:
            panel[col] = np.nan

    panel = panel[final_columns].copy()

    validation = validate_panel(panel)

    panel.to_csv(OUTPUT_PANEL_FILE, index=False)
    validation.to_csv(OUTPUT_VALIDATION_FILE, index=False)

    rv = pd.to_numeric(panel["RV_t"], errors="coerce")
    rv_alt = pd.to_numeric(panel["RV_t_alt_estimator"], errors="coerce")

    print()
    print("=" * 120)
    print("COMPLETED DAILY REGRESSION PANEL")
    print("=" * 120)
    print(f"Rows in final panel:       {len(panel)}   (attese {EXPECTED_ROW_COUNT})")
    print()
    print("--- STIMATORE DELLA REALIZED VARIANCE: CONFRONTO ---")
    print(f"primario ({RV_ESTIMATOR}):  mediana {rv.median():.6e}  "
          f"vol annualizzata {np.sqrt(rv.median() * 252):.4f}")
    print(f"alternativo:                mediana {rv_alt.median():.6e}  "
          f"vol annualizzata {np.sqrt(rv_alt.median() * 252):.4f}")
    print(f"rapporto alt/primario:      {rv_alt.median() / rv.median():.4f}")
    print("Il rendimento dentro la barra ignora il movimento fra la chiusura di una")
    print("barra e l'apertura della successiva, quindi sottostima. Lo standard di")
    print("letteratura e' il rendimento fra chiusure consecutive. Se il rapporto e'")
    print("sensibilmente sopra 1, la scelta dello stimatore va decisa esplicitamente.")
    print()
    print("--- FATTORE DI COSTRUZIONE FRA MFIV_t E RV_t ---")
    _m = pd.to_numeric(panel["MFIV_t"], errors="coerce")
    _ma = pd.to_numeric(panel["MFIV_ann_t"], errors="coerce")
    _r = pd.to_numeric(panel["RV_t"], errors="coerce")
    print(f"MFIV_t mediana      {_m.median():.4e}   RV_t mediana {_r.median():.4e}   "
          f"rapporto {_r.median() / _m.median():.4f}")
    print(f"MFIV_ann_t mediana  {_ma.median():.4e}")
    _ms = pd.to_numeric(panel["MFIV_session_t"], errors="coerce")
    print(f"MFIV_session_t mediana {_ms.median():.4e}   rapporto RV_t/MFIV_session_t "
          f"{(_r.median() / _ms.median()):.4f}")
    print("MFIV_session_t = MFIV_ann_t * tau_apertura riporta la varianza implicita")
    print("alla scala della sessione: e' su questa che il benchmark Mincer-Zarnowitz")
    print("e' (0,1) in entrambe le forme, senza aggiustamenti.")
    print("MFIV_t e' la media di sessione della varianza RESIDUA, quindi il suo")
    print("rapporto con RV_t contiene un fattore di costruzione vicino a 2 e il")
    print("benchmark di uno slope non distorto NON e' 1. MFIV_ann_t = mfiv/tau stima")
    print("sigma^2 direttamente e riporta il benchmark a 1.")
    print()
    print("--- FwdRV_full CONTRO RV_t ---")
    _rv = pd.to_numeric(panel["RV_t"], errors="coerce")
    _fw = pd.to_numeric(panel["FwdRV_full"], errors="coerce")
    print(f"giornate in cui coincidono: "
          f"{int(np.isclose(_rv, _fw, atol=1e-18, rtol=0).sum())} su {len(panel)}  "
          f"(attese: tutte)")
    print("Sotto lo stimatore a chiusure consecutive le due colonne sono identiche")
    print("per costruzione: il rendimento delle 09:31 non esiste, perche' la prima")
    print("barra non ha una precedente, quindi entrambe le finestre partono dal")
    print("rendimento delle 09:32. Design B-full ha percio' la STESSA variabile")
    print("dipendente di Design A, e i due disegni differiscono solo nel regressore.")
    print("")
    print()
    print("--- FINESTRE SULLE EARLY CLOSE ---")
    ec = panel[panel["is_early_close"] == 1]
    print(ec[["quote_date", "n_returns_full_session", "RV_t", "MFIV_t",
              "n_mfiv_ok_rows", "n_gamma_ok_rows"]].to_string(index=False))
    print()
    print("--- CONFRONTO FRA LE DUE FONTI DEL GAMMA ---")
    h1 = pd.to_numeric(panel["H_OI_t"], errors="coerce")
    h2 = pd.to_numeric(panel[f"H_OI_t{other_suffix}"], errors="coerce")
    print(f"H_OI_t primario ({GAMMA_PRIMARY_SOURCE}): mediana {h1.median():.6f}")
    print(f"H_OI_t{other_suffix}: mediana {h2.median():.6f}")
    print(f"correlazione fra le due serie: {h1.corr(h2):.6f}")
    print()
    print("Validation checks:")
    print(validation.to_string(index=False))
    print("=" * 120)


# =====================================================================
# PARTE 2: VARIANTI CON BANDA DI LOG-MONEYNESS
# =====================================================================
EXPECTED_BASELINE_ROWS = 250

# Pannello di base: quello scritto dalla parte 1, riletto dal disco.
BASELINE_PANEL_FILE = OUTPUT_PANEL_FILE

VARIANTS = [
    {
        "label": "logm_005",
        "cutoff": 0.05,
        "mfiv_folder": "03_mfiv",
        "mfiv_file": "03_mfiv_intraday_2025_all_logm_005.csv",
    },
    {
        "label": "logm_002",
        "cutoff": 0.02,
        "mfiv_folder": "03_mfiv",
        "mfiv_file": "03_mfiv_intraday_2025_all_logm_002.csv",
    },
]

OUTPUT_FILE_PREFIX = "07_"
OVERWRITE_OUTPUTS = True

# Tetto della finestra, non selezione dell'ultimo timestamp: la media prende tutte le righe fino
# a qui. Sulle early close non ci sono righe dopo le 13:00, quindi il tetto fisso non ha effetto.
SURFACE_END_TIME = REGULAR_CLOSE  # "16:00:00"
SURFACE_END_SECONDS = time_string_to_seconds(SURFACE_END_TIME)

# ---------------------------------------------------------------------
# COLONNE SOSTITUITE
# ---------------------------------------------------------------------
# Colonne prese dalla variante. Le derivate (sotto) si ricalcolano dalla MFIV della variante,
# cosi' nessuna riga mescola due regole di troncamento; tau_open resta quello della baseline,
# perche' e' una proprieta' della sessione.
MFIV_REPLACEMENT_COLUMNS = [
    "MFIV_t",
    "n_mfiv_ok_rows",
    "MFIV_0931",
    "mfiv_0931_quote_time",
    "mfiv_0931_status",
    "mfiv_0931_seconds_from_0931",
]

# Colonne derivate dalla MFIV, ricalcolate invece che sostituite.
MFIV_DERIVED_COLUMNS = [
    "MFIV_ann_t",
    "MFIV_session_t",
    "MFIV_ann_0931",
]


def require_file_logm(path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def require_columns_logm(df: pd.DataFrame, columns: list[str], label: str):
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def safe_write_csv(df: pd.DataFrame, path):
    # Versione locale: con OVERWRITE_OUTPUTS = False si ferma invece di
    # sovrascrivere (quella di common.py sovrascriverebbe comunque).
    if path.exists():
        if OVERWRITE_OUTPUTS:
            path.unlink()
        else:
            raise FileExistsError(f"Output already exists: {path}")

    df.to_csv(path, index=False)


def check_unique_key(df: pd.DataFrame, columns: list[str], label: str):
    duplicated = df.duplicated(columns, keep=False)

    if duplicated.any():
        examples = (
            df.loc[duplicated, columns]
            .sort_values(columns)
            .head(100)
        )
        raise ValueError(
            f"{label} has duplicate keys {columns}.\n"
            f"First examples:\n{examples.to_string(index=False)}"
        )


def values_identical_as_strings(
    baseline: pd.Series,
    variant: pd.Series,
) -> pd.Series:
    left = baseline.fillna("").astype(str)
    right = variant.fillna("").astype(str)
    return left == right


def add_validation_row(
    rows: list[dict],
    variant_label: str,
    check: str,
    expected,
    actual,
    passed: bool,
    details: str = "",
):
    rows.append({
        "variant": variant_label,
        "check": check,
        "expected": expected,
        "actual": actual,
        "passed": int(bool(passed)),
        "details": details,
    })


# ---------------------------------------------------------------------
# AGGREGAZIONE GIORNALIERA DELLA MFIV DELLA VARIANTE
# ---------------------------------------------------------------------
def load_intraday_mfiv(
    mfiv_file,
    variant_label: str,
    expected_cutoff: float,
) -> pd.DataFrame:
    require_file_logm(mfiv_file, f"MFIV intraday variant {variant_label}")

    df = pd.read_csv(
        mfiv_file,
        dtype=str,
        keep_default_na=False,
    )

    required = [
        "quote_date",
        "quote_time",
        "mfiv",
        # tau serve a ricalcolare MFIV_ann: un file senza tau fallisce subito
        # invece di produrre colonne vuote.
        "tau",
        "mfiv_flag",
        "log_moneyness_cutoff_applied",
        "mfiv_variant",
    ]
    require_columns_logm(df, required, f"MFIV variant {variant_label}")

    df["quote_date"] = clean_date_series(df["quote_date"])
    df["quote_time"] = df["quote_time"].astype(str).str.strip()
    df["mfiv"] = pd.to_numeric(df["mfiv"], errors="coerce")
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce")
    df["log_moneyness_cutoff_applied"] = pd.to_numeric(
        df["log_moneyness_cutoff_applied"],
        errors="coerce",
    )
    df = add_time_seconds(df)

    check_unique_key(
        df,
        ["quote_date", "quote_time"],
        f"MFIV variant {variant_label}",
    )

    labels = sorted(
        df["mfiv_variant"].astype(str).dropna().unique().tolist()
    )
    if labels != [variant_label]:
        raise ValueError(
            f"Variant label mismatch for {variant_label}: {labels}"
        )

    cutoff_values = df["log_moneyness_cutoff_applied"]
    cutoff_ok = (
        cutoff_values.notna().all()
        and np.allclose(
            cutoff_values.to_numpy(dtype=float),
            expected_cutoff,
            rtol=0.0,
            atol=1e-15,
        )
    )
    if not cutoff_ok:
        raise ValueError(
            f"Cutoff tracking mismatch for {variant_label}. "
            f"Expected {expected_cutoff}, observed "
            f"{cutoff_values.dropna().unique().tolist()}"
        )

    return df


def build_mfiv_daily_mean_logm(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["quote_date"].notna()
        & df["_time_seconds"].notna()
        & (df["_time_seconds"] >= SURFACE_START_SECONDS)
        & (df["_time_seconds"] <= SURFACE_END_SECONDS)
        & df["mfiv"].notna()
        & np.isfinite(df["mfiv"])
    )

    if USE_ONLY_OK_MFIV_ROWS:
        mask = mask & (df["mfiv_flag"] == "ok")

    clean = df[mask].copy()

    clean["_mfiv_ann"] = np.where(
        clean["tau"] > 0, clean["mfiv"] / clean["tau"], np.nan
    )

    clean = clean[clean["_mfiv_ann"].notna() & np.isfinite(clean["_mfiv_ann"])]

    return (
        clean.groupby("quote_date", as_index=False)
        .agg(
            MFIV_t=("mfiv", "mean"),
            n_mfiv_ok_rows=("mfiv", "count"),
            # media di mfiv/tau sulle stesse righe, come MFIV_ann_t nel pannello di base
            # (non MFIV_t / tau medio: la media dei rapporti non e' il rapporto delle medie)
            MFIV_ann_t=("_mfiv_ann", "mean"),
        )
    )


def build_mfiv_0931_snapshot_logm(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_distance_to_0931"] = (
        work["_time_seconds"] - SNAPSHOT_SECONDS
    ).abs()

    rows = []

    for quote_date, day in work.groupby("quote_date", sort=True):
        valid = day[
            day["mfiv"].notna()
            & np.isfinite(day["mfiv"])
            & (day["mfiv_flag"] == "ok")
            & day["_time_seconds"].notna()
        ].copy()

        exact = valid[valid["quote_time"] == SNAPSHOT_TIME].copy()

        if not exact.empty:
            selected = exact.iloc[0]
            status = "exact_0931_ok"
        else:
            nearby = valid[
                valid["_distance_to_0931"]
                <= SNAPSHOT_TOLERANCE_SECONDS
            ].copy()

            if not nearby.empty:
                nearby = nearby.sort_values("_distance_to_0931")
                selected = nearby.iloc[0]
                status = "nearest_ok_within_2min"
            else:
                rows.append({
                    "quote_date": quote_date,
                    "MFIV_0931": np.nan,
                    "MFIV_ann_0931": np.nan,
                    "mfiv_0931_quote_time": "",
                    "mfiv_0931_status": "missing_ok_within_2min",
                    "mfiv_0931_seconds_from_0931": np.nan,
                })
                continue

        _tau = float(selected["tau"]) if pd.notna(selected["tau"]) else np.nan

        rows.append({
            "quote_date": quote_date,
            "MFIV_0931": float(selected["mfiv"]),
            # stessa trasformazione del pannello di base
            "MFIV_ann_0931": (float(selected["mfiv"]) / _tau
                              if _tau and _tau > 0 else np.nan),
            "mfiv_0931_quote_time": selected["quote_time"],
            "mfiv_0931_status": status,
            "mfiv_0931_seconds_from_0931": int(
                selected["_time_seconds"] - SNAPSHOT_SECONDS
            ),
        })

    return pd.DataFrame(rows)


def build_daily_mfiv_block(
    intraday: pd.DataFrame,
    variant_label: str,
    cutoff: float,
) -> pd.DataFrame:
    daily_mean = build_mfiv_daily_mean_logm(intraday)
    snapshot = build_mfiv_0931_snapshot_logm(intraday)

    block = daily_mean.merge(
        snapshot,
        on="quote_date",
        how="outer",
        validate="one_to_one",
    )

    block["mfiv_variant"] = variant_label
    block["log_moneyness_cutoff_applied"] = cutoff
    block = block.sort_values("quote_date").reset_index(drop=True)

    check_unique_key(block, ["quote_date"], f"Daily MFIV block {variant_label}")
    return block


# ---------------------------------------------------------------------
# SOSTITUZIONE DEL BLOCCO MFIV NEL PANNELLO DI BASE
# ---------------------------------------------------------------------
def build_variant_panel(
    baseline_panel: pd.DataFrame,
    daily_block: pd.DataFrame,
    variant_label: str,
) -> pd.DataFrame:
    baseline_columns = list(baseline_panel.columns)

    require_columns_logm(
        baseline_panel,
        ["quote_date"] + MFIV_REPLACEMENT_COLUMNS,
        "Baseline daily regression panel",
    )

    required_daily = ["quote_date"] + MFIV_REPLACEMENT_COLUMNS
    require_columns_logm(
        daily_block,
        required_daily,
        f"Daily MFIV block {variant_label}",
    )

    # Si sostituiscono anche le colonne derivate dalla MFIV presenti nel pannello,
    # altrimenti la stessa riga mescolerebbe due regole di troncamento.
    derived_present = [
        c for c in MFIV_DERIVED_COLUMNS
        if c in baseline_panel.columns
    ]

    untouched = baseline_panel.drop(
        columns=MFIV_REPLACEMENT_COLUMNS + derived_present
    ).copy()

    replacement = daily_block[required_daily].copy()

    if "MFIV_ann_t" in derived_present:
        replacement["MFIV_ann_t"] = daily_block["MFIV_ann_t"].to_numpy()

    if "MFIV_ann_0931" in derived_present:
        replacement["MFIV_ann_0931"] = daily_block["MFIV_ann_0931"].to_numpy()

    output = untouched.merge(
        replacement,
        on="quote_date",
        how="left",
        validate="one_to_one",
    )

    # MFIV_session_t = MFIV_ann_t * tau_open, con tau_open della baseline: e' la
    # lunghezza della sessione in anni e non dipende dagli strike integrati.
    if "MFIV_session_t" in derived_present:
        if "tau_open" not in output.columns:
            raise ValueError(
                "MFIV_session_t va ricalcolata ma tau_open non e' nel pannello."
            )

        tau_open = pd.to_numeric(output["tau_open"], errors="coerce")
        mfiv_ann = pd.to_numeric(output["MFIV_ann_t"], errors="coerce")

        output["MFIV_session_t"] = mfiv_ann * tau_open

    # Ripristina esattamente ordine e schema delle colonne della baseline.
    missing_back = [c for c in baseline_columns if c not in output.columns]

    if missing_back:
        raise ValueError(
            f"Colonne perse nella sostituzione: {missing_back}"
        )

    output = output[baseline_columns].copy()
    return output


# ---------------------------------------------------------------------
# VALIDAZIONE E CONFRONTO
# ---------------------------------------------------------------------
def validate_variant_panel(
    baseline_panel: pd.DataFrame,
    variant_panel: pd.DataFrame,
    daily_block: pd.DataFrame,
    variant_label: str,
    cutoff: float,
) -> pd.DataFrame:
    rows = []

    add_validation_row(
        rows,
        variant_label,
        "row_count_matches_baseline",
        len(baseline_panel),
        len(variant_panel),
        len(baseline_panel) == len(variant_panel),
    )

    add_validation_row(
        rows,
        variant_label,
        "row_count_matches_expected_250",
        EXPECTED_BASELINE_ROWS,
        len(variant_panel),
        len(variant_panel) == EXPECTED_BASELINE_ROWS,
    )

    baseline_dates = baseline_panel["quote_date"].astype(str).tolist()
    variant_dates = variant_panel["quote_date"].astype(str).tolist()
    add_validation_row(
        rows,
        variant_label,
        "quote_date_sequence_identical",
        "identical",
        "identical" if baseline_dates == variant_dates else "different",
        baseline_dates == variant_dates,
    )

    add_validation_row(
        rows,
        variant_label,
        "column_sequence_identical",
        "identical",
        (
            "identical"
            if list(baseline_panel.columns) == list(variant_panel.columns)
            else "different"
        ),
        list(baseline_panel.columns) == list(variant_panel.columns),
    )

    # Escluse anche le colonne derivate, che la sostituzione ricalcola apposta; che siano
    # davvero cambiate lo verifica il controllo <colonna>_differs_from_baseline piu' sotto.
    non_mfiv_columns = [
        col for col in baseline_panel.columns
        if col not in MFIV_REPLACEMENT_COLUMNS + MFIV_DERIVED_COLUMNS
    ]

    mismatch_counts = {}
    for col in non_mfiv_columns:
        equal = values_identical_as_strings(
            baseline_panel[col],
            variant_panel[col],
        )
        mismatch_counts[col] = int((~equal).sum())

    non_mfiv_mismatches = sum(mismatch_counts.values())
    bad_columns = [
        f"{col}:{count}"
        for col, count in mismatch_counts.items()
        if count > 0
    ]
    add_validation_row(
        rows,
        variant_label,
        "all_columns_outside_mfiv_block_identical_to_baseline",
        0,
        non_mfiv_mismatches,
        non_mfiv_mismatches == 0,
        ";".join(bad_columns[:30]),
    )

    n_ok_numeric = pd.to_numeric(
        variant_panel["n_mfiv_ok_rows"],
        errors="coerce",
    )
    zero_or_missing = variant_panel[
        n_ok_numeric.isna() | (n_ok_numeric == 0)
    ].copy()
    add_validation_row(
        rows,
        variant_label,
        "days_with_zero_or_missing_ok_rows",
        "diagnostic",
        len(zero_or_missing),
        True,
        ";".join(zero_or_missing["quote_date"].astype(str).tolist()[:30]),
    )

    missing_snapshot = variant_panel[
        variant_panel["mfiv_0931_status"].astype(str)
        == "missing_ok_within_2min"
    ].copy()
    add_validation_row(
        rows,
        variant_label,
        "days_missing_ok_MFIV_0931_within_2min",
        "diagnostic",
        len(missing_snapshot),
        True,
        ";".join(missing_snapshot["quote_date"].astype(str).tolist()[:30]),
    )

    fallback_snapshot = variant_panel[
        variant_panel["mfiv_0931_status"].astype(str)
        == "nearest_ok_within_2min"
    ].copy()
    add_validation_row(
        rows,
        variant_label,
        "days_using_MFIV_0931_fallback",
        "diagnostic",
        len(fallback_snapshot),
        True,
        ";".join(fallback_snapshot["quote_date"].astype(str).tolist()[:30]),
    )

    # Le colonne derivate devono venire dalla MFIV della variante: si verifica che differiscano
    # dalla baseline e che MFIV_session_t = MFIV_ann_t * tau_open.
    for col in MFIV_DERIVED_COLUMNS:
        if col not in variant_panel.columns:
            continue

        v = pd.to_numeric(variant_panel[col], errors="coerce")
        b = pd.to_numeric(baseline_panel[col], errors="coerce")

        n_identical = int((v.round(15) == b.round(15)).sum())

        add_validation_row(
            rows,
            variant_label,
            f"{col}_differs_from_baseline",
            "< 250",
            n_identical,
            n_identical < len(variant_panel),
            "Se la colonna derivata coincidesse riga per riga con la baseline, "
            "la sostituzione non avrebbe avuto luogo e il pannello "
            "mescolerebbe due regole di troncamento.",
        )

    if ("MFIV_session_t" in variant_panel.columns
            and "MFIV_ann_t" in variant_panel.columns
            and "tau_open" in variant_panel.columns):

        lhs = pd.to_numeric(variant_panel["MFIV_session_t"], errors="coerce")
        rhs = (pd.to_numeric(variant_panel["MFIV_ann_t"], errors="coerce")
               * pd.to_numeric(variant_panel["tau_open"], errors="coerce"))

        gap = (lhs - rhs).abs()
        n_bad = int((gap > 1e-18).sum())

        add_validation_row(
            rows,
            variant_label,
            "MFIV_session_equals_ann_times_tau_open",
            0,
            n_bad,
            n_bad == 0,
            f"Scarto massimo {float(gap.max()):.3e}." if len(gap) else "",
        )

    mfiv_t = pd.to_numeric(variant_panel["MFIV_t"], errors="coerce")
    invalid_mfiv_t = variant_panel[
        mfiv_t.notna() & (~np.isfinite(mfiv_t) | (mfiv_t < 0))
    ]
    add_validation_row(
        rows,
        variant_label,
        "no_nonfinite_or_negative_daily_MFIV_t",
        0,
        len(invalid_mfiv_t),
        len(invalid_mfiv_t) == 0,
        ";".join(invalid_mfiv_t["quote_date"].astype(str).tolist()[:30]),
    )

    daily_cutoffs = pd.to_numeric(
        daily_block["log_moneyness_cutoff_applied"],
        errors="coerce",
    )
    cutoff_ok = (
        daily_cutoffs.notna().all()
        and np.allclose(
            daily_cutoffs.to_numpy(dtype=float),
            cutoff,
            rtol=0.0,
            atol=1e-15,
        )
    )
    add_validation_row(
        rows,
        variant_label,
        "daily_block_cutoff_tracking_correct",
        cutoff,
        daily_cutoffs.dropna().unique().tolist(),
        cutoff_ok,
    )

    return pd.DataFrame(rows)


def build_daily_comparison(
    baseline_panel: pd.DataFrame,
    variant_panel: pd.DataFrame,
    variant_label: str,
    cutoff: float,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "quote_date": baseline_panel["quote_date"].astype(str),
        "variant": variant_label,
        "log_moneyness_cutoff": cutoff,
        "MFIV_t_baseline": pd.to_numeric(
            baseline_panel["MFIV_t"],
            errors="coerce",
        ),
        "MFIV_t_variant": pd.to_numeric(
            variant_panel["MFIV_t"],
            errors="coerce",
        ),
        "n_mfiv_ok_rows_baseline": pd.to_numeric(
            baseline_panel["n_mfiv_ok_rows"],
            errors="coerce",
        ),
        "n_mfiv_ok_rows_variant": pd.to_numeric(
            variant_panel["n_mfiv_ok_rows"],
            errors="coerce",
        ),
        "MFIV_0931_baseline": pd.to_numeric(
            baseline_panel["MFIV_0931"],
            errors="coerce",
        ),
        "MFIV_0931_variant": pd.to_numeric(
            variant_panel["MFIV_0931"],
            errors="coerce",
        ),
        "mfiv_0931_status_baseline": (
            baseline_panel["mfiv_0931_status"].astype(str)
        ),
        "mfiv_0931_status_variant": (
            variant_panel["mfiv_0931_status"].astype(str)
        ),
    })

    out["MFIV_t_signed_diff"] = (
        out["MFIV_t_variant"] - out["MFIV_t_baseline"]
    )
    out["MFIV_t_abs_diff"] = out["MFIV_t_signed_diff"].abs()
    out["MFIV_t_pct_diff"] = np.nan
    nonzero = out["MFIV_t_baseline"].abs() > 0
    out.loc[nonzero, "MFIV_t_pct_diff"] = (
        out.loc[nonzero, "MFIV_t_abs_diff"]
        / out.loc[nonzero, "MFIV_t_baseline"].abs()
        * 100.0
    )
    out["n_ok_rows_change"] = (
        out["n_mfiv_ok_rows_variant"]
        - out["n_mfiv_ok_rows_baseline"]
    )

    return out


def daily_panel_logm():
    require_file_logm(BASELINE_PANEL_FILE, "Final validated baseline daily panel")

    baseline_panel = pd.read_csv(
        BASELINE_PANEL_FILE,
        dtype=str,
        keep_default_na=False,
    )
    baseline_panel["quote_date"] = clean_date_series(
        baseline_panel["quote_date"]
    )
    check_unique_key(
        baseline_panel,
        ["quote_date"],
        "Final validated baseline daily panel",
    )
    require_columns_logm(
        baseline_panel,
        ["quote_date"] + MFIV_REPLACEMENT_COLUMNS,
        "Final validated baseline daily panel",
    )

    print("=" * 130)
    print("BUILD DAILY REGRESSION PANELS - MFIV LOG-MONEYNESS ROBUSTNESS")
    print("=" * 130)
    print(f"DATA_DIR:                  {DATA_DIR}")
    print(f"Baseline panel:            {BASELINE_PANEL_FILE}")
    print(f"Baseline rows:             {len(baseline_panel)}")
    print(f"Output folder:             {OUTPUT_DIR}")
    print(f"Daily mean window:         {SURFACE_START_TIME} to {SURFACE_END_TIME}")
    print(f"Use only MFIV flag ok:     {USE_ONLY_OK_MFIV_ROWS}")
    print(f"09:31 fallback tolerance:  +/- {SNAPSHOT_TOLERANCE_MINUTES} minutes")
    print("Non-MFIV panel columns:    copied unchanged from validated baseline")
    print("=" * 130)

    all_validations = []
    all_comparisons = []
    structural_failure = False

    for variant in VARIANTS:
        label = variant["label"]
        cutoff = float(variant["cutoff"])
        mfiv_file = (
            DATA_DIR
            / variant["mfiv_folder"]
            / variant["mfiv_file"]
        )

        print()
        print("=" * 130)
        print(f"BUILDING DAILY PANEL: {label}")
        print("=" * 130)
        print(f"MFIV intraday input: {mfiv_file}")
        print(f"Cutoff:              |ln(K/F)| <= {cutoff}")

        intraday = load_intraday_mfiv(
            mfiv_file=mfiv_file,
            variant_label=label,
            expected_cutoff=cutoff,
        )

        daily_block = build_daily_mfiv_block(
            intraday=intraday,
            variant_label=label,
            cutoff=cutoff,
        )

        variant_panel = build_variant_panel(
            baseline_panel=baseline_panel,
            daily_block=daily_block,
            variant_label=label,
        )

        validation = validate_variant_panel(
            baseline_panel=baseline_panel,
            variant_panel=variant_panel,
            daily_block=daily_block,
            variant_label=label,
            cutoff=cutoff,
        )

        comparison = build_daily_comparison(
            baseline_panel=baseline_panel,
            variant_panel=variant_panel,
            variant_label=label,
            cutoff=cutoff,
        )

        panel_output = (
            OUTPUT_DIR
            / f"{OUTPUT_FILE_PREFIX}daily_regression_panel_{YEAR}_{label}.csv"
        )
        daily_block_output = (
            OUTPUT_DIR
            / f"{OUTPUT_FILE_PREFIX}daily_mfiv_block_{YEAR}_{label}.csv"
        )
        validation_output = (
            OUTPUT_DIR
            / f"{OUTPUT_FILE_PREFIX}daily_regression_panel_{YEAR}_{label}_validation.csv"
        )
        comparison_output = (
            OUTPUT_DIR
            / f"{OUTPUT_FILE_PREFIX}daily_mfiv_comparison_vs_baseline_{YEAR}_{label}.csv"
        )

        safe_write_csv(variant_panel, panel_output)
        safe_write_csv(daily_block, daily_block_output)
        safe_write_csv(validation, validation_output)
        safe_write_csv(comparison, comparison_output)

        all_validations.append(validation)
        all_comparisons.append(comparison)

        hard_checks = validation[
            validation["expected"].astype(str) != "diagnostic"
        ]
        passed = bool((hard_checks["passed"] == 1).all())
        structural_failure = structural_failure or (not passed)

        n_zero = int(
            pd.to_numeric(
                variant_panel["n_mfiv_ok_rows"],
                errors="coerce",
            ).isna().sum()
        )
        n_missing_0931 = int(
            (
                variant_panel["mfiv_0931_status"].astype(str)
                == "missing_ok_within_2min"
            ).sum()
        )

        print(f"Intraday rows loaded:       {len(intraday):,}")
        print(f"Daily MFIV block rows:      {len(daily_block):,}")
        print(f"Daily panel rows:           {len(variant_panel):,}")
        print(f"Days with missing daily MFIV mean: {n_zero}")
        print(f"Days missing MFIV_0931:     {n_missing_0931}")
        print(f"Structural validation pass: {int(passed)}")
        print(f"Panel output:               {panel_output}")
        print(f"Daily block output:         {daily_block_output}")
        print(f"Comparison output:          {comparison_output}")
        print(f"Validation output:          {validation_output}")
        print()
        print(validation.to_string(index=False))
        print("=" * 130)

    combined_validation = pd.concat(all_validations, ignore_index=True)
    combined_comparison = pd.concat(all_comparisons, ignore_index=True)

    safe_write_csv(
        combined_validation,
        OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}daily_logm_robustness_validation_all_variants.csv",
    )
    safe_write_csv(
        combined_comparison,
        OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}daily_mfiv_comparison_vs_baseline_all_variants.csv",
    )

    print()
    print("=" * 130)
    print("ALL DAILY LOG-MONEYNESS ROBUSTNESS PANELS COMPLETED")
    print("=" * 130)
    print(combined_validation.to_string(index=False))
    print(f"Output folder: {OUTPUT_DIR}")
    print("=" * 130)

    if structural_failure:
        raise SystemExit(
            "Parte 2: almeno un controllo strutturale dei pannelli "
            "log-moneyness e' fallito (vedi le validazioni sopra)."
        )


def main():
    daily_regression_panel()
    daily_panel_logm()


if __name__ == "__main__":
    main()
