"""
Pannello intraday 2025 su griglia a 10 minuti (09:40-15:50, 38 punti) nelle 247 sessioni regolari.
  Parte 1: varianza realizzata in avanti a orizzonte fisso (5, 15, 30, 60 minuti).
  Parte 2: varianza realizzata trailing dal primo rendimento, con cross-check su SpotRV_open.
  Parte 3: pannello unito: FwdRV, TrailingRV, MFIV, gamma dalle due fonti, chiavi di effetto fisso.
  Parte 4: varianti logm_005 e logm_002, con MFIV e MFIV_ann calcolate sulla banda |ln(K/F)|.
Input:  CBOE/Data/03_mfiv/, CBOE/Data/04_gamma_herfindahl/, CBOE/Data/05_spx_minute_bars/,
        CBOE/Data/07_daily_panel/
Output: CBOE/Data/08_intraday_panel/
"""

import re

import numpy as np
import pandas as pd

from common import (
    DATA_DIR,
    EARLY_CLOSE,
    EARLY_CLOSE_DATES,
    REGULAR_CLOSE,
    YEAR,
    clean_date_series,
    output_dir,
    safe_write_csv,
    to_numeric_series,
)

OUTPUT_DIR = output_dir("08_intraday_panel")
OUTPUT_FILE_PREFIX = "08_"


# =====================================================================
# FUNZIONI E COSTANTI COMUNI ALLE PARTI
# =====================================================================
def time_string_to_seconds(time_string: str) -> int:
    h, m, s = str(time_string).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def seconds_to_time_string(seconds_value: int) -> str:
    h = int(seconds_value // 3600)
    m = int((seconds_value % 3600) // 60)
    s = int(seconds_value % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_time_string(value) -> str:
    text = str(value).strip()

    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        h, m = text.split(":")
        return f"{int(h):02d}:{int(m):02d}:00"

    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", text):
        h, m, s = text.split(":")
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    parsed = pd.to_datetime(text, errors="coerce")

    if pd.isna(parsed):
        return ""

    return parsed.strftime("%H:%M:%S")


def parse_time_to_seconds(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).apply(normalize_time_string)
    t = pd.to_datetime(normalized, format="%H:%M:%S", errors="coerce")
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second


def normalize_colname(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def detect_column(columns, candidates):
    normalized_map = {normalize_colname(col): col for col in columns}

    for candidate in candidates:
        key = normalize_colname(candidate)

        if key in normalized_map:
            return normalized_map[key]

    return None


def require_file(path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n{path}")


def require_columns(df: pd.DataFrame, columns, label: str):
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


# Barre SPX al minuto delle parti 1 e 2: CSV di 05_spx_minute_bars, etichettati a fine intervallo.
SPX_FOLDER_NAME = "05_spx_minute_bars"

SPX_FILE_GLOBS = [
    "05_SPX_*_RV.csv",
]

SPX_DIR = DATA_DIR / SPX_FOLDER_NAME

SESSION_START_TIME = "09:30:00"
SESSION_END_TIME = REGULAR_CLOSE  # "16:00:00"

SESSION_START_SECONDS = time_string_to_seconds(SESSION_START_TIME)
SESSION_END_SECONDS = time_string_to_seconds(SESSION_END_TIME)

# Giornate dell'ambito SAMPLE_5DAY (parti 2 e 3).
SAMPLE_DATES = [
    "2025-06-16",
    "2025-04-07",
    "2025-03-21",
    "2025-01-31",
    "2025-09-19",
]

# ---------------------------------------------------------------------
# GRIGLIA, IDENTICA NELLE PARTI 1, 2 E 3
# ---------------------------------------------------------------------
# Ancorata alla chiusura (09:40, ..., 15:50): a passo di 10 minuti si tiene la coda della
# sessione, dove la concentrazione del gamma e' massima, invece del minuto 09:31. Deve essere
# la stessa nelle tre parti, perche' TrailingRV e FwdRV vanno appaiati punto per punto.
GRID_START_TIME = "09:40:00"
GRID_END_TIME = "15:50:00"
GRID_SPACING_MINUTES = 10

GRID_START_SECONDS = time_string_to_seconds(GRID_START_TIME)
GRID_END_SECONDS = time_string_to_seconds(GRID_END_TIME)
GRID_SPACING_SECONDS = GRID_SPACING_MINUTES * 60

EXPECTED_GRID_ROWS_PER_DAY = len(
    range(GRID_START_SECONDS, GRID_END_SECONDS + 1, GRID_SPACING_SECONDS)
)


def add_one_minute_returns(spx: pd.DataFrame) -> pd.DataFrame:
    out = spx.copy()
    out = out.sort_values(["quote_date", "_time_seconds"]).reset_index(drop=True)

    out["log_close"] = np.log(out["spx_close"])
    out["prev_log_close"] = out.groupby("quote_date")["log_close"].shift(1)
    out["prev_spx_close"] = out.groupby("quote_date")["spx_close"].shift(1)
    out["log_return"] = out["log_close"] - out["prev_log_close"]
    out["squared_log_return"] = out["log_return"] ** 2

    return out


# =====================================================================
# PARTE 1: VARIANZA REALIZZATA IN AVANTI A ORIZZONTE FISSO
# =====================================================================
# Ambito: PRIMARY_247 (specifica principale, senza chiusure anticipate) oppure ROBUSTNESS_250
# (con le chiusure anticipate, su griglia piu' corta). L'ambito entra nel nome dei file.
RUN_SCOPE_FWD = "PRIMARY_247"
# RUN_SCOPE_FWD = "ROBUSTNESS_250"

OVERWRITE_EXISTING = True

EXPECTED_TRADING_DAYS_BEFORE_EXCLUSION = 250
EXPECTED_TRADING_DAYS_PRIMARY_247 = 247
EXPECTED_TRADING_DAYS_ROBUSTNESS_250 = 250

FORWARD_HORIZONS_MINUTES = [5, 15, 30, 60]

if RUN_SCOPE_FWD == "PRIMARY_247":
    INCLUDE_EARLY_CLOSES_FWD = False
elif RUN_SCOPE_FWD == "ROBUSTNESS_250":
    INCLUDE_EARLY_CLOSES_FWD = True
else:
    raise ValueError("RUN_SCOPE must be PRIMARY_247 or ROBUSTNESS_250.")

FWD_STEM = f"{OUTPUT_FILE_PREFIX}fixed_horizon_fwdrv_10min_2025_{RUN_SCOPE_FWD.lower()}"

FWD_PANEL_FILE = OUTPUT_DIR / f"{FWD_STEM}.csv"
FWD_DIAGNOSTICS_FILE = OUTPUT_DIR / f"{FWD_STEM}_diagnostics.csv"
FWD_VALIDATION_FILE = OUTPUT_DIR / f"{FWD_STEM}_validation.csv"
FWD_ANOMALY_FILE = OUTPUT_DIR / f"{FWD_STEM}_anomaly_scan.csv"
FWD_WINDOW_INFO_FILE = OUTPUT_DIR / f"{FWD_STEM}_window_info.csv"
FWD_EXPECTED_PROFILE_FILE = OUTPUT_DIR / f"{FWD_STEM}_expected_day_profile.csv"


def find_spx_files():
    if not SPX_DIR.exists():
        raise FileNotFoundError(
            f"Cartella SPX non trovata:\n{SPX_DIR}\n\n"
            "Attesa CBOE/Data/05_spx_minute_bars, prodotta da 05_SPX_minute_bars.py."
        )

    files = []

    for pattern in SPX_FILE_GLOBS:
        files.extend(sorted(SPX_DIR.glob(pattern)))

    files = sorted(list(dict.fromkeys(files)))
    files = [f for f in files if not f.name.startswith("~$")]

    if not files:
        raise FileNotFoundError(
            f"Nessun CSV SPX trovato in:\n{SPX_DIR}\nPattern: {SPX_FILE_GLOBS}\n"
            "Lancia prima 05_SPX_minute_bars.py."
        )

    return files


# ---------------------------------------------------------------------
# DATI SPX AL MINUTO
# ---------------------------------------------------------------------
def load_spx_full_year() -> pd.DataFrame:
    files = find_spx_files()

    frames = []

    print(f"    SPX CSV files found: {len(files)}")

    for file in files:
        print(f"    Reading SPX file: {file.name}")

        raw = pd.read_csv(file, dtype=str, keep_default_na=False)

        date_col = detect_column(raw.columns, ["Date", "date", "quote_date"])
        time_col = detect_column(raw.columns, ["time", "Time", "quote_time"])
        close_col = detect_column(raw.columns, ["close", "Close"])

        for col, label in ((date_col, "Date"), (time_col, "time"), (close_col, "close")):
            if col is None:
                raise ValueError(
                    f"Non trovo la colonna {label} nel file:\n{file}\n"
                    f"Colonne disponibili: {list(raw.columns)}"
                )

        df = pd.DataFrame()
        df["quote_date"] = clean_date_series(raw[date_col])
        df["quote_time"] = raw[time_col].astype(str).apply(normalize_time_string)
        df["spx_close"] = pd.to_numeric(raw[close_col], errors="coerce")
        df["_source_file"] = file.name

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])

    out = out[
        out["quote_date"].notna()
        & out["quote_time"].notna()
        & out["spx_close"].notna()
        & np.isfinite(out["spx_close"])
        & (out["spx_close"] > 0)
        & out["_time_seconds"].notna()
        & out["quote_date"].astype(str).str.startswith(str(YEAR))
        & (out["_time_seconds"] > SESSION_START_SECONDS)
        & (out["_time_seconds"] <= SESSION_END_SECONDS)
    ].copy()

    if not INCLUDE_EARLY_CLOSES_FWD:
        out = out[~out["quote_date"].isin(EARLY_CLOSE_DATES)].copy()

    duplicate_count = int(out.duplicated(["quote_date", "quote_time"]).sum())

    out = (
        out.sort_values(["quote_date", "_time_seconds"])
        .drop_duplicates(subset=["quote_date", "quote_time"], keep="last")
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError("Dopo la lettura dei CSV SPX non resta nessuna riga valida.")

    print(f"    SPX rows loaded: {len(out):,}")
    print(f"    SPX days loaded: {out['quote_date'].nunique():,}")
    print(f"    SPX first date:  {out['quote_date'].min()}")
    print(f"    SPX last date:   {out['quote_date'].max()}")
    print(f"    Duplicate date-time rows before deduplication: {duplicate_count:,}")

    return out


# ---------------------------------------------------------------------
# GRIGLIA PER GIORNATA
# ---------------------------------------------------------------------
def theoretical_grid_times_for_day(actual_close_seconds: int):
    day_grid_end_seconds = min(
        GRID_END_SECONDS,
        actual_close_seconds - GRID_SPACING_SECONDS
    )

    if day_grid_end_seconds < GRID_START_SECONDS:
        return []

    return list(range(
        GRID_START_SECONDS,
        day_grid_end_seconds + 1,
        GRID_SPACING_SECONDS
    ))


def observed_grid_times_for_day(day: pd.DataFrame, actual_close_seconds: int):
    observed_times = set(day["_time_seconds"].dropna().astype(int).tolist())
    theoretical_times = theoretical_grid_times_for_day(actual_close_seconds)

    return [t for t in theoretical_times if t in observed_times]


def expected_valid_count_for_day(grid_times, actual_close_seconds: int, horizon_minutes: int) -> int:
    horizon_seconds = horizon_minutes * 60
    return sum(1 for t in grid_times if t + horizon_seconds <= actual_close_seconds)


def build_expected_day_profile(spx_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for quote_date, day in spx_returns.groupby("quote_date", sort=True):
        day = day.sort_values("_time_seconds").copy()

        actual_close_seconds = int(day["_time_seconds"].max())
        theoretical_grid_times = theoretical_grid_times_for_day(actual_close_seconds)

        row = {
            "quote_date": quote_date,
            "run_scope": RUN_SCOPE_FWD,
            "is_early_close": int(
                quote_date in EARLY_CLOSE_DATES or actual_close_seconds < SESSION_END_SECONDS
            ),
            "actual_session_close_time": seconds_to_time_string(actual_close_seconds),
            "expected_grid_rows": len(theoretical_grid_times),
        }

        for horizon in FORWARD_HORIZONS_MINUTES:
            row[f"expected_h{horizon}_valid"] = expected_valid_count_for_day(
                grid_times=theoretical_grid_times,
                actual_close_seconds=actual_close_seconds,
                horizon_minutes=horizon,
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# COSTRUZIONE DI FwdRV
# ---------------------------------------------------------------------
def compute_forward_sum_direct(day, grid_seconds, horizon_minutes, actual_close_seconds):
    horizon_seconds = horizon_minutes * 60
    target_seconds = grid_seconds + horizon_seconds

    if target_seconds > actual_close_seconds:
        return np.nan, 0, 0, "", "", np.nan

    window = day[
        (day["_time_seconds"] > grid_seconds)
        & (day["_time_seconds"] <= target_seconds)
        & day["squared_log_return"].notna()
        & np.isfinite(day["squared_log_return"])
    ].copy()

    n_returns = len(window)

    if n_returns != horizon_minutes:
        return np.nan, 0, n_returns, "", "", np.nan

    value = float(window["squared_log_return"].sum())

    first_return_time = str(window["quote_time"].iloc[0])
    last_return_time = str(window["quote_time"].iloc[-1])
    includes_grid_return = int((window["_time_seconds"] == grid_seconds).any())

    return value, 1, n_returns, first_return_time, last_return_time, includes_grid_return


def compute_fixed_horizon_for_day(day: pd.DataFrame):
    day = day.sort_values("_time_seconds").reset_index(drop=True).copy()

    quote_date = str(day["quote_date"].iloc[0])
    actual_close_seconds = int(day["_time_seconds"].max())
    grid_times = observed_grid_times_for_day(day, actual_close_seconds)

    output_rows = []
    window_rows = []

    for grid_seconds in grid_times:
        grid_time = seconds_to_time_string(grid_seconds)

        row = {"quote_date": quote_date, "grid_time": grid_time}

        for horizon in FORWARD_HORIZONS_MINUTES:
            (value, valid, n_returns, first_return_time,
             last_return_time, includes_grid_return) = compute_forward_sum_direct(
                day=day,
                grid_seconds=grid_seconds,
                horizon_minutes=horizon,
                actual_close_seconds=actual_close_seconds,
            )

            row[f"FwdRV_{horizon}"] = value
            row[f"h{horizon}_valid"] = valid

            window_rows.append({
                "quote_date": quote_date,
                "grid_time": grid_time,
                "horizon_minutes": horizon,
                "target_time": seconds_to_time_string(grid_seconds + horizon * 60),
                "actual_session_close_time": seconds_to_time_string(actual_close_seconds),
                "valid": valid,
                "n_returns_used": n_returns,
                "first_return_time": first_return_time,
                "last_return_time": last_return_time,
                "includes_grid_return": includes_grid_return,
                "expected_first_return_time": seconds_to_time_string(grid_seconds + 60),
                "expected_last_return_time": seconds_to_time_string(grid_seconds + horizon * 60),
            })

        output_rows.append(row)

    return pd.DataFrame(output_rows), pd.DataFrame(window_rows)


def build_fixed_horizon_full_year(spx_returns: pd.DataFrame):
    all_outputs = []
    all_window_info = []

    dates = sorted(spx_returns["quote_date"].dropna().unique().tolist())

    for index, quote_date in enumerate(dates, start=1):
        day = spx_returns[spx_returns["quote_date"] == quote_date].copy()

        out_day, window_info_day = compute_fixed_horizon_for_day(day)

        all_outputs.append(out_day)
        all_window_info.append(window_info_day)

        if index % 25 == 0 or index == len(dates):
            print(f"    processed {index}/{len(dates)} days", flush=True)

    output = pd.concat(all_outputs, ignore_index=True)
    window_info = pd.concat(all_window_info, ignore_index=True)

    output["_date_dt"] = pd.to_datetime(output["quote_date"], errors="coerce")
    output["_grid_seconds"] = parse_time_to_seconds(output["grid_time"])
    output = (
        output.sort_values(["_date_dt", "_grid_seconds"])
        .drop(columns=["_date_dt", "_grid_seconds"]).reset_index(drop=True)
    )

    window_info["_date_dt"] = pd.to_datetime(window_info["quote_date"], errors="coerce")
    window_info["_grid_seconds"] = parse_time_to_seconds(window_info["grid_time"])
    window_info = (
        window_info.sort_values(["_date_dt", "_grid_seconds", "horizon_minutes"])
        .drop(columns=["_date_dt", "_grid_seconds"]).reset_index(drop=True)
    )

    return output, window_info


# ---------------------------------------------------------------------
# DIAGNOSTICA
# ---------------------------------------------------------------------
def build_diagnostics(panel: pd.DataFrame, expected_profile: pd.DataFrame) -> pd.DataFrame:
    rows = []

    expected_map = expected_profile.set_index("quote_date")

    for quote_date, day in panel.groupby("quote_date", sort=True):
        grid_count = len(day)

        expected_grid_rows = int(expected_map.loc[quote_date, "expected_grid_rows"])
        is_early_close = int(expected_map.loc[quote_date, "is_early_close"])
        actual_session_close_time = str(expected_map.loc[quote_date, "actual_session_close_time"])

        for horizon in FORWARD_HORIZONS_MINUTES:
            fwd_col = f"FwdRV_{horizon}"
            valid_col = f"h{horizon}_valid"

            expected_valid = int(expected_map.loc[quote_date, f"expected_h{horizon}_valid"])

            valid = day[day[valid_col] == 1].copy()
            excluded = day[day[valid_col] == 0].copy()

            rows.append({
                "quote_date": quote_date,
                "run_scope": RUN_SCOPE_FWD,
                "is_early_close": is_early_close,
                "actual_session_close_time": actual_session_close_time,
                "horizon_minutes": horizon,
                "grid_timestamp_count": grid_count,
                "expected_grid_timestamp_count": expected_grid_rows,
                "valid_observations": int(len(valid)),
                "expected_valid_observations": expected_valid,
                "excluded_observations": int(len(excluded)),
                "expected_excluded_observations": expected_grid_rows - expected_valid,
                "min_FwdRV_valid": valid[fwd_col].min() if not valid.empty else np.nan,
                "max_FwdRV_valid": valid[fwd_col].max() if not valid.empty else np.nan,
                "mean_FwdRV_valid": valid[fwd_col].mean() if not valid.empty else np.nan,
            })

    return pd.DataFrame(rows)


def build_anomaly_scan(panel, diagnostics, expected_profile) -> pd.DataFrame:
    rows = []

    day_counts = (
        panel.groupby("quote_date", as_index=False).agg(grid_rows=("grid_time", "count"))
    )

    day_counts = day_counts.merge(
        expected_profile[[
            "quote_date", "is_early_close", "actual_session_close_time", "expected_grid_rows",
        ]],
        on="quote_date", how="right", validate="one_to_one",
    )

    day_counts["grid_rows"] = pd.to_numeric(
        day_counts["grid_rows"], errors="coerce"
    ).fillna(0).astype(int)

    bad_grid = day_counts[day_counts["grid_rows"] != day_counts["expected_grid_rows"]].copy()

    for _, row in bad_grid.iterrows():
        rows.append({
            "anomaly_type": "unexpected_grid_row_count",
            "quote_date": row["quote_date"],
            "is_early_close": int(row["is_early_close"]),
            "actual_session_close_time": row["actual_session_close_time"],
            "horizon_minutes": "",
            "expected": int(row["expected_grid_rows"]),
            "actual": int(row["grid_rows"]),
            "details": "Actual grid row count differs from day-specific expected grid.",
        })

    for _, row in diagnostics.iterrows():
        if int(row["valid_observations"]) != int(row["expected_valid_observations"]):
            rows.append({
                "anomaly_type": "unexpected_valid_count",
                "quote_date": row["quote_date"],
                "is_early_close": int(row["is_early_close"]),
                "actual_session_close_time": row["actual_session_close_time"],
                "horizon_minutes": int(row["horizon_minutes"]),
                "expected": int(row["expected_valid_observations"]),
                "actual": int(row["valid_observations"]),
                "details": f"h{int(row['horizon_minutes'])}_valid count differs from expected.",
            })

    if not rows:
        rows.append({
            "anomaly_type": "no_anomalies_detected",
            "quote_date": "", "is_early_close": "", "actual_session_close_time": "",
            "horizon_minutes": "", "expected": "", "actual": "",
            "details": "All days have expected grid rows and horizon-valid counts.",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# VALIDAZIONE
# ---------------------------------------------------------------------
def validate_row_counts(panel, diagnostics, expected_profile) -> pd.DataFrame:
    rows = []

    n_days = int(panel["quote_date"].nunique())
    n_rows = int(len(panel))

    expected_days = (
        EXPECTED_TRADING_DAYS_PRIMARY_247 if RUN_SCOPE_FWD == "PRIMARY_247"
        else EXPECTED_TRADING_DAYS_ROBUSTNESS_250
    )

    expected_total_grid_rows = int(expected_profile["expected_grid_rows"].sum())

    rows.append({
        "validation_group": "row_counts",
        "check": "trading_days_in_run_scope",
        "expected": expected_days, "actual": n_days,
        "passed": int(n_days == expected_days),
        "details": f"RUN_SCOPE={RUN_SCOPE_FWD}",
    })

    rows.append({
        "validation_group": "row_counts",
        "check": "total_grid_rows",
        "expected": expected_total_grid_rows, "actual": n_rows,
        "passed": int(n_rows == expected_total_grid_rows),
        "details": "Expected rows are computed from the day-specific grid.",
    })

    for _, row in expected_profile.iterrows():
        quote_date = row["quote_date"]
        expected_grid_rows = int(row["expected_grid_rows"])

        day = panel[panel["quote_date"] == quote_date].copy()
        actual_grid_rows = int(len(day))

        rows.append({
            "validation_group": "row_counts",
            "check": f"{quote_date}_grid_rows",
            "expected": expected_grid_rows, "actual": actual_grid_rows,
            "passed": int(actual_grid_rows == expected_grid_rows),
            "details": "Day-specific grid row count.",
        })

        valid_counts = []

        for horizon in FORWARD_HORIZONS_MINUTES:
            valid_col = f"h{horizon}_valid"
            expected_valid = int(row[f"expected_h{horizon}_valid"])
            valid_count = int(day[valid_col].sum()) if valid_col in day.columns else 0
            valid_counts.append(valid_count)

            rows.append({
                "validation_group": "row_counts",
                "check": f"{quote_date}_h{horizon}_valid_count",
                "expected": expected_valid, "actual": valid_count,
                "passed": int(valid_count == expected_valid),
                "details": f"Boundary rule: t + {horizon} minutes must be <= actual session close.",
            })

        decreasing = all(
            valid_counts[i] >= valid_counts[i + 1] for i in range(len(valid_counts) - 1)
        )

        rows.append({
            "validation_group": "row_counts",
            "check": f"{quote_date}_valid_counts_weakly_decrease_with_horizon",
            "expected": "h5 >= h15 >= h30 >= h60", "actual": str(valid_counts),
            "passed": int(decreasing),
            "details": "Longer horizons should lose more observations near the close.",
        })

    return pd.DataFrame(rows)


def validate_no_lookahead(window_info: pd.DataFrame) -> pd.DataFrame:
    rows = []

    valid_windows = window_info[window_info["valid"] == 1].copy()

    includes_grid_return = valid_windows[
        pd.to_numeric(valid_windows["includes_grid_return"], errors="coerce") == 1
    ]
    wrong_first = valid_windows[
        valid_windows["first_return_time"].astype(str)
        != valid_windows["expected_first_return_time"].astype(str)
    ]
    wrong_last = valid_windows[
        valid_windows["last_return_time"].astype(str)
        != valid_windows["expected_last_return_time"].astype(str)
    ]
    wrong_n = valid_windows[
        pd.to_numeric(valid_windows["n_returns_used"], errors="coerce")
        != pd.to_numeric(valid_windows["horizon_minutes"], errors="coerce")
    ]

    for check, bad, detail in (
        ("no_valid_window_includes_r_t", includes_grid_return,
         "Each window must start strictly after grid timestamp t."),
        ("first_return_time_equals_t_plus_1_minute", wrong_first,
         "Valid windows must start at r_{t+1}."),
        ("last_return_time_equals_t_plus_h", wrong_last,
         "Valid windows must end at r_{t+h}."),
        ("number_of_returns_equals_horizon_minutes", wrong_n,
         "A 60-minute horizon must use exactly 60 one-minute returns, etc."),
    ):
        rows.append({
            "validation_group": "no_lookahead", "check": check,
            "expected": 0, "actual": len(bad),
            "passed": int(len(bad) == 0), "details": detail,
        })

    return pd.DataFrame(rows)


def validate_scope(panel: pd.DataFrame) -> pd.DataFrame:
    present_early_closes = sorted(
        set(panel["quote_date"].astype(str).unique()) & set(EARLY_CLOSE_DATES)
    )

    if RUN_SCOPE_FWD == "PRIMARY_247":
        expected_early_count = 0
        expected_details = "Early-close sessions excluded in primary 247-day specification."
    else:
        expected_early_count = len(EARLY_CLOSE_DATES)
        expected_details = "Early-close sessions included in robustness 250-day specification."

    return pd.DataFrame([{
        "validation_group": "scope",
        "check": "early_close_dates_scope",
        "expected": expected_early_count,
        "actual": len(present_early_closes),
        "passed": int(len(present_early_closes) == expected_early_count),
        "details": expected_details + " Present: " + ";".join(present_early_closes),
    }])


def build_complete_validation(panel, diagnostics, window_info, expected_profile) -> pd.DataFrame:
    return pd.concat([
        validate_scope(panel),
        validate_row_counts(panel, diagnostics, expected_profile),
        validate_no_lookahead(window_info),
    ], ignore_index=True)


def forward_rv_10min():
    print("=" * 140)
    print("FIXED-HORIZON FORWARD REALIZED VARIANCE - 10-MINUTE GRID - FULL RUN")
    print("=" * 140)
    print(f"RUN_SCOPE:                {RUN_SCOPE_FWD}")
    print(f"DATA_DIR:                 {DATA_DIR}")
    print(f"SPX folder:               {SPX_DIR}  (CSV di 05_spx_minute_bars)")
    print(f"Output folder:            {OUTPUT_DIR}")
    print(f"Include early closes:     {INCLUDE_EARLY_CLOSES_FWD}")
    print(f"Grid:                     {GRID_START_TIME} to {GRID_END_TIME}, "
          f"every {GRID_SPACING_MINUTES} minutes")
    print(f"Horizons:                 {FORWARD_HORIZONS_MINUTES}")
    print("=" * 140)

    print("Loading SPX one-minute data...")
    spx_raw = load_spx_full_year()

    print("Adding one-minute close-to-close log returns...")
    spx_returns = add_one_minute_returns(spx_raw)

    print("Building expected day-specific grid profile...")
    expected_profile = build_expected_day_profile(spx_returns)

    print("Building fixed-horizon forward RV panel...")
    panel, window_info = build_fixed_horizon_full_year(spx_returns)

    final_columns = [
        "quote_date", "grid_time",
        "FwdRV_5", "FwdRV_15", "FwdRV_30", "FwdRV_60",
        "h5_valid", "h15_valid", "h30_valid", "h60_valid",
    ]

    for col in final_columns:
        if col not in panel.columns:
            panel[col] = np.nan

    panel = panel[final_columns].copy()

    diagnostics = build_diagnostics(panel, expected_profile)
    anomaly_scan = build_anomaly_scan(panel, diagnostics, expected_profile)
    validation = build_complete_validation(panel, diagnostics, window_info, expected_profile)

    for path in [
        FWD_PANEL_FILE, FWD_DIAGNOSTICS_FILE, FWD_VALIDATION_FILE,
        FWD_ANOMALY_FILE, FWD_WINDOW_INFO_FILE, FWD_EXPECTED_PROFILE_FILE,
    ]:
        if path.exists() and OVERWRITE_EXISTING:
            path.unlink()

    panel.to_csv(FWD_PANEL_FILE, index=False)
    diagnostics.to_csv(FWD_DIAGNOSTICS_FILE, index=False)
    validation.to_csv(FWD_VALIDATION_FILE, index=False)
    anomaly_scan.to_csv(FWD_ANOMALY_FILE, index=False)
    window_info.to_csv(FWD_WINDOW_INFO_FILE, index=False)
    expected_profile.to_csv(FWD_EXPECTED_PROFILE_FILE, index=False)

    all_validation_passed = int((validation["passed"] == 1).all())
    n_failed = int((validation["passed"] == 0).sum())

    print()
    print("=" * 140)
    print("COMPLETED FIXED-HORIZON FwdRV 10-MINUTE GRID - FULL RUN")
    print("=" * 140)
    print(f"RUN_SCOPE:                 {RUN_SCOPE_FWD}")
    print(f"Rows written:              {len(panel):,}")
    print(f"Days written:              {panel['quote_date'].nunique():,}")
    print(f"Expected total rows:       {int(expected_profile['expected_grid_rows'].sum()):,}")
    print(f"All validation passed:     {all_validation_passed}  "
          f"(controlli falliti: {n_failed} su {len(validation)})")
    print()
    print("Osservazioni valide per orizzonte, osservate contro attese:")
    for horizon in FORWARD_HORIZONS_MINUTES:
        got = int(panel[f"h{horizon}_valid"].sum())
        exp = int(expected_profile[f"expected_h{horizon}_valid"].sum())
        print(f"    h={horizon:2d}: {got:6,} osservate | {exp:6,} attese | "
              f"{'MATCH' if got == exp else 'DIVERSO'}")

    print()
    print("=" * 140)
    print("EXPECTED DAY PROFILE SUMMARY")
    print("=" * 140)
    print(
        expected_profile.groupby(
            ["run_scope", "is_early_close", "actual_session_close_time"], as_index=False
        ).agg(
            days=("quote_date", "count"),
            total_expected_grid_rows=("expected_grid_rows", "sum"),
            total_expected_h5_valid=("expected_h5_valid", "sum"),
            total_expected_h15_valid=("expected_h15_valid", "sum"),
            total_expected_h30_valid=("expected_h30_valid", "sum"),
            total_expected_h60_valid=("expected_h60_valid", "sum"),
        ).to_string(index=False)
    )

    print()
    print("=" * 140)
    print("ANOMALY SCAN")
    print("=" * 140)
    print(anomaly_scan.to_string(index=False))

    print()
    print("=" * 140)
    print("VALIDATION, SOLO CONTROLLI FALLITI")
    print("=" * 140)
    failed = validation[validation["passed"] == 0]
    print(failed.to_string(index=False) if not failed.empty
          else "Nessun controllo fallito su %d." % len(validation))

    print()
    print("VALIDATION, controlli aggregati:")
    print(validation[validation["validation_group"].isin(["scope", "no_lookahead"])].to_string(index=False))
    print(validation[validation["check"].isin(["trading_days_in_run_scope", "total_grid_rows"])].to_string(index=False))
    print("=" * 140)


# =====================================================================
# PARTE 2: VARIANZA REALIZZATA TRAILING SULLA GRIGLIA
# =====================================================================
REGRESSION_PANEL_FOLDER_NAME = "07_daily_panel"
REGRESSION_PANEL_FILE_NAME = "07_daily_regression_panel_2025.csv"

# ---------------------------------------------------------------------
# AMBITO
# ---------------------------------------------------------------------
# FULL_247, ROBUSTNESS_250 (con le chiusure anticipate, griglia per giornata come nella parte 1)
# o SAMPLE_5DAY. Sulle chiusure anticipate i CSV SPX finiscono alla chiusura ufficiale delle 13:00.
RUN_SCOPE_TRAILING = "FULL_247"
# RUN_SCOPE_TRAILING = "ROBUSTNESS_250"
# RUN_SCOPE_TRAILING = "SAMPLE_5DAY"

EXPECTED_FULL_TRADING_DAYS_AFTER_EARLY_CLOSE_EXCLUSION = 247
EXPECTED_ROBUSTNESS_TRADING_DAYS = 250

# Chiusura anticipata in secondi (EARLY_CLOSE di common.py, 13:00:00): serve
# solo all'ambito ROBUSTNESS_250, per sapere fin dove arriva la griglia corta.
EARLY_CLOSE_TIME_SECONDS = time_string_to_seconds(EARLY_CLOSE)

# Primo rendimento della sessione: la barra delle 09:31 non ha una precedente, quindi TrailingRV
# parte dal rendimento delle 09:32 e al primo punto di griglia (09:40) somma nove rendimenti.
FIRST_RETURN_TIME = "09:32:00"

SPOTRV_OPEN_CROSSCHECK_TIME = "10:00:00"

# Alle 10:00 i rendimenti da 09:32 sono 29, come quelli di SpotRV_open nel pannello giornaliero.
SPOTRV_OPEN_EXPECTED_RETURNS = 29

FLOAT_TOL = 1e-18
SPOTRV_CROSSCHECK_TOL = 1e-10

# L'ambito entra nel nome dei file, quindi le modalita' convivono nella stessa cartella.
if RUN_SCOPE_TRAILING == "SAMPLE_5DAY":
    TRAILING_STEM = "08_trailing_rv_10min_2025_sample_5day"
    INCLUDE_EARLY_CLOSES_TRAILING = False
elif RUN_SCOPE_TRAILING == "FULL_247":
    TRAILING_STEM = "08_trailing_rv_10min_2025_full_247day"
    INCLUDE_EARLY_CLOSES_TRAILING = False
elif RUN_SCOPE_TRAILING == "ROBUSTNESS_250":
    TRAILING_STEM = "08_trailing_rv_10min_2025_robustness_250day"
    INCLUDE_EARLY_CLOSES_TRAILING = True
else:
    raise ValueError("RUN_SCOPE must be SAMPLE_5DAY, FULL_247 or ROBUSTNESS_250.")

REGRESSION_PANEL_FILE = DATA_DIR / REGRESSION_PANEL_FOLDER_NAME / REGRESSION_PANEL_FILE_NAME

TRAILING_PANEL_FILE = OUTPUT_DIR / f"{TRAILING_STEM}.csv"
TRAILING_DIAGNOSTICS_FILE = OUTPUT_DIR / f"{TRAILING_STEM}_diagnostics.csv"
TRAILING_VALIDATION_FILE = OUTPUT_DIR / f"{TRAILING_STEM}_validation.csv"
TRAILING_SPOTRV_CROSSCHECK_FILE = OUTPUT_DIR / f"{TRAILING_STEM}_spotrv_open_crosscheck.csv"
TRAILING_RETURN_WINDOW_FILE = OUTPUT_DIR / f"{TRAILING_STEM}_return_window_info.csv"

FIRST_RETURN_SECONDS = time_string_to_seconds(FIRST_RETURN_TIME)

SPOTRV_OPEN_CROSSCHECK_SECONDS = time_string_to_seconds(SPOTRV_OPEN_CROSSCHECK_TIME)


def grid_times_for_day(quote_date: str, observed_max_seconds: int):
    """
    Punti di griglia della giornata. L'ultimo ammesso e' la chiusura meno la spaziatura, perche'
    un punto piu' tardi non avrebbe una finestra in avanti dentro la sessione. La chiusura e'
    l'ultimo timestamp osservato, non una costante.
    """
    last_allowed = min(GRID_END_SECONDS, observed_max_seconds - GRID_SPACING_SECONDS)

    if last_allowed < GRID_START_SECONDS:
        return []

    return list(range(GRID_START_SECONDS, last_allowed + 1, GRID_SPACING_SECONDS))


def find_spx_files_trailing():
    if not SPX_DIR.exists():
        raise FileNotFoundError(
            f"SPX folder not found:\n{SPX_DIR}\n\n"
            "Attesa CBOE/Data/05_spx_minute_bars, prodotta da 05_SPX_minute_bars.py."
        )

    files = []

    for pattern in SPX_FILE_GLOBS:
        files.extend(sorted(SPX_DIR.glob(pattern)))

    files = sorted(list(dict.fromkeys(files)))
    files = [f for f in files if not f.name.startswith("~$")]

    if not files:
        raise FileNotFoundError(
            f"No SPX CSV files found in:\n{SPX_DIR}\nPatterns: {SPX_FILE_GLOBS}"
        )

    return files


def expected_grid_times():
    return list(range(
        GRID_START_SECONDS, GRID_END_SECONDS + 1, GRID_SPACING_SECONDS
    ))


def expected_n_returns_for_grid_time(grid_seconds: int) -> int:
    """
    Numero di rendimenti da FIRST_RETURN_SECONDS (09:32) fino a grid_seconds
    incluso. Al primo punto di griglia (09:40) vale 9.
    """
    if grid_seconds < FIRST_RETURN_SECONDS:
        return 0

    return int((grid_seconds - FIRST_RETURN_SECONDS) / 60) + 1


# ---------------------------------------------------------------------
# DATI SPX AL MINUTO
# ---------------------------------------------------------------------
def load_spx_full_year_trailing() -> pd.DataFrame:
    files = find_spx_files_trailing()

    frames = []

    print(f"    SPX CSV files found: {len(files)}")

    for file in files:
        print(f"    Reading SPX file: {file.name}")

        raw = pd.read_csv(file, dtype=str, keep_default_na=False)

        date_col = detect_column(raw.columns, ["Date", "date", "quote_date"])
        time_col = detect_column(raw.columns, ["time", "Time", "quote_time"])
        close_col = detect_column(raw.columns, ["close", "Close"])

        for col, label in ((date_col, "Date"), (time_col, "time"), (close_col, "close")):
            if col is None:
                raise ValueError(
                    f"Cannot find {label} column in file:\n{file}\n"
                    f"Available columns: {list(raw.columns)}"
                )

        df = pd.DataFrame()
        df["quote_date"] = clean_date_series(raw[date_col])
        df["quote_time"] = raw[time_col].astype(str).apply(normalize_time_string)
        df["spx_close"] = pd.to_numeric(raw[close_col], errors="coerce")
        df["_source_file"] = file.name

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["_time_seconds"] = parse_time_to_seconds(out["quote_time"])

    # confronto stretto sull'inizio sessione, come nel pannello giornaliero e nella parte 1
    out = out[
        out["quote_date"].notna()
        & out["quote_time"].notna()
        & out["spx_close"].notna()
        & np.isfinite(out["spx_close"])
        & (out["spx_close"] > 0)
        & out["_time_seconds"].notna()
        & out["quote_date"].astype(str).str.startswith(str(YEAR))
        & (out["_time_seconds"] > SESSION_START_SECONDS)
        & (out["_time_seconds"] <= SESSION_END_SECONDS)
    ].copy()

    if not INCLUDE_EARLY_CLOSES_TRAILING:
        out = out[~out["quote_date"].isin(EARLY_CLOSE_DATES)].copy()

    duplicate_count = int(out.duplicated(["quote_date", "quote_time"]).sum())

    out = (
        out.sort_values(["quote_date", "_time_seconds", "_source_file"])
        .drop_duplicates(subset=["quote_date", "quote_time"], keep="last")
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError("After reading SPX files, no valid rows remain.")

    label = "incluse le early close" if INCLUDE_EARLY_CLOSES_TRAILING else "escluse le early close"
    print(f"    SPX rows loaded ({label}): {len(out):,}")
    print(f"    SPX days loaded ({label}): {out['quote_date'].nunique():,}")
    print(f"    SPX first date: {out['quote_date'].min()}")
    print(f"    SPX last date:  {out['quote_date'].max()}")
    print(f"    Duplicate date-time rows before deduplication: {duplicate_count:,}")

    return out


def restrict_run_scope(spx: pd.DataFrame) -> pd.DataFrame:
    if RUN_SCOPE_TRAILING == "SAMPLE_5DAY":
        wanted = sorted(SAMPLE_DATES)
        out = spx[spx["quote_date"].isin(wanted)].copy()

        found = sorted(out["quote_date"].dropna().unique().tolist())
        missing = sorted(set(wanted) - set(found))

        if missing:
            raise ValueError(
                "SAMPLE_5DAY is missing requested dates.\n"
                f"Requested: {wanted}\nFound: {found}\nMissing: {missing}"
            )

        return out

    # FULL_247 e ROBUSTNESS_250 non restringono le giornate: le early close sono gia'
    # state incluse o escluse in load_spx_full_year_trailing.
    if RUN_SCOPE_TRAILING in ("FULL_247", "ROBUSTNESS_250"):
        return spx.copy()

    raise ValueError(f"Unknown RUN_SCOPE: {RUN_SCOPE_TRAILING}")


# ---------------------------------------------------------------------
# PANNELLO GIORNALIERO PER IL CROSS-CHECK SU SpotRV_open
# ---------------------------------------------------------------------
def load_daily_panel_spotrv_open() -> pd.DataFrame:
    if not REGRESSION_PANEL_FILE.exists():
        raise FileNotFoundError(
            f"Daily regression panel not found:\n{REGRESSION_PANEL_FILE}"
        )

    df = pd.read_csv(REGRESSION_PANEL_FILE, dtype=str, keep_default_na=False)

    required = ["quote_date", "SpotRV_open", "n_returns_open_window"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Daily regression panel missing required columns: {missing}\n"
            f"Panel file: {REGRESSION_PANEL_FILE}"
        )

    out = pd.DataFrame()
    out["quote_date"] = clean_date_series(df["quote_date"])
    out["SpotRV_open"] = pd.to_numeric(
        df["SpotRV_open"].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    out["panel_n_returns_open_window"] = pd.to_numeric(
        df["n_returns_open_window"], errors="coerce"
    )

    out = out[
        out["quote_date"].notna()
        & out["quote_date"].astype(str).str.startswith(str(YEAR))
    ].copy()

    return out.drop_duplicates(subset=["quote_date"], keep="last").reset_index(drop=True)


# ---------------------------------------------------------------------
# COSTRUZIONE DI TrailingRV
# ---------------------------------------------------------------------
def independent_rv_through(day: pd.DataFrame, through_seconds: int) -> float:
    """
    Ricalcola la varianza realizzata direttamente dai prezzi di chiusura, dal primo rendimento
    fino a through_seconds incluso. E' il controllo aritmetico interno di TrailingRV, distinto
    dal cross-check su SpotRV_open, che confronta due pipeline.
    """
    sub_day = day[
        (day["_time_seconds"] >= FIRST_RETURN_SECONDS - 60)
        & (day["_time_seconds"] <= through_seconds)
    ].sort_values("_time_seconds")

    closes = sub_day["spx_close"].to_numpy(dtype=float)

    if len(closes) < 2:
        return 0.0

    if np.any(~np.isfinite(closes)) or np.any(closes <= 0):
        return np.nan

    log_rets = np.diff(np.log(closes))

    return float(np.sum(log_rets ** 2))


def compute_trailing_for_day(day: pd.DataFrame):
    day = day.sort_values("_time_seconds").reset_index(drop=True).copy()

    quote_date = str(day["quote_date"].iloc[0])

    observed_max = int(day["_time_seconds"].max())
    day_grid = grid_times_for_day(quote_date, observed_max)

    output_rows = []
    window_rows = []

    for grid_seconds in day_grid:
        grid_time = seconds_to_time_string(grid_seconds)

        expected_n_returns = expected_n_returns_for_grid_time(grid_seconds)

        # Finestra dal primo rendimento (09:32, incluso) fino al punto di griglia incluso.
        window = day[
            (day["_time_seconds"] >= FIRST_RETURN_SECONDS)
            & (day["_time_seconds"] <= grid_seconds)
            & day["squared_log_return"].notna()
            & np.isfinite(day["squared_log_return"])
        ].copy()

        n_returns_used = int(len(window))
        valid_return_count = int(n_returns_used == expected_n_returns)

        if valid_return_count == 1:
            trailing_rv = float(window["squared_log_return"].sum())
            first_return_time = str(window["quote_time"].iloc[0])
            last_return_time = str(window["quote_time"].iloc[-1])
        else:
            trailing_rv = np.nan
            first_return_time = str(window["quote_time"].iloc[0]) if not window.empty else ""
            last_return_time = str(window["quote_time"].iloc[-1]) if not window.empty else ""

        grid_timestamp_present = int((day["_time_seconds"] == grid_seconds).any())

        output_rows.append({
            "quote_date": quote_date,
            "grid_time": grid_time,
            "TrailingRV": trailing_rv,
            "n_returns_used": n_returns_used,
            "expected_n_returns_used": expected_n_returns,
            "valid_return_count": valid_return_count,
            "grid_timestamp_present": grid_timestamp_present,
        })

        # Ricalcolo indipendente sullo stesso punto, per il controllo aritmetico.
        independent_rv = independent_rv_through(day, grid_seconds)

        window_rows.append({
            "quote_date": quote_date,
            "grid_time": grid_time,
            "independent_rv": independent_rv,
            "abs_diff_vs_independent": (abs(trailing_rv - independent_rv)
                                        if np.isfinite(trailing_rv)
                                        and np.isfinite(independent_rv) else np.nan),
            "first_return_time": first_return_time,
            "last_return_time": last_return_time,
            "expected_first_return_time": FIRST_RETURN_TIME,
            "expected_last_return_time": grid_time,
            "n_returns_used": n_returns_used,
            "expected_n_returns_used": expected_n_returns,
            "valid_return_count": valid_return_count,
            "grid_timestamp_present": grid_timestamp_present,
        })

    return pd.DataFrame(output_rows), pd.DataFrame(window_rows)


def build_trailing_rv_panel(spx_returns: pd.DataFrame):
    all_panels = []
    all_windows = []

    dates = sorted(spx_returns["quote_date"].dropna().unique().tolist())

    for idx, quote_date in enumerate(dates, start=1):
        day = spx_returns[spx_returns["quote_date"] == quote_date].copy()

        panel_day, window_day = compute_trailing_for_day(day)

        all_panels.append(panel_day)
        all_windows.append(window_day)

        if idx % 50 == 0 or idx == len(dates):
            print(f"    processate {idx}/{len(dates)} giornate", flush=True)

    panel = pd.concat(all_panels, ignore_index=True)
    window_info = pd.concat(all_windows, ignore_index=True)

    for frame in (panel, window_info):
        frame["_date_dt"] = pd.to_datetime(frame["quote_date"], errors="coerce")
        frame["_grid_seconds"] = parse_time_to_seconds(frame["grid_time"])

    panel = (
        panel.sort_values(["_date_dt", "_grid_seconds"])
        .drop(columns=["_date_dt", "_grid_seconds"]).reset_index(drop=True)
    )

    window_info = (
        window_info.sort_values(["_date_dt", "_grid_seconds"])
        .drop(columns=["_date_dt", "_grid_seconds"]).reset_index(drop=True)
    )

    return panel, window_info


# ---------------------------------------------------------------------
# DIAGNOSTICA E VALIDAZIONE
# ---------------------------------------------------------------------
def build_spotrv_open_crosscheck(trailing_panel, daily_panel) -> pd.DataFrame:
    ten_am = trailing_panel[
        trailing_panel["grid_time"] == SPOTRV_OPEN_CROSSCHECK_TIME
    ][[
        "quote_date", "grid_time", "TrailingRV",
        "n_returns_used", "expected_n_returns_used", "valid_return_count",
    ]].copy()

    out = ten_am.merge(
        daily_panel[["quote_date", "SpotRV_open", "panel_n_returns_open_window"]],
        on="quote_date", how="left", validate="one_to_one",
    )

    out["abs_diff_TrailingRV_1000_vs_SpotRV_open"] = (
        out["TrailingRV"] - out["SpotRV_open"]
    ).abs()

    out["return_counts_match"] = (
        out["n_returns_used"] == out["panel_n_returns_open_window"]
    ).astype(int)

    out["passed"] = (
        out["SpotRV_open"].notna()
        & np.isfinite(out["SpotRV_open"])
        & out["TrailingRV"].notna()
        & np.isfinite(out["TrailingRV"])
        & (out["abs_diff_TrailingRV_1000_vs_SpotRV_open"] <= SPOTRV_CROSSCHECK_TOL)
    ).astype(int)

    out["crosscheck_tolerance"] = SPOTRV_CROSSCHECK_TOL

    return out


def build_diagnostics_trailing(trailing_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    first_grid_time = seconds_to_time_string(GRID_START_SECONDS)

    for quote_date, day in trailing_panel.groupby("quote_date", sort=True):
        day = day.sort_values("grid_time").reset_index(drop=True).copy()

        trailing_values = pd.to_numeric(day["TrailingRV"], errors="coerce")
        diffs = trailing_values.diff()

        monotonic_violations = day[diffs < -FLOAT_TOL].copy()

        row_first = day[day["grid_time"] == first_grid_time].copy()
        row_1000 = day[day["grid_time"] == SPOTRV_OPEN_CROSSCHECK_TIME].copy()

        bad_return_count = day[day["valid_return_count"] != 1].copy()
        missing_grid_timestamp = day[day["grid_timestamp_present"] != 1].copy()

        rows.append({
            "quote_date": quote_date,
            "grid_rows": int(len(day)),
            "valid_trailing_rows": int(day["TrailingRV"].notna().sum()),
            "bad_return_count_rows": int(len(bad_return_count)),
            "missing_grid_timestamp_rows": int(len(missing_grid_timestamp)),
            "first_grid_time": first_grid_time,
            "TrailingRV_first_grid": float(row_first["TrailingRV"].iloc[0])
                if not row_first.empty else np.nan,
            "n_returns_first_grid": int(row_first["n_returns_used"].iloc[0])
                if not row_first.empty else -1,
            "TrailingRV_1000": float(row_1000["TrailingRV"].iloc[0])
                if not row_1000.empty else np.nan,
            "n_returns_1000": int(row_1000["n_returns_used"].iloc[0])
                if not row_1000.empty else -1,
            "max_TrailingRV": float(trailing_values.max())
                if trailing_values.notna().any() else np.nan,
            "monotonic_non_decreasing": int(len(monotonic_violations) == 0),
            "monotonic_violation_count": int(len(monotonic_violations)),
        })

    return pd.DataFrame(rows)


def max_abs_spotrv_diff(spotrv_crosscheck: pd.DataFrame) -> float:
    """
    Massimo scarto assoluto fra TrailingRV alle 10:00 e SpotRV_open del pannello. Usata sia
    dalla validazione sia dal riepilogo, cosi' i due valori coincidono.
    """
    if len(spotrv_crosscheck) == 0:
        return float("nan")

    d = pd.to_numeric(
        spotrv_crosscheck["abs_diff_TrailingRV_1000_vs_SpotRV_open"],
        errors="coerce",
    )

    d = d[d.notna()]

    return float(d.max()) if len(d) else float("nan")


def build_validation_trailing(trailing_panel, diagnostics, spotrv_crosscheck,
                              return_window_info) -> pd.DataFrame:
    rows = []

    def add(group, check, expected, actual, passed, details=""):
        rows.append({"validation_group": group, "check": check,
                     "expected": expected, "actual": actual,
                     "passed": int(passed), "details": details})

    n_days = int(trailing_panel["quote_date"].nunique())
    n_rows = int(len(trailing_panel))

    if RUN_SCOPE_TRAILING == "SAMPLE_5DAY":
        expected_days = len(SAMPLE_DATES)
        expected_rows = expected_days * EXPECTED_GRID_ROWS_PER_DAY
    elif RUN_SCOPE_TRAILING == "FULL_247":
        expected_days = EXPECTED_FULL_TRADING_DAYS_AFTER_EARLY_CLOSE_EXCLUSION
        expected_rows = expected_days * EXPECTED_GRID_ROWS_PER_DAY
    else:
        # ROBUSTNESS_250: le tre early close hanno una griglia piu' corta, quindi il
        # conteggio atteso e' la somma dei punti per giornata, non un prodotto.
        expected_days = EXPECTED_ROBUSTNESS_TRADING_DAYS

        n_early_points = len(range(
            GRID_START_SECONDS,
            min(GRID_END_SECONDS, EARLY_CLOSE_TIME_SECONDS - GRID_SPACING_SECONDS) + 1,
            GRID_SPACING_SECONDS,
        ))

        expected_rows = (
            (expected_days - len(EARLY_CLOSE_DATES)) * EXPECTED_GRID_ROWS_PER_DAY
            + len(EARLY_CLOSE_DATES) * n_early_points
        )

    add("sample_scope", "trading_days", expected_days, n_days,
        n_days == expected_days, RUN_SCOPE_TRAILING)

    add("sample_scope", "total_grid_rows", expected_rows, n_rows,
        n_rows == expected_rows,
        f"Expected days x {EXPECTED_GRID_ROWS_PER_DAY} grid rows.")

    early_present = sorted(
        set(trailing_panel["quote_date"].astype(str).unique()) & set(EARLY_CLOSE_DATES)
    )
    expected_early = len(EARLY_CLOSE_DATES) if INCLUDE_EARLY_CLOSES_TRAILING else 0

    add("sample_scope", "early_close_dates_scope", expected_early, len(early_present),
        len(early_present) == expected_early,
        f"RUN_SCOPE={RUN_SCOPE_TRAILING}; presenti: " + ";".join(early_present))

    # Le giornate regolari devono avere la griglia piena, le early close una piu' corta.
    regular = diagnostics[~diagnostics["quote_date"].isin(EARLY_CLOSE_DATES)]
    early = diagnostics[diagnostics["quote_date"].isin(EARLY_CLOSE_DATES)]

    bad_regular = regular[regular["grid_rows"] != EXPECTED_GRID_ROWS_PER_DAY]
    add("grid", f"regular_days_have_{EXPECTED_GRID_ROWS_PER_DAY}_grid_rows", 0,
        len(bad_regular), len(bad_regular) == 0,
        ";".join(bad_regular["quote_date"].astype(str).tolist()[:50]))

    if INCLUDE_EARLY_CLOSES_TRAILING and not early.empty:
        n_early_expected = len(range(
            GRID_START_SECONDS,
            min(GRID_END_SECONDS, EARLY_CLOSE_TIME_SECONDS - GRID_SPACING_SECONDS) + 1,
            GRID_SPACING_SECONDS,
        ))

        bad_early = early[early["grid_rows"] != n_early_expected]

        add("grid", "early_close_days_have_shorter_grid", n_early_expected,
            sorted(early["grid_rows"].unique().tolist()), len(bad_early) == 0,
            "La griglia si ferma a chiusura meno dieci minuti, cioe' alle 12:50.")

    # Il primo punto (09:40) accumula nove rendimenti, da 09:32 a 09:40.
    expected_first = expected_n_returns_for_grid_time(GRID_START_SECONDS)
    bad_first = diagnostics[diagnostics["n_returns_first_grid"] != expected_first]
    add("definition", "first_grid_point_uses_expected_returns", expected_first,
        len(bad_first), len(bad_first) == 0,
        f"Al primo punto ({seconds_to_time_string(GRID_START_SECONDS)}) "
        f"TrailingRV somma {expected_first} rendimenti da {FIRST_RETURN_TIME}. "
        + ";".join(bad_first["quote_date"].astype(str).tolist()[:50]))

    bad_1000 = diagnostics[diagnostics["n_returns_1000"] != SPOTRV_OPEN_EXPECTED_RETURNS]
    add("definition", "ten_am_uses_expected_returns", SPOTRV_OPEN_EXPECTED_RETURNS,
        len(bad_1000), len(bad_1000) == 0,
        "Deve coincidere con n_returns_open_window del pannello. "
        + ";".join(bad_1000["quote_date"].astype(str).tolist()[:50]))

    monotonic_bad = diagnostics[diagnostics["monotonic_non_decreasing"] != 1]
    add("definition", "TrailingRV_monotonic_non_decreasing_all_days", 0,
        len(monotonic_bad), len(monotonic_bad) == 0,
        ";".join(monotonic_bad["quote_date"].astype(str).tolist()[:50]))

    bad_return_counts = diagnostics[diagnostics["bad_return_count_rows"] != 0]
    add("return_counts", "all_grid_rows_use_expected_number_of_returns", 0,
        len(bad_return_counts), len(bad_return_counts) == 0,
        ";".join(bad_return_counts["quote_date"].astype(str).tolist()[:50]))

    missing_grid = diagnostics[diagnostics["missing_grid_timestamp_rows"] != 0]
    add("grid", "all_grid_timestamps_present_in_raw_spx", 0, len(missing_grid),
        len(missing_grid) == 0,
        ";".join(missing_grid["quote_date"].astype(str).tolist()[:50]))

    # Controllo aritmetico: TrailingRV deve coincidere con il ricalcolo diretto dai prezzi.
    if "abs_diff_vs_independent" in return_window_info.columns:
        d = pd.to_numeric(return_window_info["abs_diff_vs_independent"],
                          errors="coerce")
        d_valid = d[d.notna()]
        n_bad = int((d_valid > 1e-12).sum())
        max_d = float(d_valid.max()) if not d_valid.empty else np.nan

        add("arithmetic", "trailing_rv_matches_independent_recomputation", 0,
            n_bad, n_bad == 0,
            f"Ricalcolo diretto dai prezzi contro la somma usata nella "
            f"costruzione, su {len(d_valid):,} punti. Scarto massimo "
            f"{max_d:.3e}, tolleranza 1e-12.")

    crosscheck_fail = spotrv_crosscheck[spotrv_crosscheck["passed"] != 1]
    add("crosscheck", "TrailingRV_1000_equals_SpotRV_open", 0, len(crosscheck_fail),
        len(crosscheck_fail) == 0,
        f"Tolerance={SPOTRV_CROSSCHECK_TOL}. Failed dates: "
        + ";".join(crosscheck_fail["quote_date"].astype(str).tolist()[:50]))

    count_mismatch = spotrv_crosscheck[spotrv_crosscheck["return_counts_match"] != 1]
    add("crosscheck", "return_counts_match_panel", 0, len(count_mismatch),
        len(count_mismatch) == 0,
        "n_returns_used alle 10:00 contro n_returns_open_window del pannello. "
        + ";".join(count_mismatch["quote_date"].astype(str).tolist()[:50]))

    max_abs_diff = max_abs_spotrv_diff(spotrv_crosscheck)

    add("crosscheck", "max_abs_diff_TrailingRV_1000_vs_SpotRV_open",
        f"<= {SPOTRV_CROSSCHECK_TOL}", max_abs_diff,
        np.isfinite(max_abs_diff) and max_abs_diff <= SPOTRV_CROSSCHECK_TOL,
        "Cross-pipeline check contro il pannello corretto.")

    return pd.DataFrame(rows)


def trailing_rv_10min():
    print("=" * 140)
    print("TRAILING RV 10-MINUTE GRID CONSTRUCTION")
    print("=" * 140)
    print(f"RUN_SCOPE:                 {RUN_SCOPE_TRAILING}")
    print(f"DATA_DIR:                  {DATA_DIR}")
    print(f"SPX folder:                {SPX_DIR}  (CSV di 05_spx_minute_bars)")
    print(f"Daily panel file:          {REGRESSION_PANEL_FILE}")
    print(f"Output folder:             {OUTPUT_DIR}")
    print(f"Grid:                      {GRID_START_TIME} to {GRID_END_TIME}, "
          f"ogni {GRID_SPACING_MINUTES} minuti, {EXPECTED_GRID_ROWS_PER_DAY} punti")
    print(f"Primo rendimento:          {FIRST_RETURN_TIME} (il 09:31 non esiste)")
    print(f"Early closes:              "
          + ("INCLUSE, con griglia piu' corta" if INCLUDE_EARLY_CLOSES_TRAILING
             else f"escluse, {sorted(EARLY_CLOSE_DATES)}"))
    print("=" * 140)

    print("Loading SPX one-minute source...")
    spx_raw_full = load_spx_full_year_trailing()

    print("Restricting run scope...")
    spx_raw = restrict_run_scope(spx_raw_full)

    print(f"    RUN_SCOPE rows: {len(spx_raw):,}")
    print(f"    RUN_SCOPE days: {spx_raw['quote_date'].nunique():,}")

    print("Adding one-minute close-to-close log returns...")
    spx_returns = add_one_minute_returns(spx_raw)

    print("Loading daily panel SpotRV_open for cross-check...")
    daily_panel = load_daily_panel_spotrv_open()

    print("Building TrailingRV 10-minute grid panel...")
    trailing_panel, return_window_info = build_trailing_rv_panel(spx_returns)

    print("Building SpotRV_open cross-check...")
    spotrv_crosscheck = build_spotrv_open_crosscheck(trailing_panel, daily_panel)

    print("Building diagnostics and validation...")
    diagnostics = build_diagnostics_trailing(trailing_panel)
    validation = build_validation_trailing(trailing_panel, diagnostics, spotrv_crosscheck,
                                           return_window_info)

    safe_write_csv(trailing_panel, TRAILING_PANEL_FILE)
    safe_write_csv(diagnostics, TRAILING_DIAGNOSTICS_FILE)
    safe_write_csv(validation, TRAILING_VALIDATION_FILE)
    safe_write_csv(spotrv_crosscheck, TRAILING_SPOTRV_CROSSCHECK_FILE)
    safe_write_csv(return_window_info, TRAILING_RETURN_WINDOW_FILE)

    all_validation_passed = int((validation["passed"] == 1).all())
    failed_validation = validation[validation["passed"] != 1].copy()

    print()
    print("=" * 140)
    print("COMPLETED TRAILING RV 10-MINUTE GRID CONSTRUCTION")
    print("=" * 140)
    print(f"Rows written:                 {len(trailing_panel):,}")
    print(f"Days written:                 {trailing_panel['quote_date'].nunique():,}")
    print(f"All validation passed:        {all_validation_passed}")

    print()
    print("--- CONTEGGI DI RENDIMENTI AI PUNTI CHIAVE ---")
    print(f"  al primo punto ({GRID_START_TIME}): attesi "
          f"{expected_n_returns_for_grid_time(GRID_START_SECONDS)}, "
          f"osservati {sorted(diagnostics['n_returns_first_grid'].unique().tolist())}")
    print(f"  alle {SPOTRV_OPEN_CROSSCHECK_TIME}: attesi {SPOTRV_OPEN_EXPECTED_RETURNS}, "
          f"osservati {sorted(diagnostics['n_returns_1000'].unique().tolist())}")

    print()
    print("--- CROSS-CHECK CONTRO SpotRV_open DEL PANNELLO ---")
    print(f"  giornate confrontate: {len(spotrv_crosscheck)}")
    print(f"  scarto massimo:       {max_abs_spotrv_diff(spotrv_crosscheck):.3e}   "
          f"(tolleranza {SPOTRV_CROSSCHECK_TOL})")
    print(f"  giornate che passano: {int(spotrv_crosscheck['passed'].sum())}")
    print("  TrailingRV alle 10:00 e SpotRV_open sono costruite da due pipeline")
    print("  diverse: se coincidono, l'aggregazione del pannello e' verificata.")

    print()
    print("VALIDATION RESULTS")
    print("=" * 140)
    print(validation.to_string(index=False))

    print()
    print("CONTROLLI FALLITI")
    print("=" * 140)
    print("Nessuno." if failed_validation.empty
          else failed_validation.to_string(index=False))
    print("=" * 140)

    if all_validation_passed != 1:
        raise SystemExit(
            "Parte 2: controlli di validazione di TrailingRV falliti "
            "(vedi CONTROLLI FALLITI sopra)."
        )


# =====================================================================
# PARTE 3: PANNELLO INTRADAY UNITO
# =====================================================================
RUN_SCOPE_PANEL = "FULL_247"
# RUN_SCOPE_PANEL = "SAMPLE_5DAY"

EXPECTED_FULL_DAYS = 247

# Alle 09:40 la superficie e' gia' misurabile: a ogni punto della griglia MFIV e misure di
# stock non devono mancare mai (controlli no_boundary_missingness).

MFIV_FOLDER_NAME = "03_mfiv"
MFIV_FILE_NAME = "03_mfiv_intraday_2025_all.csv"

GAMMA_CBOE_FOLDER_NAME = "04_gamma_herfindahl"
GAMMA_CBOE_FILE_NAME = "04_gamma_cboe_intraday_2025_all.csv"

GAMMA_INDEP_FOLDER_NAME = "04_gamma_herfindahl"
GAMMA_INDEP_FILE_NAME = "04_gamma_intraday_2025_all.csv"

# TrailingRV e FwdRV: scritti dalle parti 2 e 1, nella stessa cartella.
TRAILING_FOLDER_NAME = "08_intraday_panel"
TRAILING_FILE_NAME = "08_trailing_rv_10min_2025_full_247day.csv"

FWD_CANDIDATES = [
    (
        "08_intraday_panel",
        "08_fixed_horizon_fwdrv_10min_2025_primary_247.csv",
    ),
]

if RUN_SCOPE_PANEL == "SAMPLE_5DAY":
    MERGED_STEM = "08_merged_intraday_panel_2025_sample_5day"
    EXPECTED_DAYS = len(SAMPLE_DATES)
elif RUN_SCOPE_PANEL == "FULL_247":
    MERGED_STEM = "08_merged_intraday_panel_2025_primary_247"
    EXPECTED_DAYS = EXPECTED_FULL_DAYS
else:
    raise ValueError("RUN_SCOPE must be SAMPLE_5DAY or FULL_247.")

MFIV_FILE = DATA_DIR / MFIV_FOLDER_NAME / MFIV_FILE_NAME
GAMMA_CBOE_FILE = DATA_DIR / GAMMA_CBOE_FOLDER_NAME / GAMMA_CBOE_FILE_NAME
GAMMA_INDEP_FILE = DATA_DIR / GAMMA_INDEP_FOLDER_NAME / GAMMA_INDEP_FILE_NAME
TRAILING_FILE = DATA_DIR / TRAILING_FOLDER_NAME / TRAILING_FILE_NAME

MERGED_PANEL_FILE = OUTPUT_DIR / f"{MERGED_STEM}.csv"
MERGED_VALIDATION_FILE = OUTPUT_DIR / f"{MERGED_STEM}_validation.csv"
MERGED_MISSINGNESS_BY_GRID_FILE = OUTPUT_DIR / f"{MERGED_STEM}_missingness_by_grid_time.csv"
MERGED_SOURCE_KEY_DIAGNOSTICS_FILE = OUTPUT_DIR / f"{MERGED_STEM}_source_key_diagnostics.csv"
MERGED_SKELETON_CROSSCHECK_FILE = OUTPUT_DIR / f"{MERGED_STEM}_fwd_trailing_skeleton_crosscheck.csv"
MERGED_MISSING_ROWS_DETAIL_FILE = OUTPUT_DIR / f"{MERGED_STEM}_missing_rows_detail.csv"
MERGED_FIXED_EFFECTS_KEY_CHECK_FILE = OUTPUT_DIR / f"{MERGED_STEM}_fixed_effects_key_check.csv"


def build_expected_grid_times():
    out = []
    current = pd.Timestamp(f"2000-01-01 {GRID_START_TIME}")
    end = pd.Timestamp(f"2000-01-01 {GRID_END_TIME}")

    while current <= end:
        out.append(current.strftime("%H:%M:%S"))
        current = current + pd.Timedelta(minutes=GRID_SPACING_MINUTES)

    return out


EXPECTED_GRID_TIMES = build_expected_grid_times()
# EXPECTED_GRID_ROWS_PER_DAY (38) e' quello della sezione comune e coincide con
# len(EXPECTED_GRID_TIMES): stessi estremi e stessa spaziatura.
EXPECTED_TOTAL_ROWS = EXPECTED_DAYS * EXPECTED_GRID_ROWS_PER_DAY


def restrict_to_run_scope(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()

    if RUN_SCOPE_PANEL == "SAMPLE_5DAY":
        out = out[out["quote_date"].isin(SAMPLE_DATES)].copy()

        found = sorted(out["quote_date"].dropna().unique().tolist())
        missing = sorted(set(SAMPLE_DATES) - set(found))

        if missing:
            raise ValueError(
                f"{label} is missing requested sample dates.\n"
                f"Requested: {SAMPLE_DATES}\nFound: {found}\nMissing: {missing}"
            )

    return out.reset_index(drop=True)


def add_fixed_effect_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    date_dt = pd.to_datetime(out["quote_date"], errors="coerce")
    iso = date_dt.dt.isocalendar()

    out["date_fe"] = out["quote_date"].astype(str)
    out["iso_year"] = iso["year"].astype("Int64")
    out["iso_week"] = iso["week"].astype("Int64")
    out["iso_week_key"] = (
        out["iso_year"].astype(str) + "-W" + out["iso_week"].astype(str).str.zfill(2)
    )
    out["week_x_time_fe"] = (
        out["iso_week_key"].astype(str) + "_" + out["grid_time"].astype(str)
    )

    return out


# ---------------------------------------------------------------------
# SCHELETRI FwdRV E TrailingRV
# ---------------------------------------------------------------------
def find_fwd_file():
    checked = []

    for folder_name, file_name in FWD_CANDIDATES:
        candidate = DATA_DIR / folder_name / file_name
        checked.append(str(candidate))

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No primary 247-day fixed-horizon FwdRV file found.\nChecked:\n"
        + "\n".join(checked)
    )


def load_fwd_panel():
    fwd_file = find_fwd_file()

    df = pd.read_csv(fwd_file, dtype=str, keep_default_na=False)

    required = [
        "quote_date", "grid_time",
        "FwdRV_5", "FwdRV_15", "FwdRV_30", "FwdRV_60",
        "h5_valid", "h15_valid", "h30_valid", "h60_valid",
    ]

    require_columns(df, required, "FwdRV panel")

    out = df[required].copy()
    out["quote_date"] = clean_date_series(out["quote_date"])
    out["grid_time"] = out["grid_time"].astype(str).apply(normalize_time_string)

    for col in required[2:]:
        out[col] = to_numeric_series(out[col])

    out = out[
        out["quote_date"].notna()
        & out["quote_date"].astype(str).str.startswith(str(YEAR))
        & out["grid_time"].isin(EXPECTED_GRID_TIMES)
    ].copy()

    out = restrict_to_run_scope(out, "FwdRV panel")

    out["_date_dt"] = pd.to_datetime(out["quote_date"], errors="coerce")
    out["_grid_seconds"] = parse_time_to_seconds(out["grid_time"])
    out = (
        out.sort_values(["_date_dt", "_grid_seconds"])
        .drop(columns=["_date_dt", "_grid_seconds"]).reset_index(drop=True)
    )

    return out, fwd_file


def load_trailing_panel() -> pd.DataFrame:
    require_file(TRAILING_FILE, "TrailingRV full 247-day panel")

    df = pd.read_csv(TRAILING_FILE, dtype=str, keep_default_na=False)

    required = [
        "quote_date", "grid_time", "TrailingRV",
        "n_returns_used", "expected_n_returns_used",
        "valid_return_count", "grid_timestamp_present",
    ]

    require_columns(df, required, "TrailingRV panel")

    out = df[required].copy()
    out["quote_date"] = clean_date_series(out["quote_date"])
    out["grid_time"] = out["grid_time"].astype(str).apply(normalize_time_string)

    for col in required[2:]:
        out[col] = to_numeric_series(out[col])

    out = out[
        out["quote_date"].notna()
        & out["quote_date"].astype(str).str.startswith(str(YEAR))
        & out["grid_time"].isin(EXPECTED_GRID_TIMES)
    ].copy()

    out = restrict_to_run_scope(out, "TrailingRV panel")

    out = out.rename(columns={
        "n_returns_used": "TrailingRV_n_returns_used",
        "expected_n_returns_used": "TrailingRV_expected_n_returns_used",
        "valid_return_count": "TrailingRV_valid_return_count",
        "grid_timestamp_present": "TrailingRV_grid_timestamp_present",
    })

    out["_date_dt"] = pd.to_datetime(out["quote_date"], errors="coerce")
    out["_grid_seconds"] = parse_time_to_seconds(out["grid_time"])
    out = (
        out.sort_values(["_date_dt", "_grid_seconds"])
        .drop(columns=["_date_dt", "_grid_seconds"]).reset_index(drop=True)
    )

    return out


def build_skeleton_crosscheck(fwd: pd.DataFrame, trailing: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check, expected, actual, passed, details=""):
        rows.append({"check": check, "expected": expected, "actual": actual,
                     "passed": int(passed), "details": details})

    # Stessa griglia e stesse giornate: i due scheletri devono coincidere esattamente.
    add("fwd_rows", EXPECTED_TOTAL_ROWS, len(fwd), len(fwd) == EXPECTED_TOTAL_ROWS,
        f"{EXPECTED_DAYS} giornate x {EXPECTED_GRID_ROWS_PER_DAY} punti.")

    add("trailing_rows", EXPECTED_TOTAL_ROWS, len(trailing),
        len(trailing) == EXPECTED_TOTAL_ROWS,
        f"{EXPECTED_DAYS} giornate x {EXPECTED_GRID_ROWS_PER_DAY} punti.")

    add("fwd_days", EXPECTED_DAYS, int(fwd["quote_date"].nunique()),
        fwd["quote_date"].nunique() == EXPECTED_DAYS)

    add("trailing_days", EXPECTED_DAYS, int(trailing["quote_date"].nunique()),
        trailing["quote_date"].nunique() == EXPECTED_DAYS)

    fwd_keys = set(
        fwd["quote_date"].astype(str) + " " + fwd["grid_time"].astype(str)
    )
    trailing_keys = set(
        trailing["quote_date"].astype(str) + " " + trailing["grid_time"].astype(str)
    )

    only_fwd = sorted(fwd_keys - trailing_keys)
    only_trailing = sorted(trailing_keys - fwd_keys)

    add("skeletons_identical", 0, len(only_fwd) + len(only_trailing),
        len(only_fwd) == 0 and len(only_trailing) == 0,
        "solo in fwd: " + ";".join(only_fwd[:15])
        + " | solo in trailing: " + ";".join(only_trailing[:15]))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# DIAGNOSTICA DELLE CHIAVI DELLE FONTI
# ---------------------------------------------------------------------
def source_key_diagnostic(source_name: str, df: pd.DataFrame, value_cols) -> dict:
    duplicate_count = int(df.duplicated(["quote_date", "quote_time"]).sum())

    duplicate_keys = df[df.duplicated(["quote_date", "quote_time"], keep=False)].copy()

    return {
        "source_name": source_name,
        "rows_after_filter": int(len(df)),
        "unique_days": int(df["quote_date"].nunique()),
        "unique_times": int(df["quote_time"].nunique()),
        "duplicate_quote_date_quote_time_rows": duplicate_count,
        "duplicate_key_examples": ";".join(
            (duplicate_keys["quote_date"].astype(str) + " "
             + duplicate_keys["quote_time"].astype(str))
            .drop_duplicates().head(30).tolist()
        ),
        "missing_any_required_value_rows": int(df[value_cols].isna().any(axis=1).sum()),
        "first_time": str(df["quote_time"].min()) if not df.empty else "",
        "last_time": str(df["quote_time"].max()) if not df.empty else "",
    }


# ---------------------------------------------------------------------
# FONTI INTRADAY: MFIV E GAMMA
# ---------------------------------------------------------------------
def load_mfiv_source(base_dates, base_grid_times):
    require_file(MFIV_FILE, "MFIV intraday source")

    raw = pd.read_csv(MFIV_FILE, dtype=str, keep_default_na=False)

    required = ["quote_date", "quote_time", "mfiv"]

    # tau serve ad annualizzare la MFIV (MFIV_ann, sotto); mfiv_flag e truncation_ratio
    # servono per i filtri e per l'Appendice D.
    optional = ["mfiv_flag", "tau", "truncation_ratio", "sigma_atm",
                "mfiv_annualized_vol", "n_otm_strikes"]

    require_columns(raw, required, "MFIV intraday source")

    keep_cols = required + [c for c in optional if c in raw.columns]

    out = raw[keep_cols].copy()
    out["quote_date"] = clean_date_series(out["quote_date"])
    out["quote_time"] = out["quote_time"].astype(str).apply(normalize_time_string)
    out["MFIV"] = to_numeric_series(out["mfiv"])

    for c in ["tau", "truncation_ratio", "sigma_atm",
              "mfiv_annualized_vol", "n_otm_strikes"]:
        if c in out.columns:
            out[c] = to_numeric_series(out[c])

    out = out.drop(columns=["mfiv"])

    # MFIV annualizzata, mfiv / tau: non dipende dal tempo residuo ed e' su questa che il
    # benchmark di uno slope non distorto e' 1, come nel disegno giornaliero.
    if "tau" in out.columns:
        out["MFIV_ann"] = np.where(
            out["tau"].notna() & (out["tau"] > 0) & out["MFIV"].notna(),
            out["MFIV"] / out["tau"], np.nan,
        )

    out = out[
        out["quote_date"].isin(base_dates)
        & out["quote_time"].isin(base_grid_times)
    ].copy()

    diag = source_key_diagnostic("MFIV", out, ["MFIV"])

    if int(out.duplicated(["quote_date", "quote_time"]).sum()) != 0:
        raise ValueError(
            "MFIV source has duplicate quote_date x quote_time rows after filtering."
        )

    return out, diag


def load_gamma_cboe_source(base_dates, base_grid_times):
    require_file(GAMMA_CBOE_FILE, "CBOE gamma intraday source")

    raw = pd.read_csv(GAMMA_CBOE_FILE, dtype=str, keep_default_na=False)

    required = [
        "quote_date", "quote_time",
        "h_oi_cboe", "h_vol_cboe", "g_oi_raw_cboe", "g_vol_raw_cboe",
        "gamma_flag", "low_oi_flag", "low_vol_flag",
        "missing_r_flag", "no_valid_cboe_gamma_flag",
    ]

    optional = ["gamma_source", "h_oi_cboe_contract", "h_oi_cboe_nosolver"]

    require_columns(raw, required, "CBOE gamma intraday source")

    keep_cols = required + [c for c in optional if c in raw.columns]

    out = raw[keep_cols].copy()
    out["quote_date"] = clean_date_series(out["quote_date"])
    out["quote_time"] = out["quote_time"].astype(str).apply(normalize_time_string)

    out = out.rename(columns={
        "h_oi_cboe": "H_OI_cboe",
        "h_vol_cboe": "H_VOL_cboe",
        "g_oi_raw_cboe": "G_OI_raw_cboe",
        "g_vol_raw_cboe": "G_VOL_raw_cboe",
        "gamma_flag": "gamma_flag_cboe",
        "low_oi_flag": "low_oi_flag_cboe",
        "low_vol_flag": "low_vol_flag_cboe",
        "missing_r_flag": "missing_r_flag_cboe",
        "gamma_source": "gamma_source_cboe",
        "h_oi_cboe_contract": "H_OI_cboe_contract",
        "h_oi_cboe_nosolver": "H_OI_cboe_nosolver",
    })

    for col in ["H_OI_cboe", "H_VOL_cboe", "G_OI_raw_cboe", "G_VOL_raw_cboe",
                "low_oi_flag_cboe", "low_vol_flag_cboe", "missing_r_flag_cboe",
                "no_valid_cboe_gamma_flag", "H_OI_cboe_contract",
                "H_OI_cboe_nosolver"]:
        if col in out.columns:
            out[col] = to_numeric_series(out[col])

    out = out[
        out["quote_date"].isin(base_dates)
        & out["quote_time"].isin(base_grid_times)
    ].copy()

    # Nel diagnostico sono richieste solo le misure di stock: quelle di volume possono mancare
    # dove il volume e' zero su tutti gli strike, e lo dichiara il flag low_vol.
    diag = source_key_diagnostic(
        "gamma_cboe", out, ["H_OI_cboe", "G_OI_raw_cboe"],
    )

    if int(out.duplicated(["quote_date", "quote_time"]).sum()) != 0:
        raise ValueError(
            "CBOE gamma source has duplicate quote_date x quote_time rows after filtering."
        )

    return out, diag


def load_gamma_indep_source(base_dates, base_grid_times):
    require_file(GAMMA_INDEP_FILE, "Independent gamma intraday source")

    raw = pd.read_csv(GAMMA_INDEP_FILE, dtype=str, keep_default_na=False)

    required = [
        "quote_date", "quote_time",
        "h_oi", "h_vol", "g_oi_raw", "g_vol_raw",
        "gamma_flag", "low_oi_flag", "low_vol_flag", "missing_r_flag",
    ]

    optional = ["h_oi_contract", "h_vol_contract", "share_gamma_oi_dropped"]

    require_columns(raw, required, "Independent gamma intraday source")

    keep_cols = required + [c for c in optional if c in raw.columns]

    out = raw[keep_cols].copy()
    out["quote_date"] = clean_date_series(out["quote_date"])
    out["quote_time"] = out["quote_time"].astype(str).apply(normalize_time_string)

    out = out.rename(columns={
        "h_oi": "H_OI_indep",
        "h_vol": "H_VOL_indep",
        "g_oi_raw": "G_OI_raw_indep",
        "g_vol_raw": "G_VOL_raw_indep",
        "gamma_flag": "gamma_flag_indep",
        "low_oi_flag": "low_oi_flag_indep",
        "low_vol_flag": "low_vol_flag_indep",
        "missing_r_flag": "missing_r_flag_indep",
        "h_oi_contract": "H_OI_indep_contract",
        "h_vol_contract": "H_VOL_indep_contract",
        "share_gamma_oi_dropped": "share_gamma_oi_dropped_indep",
    })

    for col in ["H_OI_indep", "H_VOL_indep", "G_OI_raw_indep", "G_VOL_raw_indep",
                "low_oi_flag_indep", "low_vol_flag_indep", "missing_r_flag_indep",
                "H_OI_indep_contract", "H_VOL_indep_contract",
                "share_gamma_oi_dropped_indep"]:
        if col in out.columns:
            out[col] = to_numeric_series(out[col])

    out = out[
        out["quote_date"].isin(base_dates)
        & out["quote_time"].isin(base_grid_times)
    ].copy()

    # Come per il gamma CBOE: richieste solo le misure di stock.
    diag = source_key_diagnostic(
        "gamma_indep", out, ["H_OI_indep", "G_OI_raw_indep"],
    )

    if int(out.duplicated(["quote_date", "quote_time"]).sum()) != 0:
        raise ValueError(
            "Independent gamma source has duplicate quote_date x quote_time rows after filtering."
        )

    return out, diag


# ---------------------------------------------------------------------
# UNIONE
# ---------------------------------------------------------------------
def build_merged_panel(fwd, trailing, mfiv, gamma_cboe, gamma_indep) -> pd.DataFrame:
    out = fwd.copy()
    out["quote_time"] = out["grid_time"]

    out = out.merge(trailing, on=["quote_date", "grid_time"],
                    how="left", validate="one_to_one")

    for source in (mfiv, gamma_cboe, gamma_indep):
        out = out.merge(
            source,
            left_on=["quote_date", "grid_time"],
            right_on=["quote_date", "quote_time"],
            how="left", validate="many_to_one",
        )

        if "quote_time_y" in out.columns:
            out = out.drop(columns=["quote_time_y"])

        if "quote_time_x" in out.columns:
            out = out.rename(columns={"quote_time_x": "quote_time"})

    out = add_fixed_effect_keys(out)

    out["_date_dt"] = pd.to_datetime(out["quote_date"], errors="coerce")
    out["_grid_seconds"] = parse_time_to_seconds(out["grid_time"])
    out = (
        out.sort_values(["_date_dt", "_grid_seconds"])
        .drop(columns=["_date_dt", "_grid_seconds"]).reset_index(drop=True)
    )

    return out


# ---------------------------------------------------------------------
# DIAGNOSTICA DEI MANCANTI
# ---------------------------------------------------------------------
# L'open interest esiste sempre, quindi H_OI e G_OI_raw devono esserci a ogni punto. Il volume
# puo' essere zero su tutti gli strike, e allora il Herfindahl del flusso non e' definito: il
# flusso puo' mancare, ma solo dove il flag lo dichiara.
STOCK_CBOE = ["H_OI_cboe", "G_OI_raw_cboe"]
FLOW_CBOE = ["H_VOL_cboe", "G_VOL_raw_cboe"]
STOCK_INDEP = ["H_OI_indep", "G_OI_raw_indep"]
FLOW_INDEP = ["H_VOL_indep", "G_VOL_raw_indep"]

CORE_CBOE = STOCK_CBOE + FLOW_CBOE
CORE_INDEP = STOCK_INDEP + FLOW_INDEP


def add_missing_flags(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    out["missing_MFIV"] = out["MFIV"].isna().astype(int)

    out["missing_stock_cboe"] = out[STOCK_CBOE].isna().any(axis=1).astype(int)
    out["missing_flow_cboe"] = out[FLOW_CBOE].isna().any(axis=1).astype(int)
    out["missing_stock_indep"] = out[STOCK_INDEP].isna().any(axis=1).astype(int)
    out["missing_flow_indep"] = out[FLOW_INDEP].isna().any(axis=1).astype(int)

    # Indicatori complessivi per fonte (stock o flusso), mantenuti per gli script a valle.
    out["missing_gamma_cboe"] = out[CORE_CBOE].isna().any(axis=1).astype(int)
    out["missing_gamma_indep"] = out[CORE_INDEP].isna().any(axis=1).astype(int)

    # "core" = MFIV e misure di stock, che devono esserci sempre; il flusso ha un flag a parte.
    out["missing_any_core_predictor"] = (
        (out["missing_MFIV"] == 1)
        | (out["missing_stock_cboe"] == 1)
        | (out["missing_stock_indep"] == 1)
    ).astype(int)

    out["missing_any_flow_predictor"] = (
        (out["missing_flow_cboe"] == 1) | (out["missing_flow_indep"] == 1)
    ).astype(int)

    # Il flusso mancante e' atteso solo dove il flag lo dichiara.
    flag_cboe = out["gamma_flag_cboe"].astype(str)
    flag_indep = out["gamma_flag_indep"].astype(str)

    out["flow_missing_explained_by_flag"] = (
        (out["missing_any_flow_predictor"] == 0)
        | flag_cboe.isin(["low_vol", "low_oi", "no_valid_cboe_gamma", "near_expiry"])
        | flag_indep.isin(["low_vol", "low_oi", "near_expiry"])
    ).astype(int)

    # Riga utilizzabile nelle specifiche che usano il flusso.
    out["usable_for_flow_specs"] = (
        (out["missing_any_core_predictor"] == 0)
        & (out["missing_any_flow_predictor"] == 0)
    ).astype(int)

    return out


def build_missingness_by_grid_time(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for grid_time, sub in panel.groupby("grid_time", sort=True):
        row = {
            "grid_time": grid_time,
            "rows": int(len(sub)),
            "missing_MFIV_rows": int(sub["missing_MFIV"].sum()),
            "missing_stock_cboe_rows": int(sub["missing_stock_cboe"].sum()),
            "missing_flow_cboe_rows": int(sub["missing_flow_cboe"].sum()),
            "missing_stock_indep_rows": int(sub["missing_stock_indep"].sum()),
            "missing_flow_indep_rows": int(sub["missing_flow_indep"].sum()),
            "missing_any_core_predictor_rows": int(sub["missing_any_core_predictor"].sum()),
            "missing_any_flow_predictor_rows": int(sub["missing_any_flow_predictor"].sum()),
            "usable_for_flow_specs_rows": int(sub["usable_for_flow_specs"].sum()),
        }

        for col in CORE_CBOE + CORE_INDEP:
            row[f"missing_{col}_rows"] = int(sub[col].isna().sum())

        rows.append(row)

    out = pd.DataFrame(rows)
    out["_grid_seconds"] = parse_time_to_seconds(out["grid_time"])

    return (
        out.sort_values("_grid_seconds")
        .drop(columns=["_grid_seconds"]).reset_index(drop=True)
    )


def build_missing_rows_detail(panel: pd.DataFrame) -> pd.DataFrame:
    cols = ["quote_date", "grid_time", "missing_MFIV",
            "missing_stock_cboe", "missing_flow_cboe",
            "missing_stock_indep", "missing_flow_indep",
            "flow_missing_explained_by_flag",
            "gamma_flag_cboe", "gamma_flag_indep",
            "MFIV"] + CORE_CBOE + CORE_INDEP

    mask = (
        (panel["missing_any_core_predictor"] == 1)
        | (panel["missing_any_flow_predictor"] == 1)
    )

    return panel[mask][cols].copy()


# ---------------------------------------------------------------------
# CHIAVI DI EFFETTO FISSO
# ---------------------------------------------------------------------
def build_fixed_effects_key_check(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check, expected, actual, passed, details=""):
        rows.append({"check": check, "expected": expected, "actual": actual,
                     "passed": int(passed), "details": details})

    add("date_fe_nonmissing_all_rows", 0, int(panel["date_fe"].isna().sum()),
        panel["date_fe"].isna().sum() == 0)

    add("week_x_time_fe_nonmissing_all_rows", 0,
        int(panel["week_x_time_fe"].isna().sum()),
        panel["week_x_time_fe"].isna().sum() == 0)

    add("unique_date_fe_count", EXPECTED_DAYS, int(panel["date_fe"].nunique()),
        panel["date_fe"].nunique() == EXPECTED_DAYS)

    expected_unique = panel[["iso_week_key", "grid_time"]].drop_duplicates().shape[0]

    add("week_x_time_fe_unique_count_matches_iso_week_grid_pairs",
        expected_unique, int(panel["week_x_time_fe"].nunique()),
        panel["week_x_time_fe"].nunique() == expected_unique)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# VALIDAZIONE
# ---------------------------------------------------------------------
def build_validation_panel(panel, skeleton_crosscheck, missingness_by_grid,
                           source_key_diagnostics, fixed_effects_key_check) -> pd.DataFrame:
    rows = []

    def add(group, check, expected, actual, passed, details=""):
        rows.append({"validation_group": group, "check": check,
                     "expected": expected, "actual": actual,
                     "passed": int(passed), "details": details})

    add("base_panel", "total_rows", EXPECTED_TOTAL_ROWS, int(len(panel)),
        len(panel) == EXPECTED_TOTAL_ROWS,
        f"{EXPECTED_DAYS} giornate x {EXPECTED_GRID_ROWS_PER_DAY} punti di griglia.")

    add("base_panel", "unique_days", EXPECTED_DAYS,
        int(panel["quote_date"].nunique()),
        panel["quote_date"].nunique() == EXPECTED_DAYS, RUN_SCOPE_PANEL)

    dup = int(panel.duplicated(["quote_date", "grid_time"]).sum())
    add("base_panel", "duplicate_quote_date_grid_time_rows", 0, dup, dup == 0)

    skeleton_failed = skeleton_crosscheck[skeleton_crosscheck["passed"] != 1]
    add("skeleton", "fwd_and_trailing_skeleton_identical", 0, len(skeleton_failed),
        len(skeleton_failed) == 0,
        ";".join(skeleton_failed["check"].astype(str).tolist()))

    dup_src = source_key_diagnostics[
        source_key_diagnostics["duplicate_quote_date_quote_time_rows"] != 0
    ]
    add("source_keys", "no_source_duplicate_keys", 0, len(dup_src),
        len(dup_src) == 0, ";".join(dup_src["source_name"].astype(str).tolist()))

    miss_src = source_key_diagnostics[
        source_key_diagnostics["missing_any_required_value_rows"] != 0
    ]
    add("source_keys", "no_missing_required_values_inside_filtered_source_rows",
        0, len(miss_src), len(miss_src) == 0,
        "Required = MFIV e misure di STOCK. Il flusso ha il suo controllo "
        "separato in flow_missingness. "
        + ";".join(miss_src["source_name"].astype(str).tolist()))

    fe_failed = fixed_effects_key_check[fixed_effects_key_check["passed"] != 1]
    add("fixed_effects", "fixed_effect_keys_valid", 0, len(fe_failed),
        len(fe_failed) == 0, ";".join(fe_failed["check"].astype(str).tolist()))

    # Stock e flusso si controllano separatamente: vedi la nota sopra STOCK_CBOE.
    add("no_boundary_missingness", "MFIV_never_missing", 0,
        int(panel["missing_MFIV"].sum()), panel["missing_MFIV"].sum() == 0,
        "La MFIV deve esserci a ogni punto di griglia.")

    for col, label in (("missing_stock_cboe", "stock_cboe"),
                       ("missing_stock_indep", "stock_indep")):
        n = int(panel[col].sum())
        add("no_boundary_missingness", f"{label}_never_missing", 0, n, n == 0,
            "L'open interest esiste sempre: H_OI e G_OI_raw devono esserci. "
            + ";".join(
                (panel.loc[panel[col] == 1, "quote_date"].astype(str) + " "
                 + panel.loc[panel[col] == 1, "grid_time"].astype(str)).tolist()[:20]
            ))

    n_core = int(panel["missing_any_core_predictor"].sum())
    add("usable_sample", "no_missing_core_predictors_anywhere", 0, n_core,
        n_core == 0,
        "Core = MFIV piu' le misure di stock, che devono esserci ovunque.")

    # Il flusso puo' mancare, ma solo dove il flag lo dichiara.
    n_flow = int(panel["missing_any_flow_predictor"].sum())
    n_unexplained = int((panel["flow_missing_explained_by_flag"] == 0).sum())

    add("flow_missingness", "flow_missing_only_where_flag_explains_it", 0,
        n_unexplained, n_unexplained == 0,
        f"Punti con flusso mancante: {n_flow} su {len(panel)} "
        f"({100.0 * n_flow / len(panel):.3f} per cento). Non spiegati dal flag: "
        + ";".join(
            (panel.loc[panel["flow_missing_explained_by_flag"] == 0,
                       "quote_date"].astype(str) + " "
             + panel.loc[panel["flow_missing_explained_by_flag"] == 0,
                         "grid_time"].astype(str)).tolist()[:20]
        ))

    n_usable_flow = int(panel["usable_for_flow_specs"].sum())
    add("flow_missingness", "usable_rows_for_flow_specs", EXPECTED_TOTAL_ROWS - n_flow,
        n_usable_flow, n_usable_flow == EXPECTED_TOTAL_ROWS - n_flow,
        "Righe utilizzabili nelle specifiche che usano H_VOL o G_VOL_raw. "
        "Le specifiche che usano solo lo stock usano tutte le righe.")

    add("usable_sample", "usable_rows_for_stock_specs", EXPECTED_TOTAL_ROWS,
        int(len(panel)) - n_core, len(panel) - n_core == EXPECTED_TOTAL_ROWS,
        f"{EXPECTED_DAYS} giornate x {EXPECTED_GRID_ROWS_PER_DAY} punti.")

    grid_with_missing = missingness_by_grid[
        missingness_by_grid["missing_any_core_predictor_rows"] != 0
    ]
    add("missingness_by_grid_time", "all_grid_times_have_zero_core_missing", 0,
        len(grid_with_missing), len(grid_with_missing) == 0,
        ";".join(grid_with_missing["grid_time"].astype(str).tolist()))

    grid_with_flow_missing = missingness_by_grid[
        missingness_by_grid["missing_any_flow_predictor_rows"] != 0
    ]
    add("missingness_by_grid_time", "grid_times_with_flow_missing", "diagnostic",
        len(grid_with_flow_missing), True,
        "Orari con almeno un punto a flusso mancante: "
        + ";".join(grid_with_flow_missing["grid_time"].astype(str).tolist()))

    return pd.DataFrame(rows)


def intraday_panel():
    print("=" * 150)
    print("SECTION 3.8 MERGED INTRADAY PANEL BUILD")
    print("=" * 150)
    print(f"RUN_SCOPE:                    {RUN_SCOPE_PANEL}")
    print(f"DATA_DIR:                     {DATA_DIR}")
    print(f"MFIV source:                  {MFIV_FILE}")
    print(f"CBOE gamma source:            {GAMMA_CBOE_FILE}")
    print(f"Independent gamma source:     {GAMMA_INDEP_FILE}")
    print(f"TrailingRV source:            {TRAILING_FILE}")
    print(f"Output folder:                {OUTPUT_DIR}")
    print(f"Griglia:                      {GRID_START_TIME} a {GRID_END_TIME}, "
          f"{EXPECTED_GRID_ROWS_PER_DAY} punti")
    print(f"Righe attese:                 {EXPECTED_DAYS} x {EXPECTED_GRID_ROWS_PER_DAY} "
          f"= {EXPECTED_TOTAL_ROWS:,}")
    print("=" * 150)

    print("Loading FwdRV fixed-horizon primary 247-day panel...")
    fwd_panel, fwd_file_used = load_fwd_panel()
    print(f"    FwdRV source used: {fwd_file_used}")
    print(f"    FwdRV rows:        {len(fwd_panel):,}")
    print(f"    FwdRV days:        {fwd_panel['quote_date'].nunique():,}")

    print("Loading TrailingRV 247-day panel...")
    trailing_panel = load_trailing_panel()
    print(f"    Trailing rows:     {len(trailing_panel):,}")
    print(f"    Trailing days:     {trailing_panel['quote_date'].nunique():,}")

    print("Cross-checking FwdRV and TrailingRV skeletons...")
    skeleton_crosscheck = build_skeleton_crosscheck(fwd_panel, trailing_panel)

    base_dates = set(fwd_panel["quote_date"].astype(str).unique().tolist())
    base_grid_times = set(fwd_panel["grid_time"].astype(str).unique().tolist())

    print("Loading MFIV intraday source...")
    mfiv_source, mfiv_diag = load_mfiv_source(base_dates, base_grid_times)

    print("Loading CBOE gamma intraday source...")
    gamma_cboe_source, gamma_cboe_diag = load_gamma_cboe_source(base_dates, base_grid_times)

    print("Loading independent gamma intraday source...")
    gamma_indep_source, gamma_indep_diag = load_gamma_indep_source(base_dates, base_grid_times)

    source_key_diagnostics = pd.DataFrame([mfiv_diag, gamma_cboe_diag, gamma_indep_diag])

    print("Merging panel by exact quote_date x grid_time left joins...")
    merged_panel = build_merged_panel(
        fwd=fwd_panel, trailing=trailing_panel, mfiv=mfiv_source,
        gamma_cboe=gamma_cboe_source, gamma_indep=gamma_indep_source,
    )

    merged_panel = add_missing_flags(merged_panel)
    missingness_by_grid = build_missingness_by_grid_time(merged_panel)
    missing_rows_detail = build_missing_rows_detail(merged_panel)
    fixed_effects_key_check = build_fixed_effects_key_check(merged_panel)

    validation = build_validation_panel(
        panel=merged_panel,
        skeleton_crosscheck=skeleton_crosscheck,
        missingness_by_grid=missingness_by_grid,
        source_key_diagnostics=source_key_diagnostics,
        fixed_effects_key_check=fixed_effects_key_check,
    )

    all_validation_passed = int((validation["passed"] == 1).all())
    failed_validation = validation[validation["passed"] != 1].copy()

    safe_write_csv(merged_panel, MERGED_PANEL_FILE)
    safe_write_csv(validation, MERGED_VALIDATION_FILE)
    safe_write_csv(missingness_by_grid, MERGED_MISSINGNESS_BY_GRID_FILE)
    safe_write_csv(source_key_diagnostics, MERGED_SOURCE_KEY_DIAGNOSTICS_FILE)
    safe_write_csv(skeleton_crosscheck, MERGED_SKELETON_CROSSCHECK_FILE)
    safe_write_csv(missing_rows_detail, MERGED_MISSING_ROWS_DETAIL_FILE)
    safe_write_csv(fixed_effects_key_check, MERGED_FIXED_EFFECTS_KEY_CHECK_FILE)

    print()
    print("=" * 150)
    print("COMPLETED SECTION 3.8 MERGED INTRADAY PANEL BUILD")
    print("=" * 150)
    print(f"Rows written:                     {len(merged_panel):,}")
    print(f"Days written:                     {merged_panel['quote_date'].nunique():,}")
    print(f"All validation passed:            {all_validation_passed}")

    print()
    print("--- COLONNE DEL PANNELLO ---")
    print(f"  totali: {len(merged_panel.columns)}")
    print("  " + ";".join(merged_panel.columns.tolist()))

    print()
    print("--- COPERTURA DEI PREDITTORI ---")
    n_flow = int(merged_panel["missing_any_flow_predictor"].sum())
    n_core = int(merged_panel["missing_any_core_predictor"].sum())
    print(f"  MFIV e misure di STOCK mancanti: {n_core} su {len(merged_panel):,}")
    print(f"  misure di FLUSSO mancanti:       {n_flow} su {len(merged_panel):,} "
          f"({100.0 * n_flow / len(merged_panel):.3f} per cento)")
    print(f"  righe utilizzabili con il flusso: "
          f"{int(merged_panel['usable_for_flow_specs'].sum()):,}")
    print("  Il volume puo' essere zero su tutti gli strike: quando accade il")
    print("  Herfindahl del flusso non e' definito. Non e' un buco nei dati, e' un")
    print("  buco negli scambi, e il flag di giornata lo dichiara.")

    print()
    print("SOURCE KEY DIAGNOSTICS")
    print("=" * 150)
    print(source_key_diagnostics.to_string(index=False))

    print()
    print("SKELETON CROSSCHECK")
    print("=" * 150)
    print(skeleton_crosscheck.to_string(index=False))

    print()
    print("VALIDATION RESULTS")
    print("=" * 150)
    print(validation.to_string(index=False))

    print()
    print("CONTROLLI FALLITI")
    print("=" * 150)
    print("Nessuno." if failed_validation.empty else failed_validation.to_string(index=False))

    if not missing_rows_detail.empty:
        print()
        print("DETTAGLIO RIGHE CON PREDITTORI MANCANTI")
        print("=" * 150)
        print(missing_rows_detail.head(40).to_string(index=False))

    print("=" * 150)

    if all_validation_passed != 1:
        raise SystemExit(
            "Parte 3: controlli di validazione del pannello intraday "
            "falliti (vedi CONTROLLI FALLITI sopra)."
        )


# =====================================================================
# PARTE 4: VARIANTI DEL PANNELLO INTRADAY CON BANDA DI LOG-MONEYNESS
# =====================================================================
# Pannello di base: quello scritto dalla parte 3, riletto dal disco.
BASELINE_PANEL_FOLDER_NAME = "08_intraday_panel"
BASELINE_PANEL_FILE_NAME = (
    "08_merged_intraday_panel_2025_primary_247.csv"
)

KEY_COLUMNS = ["quote_date", "grid_time"]
MFIV_OUTPUT_COLUMN = "MFIV"

# ---------------------------------------------------------------------
# COLONNE DERIVATE
# ---------------------------------------------------------------------
# MFIV_ann, il regressore delle stime intraday, si ricalcola dalla variante come mfiv/tau:
# altrimenti la stessa riga avrebbe MFIV della variante e MFIV_ann della baseline.
MFIV_DERIVED_COLUMNS = ["MFIV_ann"]

VALUE_MATCH_TOL = 1e-15
EXPECTED_DAYS_LOGM = 247

# Griglia 09:40-15:50: 38 punti per ciascuna delle 247 giornate.
EXPECTED_GRID_POINTS = 38
EXPECTED_ROWS = EXPECTED_DAYS_LOGM * EXPECTED_GRID_POINTS

# Primo punto di griglia: alle 09:40 la MFIV e' gia' misurabile e non deve mancare.
FIRST_GRID_TIME = "09:40:00"

VARIANTS = {
    "logm_005": {
        "mfiv_folder": "03_mfiv",
        "mfiv_file": "03_mfiv_intraday_2025_all_logm_005.csv",
        "expected_cutoff": 0.05,
    },
    "logm_002": {
        "mfiv_folder": "03_mfiv",
        "mfiv_file": "03_mfiv_intraday_2025_all_logm_002.csv",
        "expected_cutoff": 0.02,
    },
}

BASELINE_PANEL_FILE = (
    DATA_DIR
    / BASELINE_PANEL_FOLDER_NAME
    / BASELINE_PANEL_FILE_NAME
)

COMBINED_VALIDATION_FILE = (
    OUTPUT_DIR
    / f"{OUTPUT_FILE_PREFIX}merged_intraday_panel_2025_logm_robustness_validation_all_variants.csv"
)
COMBINED_COMPARISON_FILE = (
    OUTPUT_DIR
    / f"{OUTPUT_FILE_PREFIX}mfiv_intraday_comparison_vs_baseline_all_variants.csv"
)


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    normalized_to_original = {
        normalize_colname(column): column
        for column in df.columns
    }

    for candidate in candidates:
        normalized = normalize_colname(candidate)
        if normalized in normalized_to_original:
            return normalized_to_original[normalized]

    raise ValueError(
        f"Could not find {label}.\n"
        f"Candidate names: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def check_unique_keys(df: pd.DataFrame, label: str):
    duplicated = df.duplicated(KEY_COLUMNS, keep=False)

    if duplicated.any():
        examples = (
            df.loc[duplicated, KEY_COLUMNS]
            .sort_values(KEY_COLUMNS)
            .head(100)
        )
        raise ValueError(
            f"{label} contains duplicate quote_date x grid_time keys.\n"
            f"First examples:\n{examples.to_string(index=False)}"
        )


def add_validation_row(
    rows: list[dict],
    variant: str,
    group: str,
    check: str,
    expected,
    actual,
    passed: bool,
    details: str = "",
):
    rows.append({
        "variant": variant,
        "validation_group": group,
        "check": check,
        "expected": expected,
        "actual": actual,
        "passed": int(bool(passed)),
        "details": details,
    })


# ---------------------------------------------------------------------
# PANNELLO DI BASE
# ---------------------------------------------------------------------
def load_baseline_panel() -> pd.DataFrame:
    require_file(BASELINE_PANEL_FILE, "Baseline corrected Section 3.8 merged panel")

    baseline = pd.read_csv(
        BASELINE_PANEL_FILE,
        dtype=str,
        keep_default_na=False,
    )

    require_columns(
        baseline,
        KEY_COLUMNS + [MFIV_OUTPUT_COLUMN],
        "Baseline corrected Section 3.8 merged panel",
    )

    baseline["quote_date"] = clean_date_series(baseline["quote_date"])
    baseline["grid_time"] = baseline["grid_time"].apply(normalize_time_string)

    invalid_keys = baseline[
        baseline["quote_date"].isna()
        | baseline["grid_time"].eq("")
    ]

    if not invalid_keys.empty:
        raise ValueError(
            "Baseline panel contains invalid quote_date/grid_time keys.\n"
            f"First examples:\n{invalid_keys[KEY_COLUMNS].head(50).to_string(index=False)}"
        )

    check_unique_keys(baseline, "Baseline corrected Section 3.8 merged panel")

    return baseline.reset_index(drop=True)


# ---------------------------------------------------------------------
# MFIV DI UNA VARIANTE
# ---------------------------------------------------------------------
def load_variant_mfiv(path, variant: str) -> pd.DataFrame:
    require_file(path, f"{variant} MFIV source")

    raw = pd.read_csv(path, dtype=str, keep_default_na=False)

    date_col = find_first_existing_column(
        raw,
        ["quote_date", "date", "Date"],
        "quote-date column",
    )
    time_col = find_first_existing_column(
        raw,
        ["quote_time", "grid_time", "time", "Time"],
        "quote-time column",
    )
    mfiv_col = find_first_existing_column(
        raw,
        ["mfiv", "MFIV"],
        "MFIV value column",
    )

    # tau serve a ricalcolare MFIV_ann: un file senza tau fallisce subito invece di
    # produrre una colonna derivata vuota.
    tau_col = find_first_existing_column(raw, ["tau"], "tau column")

    out = pd.DataFrame({
        "quote_date": clean_date_series(raw[date_col]),
        "grid_time": raw[time_col].apply(normalize_time_string),
        "MFIV_variant": to_numeric_series(raw[mfiv_col]),
        "tau_variant": to_numeric_series(raw[tau_col]),
    })

    optional_columns = {
        "mfiv_flag": ["mfiv_flag"],
        "mfiv_variant_label": ["mfiv_variant"],
        "log_moneyness_cutoff_applied": ["log_moneyness_cutoff_applied"],
    }

    for output_col, candidates in optional_columns.items():
        try:
            source_col = find_first_existing_column(raw, candidates, output_col)
        except ValueError:
            continue
        out[output_col] = raw[source_col].astype(str).str.strip()

    out = out[
        out["quote_date"].notna()
        & out["grid_time"].ne("")
    ].copy()

    check_unique_keys(out, f"{variant} MFIV source")

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# PANNELLO DI UNA VARIANTE
# ---------------------------------------------------------------------
def build_variant_panel(
    baseline: pd.DataFrame,
    variant: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_file = DATA_DIR / config["mfiv_folder"] / config["mfiv_file"]
    source = load_variant_mfiv(source_file, variant)

    baseline_keys = baseline[KEY_COLUMNS].copy()
    baseline_key_index = pd.MultiIndex.from_frame(baseline_keys)
    source_key_index = pd.MultiIndex.from_frame(source[KEY_COLUMNS])

    source_on_grid = source[
        source_key_index.isin(baseline_key_index)
    ].copy()
    check_unique_keys(source_on_grid, f"{variant} MFIV source on baseline grid")

    source_for_merge_columns = KEY_COLUMNS + ["MFIV_variant", "tau_variant"]
    for optional_col in [
        "mfiv_flag",
        "mfiv_variant_label",
        "log_moneyness_cutoff_applied",
    ]:
        if optional_col in source_on_grid.columns:
            source_for_merge_columns.append(optional_col)

    replacement = baseline_keys.merge(
        source_on_grid[source_for_merge_columns],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    baseline_mfiv = to_numeric_series(baseline[MFIV_OUTPUT_COLUMN])
    variant_mfiv = to_numeric_series(replacement["MFIV_variant"])

    panel = baseline.copy()
    panel[MFIV_OUTPUT_COLUMN] = variant_mfiv

    # Colonne derivate dalla MFIV, ricalcolate dalla variante (vedi MFIV_DERIVED_COLUMNS).
    variant_tau = to_numeric_series(replacement["tau_variant"])

    if "MFIV_ann" in panel.columns:
        baseline_ann = to_numeric_series(baseline["MFIV_ann"])

        panel["MFIV_ann"] = np.where(
            (variant_tau > 0) & variant_mfiv.notna(),
            variant_mfiv / variant_tau,
            np.nan,
        )

        # tau del pannello e della variante devono coincidere: la regola di troncamento
        # non tocca il tempo residuo, e il confronto deve essere a parita' di istante.
        if "tau" in baseline.columns:
            tau_panel = to_numeric_series(baseline["tau"])
            tau_gap = (tau_panel - variant_tau).abs()
            n_tau_bad = int((tau_gap > 1e-15).sum())

            if n_tau_bad > 0:
                raise ValueError(
                    f"{variant}: tau differisce fra pannello e variante su "
                    f"{n_tau_bad} righe, scarto massimo {float(tau_gap.max()):.3e}. "
                    "Il confronto non sarebbe a parita' di istante."
                )
    else:
        baseline_ann = None

    comparison = baseline_keys.copy()
    comparison["variant"] = variant
    comparison["MFIV_baseline"] = baseline_mfiv
    comparison["MFIV_variant"] = variant_mfiv
    comparison["difference_variant_minus_baseline"] = (
        variant_mfiv - baseline_mfiv
    )
    comparison["absolute_difference"] = (
        comparison["difference_variant_minus_baseline"].abs()
    )

    comparison["percent_difference_vs_baseline"] = np.nan
    nonzero_baseline = baseline_mfiv.notna() & (baseline_mfiv != 0)
    comparison.loc[
        nonzero_baseline,
        "percent_difference_vs_baseline",
    ] = (
        comparison.loc[nonzero_baseline, "difference_variant_minus_baseline"]
        / baseline_mfiv.loc[nonzero_baseline]
        * 100.0
    )

    comparison["baseline_mfiv_finite"] = np.isfinite(baseline_mfiv).astype(int)
    comparison["variant_mfiv_finite"] = np.isfinite(variant_mfiv).astype(int)

    for optional_col in [
        "mfiv_flag",
        "mfiv_variant_label",
        "log_moneyness_cutoff_applied",
    ]:
        if optional_col in replacement.columns:
            comparison[optional_col] = replacement[optional_col]

    validation_rows: list[dict] = []

    add_validation_row(
        validation_rows,
        variant,
        "input",
        "baseline_panel_file_exists",
        1,
        int(BASELINE_PANEL_FILE.exists()),
        BASELINE_PANEL_FILE.exists(),
        str(BASELINE_PANEL_FILE),
    )
    add_validation_row(
        validation_rows,
        variant,
        "input",
        "variant_mfiv_file_exists",
        1,
        int(source_file.exists()),
        source_file.exists(),
        str(source_file),
    )
    add_validation_row(
        validation_rows,
        variant,
        "scope",
        "output_row_count_matches_baseline",
        len(baseline),
        len(panel),
        len(panel) == len(baseline),
    )
    add_validation_row(
        validation_rows,
        variant,
        "scope",
        "output_row_count_is_expected",
        EXPECTED_ROWS,
        len(panel),
        len(panel) == EXPECTED_ROWS,
    )
    add_validation_row(
        validation_rows,
        variant,
        "scope",
        "output_has_247_dates",
        EXPECTED_DAYS_LOGM,
        panel["quote_date"].nunique(),
        panel["quote_date"].nunique() == EXPECTED_DAYS_LOGM,
    )
    add_validation_row(
        validation_rows,
        variant,
        "structure",
        "output_has_no_duplicate_keys",
        0,
        int(panel.duplicated(KEY_COLUMNS).sum()),
        int(panel.duplicated(KEY_COLUMNS).sum()) == 0,
    )
    add_validation_row(
        validation_rows,
        variant,
        "structure",
        "output_column_order_matches_baseline",
        1,
        int(list(panel.columns) == list(baseline.columns)),
        list(panel.columns) == list(baseline.columns),
    )

    # Escluse anche le colonne derivate, che la sostituzione ricalcola apposta.
    derived_present = [c for c in MFIV_DERIVED_COLUMNS if c in baseline.columns]

    untouched_columns = [
        column for column in baseline.columns
        if column != MFIV_OUTPUT_COLUMN and column not in derived_present
    ]
    untouched_identical = panel[untouched_columns].equals(
        baseline[untouched_columns]
    )

    add_validation_row(
        validation_rows,
        variant,
        "identity",
        "all_columns_outside_mfiv_block_identical_to_baseline",
        1,
        int(untouched_identical),
        untouched_identical,
        "Include FwdRV_h, flag di validita', TrailingRV, entrambe le fonti di "
        "gamma e le chiavi di effetto fisso.",
    )

    # Le derivate devono invece cambiare: se coincidessero riga per riga con la
    # baseline, la sostituzione non sarebbe avvenuta.
    for col in derived_present:
        v = to_numeric_series(panel[col])
        b = to_numeric_series(baseline[col])

        n_identical = int((v.round(15) == b.round(15)).sum())

        add_validation_row(
            validation_rows,
            variant,
            "identity",
            f"{col}_differs_from_baseline",
            "< " + str(len(panel)),
            n_identical,
            n_identical < len(panel),
            "Se la colonna derivata coincidesse riga per riga con la baseline, "
            "il pannello mescolerebbe due regole di troncamento e le stime intraday "
            "userebbero la baseline credendo di usare la variante.",
        )

    baseline_finite = np.isfinite(baseline_mfiv)
    variant_finite = np.isfinite(variant_mfiv)
    missing_pattern_mismatches = int((baseline_finite != variant_finite).sum())

    add_validation_row(
        validation_rows,
        variant,
        "MFIV_missingness",
        "MFIV_finite_missing_pattern_matches_baseline",
        0,
        missing_pattern_mismatches,
        missing_pattern_mismatches == 0,
    )

    rows_0930 = panel[panel["grid_time"] == FIRST_GRID_TIME]
    rows_after_0930 = panel[panel["grid_time"] != FIRST_GRID_TIME]

    add_validation_row(
        validation_rows,
        variant,
        "MFIV_missingness",
        "one_first_point_row_per_day",
        EXPECTED_DAYS_LOGM,
        len(rows_0930),
        len(rows_0930) == EXPECTED_DAYS_LOGM,
        f"Primo punto di griglia: {FIRST_GRID_TIME}.",
    )
    add_validation_row(
        validation_rows,
        variant,
        "MFIV_missingness",
        "MFIV_present_at_first_point_all_days",
        EXPECTED_DAYS_LOGM,
        int(to_numeric_series(rows_0930[MFIV_OUTPUT_COLUMN]).isna().sum()),
        int(to_numeric_series(rows_0930[MFIV_OUTPUT_COLUMN]).isna().sum()) == 0,
        f"Alle {FIRST_GRID_TIME}, primo punto della griglia, la MFIV deve essere "
        f"presente in ogni giornata.",
    )
    add_validation_row(
        validation_rows,
        variant,
        "MFIV_missingness",
        "MFIV_never_missing_after_first_point",
        0,
        int(to_numeric_series(rows_after_0930[MFIV_OUTPUT_COLUMN]).isna().sum()),
        int(to_numeric_series(rows_after_0930[MFIV_OUTPUT_COLUMN]).isna().sum()) == 0,
    )

    source_rows_on_grid = len(source_on_grid)
    expected_source_rows_on_grid = int(baseline_finite.sum())

    add_validation_row(
        validation_rows,
        variant,
        "source_grid",
        "variant_source_has_one_row_for_every_finite_baseline_MFIV_grid_key",
        expected_source_rows_on_grid,
        source_rows_on_grid,
        source_rows_on_grid == expected_source_rows_on_grid,
        (
            f"Source has {len(source):,} total intraday rows; "
            f"{len(source) - source_rows_on_grid:,} non-grid rows are intentionally ignored."
        ),
    )

    both_finite = baseline_finite & variant_finite
    n_changed = int(
        (
            (variant_mfiv[both_finite] - baseline_mfiv[both_finite]).abs()
            > VALUE_MATCH_TOL
        ).sum()
    )

    add_validation_row(
        validation_rows,
        variant,
        "MFIV_replacement",
        "variant_MFIV_differs_from_baseline_on_at_least_one_grid_row",
        ">0",
        n_changed,
        n_changed > 0,
    )

    cutoff_column_present = "log_moneyness_cutoff_applied" in source_on_grid.columns
    add_validation_row(
        validation_rows,
        variant,
        "tracking",
        "log_moneyness_cutoff_tracking_column_present",
        1,
        int(cutoff_column_present),
        cutoff_column_present,
    )

    if cutoff_column_present:
        cutoff_values = to_numeric_series(
            source_on_grid["log_moneyness_cutoff_applied"]
        ).dropna()
        cutoff_bad = int(
            (~np.isclose(
                cutoff_values,
                float(config["expected_cutoff"]),
                rtol=0.0,
                atol=1e-12,
            )).sum()
        )

        add_validation_row(
            validation_rows,
            variant,
            "tracking",
            "tracking_cutoff_matches_expected_variant",
            config["expected_cutoff"],
            ";".join(sorted(cutoff_values.astype(str).unique().tolist())),
            cutoff_bad == 0 and len(cutoff_values) > 0,
        )

    if "mfiv_variant_label" in source_on_grid.columns:
        labels = sorted(
            source_on_grid["mfiv_variant_label"]
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        add_validation_row(
            validation_rows,
            variant,
            "tracking",
            "source_variant_labels_reported",
            "reported",
            ";".join(labels),
            True,
        )

    if "mfiv_flag" in source_on_grid.columns:
        flag_counts = (
            source_on_grid["mfiv_flag"]
            .astype(str)
            .str.strip()
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        add_validation_row(
            validation_rows,
            variant,
            "tracking",
            "MFIV_flag_counts_on_estimation_grid_reported_not_filtered",
            "reported",
            str(flag_counts),
            True,
            (
                "Numeric MFIV values are imported without filtering mfiv_flag, "
                "as in the baseline intraday panel."
            ),
        )

    validation = pd.DataFrame(validation_rows)

    return panel, comparison, validation


def intraday_panel_logm():
    print("=" * 150)
    print("BUILD SECTION 3.8 MERGED INTRADAY PANELS - MFIV LOG-MONEYNESS ROBUSTNESS")
    print("=" * 150)
    print(f"Data directory:        {DATA_DIR}")
    print(f"Baseline merged panel: {BASELINE_PANEL_FILE}")
    print(f"Output directory:      {OUTPUT_DIR}")
    print("Variants:              logm_005 and logm_002")
    print("=" * 150)

    baseline_panel = load_baseline_panel()

    all_validations = []
    all_comparisons = []

    for variant_label, variant_config in VARIANTS.items():
        print()
        print("-" * 150)
        print(f"Processing {variant_label}")
        print("-" * 150)

        variant_panel, comparison, validation = build_variant_panel(
            baseline=baseline_panel,
            variant=variant_label,
            config=variant_config,
        )

        output_panel_file = (
            OUTPUT_DIR
            / (
                f"{OUTPUT_FILE_PREFIX}merged_intraday_panel_2025_"
                f"primary_247_{variant_label}.csv"
            )
        )
        output_comparison_file = (
            OUTPUT_DIR
            / f"{OUTPUT_FILE_PREFIX}mfiv_intraday_comparison_vs_baseline_{variant_label}.csv"
        )
        output_validation_file = (
            OUTPUT_DIR
            / (
                f"{OUTPUT_FILE_PREFIX}merged_intraday_panel_2025_"
                f"primary_247_{variant_label}_validation.csv"
            )
        )

        safe_write_csv(variant_panel, output_panel_file)
        safe_write_csv(comparison, output_comparison_file)
        safe_write_csv(validation, output_validation_file)

        all_validations.append(validation)
        all_comparisons.append(comparison)

        failed = validation[validation["passed"] != 1].copy()

        print(f"MFIV source:       {DATA_DIR / variant_config['mfiv_folder'] / variant_config['mfiv_file']}")
        print(f"Panel output:      {output_panel_file}")
        print(f"Comparison output: {output_comparison_file}")
        print(f"Validation output: {output_validation_file}")
        print(f"Rows:              {len(variant_panel):,}")
        print(f"Dates:             {variant_panel['quote_date'].nunique():,}")
        print(f"Failed checks:     {len(failed):,}")

        if not failed.empty:
            print()
            print("FAILED VALIDATION CHECKS")
            print(failed.to_string(index=False))

    combined_validation = pd.concat(all_validations, ignore_index=True)
    combined_comparison = pd.concat(all_comparisons, ignore_index=True)

    safe_write_csv(combined_validation, COMBINED_VALIDATION_FILE)
    safe_write_csv(combined_comparison, COMBINED_COMPARISON_FILE)

    failed_all = combined_validation[
        combined_validation["passed"] != 1
    ].copy()

    print()
    print("=" * 150)
    print("COMPLETED SECTION 3.8 MFIV LOG-MONEYNESS PANEL BUILD")
    print("=" * 150)
    print(f"Combined validation: {COMBINED_VALIDATION_FILE}")
    print(f"Combined comparison: {COMBINED_COMPARISON_FILE}")
    print(f"Total failed checks: {len(failed_all):,}")

    if failed_all.empty:
        print("All validation checks passed.")
    else:
        print()
        print(failed_all.to_string(index=False))
        print("=" * 150)
        raise SystemExit(
            "Parte 4: controlli di validazione dei pannelli intraday "
            "log-moneyness falliti (vedi sopra)."
        )

    print("=" * 150)


def main():
    forward_rv_10min()
    trailing_rv_10min()
    intraday_panel()
    intraday_panel_logm()


if __name__ == "__main__":
    main()
