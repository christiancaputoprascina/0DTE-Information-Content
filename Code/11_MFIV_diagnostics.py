"""
Diagnostica dello strip della MFIV (Capitolo 3 e Appendice B.1).
  Parte 1: strike integrati per punto di griglia e per terzile, baseline contro |k| <= 0.05.
  Parte 2: Tabella 3.1, regola di validita' alle 15:55, tau, Appendice B.1, bande di
           log-moneyness, pannello intraday e controfattuale "zero ask", confrontati con la tesi.
Input:  CBOE/Data/03_mfiv/, 07_daily_panel/, 08_intraday_panel/, 09_appendix_a_checks/,
        02_black76_iv_greeks/ (solo le giornate con quote OTM ad ask nullo)
Output: CBOE/Data/11_mfiv_diagnostics/
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (DATA_DIR, EARLY_CLOSE_DATES, YEAR, output_dir, path, require,
                    safe_write_csv, to_numeric_series)

OUTPUT_DIR = output_dir("11_mfiv_diagnostics")


# =====================================================================
# PARTE 1: QUANTI STRIKE INTEGRA LA MFIV, ORA PER ORA
# =====================================================================
# Vicino alla scadenza il troncamento sui bid nulli lascia pochi strike nello strip. Si conta,
# per punto di griglia e per terzile, quanti strike integra la baseline rispetto alla variante
# |k| <= 0.05, per vedere se il calo riguarda tutto il terzile di chiusura o solo gli ultimi minuti.

BASELINE_FILE = DATA_DIR / "03_mfiv" / "03_mfiv_intraday_2025_all.csv"
VARIANT_FILE = DATA_DIR / "03_mfiv" / "03_mfiv_intraday_2025_all_logm_005.csv"

OUTPUT_STEM = "11_mfiv_strike_count_by_time"

# Terzili della giornata sulla griglia a dieci minuti, da 09:40 a 15:50.
TERCILE_BOUNDS = [
    ("apertura", "09:40", "11:30"),
    ("meta_giornata", "11:40", "13:40"),
    ("chiusura", "13:50", "15:50"),
]

# Solo i punti della griglia a dieci minuti entrano nelle regressioni intraday.
GRID_START = "09:40"
GRID_END = "15:50"
GRID_STEP_MINUTES = 10


def hhmm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str[:5]


def load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} non trovato:\n{path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    df["quote_date"] = pd.to_datetime(
        df["quote_date"].astype(str).str[:10], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    df["_hhmm"] = hhmm(df["quote_time"])

    for col in ("n_otm_strikes", "n_otm_put_strikes", "n_otm_call_strikes",
                "tau", "mfiv"):
        if col in df.columns:
            df[col] = to_numeric_series(df[col])

    df = df[df["quote_date"].astype(str).str.startswith(str(YEAR))].copy()

    return df


def grid_times() -> list[str]:
    start = pd.Timestamp(f"2000-01-01 {GRID_START}:00")
    end = pd.Timestamp(f"2000-01-01 {GRID_END}:00")

    out = []
    cur = start

    while cur <= end:
        out.append(cur.strftime("%H:%M"))
        cur += pd.Timedelta(minutes=GRID_STEP_MINUTES)

    return out


def tercile_of(hhmm_value: str) -> str:
    for name, lo, hi in TERCILE_BOUNDS:
        if lo <= hhmm_value <= hi:
            return name

    return "fuori_griglia"


def strike_count_by_time():
    print("=" * 140)
    print("PARTE 1: CONTEGGIO DEGLI STRIKE DELLA MFIV, ORA PER ORA")
    print("=" * 140)

    baseline = load(BASELINE_FILE, "Baseline MFIV")
    variant = load(VARIANT_FILE, "Variante logm_005")

    print(f"Baseline: {len(baseline):,} righe, "
          f"{baseline['quote_date'].nunique()} giornate")
    print(f"Variante: {len(variant):,} righe")

    gt = grid_times()

    print(f"Punti di griglia considerati: {len(gt)} da {gt[0]} a {gt[-1]}")

    # --- SOLO RIGHE OK E SOLO PUNTI DI GRIGLIA
    b_ok = baseline[baseline["mfiv_flag"] == "ok"].copy()
    v_ok = variant[variant["mfiv_flag"] == "ok"].copy()

    b_grid = b_ok[b_ok["_hhmm"].isin(gt)].copy()
    v_grid = v_ok[v_ok["_hhmm"].isin(gt)].copy()

    print()
    print(f"Righe ok sulla griglia, baseline: {len(b_grid):,}")
    print(f"Righe ok sulla griglia, variante: {len(v_grid):,}")

    # --- PROFILO PER PUNTO DI GRIGLIA
    print()
    print("=" * 140)
    print("PROFILO PER PUNTO DI GRIGLIA")
    print("=" * 140)

    rows = []

    for tt in gt:
        b = b_grid[b_grid["_hhmm"] == tt]
        v = v_grid[v_grid["_hhmm"] == tt]

        bs = b["n_otm_strikes"].dropna()
        vs = v["n_otm_strikes"].dropna()

        rows.append({
            "grid_time": tt,
            "tercile": tercile_of(tt),
            "n_days_baseline_ok": int(len(b)),
            "n_days_variant_ok": int(len(v)),
            "strikes_baseline_p10": float(bs.quantile(0.10)) if len(bs) else np.nan,
            "strikes_baseline_median": float(bs.median()) if len(bs) else np.nan,
            "strikes_baseline_p90": float(bs.quantile(0.90)) if len(bs) else np.nan,
            "strikes_baseline_min": float(bs.min()) if len(bs) else np.nan,
            "strikes_variant_median": float(vs.median()) if len(vs) else np.nan,
            "ratio_baseline_over_variant": (float(bs.median() / vs.median())
                                            if len(bs) and len(vs)
                                            and vs.median() > 0 else np.nan),
            "tau_median": float(b["tau"].median()) if len(b) else np.nan,
        })

    profile = pd.DataFrame(rows)
    safe_write_csv(profile, OUTPUT_DIR / f"{OUTPUT_STEM}_by_grid_point.csv")

    print(profile[["grid_time", "tercile", "n_days_baseline_ok",
                   "strikes_baseline_min", "strikes_baseline_p10",
                   "strikes_baseline_median", "strikes_baseline_p90",
                   "strikes_variant_median",
                   "ratio_baseline_over_variant"]].to_string(index=False))

    # --- RIEPILOGO PER TERZILE
    print()
    print("=" * 140)
    print("RIEPILOGO PER TERZILE")
    print("=" * 140)

    b_grid["tercile"] = b_grid["_hhmm"].apply(tercile_of)
    v_grid["tercile"] = v_grid["_hhmm"].apply(tercile_of)

    terc_rows = []

    for name, lo, hi in TERCILE_BOUNDS:
        b = b_grid[b_grid["tercile"] == name]
        v = v_grid[v_grid["tercile"] == name]

        bs = b["n_otm_strikes"].dropna()

        terc_rows.append({
            "tercile": name,
            "intervallo": f"{lo}-{hi}",
            "n_righe": int(len(b)),
            "strikes_min": float(bs.min()) if len(bs) else np.nan,
            "strikes_p10": float(bs.quantile(0.10)) if len(bs) else np.nan,
            "strikes_median": float(bs.median()) if len(bs) else np.nan,
            "strikes_p90": float(bs.quantile(0.90)) if len(bs) else np.nan,
            "pct_righe_sotto_20_strike": float((bs < 20).mean() * 100) if len(bs) else np.nan,
            "pct_righe_sotto_50_strike": float((bs < 50).mean() * 100) if len(bs) else np.nan,
            "strikes_variant_median": float(v["n_otm_strikes"].median()) if len(v) else np.nan,
        })

    terciles = pd.DataFrame(terc_rows)
    safe_write_csv(terciles, OUTPUT_DIR / f"{OUTPUT_STEM}_by_tercile.csv")

    print(terciles.to_string(index=False))

    # --- DOVE STA IL CROLLO
    print()
    print("=" * 140)
    print("DOVE STA IL CROLLO")
    print("=" * 140)

    close_tercile = profile[profile["tercile"] == "chiusura"].copy()

    print("Terzile di chiusura, punto per punto:")
    print(close_tercile[["grid_time", "strikes_baseline_median",
                         "strikes_baseline_p10", "strikes_variant_median",
                         "ratio_baseline_over_variant"]].to_string(index=False))

    med = close_tercile["strikes_baseline_median"]

    if len(med) and med.notna().any():
        first = float(med.iloc[0])
        last = float(med.iloc[-1])

        print()
        print(f"  primo punto del terzile ({close_tercile['grid_time'].iloc[0]}): "
              f"mediana {first:.0f} strike")
        print(f"  ultimo punto ({close_tercile['grid_time'].iloc[-1]}): "
              f"mediana {last:.0f} strike")

        half = close_tercile[close_tercile["strikes_baseline_median"] < first / 2]

        if not half.empty:
            print(f"  la mediana scende sotto META' del primo punto a partire "
                  f"dalle {half['grid_time'].iloc[0]}")
        else:
            print("  la mediana NON scende mai sotto meta' del primo punto")

    print()
    print("=" * 140)
    print("COME SI LEGGE")
    print("=" * 140)
    print("Il terzile di chiusura va dalle 13:50 alle 15:50 ed e' quello dove HI3")
    print("trova l'R-quadro incrementale piu' alto, 0,1158 contro 0,00025")
    print("all'apertura.")
    print()
    print("SE il conteggio degli strike resta alto per quasi tutto il terzile e")
    print("crolla solo negli ultimi due o tre punti, allora l'R-quadro incrementale")
    print("poggia in gran parte su superfici adeguate e il problema e' circoscritto")
    print("a pochi punti su tredici.")
    print()
    print("SE invece il conteggio cala progressivamente da inizio terzile, la")
    print("superficie usata per il risultato piu' forte della tesi e' sistematicamente")
    print("piu' povera di quella usata altrove, e HI3 va riesaminato costruendo la")
    print("MFIV con la regola a moneyness.")
    print()
    print("La colonna ratio_baseline_over_variant dice quanta parte della superficie")
    print("disponibile dentro il cinque per cento la regola sui bid sta tenendo.")

    print()
    print(f"Output: {OUTPUT_STEM}_by_grid_point.csv, {OUTPUT_STEM}_by_tercile.csv")
    print("=" * 140)


# =====================================================================
# PARTE 2: DIAGNOSTICA DELLO STRIP, TABELLA 3.1, APPENDICE B.1
# =====================================================================
# Statistiche fra sessioni sugli output di 03_MFIV.py, confrontate con i valori della tesi:
# OK se il valore arrotondato alla precisione stampata coincide. Un DIVERSO non ferma lo
# script; lo fermano solo i problemi di integrita' dei dati.
#
# Tabella 3.1: 247 sessioni regolari, percentili per rango piu' vicino. La mediana nella sessione
# del numero di strike conta come strip vuoti i minuti senza strip (il valore senza di essi e'
# scritto accanto); gli strike OTM quotati comprendono quelli con bid nullo.
#
# Controfattuale "zero ask": sui minuti con quote OTM ad ask nullo si rifa' la camminata di
# 03_MFIV.py, prima identica (verifica di riproduzione) e poi trattando gli ask nulli come
# condizione di arresto, come la regola Cboe descritta nella tesi.

MFIV_INTRADAY = ("03_mfiv", "03_mfiv_intraday_2025_all.csv")
MFIV_DIAGNOSTICS = ("03_mfiv", "03_mfiv_diagnostics_2025_all.csv")
MFIV_TRUNCATION = ("03_mfiv", "03_mfiv_daily_truncation_2025_appendix_D.csv")
LOGM_INTRADAY = {
    "0.05": ("03_mfiv", "03_mfiv_intraday_2025_all_logm_005.csv"),
    "0.02": ("03_mfiv", "03_mfiv_intraday_2025_all_logm_002.csv"),
}
DAILY_PANEL = ("07_daily_panel", "07_daily_regression_panel_2025.csv")
INTRADAY_PANEL = ("08_intraday_panel", "08_merged_intraday_panel_2025_primary_247.csv")
ZERO_ASK_LIST = ("09_appendix_a_checks", "09_zero_ask_quotes_2025.csv")
BLACK76_FOLDER = "02_black76_iv_greeks"
BLACK76_PATTERN = "02_0DTE_{y}_{m}_{d}_black76_365.csv"

OUT_TABLE_3_1 = "11_table_3_1_2025.csv"
OUT_TEXT_NUMBERS = "11_text_numbers_2025.csv"
OUT_TAU_BY_DAY = "11_tau_ratio_by_day_2025.csv"
OUT_REPRO = "11_zero_ask_walk_reproduction_check_2025.csv"
OUT_COUNTERFACTUAL = "11_zero_ask_counterfactual_2025.csv"
OUT_THESIS = "11_thesis_check.csv"

# Costanti di 03_MFIV.py, copiate e non importate perche' lo script resti autonomo.
YEAR_BASIS = 365
MINUTES_PER_YEAR = YEAR_BASIS * 24 * 60          # 525600
NEAR_EXPIRY_MINUTES = 5
NEAR_EXPIRY_TAU = NEAR_EXPIRY_MINUTES * 60 / (YEAR_BASIS * 24.0 * 60.0 * 60.0)
MIN_OTM_EACH_SIDE = 5
COVERAGE_K_SD = 2.0
ATM_IV_CEILING = 5.0

EARLY_CLOSES = set(EARLY_CLOSE_DATES)
REGULAR_SESSIONS_EXPECTED = 247
ALL_SESSIONS_EXPECTED = 250

CHECK_TIME = "15:55:00"
SURFACE_START_TIME = "09:31:00"
REGULAR_EXPIRY_MINUTE = 16 * 60
EARLY_EXPIRY_MINUTE = 13 * 60
PERCENTILES = (50, 5, 95)

# Pannello intraday: terzile di chiusura e orizzonti (Tabella 5.5).
CLOSING_TERCILE = ("13:50:00", "15:50:00")
HORIZONS = [5, 15, 30, 60]

# Colonne dei file per contratto dello stadio 02 lette per la camminata (solo
# giornate interessate).
BLACK76_COLUMNS = [
    "quote_date", "quote_time", "strike", "option_type", "bid", "ask",
    "black76_bid", "black76_mid", "black76_F_365", "black76_T_365", "black76_r",
    "black76_implied_volatility_365", "black76_solver_success_365",
    "black76_iv_at_bound_365", "black76_near_expiry_flag_365",
]

# Nomi possibili delle colonne nella lista degli ask nulli (09_Appendix_A_checks.py).
ZERO_ASK_COLUMN_CANDIDATES = {
    "date": ["quote_date", "date"],
    "time": ["quote_time", "time"],
    "strike": ["strike"],
    "type": ["option_type", "type", "cp", "call_put"],
    "bid": ["bid"],
    "ask": ["ask"],
    "otm": ["is_otm", "otm", "otm_flag", "black76_is_otm_365", "OTM"],
    "F": ["F", "forward", "black76_F_365", "F_365"],
}

REL_TOL_MFIV = 1e-10


class IntegrityError(RuntimeError):
    pass


def integrity(cond: bool, message: str):
    if not cond:
        raise IntegrityError(message)


# --- funzioni statistiche

def nearest_rank(values, p: float) -> float:
    """Percentile per rango piu' vicino: valore di rango ceil(p/100 * N)."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    k = max(1, math.ceil(p / 100.0 * len(x)))
    return float(x[k - 1])


def time_to_minute(s: str) -> int:
    h, m, *_ = str(s).split(":")
    return int(h) * 60 + int(m)


def minute_to_hhmm(m: float) -> str:
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


# --- funzioni della MFIV (copiate da 03_MFIV.py, stessa logica)

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


def truncate_side(strip: pd.DataFrame, side: str) -> pd.DataFrame:
    """Camminata dello stadio 03 (cboe_truncate_side) con una colonna _zero esplicita.

    Dallo strike piu' vicino a F verso l'esterno: uno strike con _zero vero e'
    escluso; due _zero consecutivi fermano la camminata. Nello stadio 03 _zero equivale a
    bid non finito o <= 0; nella variante Cboe vale anche per gli strike con
    ask nullo aggiunti alla camminata.
    """
    if strip.empty:
        return strip.iloc[0:0]
    ordered = strip.sort_values("_strike", ascending=(side == "C")).reset_index(drop=True)
    zero = ordered["_zero"].to_numpy(dtype=bool)
    keep = []
    consecutive_zeros = 0
    for i in range(len(ordered)):
        if zero[i]:
            consecutive_zeros += 1
            if consecutive_zeros >= 2:
                break
        else:
            consecutive_zeros = 0
            keep.append(i)
    return ordered.loc[keep].sort_values("_strike").reset_index(drop=True)


def atm_sigma(valid: pd.DataFrame, F: float) -> float:
    a = valid[valid["_iv"].notna() & (valid["_iv"] > 0) & (valid["_iv"] < ATM_IV_CEILING)]
    if a.empty:
        return np.nan
    a = a.assign(_dist=(a["_strike"] - F).abs())
    return float(a.nsmallest(4, "_dist")["_iv"].mean())


def safe_median(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return np.nan
    return float(values.median())


def compute_strip_mfiv(strip: pd.DataFrame, r: float, tau: float):
    if strip.empty:
        return np.nan
    clean = strip[["_strike", "_mid"]].copy()
    clean = clean[np.isfinite(clean["_strike"]) & np.isfinite(clean["_mid"])
                  & (clean["_strike"] > 0) & (clean["_mid"] > 0)]
    clean = (clean.groupby("_strike", as_index=False).agg({"_mid": "median"})
             .sort_values("_strike").reset_index(drop=True))
    if len(clean) < 2:
        return np.nan
    strikes = clean["_strike"].to_numpy(dtype=float)
    prices = clean["_mid"].to_numpy(dtype=float)
    delta_k = compute_delta_k(strikes)
    ok = np.isfinite(delta_k) & (delta_k > 0)
    if not np.any(ok):
        return np.nan
    mfiv = 2.0 * np.exp(r * tau) * np.sum(prices[ok] / strikes[ok] ** 2 * delta_k[ok])
    if not np.isfinite(mfiv) or mfiv < 0:
        return np.nan
    return float(mfiv)


def prepare_black76_day(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "quote_date": df["quote_date"].astype(str).str.strip(),
        "quote_time": df["quote_time"].astype(str).str.strip(),
        "_strike": pd.to_numeric(df["strike"], errors="coerce"),
        "_option_type": df["option_type"].astype(str).str.strip().str.upper().str[0],
        "_mid": pd.to_numeric(df["black76_mid"], errors="coerce"),
        "_bid": pd.to_numeric(df["black76_bid"], errors="coerce"),
        "_raw_bid": pd.to_numeric(df["bid"], errors="coerce"),
        "_raw_ask": pd.to_numeric(df["ask"], errors="coerce"),
        "_F": pd.to_numeric(df["black76_F_365"], errors="coerce"),
        "_tau": pd.to_numeric(df["black76_T_365"], errors="coerce"),
        "_r": pd.to_numeric(df["black76_r"], errors="coerce"),
        "_iv": pd.to_numeric(df["black76_implied_volatility_365"], errors="coerce"),
        "_solver_success": pd.to_numeric(df["black76_solver_success_365"], errors="coerce").fillna(0).astype(int),
        "_iv_at_bound": pd.to_numeric(df["black76_iv_at_bound_365"], errors="coerce").fillna(0).astype(int),
        "_near_expiry": pd.to_numeric(df["black76_near_expiry_flag_365"], errors="coerce").fillna(0).astype(int),
    })
    return out


def strip_for_minute(group: pd.DataFrame, extra_zero: dict) -> dict:
    """Ricostruisce lo strip dello stadio 03 per un minuto.

    extra_zero: {"P": set di strike, "C": set di strike} trattati come
    condizione di arresto (variante Cboe); dizionario vuoto = regola dello stadio 03.
    """
    F = safe_median(group["_F"])
    tau = safe_median(group["_tau"])
    r = safe_median(group["_r"])
    res = {"F": F, "tau": tau, "n_put": 0, "n_call": 0, "n_strikes": 0,
           "mfiv": np.nan, "flag": "", "n_overlap_zero_ask_valid": 0}
    if not np.isfinite(F) or F <= 0 or not np.isfinite(r):
        res["flag"] = "missing_F"
        return res
    if not np.isfinite(tau) or tau <= 0 or tau < NEAR_EXPIRY_TAU:
        res["flag"] = "near_expiry"
        return res

    valid = group[
        (group["_solver_success"] == 1) & (group["_near_expiry"] == 0)
        & np.isfinite(group["_strike"]) & np.isfinite(group["_mid"])
        & np.isfinite(group["_F"]) & np.isfinite(group["_tau"]) & np.isfinite(group["_r"])
        & (group["_strike"] > 0) & (group["_mid"] > 0) & (group["_F"] > 0) & (group["_tau"] > 0)
        & group["_option_type"].isin(["C", "P"])
    ].copy()
    if valid.empty:
        res["flag"] = "no_valid_options"
        return res

    sigma = atm_sigma(valid, F)
    collapsed = (valid[["_strike", "_option_type", "_mid", "_bid", "_iv_at_bound"]]
                 .groupby(["_strike", "_option_type"], as_index=False)
                 .agg({"_mid": "median", "_bid": "median", "_iv_at_bound": "max"}))

    sides = {}
    for side in ("P", "C"):
        raw = collapsed[(collapsed["_option_type"] == side)
                        & ((collapsed["_strike"] < F) if side == "P" else (collapsed["_strike"] > F))
                        ][["_strike", "_mid", "_bid", "_iv_at_bound"]].copy()
        raw["_zero"] = ~(np.isfinite(raw["_bid"]) & (raw["_bid"] > 0))
        extra = set(extra_zero.get(side, set()))
        overlap = extra & set(raw["_strike"])
        res["n_overlap_zero_ask_valid"] += len(overlap)
        add = sorted(extra - set(raw["_strike"]))
        if add:
            raw = pd.concat([raw, pd.DataFrame({
                "_strike": add, "_mid": np.nan, "_bid": 0.0, "_iv_at_bound": 0, "_zero": True})],
                ignore_index=True)
        kept = truncate_side(raw, side)
        kept = kept[kept["_iv_at_bound"] != 1].reset_index(drop=True)
        sides[side] = kept

    puts, calls = sides["P"], sides["C"]
    parts = [d[["_strike", "_mid"]] for d in (puts, calls) if not d.empty]
    exact_atm = collapsed[np.isclose(collapsed["_strike"], F, rtol=0.0, atol=1e-8)]
    if not exact_atm.empty:
        atm_price = exact_atm["_mid"].mean()
        if np.isfinite(atm_price) and atm_price > 0:
            parts.append(pd.DataFrame({"_strike": [F], "_mid": [atm_price]}))
    if not parts:
        res["flag"] = "no_valid_options"
        return res
    full = (pd.concat(parts, ignore_index=True).groupby("_strike", as_index=False)
            .agg({"_mid": "median"}).sort_values("_strike").reset_index(drop=True))
    strikes = full["_strike"].to_numpy(dtype=float)
    res["n_put"] = int(puts["_strike"].nunique()) if not puts.empty else 0
    res["n_call"] = int(calls["_strike"].nunique()) if not calls.empty else 0
    res["n_strikes"] = int(len(strikes))
    if len(strikes) == 0:
        res["flag"] = "no_valid_options"
        return res
    sd_points = sigma * np.sqrt(tau) * F if np.isfinite(sigma) else np.nan
    if np.isfinite(sd_points) and sd_points > 0:
        coverage_fails = ((F - strikes.min()) < COVERAGE_K_SD * sd_points
                          or (strikes.max() - F) < COVERAGE_K_SD * sd_points)
    else:
        coverage_fails = False
    thin = (res["n_put"] < MIN_OTM_EACH_SIDE) or (res["n_call"] < MIN_OTM_EACH_SIDE) or coverage_fails
    res["mfiv"] = compute_strip_mfiv(full, r, tau)
    if not np.isfinite(res["mfiv"]):
        res["flag"] = "no_valid_options"
    else:
        res["flag"] = "thin_otm" if thin else "ok"
    return res


def _norm(s: str) -> str:
    return str(s).replace(",", "").replace("−", "-").replace("+", "").strip()


def mfiv_strip_diagnostics():
    # --- confronto con la tesi e numeri del testo
    thesis_rows = []

    def check(pagina, voce, valore_tesi, valore_calcolato, precisione, nota="", esito=None):
        """precisione: intero 'dN' (N decimali), 'intero', 'HH:MM' o 'testo'."""
        if esito is None:
            if precisione == "intero":
                ok = int(round(float(valore_calcolato))) == int(float(_norm(valore_tesi)))
                shown = f"{int(round(float(valore_calcolato))):,}"
            elif precisione.startswith("d"):
                nd = int(precisione[1:])
                shown = f"{float(valore_calcolato):.{nd}f}"
                ok = _norm(shown) == _norm(valore_tesi)
                if not ok and float(_norm(shown)) == 0.0 and float(_norm(valore_tesi)) == 0.0:
                    ok = True
            elif precisione == "HH:MM":
                shown = str(valore_calcolato)
                ok = shown == str(valore_tesi)
            else:
                raise ValueError("Per precisione 'testo' passare esito esplicito.")
        else:
            ok = bool(esito)
            shown = str(valore_calcolato)
        thesis_rows.append({
            "pagina": pagina, "voce": voce, "valore_tesi": valore_tesi,
            "valore_calcolato": shown, "precisione": precisione,
            "esito": "OK" if ok else "DIVERSO", "nota": nota,
        })

    text_rows = []

    def record(sezione, voce, valore, nota=""):
        text_rows.append({"sezione": sezione, "voce": voce, "valore": valore, "nota": nota})

    # --- LETTURA
    t0 = time.time()
    print("=" * 130)
    print("PARTE 2: DIAGNOSTICA DELLO STRIP MFIV: TABELLA 3.1, TESTO DEL CAPITOLO 3, APPENDICE B.1, CONTROFATTUALE ZERO ASK")
    print("=" * 130)

    for p in (path(MFIV_INTRADAY), path(MFIV_DIAGNOSTICS), path(MFIV_TRUNCATION),
              path(DAILY_PANEL), path(INTRADAY_PANEL), *[path(v) for v in LOGM_INTRADAY.values()]):
        require(p)

    zero_ask_path = path(ZERO_ASK_LIST)       # scritta da 09_Appendix_A_checks.py
    require(zero_ask_path)

    intraday = pd.read_csv(path(MFIV_INTRADAY), usecols=[
        "quote_date", "quote_time", "tau", "F", "n_otm_strikes", "n_otm_put_strikes",
        "n_otm_call_strikes", "coverage_low_sd", "coverage_high_sd", "truncation_ratio",
        "n_otm_strikes_raw", "mfiv", "mfiv_flag"])
    diag = pd.read_csv(path(MFIV_DIAGNOSTICS))
    trunc = pd.read_csv(path(MFIV_TRUNCATION))
    panel = pd.read_csv(path(DAILY_PANEL))
    ipanel = pd.read_csv(path(INTRADAY_PANEL), usecols=[
        "quote_date", "grid_time", "mfiv_flag", "h5_valid", "h15_valid", "h30_valid", "h60_valid"])

    for df in (intraday, diag, trunc, panel, ipanel):
        df["quote_date"] = df["quote_date"].astype(str).str.strip()
    intraday["quote_time"] = intraday["quote_time"].astype(str).str.strip()

    # --- integrita' ---
    integrity(intraday["quote_date"].nunique() == ALL_SESSIONS_EXPECTED,
              f"03 intraday: {intraday['quote_date'].nunique()} giornate, attese {ALL_SESSIONS_EXPECTED}")
    integrity(len(diag) == ALL_SESSIONS_EXPECTED and diag["quote_date"].is_unique,
              "03 diagnostiche: una riga per giornata attesa")
    integrity(int(diag["n_intraday_rows"].sum()) == len(intraday),
              "03: somma di n_intraday_rows diversa dal numero di righe intraday")
    integrity(not intraday.duplicated(["quote_date", "quote_time"]).any(),
              "03 intraday: coppie data-minuto duplicate")
    panel_early = set(panel.loc[pd.to_numeric(panel["is_early_close"], errors="coerce") == 1, "quote_date"])
    integrity(panel_early == EARLY_CLOSES,
              f"chiusure anticipate nel pannello 07 {sorted(panel_early)} diverse da {sorted(EARLY_CLOSES)}")

    regular_dates = sorted(set(diag["quote_date"]) - EARLY_CLOSES)
    integrity(len(regular_dates) == REGULAR_SESSIONS_EXPECTED,
              f"sessioni regolari {len(regular_dates)}, attese {REGULAR_SESSIONS_EXPECTED}")

    # le mediane per giornata del file diagnostico si ricalcolano dall'intraday
    recomp = intraday.groupby("quote_date").agg(
        med=("n_otm_strikes", "median"), mx=("n_otm_strikes", "max"),
        tr=("truncation_ratio", "median"))
    d_idx = diag.set_index("quote_date")
    integrity(np.allclose(recomp.loc[d_idx.index, "med"], d_idx["n_otm_strikes_median"]) and
              np.allclose(recomp.loc[d_idx.index, "mx"], d_idx["n_otm_strikes_max"]) and
              np.allclose(recomp.loc[d_idx.index, "tr"], d_idx["truncation_ratio_median"], equal_nan=True),
              "03: le statistiche per giornata del file diagnostico non si ricalcolano dall'intraday")

    # --- 1. TABELLA 3.1
    reg_diag = diag[diag["quote_date"].isin(regular_dates)].copy()
    at_check = intraday[(intraday["quote_time"] == CHECK_TIME) & intraday["quote_date"].isin(regular_dates)].copy()
    integrity(len(at_check) == REGULAR_SESSIONS_EXPECTED and at_check["quote_date"].is_unique,
              f"righe alle {CHECK_TIME}: {len(at_check)}, attese {REGULAR_SESSIONS_EXPECTED}")
    integrity(at_check["mfiv"].notna().all(), f"MFIV mancante alle {CHECK_TIME} in qualche sessione regolare")

    # variante: mediana nella sessione senza i minuti senza strip
    computed_rows = intraday[intraday["mfiv"].notna() & intraday["quote_date"].isin(regular_dates)]
    med_without_empty = computed_rows.groupby("quote_date")["n_otm_strikes"].median()

    last_ok_min = reg_diag["last_ok_quote_time"].map(time_to_minute)

    table_spec = [
        ("strike_nello_strip_mediana_nella_sessione", "Strikes in the strip, median within the session",
         reg_diag["n_otm_strikes_median"], "03 diagnostiche: n_otm_strikes_median",
         "mediana per giornata su tutte le righe, compresi i 5 minuti senza strip (n_otm_strikes = 0)"),
        ("strike_nello_strip_massimo_nella_sessione", "Strikes in the strip, maximum over the session",
         reg_diag["n_otm_strikes_max"], "03 diagnostiche: n_otm_strikes_max", ""),
        ("strike_nello_strip_1555", "Strikes in the strip at 15:55",
         at_check["n_otm_strikes"], "03 intraday 15:55: n_otm_strikes", ""),
        ("di_cui_put_1555", "of which puts", at_check["n_otm_put_strikes"],
         "03 intraday 15:55: n_otm_put_strikes", ""),
        ("di_cui_call_1555", "of which calls", at_check["n_otm_call_strikes"],
         "03 intraday 15:55: n_otm_call_strikes", ""),
        ("strike_otm_quotati_1555", "Listed OTM strikes at 15:55", at_check["n_otm_strikes_raw"],
         "03 intraday 15:55: n_otm_strikes_raw",
         "strike OTM con quota valida (solver riuscito, mid > 0), bid a zero inclusi"),
        ("copertura_sotto_F_1555_sd", "Reach below the forward at 15:55, standard deviations",
         at_check["coverage_low_sd"], "03 intraday 15:55: coverage_low_sd", "1 sd = sigma_ATM sqrt(tau) F"),
        ("copertura_sopra_F_1555_sd", "Reach above the forward at 15:55, standard deviations",
         at_check["coverage_high_sd"], "03 intraday 15:55: coverage_high_sd", "1 sd = sigma_ATM sqrt(tau) F"),
        ("quota_somma_non_troncata_mediana_nella_sessione", "Share of the untruncated sum kept, median within the session",
         reg_diag["truncation_ratio_median"], "03 diagnostiche: truncation_ratio_median", ""),
        ("quota_somma_non_troncata_1555", "Share of the untruncated sum kept at 15:55",
         at_check["truncation_ratio"], "03 intraday 15:55: truncation_ratio", ""),
        ("ultimo_minuto_valido", "Last valid minute", last_ok_min,
         "03 diagnostiche: last_ok_quote_time", "minuti dalla mezzanotte, riportati come HH:MM"),
    ]

    table_rows = []
    for key, label, values, source, conv in table_spec:
        v = pd.to_numeric(values, errors="coerce")
        integrity(v.notna().sum() == REGULAR_SESSIONS_EXPECTED, f"Tabella 3.1, {key}: valori mancanti")
        row = {"riga": key, "riga_tesi": label, "n_sessioni": int(v.notna().sum()),
               "fonte": source, "convenzione": conv}
        for p in PERCENTILES:
            row[{50: "mediana", 5: "p05", 95: "p95"}[p]] = nearest_rank(v, p)
        table_rows.append(row)

    table_rows.append({
        "riga": "variante_strike_mediana_senza_minuti_senza_strip",
        "riga_tesi": "(non in tesi) come la prima riga, escludendo i minuti senza strip",
        "n_sessioni": int(med_without_empty.notna().sum()),
        "fonte": "03 intraday: n_otm_strikes sulle righe con MFIV calcolata",
        "convenzione": "variante informativa",
        "mediana": nearest_rank(med_without_empty, 50), "p05": nearest_rank(med_without_empty, 5),
        "p95": nearest_rank(med_without_empty, 95),
    })
    table = pd.DataFrame(table_rows)[["riga", "riga_tesi", "mediana", "p05", "p95", "n_sessioni", "fonte", "convenzione"]]
    table.to_csv(OUTPUT_DIR / OUT_TABLE_3_1, index=False)

    print()
    print("TABELLA 3.1 (percentili per rango piu' vicino, 247 sessioni regolari)")
    for r in table_rows:
        if r["riga"] == "ultimo_minuto_valido":
            print(f"  {r['riga_tesi']:<62s} {minute_to_hhmm(r['mediana'])} {minute_to_hhmm(r['p05'])} {minute_to_hhmm(r['p95'])}")
        else:
            print(f"  {r['riga_tesi']:<62s} {r['mediana']:.4g} {r['p05']:.4g} {r['p95']:.4g}")

    T31_THESIS = {
        "strike_nello_strip_mediana_nella_sessione": (("64", "intero"), ("40.5", "d1"), ("114", "intero")),
        "strike_nello_strip_massimo_nella_sessione": (("97", "intero"), ("62", "intero"), ("151", "intero")),
        "strike_nello_strip_1555": (("7", "intero"), ("4", "intero"), ("16", "intero")),
        "di_cui_put_1555": (("4", "intero"), ("2", "intero"), ("8", "intero")),
        "di_cui_call_1555": (("3", "intero"), ("1", "intero"), ("8", "intero")),
        "strike_otm_quotati_1555": (("222", "intero"), ("193", "intero"), ("464", "intero")),
        "copertura_sotto_F_1555_sd": (("2.57", "d2"), ("1.61", "d2"), ("4.00", "d2")),
        "copertura_sopra_F_1555_sd": (("2.15", "d2"), ("1.21", "d2"), ("3.34", "d2")),
        "quota_somma_non_troncata_mediana_nella_sessione": (("0.50", "d2"), ("0.20", "d2"), ("0.87", "d2")),
        "quota_somma_non_troncata_1555": (("0.049", "d3"), ("0.018", "d3"), ("0.229", "d3")),
        "ultimo_minuto_valido": (("15:48", "HH:MM"), ("15:25", "HH:MM"), ("15:55", "HH:MM")),
    }
    for r in table_rows:
        if r["riga"] not in T31_THESIS:
            continue
        for (tv, prec), col, lab in zip(T31_THESIS[r["riga"]], ("mediana", "p05", "p95"),
                                        ("mediana", "5o pct", "95o pct")):
            calc = minute_to_hhmm(r[col]) if prec == "HH:MM" else r[col]
            nota = r["convenzione"]
            if r["riga"] == "strike_nello_strip_mediana_nella_sessione" and col == "p05":
                nota += f"; senza i minuti senza strip sarebbe {nearest_rank(med_without_empty, 5):g}"
            check(22, f"Tab. 3.1: {r['riga_tesi']}, {lab}", tv, calc, prec, nota)

    # --- 2. REGOLA DI VALIDITA' ALLE 15:55
    n_fail = int((at_check["mfiv_flag"] != "ok").sum())
    trunc_reg = trunc[trunc["quote_date"].isin(regular_dates)]
    integrity((trunc_reg["eod_quote_time"].astype(str) == CHECK_TIME).all(),
              "03 troncamento: l'ultimo minuto con MFIV non e' sempre 15:55 nelle sessioni regolari")
    n_fail_trunc = int((trunc_reg["eod_mfiv_flag"] != "ok").sum())
    integrity(n_fail == n_fail_trunc, "03: conteggio dei fallimenti alle 15:55 diverso fra intraday e troncamento")
    record("validita_1555", "sessioni regolari in cui la regola fallisce alle 15:55", n_fail,
           f"su {len(at_check)}; flag {at_check['mfiv_flag'].value_counts().to_dict()}")
    check(21, "at 15:55 the validity rule fails on 205 of the 247 regular sessions (sessioni)", "205", n_fail, "intero")
    check(21, "... of the 247 regular sessions (denominatore)", "247", len(at_check), "intero")

    # --- 3. TAU: ORIZZONTE, RAPPORTO, FATTORE, BENCHMARK
    intraday["_tau_min"] = intraday["tau"] * MINUTES_PER_YEAR
    reg_intr = intraday[intraday["quote_date"].isin(regular_dates)]
    tau_0940 = reg_intr.loc[reg_intr["quote_time"] == "09:40:00", "_tau_min"].round(6)
    tau_1550 = reg_intr.loc[reg_intr["quote_time"] == "15:50:00", "_tau_min"].round(6)
    integrity(tau_0940.nunique() == 1 and tau_1550.nunique() == 1,
              "tau alle 09:40 o alle 15:50 non uguale in tutte le sessioni regolari")
    m0940, m1550 = float(tau_0940.iloc[0]), float(tau_1550.iloc[0])
    record("tau", "minuti residui alle 09:40", m0940)
    record("tau", "minuti residui alle 15:50", m1550)
    record("tau", "rapporto 09:40 / 15:50", m0940 / m1550)
    check(22, "remaining horizon at 09:40 (minutes)", "380", m0940, "intero")
    check(22, "remaining horizon at 15:50 (minutes)", "10", m1550, "intero")
    check(22, "total implied variance would fall 38-fold", "38", m0940 / m1550, "intero")

    # tau_open dal pannello giornaliero (07) e confronto con tau alle 09:31 nello stadio 03
    p_idx = panel.set_index("quote_date")
    tau_0931 = intraday[intraday["quote_time"] == SURFACE_START_TIME].set_index("quote_date")["tau"]
    integrity(set(tau_0931.index) == set(p_idx.index), "giornate dello stadio 03 e del pannello 07 diverse")
    integrity(np.allclose(p_idx.loc[tau_0931.index, "tau_open"], tau_0931),
              "tau_open del pannello 07 diverso da tau alle 09:31 nello stadio 03")
    open_min = (p_idx["tau_open"] * MINUTES_PER_YEAR).round(6)
    open_reg = open_min[~open_min.index.isin(EARLY_CLOSES)]
    open_early = open_min[open_min.index.isin(EARLY_CLOSES)]
    integrity(open_reg.nunique() == 1 and open_early.nunique() == 1, "tau_open non costante per tipo di sessione")
    tau_open_reg, tau_open_early = float(open_reg.iloc[0]), float(open_early.iloc[0])
    check(23, "tau_open = tau(09:31), regular session (minutes)", "389", tau_open_reg, "intero")
    check(23, "tau_open on early closes (minutes)", "209", tau_open_early, "intero")

    # tau medio sui minuti validi, con la stessa maschera di 07_Daily_panel.py
    intraday["_minute"] = intraday["quote_time"].map(time_to_minute)
    end_minute = intraday["quote_date"].map(lambda d: EARLY_EXPIRY_MINUTE if d in EARLY_CLOSES else REGULAR_EXPIRY_MINUTE)
    mask_ok = ((intraday["mfiv_flag"] == "ok") & intraday["mfiv"].notna()
               & (intraday["_minute"] >= time_to_minute(SURFACE_START_TIME))
               & (intraday["_minute"] <= end_minute))
    tau_bar = intraday[mask_ok].groupby("quote_date")["tau"].mean()
    by_day = pd.DataFrame({"tau_medio_minuti": tau_bar * MINUTES_PER_YEAR})
    by_day["tau_open_minuti"] = p_idx.loc[by_day.index, "tau_open"] * MINUTES_PER_YEAR
    by_day["n_minuti_validi"] = intraday[mask_ok].groupby("quote_date").size()
    by_day["rapporto_tau_medio_su_tau_open"] = by_day["tau_medio_minuti"] / by_day["tau_open_minuti"]
    by_day["fattore_tau_open_su_tau_medio"] = 1.0 / by_day["rapporto_tau_medio_su_tau_open"]
    by_day["chiusura_anticipata"] = by_day.index.isin(EARLY_CLOSES).astype(int)
    integrity(len(by_day) == ALL_SESSIONS_EXPECTED and by_day.notna().all().all(), "tau medio mancante in qualche giornata")
    by_day.reset_index().to_csv(OUTPUT_DIR / OUT_TAU_BY_DAY, index=False)

    stats = {}
    for lab, sub in (("250", by_day), ("247", by_day[by_day["chiusura_anticipata"] == 0])):
        rr, ff = sub["rapporto_tau_medio_su_tau_open"], sub["fattore_tau_open_su_tau_medio"]
        stats[lab] = {"p05": nearest_rank(rr, 5), "p95": nearest_rank(rr, 95),
                      "fattore_mediana": float(ff.median()), "fattore_media": float(ff.mean()), "n": len(sub)}
        for k, v in stats[lab].items():
            record("tau", f"{k} (sessioni {lab})", v, "percentili per rango piu' vicino")
    print()
    print("TAU MEDIO / TAU OPEN (minuti validi)")
    for lab, s in stats.items():
        print(f"  {lab} sessioni: p05 {s['p05']:.4f}, p95 {s['p95']:.4f}; fattore mediana {s['fattore_mediana']:.4f}, media {s['fattore_media']:.4f}")

    s250, s247 = stats["250"], stats["247"]
    check(23, "tau_bar/tau_open, 5th percentile", "0.506", s250["p05"], "d3",
          f"tutte le 250 sessioni; sulle 247 regolari {s247['p05']:.4f}")
    check(23, "tau_bar/tau_open, 95th percentile", "0.551", s250["p95"], "d3",
          f"tutte le 250 sessioni; sulle 247 regolari {s247['p95']:.4f}")
    check(23, "the factor is about 1.935", "1.935", s250["fattore_mediana"], "d3",
          f"e' la MEDIANA di tau_open/tau_medio (247 regolari: {s247['fattore_mediana']:.4f}); "
          f"la media e' {s250['fattore_media']:.3f} (247: {s247['fattore_media']:.3f}); la tesi non dice 'mediana'")

    med_last = nearest_rank(last_ok_min, 50)
    mins_before = REGULAR_EXPIRY_MINUTE - med_last
    record("tau", "ultimo minuto valido mediano (247 regolari)", minute_to_hhmm(med_last))
    record("tau", "minuti dall'ultimo minuto valido mediano alla scadenza", mins_before)
    check(23, "valid minutes end a median of 12 minutes before expiry", "12", mins_before, "intero",
          f"mediana di last_ok_quote_time = {minute_to_hhmm(med_last)}")

    slope = tau_open_reg / MINUTES_PER_YEAR
    logint = math.log(tau_open_reg / MINUTES_PER_YEAR)
    record("benchmark", "tau_open in anni (389/525600)", slope)
    record("benchmark", "ln(389/525600)", logint)
    check(23, "slope benchmark tau_open (years), 7.4e-4", "7.4", slope * 1e4, "d1", "valore in unita' di 1e-4")
    check(23, "ln(389/525,600)", "-7.209", logint, "d3")

    # --- 4. APPENDICE B.1
    flags = intraday["mfiv_flag"].value_counts().to_dict()
    n_rows, n_ok, n_thin = len(intraday), flags.get("ok", 0), flags.get("thin_otm", 0)
    n_rest = n_rows - n_ok - n_thin
    rest_flags = {k: v for k, v in flags.items() if k not in ("ok", "thin_otm")}
    integrity(int(diag["n_ok"].sum()) == n_ok and int(diag["n_thin_otm"].sum()) == n_thin,
              "03: conteggi ok/thin diversi fra intraday e diagnostiche")
    for k, v in flags.items():
        record("appendice_B1", f"righe con mfiv_flag = {k}", v)
    check(68, "minute rows", "96,960", n_rows, "intero")
    check(68, "rows satisfying the validity rule", "92,828", n_ok, "intero")
    check(68, "rows falling below it", "2,881", n_thin, "intero")
    check(68, "the rest have no forward or are flagged near expiry", "solo missing_F e near_expiry",
          f"{n_rest:,} righe: {rest_flags}", "testo",
          "missing_F: 16:00 delle sessioni regolari, 13:00 delle chiusure anticipate, 30 luglio 11:23",
          esito=set(rest_flags) <= {"missing_F", "near_expiry"})

    # --- 5. BANDE DI LOG-MONEYNESS (varianti di 03_MFIV.py)
    for band, pair in LOGM_INTRADAY.items():
        b = pd.read_csv(path(pair), usecols=["quote_date", "quote_time", "mfiv", "mfiv_flag"])
        integrity(len(b) == n_rows, f"03 banda {band}: {len(b)} righe, attese {n_rows}")
        computed = b["mfiv"].notna()
        n_comp = int(computed.sum())
        n_bad = int((computed & (b["mfiv_flag"] != "ok")).sum())
        record("bande", f"|k|<={band}: minuti calcolati", n_comp)
        record("bande", f"|k|<={band}: minuti calcolati che non soddisfano la regola", n_bad)
        check(21, f"band |k|<={band}: every computed minute satisfies the rule (minuti non validi)", "0", n_bad,
              "intero", f"{n_comp:,} minuti calcolati; flag {b['mfiv_flag'].value_counts().to_dict()}")

    # --- 6. PANNELLO INTRADAY (08_Intraday_panel.py)
    thin_panel = ipanel["mfiv_flag"] == "thin_otm"
    record("pannello_intraday", "righe thin_otm conservate nel pannello", int(thin_panel.sum()),
           f"flag {ipanel['mfiv_flag'].value_counts().to_dict()}")
    check(21, "the intraday panel retains every computed grid value (righe thin_otm nel pannello)",
          "righe non valide conservate", f"{int(thin_panel.sum())} righe thin_otm", "testo",
          esito=int(thin_panel.sum()) > 0)
    gt = ipanel["grid_time"].astype(str).str.strip()
    gt = gt.where(gt.str.len() > 5, gt + ":00")
    closing = (gt >= CLOSING_TERCILE[0]) & (gt <= CLOSING_TERCILE[1])
    TAB55 = {5: "262", 15: "125", 30: "49", 60: "0"}
    for h in HORIZONS:
        n = int((thin_panel & closing & (pd.to_numeric(ipanel[f"h{h}_valid"], errors="coerce") == 1)).sum())
        record("pannello_intraday", f"righe di chiusura che falliscono la regola, h = {h}", n)
        check(46, f"Tab. 5.5 note: closing rows failing the baseline strip-validity rule, h = {h}",
              TAB55[h], n, "intero",
              f"thin_otm fuori dal terzile di chiusura: {int((thin_panel & ~closing).sum())}")

    # --- 7. CONTROFATTUALE ZERO ASK
    print()
    print("CONTROFATTUALE ZERO ASK")
    za_raw = pd.read_csv(zero_ask_path)
    colmap = {}
    for key, cands in ZERO_ASK_COLUMN_CANDIDATES.items():
        found = [c for c in cands if c in za_raw.columns]
        integrity(bool(found), f"lista zero ask: manca la colonna '{key}' (cercate {cands}); colonne {list(za_raw.columns)}")
        colmap[key] = found[0]
    za = pd.DataFrame({k: za_raw[v] for k, v in colmap.items()})
    za["date"] = pd.to_datetime(za["date"].astype(str)).dt.strftime("%Y-%m-%d")
    za["time"] = za["time"].astype(str).str.strip()
    za["time"] = za["time"].where(za["time"].str.len() > 5, za["time"] + ":00")
    za["type"] = za["type"].astype(str).str.strip().str.upper().str[0]
    za["strike"] = pd.to_numeric(za["strike"], errors="coerce")
    za["ask"] = pd.to_numeric(za["ask"], errors="coerce")
    za["bid"] = pd.to_numeric(za["bid"], errors="coerce")
    za["F"] = pd.to_numeric(za["F"], errors="coerce")
    otm_txt = za["otm"].astype(str).str.strip().str.lower()
    integrity(otm_txt.isin(["0", "1", "0.0", "1.0", "true", "false"]).all(),
              f"lista zero ask: valori inattesi nel flag OTM {sorted(otm_txt.unique())[:10]}")
    za["otm"] = otm_txt.isin(["1", "1.0", "true"]).astype(int)
    integrity(((za["ask"] <= 0) | za["ask"].isna()).all(), "lista zero ask: contiene quote con ask > 0")
    # il flag OTM della lista deve concordare con strike e forward della lista stessa
    otm_from_F = (((za["type"] == "P") & (za["strike"] < za["F"]))
                  | ((za["type"] == "C") & (za["strike"] > za["F"]))).astype(int)
    n_flag_disagree = int((otm_from_F != za["otm"]).sum())
    record("zero_ask", "quote in cui il flag OTM della lista non concorda con strike e forward", n_flag_disagree)
    integrity(n_flag_disagree == 0,
              f"lista zero ask: il flag OTM non concorda con strike e forward in {n_flag_disagree} quote")
    record("zero_ask", "quote con ask nullo nel 2025", len(za))
    record("zero_ask", "di cui OTM (flag della lista)", int(za["otm"].sum()))
    record("zero_ask", "di cui con bid > 0 (quote incrociate)", int((za["bid"] > 0).sum()))
    print(f"  quote ad ask nullo: {len(za):,}; OTM {int(za['otm'].sum()):,}; con bid > 0 {int((za['bid'] > 0).sum()):,}")

    affected = (za[za["otm"] == 1][["date", "time"]].drop_duplicates()
                .sort_values(["date", "time"]).reset_index(drop=True))
    print(f"  minuti con quote OTM ad ask nullo: {len(affected)} su {affected['date'].nunique()} giornate "
          f"({affected.groupby('date').size().to_dict()})")

    intr_idx = intraday.set_index(["quote_date", "quote_time"])
    repro_rows, cf_rows = [], []
    for day, mins in affected.groupby("date", sort=True):
        y, m, d = day.split("-")
        f = DATA_DIR / BLACK76_FOLDER / BLACK76_PATTERN.format(y=y, m=m, d=d)
        require(f)
        raw = pd.read_csv(f, usecols=BLACK76_COLUMNS, dtype={"quote_date": str, "quote_time": str})
        bd = prepare_black76_day(raw)
        del raw
        for t in mins["time"]:
            g = bd[bd["quote_time"] == t]
            integrity(not g.empty, f"{day} {t}: minuto assente dal file dello stadio 02")
            F = safe_median(g["_F"])
            # quote ad ask nullo di quel minuto nel file dello stadio 02, OTM rispetto al F del minuto
            zg = g[(g["_raw_ask"] <= 0) | g["_raw_ask"].isna()]
            z_otm = zg[((zg["_option_type"] == "P") & (zg["_strike"] < F))
                       | ((zg["_option_type"] == "C") & (zg["_strike"] > F))]
            zl = za[(za["date"] == day) & (za["time"] == t) & (za["otm"] == 1)]
            same_list = (sorted(zip(zl["type"], zl["strike"])) == sorted(zip(z_otm["_option_type"], z_otm["_strike"])))
            extra = {s: set(z_otm.loc[z_otm["_option_type"] == s, "_strike"]) for s in ("P", "C")}

            base = strip_for_minute(g, {})
            cboe = strip_for_minute(g, extra)
            ref = intr_idx.loc[(day, t)]
            mf_ref, mf_base = float(ref["mfiv"]), base["mfiv"]
            mf_ok = (np.isfinite(mf_ref) and np.isfinite(mf_base)
                     and abs(mf_base - mf_ref) <= REL_TOL_MFIV * abs(mf_ref))
            repro = (int(ref["n_otm_put_strikes"]) == base["n_put"] and int(ref["n_otm_call_strikes"]) == base["n_call"]
                     and int(ref["n_otm_strikes"]) == base["n_strikes"] and str(ref["mfiv_flag"]) == base["flag"] and mf_ok)
            repro_rows.append({
                "quote_date": day, "quote_time": t,
                "n_put_05": int(ref["n_otm_put_strikes"]), "n_put_ricostruito": base["n_put"],
                "n_call_05": int(ref["n_otm_call_strikes"]), "n_call_ricostruito": base["n_call"],
                "n_strike_05": int(ref["n_otm_strikes"]), "n_strike_ricostruito": base["n_strikes"],
                "flag_05": str(ref["mfiv_flag"]), "flag_ricostruito": base["flag"],
                "mfiv_05": mf_ref, "mfiv_ricostruita": mf_base,
                "esito": "OK" if repro else "DIVERSO",
            })
            dist = (z_otm["_strike"] - F).abs()
            cf_rows.append({
                "quote_date": day, "quote_time": t, "F": F,
                "n_zero_ask_otm_put": len(extra["P"]), "n_zero_ask_otm_call": len(extra["C"]),
                "distanza_min_zero_ask_da_F": float(dist.min()) if len(dist) else np.nan,
                "distanza_mediana_zero_ask_da_F": float(dist.median()) if len(dist) else np.nan,
                "lista_50_coincide_con_file_04": int(same_list),
                "n_strike_zero_ask_gia_validi": cboe["n_overlap_zero_ask_valid"],
                "base_n_put": base["n_put"], "base_n_call": base["n_call"], "base_flag": base["flag"],
                "base_mfiv": base["mfiv"],
                "cboe_n_put": cboe["n_put"], "cboe_n_call": cboe["n_call"], "cboe_flag": cboe["flag"],
                "cboe_mfiv": cboe["mfiv"],
                "strip_cambia": int((base["n_put"], base["n_call"], base["n_strikes"]) != (cboe["n_put"], cboe["n_call"], cboe["n_strikes"])
                                    or not (np.isfinite(base["mfiv"]) and np.isfinite(cboe["mfiv"])
                                            and abs(base["mfiv"] - cboe["mfiv"]) <= REL_TOL_MFIV * abs(base["mfiv"]))),
                "da_valido_a_non_valido": int(base["flag"] == "ok" and cboe["flag"] != "ok"),
            })
        del bd

    repro_df = pd.DataFrame(repro_rows)
    repro_df.to_csv(OUTPUT_DIR / OUT_REPRO, index=False)
    cf = pd.DataFrame(cf_rows)
    cf.to_csv(OUTPUT_DIR / OUT_COUNTERFACTUAL, index=False)

    n_repro_ok = int((repro_df["esito"] == "OK").sum()) if len(repro_df) else 0
    print(f"  riproduzione della camminata dello stadio 03: {n_repro_ok}/{len(repro_df)} minuti identici "
          f"(strike put, call, totale, flag, MFIV con tolleranza relativa {REL_TOL_MFIV:g})")
    integrity(n_repro_ok == len(repro_df),
              "la camminata ricostruita non riproduce lo stadio 03 su qualche minuto: il controfattuale non e' affidabile "
              f"(vedi {OUT_REPRO})")
    integrity(int(cf["lista_50_coincide_con_file_04"].sum()) == len(cf) if len(cf) else True,
              "le quote OTM ad ask nullo della lista non coincidono con quelle del file dello stadio 02")

    n_aff, n_changed = len(cf), int(cf["strip_cambia"].sum()) if len(cf) else 0
    to_invalid = cf.groupby("quote_date")["da_valido_a_non_valido"].sum().to_dict() if len(cf) else {}
    record("zero_ask", "minuti con quote OTM ad ask nullo", n_aff)
    record("zero_ask", "minuti in cui lo strip cambia con la regola Cboe", n_changed)
    record("zero_ask", "minuti validi che diventano non validi, per giornata", str({k: int(v) for k, v in to_invalid.items()}))
    record("zero_ask", "strike ad ask nullo gia' presenti fra le quote valide", int(cf["n_strike_zero_ask_gia_validi"].sum()) if len(cf) else 0)
    print(f"  lo strip cambia in {n_changed} minuti su {n_aff}; minuti validi che diventano non validi: "
          f"{ {k: int(v) for k, v in to_invalid.items()} }")
    if len(cf):
        print(cf[["quote_date", "quote_time", "base_n_put", "cboe_n_put", "base_n_call", "cboe_n_call",
                  "base_flag", "cboe_flag"]].to_string(index=False))
    check(21, "in 2025 the two rules [zero bid only vs zero bid or zero ask] give the same strips",
          "stessi strip (0 minuti diversi)", f"strip diversi in {n_changed} minuti su {n_aff}", "testo",
          f"minuti con quote OTM ad ask nullo per giornata {affected.groupby('date').size().to_dict()}; "
          f"validi che diventano non validi {({k: int(v) for k, v in to_invalid.items()})}; "
          "regola Cboe come descritta nella tesi (si ferma anche sugli ask nulli)",
          esito=(n_changed == 0))

    # --- USCITA E VERDETTO
    pd.DataFrame(text_rows).to_csv(OUTPUT_DIR / OUT_TEXT_NUMBERS, index=False)
    th = pd.DataFrame(thesis_rows)[["pagina", "voce", "valore_tesi", "valore_calcolato", "precisione", "esito", "nota"]]
    th.to_csv(OUTPUT_DIR / OUT_THESIS, index=False)

    n_ok_t = int((th["esito"] == "OK").sum())
    n_diff = int((th["esito"] == "DIVERSO").sum())
    print()
    print("=" * 130)
    print(f"CONFRONTO CON LA TESI: {n_ok_t} OK, {n_diff} DIVERSO su {len(th)} voci")
    for _, r in th[th["esito"] == "DIVERSO"].iterrows():
        print(f"  DIVERSO p. {r['pagina']}: {r['voce']} | tesi {r['valore_tesi']} | calcolato {r['valore_calcolato']} | {r['nota']}")
    print(f"VERDETTO: {'tutti i valori coincidono' if n_diff == 0 else 'ci sono valori DIVERSI (vedi sopra)'}")
    print(f"Output in {OUTPUT_DIR}")
    print(f"Tempo: {time.time() - t0:.1f} s")


def main():
    strike_count_by_time()
    mfiv_strip_diagnostics()


if __name__ == "__main__":
    main()
