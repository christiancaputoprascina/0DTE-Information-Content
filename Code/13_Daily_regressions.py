"""
Regressioni giornaliere di RV su MFIV (disegni A e B) per le Tab. 5.1, D.3 e D.4 della tesi.

Parte 1: controlli sul pannello e stima delle 28 specificazioni (OLS, errori di Newey-West).
Parte 2: le stesse stime sui pannelli troncati a moneyness (logm_005 e logm_002).
Parte 3: A5 della tesi ("A4" negli output) su tutte le sessioni e senza aprile.
Parte 4: A4 della tesi (solo H_VOL), con il controllo di riproduzione di A3 e A5.
Negli output delle parti 1 e 2, A5 e A6 corrispondono ad A3 e A5 della tesi.

Input:  CBOE/Data/07_daily_panel/
Output: CBOE/Data/13_daily_regressions/
"""

import hashlib
import math
import os
import traceback

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from common import (DATA_DIR, N_APRIL_SESSIONS, YEAR as _YEAR, clean_date_series,
                    make_log_column, nw_lag_length, output_dir, path, require,
                    safe_write_csv, to_numeric_col)

YEAR = str(_YEAR)  # "2025": le date si filtrano per prefisso

OUTPUT_FOLDER = "13_daily_regressions"
OUTPUT_DIR = output_dir(OUTPUT_FOLDER)
PANEL_FOLDER = "07_daily_panel"


# =====================================================================
# PARTE 1: REGRESSIONI GIORNALIERE SUL PANNELLO DI BASELINE
# =====================================================================

PANEL_FILE = DATA_DIR / PANEL_FOLDER / "07_daily_regression_panel_2025.csv"

RESULTS_OUTPUT_FILE = OUTPUT_DIR / "13_daily_regression_results_2025.csv"
PREFLIGHT_OUTPUT_FILE = OUTPUT_DIR / "13_preflight_validation_2025.csv"
A1_HANDCHECK_OUTPUT_FILE = OUTPUT_DIR / "13_A1_handcheck_2025.csv"
SPEC_SUMMARY_OUTPUT_FILE = OUTPUT_DIR / "13_specification_summary_2025.csv"

EXPECTED_N_DAYS = 250

# Valore di controllo: RV_t del 16 giugno 2025 con i rendimenti fra chiusure
# consecutive. Impedisce di stimare su un pannello diverso; non dipende dalla
# MFIV, quindi vale anche per i pannelli delle varianti (parte 2).
EXPECTED_RV_2025_06_16_DISPLAY = "1.977605e-05"

FLOAT_TOL = 1e-12
PVALUE_TOL = 1e-12


def require_columns(df: pd.DataFrame, columns, panel_file):
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            "Il daily regression panel non contiene tutte le colonne richieste.\n"
            f"Missing columns: {missing}\n"
            f"Panel file: {panel_file}"
        )


def load_panel() -> pd.DataFrame:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Daily regression panel non trovato:\n{PANEL_FILE}")

    df = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)

    if "quote_date" not in df.columns:
        raise ValueError("Il daily regression panel non contiene quote_date.")

    df["quote_date"] = clean_date_series(df["quote_date"])

    df = df[
        df["quote_date"].notna()
        & df["quote_date"].astype(str).str.startswith(YEAR)
    ].copy()

    df["_date_dt"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df = df.sort_values("_date_dt").drop(columns=["_date_dt"]).reset_index(drop=True)

    # H_OI_0931 e H_VOL_0931 (gamma allo snapshot delle 09:31) sono controllate
    # ma non entrano nelle specificazioni.
    required = [
        "RV_t",
        "MFIV_t",
        "MFIV_0931",
        "RV_lag1",
        "SpotRV_open",
        "FwdRV_short",
        "FwdRV_full",
        "H_OI_t",
        "H_VOL_t",
        "H_OI_t_indep",
        "H_VOL_t_indep",
        "H_OI_0931",
        "H_VOL_0931",
        "MFIV_ann_t",
        "MFIV_ann_0931",
        "MFIV_session_t",
        "tau_open",
    ]

    require_columns(df, required, PANEL_FILE)

    for col in required:
        df[col] = to_numeric_col(df, col)

    return df


def run_preflight_validation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "check": "panel_row_count",
        "expected": EXPECTED_N_DAYS,
        "actual": int(len(df)),
        "passed": int(len(df) == EXPECTED_N_DAYS),
        "details": "",
    })

    rows.append({
        "check": "unique_trading_days",
        "expected": EXPECTED_N_DAYS,
        "actual": int(df["quote_date"].nunique()),
        "passed": int(df["quote_date"].nunique() == EXPECTED_N_DAYS),
        "details": "",
    })

    target = df[df["quote_date"] == "2025-06-16"].copy()

    if target.empty:
        rv_display = ""
    else:
        rv_display = f"{float(target['RV_t'].iloc[0]):.6e}"

    rows.append({
        "check": "RV_t_2025_06_16_corrected_panel_check",
        "expected": EXPECTED_RV_2025_06_16_DISPLAY,
        "actual": rv_display,
        "passed": int(rv_display == EXPECTED_RV_2025_06_16_DISPLAY),
        "details": "Ancora sul pannello corretto (chiusure consecutive, 05_spx_minute_bars). "
                   "Un valore diverso significa pannello superato.",
    })

    # Con i rendimenti fra chiusure consecutive FwdRV_full coincide con RV_t per
    # costruzione: la prima barra non ha una precedente, quindi entrambe le finestre
    # partono dal rendimento delle 09:32. Il controllo segnala le giornate in cui divergono.
    rv_diverges = df[
        df["RV_t"].notna()
        & df["FwdRV_full"].notna()
        & np.isfinite(df["RV_t"])
        & np.isfinite(df["FwdRV_full"])
        & ~np.isclose(df["RV_t"], df["FwdRV_full"], atol=1e-18, rtol=0.0)
    ].copy()

    rows.append({
        "check": "FwdRV_full_identical_to_RV_t_under_consecutive_returns",
        "expected": 0,
        "actual": int(len(rv_diverges)),
        "passed": int(len(rv_diverges) == 0),
        "details": "attese identiche: il rendimento delle 09:31 non esiste. "
                   + ";".join(rv_diverges["quote_date"].astype(str).tolist()[:30]),
    })

    rv_lag1_nan = df[df["RV_lag1"].isna()].copy()
    rv_lag1_nan_dates = sorted(rv_lag1_nan["quote_date"].astype(str).tolist())

    rows.append({
        "check": "RV_lag1_nan_count_and_date",
        "expected": "1 row: 2025-01-02",
        "actual": f"{len(rv_lag1_nan)} row(s): {';'.join(rv_lag1_nan_dates)}",
        "passed": int(len(rv_lag1_nan) == 1 and rv_lag1_nan_dates == ["2025-01-02"]),
        "details": "",
    })

    strictly_positive_preflight_cols = [
        "RV_t", "MFIV_t", "MFIV_0931", "FwdRV_short", "FwdRV_full",
    ]

    for col in strictly_positive_preflight_cols:
        bad = df[df[col].isna() | ~np.isfinite(df[col]) | (df[col] <= 0)].copy()

        rows.append({
            "check": f"{col}_strictly_positive_all_rows",
            "expected": 0,
            "actual": int(len(bad)),
            "passed": int(len(bad) == 0),
            "details": ";".join(bad["quote_date"].astype(str).tolist()[:30]),
        })

    for col in ["RV_lag1", "SpotRV_open"]:
        usable = df[col].notna() & np.isfinite(df[col])
        bad = df[usable & (df[col] <= 0)].copy()

        rows.append({
            "check": f"{col}_positive_when_nonmissing_for_log_specs",
            "expected": 0,
            "actual": int(len(bad)),
            "passed": int(len(bad) == 0),
            "details": ";".join(bad["quote_date"].astype(str).tolist()[:30]),
        })

    return pd.DataFrame(rows)


def preflight_passed(preflight: pd.DataFrame) -> bool:
    return bool((preflight["passed"] == 1).all())


def add_log_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    log_vars = [
        "RV_t", "MFIV_t", "RV_lag1", "MFIV_0931",
        "SpotRV_open", "FwdRV_short", "FwdRV_full",
        "MFIV_ann_t", "MFIV_ann_0931", "MFIV_session_t",
    ]

    for col in log_vars:
        make_log_column(out, col, f"log_{col}")

    return out


def build_specifications():
    """Le 28 specificazioni, usate sia nella parte 1 sia nella parte 2."""
    specs = []

    # Su MFIV_session_t (= MFIV_ann_t * tau_open) il benchmark di Mincer-Zarnowitz e'
    # (0, 1) in entrambe le forme: A1_ses e A2_ses sono il riferimento per il test di
    # non distorsione.
    specs.append({"specification": "A1_ses", "form": "levels", "gamma_source": "NA",
                  "dependent": "RV_t", "regressors": ["MFIV_session_t"]})

    specs.append({"specification": "A2_ses", "form": "levels", "gamma_source": "NA",
                  "dependent": "RV_t", "regressors": ["MFIV_session_t", "RV_lag1"]})

    specs.append({"specification": "A6_ses", "form": "levels", "gamma_source": "CBOE",
                  "dependent": "RV_t",
                  "regressors": ["MFIV_session_t", "RV_lag1", "H_OI_t", "H_VOL_t"]})

    specs.append({"specification": "A1_ses", "form": "log", "gamma_source": "NA",
                  "dependent": "log_RV_t", "regressors": ["log_MFIV_session_t"]})

    specs.append({"specification": "A2_ses", "form": "log", "gamma_source": "NA",
                  "dependent": "log_RV_t",
                  "regressors": ["log_MFIV_session_t", "log_RV_lag1"]})

    specs.append({"specification": "A6_ses", "form": "log", "gamma_source": "CBOE",
                  "dependent": "log_RV_t",
                  "regressors": ["log_MFIV_session_t", "log_RV_lag1",
                                 "H_OI_t", "H_VOL_t"]})

    # MFIV_ann_t (media di sessione di mfiv/tau) non dipende dal tempo residuo e non
    # serve al test di non distorsione: in livelli lo slope di riferimento e' la
    # durata della sessione in anni, in log l'intercetta di riferimento non e' zero.
    specs.append({"specification": "A1_ann", "form": "levels", "gamma_source": "NA",
                  "dependent": "RV_t", "regressors": ["MFIV_ann_t"]})

    specs.append({"specification": "A2_ann", "form": "levels", "gamma_source": "NA",
                  "dependent": "RV_t", "regressors": ["MFIV_ann_t", "RV_lag1"]})

    specs.append({"specification": "A6_ann", "form": "levels", "gamma_source": "CBOE",
                  "dependent": "RV_t",
                  "regressors": ["MFIV_ann_t", "RV_lag1", "H_OI_t", "H_VOL_t"]})

    specs.append({"specification": "A1_ann", "form": "log", "gamma_source": "NA",
                  "dependent": "log_RV_t", "regressors": ["log_MFIV_ann_t"]})

    specs.append({"specification": "A2_ann", "form": "log", "gamma_source": "NA",
                  "dependent": "log_RV_t",
                  "regressors": ["log_MFIV_ann_t", "log_RV_lag1"]})

    specs.append({"specification": "A6_ann", "form": "log", "gamma_source": "CBOE",
                  "dependent": "log_RV_t",
                  "regressors": ["log_MFIV_ann_t", "log_RV_lag1", "H_OI_t", "H_VOL_t"]})

    # Su MFIV_t lo slope atteso e' circa 1,935 e non 1: MFIV_t e' la media di sessione
    # della varianza residua, non la varianza dell'intera sessione.
    specs.append({"specification": "A1", "form": "levels", "gamma_source": "NA",
                  "dependent": "RV_t", "regressors": ["MFIV_t"]})

    specs.append({"specification": "A2", "form": "levels", "gamma_source": "NA",
                  "dependent": "RV_t", "regressors": ["MFIV_t", "RV_lag1"]})

    specs.append({"specification": "A5", "form": "levels", "gamma_source": "CBOE",
                  "dependent": "RV_t", "regressors": ["MFIV_t", "RV_lag1", "H_OI_t"]})

    specs.append({"specification": "A5", "form": "levels", "gamma_source": "indep",
                  "dependent": "RV_t", "regressors": ["MFIV_t", "RV_lag1", "H_OI_t_indep"]})

    specs.append({"specification": "A6", "form": "levels", "gamma_source": "CBOE",
                  "dependent": "RV_t",
                  "regressors": ["MFIV_t", "RV_lag1", "H_OI_t", "H_VOL_t"]})

    specs.append({"specification": "A6", "form": "levels", "gamma_source": "indep",
                  "dependent": "RV_t",
                  "regressors": ["MFIV_t", "RV_lag1", "H_OI_t_indep", "H_VOL_t_indep"]})

    specs.append({"specification": "B-short", "form": "levels", "gamma_source": "NA",
                  "dependent": "FwdRV_short",
                  "regressors": ["MFIV_0931", "SpotRV_open", "RV_lag1"]})

    # FwdRV_full coincide con RV_t: B-full ha la stessa dipendente di A e cambia solo
    # il regressore (snapshot delle 09:31 invece della media di sessione).
    specs.append({"specification": "B-full", "form": "levels", "gamma_source": "NA",
                  "dependent": "FwdRV_full",
                  "regressors": ["MFIV_0931", "SpotRV_open", "RV_lag1"]})

    specs.append({"specification": "A1", "form": "log", "gamma_source": "NA",
                  "dependent": "log_RV_t", "regressors": ["log_MFIV_t"]})

    specs.append({"specification": "A2", "form": "log", "gamma_source": "NA",
                  "dependent": "log_RV_t", "regressors": ["log_MFIV_t", "log_RV_lag1"]})

    specs.append({"specification": "A5", "form": "log", "gamma_source": "CBOE",
                  "dependent": "log_RV_t",
                  "regressors": ["log_MFIV_t", "log_RV_lag1", "H_OI_t"]})

    specs.append({"specification": "A5", "form": "log", "gamma_source": "indep",
                  "dependent": "log_RV_t",
                  "regressors": ["log_MFIV_t", "log_RV_lag1", "H_OI_t_indep"]})

    specs.append({"specification": "A6", "form": "log", "gamma_source": "CBOE",
                  "dependent": "log_RV_t",
                  "regressors": ["log_MFIV_t", "log_RV_lag1", "H_OI_t", "H_VOL_t"]})

    specs.append({"specification": "A6", "form": "log", "gamma_source": "indep",
                  "dependent": "log_RV_t",
                  "regressors": ["log_MFIV_t", "log_RV_lag1",
                                 "H_OI_t_indep", "H_VOL_t_indep"]})

    specs.append({"specification": "B-short", "form": "log", "gamma_source": "NA",
                  "dependent": "log_FwdRV_short",
                  "regressors": ["log_MFIV_0931", "log_SpotRV_open", "log_RV_lag1"]})

    specs.append({"specification": "B-full", "form": "log", "gamma_source": "NA",
                  "dependent": "log_FwdRV_full",
                  "regressors": ["log_MFIV_0931", "log_SpotRV_open", "log_RV_lag1"]})

    return specs


def estimate_one_spec(df: pd.DataFrame, spec: dict):
    dependent = spec["dependent"]
    regressors = spec["regressors"]

    required = [dependent] + regressors

    work = df[["quote_date"] + required].copy()

    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=required).copy()
    work = work[np.isfinite(work[dependent])].copy()

    for col in regressors:
        work = work[np.isfinite(work[col])].copy()

    T = int(len(work))

    if T <= len(regressors) + 1:
        raise ValueError(
            f"Too few observations for {spec['specification']} {spec['form']} "
            f"{spec['gamma_source']}: n={T}, regressors={regressors}"
        )

    L = nw_lag_length(T)

    y = work[dependent].astype(float)
    X = work[regressors].astype(float)
    X = sm.add_constant(X, has_constant="add")

    fitted = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True,
    )

    rows = []

    for term in fitted.params.index:
        rows.append({
            "specification": spec["specification"],
            "form": spec["form"],
            "gamma_source": spec["gamma_source"],
            "term": term,
            "coef": float(fitted.params[term]),
            "se_nw": float(fitted.bse[term]),
            "t_stat": float(fitted.tvalues[term]),
            "p_value": float(fitted.pvalues[term]),
            "r_squared": float(fitted.rsquared),
            "adj_r_squared": float(fitted.rsquared_adj),
            "n": T,
            "nw_lag_L": L,
        })

    spec_summary = {
        "specification": spec["specification"],
        "form": spec["form"],
        "gamma_source": spec["gamma_source"],
        "dependent": dependent,
        "regressors": ";".join(regressors),
        "n": T,
        "k_including_intercept": len(regressors) + 1,
        "nw_lag_L": L,
        "first_date_used": work["quote_date"].iloc[0],
        "last_date_used": work["quote_date"].iloc[-1],
    }

    return pd.DataFrame(rows), spec_summary


def estimate_all_specs(df: pd.DataFrame, specs):
    result_frames = []
    summary_rows = []

    for spec in specs:
        print(f"Estimating {spec['specification']} | form={spec['form']} | "
              f"gamma_source={spec['gamma_source']}")

        result, summary = estimate_one_spec(df, spec)

        print(f"    n={summary['n']}, k={summary['k_including_intercept']}, "
              f"Newey-West L={summary['nw_lag_L']} from floor(4*(T/100)^(2/9))")

        result_frames.append(result)
        summary_rows.append(summary)

    return pd.concat(result_frames, ignore_index=True), pd.DataFrame(summary_rows)


def run_A1_handcheck(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    work = df[["quote_date", "RV_t", "MFIV_t"]].copy()

    work["RV_t"] = pd.to_numeric(work["RV_t"], errors="coerce")
    work["MFIV_t"] = pd.to_numeric(work["MFIV_t"], errors="coerce")
    work = work.dropna(subset=["RV_t", "MFIV_t"]).copy()

    y = work["RV_t"].astype(float)
    x = work["MFIV_t"].astype(float)

    x_bar = float(x.mean())
    y_bar = float(y.mean())

    numerator = float(((x - x_bar) * (y - y_bar)).sum())
    denominator = float(((x - x_bar) ** 2).sum())

    b1_manual = numerator / denominator
    a_manual = y_bar - b1_manual * x_bar

    a1_results = results[
        (results["specification"] == "A1")
        & (results["form"] == "levels")
        & (results["gamma_source"] == "NA")
    ].copy()

    const_coef = float(a1_results[a1_results["term"] == "const"]["coef"].iloc[0])
    mfiv_coef = float(a1_results[a1_results["term"] == "MFIV_t"]["coef"].iloc[0])

    a1_mfiv_row = a1_results[a1_results["term"] == "MFIV_t"].iloc[0]

    t_stat = float(a1_mfiv_row["t_stat"])
    n = int(a1_mfiv_row["n"])
    k = 2
    df_t = n - k

    p_manual = float(2.0 * (1.0 - scipy_stats.t.cdf(abs(t_stat), df_t)))
    p_sm = float(a1_mfiv_row["p_value"])

    out = pd.DataFrame([
        {
            "check": "A1_intercept_manual_vs_statsmodels",
            "manual_value": a_manual,
            "statsmodels_value": const_coef,
            "absolute_difference": abs(a_manual - const_coef),
            "passed": int(abs(a_manual - const_coef) <= FLOAT_TOL),
            "details": "Closed-form OLS intercept.",
        },
        {
            "check": "A1_MFIV_t_slope_manual_vs_statsmodels",
            "manual_value": b1_manual,
            "statsmodels_value": mfiv_coef,
            "absolute_difference": abs(b1_manual - mfiv_coef),
            "passed": int(abs(b1_manual - mfiv_coef) <= FLOAT_TOL),
            "details": "Closed-form OLS slope Cov(RV_t, MFIV_t) / Var(MFIV_t).",
        },
        {
            "check": "A1_MFIV_t_p_value_manual_scipy_t_vs_statsmodels",
            "manual_value": p_manual,
            "statsmodels_value": p_sm,
            "absolute_difference": abs(p_manual - p_sm),
            "passed": int(abs(p_manual - p_sm) <= PVALUE_TOL),
            "details": f"Manual p = 2*(1 - t.cdf(|t|, df)); t={t_stat}; n={n}; df={df_t}.",
        },
    ])

    print()
    print("A1 p-value independent check")
    print("-" * 130)
    print(f"t-stat from statsmodels:            {t_stat}")
    print(f"n:                                  {n}")
    print(f"df, n - k:                          {df_t}")
    print(f"p-value, manual scipy:              {p_manual}")
    print(f"p-value, statsmodels:               {p_sm}")
    print(f"absolute difference:                {abs(p_manual - p_sm)}")

    return out


def daily_regressions():
    print("=" * 130)
    print("SECTION 3.7 DAILY REGRESSION ESTIMATION")
    print("=" * 130)
    print(f"DATA_DIR:       {DATA_DIR}")
    print(f"Panel file:     {PANEL_FILE}")
    print(f"Output folder:  {OUTPUT_DIR}")
    print("=" * 130)

    print("Loading corrected daily regression panel...")
    panel = load_panel()

    print(f"Panel rows loaded: {len(panel):,}")
    print(f"Panel days loaded: {panel['quote_date'].nunique():,}")

    print()
    print("=" * 130)
    print("STEP 1 - PREFLIGHT VALIDATION")
    print("=" * 130)

    preflight = run_preflight_validation(panel)
    safe_write_csv(preflight, PREFLIGHT_OUTPUT_FILE)

    print(preflight.to_string(index=False))

    if not preflight_passed(preflight):
        print()
        print("=" * 130)
        print("STOPPING: PREFLIGHT VALIDATION FAILED")
        print("=" * 130)
        print("No regressions were estimated.")
        print(f"Preflight output saved: {PREFLIGHT_OUTPUT_FILE}")
        print("=" * 130)
        raise RuntimeError("Parte 1: preflight validation failed, no regressions estimated.")

    print()
    print("Preflight validation passed. Proceeding to estimation.")

    print()
    print("=" * 130)
    print("STEP 2-4 - LEVELS AND LOG SPECIFICATIONS")
    print("=" * 130)

    panel = add_log_variables(panel)
    specifications = build_specifications()

    print(f"Number of specifications to estimate: {len(specifications)}")
    print("Attese: 28 = 6 su MFIV_session_t + 6 su MFIV_ann_t + 16 sulle altre.")

    results, spec_summary = estimate_all_specs(panel, specifications)

    print()
    print("=" * 130)
    print("STEP 7 - A1 CLOSED-FORM AND P-VALUE HAND CHECK")
    print("=" * 130)

    a1_handcheck = run_A1_handcheck(panel, results)

    print()
    print(a1_handcheck.to_string(index=False))

    if not bool((a1_handcheck["passed"] == 1).all()):
        print()
        print("=" * 130)
        print("WARNING: A1 HAND CHECK FAILED")
        print("=" * 130)
        print("Results are still saved for inspection, but should not be trusted "
              "until resolved.")

    safe_write_csv(results, RESULTS_OUTPUT_FILE)
    safe_write_csv(spec_summary, SPEC_SUMMARY_OUTPUT_FILE)
    safe_write_csv(a1_handcheck, A1_HANDCHECK_OUTPUT_FILE)

    print()
    print("=" * 130)
    print("COMPLETED SECTION 3.7 DAILY REGRESSIONS")
    print("=" * 130)
    print(f"Results output:       {RESULTS_OUTPUT_FILE}")
    print(f"Spec summary output:  {SPEC_SUMMARY_OUTPUT_FILE}")
    print(f"Preflight output:     {PREFLIGHT_OUTPUT_FILE}")
    print(f"A1 handcheck output:  {A1_HANDCHECK_OUTPUT_FILE}")

    print()
    print("--- COEFFICIENTI PRINCIPALI, FORMA IN LIVELLI ---")
    main_lev = results[
        (results["form"] == "levels")
        & (results["term"].isin(["MFIV_t", "MFIV_ann_t", "MFIV_session_t", "MFIV_0931",
                                 "H_OI_t", "H_VOL_t",
                                 "H_OI_t_indep", "H_VOL_t_indep"]))
    ][["specification", "gamma_source", "term", "coef", "se_nw", "t_stat",
       "p_value", "r_squared", "n"]]
    print(main_lev.to_string(index=False))

    print()
    print("--- COEFFICIENTI PRINCIPALI, FORMA LOG ---")
    main_log = results[
        (results["form"] == "log")
        & (results["term"].isin(["log_MFIV_t", "log_MFIV_ann_t", "log_MFIV_session_t",
                                 "log_MFIV_0931",
                                 "H_OI_t", "H_VOL_t",
                                 "H_OI_t_indep", "H_VOL_t_indep"]))
    ][["specification", "gamma_source", "term", "coef", "se_nw", "t_stat",
       "p_value", "r_squared", "n"]]
    print(main_log.to_string(index=False))

    print()
    print("--- TEST DI NON DISTORSIONE (MINCER-ZARNOWITZ) ---")
    print("H0 congiunta: intercetta 0 e slope 1, su MFIV_session_t.")
    print()
    print("PERCHE' SU QUESTA E NON SULLE ALTRE:")
    print("  lo slope atteso e' 1 solo per MFIV_session_t.")
    print("  MFIV_t: slope atteso circa 1,935, non 1, perche' e' la media di")
    print("    sessione della varianza RESIDUA contro la varianza dell'INTERA")
    print("    sessione. Fattore misurato sui dati 2025, non assunto.")
    print("  MFIV_ann_t in livelli: slope atteso la lunghezza della sessione in")
    print("    anni, circa 7,4e-04.")
    print("  MFIV_ann_t in log: slope atteso 1, ma INTERCETTA attesa")
    print("    log(389/525600) = -7,2095, non zero. Il test (0,1) li' rifiuta")
    print("    l'intercetta per costruzione e produce F privi di contenuto.")
    print("  MFIV_session_t = MFIV_ann_t * tau_apertura: benchmark (0,1) pulito in")
    print("    entrambe le forme, che e' lo standard di letteratura.")
    print()

    for form, dep, reg in (("livelli", "RV_t", "MFIV_session_t"),
                           ("log", "log_RV_t", "log_MFIV_session_t")):
        _w = panel[["quote_date", dep, reg]].dropna()
        _y = _w[dep].astype(float)
        _X = sm.add_constant(_w[reg].astype(float), has_constant="add")
        _L = nw_lag_length(len(_w))
        _f = sm.OLS(_y, _X).fit(cov_type="HAC", cov_kwds={"maxlags": _L}, use_t=True)
        _test = _f.f_test("const = 0, " + reg + " = 1")

        _b = _f.params[reg]
        _sb = _f.bse[reg]
        _t_slope = (_b - 1.0) / _sb if _sb > 0 else np.nan

        print(f"  {form}:")
        print(f"    a = {_f.params['const']:+.6e}  (se {_f.bse['const']:.6e})  "
              f"t contro 0 = {_f.tvalues['const']:+.3f}")
        print(f"    b = {_b:.6f}  (se {_sb:.6f})  t contro 1 = {_t_slope:+.3f}")
        print(f"    F congiunta = {float(np.squeeze(_test.fvalue)):.4f}   "
              f"p = {float(np.squeeze(_test.pvalue)):.4f}   n = {len(_w)}   L = {_L}")

        # Si interpreta il valore adattato alla media del regressore, a + b * media(x),
        # confrontato con media(x): in log lo scarto e' il log del rapporto RV/MFIV.
        # L'intercetta e' il valore in x = 0, fuori dal supporto dei dati.
        _xbar = float(_w[reg].mean())
        _fit_at_mean = float(_f.params["const"] + _f.params[reg] * _xbar)
        _gap = _fit_at_mean - _xbar

        print(f"    media del regressore = {_xbar:+.6f}")
        print(f"    valore adattato alla media = {_fit_at_mean:+.6f}")

        if form == "log":
            print(f"    scarto = {_gap:+.6f}  =>  rapporto RV/MFIV = "
                  f"{np.exp(_gap):.4f}")
            print(f"    Sotto 1 significa varianza realizzata sotto quella")
            print(f"    implicita, cioe' premio al rischio di varianza POSITIVO.")
        else:
            print(f"    scarto = {_gap:+.6e}  (in livelli, stessa unita' di RV_t)")

        print()

    print("  COME LEGGERE F CONTRO I DUE t. Se la F congiunta rifiuta mentre i due")
    print("  t individuali no, non e' una contraddizione: il rifiuto avviene lungo")
    print("  una combinazione di intercetta e slope, e la direzione informativa e'")
    print("  il valore adattato alla media stampato sopra, non i due coefficienti")
    print("  presi uno alla volta.")
    print()
    print("  Un rifiuto va comunque verificato con l'analisi di influenza prima di")
    print("  essere riportato: in livelli il risultato dipende dall'episodio di")
    print("  aprile, e quel controllo non e' ripetuto su")
    print("  questi coefficienti.")

    print()
    print("SPECIFICATION SUMMARY")
    print("=" * 130)
    print(spec_summary.to_string(index=False))
    print("=" * 130)


# =====================================================================
# PARTE 2: LE STESSE REGRESSIONI SULLE VARIANTI DI TRONCAMENTO
# =====================================================================
# Verifica se il rifiuto della non distorsione (H1) regge con un'altra regola di
# troncamento: la baseline taglia sui bid nulli, le varianti su una banda di
# moneyness. Specificazioni e stimatore sono quelli della parte 1.

VARIANTS = ("logm_005", "logm_002")

EXPECTED_N_SPECIFICATIONS = 28


def load_panel_logm(panel_file) -> pd.DataFrame:
    if not panel_file.exists():
        raise FileNotFoundError(
            f"Daily regression panel non trovato:\n{panel_file}\n"
            "Lancia prima 07_Daily_panel.py."
        )

    df = pd.read_csv(panel_file, dtype=str, keep_default_na=False)

    if "quote_date" not in df.columns:
        raise ValueError("Il daily regression panel non contiene quote_date.")

    df["quote_date"] = clean_date_series(df["quote_date"])

    df = df[
        df["quote_date"].notna()
        & df["quote_date"].astype(str).str.startswith(YEAR)
    ].copy()

    df["_date_dt"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df = df.sort_values("_date_dt").drop(columns=["_date_dt"]).reset_index(drop=True)

    required = [
        "RV_t",
        "MFIV_t",
        "MFIV_0931",
        "RV_lag1",
        "SpotRV_open",
        "FwdRV_short",
        "FwdRV_full",
        "H_OI_t",
        "H_VOL_t",
        "H_OI_t_indep",
        "H_VOL_t_indep",
        "H_OI_0931",
        "H_VOL_0931",
        "MFIV_ann_t",
        "MFIV_ann_0931",
        "MFIV_session_t",
        "tau_open",
    ]

    require_columns(df, required, panel_file)

    for col in required:
        df[col] = to_numeric_col(df, col)

    return df


def run_preflight_validation_logm(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "check": "panel_row_count",
        "expected": EXPECTED_N_DAYS,
        "actual": int(len(df)),
        "passed": int(len(df) == EXPECTED_N_DAYS),
        "details": "",
    })

    rows.append({
        "check": "unique_trading_days",
        "expected": EXPECTED_N_DAYS,
        "actual": int(df["quote_date"].nunique()),
        "passed": int(df["quote_date"].nunique() == EXPECTED_N_DAYS),
        "details": "",
    })

    target = df[df["quote_date"] == "2025-06-16"].copy()

    if target.empty:
        rv_display = ""
    else:
        rv_display = f"{float(target['RV_t'].iloc[0]):.6e}"

    rows.append({
        "check": "RV_t_2025_06_16_corrected_panel_check",
        "expected": EXPECTED_RV_2025_06_16_DISPLAY,
        "actual": rv_display,
        "passed": int(rv_display == EXPECTED_RV_2025_06_16_DISPLAY),
        "details": "Ancora sul pannello corretto (chiusure consecutive, 05_spx_minute_bars). "
                   "Non dipende dalla MFIV: sotto la variante deve valere lo "
                   "stesso numero della baseline.",
    })

    # FwdRV_full coincide con RV_t per costruzione (vedi la parte 1): il controllo
    # segnala le giornate in cui divergono.
    rv_diverges = df[
        df["RV_t"].notna()
        & df["FwdRV_full"].notna()
        & np.isfinite(df["RV_t"])
        & np.isfinite(df["FwdRV_full"])
        & ~np.isclose(df["RV_t"], df["FwdRV_full"], atol=1e-18, rtol=0.0)
    ].copy()

    rows.append({
        "check": "FwdRV_full_identical_to_RV_t_under_consecutive_returns",
        "expected": 0,
        "actual": int(len(rv_diverges)),
        "passed": int(len(rv_diverges) == 0),
        "details": "attese identiche: il rendimento delle 09:31 non esiste. "
                   + ";".join(rv_diverges["quote_date"].astype(str).tolist()[:30]),
    })

    rv_lag1_nan = df[df["RV_lag1"].isna()].copy()
    rv_lag1_nan_dates = sorted(rv_lag1_nan["quote_date"].astype(str).tolist())

    rows.append({
        "check": "RV_lag1_nan_count_and_date",
        "expected": "1 row: 2025-01-02",
        "actual": f"{len(rv_lag1_nan)} row(s): {';'.join(rv_lag1_nan_dates)}",
        "passed": int(len(rv_lag1_nan) == 1 and rv_lag1_nan_dates == ["2025-01-02"]),
        "details": "",
    })

    strictly_positive_preflight_cols = [
        "RV_t", "MFIV_t", "MFIV_0931", "FwdRV_short", "FwdRV_full",
        "MFIV_session_t", "MFIV_ann_t",
    ]

    for col in strictly_positive_preflight_cols:
        bad = df[df[col].isna() | ~np.isfinite(df[col]) | (df[col] <= 0)].copy()

        rows.append({
            "check": f"{col}_strictly_positive_all_rows",
            "expected": 0,
            "actual": int(len(bad)),
            "passed": int(len(bad) == 0),
            "details": ";".join(bad["quote_date"].astype(str).tolist()[:30]),
        })

    for col in ["RV_lag1", "SpotRV_open"]:
        usable = df[col].notna() & np.isfinite(df[col])
        bad = df[usable & (df[col] <= 0)].copy()

        rows.append({
            "check": f"{col}_positive_when_nonmissing_for_log_specs",
            "expected": 0,
            "actual": int(len(bad)),
            "passed": int(len(bad) == 0),
            "details": ";".join(bad["quote_date"].astype(str).tolist()[:30]),
        })

    # MFIV_session_t = MFIV_ann_t * tau_open deve valere anche nella variante: se no,
    # le colonne derivate sarebbero rimaste quelle della baseline.
    lhs = df["MFIV_session_t"]
    rhs = df["MFIV_ann_t"] * df["tau_open"]

    # Tolleranza relativa: la scrittura e rilettura del CSV arrotonda alla
    # quindicesima cifra, e le grandezze sono dell'ordine di 1e-05.
    scale = lhs.abs().clip(lower=1e-300)
    rel_gap = (lhs - rhs).abs() / scale
    n_gap = int((rel_gap > 1e-12).sum())

    rows.append({
        "check": "MFIV_session_equals_ann_times_tau_open",
        "expected": 0,
        "actual": n_gap,
        "passed": int(n_gap == 0),
        "details": f"Scarto relativo massimo {float(rel_gap.max()):.3e}, "
                   f"tolleranza 1e-12."
                   if len(rel_gap) else "",
    })

    return pd.DataFrame(rows)


def estimate_one_spec_logm(df: pd.DataFrame, spec: dict, variant: str):
    dependent = spec["dependent"]
    regressors = spec["regressors"]

    required = [dependent] + regressors

    work = df[["quote_date"] + required].copy()

    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=required).copy()
    work = work[np.isfinite(work[dependent])].copy()

    for col in regressors:
        work = work[np.isfinite(work[col])].copy()

    T = int(len(work))

    if T <= len(regressors) + 1:
        raise ValueError(
            f"Too few observations for {spec['specification']} {spec['form']} "
            f"{spec['gamma_source']}: n={T}, regressors={regressors}"
        )

    L = nw_lag_length(T)

    y = work[dependent].astype(float)
    X = work[regressors].astype(float)
    X = sm.add_constant(X, has_constant="add")

    fitted = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True,
    )

    rows = []

    for term in fitted.params.index:
        rows.append({
            "variant": variant,
            "specification": spec["specification"],
            "form": spec["form"],
            "gamma_source": spec["gamma_source"],
            "term": term,
            "coef": float(fitted.params[term]),
            "se_nw": float(fitted.bse[term]),
            "t_stat": float(fitted.tvalues[term]),
            "p_value": float(fitted.pvalues[term]),
            "r_squared": float(fitted.rsquared),
            "adj_r_squared": float(fitted.rsquared_adj),
            "n": T,
            "nw_lag_L": L,
        })

    spec_summary = {
        "variant": variant,
        "specification": spec["specification"],
        "form": spec["form"],
        "gamma_source": spec["gamma_source"],
        "dependent": dependent,
        "regressors": ";".join(regressors),
        "n": T,
        "k_including_intercept": len(regressors) + 1,
        "nw_lag_L": L,
        "first_date_used": work["quote_date"].iloc[0],
        "last_date_used": work["quote_date"].iloc[-1],
    }

    return pd.DataFrame(rows), spec_summary


def estimate_all_specs_logm(df: pd.DataFrame, specs, variant: str):
    result_frames = []
    summary_rows = []

    for spec in specs:
        print(f"Estimating {spec['specification']} | form={spec['form']} | "
              f"gamma_source={spec['gamma_source']}")

        result, summary = estimate_one_spec_logm(df, spec, variant)

        print(f"    n={summary['n']}, k={summary['k_including_intercept']}, "
              f"Newey-West L={summary['nw_lag_L']} from floor(4*(T/100)^(2/9))")

        result_frames.append(result)
        summary_rows.append(summary)

    return pd.concat(result_frames, ignore_index=True), pd.DataFrame(summary_rows)


def daily_regressions_logm(variant: str):
    if variant not in {"logm_005", "logm_002"}:
        raise ValueError("VARIANT_LABEL deve essere 'logm_005' o 'logm_002'.")

    panel_file = DATA_DIR / PANEL_FOLDER / f"07_daily_regression_panel_2025_{variant}.csv"
    results_output_file = OUTPUT_DIR / f"13_daily_regression_results_2025_{variant}.csv"
    preflight_output_file = OUTPUT_DIR / f"13_preflight_validation_2025_{variant}.csv"
    a1_handcheck_output_file = OUTPUT_DIR / f"13_A1_handcheck_2025_{variant}.csv"
    spec_summary_output_file = OUTPUT_DIR / f"13_specification_summary_2025_{variant}.csv"
    mz_output_file = OUTPUT_DIR / f"13_mincer_zarnowitz_2025_{variant}.csv"

    print("=" * 130)
    print(f"SECTION 3.7 DAILY REGRESSIONS - VARIANTE {variant}")
    print("=" * 130)
    print(f"DATA_DIR:       {DATA_DIR}")
    print(f"Panel file:     {panel_file}")
    print(f"Output folder:  {OUTPUT_DIR}")
    print(f"Variante:       {variant}, regola a MONEYNESS e non quella sui bid")
    print("=" * 130)

    print("Loading daily regression panel...")
    panel = load_panel_logm(panel_file)

    print(f"Panel rows loaded: {len(panel):,}")
    print(f"Panel days loaded: {panel['quote_date'].nunique():,}")

    print()
    print("=" * 130)
    print("STEP 1 - PREFLIGHT VALIDATION")
    print("=" * 130)

    preflight = run_preflight_validation_logm(panel)
    safe_write_csv(preflight, preflight_output_file)

    print(preflight.to_string(index=False))

    if not preflight_passed(preflight):
        print()
        print("=" * 130)
        print("STOPPING: PREFLIGHT VALIDATION FAILED")
        print("=" * 130)
        print("No regressions were estimated.")
        print(f"Preflight output saved: {preflight_output_file}")
        print("=" * 130)
        raise RuntimeError(f"Parte 2, variante {variant}: preflight validation failed, "
                           "no regressions estimated.")

    print()
    print("Preflight validation passed. Proceeding to estimation.")

    print()
    print("=" * 130)
    print("STEP 2-4 - LEVELS AND LOG SPECIFICATIONS")
    print("=" * 130)

    panel = add_log_variables(panel)
    specifications = build_specifications()

    print(f"Number of specifications to estimate: {len(specifications)}")

    print(f"Attese: {EXPECTED_N_SPECIFICATIONS} = 6 su MFIV_session_t "
          "+ 6 su MFIV_ann_t + 16 sulle altre.")

    if len(specifications) != EXPECTED_N_SPECIFICATIONS:
        raise ValueError(
            f"Attese {EXPECTED_N_SPECIFICATIONS} specifiche, costruite "
            f"{len(specifications)}. Se il numero cambia apposta, aggiorna anche "
            "EXPECTED_N_SPECIFICATIONS."
        )

    results, spec_summary = estimate_all_specs_logm(panel, specifications, variant)

    print()
    print("=" * 130)
    print("STEP 7 - A1 CLOSED-FORM AND P-VALUE HAND CHECK")
    print("=" * 130)

    a1_handcheck = run_A1_handcheck(panel, results)

    print()
    print(a1_handcheck.to_string(index=False))

    if not bool((a1_handcheck["passed"] == 1).all()):
        print()
        print("=" * 130)
        print("WARNING: A1 HAND CHECK FAILED")
        print("=" * 130)
        print("Results are still saved for inspection, but should not be trusted "
              "until resolved.")

    safe_write_csv(results, results_output_file)
    safe_write_csv(spec_summary, spec_summary_output_file)
    safe_write_csv(a1_handcheck, a1_handcheck_output_file)

    print()
    print("=" * 130)
    print(f"COMPLETED SECTION 3.7 DAILY REGRESSIONS - {variant}")
    print("=" * 130)
    print(f"Results output:       {results_output_file}")
    print(f"Spec summary output:  {spec_summary_output_file}")
    print(f"Preflight output:     {preflight_output_file}")
    print(f"A1 handcheck output:  {a1_handcheck_output_file}")

    print()
    print("--- COEFFICIENTI PRINCIPALI, FORMA IN LIVELLI ---")
    main_lev = results[
        (results["form"] == "levels")
        & (results["term"].isin(["MFIV_t", "MFIV_ann_t", "MFIV_session_t", "MFIV_0931",
                                 "H_OI_t", "H_VOL_t",
                                 "H_OI_t_indep", "H_VOL_t_indep"]))
    ][["specification", "gamma_source", "term", "coef", "se_nw", "t_stat",
       "p_value", "r_squared", "n"]]
    print(main_lev.to_string(index=False))

    print()
    print("--- COEFFICIENTI PRINCIPALI, FORMA LOG ---")
    main_log = results[
        (results["form"] == "log")
        & (results["term"].isin(["log_MFIV_t", "log_MFIV_ann_t", "log_MFIV_session_t",
                                 "log_MFIV_0931",
                                 "H_OI_t", "H_VOL_t",
                                 "H_OI_t_indep", "H_VOL_t_indep"]))
    ][["specification", "gamma_source", "term", "coef", "se_nw", "t_stat",
       "p_value", "r_squared", "n"]]
    print(main_log.to_string(index=False))

    print()
    print("--- TEST DI NON DISTORSIONE (MINCER-ZARNOWITZ) ---")
    print(f"Variante {variant}. H0 congiunta: intercetta 0 e slope 1, "
          "su MFIV_session_t.")
    print()
    print("PERCHE' SU QUESTA E NON SULLE ALTRE:")
    print("  lo slope atteso e' 1 solo per MFIV_session_t.")
    print("  MFIV_t: slope atteso circa 1,935, non 1, perche' e' la media di")
    print("    sessione della varianza RESIDUA contro la varianza dell'INTERA")
    print("    sessione. Fattore misurato sui dati, non assunto.")
    print("  MFIV_ann_t in livelli: slope atteso la lunghezza della sessione in")
    print("    anni, circa 7,4e-04.")
    print("  MFIV_ann_t in log: slope atteso 1, ma INTERCETTA attesa")
    print("    log(389/525600) = -7,2095, non zero. Il test (0,1) li' rifiuta")
    print("    l'intercetta per costruzione e produce F privi di contenuto.")
    print("  MFIV_session_t = MFIV_ann_t * tau_apertura: benchmark (0,1) pulito in")
    print("    entrambe le forme, che e' lo standard di letteratura.")
    print()

    mz_rows = []

    for form, dep, reg in (("livelli", "RV_t", "MFIV_session_t"),
                           ("log", "log_RV_t", "log_MFIV_session_t")):
        _w = panel[["quote_date", dep, reg]].dropna()
        _y = _w[dep].astype(float)
        _X = sm.add_constant(_w[reg].astype(float), has_constant="add")
        _L = nw_lag_length(len(_w))
        _f = sm.OLS(_y, _X).fit(cov_type="HAC", cov_kwds={"maxlags": _L}, use_t=True)
        _test = _f.f_test("const = 0, " + reg + " = 1")

        _b = _f.params[reg]
        _sb = _f.bse[reg]
        _t_slope = (_b - 1.0) / _sb if _sb > 0 else np.nan

        _F = float(np.squeeze(_test.fvalue))
        _p = float(np.squeeze(_test.pvalue))

        print(f"  {form}:")
        print(f"    a = {_f.params['const']:+.6e}  (se {_f.bse['const']:.6e})  "
              f"t contro 0 = {_f.tvalues['const']:+.3f}")
        print(f"    b = {_b:.6f}  (se {_sb:.6f})  t contro 1 = {_t_slope:+.3f}")
        print(f"    F congiunta = {_F:.4f}   p = {_p:.4f}   n = {len(_w)}   L = {_L}")

        # Come nella parte 1: si interpreta il valore adattato alla media del
        # regressore, non l'intercetta.
        _xbar = float(_w[reg].mean())
        _fit_at_mean = float(_f.params["const"] + _f.params[reg] * _xbar)
        _gap = _fit_at_mean - _xbar

        print(f"    media del regressore = {_xbar:+.6f}")
        print(f"    valore adattato alla media = {_fit_at_mean:+.6f}")

        _ratio = float(np.exp(_gap)) if form == "log" else np.nan

        if form == "log":
            print(f"    scarto = {_gap:+.6f}  =>  rapporto RV/MFIV = {_ratio:.4f}")
            print("    Sotto 1 significa varianza realizzata sotto quella")
            print("    implicita, cioe' premio al rischio di varianza POSITIVO.")
        else:
            print(f"    scarto = {_gap:+.6e}  (in livelli, stessa unita' di RV_t)")

        print()

        mz_rows.append({
            "variant": variant,
            "form": form,
            "intercept": float(_f.params["const"]),
            "slope": float(_b),
            "slope_se": float(_sb),
            "t_slope_vs_1": float(_t_slope),
            "F_joint": _F,
            "p_joint": _p,
            "rejected_at_5pct": int(_p < 0.05),
            "regressor_mean": _xbar,
            "fitted_at_mean": _fit_at_mean,
            "gap": _gap,
            "rv_over_mfiv_ratio": _ratio,
            "n": int(len(_w)),
            "nw_lag_L": int(_L),
        })

    mz = pd.DataFrame(mz_rows)
    safe_write_csv(mz, mz_output_file)

    print("  COME LEGGERE F CONTRO I DUE t. Se la F congiunta rifiuta mentre i due")
    print("  t individuali no, non e' una contraddizione: il rifiuto avviene lungo")
    print("  una combinazione di intercetta e slope, e la direzione informativa e'")
    print("  il valore adattato alla media stampato sopra, non i due coefficienti")
    print("  presi uno alla volta.")
    print()
    print("  CONFRONTO CON LA BASELINE. Sulla regola dei bid la baseline da'")
    print("  F = 4,5177 in livelli (p = 0,0118) e F = 31,5152 in log (p < 0,0001),")
    print("  con rapporto RV/MFIV di 0,7943 in log. Se i numeri qui sopra")
    print("  rifiutano anch'essi, il verdetto di H1 non dipende dalla regola di")
    print("  troncamento. Se non rifiutassero, la distorsione sarebbe un effetto")
    print("  della regola e andrebbe riscritta come tale.")
    print()
    print("  In ogni caso un rifiuto va verificato con l'analisi di influenza")
    print("  prima di essere riportato: in livelli il risultato dipende")
    print("  dall'episodio di aprile, e quel controllo non e'")
    print("  ripetuto su questi coefficienti.")

    print()
    print("SPECIFICATION SUMMARY")
    print("=" * 130)
    print(spec_summary.to_string(index=False))
    print("=" * 130)


# =====================================================================
# COSTANTI COMUNI ALLE PARTI 3 E 4
# =====================================================================

REGRESSORS = ["MFIV_t", "MFIV_ann_t", "MFIV_session_t"]
FORMS = ["levels", "log"]
SAMPLES = ["full", "no_april"]
SOURCES = {"vendor": ("H_OI_t", "H_VOL_t"), "black76": ("H_OI_t_indep", "H_VOL_t_indep")}
N_APRIL_EXPECTED = N_APRIL_SESSIONS


def nw_lag(n: int) -> int:
    """Lag di Newey-West delle parti 3 e 4 (stesso valore di nw_lag_length)."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


# =====================================================================
# PARTE 3: A4 SENZA APRILE
# =====================================================================
# A4 (A5 della tesi): RV_t = a + b1 X_t + b2 RV_lag1 + b3 H_OI_t + b4 H_VOL_t + e, con
# X_t in {MFIV_t, MFIV_ann_t, MFIV_session_t}; nella forma log RV_t, X_t e RV_lag1 sono
# in log, H_OI_t e H_VOL_t no. 3 regressori x 2 forme x 2 campioni (intero e senza le
# 21 sessioni di aprile) x 2 fonti di gamma = 24 regressioni.
# RV_lag1 resta la colonna del pannello: senza aprile, la prima sessione di maggio ha
# come ritardo il 30 aprile.
#
# Le stime devono prima riprodurre HARD_TARGETS e HARD_R2 (fra cui la colonna A5 delle
# Tab. D.3 e D.4 della tesi), altrimenti lo script si ferma. Poi le celle di decisione
# (H_VOL in livelli senza aprile, con MFIV_t e MFIV_ann_t) si classificano con le regole
# R1-R3 sulle due fonti di gamma; le altre celle si riportano senza classificarle.

# md5 del pannello atteso: un valore diverso e' segnalato nei verdetti, senza fermare lo script.
PANEL_MD5 = "54e0ce24d75aaef318d85f73fc88cc5b"
OUT_A4 = OUTPUT_DIR / "13_a4_without_april.csv"
OUT_A4_VERDICTS = OUTPUT_DIR / "13_a4_without_april_verdicts.txt"

# ---------------------------------------------------------------- obiettivi della riproduzione
# (regressor, form, source, sample, term) -> dict(coef=..., se=..., p=...) come stampati.
HARD_TARGETS = {
    ("MFIV_t", "levels", "vendor", "full", "X"): {"coef": "3.68", "se": "0.281"},
    ("MFIV_t", "levels", "vendor", "full", "RV_lag1"): {"coef": "-0.294", "se": "0.0377"},
    ("MFIV_t", "levels", "vendor", "full", "H_OI"): {"coef": "-1.02e-4", "se": "1.22e-4"},
    ("MFIV_t", "levels", "vendor", "full", "H_VOL"): {"coef": "7.04e-4", "se": "2.25e-4"},
    ("MFIV_t", "levels", "vendor", "full", "const"): {"coef": "-1.16e-4", "se": "2.97e-5"},
    ("MFIV_t", "log", "vendor", "full", "X"): {"coef": "0.841", "se": "0.133"},
    ("MFIV_t", "log", "vendor", "full", "RV_lag1"): {"coef": "9.91e-4", "se": "0.0439"},
    ("MFIV_t", "log", "vendor", "full", "H_OI"): {"coef": "-0.121", "se": "0.746"},
    ("MFIV_t", "log", "vendor", "full", "H_VOL"): {"coef": "-5.97", "se": "1.98"},
    ("MFIV_t", "log", "vendor", "full", "const"): {"coef": "-0.551", "se": "0.985"},
    ("MFIV_ann_t", "log", "vendor", "full", "H_VOL"): {"coef": "-5.423", "p": "0.027"},
    ("MFIV_session_t", "log", "vendor", "full", "H_VOL"): {"coef": "-4.370", "p": "0.057"},
    ("MFIV_ann_t", "levels", "vendor", "full", "H_VOL"): {"coef": "8.24e-4"},
    ("MFIV_session_t", "levels", "vendor", "full", "H_VOL"): {"coef": "8.27e-4"},
    ("MFIV_session_t", "levels", "vendor", "no_april", "H_VOL"): {"coef": "-1.48e-4", "p": "0.0036"},
    ("MFIV_session_t", "log", "vendor", "no_april", "H_VOL"): {"coef": "-8.94"},
}
HARD_R2 = {("MFIV_t", "levels", "vendor", "full"): ("0.923", 249),
           ("MFIV_t", "log", "vendor", "full"): ("0.905", 249)}


def printed_match(value, printed):
    """True se value, arrotondato, da' la stringa stampata (mezza unita' dell'ultima cifra)."""
    t = float(printed)
    mant = printed.lower().lstrip("+-").split("e")[0].replace(".", "").lstrip("0")
    sig = max(len(mant), 1)
    if t == 0:
        return abs(value) < 10 ** (-sig)
    half = 0.5 * 10 ** (math.floor(math.log10(abs(t))) - (sig - 1))
    return abs(value - t) <= half * 1.000001


def md5_of(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_panel_a4():
    panel_file = str(PANEL_FILE)
    if not os.path.isfile(panel_file):
        raise SystemExit("STOP: panel not found: " + panel_file)
    md5 = md5_of(panel_file)
    df = pd.read_csv(panel_file, keep_default_na=False, dtype=str)
    num = ["RV_t", "RV_lag1", "MFIV_t", "MFIV_ann_t", "MFIV_session_t",
           "H_OI_t", "H_VOL_t", "H_OI_t_indep", "H_VOL_t_indep"]
    missing = [c for c in num + ["quote_date"] if c not in df.columns]
    if missing:
        raise SystemExit("STOP: columns missing from the panel: " + ", ".join(missing))
    for c in num:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df = df.sort_values("quote_date").reset_index(drop=True)
    if len(df) != 250:
        raise SystemExit("STOP: the panel has %d rows, expected 250" % len(df))
    df["is_april"] = df["quote_date"].str.startswith("2025-04")
    n_april = int(df["is_april"].sum())
    if n_april != N_APRIL_EXPECTED:
        raise SystemExit("STOP: %d April sessions, expected %d" % (n_april, N_APRIL_EXPECTED))
    return df, panel_file, md5


def fit_one(df, regressor, form, source, sample):
    hoi, hvol = SOURCES[source]
    d = df if sample == "full" else df[~df["is_april"]]
    d = d[["quote_date", "RV_t", regressor, "RV_lag1", hoi, hvol]].dropna()
    y = d["RV_t"].to_numpy(dtype=float)
    X = pd.DataFrame({"X": d[regressor].to_numpy(dtype=float),
                      "RV_lag1": d["RV_lag1"].to_numpy(dtype=float),
                      "H_OI": d[hoi].to_numpy(dtype=float),
                      "H_VOL": d[hvol].to_numpy(dtype=float)})
    if form == "log":
        for arr_name in ("X", "RV_lag1"):
            if (X[arr_name] <= 0).any():
                raise SystemExit("STOP: non-positive %s in the log form" % arr_name)
        if (y <= 0).any():
            raise SystemExit("STOP: non-positive RV_t in the log form")
        y = np.log(y)
        X["X"] = np.log(X["X"])
        X["RV_lag1"] = np.log(X["RV_lag1"])
    Xc = sm.add_constant(X, has_constant="add")
    n = len(y)
    lag = nw_lag(n)
    res = sm.OLS(y, Xc).fit(cov_type="HAC",
                            cov_kwds={"maxlags": lag, "use_correction": False},
                            use_t=True)
    ci = res.conf_int(alpha=0.05)
    # effetti per deviazione standard di H sulle sessioni del campione di stima
    mean_rv = float(d["RV_t"].mean())
    rows = []
    for term in ["const", "X", "RV_lag1", "H_OI", "H_VOL"]:
        b = float(res.params[term])
        se = float(res.bse[term])
        p = float(res.pvalues[term])
        eff = lo = hi = np.nan
        sd_h = np.nan
        if term in ("H_OI", "H_VOL"):
            sd_h = float(X[term].std(ddof=1))
            if form == "levels":
                f = lambda beta: 100.0 * beta * sd_h / mean_rv
            else:
                f = lambda beta: 100.0 * (math.exp(beta * sd_h) - 1.0)
            eff, a1, a2 = f(b), f(float(ci.loc[term, 0])), f(float(ci.loc[term, 1]))
            lo, hi = min(a1, a2), max(a1, a2)
        rows.append({"specification": "A4", "regressor": regressor, "form": form,
                     "gamma_source": source, "sample": sample, "term": term,
                     "coef": b, "se_nw": se, "t_stat": b / se if se > 0 else np.nan,
                     "p_value": p, "n": n, "nw_lag": lag, "r_squared": float(res.rsquared),
                     "sd_h": sd_h, "effect_pct_per_sd": eff,
                     "ci95_low_pct": lo, "ci95_high_pct": hi})
    return rows


def a4_without_april():
    df, panel_file, md5 = load_panel_a4()
    log = []
    log.append("13_Daily_regressions  panel: " + panel_file)
    log.append("md5 %s %s" % (md5, "(record)" if md5 == PANEL_MD5 else
                              "(WARNING: differs from the record %s)" % PANEL_MD5))

    rows = []
    for regressor in REGRESSORS:
        for form in FORMS:
            for source in SOURCES:
                for sample in SAMPLES:
                    rows.extend(fit_one(df, regressor, form, source, sample))
    out = pd.DataFrame(rows)

    # ------------------------------------------------------------ controllo di riproduzione
    failures = []
    for key, tgt in HARD_TARGETS.items():
        reg, form, src, smp, term = key
        r = out[(out.regressor == reg) & (out.form == form) & (out.gamma_source == src)
                & (out["sample"] == smp) & (out.term == term)].iloc[0]
        for field, col in (("coef", "coef"), ("se", "se_nw"), ("p", "p_value")):
            if field in tgt:
                ok = printed_match(float(r[col]), tgt[field])
                log.append("  repro %-45s %-5s %-10s got %.6g  %s" % (
                    "/".join(key), field, tgt[field], float(r[col]), "ok" if ok else "FAIL"))
                if not ok:
                    failures.append((key, field))
    for key, (r2, n) in HARD_R2.items():
        reg, form, src, smp = key
        r = out[(out.regressor == reg) & (out.form == form) & (out.gamma_source == src)
                & (out["sample"] == smp)].iloc[0]
        ok = printed_match(float(r["r_squared"]), r2) and int(r["n"]) == n
        log.append("  repro %-45s R2/n  %s/%d got %.4f/%d  %s" % (
            "/".join(key), r2, n, float(r["r_squared"]), int(r["n"]), "ok" if ok else "FAIL"))
        if not ok:
            failures.append((key, "r2/n"))
    n_noapr = int(out[(out["sample"] == "no_april")]["n"].iloc[0])
    log.append("  sample without April: n = %d (expected 228), NW lag %d" % (
        n_noapr, nw_lag(n_noapr)))
    if n_noapr != 228:
        failures.append(("no_april", "n"))

    if failures:
        print("\n".join(log))
        raise SystemExit("STOP: reproduction failed on %d items; nothing new is read." % len(failures))

    # ------------------------------------------------------------ celle di decisione
    log.append("")
    log.append("DECISION CELLS: H_VOL, levels, without April (vendor | Black '76)")
    for reg in ["MFIV_t", "MFIV_ann_t", "MFIV_session_t"]:
        cells = {}
        for src in SOURCES:
            r = out[(out.regressor == reg) & (out.form == "levels") & (out.gamma_source == src)
                    & (out["sample"] == "no_april") & (out.term == "H_VOL")].iloc[0]
            cells[src] = (float(r["coef"]), float(r["p_value"]))
        signs = [np.sign(c) for c, _ in cells.values()]
        if all(s < 0 for s in signs) and all(p < 0.05 for _, p in cells.values()):
            verdict = "R1 sign and significance attributable to April"
        elif all(s < 0 for s in signs):
            verdict = "R2 sign attributable to April, significance not"
        elif len(set(signs)) > 1:
            verdict = "R3 attribution fails (the two sources disagree in sign)"
        else:
            verdict = "R3 attribution fails (positive on both sources)"
        tag = "decision" if reg != "MFIV_session_t" else "known, check"
        log.append("  %-15s vendor %+.3e (p %.4f) | Black76 %+.3e (p %.4f)  -> %s [%s]" % (
            reg, cells["vendor"][0], cells["vendor"][1], cells["black76"][0],
            cells["black76"][1], verdict, tag))

    log.append("")
    log.append("REPORTED, NOT CLASSIFIED: H_VOL and H_OI, every cell")
    for term in ["H_VOL", "H_OI"]:
        for form in FORMS:
            for smp in SAMPLES:
                for reg in REGRESSORS:
                    parts = []
                    for src in SOURCES:
                        r = out[(out.regressor == reg) & (out.form == form)
                                & (out.gamma_source == src) & (out["sample"] == smp)
                                & (out.term == term)].iloc[0]
                        parts.append("%s %+.3e (p %.4f; %+.1f%% per sd)" % (
                            src, float(r["coef"]), float(r["p_value"]),
                            float(r["effect_pct_per_sd"])))
                    log.append("  %-5s %-6s %-8s %-15s %s" % (term, form, smp, reg,
                                                                " | ".join(parts)))

    out.to_csv(OUT_A4, index=False)
    with open(OUT_A4_VERDICTS, "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    print("\nWritten: " + str(OUTPUT_DIR))


# =====================================================================
# PARTE 4: SPECIFICAZIONE GIORNALIERA CON IL SOLO H_VOL
# =====================================================================
# Specificazioni con i nomi della tesi, X_d in {MFIV_t, MFIV_ann_t, MFIV_session_t}:
#   A3:  RV_d = a + b1 X_d + b2 RV_{d-1} + b3 H_OI_d
#   A4:  RV_d = a + b1 X_d + b2 RV_{d-1} + b3 H_VOL_d
#   A5:  RV_d = a + b1 X_d + b2 RV_{d-1} + b3 H_OI_d + b4 H_VOL_d
# Forme, fonti di gamma, campioni (249 e 228 sedute) e stimatore come nella parte 3:
# 72 regressioni. A3 e A5 devono riprodurre gli output delle parti 1 e 3 (tolleranza
# relativa 1e-9), altrimenti lo script si ferma.

PANEL = (PANEL_FOLDER, "07_daily_regression_panel_2025.csv")
RESULTS_27 = (OUTPUT_FOLDER, "13_daily_regression_results_2025.csv")   # parte 1
RESULTS_46 = (OUTPUT_FOLDER, "13_a4_without_april.csv")                # parte 3

SPECS = {"A3": ["H_OI"], "A4": ["H_VOL"], "A5": ["H_OI", "H_VOL"]}
REL_TOL = 1e-9


def existing(pair):
    """Percorso di un file di CBOE/Data che deve esistere."""
    p = path(pair)
    require(p)
    return p


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def load_panel_hvol() -> pd.DataFrame:
    df = pd.read_csv(existing(PANEL), keep_default_na=False, dtype=str)
    num = ["RV_t", "RV_lag1"] + REGRESSORS + [c for pair in SOURCES.values() for c in pair]
    missing = [c for c in num + ["quote_date"] if c not in df.columns]
    if missing:
        raise ValueError("Colonne mancanti nel panel: " + ", ".join(missing))
    for c in num:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    df = df.sort_values("quote_date").reset_index(drop=True)
    if len(df) != 250 or not df["quote_date"].is_unique:
        raise ValueError(f"Il panel ha {len(df)} righe (attese 250) o date ripetute")
    df["is_april"] = df["quote_date"].str.startswith("2025-04")
    if int(df["is_april"].sum()) != N_APRIL_EXPECTED:
        raise ValueError(f"{int(df['is_april'].sum())} sedute di aprile, attese {N_APRIL_EXPECTED}")
    return df


def estimate(panel, spec, regressor, form, source, sample):
    h_cols = dict(zip(["H_OI", "H_VOL"], SOURCES[source]))
    terms = SPECS[spec]
    d = panel if sample == "full" else panel[~panel["is_april"]]
    need = ["RV_t", regressor, "RV_lag1"] + [h_cols[t] for t in terms]
    d = d.dropna(subset=need)
    y = d["RV_t"].astype(float).copy()
    X = pd.DataFrame({"X": d[regressor].astype(float), "RV_lag1": d["RV_lag1"].astype(float)})
    for t in terms:
        X[t] = d[h_cols[t]].astype(float)
    if form == "log":
        if (y <= 0).any() or (X["X"] <= 0).any() or (X["RV_lag1"] <= 0).any():
            raise ValueError("valori non positivi nella forma in log")
        y = np.log(y)
        X["X"] = np.log(X["X"])
        X["RV_lag1"] = np.log(X["RV_lag1"])
    n = len(y)
    lag = nw_lag(n)
    res = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}, use_t=True)
    ci = res.conf_int(alpha=0.05)
    mean_rv = float(d["RV_t"].mean())
    rows = []
    for term in res.params.index:
        b, se, p = float(res.params[term]), float(res.bse[term]), float(res.pvalues[term])
        sd_h = eff = lo = hi = np.nan
        if term in ("H_OI", "H_VOL"):
            sd_h = float(X[term].std(ddof=1))
            if form == "levels":
                f = lambda beta: 100.0 * beta * sd_h / mean_rv
            else:
                f = lambda beta: 100.0 * (math.exp(beta * sd_h) - 1.0)
            eff = f(b)
            a1, a2 = f(float(ci.loc[term, 0])), f(float(ci.loc[term, 1]))
            lo, hi = min(a1, a2), max(a1, a2)
        rows.append({"specification": spec, "regressor": regressor, "form": form,
                     "gamma_source": source, "sample": sample, "term": term,
                     "coef": b, "se_nw": se, "t_stat": b / se if se > 0 else np.nan,
                     "p_value": p, "n": n, "nw_lag": lag,
                     "r_squared": float(res.rsquared), "adj_r_squared": float(res.rsquared_adj),
                     "sd_h": sd_h, "effect_pct_per_sd": eff,
                     "ci95_low_pct": lo, "ci95_high_pct": hi})
    return rows


def reproduction_check(res: pd.DataFrame) -> pd.DataFrame:
    s27 = pd.read_csv(existing(RESULTS_27))
    s46 = pd.read_csv(existing(RESULTS_46))
    out = []
    # Parte 1 (fonte "27"): A5 e A6 degli output sono A3 e A5 della tesi; MFIV_t, campione intero
    term27 = {"X": {"levels": "MFIV_t", "log": "log_MFIV_t"},
              "RV_lag1": {"levels": "RV_lag1", "log": "log_RV_lag1"}}
    for thesis_spec, reg_spec in [("A3", "A5"), ("A5", "A6")]:
        for form in FORMS:
            for source, src27 in [("vendor", "CBOE"), ("black76", "indep")]:
                mine = res[(res.specification == thesis_spec) & (res.regressor == "MFIV_t") &
                           (res.form == form) & (res.gamma_source == source) & (res["sample"] == "full")]
                ref = s27[(s27.specification == reg_spec) & (s27.form == form) & (s27.gamma_source == src27)]
                for _, r in mine.iterrows():
                    if r.term == "const":
                        name = "const"
                    elif r.term in term27:
                        name = term27[r.term][form]
                    else:
                        name = SOURCES[source][0 if r.term == "H_OI" else 1]
                    q = ref[ref.term == name]
                    if len(q) != 1:
                        raise ValueError(f"parte 1: riga {reg_spec} {form} {src27} {name} assente o doppia")
                    q = q.iloc[0]
                    for col, col27 in [("coef", "coef"), ("se_nw", "se_nw"), ("p_value", "p_value"), ("r_squared", "r_squared")]:
                        out.append({"fonte": "27", "specifica_tesi": thesis_spec, "regressor": "MFIV_t",
                                    "form": form, "gamma_source": source, "sample": "full",
                                    "term": r.term, "statistica": col,
                                    "valore_56": float(r[col]), "valore_salvato": float(q[col27]),
                                    "diff_rel": rel_diff(float(r[col]), float(q[col27]))})
    # Parte 3 (fonte "46"): le 24 regressioni di A5 ("A4" nell'output della parte 3)
    for _, q in s46.iterrows():
        mine = res[(res.specification == "A5") & (res.regressor == q.regressor) & (res.form == q.form) &
                   (res.gamma_source == q.gamma_source) & (res["sample"] == q["sample"]) & (res.term == q.term)]
        if len(mine) != 1:
            raise ValueError(f"parte 3: riga {q.regressor} {q.form} {q.gamma_source} {q['sample']} {q.term} senza corrispondente")
        m = mine.iloc[0]
        for col in ["coef", "se_nw"]:
            out.append({"fonte": "46", "specifica_tesi": "A5", "regressor": q.regressor, "form": q.form,
                        "gamma_source": q.gamma_source, "sample": q["sample"], "term": q.term,
                        "statistica": col, "valore_56": float(m[col]), "valore_salvato": float(q[col]),
                        "diff_rel": rel_diff(float(m[col]), float(q[col]))})
    chk = pd.DataFrame(out)
    chk["esito"] = np.where(chk["diff_rel"] <= REL_TOL, "OK", "DIVERSO")
    return chk


# ---------------------------------------------------------------- formato LaTeX delle Tab. D.3 e D.4
def fmt_number(x: float) -> str:
    """Tre cifre significative; notazione scientifica sotto 1e-3, come nella tesi."""
    if x == 0:
        return "0"
    if abs(x) < 1e-3:
        mant, exp = f"{x:.2e}".split("e")
        return f"{mant}\\times10^{{{int(exp)}}}"
    digits = 3 - int(math.floor(math.log10(abs(x)))) - 1
    return f"{x:.{max(digits, 0)}f}"


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(coef: float, p: float) -> str:
    s = stars(p)
    return f"${fmt_number(coef)}$" + (f"\\textsuperscript{{{s}}}" if s else "")


def table_column(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for form, table in [("levels", "D.3"), ("log", "D.4")]:
        sel = res[(res.specification == "A4") & (res.regressor == "MFIV_t") & (res.form == form) &
                  (res.gamma_source == "vendor") & (res["sample"] == "full")].set_index("term")
        for term, label in [("X", "MFIV_d"), ("RV_lag1", "RV_{d-1}"), ("H_VOL", "H^{VOL}_d"), ("const", "Constant")]:
            r = sel.loc[term]
            rows.append({"tabella": table, "riga": label, "coef": r.coef, "se": r.se_nw, "p": r.p_value,
                         "latex_coef": cell(r.coef, r.p_value), "latex_se": f"(${fmt_number(r.se_nw)}$)"})
        r = sel.iloc[0]
        rows.append({"tabella": table, "riga": "n", "coef": r.n, "latex_coef": str(int(r.n))})
        rows.append({"tabella": table, "riga": "R2", "coef": r.r_squared, "latex_coef": f"{r.r_squared:.3f}"})
    return pd.DataFrame(rows)


def daily_hvol_only():
    panel = load_panel_hvol()
    rows = []
    for spec in SPECS:
        for regressor in REGRESSORS:
            for form in FORMS:
                for source in SOURCES:
                    for sample in SAMPLES:
                        rows += estimate(panel, spec, regressor, form, source, sample)
    res = pd.DataFrame(rows)
    expected_n = {"full": 249, "no_april": 228}
    for sample, n in expected_n.items():
        got = set(res.loc[res["sample"] == sample, "n"])
        if got != {n}:
            raise ValueError(f"campione {sample}: n = {sorted(got)}, atteso {n}")

    chk = reproduction_check(res)
    chk.to_csv(OUTPUT_DIR / "13_reproduction_check.csv", index=False)
    n_bad = int((chk.esito != "OK").sum())
    if n_bad:
        print(chk[chk.esito != "OK"].to_string())
        raise SystemExit(f"STOP: {n_bad} valori di A3/A5 non riproducono gli output delle parti 1 e 3")

    res.to_csv(OUTPUT_DIR / "13_daily_hvol_only_2025.csv", index=False)
    col = table_column(res)
    col.to_csv(OUTPUT_DIR / "13_table_D3_D4_column_A4.csv", index=False)

    h = res[res.term == "H_VOL"]
    t51 = h[(h.specification == "A4") & (h.regressor == "MFIV_t")][
        ["form", "gamma_source", "sample", "coef", "p_value", "effect_pct_per_sd", "ci95_low_pct", "ci95_high_pct"]]
    t51.to_csv(OUTPUT_DIR / "13_table_5_1_row_A4.csv", index=False)

    # riepilogo
    lines = ["13_Daily_regressions: specificazione giornaliera con il solo H_VOL (A4 nuova)",
             f"Controllo di riproduzione: {len(chk)} valori di A3/A5 contro 27 e 46, tutti entro {REL_TOL:g}", ""]
    r2 = res.drop_duplicates(["specification", "regressor", "form", "gamma_source", "sample"])
    lines.append("R^2, MFIV_t, campione intero, vendor:")
    for form in FORMS:
        base = pd.read_csv(existing(RESULTS_27))
        a2 = base[(base.specification == "A2") & (base.form == form)].r_squared.iloc[0]
        vals = {s: r2[(r2.specification == s) & (r2.regressor == "MFIV_t") & (r2.form == form) &
                      (r2.gamma_source == "vendor") & (r2["sample"] == "full")].r_squared.iloc[0] for s in SPECS}
        lines.append(f"  {form:6s} A2 {a2:.4f} | A3 {vals['A3']:.4f} | A4 {vals['A4']:.4f} (+{vals['A4'] - a2:.4f}) | "
                     f"A5 {vals['A5']:.4f} (+{vals['A5'] - a2:.4f})")
    lines.append("")
    lines.append("H_VOL: coefficiente, p, effetto per d.s. (%), A4 (solo H_VOL) contro A5 (con H_OI)")
    for form in FORMS:
        for sample in SAMPLES:
            for regressor in REGRESSORS:
                for source in SOURCES:
                    parts = []
                    for spec in ["A4", "A5"]:
                        r = h[(h.specification == spec) & (h.regressor == regressor) & (h.form == form) &
                              (h.gamma_source == source) & (h["sample"] == sample)].iloc[0]
                        parts.append(f"{spec} {r.coef:+.3g} p={r.p_value:.3g} {r.effect_pct_per_sd:+.1f}%")
                    lines.append(f"  {form:6s} {sample:8s} {regressor:15s} {source:7s} | " + " | ".join(parts))
    text = "\n".join(lines)
    (OUTPUT_DIR / "13_summary.txt").write_text(text + "\n")
    print(text)
    print("\nNuova colonna A4 delle Tab. D.3 e D.4 (vendor, MFIV_t, campione intero):")
    print(col[["tabella", "riga", "latex_coef", "latex_se"]].to_string(index=False))
    print(f"\nOutput in: {OUTPUT_DIR}")


def main():
    daily_regressions()

    # Parte 2: ogni variante gira anche se l'altra fallisce; gli errori si segnalano alla fine.
    failed = []
    for variant in VARIANTS:
        try:
            daily_regressions_logm(variant)
        except Exception:
            traceback.print_exc()
            failed.append(variant)
    if failed:
        raise SystemExit(f"Varianti con errori o controlli falliti: {failed}")

    a4_without_april()
    daily_hvol_only()


if __name__ == "__main__":
    main()
