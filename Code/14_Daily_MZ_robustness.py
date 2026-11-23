"""
Robustezza del disegno giornaliero e Mincer-Zarnowitz ex ante (Tab. 5.3 e 5.4, Fig. 5.1-5.3).

Parte 1: analisi di influenza (Cook, DFBETA, leave-one-out) e ristima per sottocampione.
Parte 2: sottocampioni per regime di volatilita' e varianza realizzata a 5 minuti.
Parte 3: diagnostica di B-short (H3) in livelli con la varianza a 5 minuti.
Parte 4: giornate FOMC e CPI.
Parte 5: medie di sessione del gamma su griglie a 1, 5, 15 e 30 minuti.
Parte 6: Mincer-Zarnowitz ex ante e su MFIV^ses, log-rapporto RV/MFIV, confronto con la tesi.
Negli output A5 e A6 corrispondono ad A3 e A5 della tesi.

Input:  CBOE/Data/ (07_daily_panel, 13_daily_regressions, 05_spx_minute_bars, 04_gamma_herfindahl)
Output: CBOE/Data/14_daily_mz_robustness/
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from common import (CPI_DATES, DATA_DIR, FOMC_DATES, N_APRIL_SESSIONS, YEAR as _YEAR,
                    clean_date_series, make_log_column, nw_lag_length, output_dir,
                    path, require, safe_write_csv, to_numeric_col)

YEAR = str(_YEAR)  # "2025": le date si filtrano per prefisso

OUTPUT_FOLDER = "14_daily_mz_robustness"
OUTPUT_DIR = output_dir(OUTPUT_FOLDER)

PANEL_FILE = DATA_DIR / "07_daily_panel" / "07_daily_regression_panel_2025.csv"
# Risultati di baseline: parte 1 di 13_Daily_regressions.py.
BASELINE_RESULTS_FILE = DATA_DIR / "13_daily_regressions" / "13_daily_regression_results_2025.csv"
SPX_DIR = DATA_DIR / "05_spx_minute_bars"
SPX_FILE_PATTERN = "05_SPX_{:02d}_RV.csv"


# =====================================================================
# PARTE 1: ANALISI DI INFLUENZA
# =====================================================================
# Due domande: il rifiuto della non distorsione su MFIV_session_t poggia sulle giornate
# di aprile? Il contrasto di segno di H_VOL fra livelli e log resiste quando escono le
# giornate piu' influenti? In livelli pesano di piu' le giornate ad alta varianza.

OUT_INFLUENCE = OUTPUT_DIR / "14_influence_diagnostics_by_day_2025.csv"
OUT_SUBSAMPLE = OUTPUT_DIR / "14_coefficients_by_subsample_2025.csv"
OUT_MZ_SUBSAMPLE = OUTPUT_DIR / "14_mincer_zarnowitz_by_subsample_2025.csv"
OUT_LOO = OUTPUT_DIR / "14_leave_one_out_range_2025.csv"

# Sottocampioni da confrontare col campione pieno.
SUBSAMPLES = [
    ("full", []),
    ("no_0407_0409", ["2025-04-07", "2025-04-09"]),
    ("no_april", None),          # None = tutto aprile, calcolato a runtime
    ("no_top5_cook", "COOK5"),   # le 5 con Cook piu' alta, per specifica
]


# Soglia convenzionale per la distanza di Cook.
def cook_threshold(n: int, k: int) -> float:
    return 4.0 / float(n)


def load_panel_influence() -> pd.DataFrame:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Pannello non trovato:\n{PANEL_FILE}")

    df = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)
    df["quote_date"] = clean_date_series(df["quote_date"])

    df = df[
        df["quote_date"].notna()
        & df["quote_date"].astype(str).str.startswith(YEAR)
    ].copy()

    df["_date_dt"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df = df.sort_values("_date_dt").drop(columns=["_date_dt"]).reset_index(drop=True)

    numeric = [
        "RV_t", "MFIV_t", "MFIV_ann_t", "MFIV_session_t", "MFIV_0931",
        "RV_lag1", "SpotRV_open", "FwdRV_short", "FwdRV_full",
        "H_OI_t", "H_VOL_t", "H_OI_t_indep", "H_VOL_t_indep",
    ]

    missing = [c for c in numeric if c not in df.columns]

    if missing:
        raise ValueError(f"Colonne mancanti nel pannello: {missing}")

    for col in numeric:
        df[col] = to_numeric_col(df, col)

    for col in ["RV_t", "MFIV_t", "MFIV_ann_t", "MFIV_session_t",
                "MFIV_0931", "RV_lag1", "SpotRV_open",
                "FwdRV_short", "FwdRV_full"]:
        make_log_column(df, col, f"log_{col}")

    return df


def specs_to_check():
    return [
        # il test di non distorsione
        {"name": "MZ_ses", "form": "levels",
         "dep": "RV_t", "reg": ["MFIV_session_t"], "focus": "MFIV_session_t"},
        {"name": "MZ_ses", "form": "log",
         "dep": "log_RV_t", "reg": ["log_MFIV_session_t"],
         "focus": "log_MFIV_session_t"},

        # il contrasto di segno sul gamma
        {"name": "A6_ses", "form": "levels", "dep": "RV_t",
         "reg": ["MFIV_session_t", "RV_lag1", "H_OI_t", "H_VOL_t"],
         "focus": "H_VOL_t"},
        {"name": "A6_ses", "form": "log", "dep": "log_RV_t",
         "reg": ["log_MFIV_session_t", "log_RV_lag1", "H_OI_t", "H_VOL_t"],
         "focus": "H_VOL_t"},
        {"name": "A5", "form": "levels", "dep": "RV_t",
         "reg": ["MFIV_t", "RV_lag1", "H_OI_t"], "focus": "H_OI_t"},
        {"name": "A5", "form": "log", "dep": "log_RV_t",
         "reg": ["log_MFIV_t", "log_RV_lag1", "H_OI_t"], "focus": "H_OI_t"},
    ]


def prepare(df, spec):
    cols = [spec["dep"]] + spec["reg"]
    w = df[["quote_date"] + cols].dropna(subset=cols).copy()

    for c in cols:
        w = w[np.isfinite(w[c])]

    return w.reset_index(drop=True)


def fit_hac(w, spec):
    y = w[spec["dep"]].astype(float)
    X = sm.add_constant(w[spec["reg"]].astype(float), has_constant="add")
    L = nw_lag_length(len(w))
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True), L


def influence_for_spec(w, spec) -> pd.DataFrame:
    y = w[spec["dep"]].astype(float)
    X = sm.add_constant(w[spec["reg"]].astype(float), has_constant="add")

    # L'influenza si calcola su OLS classico: Cook e DFBETA sono definiti
    # rispetto alla varianza OLS, non a quella HAC.
    ols = sm.OLS(y, X).fit()
    infl = ols.get_influence()

    n = len(w)
    k = X.shape[1]

    cooks = infl.cooks_distance[0]
    dfb = infl.dfbetas

    focus_idx = list(X.columns).index(spec["focus"])

    out = pd.DataFrame({
        "specification": spec["name"],
        "form": spec["form"],
        "focus_term": spec["focus"],
        "quote_date": w["quote_date"].values,
        "cooks_distance": cooks,
        "cook_threshold": cook_threshold(n, k),
        "above_threshold": (cooks > cook_threshold(n, k)).astype(int),
        "dfbeta_focus": dfb[:, focus_idx],
        "leverage": infl.hat_matrix_diag,
        "student_resid": infl.resid_studentized_external,
    })

    return out.sort_values("cooks_distance", ascending=False).reset_index(drop=True)


def leave_one_out_range(w, spec) -> dict:
    """Escursione del coefficiente focus togliendo una giornata alla volta."""
    base, _ = fit_hac(w, spec)
    b0 = float(base.params[spec["focus"]])

    vals = []

    for i in range(len(w)):
        sub = w.drop(index=i)
        f, _ = fit_hac(sub.reset_index(drop=True), spec)
        vals.append(float(f.params[spec["focus"]]))

    vals = np.array(vals)

    return {
        "specification": spec["name"],
        "form": spec["form"],
        "focus_term": spec["focus"],
        "coef_full": b0,
        "loo_min": float(vals.min()),
        "loo_max": float(vals.max()),
        "loo_range_pct_of_full": float(
            100.0 * (vals.max() - vals.min()) / abs(b0)) if b0 != 0 else np.nan,
        "loo_sign_flips": int(((vals > 0) != (b0 > 0)).sum()),
        "worst_day_min": str(w["quote_date"].iloc[int(vals.argmin())]),
        "worst_day_max": str(w["quote_date"].iloc[int(vals.argmax())]),
    }


def resolve_subsample(name, spec_dates, panel, influence_df, spec):
    if spec_dates == []:
        return []

    if spec_dates is None:
        return [d for d in panel["quote_date"].unique()
                if str(d).startswith("2025-04")]

    if spec_dates == "COOK5":
        sub = influence_df[
            (influence_df["specification"] == spec["name"])
            & (influence_df["form"] == spec["form"])
        ]
        return sub.nlargest(5, "cooks_distance")["quote_date"].astype(str).tolist()

    return spec_dates


def daily_influence():
    print("=" * 130)
    print("ANALISI DI INFLUENZA SULLE STIME RIFATTE")
    print("=" * 130)
    print(f"Pannello:       {PANEL_FILE}")
    print(f"Output:         {OUTPUT_DIR}")
    print("=" * 130)

    panel = load_panel_influence()
    print(f"Giornate: {len(panel)}")

    specs = specs_to_check()

    influence_frames = []
    loo_rows = []

    print()
    print("Diagnostici di influenza e leave-one-out...")

    for spec in specs:
        w = prepare(panel, spec)
        print(f"  {spec['name']:8s} {spec['form']:7s} focus={spec['focus']:20s} n={len(w)}",
              flush=True)

        influence_frames.append(influence_for_spec(w, spec))
        loo_rows.append(leave_one_out_range(w, spec))

    influence = pd.concat(influence_frames, ignore_index=True)
    loo = pd.DataFrame(loo_rows)

    safe_write_csv(influence, OUT_INFLUENCE)
    safe_write_csv(loo, OUT_LOO)

    # --- sottocampioni ---
    print()
    print("Ristima per sottocampione...")

    sub_rows = []
    mz_rows = []

    for spec in specs:
        w_full = prepare(panel, spec)

        for name, spec_dates in SUBSAMPLES:
            drop = resolve_subsample(name, spec_dates, panel, influence, spec)
            w = w_full[~w_full["quote_date"].isin(drop)].reset_index(drop=True)

            if len(w) < len(spec["reg"]) + 5:
                continue

            f, L = fit_hac(w, spec)

            sub_rows.append({
                "specification": spec["name"],
                "form": spec["form"],
                "subsample": name,
                "n_dropped": len(drop),
                "n": len(w),
                "focus_term": spec["focus"],
                "coef": float(f.params[spec["focus"]]),
                "se_nw": float(f.bse[spec["focus"]]),
                "t_stat": float(f.tvalues[spec["focus"]]),
                "p_value": float(f.pvalues[spec["focus"]]),
                "r_squared": float(f.rsquared),
                "nw_lag_L": L,
            })

            # test di non distorsione solo sulle specifiche MZ
            if spec["name"] == "MZ_ses":
                reg = spec["reg"][0]
                test = f.f_test(f"const = 0, {reg} = 1")
                xbar = float(w[reg].mean())
                fit_at_mean = float(f.params["const"] + f.params[reg] * xbar)

                mz_rows.append({
                    "form": spec["form"],
                    "subsample": name,
                    "n_dropped": len(drop),
                    "n": len(w),
                    "a": float(f.params["const"]),
                    "b": float(f.params[reg]),
                    "se_b": float(f.bse[reg]),
                    "t_b_vs_1": float((f.params[reg] - 1.0) / f.bse[reg]),
                    "F_joint": float(np.squeeze(test.fvalue)),
                    "p_joint": float(np.squeeze(test.pvalue)),
                    "rejected_at_5pct": int(float(np.squeeze(test.pvalue)) < 0.05),
                    "gap_at_mean": fit_at_mean - xbar,
                    "ratio_RV_over_MFIV": float(np.exp(fit_at_mean - xbar))
                        if spec["form"] == "log" else np.nan,
                })

    subsample = pd.DataFrame(sub_rows)
    mz = pd.DataFrame(mz_rows)

    safe_write_csv(subsample, OUT_SUBSAMPLE)
    safe_write_csv(mz, OUT_MZ_SUBSAMPLE)

    # --- stampa ---
    print()
    print("=" * 130)
    print("1. LE GIORNATE PIU' INFLUENTI, PER SPECIFICA")
    print("=" * 130)

    for spec in specs:
        sub = influence[
            (influence["specification"] == spec["name"])
            & (influence["form"] == spec["form"])
        ].head(5)

        print()
        print(f"{spec['name']} {spec['form']}, focus {spec['focus']}  "
              f"(soglia Cook {sub['cook_threshold'].iloc[0]:.4f})")
        print(sub[["quote_date", "cooks_distance", "dfbeta_focus",
                   "leverage", "student_resid"]].to_string(index=False))

    print()
    print("=" * 130)
    print("2. LEAVE-ONE-OUT: QUANTO SI MUOVE IL COEFFICIENTE")
    print("=" * 130)
    print(loo.to_string(index=False))
    print()
    print("loo_sign_flips > 0 significa che ESISTE una singola giornata la cui")
    print("rimozione cambia il segno del coefficiente. Un risultato che dipende da")
    print("una giornata non e' un risultato.")

    print()
    print("=" * 130)
    print("3. COEFFICIENTI PER SOTTOCAMPIONE")
    print("=" * 130)

    for spec in specs:
        sub = subsample[
            (subsample["specification"] == spec["name"])
            & (subsample["form"] == spec["form"])
        ]

        if sub.empty:
            continue

        print()
        print(f"{spec['name']} {spec['form']}, focus {spec['focus']}")
        print(sub[["subsample", "n_dropped", "n", "coef", "se_nw",
                   "t_stat", "p_value", "r_squared"]].to_string(index=False))

    print()
    print("=" * 130)
    print("4. TEST DI NON DISTORSIONE PER SOTTOCAMPIONE")
    print("=" * 130)
    print(mz.to_string(index=False))
    print()
    print("Se il rifiuto sopravvive a tutti i sottocampioni, e' reale.")
    print("Se cade togliendo aprile, e' aprile.")

    print()
    print("=" * 130)
    print(f"Output: {OUT_INFLUENCE.name}, {OUT_LOO.name}, "
          f"{OUT_SUBSAMPLE.name}, {OUT_MZ_SUBSAMPLE.name}")
    print("=" * 130)


# =====================================================================
# FUNZIONI E COSTANTI COMUNI ALLE PARTI 2-6
# =====================================================================
# build_specifications() ed estimate_one_spec() sono identiche a quelle della parte 1 di
# 13_Daily_regressions.py: nelle parti 2-5 cambia solo il campione o la variabile.

RV_COLS = ["RV_t", "RV_lag1", "SpotRV_open", "FwdRV_short", "FwdRV_full"]

NEEDED = ["RV_t", "MFIV_t", "MFIV_0931", "RV_lag1", "SpotRV_open", "FwdRV_short",
          "FwdRV_full", "H_OI_t", "H_VOL_t", "H_OI_t_indep", "H_VOL_t_indep",
          "MFIV_ann_t", "MFIV_ann_0931", "MFIV_session_t", "tau_open"]

LOG_VARS = ["RV_t", "MFIV_t", "RV_lag1", "MFIV_0931", "SpotRV_open", "FwdRV_short",
            "FwdRV_full", "MFIV_ann_t", "MFIV_ann_0931", "MFIV_session_t"]

REGIME_WINDOW = 5
REGIME_LABELS = ["calmo", "intermedio", "agitato"]
RV_K = 5

SESSION_START = 9 * 60 + 31
OPEN_WINDOW_END = 10 * 60
SESSION_END = 16 * 60

REPRO_TOL = 1e-9


def build_specifications():
    """Le 28 specificazioni della parte 1 di 13_Daily_regressions.py."""
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


def load_panel() -> pd.DataFrame:
    """Pannello delle parti 2 e 3, senza logaritmi."""
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Pannello non trovato:\n{PANEL_FILE}")
    df = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)
    df["quote_date"] = clean_date_series(df["quote_date"])
    df = df[df["quote_date"].notna() & df["quote_date"].astype(str).str.startswith(YEAR)].copy()
    df = df.sort_values("quote_date").reset_index(drop=True)
    miss = [c for c in NEEDED if c not in df.columns]
    if miss:
        raise ValueError(f"Colonne mancanti nel pannello: {miss}")
    for c in NEEDED:
        df[c] = to_numeric_col(df, c)
    return df


def load_panel_logs() -> pd.DataFrame:
    """Pannello delle parti 4 e 5, con i logaritmi di LOG_VARS."""
    df = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)
    df["quote_date"] = clean_date_series(df["quote_date"])
    df = df[df["quote_date"].notna() & df["quote_date"].astype(str).str.startswith(YEAR)].copy()
    df = df.sort_values("quote_date").reset_index(drop=True)
    miss = [c for c in NEEDED if c not in df.columns]
    if miss:
        raise ValueError(f"Colonne mancanti nel pannello: {miss}")
    for c in NEEDED:
        df[c] = to_numeric_col(df, c)
    for c in LOG_VARS:
        make_log_column(df, c, f"log_{c}")
    return df


def add_logs(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        make_log_column(out, c, f"log_{c}")
    return out


def run_specs(df: pd.DataFrame, specs, sample_label: str) -> pd.DataFrame:
    frames = []
    for s in specs:
        r, _ = estimate_one_spec(df, s)
        r["sample"] = sample_label
        frames.append(r)
    return pd.concat(frames, ignore_index=True)


def substitute(specs, mapping):
    out = []
    for s in specs:
        t = dict(s)
        t["dependent"] = mapping.get(s["dependent"], s["dependent"])
        t["regressors"] = [mapping.get(x, x) for x in s["regressors"]]
        out.append(t)
    return out


def mincer_zarnowitz(df: pd.DataFrame, dep: str, reg: str):
    w = df[[dep, reg]].apply(pd.to_numeric, errors="coerce").dropna()
    w = w[np.isfinite(w).all(axis=1)]
    y = w[dep].astype(float)
    X = sm.add_constant(w[reg].astype(float), has_constant="add")
    L = nw_lag_length(len(w))
    f = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True)
    test = f.f_test(f"const = 0, {reg} = 1")
    xbar = float(w[reg].mean())
    fit_mean = float(f.params["const"] + f.params[reg] * xbar)
    return {"n": len(w), "slope": float(f.params[reg]), "se_slope": float(f.bse[reg]),
            "F": float(np.squeeze(test.fvalue)), "p_F": float(np.squeeze(test.pvalue)),
            "ratio_at_mean": float(np.exp(fit_mean - xbar)) if dep.startswith("log_") else np.nan}


# ---------------------------------------------------------------- finestre di varianza dalla serie SPX
def load_spx() -> pd.DataFrame:
    frames = []
    for m in range(1, 13):
        f = SPX_DIR / SPX_FILE_PATTERN.format(m)
        if not f.exists():
            raise FileNotFoundError(f"File SPX mancante:\n{f}")
        frames.append(pd.read_csv(f, usecols=["Underlying", "Date", "time", "close"]))
    d = pd.concat(frames, ignore_index=True)
    d = d[d["Underlying"].astype(str).str.strip() == "SPX"].copy()
    parts = d["time"].astype(str).str.strip().str.split(":", expand=True)
    return pd.DataFrame({
        "date": pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d"),
        "minute": parts[0].astype(int) * 60 + parts[1].astype(int),
        "close": pd.to_numeric(d["close"], errors="coerce"),
    })


def window_rv(lp: np.ndarray, k: int) -> float:
    """Varianza su una finestra di prezzi in log: media su tutti i k punti di
    partenza, con primo e ultimo punto della finestra sempre inclusi."""
    n = len(lp)
    if n < 2:
        return np.nan
    if k == 1:
        return float(np.sum(np.diff(lp) ** 2))
    vals = []
    for o in range(k):
        pts = np.unique(np.concatenate(([0], np.arange(o, n, k), [n - 1])))
        vals.append(float(np.sum(np.diff(lp[pts]) ** 2)))
    return float(np.mean(vals))


def build_windows(spx: pd.DataFrame, k: int) -> pd.DataFrame:
    spx = spx[(spx["minute"] >= SESSION_START) & (spx["minute"] <= SESSION_END)
              & (spx["close"] > 0)]
    rows = []
    for date, g in spx.groupby("date", sort=True):
        g = g.sort_values("minute")
        last = int(g["minute"].max())
        s = g.set_index("minute")["close"].reindex(range(SESSION_START, last + 1)).ffill()
        lp = np.log(s.to_numpy())
        i10 = OPEN_WINDOW_END - SESSION_START
        rows.append({
            "quote_date": date,
            "RV_full": window_rv(lp, k),
            "RV_open": window_rv(lp[: i10 + 1], k),
            "RV_after10": window_rv(lp[i10:], k),
        })
    return pd.DataFrame(rows)


def attach_rv(panel: pd.DataFrame, w: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """RV_t, FwdRV_full = sessione intera; SpotRV_open = 09:31-10:00;
    FwdRV_short = 10:00-chiusura; RV_lag1 = RV_t della sessione precedente
    del pannello."""
    out = panel.merge(w, on="quote_date", how="left", validate="one_to_one")
    out[f"RV_t{suffix}"] = out["RV_full"]
    out[f"FwdRV_full{suffix}"] = out["RV_full"]
    out[f"SpotRV_open{suffix}"] = out["RV_open"]
    out[f"FwdRV_short{suffix}"] = out["RV_after10"]
    out[f"RV_lag1{suffix}"] = out[f"RV_t{suffix}"].shift(1)
    return out.drop(columns=["RV_full", "RV_open", "RV_after10"])


# =====================================================================
# PARTE 2: REGIMI DI VOLATILITA' E VARIANZA A 5 MINUTI
# =====================================================================
# Regime: terzile della RV media delle cinque sessioni precedenti (componente settimanale
# dell'HAR di Corsi, 2009), nota prima dell'apertura; le prime cinque sessioni restano senza
# regime. Nei terzili righe consecutive possono essere giornate lontane: Newey-West si
# applica al sottocampione ordinato nel tempo, con il lag calcolato sulla sua numerosita'.
# Varianza a 5 minuti: tutte le variabili di RV ricostruite a 5 minuti (media su tutti i
# punti di partenza), perche' al minuto il ritardo dell'indice la sottostima; la misura al
# minuto resta la principale.
# Controlli prima dei risultati: (a) le 28 specificazioni riproducono la baseline; (b) le
# finestre ricostruite al minuto riproducono il pannello, altrimenti la stima a 5 minuti si
# salta; (c) con quelle finestre le specificazioni ridanno la baseline.

STEM_REGIMES = "14_regime_and_rv5_robustness"


def hypotheses(res: pd.DataFrame) -> pd.DataFrame:
    """Lettura delle ipotesi preregistrate."""
    out = []
    for form in ("levels", "log"):
        lg = (lambda x: f"log_{x}") if form == "log" else (lambda x: x)
        def pick(spec, src, term):
            x = res[(res["specification"] == spec) & (res["form"] == form)
                    & (res["gamma_source"] == src) & (res["term"] == term)]
            return (float(x["coef"].iloc[0]), float(x["p_value"].iloc[0])) if len(x) else (np.nan, np.nan)
        rec = {"form": form}
        rec["H1_coef"], rec["H1_p"] = pick("A1", "NA", lg("MFIV_t"))
        rec["H2_coef"], rec["H2_p"] = pick("A2", "NA", lg("MFIV_t"))
        rec["H3_coef"], rec["H3_p"] = pick("B-short", "NA", lg("MFIV_0931"))
        c, p = pick("A5", "CBOE", "H_OI_t")
        rec["H4_coef"], rec["H4_p_one_sided_neg"] = c, (p / 2 if c < 0 else 1 - p / 2)
        ci, pi_ = pick("A5", "indep", "H_OI_t_indep")
        rec["H4_indep_p_one_sided_neg"] = pi_ / 2 if ci < 0 else 1 - pi_ / 2
        rec["H5_coef"], rec["H5_p"] = pick("A6", "CBOE", "H_VOL_t")
        rec["H5_indep_p"] = pick("A6", "indep", "H_VOL_t_indep")[1]
        rec["H6_HOI_p"] = pick("A6", "CBOE", "H_OI_t")[1]
        out.append(rec)
    return pd.DataFrame(out)


def regimes_rv5():
    print("=" * 140)
    print("SEZIONE 3.7: REGIMI DI VOLATILITA' E VARIANZA A 5 MINUTI")
    print("=" * 140)

    panel = add_logs(load_panel(), LOG_VARS)
    specs = build_specifications()
    print(f"Pannello: {PANEL_FILE}  ({len(panel)} giornate)   Specifiche: {len(specs)}")

    checks = []

    # (a) riproduzione della baseline sul campione intero
    full = run_specs(panel, specs, "intero")
    if BASELINE_RESULTS_FILE.exists():
        # keep_default_na=False: gamma_source contiene la stringa "NA", che pandas
        # altrimenti legge come mancante, perdendo le specificazioni senza gamma.
        b = pd.read_csv(BASELINE_RESULTS_FILE, keep_default_na=False)
        for c in ("coef", "se_nw"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
        keys = ["specification", "form", "gamma_source", "term"]
        m = full.merge(b, on=keys, how="left", suffixes=("", "_b"))
        miss = int(m["coef_b"].isna().sum())
        gc = float(((m["coef"] - m["coef_b"]).abs() / m["coef_b"].abs().clip(lower=1e-300)).max())
        gs = float(((m["se_nw"] - m["se_nw_b"]).abs() / m["se_nw_b"].abs().clip(lower=1e-300)).max())
        ok = miss == 0 and gc < REPRO_TOL and gs < REPRO_TOL
        checks.append(("(a) le 28 specifiche riproducono la baseline", ok,
                       f"{len(m)} coefficienti, {miss} senza corrispondenza, scarto coef {gc:.1e}, se {gs:.1e}"))
    else:
        checks.append(("(a) le 28 specifiche riproducono la baseline", False,
                       f"file non trovato: {BASELINE_RESULTS_FILE}"))

    # (b) finestre al minuto dalla serie SPX contro il pannello
    spx = load_spx()
    w1 = build_windows(spx, 1)
    p1 = attach_rv(panel, w1, "_rec1")
    bad = {}
    for c in RV_COLS:
        a, r = p1[c], p1[f"{c}_rec1"]
        both = a.notna() & r.notna()
        rel = (a[both] - r[both]).abs() / a[both].abs().clip(lower=1e-300)
        nmiss = int((a.notna() != r.notna()).sum())
        bad[c] = (int((rel > REPRO_TOL).sum()), nmiss, float(rel.max()) if len(rel) else np.nan,
                  p1.loc[both[both].index[rel > REPRO_TOL], "quote_date"].head(4).tolist() if (rel > REPRO_TOL).any() else [])
    okb = all(v[0] == 0 and v[1] == 0 for v in bad.values())
    checks.append(("(b) finestre al minuto ricostruite = colonne del pannello", okb,
                   "; ".join(f"{c}: {v[0]} diverse, {v[1]} mancanti, max {v[2]:.1e}"
                             + (f" es. {v[3]}" if v[3] else "") for c, v in bad.items())))

    # (c) le 28 specifiche con le finestre ricostruite ridanno la baseline
    map1 = {c: f"{c}_rec1" for c in RV_COLS}
    map1.update({f"log_{c}": f"log_{c}_rec1" for c in RV_COLS})
    p1 = add_logs(p1, [f"{c}_rec1" for c in RV_COLS])
    rec = run_specs(p1, substitute(specs, map1), "ricostruita_1min")
    mm = rec.merge(full, on=["specification", "form", "gamma_source"], suffixes=("", "_f"))
    mm = mm[mm["term"].str.replace("_rec1", "", regex=False) == mm["term_f"]]
    gc = float(((mm["coef"] - mm["coef_f"]).abs() / mm["coef_f"].abs().clip(lower=1e-300)).max())
    checks.append(("(c) con le finestre ricostruite le 28 specifiche ridanno la baseline",
                   gc < REPRO_TOL, f"{len(mm)} coefficienti, scarto massimo {gc:.1e}"))

    print()
    print("CONTROLLI")
    for name, ok, det in checks:
        print(f"  [{'OK' if ok else 'FALLITO'}] {name}: {det}")

    results = [full]
    mz_rows = []
    for form, dep, reg in (("levels", "RV_t", "MFIV_session_t"),
                           ("log", "log_RV_t", "log_MFIV_session_t")):
        mz_rows.append({"sample": "intero", "form": form, **mincer_zarnowitz(panel, dep, reg)})

    # ---------------------------------------------------------------- 1. regimi
    print()
    print("=" * 140)
    print("PARTE 1: SOTTOCAMPIONI PER REGIME DI VOLATILITA'")
    print("=" * 140)

    panel["regime_var"] = panel["RV_t"].rolling(REGIME_WINDOW).mean().shift(1)
    panel["regime"] = pd.qcut(panel["regime_var"], 3, labels=REGIME_LABELS)
    panel["regime"] = panel["regime"].astype(object)

    desc = panel.dropna(subset=["regime"]).groupby("regime").agg(
        giornate=("quote_date", "count"),
        di_aprile=("quote_date", lambda s: int(s.str.startswith("2025-04").sum())),
        RV_media=("RV_t", "mean"),
        soglia_min=("regime_var", "min"), soglia_max=("regime_var", "max"),
    ).reindex(REGIME_LABELS)
    print(f"Regime: media di RV_t sulle {REGIME_WINDOW} sessioni precedenti; "
          f"{int(panel['regime'].isna().sum())} sessioni iniziali senza finestra.")
    print(desc.to_string(float_format=lambda v: f"{v:.3e}"))

    for lab in REGIME_LABELS:
        sub = panel[panel["regime"] == lab].copy()
        results.append(run_specs(sub, specs, lab))
        for form, dep, reg in (("levels", "RV_t", "MFIV_session_t"),
                               ("log", "log_RV_t", "log_MFIV_session_t")):
            mz_rows.append({"sample": lab, "form": form, **mincer_zarnowitz(sub, dep, reg)})

    # ---------------------------------------------------------------- 2. varianza a 5 minuti
    print()
    print("=" * 140)
    print(f"PARTE 2: VARIANZA REALIZZATA A {RV_K} MINUTI")
    print("=" * 140)

    if not okb:
        print("Il controllo (b) e' fallito: le finestre ricostruite non coincidono con il")
        print("pannello, quindi la versione a 5 minuti non misurerebbe le stesse finestre.")
        print("Parte 2 saltata.")
    else:
        wk = build_windows(spx, RV_K)
        pk = attach_rv(panel, wk, "_5m")
        pk = add_logs(pk, [f"{c}_5m" for c in RV_COLS])
        mapk = {c: f"{c}_5m" for c in RV_COLS}
        mapk.update({f"log_{c}": f"log_{c}_5m" for c in RV_COLS})
        rk = run_specs(pk, substitute(specs, mapk), f"rv_{RV_K}min")
        rk["term"] = rk["term"].str.replace("_5m", "", regex=False)
        results.append(rk)
        for form, dep, reg in (("levels", "RV_t_5m", "MFIV_session_t"),
                               ("log", "log_RV_t_5m", "log_MFIV_session_t")):
            mz_rows.append({"sample": f"rv_{RV_K}min", "form": form,
                            **mincer_zarnowitz(pk, dep, reg)})
        ratio = (pk["RV_t_5m"] / pk["RV_t"]).describe()
        print(f"RV_t a {RV_K} minuti / al minuto: media {ratio['mean']:.4f}, mediana {ratio['50%']:.4f}")

    # ---------------------------------------------------------------- output e lettura
    allres = pd.concat(results, ignore_index=True)
    allres.to_csv(OUTPUT_DIR / f"{STEM_REGIMES}_results.csv", index=False)
    mz = pd.DataFrame(mz_rows)
    mz.to_csv(OUTPUT_DIR / f"{STEM_REGIMES}_mincer_zarnowitz.csv", index=False)

    hyp = []
    for lab in ["intero"] + REGIME_LABELS + [f"rv_{RV_K}min"]:
        r = allres[allres["sample"] == lab]
        if len(r):
            h = hypotheses(r)
            h.insert(0, "campione", lab)
            hyp.append(h)
    hyp = pd.concat(hyp, ignore_index=True)
    hyp.to_csv(OUTPUT_DIR / f"{STEM_REGIMES}_hypotheses.csv", index=False)

    print()
    print("=" * 140)
    print("IPOTESI PREREGISTRATE PER CAMPIONE")
    print("=" * 140)
    print("H1 A1, H2 A2 (primaria), H3 B-short (confermativa): MFIV positivo e significativo.")
    print("H4 A5: H_OI negativo, una coda. H5 A6: H_VOL, due code. H6 A6: entrambi significativi.")
    for form in ("levels", "log"):
        print()
        print(f"-- forma {form} --")
        x = hyp[hyp["form"] == form].drop(columns=["form"])
        print(x.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("=" * 140)
    print("MINCER-ZARNOWITZ SU MFIV_session_t PER CAMPIONE")
    print("=" * 140)
    print(mz.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("Limite: nei terzili Newey-West lavora su righe non consecutive (vedi intestazione).")
    print(f"Output: {STEM_REGIMES}_results.csv, _hypotheses.csv, _mincer_zarnowitz.csv")
    print("=" * 140)

    if not all(ok for _, ok, _ in checks):
        raise SystemExit("Parte 2: controlli FALLITI (vedi sopra).")


# =====================================================================
# PARTE 3: DIAGNOSTICA DI H3 (B-SHORT) IN LIVELLI CON LA VARIANZA A 5 MINUTI
# =====================================================================
# In livelli, con la varianza a 5 minuti, il coefficiente di MFIV_0931 in B-short
# (FwdRV_short ~ MFIV_0931 + SpotRV_open + RV_lag1) perde significativita'. Le tre
# variabili di varianza si stimano in tutte le 8 combinazioni al minuto e a 5 minuti,
# sul campione intero e senza aprile, per vedere quale sostituzione lo fa sparire.

STEM_BSHORT = "14_bshort_rv5_diagnostic"


def bshort_rv5_diagnostic():
    print("=" * 130)
    print("DIAGNOSTICA DI H3 (B-SHORT) IN LIVELLI: QUALE VARIABILE A 5 MINUTI LO FA CROLLARE")
    print("=" * 130)

    panel = load_panel()
    spx = load_spx()
    w1 = attach_rv(panel, build_windows(spx, 1), "_1m")
    w5 = attach_rv(panel, build_windows(spx, RV_K), "_5m")
    for c in RV_COLS:
        panel[f"{c}_1m"] = w1[f"{c}_1m"].values
        panel[f"{c}_5m"] = w5[f"{c}_5m"].values

    gap = float(((panel["FwdRV_short_1m"] - panel["FwdRV_short"]).abs()
                 / panel["FwdRV_short"].abs()).max())
    print(f"Controllo: FwdRV_short ricostruita al minuto = pannello, scarto {gap:.1e}")
    if gap > REPRO_TOL:
        print("Le finestre non coincidono con il pannello: diagnostica non affidabile.")
        raise SystemExit("Parte 3: le finestre ricostruite non coincidono con il pannello.")

    no_april = ~panel["quote_date"].str.startswith("2025-04")
    rows = []
    for dep_k in ("1m", "5m"):
        for spot_k in ("1m", "5m"):
            for lag_k in ("1m", "5m"):
                for label, mask in (("intero", pd.Series(True, index=panel.index)),
                                    ("senza aprile", no_april)):
                    spec = {"specification": "B-short", "form": "levels", "gamma_source": "NA",
                            "dependent": f"FwdRV_short_{dep_k}",
                            "regressors": ["MFIV_0931", f"SpotRV_open_{spot_k}", f"RV_lag1_{lag_k}"]}
                    r, _ = estimate_one_spec(panel[mask].copy(), spec)
                    g = r.set_index("term")
                    rows.append({
                        "dipendente": dep_k, "controllo_apertura": spot_k, "RV_lag1": lag_k,
                        "campione": label, "n": int(g["n"].iloc[0]),
                        "coef_MFIV_0931": g.loc["MFIV_0931", "coef"],
                        "p_MFIV_0931": g.loc["MFIV_0931", "p_value"],
                        "coef_SpotRV_open": g.loc[f"SpotRV_open_{spot_k}", "coef"],
                        "p_SpotRV_open": g.loc[f"SpotRV_open_{spot_k}", "p_value"],
                        "r_squared": g["r_squared"].iloc[0],
                    })

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / f"{STEM_BSHORT}.csv", index=False)

    print()
    print("B-SHORT IN LIVELLI: coefficiente di MFIV_0931 per combinazione")
    print("(1m = al minuto, 5m = a 5 minuti; la prima riga di ciascun campione e' la baseline)")
    for label in ("intero", "senza aprile"):
        print()
        print(f"-- campione {label} --")
        x = res[res["campione"] == label].drop(columns=["campione"])
        print(x.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("CORRELAZIONI (campione intero)")
    for k in ("1m", "5m"):
        c1 = panel[["MFIV_0931", f"SpotRV_open_{k}"]].corr().iloc[0, 1]
        c2 = panel[[f"SpotRV_open_{k}", "FwdRV_short_1m"]].corr().iloc[0, 1]
        c3 = panel[[f"SpotRV_open_{k}", "FwdRV_short_5m"]].corr().iloc[0, 1]
        print(f"  SpotRV_open {k}: con MFIV_0931 {c1:.3f}; con FwdRV_short 1m {c2:.3f}, 5m {c3:.3f}")

    print()
    print("COME SI LEGGE. Si guarda in quali righe il coefficiente di MFIV_0931 perde")
    print("significativita'. Se accade solo quando il controllo d'apertura e' a 5 minuti,")
    print("il crollo viene dal controllo; se accade quando la dipendente e' a 5 minuti,")
    print("viene dalla varianza da spiegare. Il confronto con il campione senza aprile")
    print("dice se il fenomeno e' di quelle settimane.")
    print("=" * 130)


# =====================================================================
# PARTE 4: GIORNATE FOMC E CPI
# =====================================================================
# Le giornate FOMC (comunicato alle 14:00) e CPI (rilascio alle 8:30, prima dello snapshot
# delle 09:31) restano nel campione e si esaminano a parte. Date in common.py.
#   1. realizzata contro adattata per tipo di giornata; in livelli rapporto delle somme,
#      perche' l'adattato di un singolo giorno puo' essere vicino a zero o negativo;
#   2. indicatori FOMC e CPI aggiunti alle specificazioni;
#   3. RV_t / MFIV_session_t per tipo di giornata;
#   4. mattina calma nei giorni FOMC: varianza della prima mezz'ora e dopo le 10:00
#      rispetto a MFIV_0931;
#   5. MFIV_0931 contro MFIV_session_t: il rapporto basso del punto 3 nei giorni FOMC
#      viene dalla media di sessione (che include le ore dopo l'annuncio) o dal prezzo
#      all'apertura?

STEM_FOMC = "14_fomc_cpi"

EVENT_SPECS = [("A1", "NA"), ("A2", "NA"), ("B-short", "NA"), ("B-full", "NA"),
               ("A5", "CBOE"), ("A5", "indep"), ("A6", "CBOE"), ("A6", "indep"),
               ("A1_ses", "NA")]


def fit(df, spec, extra=()):
    dep, regs = spec["dependent"], list(spec["regressors"]) + list(extra)
    w = df[["quote_date", dep] + regs].copy()
    for c in [dep] + regs:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna()
    w = w[np.isfinite(w[[dep] + regs]).all(axis=1)]
    L = nw_lag_length(len(w))
    X = sm.add_constant(w[regs].astype(float), has_constant="add")
    f = sm.OLS(w[dep].astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True)
    return f, w


def fomc_cpi():
    print("=" * 140)
    print("SEZIONE 3.7, C3: GIORNATE FOMC E CPI")
    print("=" * 140)

    panel = load_panel_logs()
    panel["FOMC"] = panel["quote_date"].isin(FOMC_DATES).astype(float)
    panel["CPI"] = panel["quote_date"].isin(CPI_DATES).astype(float)
    panel["tipo"] = np.where(panel["FOMC"] == 1, "FOMC", np.where(panel["CPI"] == 1, "CPI", "altre"))

    checks = []
    miss_f = sorted(set(FOMC_DATES) - set(panel["quote_date"]))
    miss_c = sorted(set(CPI_DATES) - set(panel["quote_date"]))
    checks.append(("tutte le 19 giornate evento sono nel pannello", not miss_f and not miss_c,
                   f"FOMC mancanti {miss_f}, CPI mancanti {miss_c}"))
    overlap = sorted(set(FOMC_DATES) & set(CPI_DATES))
    checks.append(("nessuna giornata e' sia FOMC sia CPI", not overlap, f"sovrapposte {overlap}"))

    specs = {(s["specification"], s["form"], s["gamma_source"]): s for s in build_specifications()}

    # riproduzione della baseline, sulle specifiche usate qui
    if BASELINE_RESULTS_FILE.exists():
        b = pd.read_csv(BASELINE_RESULTS_FILE, keep_default_na=False)
        b["coef"] = pd.to_numeric(b["coef"], errors="coerce")
        worst, nm = 0.0, 0
        for (sp, src) in EVENT_SPECS:
            for form in ("levels", "log"):
                s = specs.get((sp, form, src))
                if s is None:
                    continue
                f, _ = fit(panel, s)
                for term, cval in f.params.items():
                    x = b[(b["specification"] == sp) & (b["form"] == form)
                          & (b["gamma_source"] == src) & (b["term"] == term)]
                    if not len(x):
                        nm += 1
                        continue
                    worst = max(worst, abs(cval - x["coef"].iloc[0]) / max(abs(x["coef"].iloc[0]), 1e-300))
        checks.append(("le specifiche riproducono la baseline", nm == 0 and worst < REPRO_TOL,
                       f"{nm} senza corrispondenza, scarto massimo {worst:.1e}"))
    else:
        checks.append(("riproduzione della baseline", False, f"file non trovato: {BASELINE_RESULTS_FILE}"))

    print()
    print("CONTROLLI")
    for n, ok, d in checks:
        print(f"  [{'OK' if ok else 'FALLITO'}] {n}: {d}")
    print(f"\nGiornate: FOMC {int(panel['FOMC'].sum())}, CPI {int(panel['CPI'].sum())}, "
          f"altre {int((panel['tipo'] == 'altre').sum())}")

    # ---------------------------------------------------------------- 1. previsioni nelle giornate evento
    rows_fit, rows_dummy = [], []
    for (sp, src) in EVENT_SPECS:
        for form in ("levels", "log"):
            s = specs.get((sp, form, src))
            if s is None:
                continue
            f, w = fit(panel, s)
            w = w.merge(panel[["quote_date", "tipo"]], on="quote_date")
            w["fitted"] = f.fittedvalues.values
            dep = s["dependent"]
            for tipo in ("FOMC", "CPI", "altre"):
                x = w[w["tipo"] == tipo]
                rec = {"specification": sp, "form": form, "gamma_source": src, "tipo": tipo, "n": len(x)}
                if form == "levels":
                    rec["rapporto_delle_somme"] = float(x[dep].sum() / x["fitted"].sum())
                    rec["residuo_medio"] = float((x[dep] - x["fitted"]).mean())
                    rec["adattati_non_positivi"] = int((x["fitted"] <= 0).sum())
                else:
                    rec["residuo_medio_log"] = float((x[dep] - x["fitted"]).mean())
                    rec["rapporto_implicito"] = float(np.exp((x[dep] - x["fitted"]).mean()))
                rows_fit.append(rec)

            g, _ = fit(panel, s, extra=("FOMC", "CPI"))
            r = {"specification": sp, "form": form, "gamma_source": src, "n": int(g.nobs)}
            for d in ("FOMC", "CPI"):
                r[f"{d}_coef"] = float(g.params[d]); r[f"{d}_p"] = float(g.pvalues[d])
            # spostamento dei coefficienti principali con gli indicatori, in errori
            # standard della stima senza indicatori
            main_terms = [t for t in f.params.index if t != "const"]
            shifts = {t: (g.params[t] - f.params[t]) / f.bse[t] for t in main_terms}
            worst = max(shifts, key=lambda t: abs(shifts[t]))
            r["max_spostamento_in_es"] = float(shifts[worst])
            r["termine_che_si_sposta_di_piu"] = worst
            rows_dummy.append(r)

    fit_tab = pd.DataFrame(rows_fit)
    dum_tab = pd.DataFrame(rows_dummy)
    fit_tab.to_csv(OUTPUT_DIR / f"{STEM_FOMC}_fitted_by_day_type.csv", index=False)
    dum_tab.to_csv(OUTPUT_DIR / f"{STEM_FOMC}_event_dummies.csv", index=False)

    print()
    print("=" * 140)
    print("1. REALIZZATA CONTRO PREVISTA, PER TIPO DI GIORNATA (modello stimato sul campione intero)")
    print("=" * 140)
    for form in ("levels", "log"):
        print(f"\n-- forma {form} --")
        x = fit_tab[fit_tab["form"] == form].dropna(axis=1, how="all")
        print(x.drop(columns=["form"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("=" * 140)
    print("2. INDICATORI DI GIORNATA EVENTO AGGIUNTI ALLE SPECIFICHE")
    print("=" * 140)
    print("Coefficiente positivo = in quelle giornate il modello SOTTOSTIMA la variabile dipendente.")
    print("Potenza bassa: 8 giornate FOMC, 11 CPI.")
    for form in ("levels", "log"):
        print(f"\n-- forma {form} --")
        x = dum_tab[dum_tab["form"] == form]
        print(x.drop(columns=["form"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---------------------------------------------------------------- 3. premio al rischio di varianza per tipo di giornata
    panel["ratio_RV_MFIV_session"] = panel["RV_t"] / panel["MFIV_session_t"]
    vrp = panel.groupby("tipo")["ratio_RV_MFIV_session"].agg(["count", "mean", "median"]).reindex(
        ["FOMC", "CPI", "altre"])
    print()
    print("=" * 140)
    print("3. RAPPORTO FRA VARIANZA REALIZZATA E IMPLICITA (RV_t / MFIV_session_t)")
    print("=" * 140)
    print(vrp.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nGiornate evento, una per una:")
    ev = panel[panel["tipo"] != "altre"][["quote_date", "tipo", "RV_t", "MFIV_session_t",
                                          "ratio_RV_MFIV_session"]]
    print(ev.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---------------------------------------------------------------- 4. verifica dell'ipotesi della mattina calma
    panel["open_su_mfiv0931"] = panel["SpotRV_open"] / panel["MFIV_0931"]
    panel["quota_dopo_10"] = panel["FwdRV_short"] / panel["RV_t"]
    panel["pomeriggio_su_mfiv0931"] = panel["FwdRV_short"] / panel["MFIV_0931"]
    tab4 = panel.groupby("tipo")[["open_su_mfiv0931", "quota_dopo_10", "pomeriggio_su_mfiv0931"]].median().reindex(
        ["FOMC", "CPI", "altre"])
    print()
    print("=" * 140)
    print("4. IPOTESI DELLA MATTINA CALMA (mediane per tipo di giornata)")
    print("=" * 140)
    print("open_su_mfiv0931: varianza 09:31-10:00 / varianza implicita alle 09:31")
    print("quota_dopo_10: varianza 10:00-chiusura / varianza della sessione")
    print("pomeriggio_su_mfiv0931: varianza 10:00-chiusura / varianza implicita alle 09:31")
    print(tab4.to_string(float_format=lambda v: f"{v:.4f}"))
    f_open = float(tab4.loc["FOMC", "open_su_mfiv0931"]); a_open = float(tab4.loc["altre", "open_su_mfiv0931"])
    f_q = float(tab4.loc["FOMC", "quota_dopo_10"]); a_q = float(tab4.loc["altre", "quota_dopo_10"])
    fo = panel[panel["tipo"] == "FOMC"]
    n_low = int((fo["open_su_mfiv0931"] < a_open).sum()); n_high = int((fo["quota_dopo_10"] > a_q).sum())
    print(f"\nGiornate FOMC con prima mezz'ora sotto la mediana delle altre, rispetto alla varianza implicita: {n_low} su 8")
    print(f"Giornate FOMC con quota dopo le 10:00 sopra la mediana delle altre: {n_high} su 8")
    if f_open < a_open and f_q > a_q:
        print("-> Coerente con l'ipotesi della mattina calma.")
    else:
        print("-> NON coerente con l'ipotesi della mattina calma: la spiegazione va cercata altrove.")
    tab4.to_csv(OUTPUT_DIR / f"{STEM_FOMC}_morning_check.csv")

    # ---------------------------------------------------------------- 5. prezzo all'apertura contro media di sessione
    panel["RV_su_mfiv0931"] = panel["RV_t"] / panel["MFIV_0931"]
    panel["session_su_mfiv0931"] = panel["MFIV_session_t"] / panel["MFIV_0931"]
    panel["ann_session_su_ann0931"] = panel["MFIV_ann_t"] / panel["MFIV_ann_0931"]
    cols5 = ["RV_su_mfiv0931", "session_su_mfiv0931", "ann_session_su_ann0931", "ratio_RV_MFIV_session"]
    tab5 = panel.groupby("tipo")[cols5].median().reindex(["FOMC", "CPI", "altre"])
    print()
    print("=" * 140)
    print("5. PREZZO ALL'APERTURA CONTRO MEDIA DI SESSIONE (mediane per tipo di giornata)")
    print("=" * 140)
    print("RV_su_mfiv0931: varianza realizzata della sessione / varianza implicita alle 09:31")
    print("session_su_mfiv0931: MFIV_session_t / varianza implicita alle 09:31")
    print("ann_session_su_ann0931: media di sessione di MFIV annualizzata / MFIV annualizzata alle 09:31")
    print("ratio_RV_MFIV_session: la colonna della tabella 3, per confronto")
    print(tab5.to_string(float_format=lambda v: f"{v:.4f}"))
    fo = panel[panel["tipo"] == "FOMC"]
    alt = panel[panel["tipo"] == "altre"]
    n_up = int((fo["session_su_mfiv0931"] > alt["session_su_mfiv0931"].median()).sum())
    n_rv = int((fo["RV_su_mfiv0931"] >= alt["RV_su_mfiv0931"].median()).sum())
    print(f"\nGiornate FOMC con MFIV_session / MFIV_0931 sopra la mediana delle altre: {n_up} su 8")
    print(f"Giornate FOMC con RV / MFIV_0931 non sotto la mediana delle altre: {n_rv} su 8")
    print("\nGiornate FOMC, una per una:")
    print(fo[["quote_date", "RV_su_mfiv0931", "session_su_mfiv0931", "ratio_RV_MFIV_session"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    up = float(tab5.loc["FOMC", "session_su_mfiv0931"]) > 1.2 * float(tab5.loc["altre", "session_su_mfiv0931"])
    notlow = float(tab5.loc["FOMC", "RV_su_mfiv0931"]) >= float(tab5.loc["altre", "RV_su_mfiv0931"])
    if up and notlow:
        print("-> Coerente con l'ipotesi: il rapporto basso della tabella 3 viene dalla media di sessione,")
        print("   non da un prezzo pagato all'apertura.")
    else:
        print("-> NON coerente con l'ipotesi: la lettura della tabella 3 va ripensata.")
    tab5.to_csv(OUTPUT_DIR / f"{STEM_FOMC}_open_vs_session.csv")

    print()
    print("Note: il 10 aprile (CPI) cade nella settimana dei dazi; il 18 dicembre (CPI) contiene due")
    print("mesi di informazione. Riportati come sono, non trattati diversamente.")
    print(f"Output: {STEM_FOMC}_fitted_by_day_type.csv, _event_dummies.csv")
    print("=" * 140)

    if not all(ok for _, ok, _ in checks):
        raise SystemExit("Parte 4: controlli FALLITI (vedi sopra).")


# =====================================================================
# PARTE 5: MISURE DI GAMMA A 5, 15 E 30 MINUTI
# =====================================================================
# H_OI e H_VOL salgono meccanicamente verso la chiusura, perche' il gamma vicino al denaro
# cresce piu' in fretta quando tau si accorcia. Le medie di sessione su griglie piu' rade
# (punti 09:31 + j*k fino alle 16:00, righe con gamma_flag "ok"; a 30 minuti l'ultimo punto
# e' alle 15:31) verificano se questa salita guida il risultato: "calo sistematico" =
# |coef| decrescente da 1 a 5, 15 e 30 minuti su entrambe le fonti di gamma.
# Controlli: (a) per ogni fonte si usa solo il file di gamma intraday che a un minuto
# riproduce le colonne del pannello; (b) a un minuto le specificazioni ridanno la baseline.

GAMMA_SOURCES = {
    "CBOE": {"folder_name": "04_gamma_herfindahl",
             "file_name": "04_gamma_cboe_intraday_2025_all.csv",
             "h_oi": "h_oi_cboe", "h_vol": "h_vol_cboe",
             "panel_oi": "H_OI_t", "panel_vol": "H_VOL_t"},
    "indep": {"folder_name": "04_gamma_herfindahl",
              "file_name": "04_gamma_intraday_2025_all.csv",
              "h_oi": "h_oi", "h_vol": "h_vol",
              "panel_oi": "H_OI_t_indep", "panel_vol": "H_VOL_t_indep"},
}

STEM_GAMMA = "14_gamma_sampling_frequency"

K_LIST = [1, 5, 15, 30]
GRID_START = 9 * 60 + 31
GRID_END = 16 * 60

GAMMA_SPECS = [("A5", "levels"), ("A5", "log"), ("A6", "levels"), ("A6", "log")]


def to_minute(s: pd.Series) -> pd.Series:
    p = s.astype(str).str.strip().str.split(":", expand=True)
    return p[0].astype(int) * 60 + p[1].astype(int)


def load_gamma(gamma_path, src: dict) -> pd.DataFrame:
    d = pd.read_csv(gamma_path, dtype=str, keep_default_na=False,
                    usecols=lambda c: c in ("quote_date", "quote_time", "gamma_flag",
                                            src["h_oi"], src["h_vol"]))
    miss = [c for c in ("quote_date", "quote_time", "gamma_flag", src["h_oi"], src["h_vol"])
            if c not in d.columns]
    if miss:
        raise ValueError(f"{gamma_path}: colonne mancanti {miss}")
    d["quote_date"] = clean_date_series(d["quote_date"])
    d["minute"] = to_minute(d["quote_time"])
    for c in (src["h_oi"], src["h_vol"]):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ok = d["gamma_flag"].astype(str).str.strip().eq("ok")
    win = (d["minute"] >= GRID_START) & (d["minute"] <= GRID_END)
    return d[ok & win].copy()


def daily_means(g: pd.DataFrame, src: dict, k: int) -> pd.DataFrame:
    on_grid = ((g["minute"] - GRID_START) % k) == 0
    x = g[on_grid]
    return (x.groupby("quote_date", as_index=False)
            .agg(h_oi=(src["h_oi"], "mean"), h_vol=(src["h_vol"], "mean"),
                 n_points=(src["h_oi"], "count"), last_minute=("minute", "max")))


def find_source(panel: pd.DataFrame, name: str, src: dict):
    cands = sorted((DATA_DIR / src["folder_name"]).rglob(src["file_name"]))
    report = []
    for p in cands:
        try:
            g = load_gamma(p, src)
        except Exception as exc:
            report.append((p, f"non leggibile: {exc}"))
            continue
        m = daily_means(g, src, 1).merge(panel[["quote_date", src["panel_oi"], src["panel_vol"]]],
                                         on="quote_date", how="right")
        r1 = ((m["h_oi"] - m[src["panel_oi"]]).abs() / m[src["panel_oi"]].abs()).max()
        r2 = ((m["h_vol"] - m[src["panel_vol"]]).abs() / m[src["panel_vol"]].abs()).max()
        nmiss = int(m["h_oi"].isna().sum())
        ok = nmiss == 0 and r1 < REPRO_TOL and r2 < REPRO_TOL
        report.append((p, f"scarto H_OI {r1:.1e}, H_VOL {r2:.1e}, giornate mancanti {nmiss}"
                          + ("  <- RIPRODUCE IL PANNELLO" if ok else "")))
        if ok:
            return p, g, report
    return None, None, report


def with_k(s, k):
    t = dict(s)
    t["regressors"] = [f"{x}_k{k}" if x in ("H_OI_t", "H_VOL_t", "H_OI_t_indep", "H_VOL_t_indep")
                       else x for x in s["regressors"]]
    return t


def gamma_sampling_frequency():
    print("=" * 140)
    print("SEZIONE 3.7, C1: MISURE DI GAMMA A 5, 15 E 30 MINUTI")
    print("=" * 140)

    panel = load_panel_logs()
    specs_all = build_specifications()
    checks = []
    gamma = {}

    for name, src in GAMMA_SOURCES.items():
        gamma_path, g, rep = find_source(panel, name, src)
        print(f"\nFonte {name}: file '{src['file_name']}' cercati sotto {DATA_DIR / src['folder_name']}")
        for p, msg in rep:
            print(f"  {p}\n      {msg}")
        checks.append((f"(a) {name}: un file riproduce il pannello a un minuto", gamma_path is not None,
                       str(gamma_path) if gamma_path else "nessun candidato riproduce il pannello"))
        if gamma_path is not None:
            gamma[name] = (g, src)

    if len(gamma) < 2:
        print()
        for n, ok, d in checks:
            print(f"  [{'OK' if ok else 'FALLITO'}] {n}: {d}")
        print("\nSenza entrambe le fonti riprodotte il test non e' affidabile: controllare")
        print("i file di gamma intraday della baseline.")
        raise SystemExit("Parte 5: le due fonti di gamma non riproducono il pannello.")

    # griglie
    grid_info = []
    for k in K_LIST:
        for name, (g, src) in gamma.items():
            dm = daily_means(g, src, k)
            panel[f"{src['panel_oi']}_k{k}"] = panel["quote_date"].map(dm.set_index("quote_date")["h_oi"])
            panel[f"{src['panel_vol']}_k{k}"] = panel["quote_date"].map(dm.set_index("quote_date")["h_vol"])
            grid_info.append({"k": k, "fonte": name, "punti_medi_per_giorno": dm["n_points"].mean(),
                              "ultimo_minuto_tipico": f"{int(dm['last_minute'].median())//60:02d}:"
                                                       f"{int(dm['last_minute'].median())%60:02d}",
                              "media_H_OI": dm["h_oi"].mean(), "media_H_VOL": dm["h_vol"].mean()})

    specs = [s for s in specs_all if (s["specification"], s["form"]) in GAMMA_SPECS]

    rows = []
    for k in K_LIST:
        for s in specs:
            r, _ = estimate_one_spec(panel, with_k(s, k))
            r["k_minutes"] = k
            r["term_base"] = r["term"].str.replace(f"_k{k}", "", regex=False)
            rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUTPUT_DIR / f"{STEM_GAMMA}_results.csv", index=False)

    # (b) a un minuto riproduce la baseline
    if BASELINE_RESULTS_FILE.exists():
        b = pd.read_csv(BASELINE_RESULTS_FILE, keep_default_na=False)
        b["coef"] = pd.to_numeric(b["coef"], errors="coerce")
        r1 = res[res["k_minutes"] == 1].merge(
            b, left_on=["specification", "form", "gamma_source", "term_base"],
            right_on=["specification", "form", "gamma_source", "term"], how="left", suffixes=("", "_b"))
        miss = int(r1["coef_b"].isna().sum())
        gap = float(((r1["coef"] - r1["coef_b"]).abs() / r1["coef_b"].abs().clip(lower=1e-300)).max())
        checks.append(("(b) a un minuto le specifiche riproducono la baseline",
                       miss == 0 and gap < REPRO_TOL,
                       f"{len(r1)} coefficienti, {miss} senza corrispondenza, scarto {gap:.1e}"))
    else:
        checks.append(("(b) riproduzione della baseline", False, f"file non trovato: {BASELINE_RESULTS_FILE}"))

    print()
    print("CONTROLLI")
    for n, ok, d in checks:
        print(f"  [{'OK' if ok else 'FALLITO'}] {n}: {d}")

    print()
    print("GRIGLIE")
    print(pd.DataFrame(grid_info).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("=" * 140)
    print("COEFFICIENTI DI GAMMA PER FREQUENZA DI CAMPIONAMENTO")
    print("=" * 140)
    g_terms = res[res["term_base"].isin(["H_OI_t", "H_VOL_t", "H_OI_t_indep", "H_VOL_t_indep"])]
    piv = g_terms.pivot_table(index=["specification", "form", "gamma_source", "term_base"],
                              columns="k_minutes", values=["coef", "p_value"], aggfunc="first")
    piv.columns = [f"{a}_k{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    print(piv.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("REGOLA FISSATA PRIMA: calo sistematico = |coef| decrescente da 1 a 5 a 15 a 30")
    print("minuti su ENTRAMBE le fonti.")
    for spec, form in GAMMA_SPECS:
        for base_cboe, base_ind in (("H_OI_t", "H_OI_t_indep"), ("H_VOL_t", "H_VOL_t_indep")):
            if spec == "A5" and base_cboe == "H_VOL_t":
                continue
            verdict = []
            for src_name, base in (("CBOE", base_cboe), ("indep", base_ind)):
                x = piv[(piv["specification"] == spec) & (piv["form"] == form)
                        & (piv["gamma_source"] == src_name) & (piv["term_base"] == base)]
                if not len(x):
                    continue
                a = [abs(float(x[f"coef_k{k}"].iloc[0])) for k in K_LIST]
                verdict.append(all(a[i] > a[i + 1] for i in range(len(a) - 1)))
            tag = "CALO SISTEMATICO" if verdict and all(verdict) else "nessun calo sistematico"
            print(f"  {spec} {form:6s} {base_cboe.replace('_t', ''):5s}: {tag}  (per fonte: {verdict})")

    print()
    print(f"Output: {STEM_GAMMA}_results.csv")
    print("=" * 140)

    if not all(ok for _, ok, _ in checks):
        raise SystemExit("Parte 5: controlli FALLITI (vedi sopra).")


# =====================================================================
# PARTE 6: MINCER-ZARNOWITZ EX ANTE (TAB. 5.4) E LOG-GAP GIORNALIERO
# =====================================================================
# Regressione di RV_d su MFIV_d(09:31) (Tab. 5.4) e su MFIV^ses_d (Tab. 5.3), in livelli e
# in log, sul campione intero, nei tre regimi e senza aprile, con le convenzioni delle
# parti 1 e 2; statistiche del log-rapporto ln(RV_d / X_d) (varianza con ddof = 1) e fatti
# del testo. Rapporto geometrico = exp della media del log-rapporto.
# Le righe su MFIV^ses_d si confrontano con gli output delle parti 1 e 2 e della parte 1 di
# 13_Daily_regressions.py (fonti "29", "30" e "27" in 14_crosscheck_27_29_30.csv); i valori
# stampati nella tesi si confrontano in 14_thesis_check.csv.
# Le colonne di 14_mz_ex_ante_2025.csv e 14_log_gap_jensen_2025.csv sono lette dai
# controlli di 17_Figures.py.

DAILY_PANEL = ("07_daily_panel", "07_daily_regression_panel_2025.csv")
RESULTS_27 = ("13_daily_regressions", "13_daily_regression_results_2025.csv")   # 13_Daily_regressions.py, parte 1
MZ_29 = (OUTPUT_FOLDER, OUT_MZ_SUBSAMPLE.name)                                    # parte 1
MZ_30 = (OUTPUT_FOLDER, f"{STEM_REGIMES}_mincer_zarnowitz.csv")                   # parte 2

OUT_MZ = "14_mz_ex_ante_2025.csv"
OUT_COOK = "14_mz_ex_ante_cook_2025.csv"
OUT_JENSEN = "14_log_gap_jensen_2025.csv"
OUT_BY_DAY = "14_log_gap_by_day_2025.csv"
OUT_FACTS = "14_log_gap_facts_2025.csv"
OUT_CROSS = "14_crosscheck_27_29_30.csv"
OUT_THESIS = "14_thesis_check.csv"

NEEDED_EX_ANTE = ["quote_date", "RV_t", "RV_lag1", "MFIV_0931", "MFIV_session_t", "tau_open"]
EXPECTED_ROWS = 250
EXPECTED_APRIL = N_APRIL_SESSIONS
SAMPLES = ["intero", "calmo", "intermedio", "agitato", "senza_aprile"]
REGRESSORS = [("MFIV_0931", "5.4"), ("MFIV_session_t", "5.3")]
BODY_QUANTILES = (0.10, 0.90)
CROSS_RTOL = 1e-9

# Tab. 5.3 e 5.4 come stampate: n, pendenza, se, t(pendenza = 1), F, stelle.
TABLE_PRINTED = {
    ("MFIV_0931", 42, "Tab. 5.4"): {
        ("levels", "intero"): (250, 1.392, 0.201, 1.95, 5.77, "***"),
        ("levels", "calmo"): (82, 0.818, 0.076, -2.41, 8.09, "***"),
        ("levels", "intermedio"): (81, 0.689, 0.075, -4.12, 16.70, "***"),
        ("levels", "agitato"): (82, 1.472, 0.236, 2.00, 5.31, "***"),
        ("levels", "senza_aprile"): (229, 0.781, 0.058, -3.78, 13.57, "***"),
        ("log", "intero"): (250, 1.053, 0.042, 1.26, 45.76, "***"),
        ("log", "calmo"): (82, 1.133, 0.077, 1.73, 16.53, "***"),
        ("log", "intermedio"): (81, 1.135, 0.089, 1.51, 22.09, "***"),
        ("log", "agitato"): (82, 1.123, 0.064, 1.92, 22.68, "***"),
        ("log", "senza_aprile"): (229, 1.042, 0.051, 0.81, 41.75, "***"),
    },
    ("MFIV_session_t", 41, "Tab. 5.3"): {
        ("levels", "intero"): (250, 1.322, 0.167, 1.93, 4.52, "**"),
        ("levels", "calmo"): (82, 0.673, 0.187, -1.75, 5.22, "***"),
        ("levels", "intermedio"): (81, 0.719, 0.080, -3.53, 7.50, "***"),
        ("levels", "agitato"): (82, 1.380, 0.196, 1.94, 3.73, "**"),
        ("levels", "senza_aprile"): (229, 0.793, 0.054, -3.85, 11.26, "***"),
        ("log", "intero"): (250, 1.037, 0.026, 1.42, 31.52, "***"),
        ("log", "calmo"): (82, 1.037, 0.087, 0.43, 17.81, "***"),
        ("log", "intermedio"): (81, 1.035, 0.058, 0.60, 11.13, "***"),
        ("log", "agitato"): (82, 1.052, 0.039, 1.32, 18.73, "***"),
        ("log", "senza_aprile"): (229, 1.040, 0.034, 1.20, 29.34, "***"),
    },
}


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def mincer_zarnowitz_ex_ante(df: pd.DataFrame, dep: str, reg: str) -> dict:
    """RV su X con HAC di Newey-West e test congiunto (0, 1), come mincer_zarnowitz della parte 2."""
    w = df[["quote_date", dep, reg]].dropna()
    w = w[np.isfinite(w[[dep, reg]]).all(axis=1)]
    y = w[dep].astype(float)
    X = sm.add_constant(w[reg].astype(float), has_constant="add")
    L = nw_lag_length(len(w))
    f = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L}, use_t=True)
    test = f.f_test(f"const = 0, {reg} = 1")
    b, sb = float(f.params[reg]), float(f.bse[reg])
    xbar = float(w[reg].mean())
    fit_mean = float(f.params["const"] + b * xbar)
    is_log = dep.startswith("log_")
    p_joint = float(np.squeeze(test.pvalue))
    return {
        "n": int(len(w)), "nw_lag_L": L,
        "slope": b, "se_slope": sb, "t_slope_vs_1": (b - 1.0) / sb,
        "F_joint": float(np.squeeze(test.fvalue)), "p_joint": p_joint, "stars": stars(p_joint),
        "intercept": float(f.params["const"]), "se_intercept": float(f.bse["const"]),
        "p_intercept": float(f.pvalues["const"]), "r_squared": float(f.rsquared),
        "regressor_mean": xbar, "fitted_at_mean": fit_mean,
        "gap_at_mean": fit_mean - xbar,
        "geometric_ratio": float(np.exp(fit_mean - xbar)) if is_log else np.nan,
        "first_date": str(w["quote_date"].min()), "last_date": str(w["quote_date"].max()),
    }


def same(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    return bool(np.isclose(float(a), float(b), rtol=CROSS_RTOL, atol=0.0))


def fmt(v, prec):
    if prec.startswith("d"):
        return f"{float(v):.{int(prec[1:])}f}"
    if prec.startswith("s"):
        return f"{float(v):.{int(prec[1:]) - 1}e}"
    return str(v)


def mz_ex_ante():
    # ---------------------------------------------------------------- pannello e controlli di integrita'
    print("=" * 120)
    print("MINCER-ZARNOWITZ EX ANTE (TABELLA 5.4) E LOG-GAP GIORNALIERO")
    print("=" * 120)

    panel_file = path(DAILY_PANEL)
    require(panel_file)

    raw = pd.read_csv(panel_file, dtype=str, keep_default_na=False)
    missing_cols = [c for c in NEEDED_EX_ANTE if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Colonne mancanti nel pannello: {missing_cols}")
    panel = raw[NEEDED_EX_ANTE].copy()
    panel["quote_date"] = clean_date_series(panel["quote_date"])
    panel = panel[panel["quote_date"].notna() & panel["quote_date"].str.startswith(YEAR)]
    panel = panel.sort_values("quote_date").reset_index(drop=True)
    for c in NEEDED_EX_ANTE[1:]:
        panel[c] = to_numeric_col(panel, c)

    if len(panel) != EXPECTED_ROWS or not panel["quote_date"].is_unique:
        raise ValueError(f"Il pannello ha {len(panel)} righe (attese {EXPECTED_ROWS}) o date ripetute.")
    core = ["RV_t", "MFIV_0931", "MFIV_session_t", "tau_open"]
    if panel[core].isna().any().any() or not (panel[core] > 0).all().all():
        raise ValueError("Valori mancanti o non positivi in RV_t, MFIV_0931, MFIV_session_t o tau_open.")
    if int(panel["RV_lag1"].notna().sum()) != EXPECTED_ROWS - 1:
        raise ValueError("RV_lag1 dovrebbe mancare solo nella prima sessione.")

    for c in ["RV_t", "MFIV_0931", "MFIV_session_t"]:
        make_log_column(panel, c, f"log_{c}")

    april = panel["quote_date"].str.startswith(f"{YEAR}-04")
    if int(april.sum()) != EXPECTED_APRIL:
        raise ValueError(f"Sessioni di aprile: {int(april.sum())}, attese {EXPECTED_APRIL}.")

    panel["regime_var"] = panel["RV_t"].rolling(REGIME_WINDOW).mean().shift(1)
    panel["regime"] = pd.qcut(panel["regime_var"], 3, labels=REGIME_LABELS).astype(object)
    n_no_regime = int(panel["regime"].isna().sum())
    if n_no_regime != REGIME_WINDOW:
        raise ValueError(f"{n_no_regime} sessioni senza regime, attese {REGIME_WINDOW}.")
    regime_counts = panel["regime"].value_counts().reindex(REGIME_LABELS)
    print(f"Pannello: {len(panel)} sessioni, {int(april.sum())} di aprile; "
          f"regimi {regime_counts.to_dict()}, {n_no_regime} sessioni iniziali senza regime.")

    subsamples = {
        "intero": panel,
        "calmo": panel[panel["regime"] == "calmo"],
        "intermedio": panel[panel["regime"] == "intermedio"],
        "agitato": panel[panel["regime"] == "agitato"],
        "senza_aprile": panel[~april],
    }

    # ---------------------------------------------------------------- 1. Mincer-Zarnowitz: Tab. 5.4 (MFIV_0931) e 5.3 (MFIV_session_t)
    rows = []
    for reg, table in REGRESSORS:
        for form, dep, x in (("levels", "RV_t", reg), ("log", "log_RV_t", f"log_{reg}")):
            for s in SAMPLES:
                rows.append({"tabella_tesi": table, "regressor": reg, "form": form, "sample": s,
                             **mincer_zarnowitz_ex_ante(subsamples[s], dep, x)})
    mz = pd.DataFrame(rows)
    mz.to_csv(OUTPUT_DIR / OUT_MZ, index=False)

    def mzrow(reg, form, s):
        r = mz[(mz["regressor"] == reg) & (mz["form"] == form) & (mz["sample"] == s)]
        assert len(r) == 1
        return r.iloc[0]

    print()
    print("TABELLA 5.4: RV_d su MFIV_d(09:31)            (e, sotto, Tabella 5.3 su MFIV^ses_d)")
    show = mz[["tabella_tesi", "form", "sample", "n", "nw_lag_L", "slope", "se_slope", "t_slope_vs_1",
               "F_joint", "stars", "intercept", "se_intercept", "geometric_ratio"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # ---------------------------------------------------------------- 2. distanza di Cook (regressione ex ante, OLS semplice come la parte 1)
    cook_frames = []
    for form, dep, x in (("levels", "RV_t", "MFIV_0931"), ("log", "log_RV_t", "log_MFIV_0931")):
        ols = sm.OLS(panel[dep].astype(float), sm.add_constant(panel[x].astype(float))).fit()
        infl = OLSInfluence(ols)
        cf = pd.DataFrame({
            "regressor": "MFIV_0931", "form": form, "quote_date": panel["quote_date"],
            "cooks_distance": infl.cooks_distance[0], "leverage": infl.hat_matrix_diag,
            "student_resid": infl.resid_studentized_internal,
        }).sort_values("cooks_distance", ascending=False).reset_index(drop=True)
        cf.insert(3, "rank", np.arange(1, len(cf) + 1))
        cook_frames.append(cf)
    cook = pd.concat(cook_frames, ignore_index=True)
    cook.to_csv(OUTPUT_DIR / OUT_COOK, index=False)
    cook_lev = cook[cook["form"] == "levels"].reset_index(drop=True)
    print()
    print("Cook, ex ante in livelli, prime 3:",
          "; ".join(f"{r.quote_date} {r.cooks_distance:.3f}" for r in cook_lev.head(3).itertuples()))

    # ---------------------------------------------------------------- 3. log-rapporto e Jensen
    jrows = []
    for reg, _ in REGRESSORS:
        for s, sel in (("intero", np.ones(len(panel), bool)), ("senza_aprile", (~april).to_numpy())):
            rv = panel.loc[sel, "RV_t"].to_numpy(float)
            xx = panel.loc[sel, reg].to_numpy(float)
            ratio = rv / xx
            lr = np.log(ratio)
            jrows.append({
                "regressor": reg, "sample": s, "n": int(len(lr)),
                "mean_log_ratio": float(lr.mean()), "var_log_ratio": float(lr.var(ddof=1)),
                "sd_log_ratio": float(lr.std(ddof=1)), "geometric_ratio": float(np.exp(lr.mean())),
                "mean_of_ratios": float(ratio.mean()), "median_ratio": float(np.median(ratio)),
                "ratio_of_sample_means": float(rv.mean() / xx.mean()),
                "min_log_ratio": float(lr.min()), "max_log_ratio": float(lr.max()),
            })
    jensen = pd.DataFrame(jrows)
    jensen.to_csv(OUTPUT_DIR / OUT_JENSEN, index=False)

    def jrow(reg, s):
        r = jensen[(jensen["regressor"] == reg) & (jensen["sample"] == s)]
        assert len(r) == 1
        return r.iloc[0]

    print()
    print("LOG-RAPPORTO ln(RV_d / X_d) (varianza e scarto con ddof = 1)")
    print(jensen.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    by_day = panel[["quote_date", "regime", "regime_var", "RV_t", "MFIV_0931", "MFIV_session_t",
                    "tau_open"]].copy()
    by_day.insert(1, "april", april.astype(int).to_numpy())
    by_day["log_ratio_open"] = np.log(by_day["RV_t"] / by_day["MFIV_0931"])
    by_day["log_ratio_ses"] = np.log(by_day["RV_t"] / by_day["MFIV_session_t"])
    by_day["ann_vol_implied_open_pct"] = 100.0 * np.sqrt(by_day["MFIV_0931"] / by_day["tau_open"])
    by_day["ann_vol_realized_pct"] = 100.0 * np.sqrt(by_day["RV_t"] / by_day["tau_open"])
    by_day.to_csv(OUTPUT_DIR / OUT_BY_DAY, index=False)

    # ---------------------------------------------------------------- 4. fatti del testo
    facts = []

    def fact(voce, valore, dettaglio=""):
        facts.append({"voce": voce, "valore": valore, "dettaglio": dettaglio})

    lo_open = by_day.loc[by_day["log_ratio_open"].idxmin()]
    hi_open = by_day.loc[by_day["log_ratio_open"].idxmax()]
    fact("log_ratio_open_min", lo_open["log_ratio_open"], f"{lo_open['quote_date']}, aprile={lo_open['april']}")
    fact("log_ratio_open_max", hi_open["log_ratio_open"], f"{hi_open['quote_date']}, aprile={hi_open['april']}")
    fact("extreme_open_sessions_in_april", int(lo_open["april"]) + int(hi_open["april"]),
         "numero di sessioni di aprile fra le due piu' estreme contro il prezzo di apertura")

    for col, lab in (("log_ratio_open", "open"), ("log_ratio_ses", "ses")):
        q_lo, q_hi = by_day[col].quantile(list(BODY_QUANTILES))
        a = by_day.loc[by_day["april"] == 1, col]
        fact(f"april_inside_p10_p90_{lab}", int(((a >= q_lo) & (a <= q_hi)).sum()),
             f"su {len(a)} sessioni di aprile; p10 {q_lo:.4f}, p90 {q_hi:.4f}")

    n_open_above = int((by_day["MFIV_0931"] > by_day["RV_t"]).sum())
    fact("days_opening_price_above_realized", n_open_above, f"su {len(by_day)} sessioni")

    i_rv = by_day["ann_vol_realized_pct"].idxmax()
    i_iv = by_day["ann_vol_implied_open_pct"].idxmax()
    fact("max_ann_vol_realized_pct", by_day.loc[i_rv, "ann_vol_realized_pct"], by_day.loc[i_rv, "quote_date"])
    fact("ann_vol_implied_open_same_day_pct", by_day.loc[i_rv, "ann_vol_implied_open_pct"], by_day.loc[i_rv, "quote_date"])
    fact("max_ann_vol_implied_open_pct", by_day.loc[i_iv, "ann_vol_implied_open_pct"], by_day.loc[i_iv, "quote_date"])
    highest_is_rv = bool(by_day.loc[i_rv, "ann_vol_realized_pct"] > by_day.loc[i_iv, "ann_vol_implied_open_pct"])
    fact("highest_point_is_realized", int(highest_is_rv), "1 = il punto piu' alto delle due serie e' la realizzata")

    april_regimes = by_day.loc[by_day["april"] == 1, "regime"].value_counts(dropna=False)
    fact("april_sessions_in_agitato", int(april_regimes.get("agitato", 0)), f"su {int(april.sum())}")
    fact("sessions_with_regime", int(panel["regime"].notna().sum()), str(regime_counts.to_dict()))
    fact("sessions_without_regime", n_no_regime, "prime sessioni senza 5 sessioni precedenti")
    fact("cook_max_ex_ante_levels", cook_lev.loc[0, "cooks_distance"], cook_lev.loc[0, "quote_date"])
    fact("cook_second_ex_ante_levels", cook_lev.loc[1, "cooks_distance"], cook_lev.loc[1, "quote_date"])
    other = mz[(mz["regressor"] == "MFIV_0931") & (mz["form"] == "log") & (mz["sample"] != "intero")]
    fact("geometric_ratio_open_other_min", other["geometric_ratio"].min(),
         other.loc[other["geometric_ratio"].idxmin(), "sample"])
    fact("geometric_ratio_open_other_max", other["geometric_ratio"].max(),
         other.loc[other["geometric_ratio"].idxmax(), "sample"])
    facts_df = pd.DataFrame(facts)
    facts_df.to_csv(OUTPUT_DIR / OUT_FACTS, index=False)
    print()
    print("FATTI DEL TESTO")
    print(facts_df.to_string(index=False))

    # ---------------------------------------------------------------- 5. controllo incrociato: righe MFIV^ses contro gli output salvati
    for p in (path(RESULTS_27), path(MZ_29), path(MZ_30)):
        require(p)
    r27 = pd.read_csv(path(RESULTS_27))
    m29 = pd.read_csv(path(MZ_29))
    m30 = pd.read_csv(path(MZ_30))
    cross = []

    def cross_add(fonte, chiave, campo, salvato, calcolato):
        cross.append({"fonte": fonte, "chiave": chiave, "campo": campo, "valore_salvato": salvato,
                      "valore_49": calcolato, "esito": "OK" if same(salvato, calcolato) else "DIVERSO"})

    SUB29 = {"intero": "full", "senza_aprile": "no_april"}
    for form in ("levels", "log"):
        for s, s29 in SUB29.items():
            mine = mzrow("MFIV_session_t", form, s)
            r = m29[(m29["form"] == form) & (m29["subsample"] == s29)].iloc[0]
            for c29, c49 in (("n", "n"), ("a", "intercept"), ("b", "slope"), ("se_b", "se_slope"),
                             ("t_b_vs_1", "t_slope_vs_1"), ("F_joint", "F_joint"), ("p_joint", "p_joint"),
                             ("gap_at_mean", "gap_at_mean"), ("ratio_RV_over_MFIV", "geometric_ratio")):
                cross_add("29", f"{form}/{s}", c49, r[c29], mine[c49])
        for s in ("intero", "calmo", "intermedio", "agitato"):
            mine = mzrow("MFIV_session_t", form, s)
            r = m30[(m30["form"] == form) & (m30["sample"] == s)].iloc[0]
            for c30, c49 in (("n", "n"), ("slope", "slope"), ("se_slope", "se_slope"), ("F", "F_joint"),
                             ("p_F", "p_joint"), ("ratio_at_mean", "geometric_ratio")):
                cross_add("30", f"{form}/{s}", c49, r[c30], mine[c49])
        mine = mzrow("MFIV_session_t", form, "intero")
        r = r27[(r27["specification"] == "A1_ses") & (r27["form"] == form) & (r27["term"] == "const")].iloc[0]
        for c27, c49 in (("coef", "intercept"), ("se_nw", "se_intercept"), ("p_value", "p_intercept"),
                         ("r_squared", "r_squared"), ("n", "n"), ("nw_lag_L", "nw_lag_L")):
            cross_add("27", f"A1_ses/{form}/const", c49, r[c27], mine[c49])
    cross_df = pd.DataFrame(cross)
    cross_df.to_csv(OUTPUT_DIR / OUT_CROSS, index=False)
    n_cross_bad = int((cross_df["esito"] != "OK").sum())
    print()
    print(f"CONTROLLO INCROCIATO MFIV^ses contro 27/29/30: {len(cross_df) - n_cross_bad} OK, "
          f"{n_cross_bad} DIVERSO (tolleranza relativa {CROSS_RTOL:g})")
    if n_cross_bad:
        print(cross_df[cross_df["esito"] != "OK"].to_string(index=False))

    # ---------------------------------------------------------------- 6. confronto con i valori stampati nella tesi
    # pagina: numero stampato (nel PDF e' +4). precisione: "dN" = N decimali, "sN" = N cifre
    # significative, "esatto" = intero o testo, "logico" = condizione del testo verificata.
    checks = []

    def check(pagina, voce, tesi, calcolato, prec, nota=""):
        if prec == "logico":
            ok = bool(tesi)
            tesi_txt, calc_txt = nota, calcolato
            nota_out = ""
        else:
            tesi_txt, calc_txt, nota_out = tesi, calcolato, nota
            if prec == "esatto":
                ok = str(tesi) == str(calcolato)
            else:
                ok = fmt(tesi, prec) == fmt(calcolato, prec)
        checks.append({"pagina": pagina, "voce": voce, "valore_tesi": tesi_txt, "valore_calcolato": calc_txt,
                       "precisione": prec, "esito": "OK" if ok else "DIVERSO", "nota": nota_out})

    # Tab. 5.3 e 5.4: n, pendenza, se, t(pendenza = 1), F, stelle.
    for (reg, pag, lab), cells in TABLE_PRINTED.items():
        for (form, s), (n, b, se, t, F, st) in cells.items():
            r = mzrow(reg, form, s)
            k = f"{lab} {form} {s}"
            check(pag, f"{k}: n", n, int(r["n"]), "esatto")
            check(pag, f"{k}: pendenza", b, r["slope"], "d3")
            check(pag, f"{k}: se", se, r["se_slope"], "d3")
            check(pag, f"{k}: t(pendenza=1)", t, r["t_slope_vs_1"], "d2")
            check(pag, f"{k}: F congiunta", F, r["F_joint"], "d2")
            check(pag, f"{k}: stelle F", st, r["stars"], "esatto", f"p = {r['p_joint']:.3g}")

    # Nota della Tab. 5.3
    ses_lev, ses_log = mzrow("MFIV_session_t", "levels", "intero"), mzrow("MFIV_session_t", "log", "intero")
    check(41, "Tab. 5.3 nota: sessioni con regime", 245, int(panel["regime"].notna().sum()), "esatto")
    check(41, "Tab. 5.3 nota: ritardo NW, campione intero", 4, int(ses_lev["nw_lag_L"]), "esatto")
    check(41, "Tab. 5.3 nota: ritardo NW, regimi", 3,
          int(mz[mz["sample"].isin(REGIME_LABELS)]["nw_lag_L"].max()), "esatto",
          "massimo sui regimi (min = %d)" % int(mz[mz["sample"].isin(REGIME_LABELS)]["nw_lag_L"].min()))
    check(41, "Tab. 5.3 nota: intercetta livelli", -3.02e-5, ses_lev["intercept"], "s3")
    check(41, "Tab. 5.3 nota: se intercetta livelli", 1.12e-5, ses_lev["se_intercept"], "s3")
    check(41, "Tab. 5.3 nota: p intercetta livelli", 0.007, ses_lev["p_intercept"], "d3")
    check(41, "Tab. 5.3 nota: intercetta log", 0.136, ses_log["intercept"], "d3")
    check(41, "Tab. 5.3 nota: se intercetta log", 0.260, ses_log["se_intercept"], "d3")
    check(41, "Tab. 5.3 nota: p intercetta log", 0.602, ses_log["p_intercept"], "d3")
    check(41, "Tab. 5.3 nota: rapporto geometrico intero", 0.794, ses_log["geometric_ratio"], "d3")
    reg_ses = mz[(mz["regressor"] == "MFIV_session_t") & (mz["form"] == "log") & mz["sample"].isin(REGIME_LABELS)]
    check(41, "Tab. 5.3 nota: rapporto geometrico regimi, minimo", 0.774, reg_ses["geometric_ratio"].min(), "d3")
    check(41, "Tab. 5.3 nota: rapporto geometrico regimi, massimo", 0.798, reg_ses["geometric_ratio"].max(), "d3")
    check(41, "Tab. 5.3 nota: rapporto geometrico senza aprile", 0.795,
          mzrow("MFIV_session_t", "log", "senza_aprile")["geometric_ratio"], "d3")

    # Nota della Tab. 5.4
    ea_lev, ea_log = mzrow("MFIV_0931", "levels", "intero"), mzrow("MFIV_0931", "log", "intero")
    check(42, "Tab. 5.4 nota: intercetta livelli", -3.83e-5, ea_lev["intercept"], "s3")
    check(42, "Tab. 5.4 nota: se intercetta livelli", 1.33e-5, ea_lev["se_intercept"], "s3")
    check(42, "Tab. 5.4 nota: intercetta log", 0.182, ea_log["intercept"], "d3")
    check(42, "Tab. 5.4 nota: se intercetta log", 0.416, ea_log["se_intercept"], "d3")
    check(42, "Tab. 5.4 nota: distanza di Cook massima (livelli, intero)", 25.6, cook_lev.loc[0, "cooks_distance"], "d1")
    check(42, "Tab. 5.4 nota: giorno della distanza di Cook massima", "2025-04-07", cook_lev.loc[0, "quote_date"], "esatto")
    check(42, "Tab. 5.4 nota: rapporto geometrico intero", 0.712, ea_log["geometric_ratio"], "d3")
    check(42, "Tab. 5.4 nota: rapporto geometrico altri campioni, minimo", 0.695, other["geometric_ratio"].min(), "d3")
    check(42, "Tab. 5.4 nota: rapporto geometrico altri campioni, massimo", 0.725, other["geometric_ratio"].max(), "d3")

    # Fig. 5.1, 5.2 e 5.3
    for s, v in (("intero", 1.04), ("calmo", 1.04), ("intermedio", 1.03), ("agitato", 1.05)):
        check(42, f"Fig. 5.1 legenda: pendenza log MFIV^ses, {s}", v, mzrow("MFIV_session_t", "log", s)["slope"], "d2")
    check(42, "Fig. 5.1 legenda: prime cinque sessioni senza regime", 5, n_no_regime, "esatto")
    n_apr_ag = int(april_regimes.get("agitato", 0))
    check(42, "Fig. 5.1 legenda: 'April 2025 (agitated regime)'", n_apr_ag == int(april.sum()),
          f"{n_apr_ag} di {int(april.sum())} nel regime agitato", "logico", "tutte le sessioni di aprile nel regime agitato")
    check(43, "Fig. 5.2 legenda: sessioni di aprile", 21, int(april.sum()), "esatto")
    check(43, "Fig. 5.3 titolo (a): rapporto geometrico, prezzo di apertura", 0.712,
          jrow("MFIV_0931", "intero")["geometric_ratio"], "d3")
    check(43, "Fig. 5.3 titolo (b): rapporto geometrico, MFIV^ses", 0.794,
          jrow("MFIV_session_t", "intero")["geometric_ratio"], "d3")
    check(43, "Fig. 5.3 legenda: altre sessioni", 229, int((~april).sum()), "esatto")
    check(43, "Fig. 5.3 legenda: sessioni di aprile", 21, int(april.sum()), "esatto")

    # Testo della sezione 4.1: numero di sessioni
    check(30, "Sez. 4.1: sessioni del disegno A", 250, len(panel), "esatto")
    check(30, "Sez. 4.1: sessioni con RV_{d-1}", 249, int(panel["RV_lag1"].notna().sum()), "esatto")

    # Testo della sezione 5.1
    b_na = mzrow("MFIV_0931", "levels", "senza_aprile")
    check(40, "Sez. 5.1: pendenza ex ante senza aprile ('about 0.78')", 0.78, b_na["slope"], "d2")
    check(40, "Sez. 5.1: senza aprile significativamente sotto uno", b_na["t_slope_vs_1"] < -1.96,
          f"t = {b_na['t_slope_vs_1']:.2f}", "logico", "t(pendenza=1) < -1.96")
    c_b, i_b, a_b = (mzrow("MFIV_0931", "levels", s)["slope"] for s in ("calmo", "intermedio", "agitato"))
    check(40, "Sez. 5.1: regimi calmo e intermedio sotto uno, agitato no", (c_b < 1) and (i_b < 1) and (a_b >= 1),
          f"{c_b:.3f}, {i_b:.3f}, {a_b:.3f}", "logico", "pendenze ex ante in livelli: calmo < 1, intermedio < 1, agitato >= 1")
    s_full, s_c, s_i, s_na = (mzrow("MFIV_session_t", "levels", s)["slope"]
                              for s in ("intero", "calmo", "intermedio", "senza_aprile"))
    check(40, "Sez. 5.1: MFIV^ses, pendenza intera sopra uno, fuori dal regime agitato e senza aprile sotto",
          (s_full > 1) and (s_c < 1) and (s_i < 1) and (s_na < 1),
          f"{s_full:.3f}; {s_c:.3f}, {s_i:.3f}; {s_na:.3f}", "logico", "intero > 1; calmo, intermedio, senza aprile < 1")
    check(40, "Sez. 5.1: test congiunti rifiutati in ogni campione (Tab. 5.3 e 5.4, senza RV a 5 min)",
          bool((mz["p_joint"] < 0.05).all()), f"p massimo {mz['p_joint'].max():.3g}", "logico", "tutti i p congiunti < 0.05")
    check(40, "Sez. 5.1: prezzo di apertura sopra la realizzata 'on most days'", n_open_above > len(panel) / 2,
          f"{n_open_above} di {len(panel)}", "logico", "piu' della meta' dei giorni")
    check(40, "Sez. 5.1: volatilita' realizzata del 7 aprile, annualizzata ('about 200 percent')", 200,
          by_day.loc[i_rv, "ann_vol_realized_pct"], "d0", f"giorno del massimo: {by_day.loc[i_rv, 'quote_date']}")
    check(40, "Sez. 5.1: il massimo delle due serie e' la realizzata del 7 aprile",
          highest_is_rv and by_day.loc[i_rv, "quote_date"] == "2025-04-07",
          f"realizzata {by_day.loc[i_rv, 'ann_vol_realized_pct']:.1f}% il {by_day.loc[i_rv, 'quote_date']}; "
          f"implicita massima {by_day.loc[i_iv, 'ann_vol_implied_open_pct']:.1f}% il {by_day.loc[i_iv, 'quote_date']}",
          "logico", "massimo delle due serie = realizzata del 7 aprile")
    check(40, "Sez. 5.1: realizzata del 7 aprile sopra il suo prezzo di apertura",
          by_day.loc[i_rv, "ann_vol_realized_pct"] > by_day.loc[i_rv, "ann_vol_implied_open_pct"],
          f"{by_day.loc[i_rv, 'ann_vol_realized_pct']:.1f}% contro {by_day.loc[i_rv, 'ann_vol_implied_open_pct']:.1f}%",
          "logico", "realizzata > implicita di apertura")
    check(40, "Sez. 5.1: rapporto geometrico contro MFIV^ses (circa)", 0.79, jrow("MFIV_session_t", "intero")["geometric_ratio"], "d2")
    check(40, "Sez. 5.1: rapporto geometrico contro l'apertura (circa)", 0.71, jrow("MFIV_0931", "intero")["geometric_ratio"], "d2")
    check(40, "Sez. 5.1: scarto del log-rapporto, apertura (circa)", 0.48, jrow("MFIV_0931", "intero")["sd_log_ratio"], "d2", "ddof = 1")
    check(40, "Sez. 5.1: scarto del log-rapporto, MFIV^ses (circa)", 0.34, jrow("MFIV_session_t", "intero")["sd_log_ratio"], "d2", "ddof = 1")
    check(40, "Sez. 5.1: media dei rapporti giornalieri, apertura", 0.81, jrow("MFIV_0931", "intero")["mean_of_ratios"], "d2")
    check(40, "Sez. 5.1: media dei rapporti giornalieri, MFIV^ses", 0.84, jrow("MFIV_session_t", "intero")["mean_of_ratios"], "d2")
    n_body_open = int(facts_df.loc[facts_df["voce"] == "april_inside_p10_p90_open", "valore"].iloc[0])
    n_body_ses = int(facts_df.loc[facts_df["voce"] == "april_inside_p10_p90_ses", "valore"].iloc[0])
    check(40, "Sez. 5.1: sessioni di aprile 'mostly in the body' (fra 10o e 90o percentile)",
          min(n_body_open, n_body_ses) > int(april.sum()) / 2,
          f"apertura {n_body_open} di 21, MFIV^ses {n_body_ses} di 21", "logico", "piu' della meta' in entrambi i pannelli")

    # Testo della sezione 5.1: log-rapporto e sessioni estreme
    check(41, "Sez. 5.1: varianza log-rapporto apertura, senza aprile (primo numero)", 0.237,
          jrow("MFIV_0931", "senza_aprile")["var_log_ratio"], "d3", "ddof = 1")
    check(41, "Sez. 5.1: varianza log-rapporto apertura, campione intero (secondo numero)", 0.234,
          jrow("MFIV_0931", "intero")["var_log_ratio"], "d3", "ddof = 1")
    check(41, "Sez. 5.1: varianza log-rapporto MFIV^ses, senza aprile (primo numero)", 0.115,
          jrow("MFIV_session_t", "senza_aprile")["var_log_ratio"], "d3", "ddof = 1")
    check(41, "Sez. 5.1: varianza log-rapporto MFIV^ses, campione intero (secondo numero)", 0.116,
          jrow("MFIV_session_t", "intero")["var_log_ratio"], "d3", "ddof = 1")
    check(41, "Sez. 5.1: sessione piu' estrema in basso contro l'apertura ('near -2')", -2,
          lo_open["log_ratio_open"], "d0", lo_open["quote_date"])
    check(41, "Sez. 5.1: sessione piu' estrema in alto contro l'apertura ('near +2')", 2,
          hi_open["log_ratio_open"], "d0", hi_open["quote_date"])
    check(41, "Sez. 5.1: le due sessioni piu' estreme non sono di aprile",
          int(lo_open["april"]) + int(hi_open["april"]) == 0,
          f"{lo_open['quote_date']}, {hi_open['quote_date']}", "logico", "nessuna delle due in aprile")
    check(41, "Sez. 5.1: RV a 5 minuti restringe il divario (0.811 contro 0.794): parte a 1 minuto", 0.794,
          ses_log["geometric_ratio"], "d3", "il valore a 5 minuti (0.811) e' della parte 2")

    # Testo delle sezioni 5.5, 6.1 e 6.4
    check(51, "Sez. 5.5: il regime agitato contiene tutte le sessioni di aprile", n_apr_ag == int(april.sum()),
          f"{n_apr_ag} di {int(april.sum())}", "logico", "tutte le sessioni di aprile nel regime agitato")
    check(55, "Sez. 6.1: log-rapporto medio negativo con entrambe le misure",
          bool((jensen.loc[jensen["sample"] == "intero", "mean_log_ratio"] < 0).all()),
          "; ".join(f"{r.regressor} {r.mean_log_ratio:.4f}" for r in jensen[jensen["sample"] == "intero"].itertuples()),
          "logico", "media del log-rapporto < 0 per MFIV_0931 e MFIV^ses")
    check(55, "Sez. 6.1: rapporto 'about 0.79' (MFIV^ses)", 0.79, jrow("MFIV_session_t", "intero")["geometric_ratio"], "d2")
    check(55, "Sez. 6.1: rapporto 'about 0.71' (apertura)", 0.71, jrow("MFIV_0931", "intero")["geometric_ratio"], "d2")
    check(58, "Sez. 6.4: pendenza ex ante senza aprile 'about four fifths'", 0.8, b_na["slope"], "d1")

    chk = pd.DataFrame(checks)
    chk.to_csv(OUTPUT_DIR / OUT_THESIS, index=False)

    # ---------------------------------------------------------------- sintesi
    n_ok = int((chk["esito"] == "OK").sum())
    n_bad = int((chk["esito"] != "OK").sum())
    print()
    print("=" * 120)
    print(f"CONFRONTO CON LA TESI: {len(chk)} voci, {n_ok} OK, {n_bad} DIVERSO")
    if n_bad:
        print(chk[chk["esito"] != "OK"].to_string(index=False))
    print(f"Output in {OUTPUT_DIR}:")
    for f in (OUT_MZ, OUT_COOK, OUT_JENSEN, OUT_BY_DAY, OUT_FACTS, OUT_CROSS, OUT_THESIS):
        print(f"  {f}")
    if n_bad == 0 and n_cross_bad == 0:
        print("VERDETTO: OK. Tutti i valori stampati nella tesi coincidono e le righe MFIV^ses "
              "riproducono gli output di 13_daily_regressions e delle parti 1 e 2.")
    else:
        print(f"VERDETTO: DA CONTROLLARE. {n_bad} valori diversi dalla tesi, "
              f"{n_cross_bad} differenze con gli output di 27/29/30.")


def main():
    daily_influence()
    regimes_rv5()
    bshort_rv5_diagnostic()
    fomc_cpi()
    gamma_sampling_frequency()
    mz_ex_ante()


if __name__ == "__main__":
    main()
