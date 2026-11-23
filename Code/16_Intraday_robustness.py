"""
16_Intraday_robustness.py: robustezza del disegno intraday (Capitoli 4-5, Appendici C-D).

  1. concentrazione dell'inferenza per giornata e stima senza la giornata piu' pesante;
  2. esclusione delle date di stress (aprile e scadenze trimestrali), tre varianti;
  3. bootstrap wild restricted per cluster di settimana, con diagnostiche di degenerazione;
  4. correlazione fra stock e flusso di convessita' e VIF;
  5. effetto di una deviazione standard dei termini di gamma, con IC al 95%;
  6. H_VOL con i controlli per ampiezza (logN) e quantita' (logG) degli scambi;
  7. Holm e Benjamini-Hochberg sui test di I3, conteggi intraday e confronto con la tesi.

Input:  CBOE/Data/03_mfiv/, 04_gamma_herfindahl/, 07_daily_panel/, 08_intraday_panel/,
        13_daily_regressions/, 15_intraday_regressions/
Output: CBOE/Data/16_intraday_robustness/
"""

import functools
import math
import os
import re
import traceback
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm
from scipy import stats

from common import (DATA_DIR, FOMC_DATES, STRESS_DAY, YEAR as YEAR_INT,
                    clean_date_series, output_dir, path, require,
                    safe_write_csv, to_numeric_series)

OUTPUT_FOLDER = "16_intraday_robustness"
OUTPUT_DIR = output_dir(OUTPUT_FOLDER)

YEAR = str(YEAR_INT)  # "2025": le date si filtrano con str.startswith(YEAR)

# Input e output condivisi da piu' parti, come (cartella, file) dentro CBOE/Data.
PANEL = ("08_intraday_panel", "08_merged_intraday_panel_2025_primary_247.csv")
PANEL_FILE = path(PANEL)
REGISTERED = ("15_intraday_regressions", "15_registered_spec_check_results.csv")
DAILY_PANEL = ("07_daily_panel", "07_daily_regression_panel_2025.csv")
CONCENTRATION = {
    "levels": (OUTPUT_FOLDER, "16_day_concentration_levels.csv"),
    "log": (OUTPUT_FOLDER, "16_day_concentration_log.csv"),
}

HORIZONS = [5, 15, 30, 60]
REGRESSORS = ["MFIV", "MFIV_ann"]
FE_COLS = ["date_fe", "week_x_time_fe"]
FE_FORMULA = "date_fe + week_x_time_fe"
ALPHA = 0.05

DEMEAN_TOL = 1e-13
DEMEAN_MAXIT = 20000


# =====================================================================
# FUNZIONI COMUNI ALLE PARTI
# =====================================================================
def ignoring_warnings(pattern: str):
    """Ignora i warning il cui messaggio corrisponde a pattern, solo durante la funzione
    decorata."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=pattern)
                return func(*args, **kwargs)
        return wrapper
    return decorator


def run_variants(func, variants):
    """Stima ogni variante in ordine; un errore non ferma le successive e viene
    segnalato alla fine."""
    failed = []
    for variant in variants:
        try:
            func(variant)
        except Exception:
            traceback.print_exc()
            failed.append(variant)
    if failed:
        raise SystemExit(f"Varianti con errori o controlli falliti: {failed}")


def normalize_time_string(value) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d{1,2}:\d{2}", text):
        h, m = text.split(":")
        return f"{int(h):02d}:{int(m):02d}:00"
    return text


def demean(values: np.ndarray, frame: pd.DataFrame, fe_cols) -> np.ndarray:
    """Toglie gli effetti fissi per proiezioni alternate (parti 1 e 3); i coefficienti che
    ne derivano sono controllati contro pyfixest."""
    X = np.array(values, dtype=float, copy=True)
    if X.ndim == 1:
        X = X[:, None]
    X = X - X.mean(axis=0)
    codes = [pd.factorize(frame[c].astype(str))[0] for c in fe_cols]
    counts = [np.bincount(c) for c in codes]
    scale = np.maximum(X.std(axis=0), 1e-300)
    for _ in range(DEMEAN_MAXIT):
        old = X.copy()
        for c, cnt in zip(codes, counts):
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(c, weights=X[:, j]) / cnt)[c]
        if np.max(np.abs(X - old) / scale) <= DEMEAN_TOL:
            break
    return X


# =====================================================================
# PARTE 1: CONCENTRAZIONE DELL'INFERENZA PER GIORNO
# =====================================================================
# Quanto l'inferenza a cluster per giorno dipende da una sola giornata. Tolti gli effetti
# fissi, con q = colonna j di X (X'X)^-1, i contributi di giornata di q*u (u: residui senza il
# termine j) sommano a beta_j, quelli di q*e danno la varianza a cluster SOMMA_d c_d^2.
#
# Giornate effettive (SOMMA c^2)^2 / SOMMA c^4 (anche per settimana): G se pesano tutte uguale,
# 1 se ne pesa una sola. Cella CONCENTRATA se una giornata porta piu' di meta' della varianza;
# ogni cella si ristima anche senza la giornata piu' influente sul coefficiente.

# Forme, nell'ordine di stima. In log si trasformano FwdRV, il regressore e TrailingRV, non le
# misure di gamma.
CONCENTRATION_FORMS = ("log", "levels")

CONCENTRATION_SHARE = 0.5


def concentration(values: np.ndarray, codes: np.ndarray, labels):
    """Quote e numero effettivo dei contributi di cluster (al quadrato)."""
    c = np.bincount(codes, weights=values)
    c2 = c ** 2
    tot = c2.sum()
    s = np.sort(c2)[::-1] / tot if tot > 0 else np.full(len(c2), np.nan)
    n_eff = tot ** 2 / np.sum(c2 ** 2) if tot > 0 else np.nan
    return {"top1": float(s[0]), "top3": float(s[:3].sum()),
            "n_eff": float(n_eff), "top_label": labels[int(np.argmax(c2))],
            "contrib": c}


def load_panel_concentration() -> pd.DataFrame:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Pannello non trovato:\n{PANEL_FILE}")
    df = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)
    df["grid_time"] = df["grid_time"].astype(str).apply(normalize_time_string)
    for c in ("date_fe", "iso_week_key", "week_x_time_fe"):
        df[c] = df[c].astype(str).str.strip()
    num = (["MFIV", "MFIV_ann", "TrailingRV", "H_OI_cboe", "H_VOL_cboe",
            "H_OI_indep", "H_VOL_indep", "usable_for_flow_specs"]
           + [f"FwdRV_{h}" for h in HORIZONS] + [f"h{h}_valid" for h in HORIZONS])
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["MFIV", "MFIV_ann", "TrailingRV"] + [f"FwdRV_{h}" for h in HORIZONS]:
        df[f"log_{c}"] = np.where(df[c] > 0, np.log(df[c].where(df[c] > 0)), np.nan)
    return df[df["quote_date"].astype(str).str.startswith(YEAR)].copy()


def ivname(iv, form):
    return f"log_{iv}" if form == "log" else iv


def cells_for_concentration(iv, form):
    v = ivname(iv, form)
    tr = "log_TrailingRV" if form == "log" else "TrailingRV"
    return [
        ("IH1", "none", False, [v], [v]),
        ("IH2", "none", False, [v, tr], [v]),
        ("IH3", "cboe", True, [v, tr, "H_OI_cboe", "H_VOL_cboe"],
         [v, "H_OI_cboe", "H_VOL_cboe"]),
        ("IH3", "indep", True, [v, tr, "H_OI_indep", "H_VOL_indep"],
         [v, "H_OI_indep", "H_VOL_indep"]),
    ]


@ignoring_warnings(".*singleton.*")
def day_concentration(form: str):
    output_file = path(CONCENTRATION[form])

    print("=" * 150)
    print(f"CONCENTRAZIONE DELL'INFERENZA PER GIORNO - FORMA {form.upper()}")
    print("=" * 150)

    panel = load_panel_concentration()
    # Risultati analitici di riferimento, dalla parte 4 di 15_Intraday_regressions.py.
    apath = path(REGISTERED)
    if not apath.exists():
        raise FileNotFoundError(f"Servono i risultati analitici:\n{apath}")
    analytic = pd.read_csv(apath)
    analytic = analytic[(analytic["form"] == form) & (analytic["vcov"] == "giorno")]

    rows, coef_gaps, se_ratios = [], [], []

    for iv in REGRESSORS:
        for spec, src, uses_flow, regs, tests in cells_for_concentration(iv, form):
            for h in HORIZONS:
                dep = f"FwdRV_{h}" if form == "levels" else f"log_FwdRV_{h}"
                w = panel[panel[f"h{h}_valid"] == 1]
                if uses_flow:
                    w = w[w["usable_for_flow_specs"] == 1]
                w = w.dropna(subset=[dep] + regs + FE_COLS).copy()
                for c in [dep] + regs:
                    w = w[np.isfinite(w[c])]
                w = w.reset_index(drop=True)

                Z = demean(w[[dep] + regs].to_numpy(), w, FE_COLS)
                y, X = Z[:, 0], Z[:, 1:]
                A = np.linalg.inv(X.T @ X)
                beta = A @ X.T @ y
                e = y - X @ beta

                dcode, dlab = pd.factorize(w["date_fe"])
                wcode, wlab = pd.factorize(w["iso_week_key"])

                for term in tests:
                    j = regs.index(term)
                    q = (X @ A)[:, j]
                    keep = [k for k in range(X.shape[1]) if k != j]
                    if keep:
                        Xr = X[:, keep]
                        u_r = y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
                    else:
                        u_r = y.copy()

                    # contributi al coefficiente (residui ristretti): sommano a beta_j
                    coef_day = concentration(q * u_r, dcode, dlab)
                    # varianza a cluster (residui del modello completo)
                    var_day = concentration(q * e, dcode, dlab)
                    var_week = concentration(q * e, wcode, wlab)
                    coef_week = concentration(q * u_r, wcode, wlab)

                    sum_check = abs(coef_day["contrib"].sum() - beta[j]) / max(abs(beta[j]), 1e-300)

                    # la giornata con il contributo maggiore (in valore assoluto) al coefficiente
                    top_day = dlab[int(np.argmax(np.abs(coef_day["contrib"])))]
                    top_day_coef_share = float(coef_day["contrib"][list(dlab).index(top_day)] / beta[j])

                    w_lo = w[w["date_fe"] != top_day]
                    fit_lo = pf.feols(f"{dep} ~ {' + '.join(regs)} | {FE_FORMULA}",
                                      data=w_lo, vcov={"CRV1": "date_fe"})

                    a = analytic[(analytic["regressor"] == iv)
                                 & (analytic["specification"] == spec)
                                 & (analytic["gamma_source"] == src)
                                 & (analytic["horizon"] == h)
                                 & (analytic["term"] == term)]
                    if len(a):
                        coef_gaps.append(abs(beta[j] - a["coef"].iloc[0])
                                         / max(abs(a["coef"].iloc[0]), 1e-300))
                        se_ratios.append(a["se"].iloc[0]
                                         / np.sqrt(np.sum(var_day["contrib"] ** 2)))

                    rows.append({
                        "regressor": iv, "specification": spec, "gamma_source": src,
                        "horizon": h, "term": term, "n": len(w),
                        "coef": beta[j],
                        "p_day": a["p_value"].iloc[0] if len(a) else np.nan,
                        "sum_of_day_contributions_gap": sum_check,
                        "var_day_top1": var_day["top1"], "var_day_top3": var_day["top3"],
                        "var_day_n_eff": var_day["n_eff"], "var_day_top_label": var_day["top_label"],
                        "coef_day_top1": coef_day["top1"], "coef_day_n_eff": coef_day["n_eff"],
                        "top_day_for_coef": top_day,
                        "top_day_share_of_coef": top_day_coef_share,
                        "var_week_n_eff": var_week["n_eff"],
                        "coef_week_top1": coef_week["top1"], "coef_week_n_eff": coef_week["n_eff"],
                        "coef_without_top_day": float(fit_lo.coef()[term]),
                        "p_day_without_top_day": float(fit_lo.pvalue()[term]),
                        "concentrated_day": bool(var_day["top1"] > CONCENTRATION_SHARE),
                    })
                print(f"  {iv:9s} {spec} {src:5s} h={h:2d}: fatto")

    res = pd.DataFrame(rows)
    res.to_csv(output_file, index=False)

    # CONTROLLI
    print()
    print("=" * 150)
    print("CONTROLLI")
    print("=" * 150)
    g1 = max(coef_gaps) if coef_gaps else np.nan
    ok1 = bool(coef_gaps) and g1 < 1e-8
    print(f"  [{'OK' if ok1 else 'FALLITO'}] coefficienti uguali a quelli del file analitico: "
          f"{len(coef_gaps)} celle, scarto relativo massimo {g1:.2e}")
    g2 = float(res["sum_of_day_contributions_gap"].max())
    ok2 = g2 < 1e-8
    print(f"  [{'OK' if ok2 else 'FALLITO'}] i contributi di giornata sommano al coefficiente: "
          f"scarto relativo massimo {g2:.2e}")
    r = pd.Series(se_ratios)
    ok3 = len(r) > 0 and (r.max() - r.min()) / r.mean() < 0.01
    print(f"  [{'OK' if ok3 else 'FALLITO'}] errore standard per giorno = radice della somma dei "
          f"contributi al quadrato per una costante: rapporto fra {r.min():.6f} e {r.max():.6f}")

    # LETTURA
    print()
    print("=" * 150)
    print("CONCENTRAZIONE PER GIORNO, CON IL CONFRONTO PER SETTIMANA")
    print("=" * 150)
    print(f"Numero effettivo di cluster: {panel['date_fe'].nunique()} giornate e "
          f"{panel['iso_week_key'].nunique()} settimane se pesassero tutte uguale.")
    print(f"Regola fissata prima: CONCENTRATA se una giornata porta oltre il "
          f"{CONCENTRATION_SHARE:.0%} della varianza a cluster per giorno.")
    print()
    show = ["regressor", "specification", "gamma_source", "horizon", "term", "coef",
            "p_day", "var_day_top1", "var_day_n_eff", "var_day_top_label",
            "top_day_for_coef", "top_day_share_of_coef", "var_week_n_eff",
            "coef_without_top_day", "p_day_without_top_day", "concentrated_day"]
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    nc = int(res["concentrated_day"].sum())
    print(f"Celle con inferenza per giorno concentrata: {nc} su {len(res)}.")
    print("Giornata piu' pesante per la varianza, conteggio:")
    print(res["var_day_top_label"].value_counts().head(8).to_string())

    print()
    print("=" * 150)
    print("IPOTESI PREREGISTRATE, CLUSTER PER GIORNO, CON E SENZA LA GIORNATA PIU' PESANTE")
    print("=" * 150)
    for iv in REGRESSORS:
        s = res[res["regressor"] == iv]
        def cnt(spec, term, src, col_p, col_c, mode):
            x = s[(s["specification"] == spec) & (s["term"] == term) & (s["gamma_source"] == src)]
            if mode == "pos":
                return int(((x[col_c] > 0) & (x[col_p] < ALPHA)).sum())
            if mode == "neg1":
                p1 = np.where(x[col_c] < 0, x[col_p] / 2, 1 - x[col_p] / 2)
                return int((p1 < ALPHA).sum())
            return int((x[col_p] < ALPHA).sum())
        for label, cp, cc in (("con", "p_day", "coef"),
                              ("senza", "p_day_without_top_day", "coef_without_top_day")):
            print(f"  {iv:9s} {label:5s} la giornata piu' pesante: "
                  f"H-i-1 {cnt('IH1', ivname(iv, form), 'none', cp, cc, 'pos')}/4   "
                  f"H-i-2 {cnt('IH2', ivname(iv, form), 'none', cp, cc, 'pos')}/4   "
                  f"H_OI neg. CBOE {cnt('IH3', 'H_OI_cboe', 'cboe', cp, cc, 'neg1')}/4   "
                  f"H_VOL CBOE {cnt('IH3', 'H_VOL_cboe', 'cboe', cp, cc, 'two')}/4, "
                  f"indip. {cnt('IH3', 'H_VOL_indep', 'indep', cp, cc, 'two')}/4")

    print()
    print(f"Output: {output_file.name}")
    print("=" * 150)

    if not (ok1 and ok2 and ok3):
        raise RuntimeError(f"Concentrazione per giorno ({form}): controlli falliti.")


# =====================================================================
# PARTE 2: ESCLUSIONE DEL CLUSTER DI DATE
# =====================================================================
# Ristima senza un gruppo di date di stress, come nel disegno giornaliero: l'insieme completo,
# le sole giornate di aprile e le sole scadenze trimestrali, per separare i due fenomeni.

EXCLUSION_FILE = OUTPUT_DIR / "16_cluster_exclusion_diagnostic.csv"
EXCLUSION_SUMMARY_FILE = OUTPUT_DIR / "16_cluster_exclusion_summary.csv"

FORMS = ["levels", "log"]

VCOV = {"CRV1": "date_fe"}

IV = "MFIV_ann"
LOG_IV = "log_MFIV_ann"

# Quindici date, quattordici nel pannello (il 18 aprile, Venerdi' Santo, il mercato e'
# chiuso): dodici della sequenza di aprile e due scadenze trimestrali (21 marzo e 20 giugno).
APRIL_DATES = [
    "2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11",
    "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-18",
    "2025-04-21", "2025-04-22", "2025-04-23",
]

QUARTERLY_EXPIRY_DATES = ["2025-03-21", "2025-06-20"]

VARIANTS = [
    ("insieme_completo", APRIL_DATES + QUARTERLY_EXPIRY_DATES),
    ("solo_aprile", APRIL_DATES),
    ("solo_scadenze_trimestrali", QUARTERLY_EXPIRY_DATES),
]

SPECIFICATIONS = [
    {"specification": "IH1", "gamma_source": "none", "uses_flow": False,
     "regressors_levels": [IV], "regressors_log": [LOG_IV]},
    {"specification": "IH2", "gamma_source": "none", "uses_flow": False,
     "regressors_levels": [IV, "TrailingRV"],
     "regressors_log": [LOG_IV, "log_TrailingRV"]},
    {"specification": "IH3", "gamma_source": "cboe", "uses_flow": True,
     "regressors_levels": [IV, "TrailingRV", "H_OI_cboe", "H_VOL_cboe"],
     "regressors_log": [LOG_IV, "log_TrailingRV", "H_OI_cboe", "H_VOL_cboe"]},
    {"specification": "IH3", "gamma_source": "indep", "uses_flow": True,
     "regressors_levels": [IV, "TrailingRV", "H_OI_indep", "H_VOL_indep"],
     "regressors_log": [LOG_IV, "log_TrailingRV", "H_OI_indep", "H_VOL_indep"]},
]

LOG_COLUMNS = ["MFIV_ann", "TrailingRV",
               "FwdRV_5", "FwdRV_15", "FwdRV_30", "FwdRV_60"]

NUMERIC_COLUMNS = [
    "MFIV", "MFIV_ann", "TrailingRV",
    "H_OI_cboe", "H_VOL_cboe", "H_OI_indep", "H_VOL_indep",
    "usable_for_flow_specs",
] + [f"FwdRV_{h}" for h in HORIZONS] + [f"h{h}_valid" for h in HORIZONS]


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in LOG_COLUMNS:
        if col not in out.columns:
            continue

        log_col = f"log_{col}"
        out[log_col] = np.nan

        mask = out[col].notna() & np.isfinite(out[col]) & (out[col] > 0)
        out.loc[mask, log_col] = np.log(out.loc[mask, col])

    return out


def fit(fml, data):
    try:
        return pf.feols(fml=fml, data=data, vcov=VCOV)
    except TypeError:
        return pf.feols(fml, data, vcov=VCOV)


def term_values(tidy, term):
    if term in tidy.index:
        row = tidy.loc[[term]]
    else:
        t2 = tidy.reset_index()
        col = "Coefficient" if "Coefficient" in t2.columns else t2.columns[0]
        row = t2[t2[col].astype(str) == term]

    if len(row) == 0:
        raise KeyError(f"Termine {term} non trovato.")

    return (float(row["Estimate"].iloc[0]), float(row["Std. Error"].iloc[0]),
            float(row["t value"].iloc[0]), float(row["Pr(>|t|)"].iloc[0]))


@ignoring_warnings(".*singleton fixed effect.*")
def cluster_exclusion():
    # LOAD
    print("=" * 140)
    print("DIAGNOSTICA DI ESCLUSIONE DEL CLUSTER DI DATE - DISEGNO INTRADAY")
    print("=" * 140)
    print(f"Pannello: {PANEL_FILE}")
    print("=" * 140)

    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Pannello non trovato:\n{PANEL_FILE}")

    panel = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)

    panel["quote_date"] = clean_date_series(panel["quote_date"])
    panel["grid_time"] = panel["grid_time"].astype(str).apply(normalize_time_string)

    for col in ("date_fe", "week_x_time_fe", "iso_week_key"):
        if col in panel.columns:
            panel[col] = panel[col].astype(str).str.strip()

    for col in NUMERIC_COLUMNS:
        if col in panel.columns:
            panel[col] = to_numeric_series(panel[col])

    panel = panel[
        panel["quote_date"].notna()
        & panel["quote_date"].astype(str).str.startswith(YEAR)
    ].copy()

    panel = add_log_columns(panel)

    print(f"Righe: {len(panel):,} | giornate: {panel['quote_date'].nunique()}")

    print()
    print("--- PRESENZA DELLE DATE DEL CLUSTER ---")

    present_dates = set(panel["quote_date"].unique())

    for name, dates in VARIANTS:
        present = sorted(set(dates) & present_dates)
        missing = sorted(set(dates) - present_dates)

        print(f"  {name:28s}: {len(present)} presenti su {len(dates)} specificate")

        if missing:
            print(f"    assenti (attese se festivi): {missing}")

    # STIMA
    rows = []

    total = len(HORIZONS) * len(FORMS) * len(SPECIFICATIONS)
    done = 0

    for horizon in HORIZONS:
        for form in FORMS:
            for spec in SPECIFICATIONS:
                dep = (f"FwdRV_{horizon}" if form == "levels"
                       else f"log_FwdRV_{horizon}")
                regs = (spec["regressors_levels"] if form == "levels"
                        else spec["regressors_log"])

                cols = [dep] + regs + ["date_fe", "week_x_time_fe",
                                       "quote_date", "grid_time"]

                w = panel[panel[f"h{horizon}_valid"] == 1].copy()

                if spec["uses_flow"] and "usable_for_flow_specs" in w.columns:
                    w = w[w["usable_for_flow_specs"] == 1].copy()

                w = w.dropna(subset=cols).copy()

                for c in [dep] + regs:
                    w = w[np.isfinite(w[c])]

                fml = f"{dep} ~ {' + '.join(regs)} | {FE_FORMULA}"

                tidy_full = fit(fml, w).tidy()

                for variant_name, variant_dates in VARIANTS:
                    sub = w[~w["quote_date"].isin(variant_dates)].copy()

                    if len(sub) < 200:
                        continue

                    tidy_sub = fit(fml, sub).tidy()

                    for term in regs:
                        cf, sf, tf, pf_full = term_values(tidy_full, term)
                        ce, se, te, pe = term_values(tidy_sub, term)

                        rows.append({
                            "specification": spec["specification"],
                            "horizon": horizon,
                            "form": form,
                            "gamma_source": spec["gamma_source"],
                            "variant": variant_name,
                            "term": term,
                            "n_full": len(w),
                            "n_excl": len(sub),
                            "n_days_excluded": int(
                                w["quote_date"].isin(variant_dates).sum()
                                / max(w["grid_time"].nunique(), 1)
                            ),
                            "coef_full": cf, "se_full": sf,
                            "t_full": tf, "p_full": pf_full,
                            "coef_excl": ce, "se_excl": se,
                            "t_excl": te, "p_excl": pe,
                            "coef_change_raw": ce - cf,
                            "pct_change_abs": (abs(ce) - abs(cf)) / abs(cf) * 100
                                if cf != 0 else np.nan,
                            "sign_flip": int((cf > 0) != (ce > 0)),
                            # attraversamento della soglia del 5%, in un senso o nell'altro
                            "crosses_5pct_threshold": int((pf_full < 0.05) != (pe < 0.05)),
                            "became_significant": int((pf_full >= 0.05) and (pe < 0.05)),
                            "lost_significance": int((pf_full < 0.05) and (pe >= 0.05)),
                        })

                done += 1

                if done % 8 == 0 or done == total:
                    print(f"    {done}/{total} specifiche", flush=True)

    result = pd.DataFrame(rows)
    safe_write_csv(result, EXCLUSION_FILE)

    # RIEPILOGO PER FORMA E VARIANTE
    summary_rows = []

    for form in FORMS:
        for variant_name, _ in VARIANTS:
            sub = result[(result["form"] == form) & (result["variant"] == variant_name)]

            if sub.empty:
                continue

            summary_rows.append({
                "disegno": "intraday",
                "forma": form,
                "variante": variant_name,
                "celle_totali": len(sub),
                "celle_che_attraversano_5pct": int(sub["crosses_5pct_threshold"].sum()),
                "diventano_significative": int(sub["became_significant"].sum()),
                "perdono_significativita": int(sub["lost_significance"].sum()),
                "cambi_di_segno": int(sub["sign_flip"].sum()),
                "variazione_mediana_pct": float(sub["pct_change_abs"].median()),
            })

    summary = pd.DataFrame(summary_rows)
    safe_write_csv(summary, EXCLUSION_SUMMARY_FILE)

    print()
    print("=" * 140)
    print("RIEPILOGO")
    print("=" * 140)
    print(summary.to_string(index=False))
    print()
    print("La colonna celle_che_attraversano_5pct e' la stessa riportata per il")
    print("disegno giornaliero. Con tre varianti la riga intraday diventa")
    print("confrontabile con quella giornaliera invece che monca.")

    print()
    print("=" * 140)
    print("DETTAGLIO SULLE MISURE DI CONVESSITA'")
    print("=" * 140)

    gamma_terms = result[result["term"].astype(str).str.startswith(("H_OI", "H_VOL"))]

    for form in FORMS:
        sub = gamma_terms[gamma_terms["form"] == form]

        if sub.empty:
            continue

        print()
        print(f"  forma {form}:")

        for variant_name, _ in VARIANTS:
            s = sub[sub["variant"] == variant_name]

            if s.empty:
                continue

            hv = s[s["term"].astype(str).str.startswith("H_VOL")]
            ho = s[s["term"].astype(str).str.startswith("H_OI")]

            print(f"    {variant_name:28s}: "
                  f"H_VOL {int((hv['p_excl'] < 0.05).sum())}/{len(hv)} significativi, "
                  f"{int((hv['coef_excl'] < 0).sum())}/{len(hv)} negativi | "
                  f"H_OI {int((ho['p_excl'] < 0.05).sum())}/{len(ho)} significativi")

    print()
    print("--- CONFRONTO CON IL CAMPIONE PIENO ---")

    for form in FORMS:
        sub = gamma_terms[gamma_terms["form"] == form].drop_duplicates(
            subset=["specification", "horizon", "gamma_source", "term"]
        )

        hv = sub[sub["term"].astype(str).str.startswith("H_VOL")]
        ho = sub[sub["term"].astype(str).str.startswith("H_OI")]

        print(f"  forma {form}: campione pieno, "
              f"H_VOL {int((hv['p_full'] < 0.05).sum())}/{len(hv)} significativi | "
              f"H_OI {int((ho['p_full'] < 0.05).sum())}/{len(ho)} significativi")

    print()
    print(f"Output: {EXCLUSION_FILE.name}, {EXCLUSION_SUMMARY_FILE.name}")
    print("=" * 140)


# =====================================================================
# PARTE 3: BOOTSTRAP WILD PER CLUSTER DI SETTIMANA
# =====================================================================
# Il cluster a due vie coincide con quello per settimana (giornate annidate nelle settimane):
# con 52 cluster e molti parametri assorbiti l'inferenza analitica e' poco affidabile.
#
# Wild cluster restricted bootstrap (WCR): per ogni coefficiente si stima il modello con il
# coefficiente a zero, si moltiplicano i residui ristretti per un peso di Rademacher costante
# nella settimana e si ristima; il p-value e' la quota di t bootstrap piu' estreme dell'osservata.
#
# Gli effetti fissi sono annidati nella settimana: i residui ripesati restano ortogonali a
# essi e il demeaning non va ripetuto a ogni estrazione. Per il giorno non vale
# (week_x_time_fe non e' annidato nel giorno).
#
# La t osservata e quelle bootstrap usano lo stesso sandwich senza correzioni di piccolo
# campione: le costanti si cancellano e il p-value non ne dipende.
#
# Se un contributo di settimana c_g sui residui ristretti domina, t* ~ +-1. Cella DEGENERE se
# max |t*| sta sotto 2,5: con 51 gradi di liberta' il massimo su migliaia di estrazioni
# dovrebbe superare 4. Il p di una cella degenere non va usato.

BOOTSTRAP_FILE = OUTPUT_DIR / "16_wild_cluster_bootstrap_week_diagnostics.csv"
DEGENERATE_MAX_ABS_T = 2.5

CLUSTER_COL = "iso_week_key"

B = 9999
CHUNK = 500
SEED = 20260906


def cluster_sums(M: np.ndarray, order: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Somme per cluster delle righe di M (N x B) -> (G x B)."""
    return np.add.reduceat(M[order], starts, axis=0)


def load_panel_bootstrap() -> pd.DataFrame:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Pannello non trovato:\n{PANEL_FILE}")
    df = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)
    df["grid_time"] = df["grid_time"].astype(str).apply(normalize_time_string)
    for c in ("date_fe", "iso_week_key", "week_x_time_fe"):
        df[c] = df[c].astype(str).str.strip()
    num = (["MFIV", "MFIV_ann", "TrailingRV", "H_OI_cboe", "H_VOL_cboe",
            "H_OI_indep", "H_VOL_indep", "usable_for_flow_specs"]
           + [f"FwdRV_{h}" for h in HORIZONS] + [f"h{h}_valid" for h in HORIZONS])
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["quote_date"].astype(str).str.startswith(YEAR)].copy()


def cells_for_bootstrap(iv):
    """(specifica, fonte, usa_flusso, regressori, termini da testare)."""
    return [
        ("IH1", "none", False, [iv], [iv]),
        ("IH2", "none", False, [iv, "TrailingRV"], [iv]),
        ("IH3", "cboe", True, [iv, "TrailingRV", "H_OI_cboe", "H_VOL_cboe"],
         [iv, "H_OI_cboe", "H_VOL_cboe"]),
        ("IH3", "indep", True, [iv, "TrailingRV", "H_OI_indep", "H_VOL_indep"],
         [iv, "H_OI_indep", "H_VOL_indep"]),
    ]


def wild_bootstrap_week():
    print("=" * 150)
    print("BOOTSTRAP WILD PER CLUSTER DI SETTIMANA (WCR, RADEMACHER)")
    print("=" * 150)

    panel = load_panel_bootstrap()
    print(f"Pannello: {PANEL_FILE}")
    print(f"Estrazioni per cella: {B}, seme {SEED}, effetti fissi {' + '.join(FE_COLS)}")

    # Risultati analitici di riferimento, dalla parte 4 di 15_Intraday_regressions.py.
    apath = path(REGISTERED)
    if not apath.exists():
        raise FileNotFoundError(f"Servono i risultati analitici:\n{apath}")
    analytic = pd.read_csv(apath)
    analytic = analytic[analytic["form"] == "levels"]

    rng = np.random.default_rng(SEED)
    rows = []
    coef_gaps = []

    for iv in REGRESSORS:
        for spec, src, uses_flow, regs, tests in cells_for_bootstrap(iv):
            for h in HORIZONS:
                dep = f"FwdRV_{h}"
                w = panel[panel[f"h{h}_valid"] == 1]
                if uses_flow:
                    w = w[w["usable_for_flow_specs"] == 1]
                w = w.dropna(subset=[dep] + regs + FE_COLS + [CLUSTER_COL]).copy()
                for c in [dep] + regs:
                    w = w[np.isfinite(w[c])]
                w = w.reset_index(drop=True)

                Z = demean(w[[dep] + regs].to_numpy(), w, FE_COLS)
                y, X = Z[:, 0], Z[:, 1:]
                A = np.linalg.inv(X.T @ X)
                beta = A @ X.T @ y
                e = y - X @ beta

                gcode, glabels = pd.factorize(w[CLUSTER_COL])
                order = np.argsort(gcode, kind="stable")
                starts = np.flatnonzero(np.r_[True, np.diff(gcode[order]) != 0])
                G = len(starts)

                for term in tests:
                    j = regs.index(term)
                    q = (X @ A)[:, j]
                    se_raw = np.sqrt(np.sum(cluster_sums((q * e)[:, None], order, starts) ** 2))
                    t_obs = beta[j] / se_raw

                    keep = [k for k in range(X.shape[1]) if k != j]
                    if keep:
                        Xr = X[:, keep]
                        br = np.linalg.lstsq(Xr, y, rcond=None)[0]
                        yhat_r = Xr @ br
                    else:
                        yhat_r = np.zeros_like(y)
                    u_r = y - yhat_r

                    # Contributi di settimana al coefficiente, sui residui del modello
                    # completo e su quelli ristretti che il bootstrap moltiplica per i pesi.
                    c_obs = cluster_sums((q * e)[:, None], order, starts)[:, 0]
                    c_res = cluster_sums((q * u_r)[:, None], order, starts)[:, 0]
                    s_obs = np.sort(c_obs ** 2)[::-1] / np.sum(c_obs ** 2)
                    s_res = np.sort(c_res ** 2)[::-1] / np.sum(c_res ** 2)
                    top_obs = glabels[int(np.argmax(c_obs ** 2))]
                    top_res = glabels[int(np.argmax(c_res ** 2))]

                    t_star = np.empty(B)
                    done = 0
                    while done < B:
                        b = min(CHUNK, B - done)
                        wts = rng.choice(np.array([-1.0, 1.0]), size=(G, b))
                        Ystar = yhat_r[:, None] + u_r[:, None] * wts[gcode]
                        Bstar = A @ (X.T @ Ystar)
                        Estar = Ystar - X @ Bstar
                        S = cluster_sums(q[:, None] * Estar, order, starts)
                        t_star[done:done + b] = Bstar[j] / np.sqrt(np.sum(S ** 2, axis=0))
                        done += b

                    p_two = float(np.mean(np.abs(t_star) >= abs(t_obs)))
                    p_neg = float(np.mean(t_star <= t_obs))

                    a = analytic[(analytic["regressor"] == iv)
                                 & (analytic["specification"] == spec)
                                 & (analytic["gamma_source"] == src)
                                 & (analytic["horizon"] == h)
                                 & (analytic["term"] == term)]
                    ad = a[a["vcov"] == "giorno"]
                    aw = a[a["vcov"] == "settimana"]
                    if len(ad):
                        coef_gaps.append(abs(beta[j] - ad["coef"].iloc[0])
                                         / max(abs(ad["coef"].iloc[0]), 1e-300))

                    rows.append({
                        "regressor": iv, "specification": spec, "gamma_source": src,
                        "horizon": h, "term": term, "n": len(w), "clusters": G,
                        "coef": beta[j], "t_week_raw": t_obs,
                        "se_ratio_pyfixest_week_over_raw": (aw["se"].iloc[0] / se_raw
                                                            if len(aw) else np.nan),
                        "p_analytic_day": ad["p_value"].iloc[0] if len(ad) else np.nan,
                        "p_analytic_week": aw["p_value"].iloc[0] if len(aw) else np.nan,
                        "p_boot_week": p_two,
                        "p_boot_week_one_sided_negative": p_neg,
                        "B": B,
                        "top_week_restricted": top_res,
                        "share_top1_restricted": float(s_res[0]),
                        "share_top3_restricted": float(s_res[:3].sum()),
                        "top_week_observed": top_obs,
                        "share_top1_observed": float(s_obs[0]),
                        "t_star_abs_q50": float(np.quantile(np.abs(t_star), 0.50)),
                        "t_star_abs_q95": float(np.quantile(np.abs(t_star), 0.95)),
                        "t_star_abs_q99": float(np.quantile(np.abs(t_star), 0.99)),
                        "t_star_abs_max": float(np.max(np.abs(t_star))),
                        "boot_degenerate": bool(np.max(np.abs(t_star)) < DEGENERATE_MAX_ABS_T),
                    })
                print(f"  {iv:9s} {spec} {src:5s} h={h:2d}: fatto")

    res = pd.DataFrame(rows)
    res.to_csv(BOOTSTRAP_FILE, index=False)

    # CONTROLLI
    print()
    print("=" * 150)
    print("CONTROLLI")
    print("=" * 150)

    gap = max(coef_gaps) if coef_gaps else np.nan
    ok1 = bool(coef_gaps) and gap < 1e-8
    print(f"  [{'OK' if ok1 else 'FALLITO'}] coefficienti dal demeaning uguali a pyfixest: "
          f"{len(coef_gaps)} celle, scarto relativo massimo {gap:.2e}")

    r = res["se_ratio_pyfixest_week_over_raw"].dropna()
    ok2 = len(r) > 0 and (r.max() - r.min()) / r.mean() < 0.01
    print(f"  [{'OK' if ok2 else 'FALLITO'}] errore standard per settimana di pyfixest = "
          f"sandwich grezzo per una costante di piccolo campione: rapporto fra "
          f"{r.min():.6f} e {r.max():.6f}")

    print(f"  [INFO] seme {SEED}, {B} estrazioni per cella: rilanciando si ottengono "
          f"gli stessi numeri.")

    # LETTURA
    print()
    print("=" * 150)
    print("P ANALITICI E BOOTSTRAP A CONFRONTO")
    print("=" * 150)

    show = ["regressor", "specification", "gamma_source", "horizon", "term", "coef",
            "p_analytic_day", "p_analytic_week", "p_boot_week",
            "p_boot_week_one_sided_negative"]
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("=" * 150)
    print("IPOTESI PREREGISTRATE CON INFERENZA BOOTSTRAP PER SETTIMANA")
    print("=" * 150)
    for iv in REGRESSORS:
        s = res[res["regressor"] == iv]
        def hit(spec, term, src="none", one_sided_neg=False, positive=False):
            x = s[(s["specification"] == spec) & (s["term"] == term) & (s["gamma_source"] == src)]
            if one_sided_neg:
                return int((x["p_boot_week_one_sided_negative"] < ALPHA).sum()), x
            if positive:
                return int(((x["coef"] > 0) & (x["p_boot_week"] < ALPHA)).sum()), x
            return int((x["p_boot_week"] < ALPHA).sum()), x
        n1, _ = hit("IH1", iv, positive=True)
        n2, _ = hit("IH2", iv, positive=True)
        oc, _ = hit("IH3", "H_OI_cboe", "cboe", one_sided_neg=True)
        oi, _ = hit("IH3", "H_OI_indep", "indep", one_sided_neg=True)
        vc, _ = hit("IH3", "H_VOL_cboe", "cboe")
        vi, _ = hit("IH3", "H_VOL_indep", "indep")
        print(f"  {iv:9s}: H-i-1 {n1}/4   H-i-2 {n2}/4   H_OI negativo CBOE {oc}/4, "
              f"indip. {oi}/4   H_VOL CBOE {vc}/4, indip. {vi}/4")

    print()
    print("=" * 150)
    print("DIAGNOSTICHE DI DEGENERAZIONE DEL BOOTSTRAP")
    print("=" * 150)
    _G = int(res["clusters"].median())
    print(f"Riferimento, t di Student a {_G - 1} gradi di liberta': |t| al 95esimo "
          f"percentile {stats.t.ppf(0.975, _G - 1):.2f}, al 99esimo {stats.t.ppf(0.995, _G - 1):.2f}; "
          f"massimo atteso su {B} estrazioni circa {stats.t.ppf(1 - 1 / (2 * B), _G - 1):.1f}.")
    print(f"Regola fissata prima: DEGENERE se max |t*| < {DEGENERATE_MAX_ABS_T}.")
    print()
    dshow = ["regressor", "specification", "gamma_source", "horizon", "term",
             "p_analytic_week", "p_boot_week", "t_star_abs_q95", "t_star_abs_max",
             "share_top1_restricted", "share_top3_restricted", "top_week_restricted",
             "boot_degenerate"]
    print(res[dshow].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    nd = int(res["boot_degenerate"].sum())
    print()
    print(f"Celle degeneri: {nd} su {len(res)}.")
    if nd:
        print("Settimane dominanti nelle celle degeneri (residui ristretti):")
        print(res[res["boot_degenerate"]]["top_week_restricted"].value_counts().to_string())

    print()
    print("=" * 150)
    print("IPOTESI PREREGISTRATE, SOLO CELLE CON BOOTSTRAP NON DEGENERE")
    print("=" * 150)
    ok_cells = res[~res["boot_degenerate"]]
    for iv in REGRESSORS:
        s = ok_cells[ok_cells["regressor"] == iv]
        def cnt(spec, term, src="none", mode="two"):
            x = s[(s["specification"] == spec) & (s["term"] == term) & (s["gamma_source"] == src)]
            if mode == "neg":
                return int((x["p_boot_week_one_sided_negative"] < ALPHA).sum()), len(x)
            if mode == "pos":
                return int(((x["coef"] > 0) & (x["p_boot_week"] < ALPHA)).sum()), len(x)
            return int((x["p_boot_week"] < ALPHA).sum()), len(x)
        a1 = cnt("IH1", iv, mode="pos"); a2 = cnt("IH2", iv, mode="pos")
        oc = cnt("IH3", "H_OI_cboe", "cboe", "neg"); vc = cnt("IH3", "H_VOL_cboe", "cboe")
        vi = cnt("IH3", "H_VOL_indep", "indep")
        print(f"  {iv:9s}: H-i-1 {a1[0]}/{a1[1]} usabili   H-i-2 {a2[0]}/{a2[1]}   "
              f"H_OI neg. CBOE {oc[0]}/{oc[1]}   H_VOL CBOE {vc[0]}/{vc[1]}, indip. {vi[0]}/{vi[1]}")

    print()
    print("COME SI LEGGE. Se p_boot_week sta vicino a p_analytic_week, gli errori")
    print("standard analitici per settimana sono affidabili e il cluster a due vie")
    print("preregistrato si puo' riportare. Se p_boot_week e' molto piu' grande, gli")
    print("analitici erano ottimistici e si riporta il bootstrap. In entrambi i casi il")
    print("confronto con p_analytic_day dice quanto cambia l'inferenza passando dal")
    print("giorno alla settimana.")
    print()
    print(f"Output: {BOOTSTRAP_FILE.name}")
    print("=" * 150)

    if not (ok1 and ok2):
        raise RuntimeError("Bootstrap wild per settimana: controlli falliti.")


# =====================================================================
# PARTE 4: CORRELAZIONE FRA STOCK E FLUSSO DI CONVESSITA' E VIF
# =====================================================================
VIF_SUMMARY_FILE = OUTPUT_DIR / "16_gamma_correlation_vif_check_2025_primary_247.csv"
VIF_MISSING_GAMMA_ROWS_FILE = OUTPUT_DIR / "16_gamma_correlation_vif_missing_gamma_rows_2025_primary_247.csv"
VIF_AUX_FILE = OUTPUT_DIR / "16_auxiliary_regression_vif_2025_primary_247.csv"

# ATTESE
# La griglia parte dalle 09:40: nessuna riga di frontiera delle 09:30 da scartare.
EXPECTED_0930_ROWS_DROPPED = 0

# Righe senza misure di flusso: con volume nullo su tutti gli strike il Herfindahl del
# flusso non e' definito (righe marcate low_vol in entrambe le fonti).
EXPECTED_MISSING_GAMMA_ROWS = 4

# Valori di riferimento da un pannello con MFIV senza troncamento e Herfindahl per contratto:
# non confrontabili con questi, riportati solo come traccia.
HISTORICAL_CORR_CBOE = 0.5244
HISTORICAL_CORR_INDEP = 0.5410
HISTORICAL_VIF_CBOE = 1.379
HISTORICAL_VIF_INDEP = 1.414

GAMMA_COLUMNS = [
    "H_OI_cboe",
    "H_VOL_cboe",
    "H_OI_indep",
    "H_VOL_indep",
]

# Regressori di IH3, per il VIF da regressione ausiliaria (1/(1-rho^2) vale solo con due
# regressori).
SPEC_REGRESSORS_CBOE = ["MFIV", "TrailingRV", "H_OI_cboe", "H_VOL_cboe"]
SPEC_REGRESSORS_INDEP = ["MFIV", "TrailingRV", "H_OI_indep", "H_VOL_indep"]


def require_columns(df: pd.DataFrame, columns, label: str):
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def numeric_convert_in_place(df: pd.DataFrame, columns) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        out[col] = pd.to_numeric(
            out[col].astype(str).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )

    return out


def pass_fail(actual, expected) -> int:
    return int(actual == expected)


def auxiliary_vif(df: pd.DataFrame, regressors) -> pd.DataFrame:
    """VIF = 1/(1-R^2), con R^2 della regressione di ciascun regressore su tutti gli altri."""
    rows = []

    work = df[regressors].dropna().astype(float)
    n = len(work)

    for target in regressors:
        others = [c for c in regressors if c != target]

        y = work[target].to_numpy()
        X = np.column_stack([np.ones(n)] + [work[c].to_numpy() for c in others])

        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta

        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        vif = 1.0 / (1.0 - r2) if np.isfinite(r2) and r2 < 1.0 else np.inf

        rows.append({
            "regressor": target,
            "regressed_on": ";".join(others),
            "r_squared_auxiliary": r2,
            "vif_true": vif,
            "n": n,
        })

    return pd.DataFrame(rows)


def gamma_vif():
    print("=" * 120)
    print("SECTION 3.8 GAMMA CORRELATION AND VIF CHECK")
    print("=" * 120)
    print(f"DATA_DIR:            {DATA_DIR}")
    print(f"Merged panel file:   {PANEL_FILE}")
    print("=" * 120)

    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Merged Section 3.8 panel not found:\n{PANEL_FILE}")

    print("Loading merged panel...")
    df = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)

    required_columns = ["quote_date", "grid_time"] + GAMMA_COLUMNS
    require_columns(df, required_columns, "Merged Section 3.8 panel")

    numeric_cols = GAMMA_COLUMNS + [
        c for c in ["MFIV", "MFIV_ann", "TrailingRV", "usable_for_flow_specs",
                    "missing_any_flow_predictor"]
        if c in df.columns
    ]

    print("Converting columns to numeric...")
    df = numeric_convert_in_place(df, numeric_cols)

    total_rows = len(df)

    EXPECTED_TOTAL_ROWS = total_rows
    EXPECTED_ROWS_AFTER_DROPPING_0930 = EXPECTED_TOTAL_ROWS - EXPECTED_0930_ROWS_DROPPED
    EXPECTED_ROWS_AFTER_DROPPING_MISSING_GAMMA = (
        EXPECTED_ROWS_AFTER_DROPPING_0930 - EXPECTED_MISSING_GAMMA_ROWS
    )

    print()
    print("=" * 120)
    print("STEP 1: RAW MERGED PANEL")
    print("=" * 120)
    print(f"Raw merged panel rows: {total_rows:,}")

    print()
    print("=" * 120)
    print("STEP 2: RIGHE DI FRONTIERA")
    print("=" * 120)

    usable_after_boundary = df[df["grid_time"] != "09:30:00"].copy()
    rows_dropped_boundary = total_rows - len(usable_after_boundary)

    print(f"Righe scartate per la frontiera delle 09:30: {rows_dropped_boundary}")
    print(f"Attese: {EXPECTED_0930_ROWS_DROPPED}")
    print("Con la griglia ancorata alle 09:40 non esiste piu' un punto di frontiera:")
    print("questo passo deve scartare ZERO righe. Se ne scartasse, la griglia non e'")
    print("quella che il resto della pipeline usa.")
    print(f"Passato: {pass_fail(rows_dropped_boundary, EXPECTED_0930_ROWS_DROPPED)}")

    print()
    print("=" * 120)
    print("STEP 3: SELEZIONE DELLE RIGHE UTILIZZABILI CON IL FLUSSO")
    print("=" * 120)

    # Criterio del pannello (usable_for_flow_specs); il dropna sulle colonne gamma serve solo
    # se la colonna manca.
    if "usable_for_flow_specs" in usable_after_boundary.columns:
        missing_gamma_rows = usable_after_boundary[
            usable_after_boundary["usable_for_flow_specs"] != 1
        ].copy()

        usable = usable_after_boundary[
            usable_after_boundary["usable_for_flow_specs"] == 1
        ].copy()

        selection_method = "usable_for_flow_specs dal pannello"
    else:
        missing_gamma_rows = usable_after_boundary[
            usable_after_boundary[GAMMA_COLUMNS].isna().any(axis=1)
        ].copy()

        usable = usable_after_boundary.dropna(subset=GAMMA_COLUMNS).copy()
        selection_method = "dropna sulle colonne gamma (fallback)"

    print(f"Metodo di selezione: {selection_method}")
    print(f"Righe utilizzabili: {len(usable):,}")
    print(f"Righe escluse:      {len(missing_gamma_rows)}  (attese {EXPECTED_MISSING_GAMMA_ROWS})")
    print(f"Passato: {pass_fail(len(missing_gamma_rows), EXPECTED_MISSING_GAMMA_ROWS)}")

    print()
    print("Righe escluse:")
    if missing_gamma_rows.empty:
        print("Nessuna.")
    else:
        show_cols = ["quote_date", "grid_time"] + GAMMA_COLUMNS
        show_cols += [c for c in ["gamma_flag_cboe", "gamma_flag_indep"]
                      if c in missing_gamma_rows.columns]
        print(missing_gamma_rows[show_cols].to_string(index=False))

    safe_write_csv(
        missing_gamma_rows[["quote_date", "grid_time"] + GAMMA_COLUMNS],
        VIF_MISSING_GAMMA_ROWS_FILE,
    )

    print()
    print("=" * 120)
    print("STEP 4: CORRELAZIONI FRA STOCK E FLUSSO")
    print("=" * 120)

    corr_cboe = usable["H_OI_cboe"].corr(usable["H_VOL_cboe"])
    corr_indep = usable["H_OI_indep"].corr(usable["H_VOL_indep"])

    print(f"corr(H_OI_cboe, H_VOL_cboe)   = {corr_cboe:.6f}")
    print(f"corr(H_OI_indep, H_VOL_indep) = {corr_indep:.6f}")
    print()
    print("Una correlazione lontana da uno sostiene che stock e flusso siano oggetti")
    print("distinti, che e' l'affermazione centrale della tesi sulla distinzione.")

    print()
    print("=" * 120)
    print("STEP 5: VIF A DUE REGRESSORI, LIMITE INFERIORE")
    print("=" * 120)

    vif_cboe = 1.0 / (1.0 - corr_cboe ** 2)
    vif_indep = 1.0 / (1.0 - corr_indep ** 2)

    print(f"VIF cboe  = {vif_cboe:.4f}")
    print(f"VIF indep = {vif_indep:.4f}")
    print()
    print("ATTENZIONE ALLA LETTURA. La formula 1/(1-rho^2) e' il VIF corretto solo")
    print("con DUE regressori. Le specifiche intraday ne hanno quattro (MFIV,")
    print("TrailingRV, H_OI, H_VOL), quindi questo numero e' un LIMITE INFERIORE del")
    print("VIF effettivo e non il VIF della specifica. Il VIF vero richiede la")
    print("regressione ausiliaria di ciascun regressore su tutti gli altri, ed e'")
    print("calcolato nel passo successivo.")

    print()
    print("=" * 120)
    print("STEP 6: VIF VERO PER REGRESSIONE AUSILIARIA")
    print("=" * 120)

    aux_frames = []

    for label, regressors in (("CBOE", SPEC_REGRESSORS_CBOE),
                              ("indep", SPEC_REGRESSORS_INDEP)):
        available = [c for c in regressors if c in usable.columns]

        if len(available) < len(regressors):
            print(f"  {label}: colonne mancanti "
                  f"{sorted(set(regressors) - set(available))}, salto.")
            continue

        aux = auxiliary_vif(usable, available)
        aux.insert(0, "gamma_source", label)
        aux_frames.append(aux)

        print()
        print(f"  Fonte gamma: {label}")
        print(aux.to_string(index=False))

    if aux_frames:
        aux_all = pd.concat(aux_frames, ignore_index=True)
        safe_write_csv(aux_all, VIF_AUX_FILE)

        worst = aux_all.loc[aux_all["vif_true"].idxmax()]
        print()
        print(f"VIF vero massimo: {worst['vif_true']:.4f} su {worst['regressor']} "
              f"({worst['gamma_source']})")
        print("Soglia convenzionale di allarme: 10. Sotto 5 la collinearita' non e'")
        print("un problema per l'interpretazione dei coefficienti.")
    else:
        aux_all = pd.DataFrame()
        print("  Nessuna regressione ausiliaria calcolata.")

    print()
    print("=" * 120)
    print("RIEPILOGO")
    print("=" * 120)

    summary_rows = [
        {"metric": "raw_merged_panel_rows", "actual": total_rows,
         "expected_or_reference": EXPECTED_TOTAL_ROWS,
         "difference": 0, "passed_exact_expected": 1},
        {"metric": "rows_dropped_at_0930_boundary", "actual": rows_dropped_boundary,
         "expected_or_reference": EXPECTED_0930_ROWS_DROPPED,
         "difference": rows_dropped_boundary - EXPECTED_0930_ROWS_DROPPED,
         "passed_exact_expected": pass_fail(rows_dropped_boundary,
                                            EXPECTED_0930_ROWS_DROPPED)},
        {"metric": "rows_excluded_missing_flow", "actual": len(missing_gamma_rows),
         "expected_or_reference": EXPECTED_MISSING_GAMMA_ROWS,
         "difference": len(missing_gamma_rows) - EXPECTED_MISSING_GAMMA_ROWS,
         "passed_exact_expected": pass_fail(len(missing_gamma_rows),
                                            EXPECTED_MISSING_GAMMA_ROWS)},
        {"metric": "usable_rows", "actual": len(usable),
         "expected_or_reference": EXPECTED_ROWS_AFTER_DROPPING_MISSING_GAMMA,
         "difference": len(usable) - EXPECTED_ROWS_AFTER_DROPPING_MISSING_GAMMA,
         "passed_exact_expected": pass_fail(
             len(usable), EXPECTED_ROWS_AFTER_DROPPING_MISSING_GAMMA)},
        {"metric": "corr_H_OI_cboe_H_VOL_cboe", "actual": corr_cboe,
         "expected_or_reference": HISTORICAL_CORR_CBOE,
         "difference": corr_cboe - HISTORICAL_CORR_CBOE,
         "passed_exact_expected": ""},
        {"metric": "corr_H_OI_indep_H_VOL_indep", "actual": corr_indep,
         "expected_or_reference": HISTORICAL_CORR_INDEP,
         "difference": corr_indep - HISTORICAL_CORR_INDEP,
         "passed_exact_expected": ""},
        {"metric": "vif_two_regressor_cboe", "actual": vif_cboe,
         "expected_or_reference": HISTORICAL_VIF_CBOE,
         "difference": vif_cboe - HISTORICAL_VIF_CBOE,
         "passed_exact_expected": ""},
        {"metric": "vif_two_regressor_indep", "actual": vif_indep,
         "expected_or_reference": HISTORICAL_VIF_INDEP,
         "difference": vif_indep - HISTORICAL_VIF_INDEP,
         "passed_exact_expected": ""},
    ]

    if not aux_all.empty:
        for _, r in aux_all.iterrows():
            summary_rows.append({
                "metric": f"vif_true_{r['gamma_source']}_{r['regressor']}",
                "actual": r["vif_true"],
                "expected_or_reference": "",
                "difference": "",
                "passed_exact_expected": "",
            })

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    safe_write_csv(summary, VIF_SUMMARY_FILE)

    print()
    print("I valori di riferimento storici (correlazioni 0,5244 e 0,5410, VIF 1,379 e")
    print("1,414) vengono dal pannello costruito sulla MFIV senza troncamento e sul")
    print("Herfindahl per contratto. NON sono confrontabili con questi e restano")
    print("solo come traccia.")

    print()
    print("Output:")
    print(f"    {VIF_SUMMARY_FILE.name}")
    print(f"    {VIF_MISSING_GAMMA_ROWS_FILE.name}")
    print(f"    {VIF_AUX_FILE.name}")
    print("=" * 120)


# =====================================================================
# PARTE 5: CHE TIPO DI NULLI SONO: DIMENSIONI D'EFFETTO E IC
# =====================================================================
# Un risultato non significativo dice poco finche' non si sa quali effetti i dati escludono:
# per i termini di gamma, effetto di una deviazione standard con IC al 95%, dai risultati gia'
# stimati.
#
# Effetto di 1 sd: coef * sd in % della media della dipendente (livelli), 100 (exp(coef sd) - 1)
# (log). Intraday: sd entro gli effetti fissi, t con G-1 gradi di liberta' (cluster per giorno).
# Giornaliero: sd del campione, t con n-k gradi di liberta' (Newey-West).
#
# Intraday con entrambi i regressori, perche' il risultato su H_VOL cambia con il regressore,
# e con le giornate effettive della parte 1.

DAILY_RESULTS = ("13_daily_regressions", "13_daily_regression_results_2025.csv")
EFFECT_SIZE_OUTPUT = (OUTPUT_FOLDER, "16_gamma_effect_size_ci.csv")


def demean_ci(x: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    """Proiezioni alternate su un solo vettore (parte 5), come demean."""
    x = np.asarray(x, dtype=float).copy()
    x -= x.mean()
    codes = [pd.factorize(frame[c].astype(str))[0] for c in FE_COLS]
    counts = [np.bincount(c) for c in codes]
    scale = max(x.std(), 1e-300)
    for _ in range(DEMEAN_MAXIT):
        old = x.copy()
        for c, cnt in zip(codes, counts):
            x -= (np.bincount(c, weights=x) / cnt)[c]
        if np.max(np.abs(x - old)) / scale <= DEMEAN_TOL:
            break
    return x


def effect(coef, se, sd, dep_mean, form, df):
    tcrit = stats.t.ppf(0.975, df)
    lo, hi = coef - tcrit * se, coef + tcrit * se
    if form == "levels":
        f = lambda b: 100.0 * b * sd / dep_mean
    else:
        f = lambda b: 100.0 * (np.exp(b * sd) - 1.0)
    e, a, b = f(coef), f(lo), f(hi)
    return e, min(a, b), max(a, b)


def gamma_effect_size_ci():
    # INTRADAY
    print("=" * 140)
    print("CHE TIPO DI NULLI SONO: EFFETTO DI UNA DEVIAZIONE STANDARD DEL GAMMA, CON IC AL 95%")
    print("=" * 140)

    checks = []
    rows = []

    for p in (path(PANEL), path(REGISTERED)):
        require(p)

    ip = pd.read_csv(path(PANEL), dtype=str, keep_default_na=False)
    ip = ip[ip["quote_date"].astype(str).str.startswith(YEAR)].copy()
    for c in (["MFIV", "MFIV_ann", "TrailingRV", "H_OI_cboe", "H_VOL_cboe", "H_OI_indep",
               "H_VOL_indep", "usable_for_flow_specs"]
              + [f"FwdRV_{h}" for h in HORIZONS] + [f"h{h}_valid" for h in HORIZONS]):
        ip[c] = pd.to_numeric(ip[c], errors="coerce")
    for c in FE_COLS:
        ip[c] = ip[c].astype(str).str.strip()
    ip["n_days"] = ip["quote_date"].nunique()

    ir = pd.read_csv(path(REGISTERED), keep_default_na=False)
    for c in ("coef", "se", "p_value", "n", "horizon"):
        ir[c] = pd.to_numeric(ir[c], errors="coerce")
    ir = ir[(ir["specification"] == "IH3") & (ir["vcov"] == "giorno")]

    conc = {}
    for form, pair in CONCENTRATION.items():
        p = path(pair)
        if p.exists():
            c = pd.read_csv(p, keep_default_na=False)
            c["horizon"] = pd.to_numeric(c["horizon"], errors="coerce")
            conc[form] = c
        else:
            print(f"  Nota: file di concentrazione {form} non trovato, colonna vuota: {p}")

    G = int(ip["quote_date"].nunique())
    n_mismatch = 0

    for iv in ("MFIV", "MFIV_ann"):
        for src in ("cboe", "indep"):
            for h in HORIZONS:
                for form in ("levels", "log"):
                    ivc = iv if form == "levels" else f"log_{iv}"
                    w = ip[(ip[f"h{h}_valid"] == 1) & (ip["usable_for_flow_specs"] == 1)].copy()
                    dep = f"FwdRV_{h}"
                    regs = [iv, "TrailingRV", f"H_OI_{src}", f"H_VOL_{src}"]
                    w = w.dropna(subset=[dep] + regs + FE_COLS)
                    w = w[(w[dep] > 0) & (w[iv] > 0) & (w["TrailingRV"] > 0)] if form == "log" else w
                    w = w[np.isfinite(w[[dep] + regs]).all(axis=1)]
                    dep_mean = float(w[dep].mean())
                    for kind in ("H_OI", "H_VOL"):
                        term = f"{kind}_{src}"
                        x = ir[(ir["regressor"] == iv) & (ir["form"] == form) & (ir["gamma_source"] == src)
                               & (ir["horizon"] == h) & (ir["term"] == term)]
                        if len(x) != 1:
                            n_mismatch += 1
                            continue
                        x = x.iloc[0]
                        if int(x["n"]) != len(w):
                            n_mismatch += 1
                        sd = float(np.std(demean_ci(w[term].to_numpy(), w), ddof=1))
                        e, lo, hi = effect(float(x["coef"]), float(x["se"]), sd, dep_mean, form, G - 1)
                        neff = np.nan
                        if form in conc:
                            cc = conc[form]
                            yy = cc[(cc["regressor"] == iv) & (cc["specification"] == "IH3")
                                    & (cc["gamma_source"] == src) & (cc["horizon"] == h) & (cc["term"] == term)]
                            if len(yy):
                                neff = float(yy["var_day_n_eff"].iloc[0])
                        rows.append({"design": "intraday", "form": form, "regressor": iv, "source": src,
                                     "horizon": h, "term": kind, "coef": float(x["coef"]),
                                     "p_value": float(x["p_value"]), "sd_used": sd,
                                     "effect_pct": e, "ci_low_pct": lo, "ci_high_pct": hi,
                                     "excludes_beyond_pct": max(abs(lo), abs(hi)),
                                     "effective_days": neff, "n": int(x["n"])})

    checks.append(("intraday: ogni cella trovata una volta e con la numerosita' delle stime",
                   n_mismatch == 0, f"celle non corrispondenti: {n_mismatch}"))

    # GIORNALIERO
    for p in (path(DAILY_PANEL), path(DAILY_RESULTS)):
        require(p)

    dp = pd.read_csv(path(DAILY_PANEL), dtype=str, keep_default_na=False)
    dp = dp[dp["quote_date"].astype(str).str.startswith(YEAR)].copy()
    for c in ("RV_t", "MFIV_t", "RV_lag1", "H_OI_t", "H_VOL_t", "H_OI_t_indep", "H_VOL_t_indep"):
        dp[c] = pd.to_numeric(dp[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    dr = pd.read_csv(path(DAILY_RESULTS), keep_default_na=False)
    for c in ("coef", "se_nw", "p_value", "n"):
        dr[c] = pd.to_numeric(dr[c], errors="coerce")

    d_mismatch = 0
    for spec, k in (("A5", 4), ("A6", 5)):
        for src, hoi, hvol in (("CBOE", "H_OI_t", "H_VOL_t"), ("indep", "H_OI_t_indep", "H_VOL_t_indep")):
            regs = ["MFIV_t", "RV_lag1", hoi] + ([hvol] if spec == "A6" else [])
            for form in ("levels", "log"):
                w = dp.dropna(subset=["RV_t"] + regs)
                if form == "log":
                    w = w[(w["RV_t"] > 0) & (w["MFIV_t"] > 0) & (w["RV_lag1"] > 0)]
                dep_mean = float(w["RV_t"].mean())
                for kind, col in (("H_OI", hoi), ("H_VOL", hvol)):
                    if col not in regs:
                        continue
                    x = dr[(dr["specification"] == spec) & (dr["form"] == form)
                           & (dr["gamma_source"] == src) & (dr["term"] == col)]
                    if len(x) != 1:
                        d_mismatch += 1
                        continue
                    x = x.iloc[0]
                    if int(x["n"]) != len(w):
                        d_mismatch += 1
                    sd = float(w[col].std(ddof=1))
                    e, lo, hi = effect(float(x["coef"]), float(x["se_nw"]), sd, dep_mean, form, int(x["n"]) - k)
                    rows.append({"design": "giornaliero", "form": form, "regressor": "MFIV_t", "source": src,
                                 "horizon": np.nan, "term": f"{spec}:{kind}", "coef": float(x["coef"]),
                                 "p_value": float(x["p_value"]), "sd_used": sd,
                                 "effect_pct": e, "ci_low_pct": lo, "ci_high_pct": hi,
                                 "excludes_beyond_pct": max(abs(lo), abs(hi)),
                                 "effective_days": np.nan, "n": int(x["n"])})

    checks.append(("giornaliero: ogni cella trovata una volta e con la numerosita' delle stime",
                   d_mismatch == 0, f"celle non corrispondenti: {d_mismatch}"))

    res = pd.DataFrame(rows)
    res.to_csv(path(EFFECT_SIZE_OUTPUT), index=False)

    # STAMPA
    print()
    print("CONTROLLI")
    for n, ok, d in checks:
        print(f"  [{'OK' if ok else 'FALLITO'}] {n}: {d}")

    fmt = lambda v: f"{v:.3g}"
    show = ["form", "regressor", "source", "horizon", "term", "coef", "p_value",
            "effect_pct", "ci_low_pct", "ci_high_pct", "effective_days"]

    print()
    print("=" * 140)
    print("INTRADAY, IH3, cluster per giorno (effetto di 1 sd dentro gli effetti fissi)")
    print("=" * 140)
    print("Livelli: % della varianza realizzata media nella finestra. Log: variazione % della varianza.")
    for form in ("levels", "log"):
        print(f"\n-- forma {form} --")
        x = res[(res["design"] == "intraday") & (res["form"] == form)]
        print(x[show].to_string(index=False, float_format=fmt))

    print()
    print("=" * 140)
    print("GIORNALIERO, Newey-West (effetto di 1 sd del campione)")
    print("=" * 140)
    x = res[res["design"] == "giornaliero"]
    print(x[[c for c in show if c not in ("horizon", "effective_days")]].to_string(index=False, float_format=fmt))

    print()
    print("COME SI LEGGE. effect_pct e' l'effetto stimato; l'intervallo dice quali effetti i dati")
    print("non escludono. Se l'intervallo sta tutto vicino a zero, il nullo e' informativo; se e'")
    print("largo, i dati non bastano. Nelle celle intraday con poche giornate effettive l'intervallo")
    print("stesso e' poco affidabile (Sezione 81).")
    print(f"Output: {EFFECT_SIZE_OUTPUT[1]}")
    print("=" * 140)

    if not all(ok for _, ok, _ in checks):
        raise RuntimeError("Dimensioni d'effetto: controlli falliti.")


# =====================================================================
# PARTE 6: H_VOL CON I CONTROLLI PER L'AMPIEZZA DEGLI SCAMBI
# =====================================================================
# Se l'associazione negativa di H_VOL con la varianza realizzata regge ai controlli per gli
# scambi: logN, log del numero di strike con volume positivo (H >= 1/N, quindi un H_VOL basso
# puo' voler dire solo piu' strike scambiati), e logG, log del volume pesato per il gamma.
#
# Intraday (I3): FwdRV_h ~ X + TrailingRV + H_OI + H_VOL [+ controlli] | date + week x time, con
# X = MFIV_ann o MFIV in livelli e log MFIV_ann in log, cluster per giorno. Giornaliero (A4),
# solo diagnostico perche' volume e RV_t sono della stessa sessione, con Newey-West.
#
# Verdetti su H_VOL, insieme NG contro base: S1 stesso segno, p < 0.05, |b_NG| >= 0.5 |b_base|;
# S2a stesso segno, |b_NG| < 0.5 |b_base|; S2b stesso segno, |b_NG| >= 0.5 |b_base|, p >= 0.05;
# S3 segno opposto; B0 base non significativa. Conta solo se le due fonti di gamma concordano.
#
# Celle di decisione: intraday in log e in livelli con MFIV, dove la base e' negativa e
# significativa; A4 log con MFIV_t e' diagnostica, le altre celle sono solo riportate.
#
# Prima si riproducono, fermandosi se un controllo fallisce, le medie giornaliere di H_VOL e
# G_VOL_raw del pannello, l'unione dei file gamma con il pannello intraday e le stime base
# della tesi; su ogni minuto deve valere H_VOL x N >= 1.

PANEL_D = os.path.join(*DAILY_PANEL)
PANEL_I = os.path.join(*PANEL)
GAMMA = {"vendor": (os.path.join("04_gamma_herfindahl", "04_gamma_cboe_intraday_2025_all.csv"),
                    "h_vol_cboe", "g_vol_raw_cboe"),
         "black76": (os.path.join("04_gamma_herfindahl", "04_gamma_intraday_2025_all.csv"),
                     "h_vol", "g_vol_raw")}
MFIV_I = os.path.join("03_mfiv", "03_mfiv_intraday_2025_all.csv")

D_SRC = {"vendor": ("H_OI_t", "H_VOL_t", "G_VOL_raw_t"),
         "black76": ("H_OI_t_indep", "H_VOL_t_indep", "G_VOL_raw_t_indep")}
I_SRC = {"vendor": ("H_OI_cboe", "H_VOL_cboe", "G_VOL_raw_cboe"),
         "black76": ("H_OI_indep", "H_VOL_indep", "G_VOL_raw_indep")}
CONTROL_SETS = {"base": [], "N": ["logN"], "G": ["logG"], "NG": ["logN", "logG"]}
D_REGRESSORS = ["MFIV_t", "MFIV_ann_t", "MFIV_session_t"]
I_SPECS = {"levels_MFIV_ann": ("levels", "MFIV_ann"),
           "levels_MFIV": ("levels", "MFIV"),
           "log_MFIV_ann": ("log", "MFIV_ann")}

# Coefficienti ed errori standard di A4 base (gamma vendor, MFIV_t) come stampati nella tesi.
D_TARGETS = {("levels", "X"): ("3.68", "0.281"), ("levels", "RV_lag1"): ("-0.294", "0.0377"),
             ("levels", "H_OI"): ("-1.02e-4", "1.22e-4"), ("levels", "H_VOL"): ("7.04e-4", "2.25e-4"),
             ("log", "X"): ("0.841", "0.133"), ("log", "H_OI"): ("-0.121", "0.746"),
             ("log", "H_VOL"): ("-5.97", "1.98")}
# Lo stesso per I3 base (gamma vendor): livelli con MFIV_ann e con MFIV, log con MFIV_ann.
I_TARGETS = {
    ("levels_MFIV_ann", 5, "H_VOL"): ("3.89e-6", "2.69e-6"),
    ("levels_MFIV_ann", 15, "H_VOL"): ("4.98e-6", "5.57e-6"),
    ("levels_MFIV_ann", 30, "H_VOL"): ("-1.68e-6", "1.03e-5"),
    ("levels_MFIV_ann", 60, "H_VOL"): ("-2.44e-5", "1.83e-5"),
    ("levels_MFIV_ann", 5, "X"): ("2.62e-5", "8.17e-6"),
    ("levels_MFIV", 5, "H_VOL"): ("-4.42e-6", "1.59e-6"),
    ("levels_MFIV", 15, "H_VOL"): ("-1.10e-5", "3.07e-6"),
    ("levels_MFIV", 30, "H_VOL"): ("-2.68e-5", "6.15e-6"),
    ("levels_MFIV", 60, "H_VOL"): ("-5.84e-5", "1.43e-5"),
    ("log_MFIV_ann", 5, "H_VOL"): ("-1.16", "0.535"),
    ("log_MFIV_ann", 15, "H_VOL"): ("-1.13", "0.411"),
    ("log_MFIV_ann", 30, "H_VOL"): ("-0.928", "0.381"),
    ("log_MFIV_ann", 60, "H_VOL"): ("-0.735", "0.367"),
    ("log_MFIV_ann", 5, "H_OI"): ("-0.291", "0.618"),
    ("log_MFIV_ann", 60, "H_OI"): ("-1.70", "0.838"),
    ("log_MFIV_ann", 5, "X"): ("1.34", "0.0924"),
}
I_N = {5: 9382, 15: 9135, 30: 8888, 60: 8147}
I_R2W = {"levels_MFIV_ann": ["0.140", "0.178", "0.266", "0.398"],
         "levels_MFIV": ["0.124", "0.149", "0.230", "0.364"],
         "log_MFIV_ann": ["0.150", "0.223", "0.229", "0.223"]}


def printed_match(value, printed):
    t = float(printed)
    mant = printed.lower().lstrip("+-").split("e")[0].replace(".", "").lstrip("0")
    sig = max(len(mant), 1)
    half = 0.5 * 10 ** (math.floor(math.log10(abs(t))) - (sig - 1))
    return abs(value - t) <= half * 1.000001


def nw_lag(n):
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def hhmmss(s):
    s = s.astype(str).str.strip()
    return s.where(s.str.len() != 5, s + ":00")


def read_str_csv(path, usecols=None):
    if not os.path.isfile(path):
        raise SystemExit("STOP: file not found: " + path)
    return pd.read_csv(path, keep_default_na=False, dtype=str, usecols=usecols)


def num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    return df


def classify(b0, p0, b1, p1):
    if p0 >= 0.05:
        return "B0 (base not significant: no association to explain)"
    if np.sign(b1) != np.sign(b0):
        return "S3"
    if abs(b1) < 0.5 * abs(b0):
        return "S2a"
    if p1 < 0.05:
        return "S1"
    return "S2b"


def herfindahl_floor_gate(g, hcol, src, log):
    """H >= 1/N must hold minute by minute if N is counted on the universe of H."""
    z = g[[hcol, "n_vol_positive_strikes"]].dropna()
    prod = z[hcol] * z["n_vol_positive_strikes"]
    n_bad = int((prod < 1.0 - 1e-9).sum())
    log.append("  floor %-7s minutes %d, min H_VOL x N %.6f, violations %d" % (
        src, len(z), float(prod.min()) if len(z) else np.nan, n_bad))
    if n_bad > 0:
        print("\n".join(log))
        raise SystemExit("STOP: H_VOL < 1/N on %d minutes (%s): n_vol_positive_strikes is not "
                         "counted on the row universe of H_VOL." % (n_bad, src))


def daily_breadth(data_dir, panel, log):
    """Session means of n_vol_positive_strikes and g_vol_raw over the minutes that
    reproduce the panel's H_VOL_t and G_VOL_raw_t."""
    out = pd.DataFrame({"quote_date": panel["quote_date"]})
    mfiv_flags = None
    for src, (rel, hcol, gcol) in GAMMA.items():
        g = read_str_csv(os.path.join(data_dir, rel),
                         ["quote_date", "quote_time", "gamma_flag", hcol, gcol, "n_vol_positive_strikes"])
        g = num(g, [hcol, gcol, "n_vol_positive_strikes"])
        g["quote_time"] = hhmmss(g["quote_time"])
        herfindahl_floor_gate(g, hcol, src, log)
        _, hpan, gpan = D_SRC[src][0], D_SRC[src][1], D_SRC[src][2]
        target = panel.set_index("quote_date")[[hpan, gpan]]
        chosen = None
        for rule in ["A_gamma_ok", "B_gamma_and_mfiv_ok", "C_all_nonmissing"]:
            if rule == "A_gamma_ok":
                mask = g["gamma_flag"].str.strip() == "ok"
            elif rule == "B_gamma_and_mfiv_ok":
                if mfiv_flags is None:
                    m = read_str_csv(os.path.join(data_dir, MFIV_I), ["quote_date", "quote_time", "mfiv_flag"])
                    m["quote_time"] = hhmmss(m["quote_time"])
                    mfiv_flags = m
                gg = g.merge(mfiv_flags, on=["quote_date", "quote_time"], how="left",
                             validate="m:1")
                if len(gg) != len(g):
                    raise SystemExit("STOP: the MFIV flags do not merge one to one on the gamma minutes.")
                mask = ((gg["gamma_flag"].str.strip() == "ok")
                        & (gg["mfiv_flag"].fillna("").str.strip() == "ok")).to_numpy()
            else:
                mask = g[hcol].notna()
            sub = g[np.asarray(mask)]
            means = sub.groupby("quote_date")[[hcol, gcol, "n_vol_positive_strikes"]].mean()
            means = means.reindex(target.index)
            rel_h = ((means[hcol] - target[hpan]).abs() / target[hpan].abs()).max()
            rel_g = ((means[gcol] - target[gpan]).abs() / target[gpan].abs()).max()
            log.append("  daily rule %-22s %-7s max rel diff H_VOL %.2e, G_VOL %.2e" % (
                rule, src, rel_h, rel_g))
            if np.isfinite(rel_h) and np.isfinite(rel_g) and rel_h < 1e-9 and rel_g < 1e-9:
                chosen = rule
                out["N_vol_mean_" + src] = means["n_vol_positive_strikes"].to_numpy()
                out["rule_" + src] = rule
                break
        if chosen is None:
            return None
    return out


def daily_fit(d, reg, form, src, cset):
    hoi, hvol, gvol = D_SRC[src]
    cols = ["RV_t", reg, "RV_lag1", hoi, hvol, "N_vol_mean_" + src, gvol]
    z = d[cols].dropna()
    if (z["N_vol_mean_" + src] <= 0).any() or (z[gvol] <= 0).any():
        raise SystemExit("STOP: non-positive N or G_VOL_raw in the daily sample (%s)." % src)
    y = z["RV_t"].to_numpy(float)
    X = pd.DataFrame({"X": z[reg].to_numpy(float), "RV_lag1": z["RV_lag1"].to_numpy(float),
                      "H_OI": z[hoi].to_numpy(float), "H_VOL": z[hvol].to_numpy(float)})
    if form == "log":
        y = np.log(y)
        X["X"] = np.log(X["X"])
        X["RV_lag1"] = np.log(X["RV_lag1"])
    if "logN" in CONTROL_SETS[cset]:
        X["logN"] = np.log(z["N_vol_mean_" + src].to_numpy(float))
    if "logG" in CONTROL_SETS[cset]:
        X["logG"] = np.log(z[gvol].to_numpy(float))
    n = len(y)
    res = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="HAC", cov_kwds={"maxlags": nw_lag(n), "use_correction": False}, use_t=True)
    rows = []
    for term in res.params.index:
        rows.append({"design": "daily_A4", "spec": reg, "form": form, "gamma_source": src,
                     "controls": cset, "term": term, "coef": float(res.params[term]),
                     "se": float(res.bse[term]), "p_value": float(res.pvalues[term]),
                     "n": n, "r_squared": float(res.rsquared)})
    return rows


def intraday_fit(ip, spec, h, src, cset, dropped_log):
    form, xcol = I_SPECS[spec]
    hoi, hvol, gvol = I_SRC[src]
    ycol = "FwdRV_%d" % h
    m = ip["usable_for_flow_specs_b"] & ip["h%d_valid_b" % h]
    cols = [ycol, xcol, "TrailingRV", hoi, hvol, "nvol_" + src, gvol, "date_fe", "week_x_time_fe"]
    z = ip.loc[m, cols].dropna().copy()
    if (z["nvol_" + src] <= 0).any() or (z[gvol] <= 0).any():
        raise SystemExit("STOP: non-positive N or G_VOL_raw in the intraday sample (%s, h=%d)." % (src, h))
    z = z.rename(columns={ycol: "y", xcol: "x", "TrailingRV": "trv", hoi: "hoi", hvol: "hvol"})
    if form == "log":
        pos = (z["y"] > 0) & (z["x"] > 0) & (z["trv"] > 0)
        if (~pos).any():
            dropped_log.append((spec, h, src, int((~pos).sum())))
        z = z[pos]
        z["y"] = np.log(z["y"])
        z["x"] = np.log(z["x"])
        z["trv"] = np.log(z["trv"])
    z["logN"] = np.log(z["nvol_" + src])
    z["logG"] = np.log(z[gvol])
    rhs = "x + trv + hoi + hvol" + "".join(" + " + c for c in CONTROL_SETS[cset])
    fit = pf.feols("y ~ " + rhs + " | date_fe + week_x_time_fe", data=z,
                   vcov={"CRV1": "date_fe"})
    tid = fit.tidy()
    r2w = float(getattr(fit, "_r2_within", np.nan))
    rows = []
    for term in tid.index:
        name = {"x": "X", "trv": "TrailingRV", "hoi": "H_OI", "hvol": "H_VOL"}.get(term, term)
        rows.append({"design": "intraday_I3", "spec": spec, "horizon": h, "gamma_source": src,
                     "controls": cset, "term": name, "coef": float(tid.loc[term, "Estimate"]),
                     "se": float(tid.loc[term, "Std. Error"]),
                     "p_value": float(tid.loc[term, "Pr(>|t|)"]), "n": int(fit._N),
                     "r2_within": r2w})
    return rows


def within_r2_hvol_on_controls(ip, src):
    """Share of the within (date, week x time) variation of H_VOL explained by logN, logG."""
    hoi, hvol, gvol = I_SRC[src]
    m = ip["usable_for_flow_specs_b"] & ip["h5_valid_b"]
    z = ip.loc[m, [hvol, "nvol_" + src, gvol, "date_fe", "week_x_time_fe"]].dropna().copy()
    z = z.rename(columns={hvol: "hvol"})
    z["logN"] = np.log(z["nvol_" + src])
    z["logG"] = np.log(z[gvol])
    out = {}
    for name, rhs in (("logN", "logN"), ("logG", "logG"), ("logN+logG", "logN + logG")):
        f = pf.feols("hvol ~ " + rhs + " | date_fe + week_x_time_fe", data=z,
                     vcov={"CRV1": "date_fe"})
        out[name] = float(getattr(f, "_r2_within", np.nan))
    return out


def hvol_breadth_controls():
    data_dir = str(DATA_DIR)
    log = ["16_Intraday_robustness  data: " + data_dir]
    fails = []

    # ---------------- daily
    panel = read_str_csv(os.path.join(data_dir, PANEL_D))
    numc = ["RV_t", "RV_lag1", "MFIV_t", "MFIV_ann_t", "MFIV_session_t"] + \
           [c for v in D_SRC.values() for c in v]
    panel = num(panel, numc).sort_values("quote_date").reset_index(drop=True)
    log.append("GATE 1, daily means:")
    breadth = daily_breadth(data_dir, panel, log)
    if breadth is None:
        print("\n".join(log))
        raise SystemExit("STOP: no rule reproduces the panel's H_VOL_t and G_VOL_raw_t.")
    d = panel.merge(breadth, on="quote_date", how="left")
    drows = []
    for reg in D_REGRESSORS:
        for form in ["levels", "log"]:
            for src in D_SRC:
                for cset in CONTROL_SETS:
                    drows.extend(daily_fit(d, reg, form, src, cset))
    dres = pd.DataFrame(drows)
    dres["term"] = dres["term"].replace({"const": "const"})

    log.append("GATE 3, daily base A4 (vendor, MFIV_t) against Tables D.3-D.4:")
    for (form, term), (c, s) in D_TARGETS.items():
        r = dres[(dres.spec == "MFIV_t") & (dres.form == form) & (dres.gamma_source == "vendor")
                 & (dres.controls == "base") & (dres.term == term)].iloc[0]
        ok = printed_match(r["coef"], c) and printed_match(r["se"], s)
        log.append("  %-6s %-8s coef %s se %s  got %.4g (%.4g)  %s" % (
            form, term, c, s, r["coef"], r["se"], "ok" if ok else "FAIL"))
        if not ok:
            fails.append(("daily", form, term))

    # ---------------- intraday
    ip = read_str_csv(os.path.join(data_dir, PANEL_I))
    inum = ["FwdRV_5", "FwdRV_15", "FwdRV_30", "FwdRV_60", "TrailingRV", "MFIV", "MFIV_ann"] + \
           [c for v in I_SRC.values() for c in v]
    ip = num(ip, inum)
    ip["usable_for_flow_specs_b"] = as_bool(ip["usable_for_flow_specs"])
    for h in HORIZONS:
        ip["h%d_valid_b" % h] = as_bool(ip["h%d_valid" % h])
    ip["date_fe"] = ip["date_fe"].astype(str)
    ip["week_x_time_fe"] = ip["week_x_time_fe"].astype(str)
    ip["grid_key"] = hhmmss(ip["grid_time"])
    log.append("GATE 2, intraday merge:")
    for src, (rel, hcol, gcol) in GAMMA.items():
        g = read_str_csv(os.path.join(data_dir, rel), ["quote_date", "quote_time", hcol, "n_vol_positive_strikes"])
        g = num(g, [hcol, "n_vol_positive_strikes"])
        g["grid_key"] = hhmmss(g["quote_time"])
        g = g.drop(columns=["quote_time"]).rename(columns={hcol: "hchk_" + src,
                                                           "n_vol_positive_strikes": "nvol_" + src})
        if g.duplicated(["quote_date", "grid_key"]).any():
            raise SystemExit("STOP: duplicate minutes in " + rel)
        ip = ip.merge(g, on=["quote_date", "grid_key"], how="left")
        hpan = I_SRC[src][1]
        both = ip[[hpan, "hchk_" + src]].dropna()
        unmatched = int(ip["nvol_" + src].isna().sum())
        rel_diff = float(((both[hpan] - both["hchk_" + src]).abs()
                          / both[hpan].abs()).max()) if len(both) else np.inf
        log.append("  %-7s rows %d, unmatched %d, compared %d, max rel diff H_VOL %.2e" % (
            src, len(ip), unmatched, len(both), rel_diff))
        if unmatched > 0 or not rel_diff < 1e-9:
            print("\n".join(log))
            raise SystemExit("STOP: the gamma file does not merge exactly on the intraday panel.")

    irows, dropped_log = [], []
    for spec in I_SPECS:
        for h in HORIZONS:
            for src in I_SRC:
                for cset in CONTROL_SETS:
                    irows.extend(intraday_fit(ip, spec, h, src, cset, dropped_log))
    ires = pd.DataFrame(irows)
    for item in dropped_log:
        log.append("  log form: %s h=%d %s dropped %d non-positive rows" % item)

    log.append("GATE 4, intraday base I3 (vendor) against Tables D.5-D.7:")
    for (spec, h, term), (c, s) in I_TARGETS.items():
        r = ires[(ires.spec == spec) & (ires.horizon == h) & (ires.gamma_source == "vendor")
                 & (ires.controls == "base") & (ires.term == term)].iloc[0]
        ok = printed_match(r["coef"], c) and printed_match(r["se"], s)
        log.append("  %-16s h=%-2d %-6s coef %s se %s  got %.4g (%.4g)  %s" % (
            spec, h, term, c, s, r["coef"], r["se"], "ok" if ok else "FAIL"))
        if not ok:
            fails.append(("intraday", spec, h, term))
    for spec, r2s in I_R2W.items():
        for h, r2 in zip(HORIZONS, r2s):
            r = ires[(ires.spec == spec) & (ires.horizon == h) & (ires.gamma_source == "vendor")
                     & (ires.controls == "base") & (ires.term == "H_VOL")].iloc[0]
            ok = printed_match(r["r2_within"], r2) and int(r["n"]) == I_N[h]
            log.append("  %-16s h=%-2d within R2 %s n %d  got %.4f %d  %s" % (
                spec, h, r2, I_N[h], r["r2_within"], r["n"], "ok" if ok else "FAIL"))
            if not ok:
                fails.append(("intraday_r2n", spec, h))

    if fails:
        print("\n".join(log))
        raise SystemExit("STOP: reproduction failed on %d items; nothing new is read." % len(fails))

    # ---------------- diagnostica: quanta parte di H_VOL e' ampiezza e quantita'
    log.append("")
    log.append("DIAGNOSTIC: within R2 of H_VOL on the controls (date and week x time effects, h=5 rows)")
    for src in I_SRC:
        w = within_r2_hvol_on_controls(ip, src)
        log.append("  %-7s logN %.3f | logG %.3f | logN+logG %.3f" % (
            src, w["logN"], w["logG"], w["logN+logG"]))
    for src in D_SRC:
        hvol = D_SRC[src][1]
        z = d[[hvol, "N_vol_mean_" + src, D_SRC[src][2]]].dropna()
        c1 = np.corrcoef(z[hvol], np.log(z["N_vol_mean_" + src]))[0, 1]
        c2 = np.corrcoef(z[hvol], np.log(z[D_SRC[src][2]]))[0, 1]
        log.append("  daily %-7s corr(H_VOL_d, logN_d) %.3f | corr(H_VOL_d, logG_d) %.3f" % (src, c1, c2))

    # ---------------- verdetti
    def cell(res, key_cols, key_vals, cset):
        q = res.copy()
        for k, v in zip(key_cols, key_vals):
            q = q[q[k] == v]
        q = q[(q.controls == cset) & (q.term == "H_VOL")]
        return q.iloc[0]

    log.append("")
    log.append("VERDICTS (H_VOL, set NG against base; S1 survives / S2a attenuated / "
               "S2b not separable / S3 reversed / B0 no base association)")
    log.append(" intraday:")
    for spec in I_SPECS:
        decision = spec in ("log_MFIV_ann", "levels_MFIV")
        for h in HORIZONS:
            parts, classes = [], []
            for src in I_SRC:
                b0 = cell(ires, ["spec", "horizon", "gamma_source"], [spec, h, src], "base")
                parts_src = []
                for cset in ["N", "G", "NG"]:
                    b1 = cell(ires, ["spec", "horizon", "gamma_source"], [spec, h, src], cset)
                    parts_src.append("%s %+.3g (p %.3f)" % (cset, b1["coef"], b1["p_value"]))
                bng = cell(ires, ["spec", "horizon", "gamma_source"], [spec, h, src], "NG")
                cls = classify(b0["coef"], b0["p_value"], bng["coef"], bng["p_value"])
                classes.append(cls)
                parts.append("%s base %+.3g (p %.3f); %s; se NG/base %.2f -> %s" % (
                    src, b0["coef"], b0["p_value"], "; ".join(parts_src),
                    bng["se"] / b0["se"], cls))
            agree = classes[0] if classes[0] == classes[1] else "sources disagree"
            log.append("  %-16s h=%-2d [%s] %s" % (spec, h, "decision" if decision else "reported",
                                                   agree))
            for p in parts:
                log.append("      " + p)
    log.append(" daily A4 (diagnostic: controls contemporaneous with RV_t):")
    for reg in D_REGRESSORS:
        for form in ["log", "levels"]:
            decision = (reg == "MFIV_t" and form == "log")
            parts, classes = [], []
            for src in D_SRC:
                b0 = cell(dres, ["spec", "form", "gamma_source"], [reg, form, src], "base")
                bng = cell(dres, ["spec", "form", "gamma_source"], [reg, form, src], "NG")
                cls = classify(b0["coef"], b0["p_value"], bng["coef"], bng["p_value"])
                classes.append(cls)
                extra = "; ".join("%s %+.3g (p %.3f)" % (
                    cs, cell(dres, ["spec", "form", "gamma_source"], [reg, form, src], cs)["coef"],
                    cell(dres, ["spec", "form", "gamma_source"], [reg, form, src], cs)["p_value"])
                    for cs in ["N", "G", "NG"])
                parts.append("%s base %+.3g (p %.3f); %s; se NG/base %.2f -> %s" % (
                    src, b0["coef"], b0["p_value"], extra, bng["se"] / b0["se"], cls))
            agree = classes[0] if classes[0] == classes[1] else "sources disagree"
            log.append("  %-15s %-6s [%s] %s" % (reg, form, "diagnostic" if decision else "reported", agree))
            for p in parts:
                log.append("      " + p)

    out_dir = str(OUTPUT_DIR)
    ires.to_csv(os.path.join(out_dir, "16_hvol_breadth_controls_intraday.csv"), index=False)
    dres.to_csv(os.path.join(out_dir, "16_hvol_breadth_controls_daily.csv"), index=False)
    breadth.to_csv(os.path.join(out_dir, "16_hvol_breadth_daily_inputs.csv"), index=False)
    with open(os.path.join(out_dir, "16_hvol_breadth_controls_verdicts.txt"), "w") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    print("\nWritten: " + out_dir)


# =====================================================================
# PARTE 7: CORREZIONI PER TEST MULTIPLI E CONTEGGI DEL DISEGNO INTRADAY
# =====================================================================
# Ricalcola dagli output gia' prodotti, senza nuove stime, i numeri della tesi sul disegno
# intraday e li confronta con la tesi stampata (pagina stampata = pagina del PDF - 4).
#
# Esiti: OK / DIVERSO per i confronti, INFO per i numeri non stampati. Un DIVERSO non ferma la
# parte; la fermano solo problemi di integrita' dei dati (file o colonne mancanti, chiavi
# duplicate, famiglie di test incomplete, conteggi che non tornano fra gli output a monte).

PYFIXEST_RESULTS = ("15_intraday_regressions",
                    "15_pyfixest_estimation_2025_full_247_results.csv")
PYFIXEST_STATUS = ("15_intraday_regressions",
                   "15_pyfixest_estimation_2025_full_247_model_status.csv")
TERCILES = ("15_intraday_regressions", "15_tercile_incremental_r2_full_247.csv")
EXCLUSION = (OUTPUT_FOLDER, "16_cluster_exclusion_diagnostic.csv")

OUT_TESTS = "16_multiple_testing_2025.csv"
OUT_TESTS_SUMMARY = "16_multiple_testing_summary_2025.csv"
OUT_COUNTS = "16_counts_and_diagnostics_2025.csv"
OUT_CHECK = "16_thesis_check.csv"

PRIMARY_FE = "date_x_weektime"
LIGHT_FE = "date_x_time"

# Terzili fissati sulla griglia (Sezione 4.2).
TERCILES_DEF = {
    "opening": ("09:40:00", "11:30:00"),
    "middle": ("11:40:00", "13:40:00"),
    "closing": ("13:50:00", "15:50:00"),
}
TERCILE_LABEL_35 = {"opening": "early", "middle": "middle", "closing": "late"}

APRIL_7 = STRESS_DAY
STRESS_APRIL_FIRST, STRESS_APRIL_LAST = "2025-04-07", "2025-04-23"
LATE_FROM, LATE_TO = "13:50:00", "15:50:00"

# Tolleranza per "identici" (pyfixest ricalcola gli errori standard con rumore
# fino a circa 1e-6 relativo).
IDENTICAL_TOL = 1e-6

FOMC_SOURCE_NOTE = "date FOMC copiate da 14_Daily_MZ_robustness.py"


class IntegrityError(RuntimeError):
    pass


def read_output(pair, columns=None) -> pd.DataFrame:
    p = path(pair)
    if not p.exists():
        raise FileNotFoundError(f"File richiesto non trovato:\n{p}")
    df = pd.read_csv(p)
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise IntegrityError(f"{p.name}: colonne mancanti {missing}")
    return df


def integrity(cond: bool, message: str):
    if not cond:
        raise IntegrityError(message)


def holm_adjust(p: np.ndarray) -> np.ndarray:
    """p-value aggiustati di Holm (step-down), come statsmodels 'holm'."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty(m)
    out[order] = adj_sorted
    return out


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """p-value aggiustati di Benjamini-Hochberg (step-up), come 'fdr_bh'."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty(m)
    out[order] = adj_sorted
    return out


def fmt_int(x) -> str:
    return f"{int(x):,}"


def fmt_list_int(values) -> str:
    return " / ".join(fmt_int(v) for v in values)


def tercile_of(t):
    for name, (a, b) in TERCILES_DEF.items():
        if a <= t <= b:
            return name
    return None


def multiple_testing_and_counts():
    checks_rows = []
    counts_rows = []

    def check(pagina, voce, valore_tesi, valore_calcolato, precisione, ok, nota=""):
        esito = "INFO" if ok is None else ("OK" if ok else "DIVERSO")
        checks_rows.append({"pagina": pagina, "voce": voce, "valore_tesi": valore_tesi,
                            "valore_calcolato": valore_calcolato, "precisione": precisione,
                            "esito": esito, "nota": nota})

    def count(sezione, voce, valore, fonte):
        counts_rows.append({"sezione": sezione, "voce": voce, "valore": valore, "fonte": fonte})

    print("=" * 120)
    print("16 (parte 7)  CORREZIONI PER TEST MULTIPLI E CONTEGGI DEL DISEGNO INTRADAY")
    print("=" * 120)
    print(f"Cartella dati: {DATA_DIR}")

    # =====================================================
    # 1. HOLM E BENJAMINI-HOCHBERG (I3, MFIV_ann, 32 TEST PER MISURA)
    # =====================================================
    res34 = read_output(PYFIXEST_RESULTS, ["specification", "horizon", "form", "gamma_source",
                                           "fe_scheme", "term", "coef", "se", "p_value", "n"])
    ih3 = res34[res34["specification"] == "IH3"].copy()
    ih3["measure"] = ih3["term"].str.extract(r"^(H_OI|H_VOL)_")[0]
    ih3 = ih3[ih3["measure"].notna()].copy()

    families = []
    for measure in ["H_OI", "H_VOL"]:
        fam = ih3[ih3["measure"] == measure].copy()
        key = ["horizon", "form", "fe_scheme", "gamma_source"]
        integrity(len(fam) == 32, f"famiglia {measure}: {len(fam)} test invece di 32")
        integrity(not fam.duplicated(key).any(), f"famiglia {measure}: celle duplicate")
        integrity(set(fam["horizon"]) == set(HORIZONS)
                  and set(fam["form"]) == {"levels", "log"}
                  and set(fam["fe_scheme"]) == {PRIMARY_FE, LIGHT_FE}
                  and set(fam["gamma_source"]) == {"cboe", "indep"},
                  f"famiglia {measure}: la griglia 4x2x2x2 non e' completa")
        integrity(fam["p_value"].between(0, 1).all() and np.isfinite(fam["p_value"]).all(),
                  f"famiglia {measure}: p-value non validi")
        fam["p_holm"] = holm_adjust(fam["p_value"].to_numpy())
        fam["p_bh"] = bh_adjust(fam["p_value"].to_numpy())
        fam["reject_holm_5pct"] = (fam["p_holm"] <= ALPHA).astype(int)
        fam["reject_bh_5pct"] = (fam["p_bh"] <= ALPHA).astype(int)
        fam["rank_p"] = fam["p_value"].rank(method="first").astype(int)
        fam["fe_role"] = np.where(fam["fe_scheme"] == PRIMARY_FE, "primary", "lighter")
        families.append(fam)

    tests = pd.concat(families, ignore_index=True)

    # Controllo incrociato con statsmodels, se presente: deve dare gli stessi rifiuti.
    try:
        from statsmodels.stats.multitest import multipletests
        for measure, fam in tests.groupby("measure"):
            rh, ph, _, _ = multipletests(fam["p_value"], ALPHA, "holm")
            rb, pb, _, _ = multipletests(fam["p_value"], ALPHA, "fdr_bh")
            integrity(np.allclose(ph, fam["p_holm"], rtol=0, atol=1e-12)
                      and np.allclose(pb, fam["p_bh"], rtol=0, atol=1e-12)
                      and (rh.astype(int) == fam["reject_holm_5pct"]).all()
                      and (rb.astype(int) == fam["reject_bh_5pct"]).all(),
                      f"{measure}: Holm/BH diversi da statsmodels")
        sm_note = "Holm e BH identici a statsmodels.multipletests"
    except ImportError:
        sm_note = "statsmodels non disponibile: nessun controllo incrociato"
    print(f"\n[1] {sm_note}")

    tests = tests.sort_values(["measure", "p_value"]).reset_index(drop=True)
    tests_out = tests[["measure", "term", "horizon", "form", "fe_scheme", "fe_role",
                       "gamma_source", "coef", "se", "p_value", "rank_p", "p_holm",
                       "p_bh", "reject_holm_5pct", "reject_bh_5pct", "n"]]
    tests_out.to_csv(OUTPUT_DIR / OUT_TESTS, index=False)

    summary_rows = []
    for measure, fam in tests.groupby("measure"):
        prim = fam[fam["fe_role"] == "primary"]
        light = fam[fam["fe_role"] == "lighter"]
        best = prim.loc[prim["p_holm"].idxmin()]
        summary_rows.append({
            "measure": measure, "n_tests": len(fam),
            "holm_rejections": int(fam["reject_holm_5pct"].sum()),
            "holm_rejections_primary": int(prim["reject_holm_5pct"].sum()),
            "holm_rejections_lighter": int(light["reject_holm_5pct"].sum()),
            "bh_rejections": int(fam["reject_bh_5pct"].sum()),
            "bh_rejections_primary": int(prim["reject_bh_5pct"].sum()),
            "bh_rejections_lighter": int(light["reject_bh_5pct"].sum()),
            "min_p_holm_primary": float(best["p_holm"]),
            "min_p_holm_primary_cell": f"{best['form']} h={int(best['horizon'])} {best['gamma_source']}",
            "min_p_raw_primary": float(prim["p_value"].min()),
        })
    mt = pd.DataFrame(summary_rows)
    mt.to_csv(OUTPUT_DIR / OUT_TESTS_SUMMARY, index=False)
    print(mt.to_string(index=False))

    s_oi = mt.set_index("measure").loc["H_OI"]
    s_vol = mt.set_index("measure").loc["H_VOL"]

    check(71, "Test per misura di convessita' (I3, MFIV_ann)", "32", str(int(s_vol["n_tests"])),
          "intero", int(s_vol["n_tests"]) == 32 and int(s_oi["n_tests"]) == 32,
          "4 orizzonti x 2 forme x 2 schemi FE x 2 fonti; stesso numero per H_OI")
    check(71, "H_OI: celle che sopravvivono a Holm / BH", "0 / 0 (\"No H_OI result survives\")",
          f"{int(s_oi['holm_rejections'])} / {int(s_oi['bh_rejections'])}", "intero",
          int(s_oi["holm_rejections"]) == 0 and int(s_oi["bh_rejections"]) == 0)
    check(71, "H_VOL: rifiuti di Holm, tutti con effetti fissi leggeri",
          "8, tutti leggeri",
          f"{int(s_vol['holm_rejections'])} (primario {int(s_vol['holm_rejections_primary'])}, "
          f"leggero {int(s_vol['holm_rejections_lighter'])})", "intero",
          int(s_vol["holm_rejections"]) == 8 and int(s_vol["holm_rejections_primary"]) == 0)
    check(71, "H_VOL: celle che sopravvivono a BH con effetti primari", "4",
          str(int(s_vol["bh_rejections_primary"])), "intero",
          int(s_vol["bh_rejections_primary"]) == 4)
    check(73, "Tab. D.1: nessuna misura sopravvive a Holm (schema primario); 8 rifiuti H_VOL "
          "leggeri; BH 4 celle primarie", "0 / 8 / 4",
          f"{int(s_oi['holm_rejections_primary']) + int(s_vol['holm_rejections_primary'])} / "
          f"{int(s_vol['holm_rejections_lighter'])} / {int(s_vol['bh_rejections_primary'])}",
          "intero",
          int(s_oi["holm_rejections_primary"]) + int(s_vol["holm_rejections_primary"]) == 0
          and int(s_vol["holm_rejections_lighter"]) == 8
          and int(s_vol["bh_rejections_primary"]) == 4)
    for pg, voce in [(39, "Tab. 5.2, H_VOL log: \"none survives Holm\""),
                     (46, "Sez. 5.4: nessuna cella intraday dello schema primario sopravvive a Holm"),
                     (53, "Sez. 5.6: nessuna cella dello schema primario sopravvive a Holm")]:
        check(pg, voce, "0", str(int(s_vol["holm_rejections_primary"])), "intero",
              int(s_vol["holm_rejections_primary"]) == 0,
              f"p di Holm minimo nello schema primario {s_vol['min_p_holm_primary']:.3f} "
              f"({s_vol['min_p_holm_primary_cell']})")
    check(48, "Sez. 5.4: nessuna cella primaria sopravvive a Holm; BH \"retains some\"",
          "Holm 0; BH > 0",
          f"Holm {int(s_vol['holm_rejections_primary'])}; BH {int(s_vol['bh_rejections_primary'])}",
          "qualitativo",
          int(s_vol["holm_rejections_primary"]) == 0 and int(s_vol["bh_rejections_primary"]) > 0)
    check("-", "H_VOL: rifiuti BH totali (primario + leggero)", "non stampato",
          f"{int(s_vol['bh_rejections'])} ({int(s_vol['bh_rejections_primary'])} primari)",
          "intero", None)
    check("-", "H_VOL: p di Holm minimo nello schema primario", "non stampato",
          f"{s_vol['min_p_holm_primary']:.4f} ({s_vol['min_p_holm_primary_cell']}; "
          f"p grezzo {s_vol['min_p_raw_primary']:.4f})", "4 decimali", None)

    # H_OI in log con effetti leggeri: significativo a 30 e 60 minuti su entrambe le fonti,
    # ma nessuna di queste celle sopravvive a Holm o BH.
    hoi_light_log = tests[(tests["measure"] == "H_OI") & (tests["fe_role"] == "lighter")
                          & (tests["form"] == "log") & (tests["horizon"].isin([30, 60]))]
    integrity(len(hoi_light_log) == 4, "H_OI log leggero 30/60: celle mancanti")
    ok_nom = bool((hoi_light_log["p_value"] < ALPHA).all())
    ok_none = bool((hoi_light_log["reject_holm_5pct"] == 0).all()
                   and (hoi_light_log["reject_bh_5pct"] == 0).all())
    check(49, "H_OI log, effetti leggeri: significativo a 30 e 60 min su entrambe le fonti; "
          "nessuna cella sopravvive a Holm o BH", "si'; nessuna",
          "p = " + ", ".join(f"{r.gamma_source} h={int(r.horizon)} {r.p_value:.4f}"
                             for r in hoi_light_log.sort_values(["horizon", "gamma_source"]).itertuples())
          + f"; Holm/BH {int(hoi_light_log['reject_holm_5pct'].sum())}/"
            f"{int(hoi_light_log['reject_bh_5pct'].sum())}",
          "5% nominale", ok_nom and ok_none,
          "indep a 30 minuti ha p = 0.0498, appena sotto il 5%")

    # =====================================================
    # 2-4. CONTEGGI DEL PANNELLO INTRADAY
    # =====================================================
    panel = read_output(PANEL, ["quote_date", "grid_time", "date_fe", "week_x_time_fe", "iso_week_key",
                                "mfiv_flag", "TrailingRV_n_returns_used", "usable_for_flow_specs",
                                "sigma_atm", "MFIV", "MFIV_ann", "tau"]
                        + [f"h{h}_valid" for h in HORIZONS] + [f"FwdRV_{h}" for h in HORIZONS])
    panel["grid_time"] = panel["grid_time"].map(normalize_time_string)
    panel["quote_date"] = panel["quote_date"].astype(str)
    integrity(not panel.duplicated(["quote_date", "grid_time"]).any(),
              "pannello 13: righe duplicate per (quote_date, grid_time)")

    n_rows = len(panel)
    n_days = panel["quote_date"].nunique()
    grid = sorted(panel["grid_time"].unique())
    n_grid = len(grid)
    count("pannello", "righe", n_rows, PANEL[1])
    count("pannello", "giornate", n_days, PANEL[1])
    count("pannello", "punti di griglia", n_grid, PANEL[1])
    count("pannello", "primo / ultimo punto", f"{grid[0]} / {grid[-1]}", PANEL[1])
    check(32, "Giornate x punti di griglia = righe",
          "247 x 38 = 9,386 (09:40-15:50)",
          f"{n_days} x {n_grid} = {fmt_int(n_rows)} ({grid[0][:5]}-{grid[-1][:5]})", "intero",
          n_days == 247 and n_grid == 38 and n_rows == 9386
          and grid[0] == "09:40:00" and grid[-1] == "15:50:00")
    check(36, "Cluster per giorno", "247", str(n_days), "intero", n_days == 247)

    # Osservazioni per orizzonte, senza e con le misure di convessita'.
    n_h = [int((panel[f"h{h}_valid"] == 1).sum()) for h in HORIZONS]
    n_h_flow = [int(((panel[f"h{h}_valid"] == 1) & (panel["usable_for_flow_specs"] == 1)).sum())
                for h in HORIZONS]
    for h, a, b in zip(HORIZONS, n_h, n_h_flow):
        count("osservazioni", f"h={h} (I1, I2)", a, PANEL[1])
        count("osservazioni", f"h={h} (I3, con convessita')", b, PANEL[1])
        integrity(int(panel.loc[panel[f"h{h}_valid"] == 1, f"FwdRV_{h}"].notna().sum()) == a,
                  f"h={h}: FwdRV mancante in righe valide")

    status = read_output(PYFIXEST_STATUS, ["specification", "horizon", "form", "fe_scheme",
                                           "n_final_estimation_rows", "n_fe_absorbed", "n_days"])
    st = status[(status["fe_scheme"] == PRIMARY_FE) & (status["form"] == "levels")]
    for h, a, b in zip(HORIZONS, n_h, n_h_flow):
        s2 = st[(st["specification"] == "IH2") & (st["horizon"] == h)]
        s3 = st[(st["specification"] == "IH3") & (st["horizon"] == h)]
        integrity(len(s2) == 1 and int(s2["n_final_estimation_rows"].iloc[0]) == a,
                  f"h={h}: righe del pannello diverse da 34 (IH2)")
        integrity(len(s3) >= 1 and (s3["n_final_estimation_rows"] == b).all(),
                  f"h={h}: righe con convessita' diverse da 34 (IH3)")
    check(32, "Osservazioni per h = 5, 15, 30, 60", "9,386 / 9,139 / 8,892 / 8,151",
          fmt_list_int(n_h), "intero", n_h == [9386, 9139, 8892, 8151],
          "uguali a 34 model_status (IH2, livelli)")
    check(32, "Quattro in meno con le misure di convessita' (n di I3 in Tab. D.5, p. 77)",
          "9,382 / 9,135 / 8,888 / 8,147", fmt_list_int(n_h_flow), "intero",
          n_h_flow == [9382, 9135, 8888, 8147]
          and all(a - b == 4 for a, b in zip(n_h, n_h_flow)))

    # Effetti fissi e celle settimana x ora.
    n_date_fe = panel["date_fe"].nunique()
    n_wxt = panel["week_x_time_fe"].nunique()
    n_weeks = panel["iso_week_key"].nunique()
    cells = panel.groupby("week_x_time_fe").size()
    dist = cells.value_counts().sort_index()
    mean_cell = float(cells.mean())
    fe34 = st[(st["specification"] == "IH2") & (st["horizon"] == 5)]["n_fe_absorbed"].iloc[0]
    integrity(int(fe34) == n_date_fe + n_wxt, "livelli di effetti fissi diversi da 34 (h=5)")
    count("effetti fissi", "date", n_date_fe, PANEL[1])
    count("effetti fissi", "celle settimana x ora", n_wxt, PANEL[1])
    count("effetti fissi", "settimane ISO", n_weeks, PANEL[1])
    count("effetti fissi", "livelli assorbiti a h=5 (34)", int(fe34), PYFIXEST_STATUS[1])
    count("effetti fissi", "osservazioni medie per cella", round(mean_cell, 4), PANEL[1])
    for size, k in dist.items():
        count("effetti fissi", f"celle con {size} osservazioni", int(k), PANEL[1])
    check(32, "Livelli di effetti fissi = date + celle settimana x ora", "2,261 = 247 + 2,014",
          f"{fmt_int(n_date_fe + n_wxt)} = {n_date_fe} + {fmt_int(n_wxt)}", "intero",
          n_date_fe == 247 and n_wxt == 2014 and int(fe34) == 2261,
          f"{n_weeks} settimane ISO x {n_grid} punti; 34 n_fe_absorbed = {int(fe34)}")
    check(32, "Osservazioni per cella settimana x ora (\"about five ... one per weekday\")",
          "circa 5, al piu' una per giorno feriale", f"media {mean_cell:.2f}; massimo {int(cells.max())}",
          "qualitativo", 4.5 <= mean_cell <= 5.0 and int(cells.max()) <= 5,
          "distribuzione: " + ", ".join(f"{int(k):,} celle con {s}" for s, k in dist.items()))

    # TrailingRV: nove rendimenti alle 09:40.
    tr_first = panel.loc[panel["grid_time"] == grid[0], "TrailingRV_n_returns_used"].unique()
    count("TrailingRV", "rendimenti usati alle 09:40", ";".join(str(int(x)) for x in tr_first), PANEL[1])
    check(32, "Primi nove minuti solo in TrailingRV (rendimenti a 1 minuto alle 09:40)", "9",
          ";".join(str(int(x)) for x in tr_first), "intero",
          len(tr_first) == 1 and int(tr_first[0]) == 9)

    # Terzili e ultimo punto a 60 minuti.
    panel["tercile"] = panel["grid_time"].map(tercile_of)
    integrity(panel["tercile"].notna().all(), "punti di griglia fuori dai terzili")
    pts = {k: sorted(panel.loc[panel["tercile"] == k, "grid_time"].unique()) for k in TERCILES_DEF}
    check(33, "Terzili: punti e orari",
          "12 (09:40-11:30); 13 (11:40-13:40); 13 (13:50-15:50)",
          "; ".join(f"{len(v)} ({v[0][:5]}-{v[-1][:5]})" for v in pts.values()), "intero",
          [len(v) for v in pts.values()] == [12, 13, 13])
    last60 = panel.loc[panel["h60_valid"] == 1, "grid_time"].max()
    check(33, "Ultimo punto con finestra di 60 minuti", "15:00", last60[:5], "esatto",
          last60 == "15:00:00")

    terc35 = read_output(TERCILES, ["horizon", "tercile", "n_rows", "n_grid_times"])
    terc_rows = {}
    for h in HORIZONS:
        v = panel[panel[f"h{h}_valid"] == 1]
        for k in TERCILES_DEF:
            n = int((v["tercile"] == k).sum())
            g = int(v.loc[v["tercile"] == k, "grid_time"].nunique())
            terc_rows[(h, k)] = (n, g)
            r35 = terc35[(terc35["horizon"] == h) & (terc35["tercile"] == TERCILE_LABEL_35[k])]
            integrity(len(r35) == 1 and int(r35["n_rows"].iloc[0]) == n
                      and int(r35["n_grid_times"].iloc[0]) == g,
                      f"terzile {k} h={h}: righe diverse da 35")
            count("terzili", f"h={h} {k}: righe / punti", f"{n} / {g}", PANEL[1])
    op = [terc_rows[(h, "opening")][0] for h in HORIZONS]
    mi = [terc_rows[(h, "middle")][0] for h in HORIZONS]
    cl = [terc_rows[(h, "closing")][0] for h in HORIZONS]
    check(46, "Tab. 5.5, righe di apertura e meta' giornata a ogni orizzonte", "2,964 e 3,211",
          f"{fmt_list_int(op)} e {fmt_list_int(mi)}", "intero",
          set(op) == {2964} and set(mi) == {3211}, "uguali a 35 n_rows")
    check(46, "Tab. 5.5, righe di chiusura per h = 5, 15, 30, 60", "3,211 / 2,964 / 2,717 / 1,976",
          fmt_list_int(cl), "intero", cl == [3211, 2964, 2717, 1976], "uguali a 35 n_rows")
    check(46, "Tab. 5.5, punti di griglia di chiusura a h = 60", "8",
          str(terc_rows[(60, "closing")][1]), "intero", terc_rows[(60, "closing")][1] == 8)

    # Righe che non passano la regola di validita' della strip.
    bad = panel["mfiv_flag"] != "ok"
    fail = []
    fail_outside = []
    for h in HORIZONS:
        v = panel[f"h{h}_valid"] == 1
        fail.append(int((v & bad & (panel["tercile"] == "closing")).sum()))
        fail_outside.append(int((v & bad & (panel["tercile"] != "closing")).sum()))
        count("validita' della strip", f"h={h}: righe di chiusura non valide", fail[-1], PANEL[1])
        count("validita' della strip", f"h={h}: righe non valide fuori dalla chiusura",
              fail_outside[-1], PANEL[1])
    n_bad = int(bad.sum())
    count("validita' della strip", "righe non valide nel pannello",
          n_bad, PANEL[1] + " (" + ", ".join(f"{k}={v}" for k, v in
                                              panel["mfiv_flag"].value_counts().items()) + ")")
    check(46, "Righe di chiusura che non passano la regola di validita', h = 5, 15, 30, 60",
          "262 / 125 / 49 / 0", fmt_list_int(fail), "intero", fail == [262, 125, 49, 0],
          f"fuori dalla chiusura: {fmt_list_int(fail_outside)}")
    check(33, "Il pannello tiene le righe sotto la regola di validita'", "tenute",
          f"{n_bad} righe con mfiv_flag != ok fra le {fmt_int(n_rows)}, tutte nel terzile di "
          f"chiusura; il campione di I2 a h=5 ha {fmt_int(n_h[0])} righe", "qualitativo",
          n_bad > 0 and sum(fail_outside) == 0 and n_h[0] == n_rows)

    # =====================================================
    # 5. LOG: MFIV TOTALE E ANNUALIZZATA COINCIDONO
    # =====================================================
    reg = read_output(REGISTERED, ["regressor", "vcov", "specification", "gamma_source", "horizon",
                                   "form", "term", "coef", "se", "r2_within"])
    lg = reg[(reg["vcov"] == "giorno") & (reg["form"] == "log")].copy()
    lg["term_std"] = lg["term"].replace({"log_MFIV": "log_MFIV_ann"})
    keys = ["specification", "gamma_source", "horizon", "term_std"]
    tot = lg[lg["regressor"] == "MFIV"].set_index(keys)
    ann = lg[lg["regressor"] == "MFIV_ann"].set_index(keys)
    integrity(len(tot) > 0 and len(tot) == len(ann) and set(tot.index) == set(ann.index)
              and not tot.index.duplicated().any() and not ann.index.duplicated().any(),
              "37: celle log non appaiate fra MFIV e MFIV_ann")
    ann = ann.loc[tot.index]
    rel_coef = float(((tot["coef"] - ann["coef"]).abs() / ann["coef"].abs()).max())
    rel_se = float(((tot["se"] - ann["se"]).abs() / ann["se"]).max())
    abs_r2 = float((tot["r2_within"] - ann["r2_within"]).abs().max())
    tau_var = int(panel.groupby("grid_time")["tau"].nunique().max())
    count("log", "celle confrontate", len(tot), REGISTERED[1])
    count("log", "max diff. relativa coef", rel_coef, REGISTERED[1])
    count("log", "max diff. relativa se", rel_se, REGISTERED[1])
    count("log", "max diff. assoluta R2 within", abs_r2, REGISTERED[1])
    count("log", "valori distinti di tau per punto di griglia (massimo)", tau_var, PANEL[1])
    for pg, voce in [(33, "In log MFIV totale e annualizzata coincidono"),
                     (79, "Tab. D.7: identica sotto MFIV totale e annualizzata")]:
        check(pg, voce, "identiche",
              f"{len(tot)} celle; max diff. rel. coef {rel_coef:.1e}, se {rel_se:.1e}; R2 {abs_r2:.1e}",
              f"tolleranza {IDENTICAL_TOL:g}",
              rel_coef < IDENTICAL_TOL and rel_se < IDENTICAL_TOL and abs_r2 < IDENTICAL_TOL,
              "tau(t) costante a ogni punto di griglia (nessuna chiusura anticipata fra le 247)"
              if tau_var == 1 else "tau(t) NON costante per punto di griglia")

    # =====================================================
    # 6. CONCENTRAZIONE PER GIORNATA (parte 1)
    # =====================================================
    conc_cols = ["regressor", "specification", "gamma_source", "horizon", "term", "var_day_n_eff",
                 "var_day_top_label", "top_day_for_coef", "coef", "coef_without_top_day",
                 "p_day", "p_day_without_top_day"]
    cl_ = read_output(CONCENTRATION["levels"], conc_cols)
    cg_ = read_output(CONCENTRATION["log"], conc_cols)
    integrity(len(cl_) == 64 and len(cg_) == 64, "40: attese 64 celle per forma")
    for df in (cl_, cg_):
        df["var_day_top_label"] = df["var_day_top_label"].astype(str)
        df["top_day_for_coef"] = df["top_day_for_coef"].astype(str)

    iv_lev = cl_[cl_["term"].isin(["MFIV", "MFIV_ann"])]
    integrity(len(iv_lev) == 32, "40 livelli: attese 32 celle di varianza implicita")
    i2 = iv_lev[(iv_lev["regressor"] == "MFIV_ann") & (iv_lev["specification"] == "IH2")].sort_values("horizon")
    integrity(list(i2["horizon"]) == HORIZONS, "40 livelli: I2 MFIV_ann incompleto")
    r_i2 = f"{i2['var_day_n_eff'].min():.1f}-{i2['var_day_n_eff'].max():.1f}"
    r_all = f"{iv_lev['var_day_n_eff'].min():.1f}-{iv_lev['var_day_n_eff'].max():.1f}"
    check(44, "Giorni effettivi, I2 con MFIV_ann, livelli", "2.1-3.0", r_i2, "1 decimale",
          r_i2 == "2.1-3.0", "valori " + ", ".join(f"{x:.2f}" for x in i2["var_day_n_eff"]))
    check(51, "Giorni effettivi, coefficienti di varianza implicita in livelli (MFIV totale e annualizzata)",
          "1.3-3.1", r_all, "1 decimale", r_all == "1.3-3.1",
          f"{len(iv_lev)} celle: I1-I3, due regressori, due fonti")

    n_top_coef = int((iv_lev["top_day_for_coef"] == APRIL_7).sum())
    n_top_var = int((iv_lev["var_day_top_label"] == APRIL_7).sum())
    check(44, "Le stime in livelli dipendono soprattutto dal 7 aprile 2025 (I2, MFIV_ann)",
          "\"particularly 7 April 2025\"",
          "giornata piu' influente sul coefficiente: " + ", ".join(
              f"h={int(r.horizon)} {r.top_day_for_coef}" for r in i2.itertuples()),
          "qualitativo", bool((i2["top_day_for_coef"] == APRIL_7).all()))
    check(51, "Il 7 aprile e' di solito la giornata piu' influente (celle di varianza implicita in livelli)",
          "\"usually the most influential\"",
          f"giornata piu' influente sul coefficiente in {n_top_coef} di {len(iv_lev)} celle; "
          f"giornata con la quota di varianza maggiore in {n_top_var} di {len(iv_lev)}",
          "qualitativo", n_top_coef > len(iv_lev) / 2 and n_top_var > len(iv_lev) / 2)

    p_wo = list(i2["p_day_without_top_day"])
    check(44, "Senza la giornata piu' pesante (7 aprile) il risultato a 5 minuti resta, oltre si indebolisce",
          "5 min resta; oltre no",
          "p senza il 7 aprile: " + " / ".join(f"{p:.3f}" for p in p_wo)
          + " (con: " + " / ".join(f"{p:.3f}" for p in i2["p_day"]) + ")", "5% nominale",
          p_wo[0] < ALPHA and all(p >= ALPHA for p in p_wo[1:])
          and bool((i2["top_day_for_coef"] == APRIL_7).all()))
    check(53, "In livelli, oltre i cinque minuti il risultato poggia soprattutto sul 7 aprile",
          "si'", "p senza il 7 aprile a 15/30/60: " + " / ".join(f"{p:.3f}" for p in p_wo[1:]),
          "5% nominale", all(p >= ALPHA for p in p_wo[1:]))

    iv_log = cg_[(cg_["regressor"] == "MFIV_ann") & (cg_["term"] == "log_MFIV_ann")]
    i2_log = iv_log[iv_log["specification"] == "IH2"].sort_values("horizon")
    integrity(len(i2_log) == 4, "40 log: I2 incompleto")
    rel_change = float(((i2_log["coef_without_top_day"] - i2_log["coef"]).abs() / i2_log["coef"]).max())
    check(44, "Elasticita' di I2 in log: cambia poco togliendo la giornata piu' influente",
          "cambia poco", f"max variazione relativa {100 * rel_change:.1f}% ("
          + ", ".join(f"{a:.2f}->{b:.2f}" for a, b in zip(i2_log["coef"], i2_log["coef_without_top_day"])) + ")",
          "qualitativo (< 10%)", rel_change < 0.10)

    log_cells = cg_[cg_["regressor"] == "MFIV_ann"]
    integrity(len(log_cells) == 32, "40 log: attese 32 celle con MFIV_ann")
    log_iv = log_cells[log_cells["term"] == "log_MFIV_ann"]
    check(51, "In log l'influenza si distribuisce su piu' giornate",
          "piu' giornate", f"celle di varianza implicita: {log_iv['var_day_n_eff'].min():.1f}-"
          f"{log_iv['var_day_n_eff'].max():.1f} giorni effettivi (livelli {r_all})", "qualitativo",
          log_iv["var_day_n_eff"].min() > iv_lev["var_day_n_eff"].max())
    n_fomc = int(log_cells["var_day_top_label"].isin(FOMC_DATES).sum())
    top_mode = log_cells["var_day_top_label"].value_counts()
    check(51, "In log le giornate FOMC diventano influenti",
          "\"FOMC decisions become influential\"",
          f"giornata con la quota di varianza maggiore e' FOMC in {n_fomc} di {len(log_cells)} celle "
          f"(la piu' frequente: {top_mode.index[0]}, {int(top_mode.iloc[0])} celle)", "qualitativo",
          n_fomc > len(log_cells) / 2 and top_mode.index[0] in FOMC_DATES,
          FOMC_SOURCE_NOTE)
    check(52, "In log le giornate FOMC sono le piu' pesanti", "si'",
          f"{n_fomc} di {len(log_cells)} celle", "qualitativo", n_fomc > len(log_cells) / 2)
    conv_log = log_cells[log_cells["term"].str.startswith("H_")]
    n_lt10 = int((conv_log["var_day_n_eff"] < 10).sum())
    check(51, "Alcune celle di convessita' in log hanno meno di dieci giorni effettivi", "alcune",
          f"{n_lt10} di {len(conv_log)} (" + ", ".join(
              f"{r.term} h={int(r.horizon)} {r.var_day_n_eff:.2f}"
              for r in conv_log[conv_log["var_day_n_eff"] < 10].itertuples()) + ")",
          "qualitativo", 0 < n_lt10 < len(conv_log))
    hv_tot_lev = cl_[(cl_["regressor"] == "MFIV") & (cl_["term"].str.startswith("H_VOL"))]
    hv_log = log_cells[log_cells["term"].str.startswith("H_VOL")].sort_values(["horizon", "gamma_source"])
    integrity(len(hv_tot_lev) == 8 and len(hv_log) == 8, "40: celle H_VOL incomplete")
    surv_log = sorted(set(int(r.horizon) for r in hv_log.itertuples() if r.p_day_without_top_day < ALPHA))
    check(51, "H_VOL in livelli con MFIV totale sopravvive alla rimozione della giornata piu' pesante",
          "sopravvive", f"p senza la giornata: massimo {hv_tot_lev['p_day_without_top_day'].max():.4f} "
          f"sulle 8 celle", "5% nominale", bool((hv_tot_lev["p_day_without_top_day"] < ALPHA).all()))
    check(51, "... la controparte in log solo ad alcuni orizzonti", "alcuni orizzonti",
          f"sopravvive a h = {surv_log} (p: " + ", ".join(
              f"{r.gamma_source} h={int(r.horizon)} {r.p_day_without_top_day:.3f}"
              for r in hv_log.itertuples()) + ")", "5% nominale",
          0 < len(surv_log) < len(HORIZONS))
    five_iv = iv_lev[iv_lev["horizon"] == 5]
    check(52, "Due risultati in livelli sopravvivono alla rimozione della giornata piu' pesante: "
          "varianza implicita a 5 minuti e H_VOL con MFIV totale", "entrambi",
          f"varianza implicita a 5 min: p senza la giornata <= {five_iv['p_day_without_top_day'].max():.1e} "
          f"({len(five_iv)} celle); H_VOL totale: <= {hv_tot_lev['p_day_without_top_day'].max():.4f}",
          "5% nominale", bool((five_iv["p_day_without_top_day"] < ALPHA).all()
                              and (hv_tot_lev["p_day_without_top_day"] < ALPHA).all()))
    count("concentrazione", "giorni effettivi I2 MFIV_ann livelli",
          ", ".join(f"{x:.4f}" for x in i2["var_day_n_eff"]), CONCENTRATION["levels"][1])
    count("concentrazione", "giorni effettivi, celle IV in livelli (min-max)",
          f"{iv_lev['var_day_n_eff'].min():.4f}-{iv_lev['var_day_n_eff'].max():.4f}", CONCENTRATION["levels"][1])
    count("concentrazione", "7 aprile giornata piu' influente sul coefficiente (celle IV livelli)",
          f"{n_top_coef}/{len(iv_lev)}", CONCENTRATION["levels"][1])
    count("concentrazione", "7 aprile giornata con quota di varianza maggiore (celle IV livelli)",
          f"{n_top_var}/{len(iv_lev)}", CONCENTRATION["levels"][1])
    count("concentrazione", "I2 MFIV_ann livelli: p senza il 7 aprile",
          ", ".join(f"{p:.6g}" for p in p_wo), CONCENTRATION["levels"][1])
    count("concentrazione", "log: giornata di varianza maggiore FOMC", f"{n_fomc}/{len(log_cells)}",
          CONCENTRATION["log"][1])

    # =====================================================
    # 7. PROFILO DI FINE GIORNATA E VOLATILITA' DI APRILE
    # =====================================================
    med = panel.groupby("grid_time")[["sigma_atm", "MFIV", "MFIV_ann"]].median()
    a, b = med.loc[LATE_FROM], med.loc[LATE_TO]
    for col in ["sigma_atm", "MFIV", "MFIV_ann"]:
        count("profilo di fine giornata", f"mediana {col} {LATE_FROM[:5]} -> {LATE_TO[:5]}",
              f"{a[col]:.6g} -> {b[col]:.6g}", PANEL[1])
    check(45, "Fine giornata: sigma_ATM sale mentre l'orizzonte si accorcia; la MFIV totale scende, "
          "la MFIV_ann sale", "qualitativo (nessun numero stampato)",
          f"mediane {LATE_FROM[:5]}->{LATE_TO[:5]}: sigma_ATM {a['sigma_atm']:.3f}->{b['sigma_atm']:.3f}; "
          f"MFIV {a['MFIV']:.2g}->{b['MFIV']:.2g}; MFIV_ann {a['MFIV_ann']:.3f}->{b['MFIV_ann']:.3f}",
          "qualitativo",
          b["sigma_atm"] > a["sigma_atm"] and b["MFIV"] < a["MFIV"] and b["MFIV_ann"] > a["MFIV_ann"])

    daily = read_output(DAILY_PANEL, ["quote_date", "MFIV_ann_t", "MFIV_ann_0931"])
    daily["quote_date"] = daily["quote_date"].astype(str)
    integrity(not daily["quote_date"].duplicated().any(), "pannello 10: date duplicate")
    apr = daily["quote_date"].str.startswith(f"{YEAR}-04")
    ratios = {}
    for col in ["MFIV_ann_t", "MFIV_ann_0931"]:
        vol = np.sqrt(daily[col])
        ratios[(col, "media")] = float(vol[apr].mean() / vol[~apr].mean())
        ratios[(col, "mediana")] = float(vol[apr].median() / vol[~apr].median())
        count("aprile", f"volatilita' implicita annualizzata ({col}): aprile / resto, media",
              round(ratios[(col, "media")], 4), DAILY_PANEL[1])
        count("aprile", f"volatilita' implicita annualizzata ({col}): aprile / resto, mediana",
              round(ratios[(col, "mediana")], 4), DAILY_PANEL[1])
    check(36, "Volatilita' implicita di aprile piu' che doppia rispetto al resto dell'anno",
          "> 2", "sqrt(MFIV_ann), media della seduta: " + f"{ratios[('MFIV_ann_t', 'media')]:.2f} "
          f"(apertura {ratios[('MFIV_ann_0931', 'media')]:.2f}; mediane "
          f"{ratios[('MFIV_ann_t', 'mediana')]:.2f} e {ratios[('MFIV_ann_0931', 'mediana')]:.2f})",
          "2 decimali", min(ratios.values()) > 2,
          f"{int(apr.sum())} sedute di aprile; la Sez. 2.2 puo' usare un'altra misura (IV ATM)")

    # =====================================================
    # 8. ESCLUSIONE DELLE SESSIONI DI STRESS (parte 2)
    # =====================================================
    exc = read_output(EXCLUSION, ["specification", "horizon", "form", "gamma_source", "variant", "term",
                                  "n_days_excluded", "coef_full", "p_full", "coef_excl", "t_excl", "p_excl"])
    apr_sessions = sorted(d for d in panel["quote_date"].unique()
                          if STRESS_APRIL_FIRST <= d <= STRESS_APRIL_LAST)
    nd = exc.groupby("variant")["n_days_excluded"].unique()
    integrity(all(len(v) == 1 for v in nd), "41: n_days_excluded non unico per variante")
    nd = {k: int(v[0]) for k, v in nd.items()}
    check(36, "Gruppo di stress: 14 sessioni = 12 dal 7 al 23 aprile + 21 marzo e 20 giugno",
          "14 = 12 + 2", f"{nd.get('insieme_completo')} = {nd.get('solo_aprile')} + "
          f"{nd.get('solo_scadenze_trimestrali')}; sessioni 7-23 aprile nel pannello: {len(apr_sessions)}",
          "intero", nd.get("insieme_completo") == 14 and nd.get("solo_aprile") == 12
          and nd.get("solo_scadenze_trimestrali") == 2 and len(apr_sessions) == 12)
    for variant, label in [("insieme_completo", "14 sessioni"), ("solo_aprile", "12 sessioni di aprile")]:
        x = exc[(exc["specification"] == "IH2") & (exc["form"] == "levels")
                & (exc["term"] == "MFIV_ann") & (exc["variant"] == variant)].sort_values("horizon")
        integrity(list(x["horizon"]) == HORIZONS, f"41: I2 MFIV_ann livelli incompleto ({variant})")
        for r in x.itertuples():
            count("esclusione stress", f"I2 MFIV_ann livelli, senza {label}, h={int(r.horizon)}: t / p",
                  f"{r.t_excl:.4f} / {r.p_excl:.3g}", EXCLUSION[1])
        check(50, f"Senza le {label}: t di MFIV_ann in I2, livelli, h = 5, 15, 30, 60",
              "non stampato", " / ".join(f"{t:.1f}" for t in x["t_excl"])
              + f" (p massimo {x['p_excl'].max():.1e}; sul campione intero p = "
              + " / ".join(f"{p:.3f}" for p in x["p_full"]) + ")", "1 decimale", None,
              "la tesi dice solo che le stime in livelli restano sensibili all'episodio di aprile; "
              "togliere l'intero gruppo rende il coefficiente significativo a ogni orizzonte")

    # =====================================================
    # OUTPUT E VERDETTO
    # =====================================================
    pd.DataFrame(counts_rows).to_csv(OUTPUT_DIR / OUT_COUNTS, index=False)
    chk = pd.DataFrame(checks_rows, columns=["pagina", "voce", "valore_tesi", "valore_calcolato",
                                             "precisione", "esito", "nota"])
    chk.to_csv(OUTPUT_DIR / OUT_CHECK, index=False)

    print()
    print("=" * 120)
    print("CONFRONTO CON LA TESI (pagine stampate)")
    print("=" * 120)
    for r in chk.itertuples():
        print(f"[{r.esito:7s}] p. {r.pagina}: {r.voce}")
        print(f"           tesi: {r.valore_tesi} | calcolato: {r.valore_calcolato}"
              + (f" | {r.nota}" if r.nota else ""))

    n_ok = int((chk["esito"] == "OK").sum())
    n_diff = int((chk["esito"] == "DIVERSO").sum())
    n_info = int((chk["esito"] == "INFO").sum())
    print()
    print(f"Riepilogo: {n_ok} OK, {n_diff} DIVERSO, {n_info} INFO")
    print("Output: " + ", ".join([OUT_TESTS, OUT_TESTS_SUMMARY, OUT_COUNTS, OUT_CHECK]))
    print(f"Cartella: {OUTPUT_DIR}")
    if n_diff == 0:
        print("VERDETTO: tutti i numeri della tesi controllati sono riprodotti.")
    else:
        print("VERDETTO: alcune voci sono DIVERSE dalla tesi:")
        for r in chk[chk["esito"] == "DIVERSO"].itertuples():
            print(f"  p. {r.pagina}: {r.voce} | tesi {r.valore_tesi} | calcolato {r.valore_calcolato}")
    print("=" * 120)


def main():
    run_variants(day_concentration, CONCENTRATION_FORMS)   # parte 1
    cluster_exclusion()                                    # parte 2
    wild_bootstrap_week()                                  # parte 3
    gamma_vif()                                            # parte 4
    gamma_effect_size_ci()                                 # parte 5
    hvol_breadth_controls()                                # parte 6
    multiple_testing_and_counts()                          # parte 7


if __name__ == "__main__":
    main()
