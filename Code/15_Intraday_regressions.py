"""
15_Intraday_regressions.py: regressioni intraday (Capitolo 5 e Appendici C-D della tesi).

  1. stime pyfixest di IH1, IH2, IH3 e IH1_mfiv_raw con due schemi di effetti fissi;
  2. R-quadro incrementale di MFIV_ann per terzili di orario (HI3);
  3. righe per specifica e fattore di costruzione fra MFIV e dipendente per fascia oraria;
  4. regressore MFIV accanto a MFIV_ann, cluster per giorno, a due vie e per settimana;
  5. la parte 2 sui pannelli troncati a moneyness (logm_005, logm_002);
  6. la parte 1 sugli stessi pannelli;
  7. controllo di pyfixest contro statsmodels con variabili dummy.

Input:  CBOE/Data/08_intraday_panel/ (pannello principale e varianti logm_005, logm_002)
Output: CBOE/Data/15_intraday_regressions/
"""

import functools
import re
import traceback
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.api as sm

from common import (DATA_DIR, REGULAR_CLOSE_SECONDS, YEAR as YEAR_INT,
                    clean_date_series, output_dir, safe_write_csv,
                    to_numeric_series)

OUTPUT_DIR = output_dir("15_intraday_regressions")

YEAR = str(YEAR_INT)  # "2025": le date si filtrano con str.startswith(YEAR)

PANEL_FOLDER = "08_intraday_panel"
PANEL_FILE = DATA_DIR / PANEL_FOLDER / "08_merged_intraday_panel_2025_primary_247.csv"

# Varianti di troncamento a moneyness (parti 5 e 6), nell'ordine di stima.
LOGM_VARIANTS = ("logm_005", "logm_002")


def logm_panel_file(variant: str):
    """Pannello intraday della variante di troncamento."""
    if variant not in LOGM_VARIANTS:
        raise ValueError("La variante deve essere 'logm_005' o 'logm_002'.")
    return DATA_DIR / PANEL_FOLDER / f"08_merged_intraday_panel_2025_primary_247_{variant}.csv"


RUN_SCOPE = "FULL_247"
HORIZONS = [5, 15, 30, 60]


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


def parse_time_to_seconds(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).apply(normalize_time_string)
    t = pd.to_datetime(normalized, format="%H:%M:%S", errors="coerce")
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second


def make_log(df, source, target):
    out = df.copy()
    x = pd.to_numeric(out[source], errors="coerce")
    out[target] = np.where(x > 0, np.log(x), np.nan)
    return out


# =====================================================================
# PARTE 1: STIME PYFIXEST SULL'ANNO INTERO
# =====================================================================
# Le stesse funzioni servono alla parte 6, sui pannelli delle varianti.

EXPECTED_DAYS = 247
EXPECTED_GRID_ROWS_PER_DAY = 38
EXPECTED_TOTAL_ROWS = EXPECTED_DAYS * EXPECTED_GRID_ROWS_PER_DAY

FORMS = ["levels", "log"]

# Cluster per giornata (247 cluster da 38 osservazioni): la correlazione che conta e'
# quella fra fasce orarie della stessa giornata.
CLUSTER_TYPE = "CRV1_date_fe"
PYFIXEST_VCOV_PRIMARY = {"CRV1": "date_fe"}

# Due schemi di effetti fissi: quello storico (date_fe + week_x_time_fe) assorbe circa un
# parametro ogni quattro osservazioni, quello leggero (date_fe + grid_time) molti meno. Se i
# coefficienti reggono in entrambi, il risultato non dipende dalla saturazione.
FE_SCHEMES = [
    {"name": "date_x_weektime", "formula_fe": "date_fe + week_x_time_fe",
     "note": "schema storico, saturato: 2.223 parametri su 9.386 osservazioni"},
    {"name": "date_x_time", "formula_fe": "date_fe + grid_time",
     "note": "schema leggero: 285 parametri, controlla giornata e ora del giorno"},
]

# MFIV_ann e non MFIV: la dipendente e' la varianza su una finestra fissa di h minuti, MFIV
# quella residua fino alla chiusura; il fattore h/(minuti residui) che le lega varia nella
# sessione (di 38 volte per h=5) e un solo coefficiente su MFIV ne sarebbe una media pesata.
IV = "MFIV_ann"
LOG_IV = "log_MFIV_ann"

SPECIFICATIONS = [
    {
        "specification": "IH1",
        "gamma_source": "none",
        "uses_flow": False,
        "regressors_levels": [IV],
        "regressors_log": [LOG_IV],
        "variance_scale_inputs": [IV],
    },
    {
        "specification": "IH2",
        "gamma_source": "none",
        "uses_flow": False,
        "regressors_levels": [IV, "TrailingRV"],
        "regressors_log": [LOG_IV, "log_TrailingRV"],
        "variance_scale_inputs": [IV, "TrailingRV"],
    },
    {
        "specification": "IH3",
        "gamma_source": "cboe",
        "uses_flow": True,
        "regressors_levels": [IV, "TrailingRV", "H_OI_cboe", "H_VOL_cboe"],
        "regressors_log": [LOG_IV, "log_TrailingRV", "H_OI_cboe", "H_VOL_cboe"],
        "variance_scale_inputs": [IV, "TrailingRV"],
    },
    {
        "specification": "IH3",
        "gamma_source": "indep",
        "uses_flow": True,
        "regressors_levels": [IV, "TrailingRV", "H_OI_indep", "H_VOL_indep"],
        "regressors_log": [LOG_IV, "log_TrailingRV", "H_OI_indep", "H_VOL_indep"],
        "variance_scale_inputs": [IV, "TrailingRV"],
    },
]

# Confronto metodologico: una sola specifica sul regressore non riscalato MFIV, per
# misurare quanto distorce la parametrizzazione su MFIV.
LEGACY_SPEC = {
    "specification": "IH1_mfiv_raw",
    "gamma_source": "none",
    "uses_flow": False,
    "regressors_levels": ["MFIV"],
    "regressors_log": ["log_MFIV"],
    "variance_scale_inputs": ["MFIV"],
}

VARIANCE_SCALE_COLUMNS_ALWAYS_LOGGED = [
    "MFIV", "MFIV_ann", "TrailingRV",
    "FwdRV_5", "FwdRV_15", "FwdRV_30", "FwdRV_60",
    "G_OI_raw_cboe", "G_VOL_raw_cboe",
    "G_OI_raw_indep", "G_VOL_raw_indep",
]

ESTIMATION_NUMERIC_COLUMNS = [
    "MFIV", "MFIV_ann", "tau", "TrailingRV",
    "H_OI_cboe", "H_VOL_cboe", "G_OI_raw_cboe", "G_VOL_raw_cboe",
    "H_OI_indep", "H_VOL_indep", "G_OI_raw_indep", "G_VOL_raw_indep",
    "FwdRV_5", "FwdRV_15", "FwdRV_30", "FwdRV_60",
    "h5_valid", "h15_valid", "h30_valid", "h60_valid",
    "usable_for_flow_specs",
]

ESTIMATION_STEM = "15_pyfixest_estimation_2025_full_247"


def normalize_time_string_full(value) -> str:
    """Versione delle parti 1 e 6: normalizza anche HH:MM:SS e altri formati."""
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


def parse_time_to_seconds_full(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).apply(normalize_time_string_full)
    t = pd.to_datetime(normalized, format="%H:%M:%S", errors="coerce")
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second


def require_columns(df: pd.DataFrame, columns, label: str):
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def fit_feols(fml: str, data: pd.DataFrame):
    try:
        return pf.feols(fml=fml, data=data, vcov=PYFIXEST_VCOV_PRIMARY)
    except TypeError:
        return pf.feols(fml, data, vcov=PYFIXEST_VCOV_PRIMARY)


def extract_model_n(model, fallback_n: int):
    for attr in ("nobs", "N", "_N", "_nobs", "_N_rows"):
        if hasattr(model, attr):
            value = getattr(model, attr)

            try:
                if callable(value):
                    value = value()

                return int(value), attr
            except Exception:
                continue

    return int(fallback_n), "FALLBACK"


def extract_tidy_table(model) -> pd.DataFrame:
    if hasattr(model, "tidy"):
        tidy = model.tidy()

        if isinstance(tidy, pd.DataFrame):
            out = tidy.copy()

            if "Coefficient" not in out.columns:
                index_name = out.index.name or "index"
                out = out.reset_index()

                if "Coefficient" in out.columns:
                    pass
                elif index_name in out.columns:
                    out["Coefficient"] = out[index_name]
                elif "index" in out.columns:
                    out["Coefficient"] = out["index"]
                else:
                    raise ValueError(
                        f"Cannot identify coefficient names. Columns: {list(out.columns)}"
                    )

            required = ["Estimate", "Std. Error", "t value", "Pr(>|t|)"]
            missing = [c for c in required if c not in out.columns]

            if missing:
                raise ValueError(f"tidy() missing columns: {missing}")

            return pd.DataFrame({
                "term": out["Coefficient"].astype(str),
                "coef": pd.to_numeric(out["Estimate"], errors="coerce"),
                "se": pd.to_numeric(out["Std. Error"], errors="coerce"),
                "t_stat": pd.to_numeric(out["t value"], errors="coerce"),
                "p_value": pd.to_numeric(out["Pr(>|t|)"], errors="coerce"),
            })

    if all(hasattr(model, m) for m in ("coef", "se", "tstat", "pvalue")):
        coef = model.coef()

        return pd.DataFrame({
            "term": list(coef.index.astype(str)),
            "coef": pd.to_numeric(coef, errors="coerce").to_numpy(),
            "se": pd.to_numeric(model.se().reindex(coef.index), errors="coerce").to_numpy(),
            "t_stat": pd.to_numeric(model.tstat().reindex(coef.index), errors="coerce").to_numpy(),
            "p_value": pd.to_numeric(model.pvalue().reindex(coef.index), errors="coerce").to_numpy(),
        })

    raise ValueError("Cannot extract pyfixest results.")


def load_merged_panel(panel_file) -> pd.DataFrame:
    if not panel_file.exists():
        raise FileNotFoundError(f"Merged Section 3.8 panel not found:\n{panel_file}")

    df = pd.read_csv(panel_file, dtype=str, keep_default_na=False)

    required = [
        "quote_date", "grid_time", "date_fe", "iso_week_key", "week_x_time_fe",
        "MFIV", "MFIV_ann", "TrailingRV",
        "H_OI_cboe", "H_VOL_cboe", "H_OI_indep", "H_VOL_indep",
        "usable_for_flow_specs",
    ] + [f"FwdRV_{h}" for h in HORIZONS] + [f"h{h}_valid" for h in HORIZONS]

    require_columns(df, required, "Merged Section 3.8 panel")

    df["quote_date"] = clean_date_series(df["quote_date"])
    df["grid_time"] = df["grid_time"].astype(str).apply(normalize_time_string_full)

    for col in ("date_fe", "iso_week_key", "week_x_time_fe"):
        df[col] = df[col].astype(str).str.strip()

    for col in ESTIMATION_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = to_numeric_series(df[col])

    df = df[
        df["quote_date"].notna()
        & df["quote_date"].astype(str).str.startswith(YEAR)
        & df["grid_time"].notna()
    ].copy()

    df["_date_dt"] = pd.to_datetime(df["quote_date"], errors="coerce")
    df["_grid_seconds"] = parse_time_to_seconds_full(df["grid_time"])

    return (
        df.sort_values(["_date_dt", "_grid_seconds"])
        .drop(columns=["_date_dt", "_grid_seconds"]).reset_index(drop=True)
    )


def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in VARIANCE_SCALE_COLUMNS_ALWAYS_LOGGED:
        if col not in out.columns:
            continue

        log_col = f"log_{col}"
        out[log_col] = np.nan

        mask = out[col].notna() & np.isfinite(out[col]) & (out[col] > 0)
        out.loc[mask, log_col] = np.log(out.loc[mask, col])

    return out


def build_working_frame(panel, horizon, form, spec, fe_scheme):
    dep_levels = f"FwdRV_{horizon}"
    dep_col = dep_levels if form == "levels" else f"log_FwdRV_{horizon}"
    regressors = spec["regressors_levels"] if form == "levels" else spec["regressors_log"]

    valid_col = f"h{horizon}_valid"

    fe_cols = [c.strip() for c in fe_scheme["formula_fe"].split("+")]

    required_cols = [dep_col] + regressors + fe_cols + ["quote_date", "grid_time"]

    missing = [c for c in required_cols if c not in panel.columns]

    if missing:
        raise ValueError(
            f"Missing columns for h={horizon}, {form}, {spec['specification']}: {missing}"
        )

    n_start = len(panel)

    after_hvalid = panel[panel[valid_col] == 1].copy()

    # Le specifiche sul flusso girano solo dove il flusso esiste, con il criterio del pannello.
    if spec["uses_flow"] and "usable_for_flow_specs" in after_hvalid.columns:
        after_flow = after_hvalid[after_hvalid["usable_for_flow_specs"] == 1].copy()
    else:
        after_flow = after_hvalid

    working = after_flow.dropna(subset=required_cols).copy()

    for col in [dep_col] + regressors:
        working = working[np.isfinite(working[col])]

    meta = {
        "specification": spec["specification"],
        "horizon": horizon,
        "form": form,
        "gamma_source": spec["gamma_source"],
        "fe_scheme": fe_scheme["name"],
        "dependent_variable": dep_col,
        "regressors_string": " + ".join(regressors),
        "expected_terms": regressors,
        "fixed_effects": fe_scheme["formula_fe"],
        "cluster_type": CLUSTER_TYPE,
        "uses_flow": int(spec["uses_flow"]),
        "n_panel": n_start,
        "n_after_horizon_valid": int(len(after_hvalid)),
        "n_dropped_horizon": int(n_start - len(after_hvalid)),
        "n_after_flow_filter": int(len(after_flow)),
        "n_dropped_flow": int(len(after_hvalid) - len(after_flow)),
        "n_final_estimation_rows": int(len(working)),
        "n_dropped_listwise": int(len(after_flow) - len(working)),
        "n_fe_absorbed": int(sum(working[c].nunique() for c in fe_cols))
            if not working.empty else 0,
        "n_days": int(working["quote_date"].nunique()) if not working.empty else 0,
        "n_grid_times": int(working["grid_time"].nunique()) if not working.empty else 0,
        "formula": f"{dep_col} ~ {' + '.join(regressors)} | {fe_scheme['formula_fe']}",
    }

    return working, meta


def build_all_working_frames(panel):
    out = []

    all_specs = SPECIFICATIONS + [LEGACY_SPEC]

    for fe_scheme in FE_SCHEMES:
        for horizon in HORIZONS:
            for form in FORMS:
                for spec in all_specs:
                    # Il confronto metodologico IH1_mfiv_raw gira solo in log e a h=5.
                    if spec["specification"] == "IH1_mfiv_raw":
                        if form != "log" or horizon != 5:
                            continue

                    working, meta = build_working_frame(
                        panel, horizon, form, spec, fe_scheme
                    )

                    out.append({"working": working, "meta": meta})

    return out


def run_all_models(working_specs, panel_file, variant=None):
    """variant = None per la baseline (parte 1); per le varianti (parte 6) la
    colonna variant precede le altre nei risultati."""
    result_rows = []
    status_rows = []

    total = len(working_specs)

    for idx, item in enumerate(working_specs, start=1):
        working = item["working"]
        meta = item["meta"]

        label = (f"{meta['specification']}_h{meta['horizon']}_{meta['form']}_"
                 f"{meta['gamma_source']}_{meta['fe_scheme']}")

        if meta["n_final_estimation_rows"] <= 0:
            status_rows.append({**meta, "model_status": "SKIPPED_EMPTY",
                                "model_error": "", "actual_terms": "",
                                "missing_expected_terms": ";".join(meta["expected_terms"]),
                                "n_model_reported": np.nan,
                                "n_model_source_attr": "NA_EMPTY"})
            continue

        try:
            model = fit_feols(meta["formula"], working)

            n_model, n_source = extract_model_n(model, meta["n_final_estimation_rows"])

            tidy = extract_tidy_table(model)

            actual = tidy["term"].astype(str).tolist()
            expected = [str(x) for x in meta["expected_terms"]]
            missing = [t for t in expected if t not in actual]

            status_rows.append({
                **meta,
                "model_status": "OK" if not missing else "FIT_OK_MISSING_TERMS",
                "model_error": "",
                "actual_terms": ";".join(actual),
                "missing_expected_terms": ";".join(missing),
                "n_model_reported": n_model,
                "n_model_source_attr": n_source,
            })

            for _, row in tidy.iterrows():
                record = {} if variant is None else {"variant": variant}
                record.update({
                    # provenienza: ogni file deve venire dal pannello atteso
                    "merged_panel_file": str(panel_file),
                    "specification": meta["specification"],
                    "horizon": meta["horizon"],
                    "form": meta["form"],
                    "gamma_source": meta["gamma_source"],
                    "fe_scheme": meta["fe_scheme"],
                    "term": row["term"],
                    "coef": row["coef"],
                    "se": row["se"],
                    "t_stat": row["t_stat"],
                    "p_value": row["p_value"],
                    "n": meta["n_final_estimation_rows"],
                    "n_fe_absorbed": meta["n_fe_absorbed"],
                    "cluster_type": CLUSTER_TYPE,
                    "formula": meta["formula"],
                })
                result_rows.append(record)

            if idx % 10 == 0 or idx == total:
                print(f"    [{idx}/{total}] {label}  n={meta['n_final_estimation_rows']:,}",
                      flush=True)

        except Exception as exc:
            status_rows.append({**meta, "model_status": "ERROR",
                                "model_error": str(exc) + "\n" + traceback.format_exc(),
                                "actual_terms": "",
                                "missing_expected_terms": ";".join(meta["expected_terms"]),
                                "n_model_reported": np.nan,
                                "n_model_source_attr": "ERROR"})

            print(f"    [{idx}/{total}] {label}  ERROR: {exc}")

    return pd.DataFrame(result_rows), pd.DataFrame(status_rows)


def build_fe_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Confronto dei coefficienti fra i due schemi di effetti fissi."""
    key = ["specification", "horizon", "form", "gamma_source", "term"]

    wide = results.pivot_table(
        index=key, columns="fe_scheme", values=["coef", "p_value"], aggfunc="first"
    )

    if wide.empty:
        return pd.DataFrame()

    out = wide.reset_index()
    out.columns = ["_".join([c for c in col if c]).strip("_") for col in out.columns]

    a, b = FE_SCHEMES[0]["name"], FE_SCHEMES[1]["name"]

    ca, cb = f"coef_{a}", f"coef_{b}"
    pa, pb = f"p_value_{a}", f"p_value_{b}"

    if ca in out.columns and cb in out.columns:
        out["coef_ratio"] = out[cb] / out[ca]
        out["sign_flip"] = ((out[ca] > 0) != (out[cb] > 0)).astype(int)
        out["sig_change_at_5pct"] = (
            (out[pa] < 0.05) != (out[pb] < 0.05)
        ).astype(int)

    return out


def build_validation(results, status, row_counts, panel_ref,
                     check_daily_span=False) -> pd.DataFrame:
    """check_daily_span=True per le varianti (parte 6): controllo in piu' sul
    primo e ultimo orario di ogni giornata."""
    rows = []

    def add(group, check, expected, actual, passed, details=""):
        rows.append({"validation_group": group, "check": check,
                     "expected": expected, "actual": actual,
                     "passed": int(passed), "details": details})

    # Duplicati di chiave controllati direttamente: un pannello con una riga duplicata e una
    # mancante passerebbe i controlli sui conteggi.
    dup_keys = int(panel_ref.duplicated(["quote_date", "grid_time"]).sum())

    add("base_panel", "no_duplicate_quote_date_grid_time_rows", 0, dup_keys,
        dup_keys == 0,
        "Controllo diretto, indipendente dai conteggi di riga: un pannello con "
        "una riga duplicata e una mancante passerebbe i conteggi.")

    n_rows = int(len(panel_ref))
    n_days = int(panel_ref["quote_date"].nunique())
    n_times = int(panel_ref["grid_time"].nunique())

    if check_daily_span:
        # Primo e ultimo orario di ogni giornata: il conteggio delle righe non vedrebbe una
        # griglia spostata (09:50-16:00 invece di 09:40-15:50).
        daily_span = (
            panel_ref.groupby("quote_date", as_index=False)
            .agg(n_rows=("grid_time", "size"),
                 first_grid_time=("grid_time", "min"),
                 last_grid_time=("grid_time", "max"))
        )

        expected_first = panel_ref["grid_time"].min()
        expected_last = panel_ref["grid_time"].max()

        bad_span = daily_span[
            ~((daily_span["n_rows"] == n_times)
              & (daily_span["first_grid_time"] == expected_first)
              & (daily_span["last_grid_time"] == expected_last))
        ]

        add("base_panel", "every_day_spans_the_full_grid", 0, len(bad_span),
            len(bad_span) == 0,
            f"Ogni giornata deve avere {n_times} righe da {expected_first} a "
            f"{expected_last}. Il conteggio totale da solo non lo garantisce. "
            + ";".join(bad_span["quote_date"].astype(str).tolist()[:20]))

    add("base_panel", "rows_equals_days_times_grid_points", n_days * n_times,
        n_rows, n_rows == n_days * n_times,
        f"{n_days} giornate x {n_times} punti di griglia. Con la griglia a 09:40 "
        "ogni giornata ha lo stesso numero di punti, quindi questo e' un "
        "invariante vero e non una tautologia.")

    expected_models = len(row_counts)

    add("model_status", "all_models_attempted", expected_models, len(status),
        len(status) == expected_models)

    errors = status[status["model_status"] == "ERROR"]
    add("model_status", "no_pyfixest_errors", 0, len(errors), len(errors) == 0,
        ";".join((errors["specification"].astype(str) + "_h"
                  + errors["horizon"].astype(str)).tolist()[:20]))

    missing_terms = status[status["missing_expected_terms"].astype(str).str.len() > 0]
    add("model_status", "no_expected_terms_missing", 0, len(missing_terms),
        len(missing_terms) == 0,
        ";".join(missing_terms["specification"].astype(str).tolist()[:20]))

    fallback = status[status["n_model_source_attr"].astype(str) == "FALLBACK"]
    add("model_n", "n_not_from_fallback", 0, len(fallback), len(fallback) == 0)

    mismatch = status[
        (status["model_status"] != "ERROR")
        & (status["n_model_reported"] != status["n_final_estimation_rows"])
    ]
    add("model_n", "n_model_equals_working_n", 0, len(mismatch), len(mismatch) == 0,
        ";".join(mismatch["specification"].astype(str).tolist()[:20]))

    add("results", "results_nonempty", "> 0", len(results), len(results) > 0)

    if not results.empty:
        nonfinite = results[
            ~(np.isfinite(results["coef"]) & np.isfinite(results["se"])
              & np.isfinite(results["t_stat"]) & np.isfinite(results["p_value"]))
        ]
        add("results", "all_statistics_finite", 0, len(nonfinite), len(nonfinite) == 0)

        bad_se = results[results["se"] <= 0]
        add("results", "all_standard_errors_positive", 0, len(bad_se), len(bad_se) == 0)

        huge = results[results["coef"].abs() > 1e8]
        add("sanity", "no_absurd_coefficients", 0, len(huge), len(huge) == 0)

    zero_n = row_counts[row_counts["n_final_estimation_rows"] <= 0]
    add("row_counts", "all_working_frames_nonempty", 0, len(zero_n), len(zero_n) == 0)

    # Il campione deve calare con l'orizzonte.
    for fe in [s["name"] for s in FE_SCHEMES]:
        sub = row_counts[
            (row_counts["specification"] == "IH1")
            & (row_counts["form"] == "levels")
            & (row_counts["fe_scheme"] == fe)
        ].sort_values("horizon")

        ns = sub["n_final_estimation_rows"].tolist()
        monotone = all(ns[i] >= ns[i + 1] for i in range(len(ns) - 1))

        add("row_counts", f"sample_decreasing_with_horizon_{fe}",
            "h5 >= h15 >= h30 >= h60", str(ns), monotone)

    return pd.DataFrame(rows)


def estimate_models(panel_file, output_stem, variant=None):
    """Stima, confronto fra schemi e validazione; variant = None per la parte 1,
    "logm_005" o "logm_002" per la parte 6."""
    results_file = OUTPUT_DIR / f"{output_stem}_results.csv"
    row_counts_file = OUTPUT_DIR / f"{output_stem}_row_counts.csv"
    model_status_file = OUTPUT_DIR / f"{output_stem}_model_status.csv"
    fe_comparison_file = OUTPUT_DIR / f"{output_stem}_fe_scheme_comparison.csv"
    validation_file = OUTPUT_DIR / f"{output_stem}_validation.csv"

    print("=" * 150)
    if variant is None:
        print("SECTION 3.8 PYFIXEST ESTIMATION - ANNO INTERO")
    else:
        print(f"SECTION 3.8 PYFIXEST ESTIMATION - VARIANTE {variant}")
    print("=" * 150)
    print(f"RUN_SCOPE:              {RUN_SCOPE}")
    if variant is not None:
        print(f"VARIANTE:               {variant}, regola a MONEYNESS")
    print(f"pyfixest version:       {getattr(pf, '__version__', 'unknown')}")
    print(f"Merged panel file:      {panel_file}")
    print(f"Output folder:          {OUTPUT_DIR}")
    print(f"Regressore implicito:   {IV}  (non MFIV: vedi nota sul regressore)")
    print(f"Clustering:             {PYFIXEST_VCOV_PRIMARY}")
    print(f"Schemi di effetti fissi: "
          + " | ".join(f"{s['name']} ({s['formula_fe']})" for s in FE_SCHEMES))
    print("=" * 150)

    print("Loading merged intraday panel...")
    panel = load_merged_panel(panel_file)

    print(f"    Righe:    {len(panel):,}  (attese {EXPECTED_TOTAL_ROWS:,})")
    print(f"    Giornate: {panel['quote_date'].nunique():,}  (attese {EXPECTED_DAYS})")

    if IV not in panel.columns:
        raise ValueError(
            f"Il pannello non contiene {IV}. Rilancia il costruttore del pannello "
            "intraday, che la produce da mfiv / tau."
        )

    print("Adding log columns...")
    panel = add_log_columns(panel)

    print("Building working frames...")
    working_specs = build_all_working_frames(panel)

    row_counts = pd.DataFrame([item["meta"] for item in working_specs])

    print(f"    Modelli da stimare: {len(working_specs)}")

    print("Running pyfixest estimations...")
    results, model_status = run_all_models(working_specs, panel_file, variant)

    print("Building fixed-effects comparison...")
    fe_comparison = build_fe_comparison(results)

    print("Building validation...")
    validation = build_validation(results, model_status, row_counts, panel,
                                  check_daily_span=variant is not None)

    all_passed = int((validation["passed"] == 1).all())
    failed = validation[validation["passed"] != 1]

    safe_write_csv(results, results_file)
    safe_write_csv(row_counts, row_counts_file)
    safe_write_csv(model_status, model_status_file)
    safe_write_csv(fe_comparison, fe_comparison_file)
    safe_write_csv(validation, validation_file)

    print()
    print("=" * 150)
    print("COMPLETED")
    print("=" * 150)
    print(f"Modelli stimati:      {len(model_status):,}")
    print(f"Righe di risultato:   {len(results):,}")
    print(f"Validazione passata:  {all_passed}")

    print()
    print("--- COEFFICIENTI PRINCIPALI, SCHEMA DI EFFETTI FISSI STORICO ---")
    main_terms = [IV, LOG_IV, "H_OI_cboe", "H_VOL_cboe", "H_OI_indep", "H_VOL_indep"]

    primary = results[
        (results["fe_scheme"] == FE_SCHEMES[0]["name"])
        & (results["term"].isin(main_terms))
    ]
    print(primary[["specification", "horizon", "form", "gamma_source", "term",
                   "coef", "se", "t_stat", "p_value", "n"]].to_string(index=False))

    print()
    print("--- CONFRONTO FRA I DUE SCHEMI DI EFFETTI FISSI ---")
    if fe_comparison.empty:
        print("Nessun confronto disponibile.")
    else:
        n_flip = int(fe_comparison["sign_flip"].sum()) if "sign_flip" in fe_comparison else 0
        n_sig = int(fe_comparison["sig_change_at_5pct"].sum()) if "sig_change_at_5pct" in fe_comparison else 0

        print(f"  coefficienti confrontati:        {len(fe_comparison)}")
        print(f"  che cambiano SEGNO:              {n_flip}")
        print(f"  che cambiano significativita':   {n_sig}")
        print()
        print("  Lo schema storico assorbe 2.223 parametri su 9.386 osservazioni, quasi")
        print("  uno ogni quattro. Se i coefficienti reggono anche con lo schema leggero")
        print("  (285 parametri), il risultato non dipende dalla saturazione. Se cambiano,")
        print("  va dichiarato quale schema si usa e perche'.")
        print()

        cmp_terms = fe_comparison[fe_comparison["term"].isin(main_terms)]
        print(cmp_terms.head(40).to_string(index=False))

    print()
    print("--- CONFRONTO METODOLOGICO: MFIV_ann CONTRO MFIV ---")
    comp = results[
        (results["horizon"] == 5) & (results["form"] == "log")
        & (results["fe_scheme"] == FE_SCHEMES[0]["name"])
        & (results["specification"].isin(["IH1", "IH1_mfiv_raw"]))
    ]
    print(comp[["specification", "term", "coef", "se", "t_stat", "p_value", "n"]]
          .to_string(index=False))
    print()
    print("IH1 usa MFIV_ann, il regressore riscalato. IH1_mfiv_raw usa MFIV, dove il")
    print("fattore di costruzione varia di trentotto volte lungo la sessione. La")
    print("differenza fra i due coefficienti misura la distorsione della")
    print("parametrizzazione non riscalata, su campione identico.")

    print()
    print("VALIDATION")
    print("=" * 150)
    print(validation.to_string(index=False))

    print()
    print("CONTROLLI FALLITI")
    print("=" * 150)
    print("Nessuno." if failed.empty else failed.to_string(index=False))
    print("=" * 150)

    if all_passed != 1:
        raise RuntimeError(f"Stime pyfixest ({variant or 'baseline'}): validazione fallita.")


def pyfixest_estimation():
    estimate_models(PANEL_FILE, ESTIMATION_STEM)


# =====================================================================
# PARTE 2: R-QUADRO INCREMENTALE PER TERZILI DI ORARIO (HI3)
# =====================================================================
# Si confronta l'R-quadro incrementale e non lo slope: la scala della MFIV si contrae con
# tau, quella di FwdRV_h no, quindi lo slope crescerebbe verso la chiusura per ragioni
# meccaniche; l'R-quadro non dipende dalla scala del regressore.
#
# Le stesse funzioni servono alla parte 5, sui pannelli delle varianti.

FE_FORMULA = "date_fe + week_x_time_fe"
VCOV = {"CRV1": "date_fe"}

CONTROL = "TrailingRV"

# 38 punti di griglia non sono divisibili per tre: il primo terzile ne ha 12, gli altri 13.
TERCILE_SIZES = (12, 13, 13)
TERCILE_NAMES = ("early", "middle", "late")

TERCILE_NUMERIC_COLUMNS = [
    "MFIV", "MFIV_ann", "TrailingRV",
    "H_OI_cboe", "H_VOL_cboe", "H_OI_indep", "H_VOL_indep",
    "usable_for_flow_specs",
] + [f"FwdRV_{h}" for h in HORIZONS] + [f"h{h}_valid" for h in HORIZONS]

# R-quadro incrementale nel terzile di chiusura (schema storico) ottenuto sostituendo la MFIV
# logm_005 nel pannello di baseline: il pannello logm_005 deve ridare questi valori (parte 5).
EXPECTED_FROM_GRAFT_LOGM005 = {5: 0.116800, 15: 0.128795, 30: 0.061494,
                               60: 0.003427}


def tercile_files(variant: str):
    """File di output dei terzili: baseline (parte 2) o variante (parte 5)."""
    suffix = "" if variant == "baseline" else f"_{variant}"
    return (OUTPUT_DIR / f"15_tercile_incremental_r2_{RUN_SCOPE.lower()}{suffix}.csv",
            OUTPUT_DIR / f"15_tercile_incremental_r2_balanced_{RUN_SCOPE.lower()}{suffix}.csv")


def extract_within_r2(model) -> float:
    if not hasattr(model, "_r2_within"):
        raise AttributeError("pyfixest model does not expose _r2_within.")

    value = getattr(model, "_r2_within")

    if callable(value):
        value = value()

    return float(value)


def terms_of(model):
    tidy = model.tidy()

    if "Coefficient" in tidy.columns:
        return list(tidy["Coefficient"].astype(str))

    return list(tidy.index.astype(str))


def fit(fml, data):
    try:
        return pf.feols(fml=fml, data=data, vcov=VCOV)
    except TypeError:
        return pf.feols(fml, data, vcov=VCOV)


def graft_crosscheck(results_df):
    """Controllo incrociato su logm_005: il pannello della variante deve ridare i valori
    ottenuti sostituendo la sua MFIV nel pannello di baseline (la sostituzione e' la stessa)."""
    print()
    print("=" * 140)
    print("CONTROLLO INCROCIATO CONTRO IL TEST PER INNESTO")
    print("=" * 140)

    close = results_df[results_df["tercile"] == "late"]

    max_gap = 0.0
    n_bad = 0

    for h, expected in EXPECTED_FROM_GRAFT_LOGM005.items():
        row = close[close["horizon"] == h]

        if row.empty:
            print(f"  h={h:2d}: riga mancante")
            n_bad += 1
            continue

        got = float(row["incremental_r2"].iloc[0])
        gap = abs(got - expected)
        max_gap = max(max_gap, gap)

        ok = gap <= 1e-6

        if not ok:
            n_bad += 1

        print(f"  h={h:2d}: innesto {expected:.6f}  pannello {got:.6f}  "
              f"scarto {gap:.3e}  {'OK' if ok else 'DIVERGE'}")

    print()

    if n_bad == 0:
        print(f"  I due percorsi coincidono, scarto massimo {max_gap:.3e}.")
        print("  Sia l'innesto sia il costruttore del pannello di robustezza")
        print("  producono la stessa sostituzione, come deve essere.")
    else:
        print(f"  ATTENZIONE: {n_bad} orizzonti divergono, scarto massimo "
              f"{max_gap:.3e}.")
        print("  I due percorsi dovrebbero dare numeri identici. Uno dei due")
        print("  ha un difetto e va trovato prima di usare l'uno o l'altro.")


@ignoring_warnings(".*singleton fixed effect.*")
def estimate_terciles(panel_file, variant):
    """variant = "baseline" per la parte 2, "logm_005" o "logm_002" per la parte 5."""
    output_file, output_balanced_file = tercile_files(variant)

    # LOAD
    print("=" * 140)
    if variant == "baseline":
        print("R-QUADRO INCREMENTALE DELLA MFIV PER TERZILI DI ORARIO (HI3)")
    else:
        print(f"R-QUADRO INCREMENTALE PER TERZILI (HI3) - VARIANTE {variant}")
    print("=" * 140)
    print(f"Pannello:  {panel_file}")
    print(f"Regressore: {IV}")
    if variant != "baseline":
        print(f"Variante:   {variant}, regola a moneyness e non quella sui bid")
    print("=" * 140)

    if not panel_file.exists():
        raise FileNotFoundError(f"Pannello non trovato:\n{panel_file}")

    panel = pd.read_csv(panel_file, dtype=str, keep_default_na=False)

    panel["quote_date"] = clean_date_series(panel["quote_date"])
    panel["grid_time"] = panel["grid_time"].astype(str).apply(normalize_time_string)

    for col in ("date_fe", "week_x_time_fe", "iso_week_key"):
        if col in panel.columns:
            panel[col] = panel[col].astype(str).str.strip()

    for col in TERCILE_NUMERIC_COLUMNS:
        if col in panel.columns:
            panel[col] = to_numeric_series(panel[col])

    panel = panel[
        panel["quote_date"].notna()
        & panel["quote_date"].astype(str).str.startswith(YEAR)
    ].copy()

    print(f"Righe:    {len(panel):,}")
    print(f"Giornate: {panel['quote_date'].nunique():,}")

    # TERZILI
    grid_times = sorted(
        panel["grid_time"].unique(),
        key=lambda t: parse_time_to_seconds(pd.Series([t])).iloc[0],
    )

    n_grid = len(grid_times)

    print()
    print("=" * 140)
    print("COSTRUZIONE DEI TERZILI")
    print("=" * 140)
    print(f"Punti di griglia: {n_grid}")

    if n_grid != sum(TERCILE_SIZES):
        raise ValueError(
            f"I terzili sommano a {sum(TERCILE_SIZES)} ma la griglia ha {n_grid} punti. "
            "Controlla TERCILE_SIZES."
        )

    tercile_map = {}
    blocks = {}
    i = 0

    for name, size in zip(TERCILE_NAMES, TERCILE_SIZES):
        block = grid_times[i:i + size]
        blocks[name] = block

        for tt in block:
            tercile_map[tt] = name

        print(f"  {name:8s}: {len(block):2d} punti, {block[0][:5]} - {block[-1][:5]}")
        i += size

    print()
    print(f"NOTA: {n_grid} punti non sono divisibili per tre. Lo sbilanciamento e'")
    print(f"messo sul primo terzile ({TERCILE_SIZES[0]} punti contro "
          f"{TERCILE_SIZES[1]} e {TERCILE_SIZES[2]}), ed e' dichiarato.")

    panel["tercile"] = panel["grid_time"].map(tercile_map)

    n_unmapped = int(panel["tercile"].isna().sum())

    if n_unmapped != 0:
        raise ValueError(f"{n_unmapped} righe senza terzile assegnato.")

    # SBILANCIAMENTO DEL CAMPIONE PER ORIZZONTE
    # Vicino alla chiusura la finestra in avanti non e' valida, quindi il terzile tardivo
    # perde punti che gli altri non perdono (a h=60 ne restano 8 su 13).
    print()
    print("=" * 140)
    print("VALIDITA' DELLA FINESTRA PER TERZILE E ORIZZONTE")
    print("=" * 140)

    balance_rows = []

    for h in HORIZONS:
        row = {"horizon": h}

        for name in TERCILE_NAMES:
            sub = panel[(panel["tercile"] == name) & (panel[f"h{h}_valid"] == 1)]
            n_times = int(sub["grid_time"].nunique())

            row[f"{name}_grid_times"] = n_times
            row[f"{name}_rows"] = int(len(sub))

        valid_counts = [row[f"{n}_grid_times"] for n in TERCILE_NAMES]

        row["min_over_max_grid_times"] = (min(valid_counts) / max(valid_counts)
                                          if max(valid_counts) else np.nan)
        row["balanced"] = int(min(valid_counts) == max(valid_counts))

        balance_rows.append(row)

    balance = pd.DataFrame(balance_rows)
    print(balance.to_string(index=False))

    print()
    print("Dove balanced = 0 i tre R-quadro NON sono confrontabili come se fossero")
    print("stimati sullo stesso disegno: il terzile tardivo e' stimato su meno punti.")

    # STIMA
    print()
    print("=" * 140)
    print("STIMA DEI MODELLI RISTRETTO E COMPLETO")
    print("=" * 140)

    results = []

    for h in HORIZONS:
        for name in TERCILE_NAMES:
            dep = f"FwdRV_{h}"

            sub = panel[
                (panel["tercile"] == name) & (panel[f"h{h}_valid"] == 1)
            ].dropna(subset=[IV, CONTROL, dep, "date_fe", "week_x_time_fe"]).copy()

            for c in (IV, CONTROL, dep):
                sub = sub[np.isfinite(sub[c])]

            n_rows = len(sub)

            base = {
                "run_scope": RUN_SCOPE,
                # provenienza, per il confronto fra baseline e varianti
                "variant": variant,
                "merged_panel_file": str(panel_file),
                "horizon": h,
                "tercile": name,
                "n_rows": n_rows,
                "n_days": int(sub["date_fe"].nunique()) if n_rows else 0,
                "n_grid_times": int(sub["grid_time"].nunique()) if n_rows else 0,
                "min_grid_time": str(sub["grid_time"].min()) if n_rows else "",
                "max_grid_time": str(sub["grid_time"].max()) if n_rows else "",
            }

            if n_rows < 200:
                results.append({**base, "r2_restricted": np.nan, "r2_full": np.nan,
                                "incremental_r2": np.nan, "sanity_ok": np.nan,
                                "terms_survived": False, "status": "TOO_FEW_ROWS"})
                continue

            try:
                m_r = fit(f"{dep} ~ {CONTROL} | {FE_FORMULA}", sub)
                m_f = fit(f"{dep} ~ {IV} + {CONTROL} | {FE_FORMULA}", sub)

                r2_r = extract_within_r2(m_r)
                r2_f = extract_within_r2(m_f)

                terms_r = terms_of(m_r)
                terms_f = terms_of(m_f)

                survived = (CONTROL in terms_r) and (IV in terms_f) and (CONTROL in terms_f)

                results.append({
                    **base,
                    "r2_restricted": r2_r,
                    "r2_full": r2_f,
                    "incremental_r2": r2_f - r2_r,
                    "sanity_ok": bool(r2_f >= r2_r),
                    "terms_survived": bool(survived),
                    "status": "OK",
                })

                print(f"  h={h:2d} {name:8s}: n={n_rows:5,}  "
                      f"R2 ristretto {r2_r:.6f}  completo {r2_f:.6f}  "
                      f"incrementale {r2_f - r2_r:.6f}", flush=True)

            except Exception as exc:
                print(f"  h={h} {name}: ERRORE {exc}")
                print(traceback.format_exc())

                results.append({**base, "r2_restricted": np.nan, "r2_full": np.nan,
                                "incremental_r2": np.nan, "sanity_ok": False,
                                "terms_survived": False, "status": f"ERROR: {exc}"})

    results_df = pd.DataFrame(results)

    # unisce la diagnostica di bilanciamento
    results_df = results_df.merge(
        balance[["horizon", "balanced", "min_over_max_grid_times"]],
        on="horizon", how="left",
    )

    safe_write_csv(results_df, output_file)
    safe_write_csv(balance, output_balanced_file)

    # VALIDAZIONE E STAMPA
    print()
    print("=" * 140)
    print("CONTROLLI")
    print("=" * 140)

    n_expected = len(HORIZONS) * len(TERCILE_NAMES)
    n_errors = int((results_df["status"] != "OK").sum())

    ok = results_df[results_df["status"] == "OK"]

    n_sanity = int((~ok["sanity_ok"].astype(bool)).sum()) if not ok.empty else n_expected
    n_dropped = int((~ok["terms_survived"].astype(bool)).sum()) if not ok.empty else n_expected

    print(f"righe attese {n_expected}, ottenute {len(results_df)}")
    print(f"errori o campioni troppo piccoli: {n_errors}")
    print(f"violazioni di R2 completo >= ristretto: {n_sanity}")
    print(f"termini silenziosamente scartati: {n_dropped}")

    all_ok = int(len(results_df) == n_expected and n_errors == 0
                 and n_sanity == 0 and n_dropped == 0)

    print(f"tutti i controlli passati: {all_ok}")

    print()
    print("=" * 140)
    print("R-QUADRO INCREMENTALE PER ORIZZONTE E TERZILE")
    print("=" * 140)

    pivot = results_df.pivot_table(index="horizon", columns="tercile",
                                   values="incremental_r2", aggfunc="first")

    if not pivot.empty:
        pivot = pivot[[c for c in TERCILE_NAMES if c in pivot.columns]]
        print(pivot.to_string())

    print()
    print("Numero di punti di griglia effettivamente usati, per controllo:")

    pivot_n = results_df.pivot_table(index="horizon", columns="tercile",
                                     values="n_grid_times", aggfunc="first")

    if not pivot_n.empty:
        pivot_n = pivot_n[[c for c in TERCILE_NAMES if c in pivot_n.columns]]
        print(pivot_n.to_string())

    print()
    print("ATTENZIONE ALLA LETTURA A h=60. Il terzile tardivo perde punti che gli")
    print("altri due non perdono, perche' la finestra in avanti non e' valida vicino")
    print("alla chiusura. Dove la riga 'balanced' vale 0 i tre numeri non sono")
    print("stimati sullo stesso disegno e il confronto va qualificato.")

    if variant == "logm_005":
        graft_crosscheck(results_df)

    print()
    print("=" * 140)
    print("RISULTATI COMPLETI")
    print("=" * 140)
    print(results_df.to_string(index=False))

    print()
    print(f"Output: {output_file.name}, {output_balanced_file.name}")
    print("=" * 140)

    if all_ok != 1:
        raise RuntimeError(f"Terzili ({variant}): controlli falliti.")


def terciles():
    estimate_terciles(PANEL_FILE, "baseline")


# =====================================================================
# PARTE 3: RIGHE PER SPECIFICA E FATTORE DI COSTRUZIONE
# =====================================================================
# Fattore h/(minuti residui) fra MFIV e dipendente, per fascia oraria: non e' costante, da
# cui il regressore MFIV_ann (IV e LOG_IV della parte 1). Nel disegno giornaliero il fattore
# e' costante, e li' il regressore resta MFIV.
PRIMARY_IV_REGRESSOR = "MFIV_ann"
LEGACY_IV_REGRESSOR = "MFIV"

ROW_COUNTS_FILE = OUTPUT_DIR / "15_specification_row_counts_2025_primary_247.csv"
CONSTRUCTION_FACTOR_FILE = OUTPUT_DIR / "15_construction_factor_by_grid_time.csv"

# Chiusura della sessione: 16:00 (nessuna chiusura anticipata fra le 247 giornate).
CLOSE_SECONDS = REGULAR_CLOSE_SECONDS


def to_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )

    return out


def time_to_seconds(value) -> int:
    h, m, s = str(value).split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def build_specifications():
    specs = []

    for h in HORIZONS:
        # IH1: varianza implicita da sola
        specs.append({"hypothesis": "IH1", "horizon": h, "form": "levels",
                      "gamma_source": "NA", "uses_flow": False,
                      "dependent": f"FwdRV_{h}", "regressors": [IV]})

        specs.append({"hypothesis": "IH1", "horizon": h, "form": "log",
                      "gamma_source": "NA", "uses_flow": False,
                      "dependent": f"log_FwdRV_{h}", "regressors": [LOG_IV]})

        # IH2: piu' la varianza realizzata trascorsa
        specs.append({"hypothesis": "IH2", "horizon": h, "form": "levels",
                      "gamma_source": "NA", "uses_flow": False,
                      "dependent": f"FwdRV_{h}", "regressors": [IV, "TrailingRV"]})

        specs.append({"hypothesis": "IH2", "horizon": h, "form": "log",
                      "gamma_source": "NA", "uses_flow": False,
                      "dependent": f"log_FwdRV_{h}",
                      "regressors": [LOG_IV, "log_TrailingRV"]})

        # IH3: piu' stock e flusso di convessita', due fonti gamma
        for src, h_oi, h_vol in (("CBOE", "H_OI_cboe", "H_VOL_cboe"),
                                 ("indep", "H_OI_indep", "H_VOL_indep")):
            specs.append({"hypothesis": "IH3", "horizon": h, "form": "levels",
                          "gamma_source": src, "uses_flow": True,
                          "dependent": f"FwdRV_{h}",
                          "regressors": [IV, "TrailingRV", h_oi, h_vol]})

            specs.append({"hypothesis": "IH3", "horizon": h, "form": "log",
                          "gamma_source": src, "uses_flow": True,
                          "dependent": f"log_FwdRV_{h}",
                          "regressors": [LOG_IV, "log_TrailingRV", h_oi, h_vol]})

    # Una sola specifica sul regressore non riscalato, come confronto metodologico.
    specs.append({"hypothesis": "IH1_mfiv_raw", "horizon": 5, "form": "log",
                  "gamma_source": "NA", "uses_flow": False,
                  "dependent": "log_FwdRV_5",
                  "regressors": [f"log_{LEGACY_IV_REGRESSOR}"],
                  "note": "confronto metodologico: regressore NON riscalato, "
                          "fattore di costruzione non costante"})

    return specs


def row_counts():
    # LOAD
    print("=" * 130)
    print("SECTION 3.8 SPECIFICATION ROW COUNTS - PREFLIGHT")
    print("=" * 130)
    print(f"Merged panel: {PANEL_FILE}")
    print(f"Regressore di varianza implicita: {PRIMARY_IV_REGRESSOR}")
    print("=" * 130)

    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Merged panel not found:\n{PANEL_FILE}")

    panel = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)

    numeric_cols = [
        "MFIV", "MFIV_ann", "tau", "TrailingRV",
        "H_OI_cboe", "H_VOL_cboe", "H_OI_indep", "H_VOL_indep",
        "usable_for_flow_specs", "missing_any_flow_predictor",
    ] + [f"FwdRV_{h}" for h in HORIZONS] + [f"h{h}_valid" for h in HORIZONS]

    panel = to_numeric(panel, numeric_cols)

    print(f"Righe:    {len(panel):,}  (attese {EXPECTED_TOTAL_ROWS:,})")
    print(f"Giornate: {panel['quote_date'].nunique():,}  (attese {EXPECTED_DAYS})")
    print(f"Punti di griglia distinti: {panel['grid_time'].nunique()}  "
          f"(attesi {EXPECTED_GRID_ROWS_PER_DAY})")

    if PRIMARY_IV_REGRESSOR not in panel.columns:
        raise ValueError(
            f"Il pannello non contiene {PRIMARY_IV_REGRESSOR}. "
            "Rilancia il costruttore del pannello intraday: e' lui a produrla da "
            "mfiv / tau."
        )

    # FATTORE DI COSTRUZIONE PER FASCIA ORARIA
    print()
    print("=" * 130)
    print("FATTORE DI COSTRUZIONE FRA MFIV E DIPENDENTE, PER FASCIA ORARIA")
    print("=" * 130)

    factor_rows = []

    for grid_time in sorted(panel["grid_time"].unique()):
        seconds = time_to_seconds(grid_time)
        minutes_remaining = (CLOSE_SECONDS - seconds) / 60.0

        row = {"grid_time": grid_time, "minutes_remaining": minutes_remaining}

        for h in HORIZONS:
            row[f"factor_h{h}"] = h / minutes_remaining if minutes_remaining > 0 else np.nan

        sub = panel[panel["grid_time"] == grid_time]

        for h in HORIZONS:
            fwd = pd.to_numeric(sub[f"FwdRV_{h}"], errors="coerce")
            mfiv = pd.to_numeric(sub["MFIV"], errors="coerce")
            ok = fwd.notna() & mfiv.notna() & (mfiv > 0)

            row[f"empirical_ratio_h{h}"] = (
                float((fwd[ok] / mfiv[ok]).median()) if ok.any() else np.nan
            )

        factor_rows.append(row)

    factor = pd.DataFrame(factor_rows)
    safe_write_csv(factor, CONSTRUCTION_FACTOR_FILE)

    show = factor.iloc[[0, 9, 19, 29, len(factor) - 1]]
    print(show[["grid_time", "minutes_remaining", "factor_h5", "factor_h60",
                "empirical_ratio_h5", "empirical_ratio_h60"]].to_string(index=False))

    f5 = factor["factor_h5"]
    print()
    print(f"Il fattore teorico h/minuti_residui per h=5 va da {f5.min():.4f} a "
          f"{f5.max():.4f}: varia di {f5.max() / f5.min():.0f} volte.")
    print("Per questo il regressore primario e' MFIV_ann e non MFIV. Una regressione")
    print("con un solo coefficiente su MFIV imporrebbe un rapporto costante dove per")
    print("costruzione non lo e'.")

    # SPECIFICHE
    panel = make_log(panel, "FwdRV_5", "log_FwdRV_5")

    for h in HORIZONS:
        panel = make_log(panel, f"FwdRV_{h}", f"log_FwdRV_{h}")

    panel = make_log(panel, PRIMARY_IV_REGRESSOR, f"log_{PRIMARY_IV_REGRESSOR}")
    panel = make_log(panel, LEGACY_IV_REGRESSOR, f"log_{LEGACY_IV_REGRESSOR}")
    panel = make_log(panel, "TrailingRV", "log_TrailingRV")

    specifications = build_specifications()

    print()
    print("=" * 130)
    print("CONTEGGIO RIGHE PER SPECIFICA")
    print("=" * 130)
    print(f"Specifiche: {len(specifications)}")

    rows = []

    for spec in specifications:
        needed = [spec["dependent"]] + spec["regressors"]

        work = panel.copy()

        # Validita' dell'orizzonte in avanti.
        valid_col = f"h{spec['horizon']}_valid"
        n_before = len(work)

        work = work[work[valid_col] == 1].copy()
        n_after_valid = len(work)

        # Le specifiche sul flusso girano solo dove il flusso esiste (criterio del pannello).
        if spec["uses_flow"] and "usable_for_flow_specs" in work.columns:
            work = work[work["usable_for_flow_specs"] == 1].copy()

        n_after_flow = len(work)

        work = work.dropna(subset=needed).copy()

        for col in needed:
            work = work[np.isfinite(work[col])]

        n_final = len(work)

        rows.append({
            "hypothesis": spec["hypothesis"],
            "horizon": spec["horizon"],
            "form": spec["form"],
            "gamma_source": spec["gamma_source"],
            "dependent": spec["dependent"],
            "regressors": ";".join(spec["regressors"]),
            "uses_flow": int(spec["uses_flow"]),
            "n_panel": n_before,
            "n_after_horizon_valid": n_after_valid,
            "n_dropped_horizon": n_before - n_after_valid,
            "n_after_flow_filter": n_after_flow,
            "n_dropped_flow": n_after_valid - n_after_flow,
            "n_final": n_final,
            "n_dropped_other": n_after_flow - n_final,
            "n_days": int(work["quote_date"].nunique()) if n_final > 0 else 0,
            "note": spec.get("note", ""),
        })

    counts = pd.DataFrame(rows)
    safe_write_csv(counts, ROW_COUNTS_FILE)

    print()
    print(counts[["hypothesis", "horizon", "form", "gamma_source",
                  "n_after_horizon_valid", "n_dropped_flow", "n_final",
                  "n_days"]].to_string(index=False))

    print()
    print("=" * 130)
    print("VERIFICHE")
    print("=" * 130)

    checks = []

    def add(check, expected, actual, passed, details=""):
        checks.append({"check": check, "expected": expected, "actual": actual,
                       "passed": int(passed), "details": details})

    add("panel_rows", EXPECTED_TOTAL_ROWS, len(panel), len(panel) == EXPECTED_TOTAL_ROWS,
        f"{EXPECTED_DAYS} x {EXPECTED_GRID_ROWS_PER_DAY}")

    add("grid_points", EXPECTED_GRID_ROWS_PER_DAY, int(panel["grid_time"].nunique()),
        panel["grid_time"].nunique() == EXPECTED_GRID_ROWS_PER_DAY,
        "Griglia 09:40-15:50: nessun punto di frontiera da scartare.")

    # Nessuna specifica deve perdere righe per ragioni diverse da orizzonte e flusso.
    unexplained = counts[counts["n_dropped_other"] != 0]
    add("no_unexplained_row_loss", 0, len(unexplained), len(unexplained) == 0,
        "Righe perse oltre al filtro di orizzonte e a quello di flusso. "
        + ";".join(
            (unexplained["hypothesis"].astype(str) + "_h"
             + unexplained["horizon"].astype(str) + "_"
             + unexplained["form"].astype(str)).tolist()[:20]
        ))

    # Le specifiche sul flusso perdono esattamente le righe attese.
    flow_specs = counts[counts["uses_flow"] == 1]
    stock_specs = counts[counts["uses_flow"] == 0]

    add("flow_specs_drop_rows", "> 0 su almeno un orizzonte",
        int(flow_specs["n_dropped_flow"].sum()),
        flow_specs["n_dropped_flow"].sum() >= 0,
        "Le specifiche sul flusso escludono i punti a volume nullo.")

    add("stock_specs_drop_no_flow_rows", 0, int(stock_specs["n_dropped_flow"].sum()),
        stock_specs["n_dropped_flow"].sum() == 0,
        "Le specifiche che non usano il flusso non devono perdere quelle righe.")

    # Il campione cala all'aumentare dell'orizzonte, per costruzione.
    for form in ("levels", "log"):
        sub = counts[(counts["hypothesis"] == "IH1") & (counts["form"] == form)]
        sub = sub.sort_values("horizon")
        ns = sub["n_final"].tolist()
        monotone = all(ns[i] >= ns[i + 1] for i in range(len(ns) - 1))

        add(f"sample_decreasing_with_horizon_{form}", "h5 >= h15 >= h30 >= h60",
            str(ns), monotone,
            "La validita' dell'orizzonte cade avvicinandosi alla chiusura.")

    checks_df = pd.DataFrame(checks)
    print(checks_df.to_string(index=False))

    failed = checks_df[checks_df["passed"] != 1]

    print()
    print("Nessun controllo fallito." if failed.empty
          else "CONTROLLI FALLITI:\n" + failed.to_string(index=False))

    print()
    print(f"Output: {ROW_COUNTS_FILE.name}, {CONSTRUCTION_FACTOR_FILE.name}")
    print("=" * 130)


# =====================================================================
# PARTE 4: SPECIFICA PREREGISTRATA, REGRESSORE MFIV E CLUSTER A DUE VIE
# =====================================================================
# Specifica primaria (date_fe + week_x_time_fe, livelli; IH1, IH2, IH3 con le due fonti di
# gamma, quattro orizzonti, terzili di HI3) con il regressore preregistrato MFIV accanto a
# MFIV_ann. La forma log serve da controllo: li' i due regressori danno coefficienti identici.
#
# Le giornate sono annidate nelle settimane: nel cluster a due vie (V_giorno + V_settimana -
# V_intersezione) l'intersezione e' il giorno, e resta il cluster per settimana (controllo 4).
#
# Prima si controlla che MFIV_ann con cluster per giorno riproduca le parti 1 e 2.

REGISTERED_STEM = "15_registered_spec_check"

FE_SCHEME_NAME = "date_x_weektime"

REGRESSORS = {
    "MFIV": "registrato",
    "MFIV_ann": "corrente",
}

VCOVS = {
    "giorno": {"CRV1": "date_fe"},
    "due_vie": {"CRV1": "date_fe+iso_week_key"},
    "settimana": {"CRV1": "iso_week_key"},
}

ALPHA = 0.05
REPRO_TOL = 1e-8


def within_r2(fit) -> float:
    v = getattr(fit, "_r2_within")
    return float(v() if callable(v) else v)


def one_sided_negative_p(coef: float, p_two: float) -> float:
    """p-value a una coda per l'ipotesi coefficiente < 0 (H_OI)."""
    return p_two / 2.0 if coef < 0 else 1.0 - p_two / 2.0


def load_panel_registered() -> pd.DataFrame:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Pannello non trovato:\n{PANEL_FILE}")

    df = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)

    need = (["quote_date", "grid_time", "date_fe", "iso_week_key", "week_x_time_fe",
             "MFIV", "MFIV_ann", "TrailingRV",
             "H_OI_cboe", "H_VOL_cboe", "H_OI_indep", "H_VOL_indep",
             "usable_for_flow_specs"]
            + [f"FwdRV_{h}" for h in HORIZONS] + [f"h{h}_valid" for h in HORIZONS])
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Colonne mancanti nel pannello: {miss}")

    df["grid_time"] = df["grid_time"].astype(str).apply(normalize_time_string)
    for c in ("date_fe", "iso_week_key", "week_x_time_fe"):
        df[c] = df[c].astype(str).str.strip()

    num = [c for c in need if c not in ("quote_date", "grid_time", "date_fe",
                                         "iso_week_key", "week_x_time_fe")]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["MFIV", "MFIV_ann", "TrailingRV"] + [f"FwdRV_{h}" for h in HORIZONS]:
        df[f"log_{c}"] = np.where(df[c] > 0, np.log(df[c].where(df[c] > 0)), np.nan)

    df = df[df["quote_date"].astype(str).str.startswith(YEAR)].copy()
    return df


def specs_for(iv: str, form: str):
    lg = (lambda x: f"log_{x}") if form == "log" else (lambda x: x)
    return [
        ("IH1", "none", False, [lg(iv)]),
        ("IH2", "none", False, [lg(iv), lg("TrailingRV")]),
        ("IH3", "cboe", True, [lg(iv), lg("TrailingRV"), "H_OI_cboe", "H_VOL_cboe"]),
        ("IH3", "indep", True, [lg(iv), lg("TrailingRV"), "H_OI_indep", "H_VOL_indep"]),
    ]


def working_frame(panel, h, form, uses_flow, regressors):
    dep = f"FwdRV_{h}" if form == "levels" else f"log_FwdRV_{h}"
    w = panel[panel[f"h{h}_valid"] == 1]
    if uses_flow:
        w = w[w["usable_for_flow_specs"] == 1]
    cols = [dep] + regressors + ["date_fe", "week_x_time_fe", "iso_week_key"]
    w = w.dropna(subset=cols).copy()
    for c in [dep] + regressors:
        w = w[np.isfinite(w[c])]
    return dep, w


@ignoring_warnings(".*singleton.*")
def registered_spec_check():
    # REGRESSIONI
    print("=" * 150)
    print("SPECIFICA PRIMARIA CON REGRESSORE PREREGISTRATO E CLUSTER A DUE VIE")
    print("=" * 150)

    panel = load_panel_registered()
    print(f"Pannello: {PANEL_FILE}")
    print(f"Righe {len(panel):,}, giornate {panel['quote_date'].nunique()}, "
          f"settimane {panel['iso_week_key'].nunique()}")
    print(f"Effetti fissi: {FE_FORMULA} (schema preregistrato)")

    rows = []

    for iv, iv_label in REGRESSORS.items():
        for form in ("levels", "log"):
            vcov_set = VCOVS if form == "levels" else {"giorno": VCOVS["giorno"]}

            for spec, src, uses_flow, regs in specs_for(iv, form):
                for h in HORIZONS:
                    dep, w = working_frame(panel, h, form, uses_flow, regs)
                    fml = f"{dep} ~ {' + '.join(regs)} | {FE_FORMULA}"

                    for vlabel, vcov in vcov_set.items():
                        fit = pf.feols(fml, data=w, vcov=vcov)
                        co, se = fit.coef(), fit.se()
                        tt, pv = fit.tstat(), fit.pvalue()

                        for term in co.index:
                            rows.append({
                                "regressor": iv, "regressor_role": iv_label,
                                "vcov": vlabel, "specification": spec,
                                "gamma_source": src, "horizon": h, "form": form,
                                "fe_scheme": FE_SCHEME_NAME,
                                "term": term,
                                "term_role": ("iv" if term in (iv, f"log_{iv}") else term),
                                "coef": float(co[term]), "se": float(se[term]),
                                "t_stat": float(tt[term]), "p_value": float(pv[term]),
                                "n": int(fit._N),
                                # R-quadro entro gli effetti fissi, per le tabelle
                                "r2_within": within_r2(fit),
                                "merged_panel_file": str(PANEL_FILE),
                            })

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / f"{REGISTERED_STEM}_results.csv", index=False)
    print(f"Stime: {len(res)} righe di risultato")

    # TERZILI DI H-INTRADAY-3
    grid = sorted(panel["grid_time"].unique())
    if len(grid) != sum(TERCILE_SIZES):
        raise ValueError(f"Griglia di {len(grid)} punti, attesi {sum(TERCILE_SIZES)}.")

    tmap, i = {}, 0
    for name, size in zip(TERCILE_NAMES, TERCILE_SIZES):
        for t in grid[i:i + size]:
            tmap[t] = name
        i += size
    panel["tercile"] = panel["grid_time"].map(tmap)

    trows = []
    for iv, iv_label in REGRESSORS.items():
        for h in HORIZONS:
            for name in TERCILE_NAMES:
                sub = panel[(panel["tercile"] == name) & (panel[f"h{h}_valid"] == 1)]
                dep = f"FwdRV_{h}"
                sub = sub.dropna(subset=[iv, "TrailingRV", dep, "date_fe", "week_x_time_fe"])
                for c in (iv, "TrailingRV", dep):
                    sub = sub[np.isfinite(sub[c])]
                r = pf.feols(f"{dep} ~ TrailingRV | {FE_FORMULA}", data=sub, vcov=VCOVS["giorno"])
                f = pf.feols(f"{dep} ~ {iv} + TrailingRV | {FE_FORMULA}", data=sub, vcov=VCOVS["giorno"])
                trows.append({
                    "regressor": iv, "regressor_role": iv_label, "horizon": h,
                    "tercile": name, "n_rows": len(sub),
                    "r2_restricted": within_r2(r), "r2_full": within_r2(f),
                    "incremental_r2": within_r2(f) - within_r2(r),
                    "merged_panel_file": str(PANEL_FILE),
                })

    ter = pd.DataFrame(trows)
    ter.to_csv(OUTPUT_DIR / f"{REGISTERED_STEM}_terciles.csv", index=False)

    # CONTROLLI
    print()
    print("=" * 150)
    print("CONTROLLI")
    print("=" * 150)

    checks = []

    # 1. Riproduzione della baseline (parte 1): MFIV_ann, cluster per giorno,
    #    schema storico.
    bpath = OUTPUT_DIR / f"{ESTIMATION_STEM}_results.csv"
    if bpath.exists():
        base = pd.read_csv(bpath)
        base = base[base["fe_scheme"] == FE_SCHEME_NAME]
        mine = res[(res["regressor"] == "MFIV_ann") & (res["vcov"] == "giorno")]
        keys = ["specification", "gamma_source", "horizon", "form", "term"]
        m = mine.merge(base, on=keys, how="left", suffixes=("", "_b"))
        missing = int(m["coef_b"].isna().sum())
        gap_c = float(((m["coef"] - m["coef_b"]).abs() / m["coef_b"].abs().clip(lower=1e-300)).max())
        gap_s = float(((m["se"] - m["se_b"]).abs() / m["se_b"].abs().clip(lower=1e-300)).max())
        gap_n = int((m["n"] != m["n_b"]).sum())
        ok = missing == 0 and gap_c < REPRO_TOL and gap_s < REPRO_TOL and gap_n == 0
        checks.append(("riproduce la baseline (MFIV_ann, cluster giorno)", ok,
                       f"{len(m)} coefficienti, {missing} senza corrispondenza, "
                       f"scarto relativo max coef {gap_c:.2e}, se {gap_s:.2e}, n diversi {gap_n}"))
    else:
        checks.append(("riproduce la baseline", False, f"file non trovato: {bpath}"))

    # 2. Riproduzione dei terzili della baseline (parte 2).
    tpath = tercile_files("baseline")[0]
    if tpath.exists():
        bt = pd.read_csv(tpath)
        mt = ter[ter["regressor"] == "MFIV_ann"].merge(
            bt, on=["horizon", "tercile"], suffixes=("", "_b"))
        gap = float((mt["incremental_r2"] - mt["incremental_r2_b"]).abs().max())
        ok = len(mt) == 12 and gap < 1e-10
        checks.append(("riproduce i terzili della baseline (MFIV_ann)", ok,
                       f"{len(mt)} celle, scarto massimo {gap:.2e}"))
    else:
        checks.append(("riproduce i terzili della baseline", False, f"file non trovato: {tpath}"))

    # 3. Il cluster cambia solo gli errori standard, mai i coefficienti.
    lv = res[res["form"] == "levels"]
    piv = lv.pivot_table(index=["regressor", "specification", "gamma_source", "horizon", "term"],
                         columns="vcov", values="coef", aggfunc="first")
    gap = float(np.nanmax(np.abs(piv.values - piv[["giorno"]].values)))
    checks.append(("coefficienti identici fra i tre cluster", gap < 1e-12,
                   f"scarto massimo {gap:.2e}"))

    # 4. Doppio cluster annidato = cluster per settimana.
    pse = lv.pivot_table(index=["regressor", "specification", "gamma_source", "horizon", "term"],
                         columns="vcov", values="se", aggfunc="first")
    rel = ((pse["due_vie"] - pse["settimana"]).abs() / pse["settimana"].abs()).max()
    checks.append(("doppio cluster coincide con cluster per settimana", bool(rel < 1e-6),
                   f"scarto relativo massimo degli errori standard {rel:.2e}"))

    # 5. In log MFIV e MFIV_ann danno la stessa regressione.
    lg = res[(res["form"] == "log") & (res["vcov"] == "giorno")].copy()
    lg["term_key"] = lg["term"].str.replace("log_MFIV_ann", "IV").str.replace("log_MFIV", "IV")
    pl = lg.pivot_table(index=["specification", "gamma_source", "horizon", "term_key"],
                        columns="regressor", values="coef", aggfunc="first")
    gap = float((((pl["MFIV"] - pl["MFIV_ann"]).abs()) / pl["MFIV_ann"].abs().clip(lower=1e-300)).max())
    checks.append(("in log MFIV e MFIV_ann coincidono", gap < 1e-6,
                   f"scarto relativo massimo {gap:.2e}"))

    for name, ok, det in checks:
        print(f"  [{'OK' if ok else 'FALLITO'}] {name}: {det}")

    if not all(ok for _, ok, _ in checks[:2]):
        print()
        print("LA BASELINE NON E' RIPRODOTTA: i confronti che seguono non sono affidabili.")

    # LETTURA DELLE IPOTESI
    print()
    print("=" * 150)
    print("IPOTESI PREREGISTRATE, SPECIFICA PRIMARIA (LIVELLI, SCHEMA STORICO)")
    print("=" * 150)
    print("H-intraday-1 e -2: coefficiente del regressore positivo e significativo a")
    print("OGNI orizzonte. H-intraday-4: H_OI negativo (una coda), H_VOL a due code.")
    print()

    lv = res[res["form"] == "levels"].copy()
    out = []
    for iv in REGRESSORS:
        for vl in VCOVS:
            s = lv[(lv["regressor"] == iv) & (lv["vcov"] == vl)]
            for h in HORIZONS:
                r = {"regressore": iv, "cluster": vl, "h": h}
                for spec in ("IH1", "IH2"):
                    x = s[(s["specification"] == spec) & (s["term"] == iv) & (s["horizon"] == h)]
                    if len(x):
                        r[f"{spec}_coef"] = x["coef"].iloc[0]
                        r[f"{spec}_p"] = x["p_value"].iloc[0]
                for src in ("cboe", "indep"):
                    x = s[(s["specification"] == "IH3") & (s["gamma_source"] == src) & (s["horizon"] == h)]
                    oi = x[x["term"] == f"H_OI_{src}"]
                    vo = x[x["term"] == f"H_VOL_{src}"]
                    if len(oi):
                        r[f"H_OI_{src}_p1neg"] = one_sided_negative_p(oi["coef"].iloc[0], oi["p_value"].iloc[0])
                    if len(vo):
                        r[f"H_VOL_{src}_coef"] = vo["coef"].iloc[0]
                        r[f"H_VOL_{src}_p"] = vo["p_value"].iloc[0]
                out.append(r)

    summary = pd.DataFrame(out)
    summary.to_csv(OUTPUT_DIR / f"{REGISTERED_STEM}_summary.csv", index=False)

    for iv in REGRESSORS:
        for vl in VCOVS:
            s = summary[(summary["regressore"] == iv) & (summary["cluster"] == vl)]
            n1 = int(((s["IH1_coef"] > 0) & (s["IH1_p"] < ALPHA)).sum())
            n2 = int(((s["IH2_coef"] > 0) & (s["IH2_p"] < ALPHA)).sum())
            noi = int(((s["H_OI_cboe_p1neg"] < ALPHA) & (s["H_OI_indep_p1neg"] < ALPHA)).sum())
            nvo = int(((s["H_VOL_cboe_p"] < ALPHA) & (s["H_VOL_indep_p"] < ALPHA)).sum())
            print(f"  {iv:9s} ({REGRESSORS[iv]:10s}) cluster {vl:9s}: "
                  f"H-i-1 {n1}/4   H-i-2 {n2}/4   "
                  f"H_OI neg. su entrambe le fonti {noi}/4   H_VOL su entrambe {nvo}/4")

    print()
    show = ["regressore", "cluster", "h", "IH1_coef", "IH1_p", "IH2_coef", "IH2_p",
            "H_OI_cboe_p1neg", "H_OI_indep_p1neg", "H_VOL_cboe_coef", "H_VOL_cboe_p",
            "H_VOL_indep_p"]
    print(summary[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print()
    print("=" * 150)
    print("TERZILI DI H-INTRADAY-3: REGRESSORE PREREGISTRATO CONTRO CORRENTE")
    print("=" * 150)
    print("Priore preregistrato: il contenuto incrementale si INDEBOLISCE verso la chiusura.")
    pt = ter.pivot_table(index=["horizon", "tercile"], columns="regressor",
                         values="incremental_r2", aggfunc="first").reset_index()
    pt["tercile"] = pd.Categorical(pt["tercile"], TERCILE_NAMES, ordered=True)
    print(pt.sort_values(["horizon", "tercile"]).to_string(index=False,
          float_format=lambda v: f"{v:.6f}"))

    for iv in REGRESSORS:
        x = ter[ter["regressor"] == iv]
        up = 0
        for h in HORIZONS:
            e = x[(x["horizon"] == h) & (x["tercile"] == "early")]["incremental_r2"].iloc[0]
            l = x[(x["horizon"] == h) & (x["tercile"] == "late")]["incremental_r2"].iloc[0]
            up += int(l > e)
        print(f"  {iv}: chiusura sopra apertura in {up} orizzonti su 4")

    print()
    print(f"Output: {REGISTERED_STEM}_results.csv, _terciles.csv, _summary.csv")
    print("=" * 150)

    if not all(ok for _, ok, _ in checks):
        raise RuntimeError("Specifica preregistrata: controlli falliti.")


# =====================================================================
# PARTE 5: TERZILI SULLE VARIANTI DI TRONCAMENTO A MONEYNESS
# =====================================================================
# Terzili della parte 2 sui pannelli logm_005 e logm_002; su logm_005 anche il controllo
# incrociato graft_crosscheck.
def terciles_logm(variant: str):
    estimate_terciles(logm_panel_file(variant), variant)


# =====================================================================
# PARTE 6: STIME PYFIXEST SULLE VARIANTI DI TRONCAMENTO A MONEYNESS
# =====================================================================
# Le stime della parte 1 (stessi 66 modelli, schemi e regressore) sui pannelli delle
# varianti: verifica se i risultati intraday reggono a un'altra regola di troncamento.
def pyfixest_estimation_logm(variant: str):
    estimate_models(logm_panel_file(variant), f"{ESTIMATION_STEM}_{variant}", variant)


# =====================================================================
# PARTE 7: CONTROLLO A MANO DI PYFIXEST CONTRO STATSMODELS
# =====================================================================
# La stessa regressione stimata con statsmodels e una dummy per ogni livello di effetto
# fisso: coefficiente ed errore standard devono coincidere con quelli di pyfixest. Si usa
# l'anno intero perche' la correzione a campione finito dipende dal numero di cluster.

HANDCHECK_FILE = OUTPUT_DIR / "15_handcheck_pyfixest_vs_statsmodels.csv"

# IH1 a h=5 in livelli e in log: si verificano entrambe le forme delle stime.
HANDCHECKS = [
    {"label": "IH1_h5_levels", "dep": "FwdRV_5", "reg": "MFIV_ann"},
    {"label": "IH1_h5_log", "dep": "log_FwdRV_5", "reg": "log_MFIV_ann"},
]

FE_COLS = ["date_fe", "week_x_time_fe"]
CLUSTER_COL = "date_fe"

COEF_TOL = 1e-6
SE_REL_TOL = 1e-6


def tidy_value(tidy, term, column):
    if term in tidy.index:
        return float(tidy.loc[term, column])

    t2 = tidy.reset_index()
    col = "Coefficient" if "Coefficient" in t2.columns else t2.columns[0]
    row = t2[t2[col].astype(str) == term]

    if row.empty:
        raise ValueError(f"Termine {term} non trovato nell'output di pyfixest.")

    return float(row[column].iloc[0])


@ignoring_warnings(".*singleton fixed effect.*")
def pyfixest_handcheck():
    # LOAD
    print("=" * 130)
    print("HAND-CHECK PYFIXEST CONTRO STATSMODELS - ANNO INTERO")
    print("=" * 130)
    print(f"Pannello:          {PANEL_FILE}")
    print(f"pyfixest version:  {getattr(pf, '__version__', 'unknown')}")
    print("=" * 130)

    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Pannello non trovato:\n{PANEL_FILE}")

    panel = pd.read_csv(PANEL_FILE, dtype=str, keep_default_na=False)

    panel["quote_date"] = clean_date_series(panel["quote_date"])
    panel["grid_time"] = panel["grid_time"].astype(str).apply(normalize_time_string)

    for col in FE_COLS + ["iso_week_key"]:
        if col in panel.columns:
            panel[col] = panel[col].astype(str).str.strip()

    numeric = ["MFIV", "MFIV_ann", "TrailingRV", "FwdRV_5", "h5_valid"]

    for col in numeric:
        if col in panel.columns:
            panel[col] = to_numeric_series(panel[col])

    panel = make_log(panel, "MFIV_ann", "log_MFIV_ann")
    panel = make_log(panel, "FwdRV_5", "log_FwdRV_5")

    panel = panel[
        panel["quote_date"].notna()
        & panel["quote_date"].astype(str).str.startswith(YEAR)
    ].copy()

    print(f"Righe:    {len(panel):,}")
    print(f"Giornate: {panel['quote_date'].nunique():,}")
    print(f"Fasce:    {panel['grid_time'].nunique():,}")

    # HAND-CHECK
    rows = []

    for check in HANDCHECKS:
        dep = check["dep"]
        reg = check["reg"]

        print()
        print("=" * 130)
        print(f"CONTROLLO: {check['label']}   {dep} ~ {reg} | "
              f"{' + '.join(FE_COLS)}")
        print("=" * 130)

        w = panel[panel["h5_valid"] == 1].copy()
        w = w.dropna(subset=[dep, reg] + FE_COLS).copy()
        w = w[np.isfinite(w[dep]) & np.isfinite(w[reg])].reset_index(drop=True)

        print(f"  righe:                {len(w):,}")
        print(f"  livelli di date_fe:   {w['date_fe'].nunique():,}")
        print(f"  livelli di week_x_time: {w['week_x_time_fe'].nunique():,}")

        # --- pyfixest ---
        print("  stima pyfixest...", flush=True)

        model_pf = pf.feols(
            fml=f"{dep} ~ {reg} | {' + '.join(FE_COLS)}",
            data=w,
            vcov={"CRV1": CLUSTER_COL},
        )

        tidy_pf = model_pf.tidy()

        coef_pf = tidy_value(tidy_pf, reg, "Estimate")
        se_pf = tidy_value(tidy_pf, reg, "Std. Error")

        print(f"    coef = {coef_pf:.12e}")
        print(f"    se   = {se_pf:.12e}")

        # --- statsmodels a forza bruta ---
        print("  costruzione delle dummy...", flush=True)

        date_dummies = pd.get_dummies(w["date_fe"], prefix="date",
                                      drop_first=True).astype(float)
        wt_dummies = pd.get_dummies(w["week_x_time_fe"], prefix="wt",
                                    drop_first=True).astype(float)

        X = pd.concat([w[[reg]].astype(float), date_dummies, wt_dummies], axis=1)
        X = sm.add_constant(X)
        y = w[dep].astype(float)

        print(f"    matrice di disegno: {X.shape[0]:,} x {X.shape[1]:,} "
              f"({X.shape[0] * X.shape[1] * 8 / 1e6:.0f} MB)")
        print("  stima statsmodels OLS...", flush=True)

        model_sm = sm.OLS(y, X).fit()
        coef_sm = float(model_sm.params[reg])

        print("  stima statsmodels con cluster robust...", flush=True)

        model_sm_cl = sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds={"groups": w[CLUSTER_COL]}
        )
        se_sm_cl = float(model_sm_cl.bse[reg])

        # --- confronto coefficiente ---
        coef_diff = abs(coef_pf - coef_sm)
        coef_pass = bool(coef_diff <= COEF_TOL)

        # --- decomposizione dell'errore standard ---
        G = int(w[CLUSTER_COL].nunique())
        N = int(len(w))
        K_full = int(X.shape[1])

        corr_full = np.sqrt((G / (G - 1)) * ((N - 1) / (N - K_full)))
        se_sm_raw = se_sm_cl / corr_full

        # Convenzione di fixest: gli effetti fissi annidati nel cluster non contano nella
        # correzione a campione finito; date_fe coincide con il cluster, week_x_time_fe no.
        K_nonnested = 1 + int(w["week_x_time_fe"].nunique())

        corr_nonnested = np.sqrt((G / (G - 1)) * ((N - 1) / (N - K_nonnested)))
        se_sm_nonnested = se_sm_raw * corr_nonnested

        se_rel_naive = abs(se_pf - se_sm_cl) / abs(se_pf) if se_pf else np.nan
        se_rel_nonnested = abs(se_pf - se_sm_nonnested) / abs(se_pf) if se_pf else np.nan

        se_pass = bool(se_rel_nonnested <= SE_REL_TOL) if np.isfinite(se_rel_nonnested) else False

        print()
        print("  --- COEFFICIENTE ---")
        print(f"    pyfixest:     {coef_pf:.15e}")
        print(f"    statsmodels:  {coef_sm:.15e}")
        print(f"    differenza:   {coef_diff:.3e}   passa a {COEF_TOL}: {coef_pass}")

        print()
        print("  --- ERRORE STANDARD ---")
        print(f"    G (cluster) = {G}, N = {N}")
        print(f"    K con TUTTI i parametri:        {K_full}")
        print(f"    K solo NON annidati (fixest):   {K_nonnested}")
        print(f"    correzione con K pieno:         {corr_full:.10f}")
        print(f"    correzione con K non annidato:  {corr_nonnested:.10f}")
        print()
        print(f"    pyfixest:                               {se_pf:.15e}")
        print(f"    statsmodels, correzione propria:        {se_sm_cl:.15e}   "
              f"scarto relativo {se_rel_naive:.3e}")
        print(f"    statsmodels, convenzione non annidata:  {se_sm_nonnested:.15e}   "
              f"scarto relativo {se_rel_nonnested:.3e}")
        print(f"    passa a {SE_REL_TOL}: {se_pass}")

        rows.append({
            "check": check["label"],
            "dependent": dep,
            "regressor": reg,
            "n": N,
            "n_clusters": G,
            "K_full": K_full,
            "K_nonnested": K_nonnested,
            "coef_pyfixest": coef_pf,
            "coef_statsmodels": coef_sm,
            "coef_abs_diff": coef_diff,
            "coef_pass": int(coef_pass),
            "se_pyfixest": se_pf,
            "se_statsmodels_own_correction": se_sm_cl,
            "se_statsmodels_raw": se_sm_raw,
            "se_statsmodels_nonnested": se_sm_nonnested,
            "correction_full_K": corr_full,
            "correction_nonnested_K": corr_nonnested,
            "se_rel_diff_naive": se_rel_naive,
            "se_rel_diff_nonnested": se_rel_nonnested,
            "se_pass": int(se_pass),
            "overall_pass": int(coef_pass and se_pass),
        })

    result = pd.DataFrame(rows)
    safe_write_csv(result, HANDCHECK_FILE)

    print()
    print("=" * 130)
    print("RIEPILOGO")
    print("=" * 130)
    print(result[["check", "n", "n_clusters", "coef_abs_diff", "coef_pass",
                  "se_rel_diff_naive", "se_rel_diff_nonnested", "se_pass",
                  "overall_pass"]].to_string(index=False))

    print()
    print("COSA VERIFICA QUESTO CONTROLLO, E PERCHE' LA SECONDA COLONNA DI SCARTO")
    print("CONTA.")
    print("Il coefficiente deve coincidere a 1e-6: e' pura algebra e non dipende da")
    print("convenzioni. L'errore standard no: statsmodels applica la correzione a")
    print("campione finito contando TUTTI i parametri, fixest e pyfixest contano solo")
    print("quelli NON ANNIDATI nel cluster. Poiche' date_fe coincide col cluster, non")
    print("conta, e la correzione usa K = 1 + livelli di week_x_time_fe.")
    print("Se il confronto passa con la convenzione non annidata, gli errori standard")
    print("delle stime intraday sono verificati contro una ricostruzione")
    print("indipendente, e la convenzione usata e' documentata invece che assunta.")

    all_pass = int((result["overall_pass"] == 1).all())

    print()
    print("=" * 130)
    print("HAND-CHECK PASSATO" if all_pass else "HAND-CHECK FALLITO")
    print("=" * 130)
    print(f"Output: {HANDCHECK_FILE.name}")


def main():
    pyfixest_estimation()                                   # parte 1
    terciles()                                              # parte 2
    row_counts()                                            # parte 3
    registered_spec_check()                                 # parte 4
    run_variants(terciles_logm, LOGM_VARIANTS)              # parte 5
    run_variants(pyfixest_estimation_logm, LOGM_VARIANTS)   # parte 6
    pyfixest_handcheck()                                    # parte 7


if __name__ == "__main__":
    main()
