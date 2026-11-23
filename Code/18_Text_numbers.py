"""
Numeri derivati del testo (Sezioni 5.4-5.6, Appendice C, Tabelle D.1-D.2): intervalli, rapporti,
differenze, minimi e massimi ricostruiti dai risultati delle regressioni gia' salvati, senza
nuove stime, e confrontati con i valori stampati nella tesi. Un numero coincide se il valore
calcolato arrotonda alla cifra stampata; un DIVERSO non ferma lo script, lo fermano solo i
problemi di integrita' dei dati (file o righe mancanti, output che non coincidono fra loro).
Input:  CBOE/Data/13_daily_regressions/, 14_daily_mz_robustness/, 15_intraday_regressions/,
        16_intraday_robustness/ (elenco completo in INPUTS)
Output: CBOE/Data/18_text_numbers/
"""

import math
import re
import time

import numpy as np
import pandas as pd

from common import DATA_DIR, STRESS_DAY, output_dir, require


# =====================================================
# CONFIGURAZIONE
# =====================================================

OUTPUT_DIR = output_dir("18_text_numbers")
OUT_DETAIL = "18_text_derived_numbers.csv"
OUT_THESIS = "18_thesis_check.csv"
OUT_B1 = "18_b1_opening_control_without_april.csv"

# Chiavi dei file di input: d = giornaliero, i = intraday, e = effetti, b = controlli di
# ampiezza. Nei messaggi i file sono indicati con il numero della chiave (27 = d27).
INPUTS = {
    "d27": ("13_daily_regressions", "13_daily_regression_results_2025.csv"),
    "d28_005": ("13_daily_regressions", "13_daily_regression_results_2025_logm_005.csv"),
    "d28_002": ("13_daily_regressions", "13_daily_regression_results_2025_logm_002.csv"),
    "d30": ("14_daily_mz_robustness", "14_regime_and_rv5_robustness_results.csv"),
    "d31": ("14_daily_mz_robustness", "14_bshort_rv5_diagnostic.csv"),
    "d33": ("14_daily_mz_robustness", "14_gamma_sampling_frequency_results.csv"),
    "i34": ("15_intraday_regressions", "15_pyfixest_estimation_2025_full_247_results.csv"),
    "i37": ("15_intraday_regressions", "15_registered_spec_check_results.csv"),
    "i39_005": ("15_intraday_regressions", "15_pyfixest_estimation_2025_full_247_logm_005_results.csv"),
    "i39_002": ("15_intraday_regressions", "15_pyfixest_estimation_2025_full_247_logm_002_results.csv"),
    "i40_lev": ("16_intraday_robustness", "16_day_concentration_levels.csv"),
    "i41_sum": ("16_intraday_robustness", "16_cluster_exclusion_summary.csv"),
    "i41_diag": ("16_intraday_robustness", "16_cluster_exclusion_diagnostic.csv"),
    "i42": ("16_intraday_robustness", "16_wild_cluster_bootstrap_week_diagnostics.csv"),
    "i43": ("15_intraday_regressions", "15_handcheck_pyfixest_vs_statsmodels.csv"),
    "i44": ("16_intraday_robustness", "16_auxiliary_regression_vif_2025_primary_247.csv"),
    "e45": ("16_intraday_robustness", "16_gamma_effect_size_ci.csv"),
    "d46": ("13_daily_regressions", "13_a4_without_april.csv"),
    "b47_i": ("16_intraday_robustness", "16_hvol_breadth_controls_intraday.csv"),
    "b47_d": ("16_intraday_robustness", "16_hvol_breadth_controls_daily.csv"),
    "b47_v": ("16_intraday_robustness", "16_hvol_breadth_controls_verdicts.txt"),
}

HORIZONS = [5, 15, 30, 60]
ALPHA = 0.05
DAILY_REGRESSORS = ["MFIV_t", "MFIV_ann_t", "MFIV_session_t"]
SOURCES_46 = ["vendor", "black76"]          # nomi delle fonti in d46 e b47
SOURCES_45 = ["cboe", "indep"]              # nomi delle fonti in i34, i37, i40 ed e45
GAMMA_K = [1, 5, 15, 30]                    # griglie del gamma in d33 (minuti)
REGIMES = ["calmo", "intermedio", "agitato"]
APRIL_PEAK_DAY = STRESS_DAY                 # la settimana di questa data e' quella del bootstrap
EXPLORATORY_DAY = "2025-04-04"              # sessione tolta nell'esplorativo su H_OI
DEGENERATE_MAX_ABS_T = 2.5                  # soglia di |t*| massimo per le celle degeneri del bootstrap
CROSS_RTOL = 1e-6                           # controllo incrociato fra output (pyfixest: rumore fino a 1.5e-7)
SPOT_RV_COEF_BASE = 1.894                   # coefficiente della voce INFO (b); non usato nei calcoli


# =====================================================================
# NUMERI DERIVATI DEL TESTO E CONFRONTO CON LA TESI
# =====================================================================

# -----------------------------------------------------
# FUNZIONI DI SUPPORTO
# -----------------------------------------------------

def path(key):
    pair = INPUTS[key]
    return DATA_DIR / pair[0] / pair[1]


def label(key):
    pair = INPUTS[key]
    return f"{pair[0]}/{pair[1]}"


_CACHE = {}


def read(key):
    if key not in _CACHE:
        p = path(key)
        require(p)
        _CACHE[key] = pd.read_csv(p)
    return _CACHE[key]


def one(df: pd.DataFrame, what: str, **kw) -> pd.Series:
    """Esattamente una riga che soddisfa i filtri, altrimenti errore di integrita'."""
    m = pd.Series(True, index=df.index)
    for col, val in kw.items():
        if col not in df.columns:
            raise KeyError(f"[{what}] colonna mancante: {col}")
        m &= (df[col] == val)
    sub = df[m]
    if len(sub) != 1:
        raise RuntimeError(f"[{what}] attesa una riga con {kw}, trovate {len(sub)}")
    return sub.iloc[0]


def val(row: pd.Series, col: str, what: str) -> float:
    x = float(row[col])
    if not np.isfinite(x):
        raise RuntimeError(f"[{what}] valore non finito in {col}")
    return x


def printed_match(value, printed):
    """True se value arrotonda alla stringa stampata (mezza unita' dell'ultima cifra),
    come in 13_Daily_regressions.py."""
    t = float(printed)
    mant = printed.lower().lstrip("+-").split("e")[0].replace(".", "").lstrip("0")
    sig = max(len(mant), 1)
    if t == 0:
        return abs(value) < 10 ** (-sig)
    half = 0.5 * 10 ** (math.floor(math.log10(abs(t))) - (sig - 1))
    return abs(value - t) <= half * 1.000001


def half_unit(printed):
    t = float(printed)
    mant = printed.lower().lstrip("+-").split("e")[0].replace(".", "").lstrip("0")
    sig = max(len(mant), 1)
    return 0.5 * 10 ** (math.floor(math.log10(abs(t))) - (sig - 1))


def nw_lag(n):
    """Ritardo di Newey-West floor(4 (T/100)^(2/9)), come in 13_Daily_regressions.py."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def g(x, d=4):
    return f"{x:.{d}g}"


def same(a, b, rtol=CROSS_RTOL):
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300)


def cross(what, a, b, rtol=CROSS_RTOL):
    """Controllo di integrita': due output devono contenere lo stesso numero."""
    if not same(a, b, rtol):
        raise RuntimeError(f"[integrita'] {what}: {a!r} contro {b!r} (rtol {rtol:g})")


DETAIL = []


def det(gruppo, chiave, valore, fonte, colonna):
    DETAIL.append({"gruppo": gruppo, "chiave": chiave, "valore": float(valore),
                   "file_fonte": fonte, "colonna_fonte": colonna})


CHECKS = []


def add(pagina, voce, valore_tesi, valore_calcolato, precisione, ok, fonte, colonna,
        nota="", info=False):
    CHECKS.append({
        "pagina": pagina, "voce": voce, "valore_tesi": valore_tesi,
        "valore_calcolato": valore_calcolato, "precisione": precisione,
        "esito": "INFO" if info else ("OK" if ok else "DIVERSO"),
        "nota": nota, "file_fonte": fonte, "colonna_fonte": colonna})


def chk_value(pagina, voce, tesi, calc, fonte, colonna, nota="", d=4, shown=None):
    add(pagina, voce, shown or tesi, g(calc, d), "arrotondamento alla cifra stampata",
        printed_match(calc, tesi), fonte, colonna, nota)


def chk_list(pagina, voce, tesi_list, calc_list, fonte, colonna, nota="", d=4):
    ok = len(tesi_list) == len(calc_list) and all(
        printed_match(c, t) for t, c in zip(tesi_list, calc_list))
    add(pagina, voce, ", ".join(tesi_list), ", ".join(g(c, d) for c in calc_list),
        "ogni valore alla cifra stampata", ok, fonte, colonna, nota)


def chk_range(pagina, voce, lo, hi, values, fonte, colonna, nota="", d=4, unit=""):
    vmin, vmax = float(np.min(values)), float(np.max(values))
    ok = printed_match(vmin, lo) and printed_match(vmax, hi)
    add(pagina, voce, f"{lo}{unit} - {hi}{unit}", f"{g(vmin, d)}{unit} - {g(vmax, d)}{unit}",
        "estremi (min, max) alla cifra stampata", ok, fonte, colonna, nota)


def chk_bound(pagina, voce, op, bound, values, fonte, colonna, nota="", d=4, rounded=False):
    """op in {'<=', '<', '>', '>='}. rounded=True: "al massimo circa", il valore estremo
    deve arrotondare a non piu' (non meno) del limite stampato."""
    b = float(bound)
    tol = half_unit(bound) if rounded else 0.0
    if op in ("<=", "<"):
        ext = float(np.max(values))
        ok = (ext <= b + tol) if (op == "<=" or rounded) else (ext < b)
        ext_txt = f"max {g(ext, d)}"
    else:
        ext = float(np.min(values))
        ok = (ext >= b - tol) if (op == ">=" or rounded) else (ext > b)
        ext_txt = f"min {g(ext, d)}"
    prec = ("limite, valore estremo arrotondato alla cifra stampata" if rounded
            else "limite sul valore non arrotondato")
    add(pagina, voce, f"{op} {bound}" if not rounded else f"{op} ~{bound}", ext_txt, prec,
        ok, fonte, colonna, nota)


def chk_approx(pagina, voce, tesi, values, tol, fonte, colonna, nota="", d=4, unit=""):
    t = float(tesi)
    ok = all(abs(v - t) <= tol for v in values)
    add(pagina, voce, f"circa {tesi}{unit}", ", ".join(f"{g(v, d)}{unit}" for v in values),
        f"circa: ogni valore entro +/-{tol:g} da {tesi}", ok, fonte, colonna, nota)


def chk_cond(pagina, voce, tesi_text, ok, calc_text, fonte, colonna, nota=""):
    add(pagina, voce, tesi_text, calc_text, "condizione", bool(ok), fonte, colonna, nota)


def text_derived_numbers():
    """Ricostruisce i numeri derivati, li salva e li confronta con la tesi."""
    # DETAIL e CHECKS sono a livello di modulo: si svuotano qui, cosi' una seconda chiamata
    # nello stesso processo non accoda righe doppie.
    DETAIL.clear()
    CHECKS.clear()
    t_start = time.time()
    print("=" * 140)
    print("18: NUMERI DERIVATI DEL TESTO (SEZ. 5.4-5.6, APP. C, TAB. D.1-D.2)")
    print("=" * 140)
    print(f"Cartella dati: {DATA_DIR}")

    # =====================================================
    # LETTURA E CONTROLLI DI INTEGRITA'
    # =====================================================

    for key in INPUTS:
        require(path(key))

    d27, d46, e45 = read("d27"), read("d46"), read("e45")
    b47i, b47d = read("b47_i"), read("b47_d")
    i34, i37 = read("i34"), read("i37")

    def c46(regressor, form, source, sample, term):
        return one(d46, "46", regressor=regressor, form=form, gamma_source=source,
                   sample=sample, term=term)

    def c45_intra(regressor, form, source, horizon, term):
        return one(e45, "45", design="intraday", form=form, regressor=regressor,
                   source=source, horizon=float(horizon), term=term)

    def c47i(spec, horizon, source, controls, term="H_VOL"):
        return one(b47i, "47 intraday", spec=spec, horizon=horizon, gamma_source=source,
                   controls=controls, term=term)

    def c47d(spec, form, source, controls, term="H_VOL"):
        return one(b47d, "47 daily", spec=spec, form=form, gamma_source=source,
                   controls=controls, term=term)

    # d27 <-> d46: A4 sul campione intero, vendor.
    SPEC27 = {"MFIV_t": "A6", "MFIV_ann_t": "A6_ann", "MFIV_session_t": "A6_ses"}
    for reg, spec in SPEC27.items():
        for form in ["levels", "log"]:
            r27 = one(d27, "27", specification=spec, form=form, gamma_source="CBOE", term="H_VOL_t")
            r46 = c46(reg, form, "vendor", "full", "H_VOL")
            cross(f"27 {spec} {form} H_VOL coef vs 46", val(r27, "coef", "27"), val(r46, "coef", "46"))
            cross(f"27 {spec} {form} H_VOL p vs 46", val(r27, "p_value", "27"), val(r46, "p_value", "46"))
    # b47 (base) <-> d46: A4 sul campione intero, entrambe le fonti.
    for reg in DAILY_REGRESSORS:
        for form in ["levels", "log"]:
            for src in SOURCES_46:
                cross(f"47 daily base {reg} {form} {src} vs 46",
                      val(c47d(reg, form, src, "base"), "coef", "47"),
                      val(c46(reg, form, src, "full", "H_VOL"), "coef", "46"))
    # e45 <-> d46: effetto per deviazione standard di H_VOL in A4.
    for form in ["levels", "log"]:
        for s45, s46 in zip(["CBOE", "indep"], SOURCES_46):
            r45 = one(e45, "45 daily", design="giornaliero", form=form, source=s45, term="A6:H_VOL")
            cross(f"45 vs 46 effetto A4 {form} {s45}", val(r45, "effect_pct", "45"),
                  val(c46("MFIV_t", form, s46, "full", "H_VOL"), "effect_pct_per_sd", "46"))
    # b47 (base, I3) <-> i34 ed e45: coefficienti di H_VOL intraday.
    for h in HORIZONS:
        for s45, s46 in zip(SOURCES_45, SOURCES_46):
            term34 = f"H_VOL_{s45}"
            for form, spec47 in [("levels", "levels_MFIV_ann"), ("log", "log_MFIV_ann")]:
                r34 = one(i34, "34", specification="IH3", horizon=h, form=form, gamma_source=s45,
                          fe_scheme="date_x_weektime", term=term34)
                cross(f"34 vs 47 {spec47} h{h} {s45}", val(r34, "coef", "34"),
                      val(c47i(spec47, h, s46, "base"), "coef", "47"))
            cross(f"45 vs 47 levels_MFIV h{h} {s45}",
                  val(c45_intra("MFIV", "levels", s45, h, "H_VOL"), "coef", "45"),
                  val(c47i("levels_MFIV", h, s46, "base"), "coef", "47"))
    print("Controlli incrociati fra gli output 27, 34, 45, 46, 47: superati.")

    # =====================================================
    # TAB. 5.1 (p. 38) E SEZ. 5.4 GIORNALIERO (pp. 46-48)
    # =====================================================

    F45, F46, F27 = label("e45"), label("d46"), label("d27")

    eff_lev = val(one(e45, "45", design="giornaliero", form="levels", source="CBOE", term="A6:H_VOL"),
                  "effect_pct", "45")
    eff_log = val(one(e45, "45", design="giornaliero", form="log", source="CBOE", term="A6:H_VOL"),
                  "effect_pct", "45")
    eff_noapr = val(c46("MFIV_t", "levels", "vendor", "no_april", "H_VOL"), "effect_pct_per_sd", "46")
    chk_value(38, "Tab. 5.1, A4 H_VOL livelli, effetto per sd (vendor)", "39.1", eff_lev, F45,
              "effect_pct (giornaliero, levels, CBOE, A6:H_VOL)", shown="+39.1%")
    chk_value(38, "Tab. 5.1, A4 H_VOL livelli senza aprile, effetto per sd (vendor)", "-17.3", eff_noapr,
              F46, "effect_pct_per_sd (MFIV_t, levels, vendor, no_april)", shown="-17.3%")
    chk_value(38, "Tab. 5.1, A4 H_VOL log, effetto per sd (vendor)", "-22.7", eff_log, F45,
              "effect_pct (giornaliero, log, CBOE, A6:H_VOL)", shown="-22.7%")
    chk_value(47, "A4 log con MFIV_d: H_VOL, riduzione per sd", "22.7", -eff_log, F45,
              "effect_pct (giornaliero, log, CBOE, A6:H_VOL)", shown="22.7% in meno")

    # p sul campione intero: vendor e Black '76.
    p_full = {src: [val(c46(reg, "log", src, "full", "H_VOL"), "p_value", "46") for reg in DAILY_REGRESSORS]
              for src in SOURCES_46}
    for src in SOURCES_46:
        for reg, p in zip(DAILY_REGRESSORS, p_full[src]):
            det("A4 log H_VOL p, campione intero", f"{reg} {src}", p, F46, "p_value")
    chk_list(47, "A4 log H_VOL, p con MFIV_d, MFIV^ann_d, MFIV^ses_d (vendor)",
             ["0.003", "0.027", "0.057"], p_full["vendor"], F46, "p_value (log, full, H_VOL)")
    chk_list(47, "A4 log H_VOL, p con MFIV_d, MFIV^ann_d, MFIV^ses_d (Black '76)",
             ["0.003", "0.028", "0.060"], p_full["black76"], F46, "p_value (log, full, H_VOL)")

    coef_full_log = {(reg, src): val(c46(reg, "log", src, "full", "H_VOL"), "coef", "46")
                     for reg in DAILY_REGRESSORS for src in SOURCES_46}
    coef_noapr_log = {(reg, src): val(c46(reg, "log", src, "no_april", "H_VOL"), "coef", "46")
                      for reg in DAILY_REGRESSORS for src in SOURCES_46}
    p_noapr_log = [val(c46(reg, "log", src, "no_april", "H_VOL"), "p_value", "46")
                   for reg in DAILY_REGRESSORS for src in SOURCES_46]
    bigger = all(abs(coef_noapr_log[k]) > abs(coef_full_log[k]) for k in coef_full_log)
    chk_cond(47, "A4 log senza aprile: H_VOL piu' grande in valore assoluto sotto i tre regressori, due fonti",
             "piu' grande in valore assoluto",
             bigger, "; ".join(f"{r} {s}: {g(coef_full_log[(r, s)])} -> {g(coef_noapr_log[(r, s)])}"
                               for r, s in coef_full_log), F46, "coef (log, full vs no_april)")
    chk_bound(47, "A4 log senza aprile: p di H_VOL, tre regressori, due fonti", "<", "0.001",
              p_noapr_log, F46, "p_value (log, no_april, H_VOL)")

    # Senza aprile in livelli.
    lev_noapr = {(reg, src): c46(reg, "levels", src, "no_april", "H_VOL")
                 for reg in DAILY_REGRESSORS for src in SOURCES_46}
    neg_sig = all(val(r, "coef", "46") < 0 and val(r, "p_value", "46") < ALPHA for r in lev_noapr.values())
    chk_cond(46, "Senza aprile H_VOL_d in livelli negativo e significativo con i tre regressori",
             "negativo e significativo", neg_sig,
             "; ".join(f"{r} {s}: {g(val(x, 'coef', '46'), 3)} (p {g(val(x, 'p_value', '46'), 2)})"
                       for (r, s), x in lev_noapr.items()), F46, "coef, p_value (levels, no_april)")
    p_lev_noapr = [val(r, "p_value", "46") for r in lev_noapr.values()]
    chk_bound(48, "A4 livelli senza aprile: p di H_VOL, tre regressori, due fonti", "<=", "0.004",
              p_lev_noapr, F46, "p_value (levels, no_april, H_VOL)")
    eff_noapr_v = [-val(lev_noapr[(reg, "vendor")], "effect_pct_per_sd", "46") for reg in DAILY_REGRESSORS]
    for reg, e in zip(DAILY_REGRESSORS, eff_noapr_v):
        det("A4 livelli senza aprile, riduzione per sd (vendor)", reg, e, F46, "effect_pct_per_sd")
    chk_range(48, "A4 livelli senza aprile: riduzione di RV_d per sd (vendor)", "14.5", "17.3",
              eff_noapr_v, F46, "effect_pct_per_sd (levels, no_april, vendor)", unit="%",
              nota="Black '76: " + ", ".join(
                  g(-val(lev_noapr[(r, 'black76')], 'effect_pct_per_sd', '46')) for r in DAILY_REGRESSORS))

    # Regimi (d30).
    d30 = read("d30")
    F30 = label("d30")
    regime_cells = [("A6", "CBOE", "H_VOL_t"), ("A6", "indep", "H_VOL_t_indep"),
                    ("A6_ann", "CBOE", "H_VOL_t"), ("A6_ses", "CBOE", "H_VOL_t")]
    reg_signs, txt = True, []
    for spec, src, term in regime_cells:
        signs = {}
        for smp in REGIMES:
            r = one(d30, "30", specification=spec, form="levels", gamma_source=src, term=term, sample=smp)
            signs[smp] = val(r, "coef", "30")
            det("A4 livelli H_VOL per regime", f"{spec} {src} {smp}", signs[smp], F30, "coef")
        reg_signs &= signs["calmo"] < 0 and signs["intermedio"] < 0 and signs["agitato"] > 0
        txt.append(f"{spec} {src}: " + "/".join(f"{signs[s]:+.2e}" for s in REGIMES))
    chk_cond(48, "Regime agitato: l'unico con associazione giornaliera positiva in livelli (H_VOL_d)",
             "positivo solo nel regime agitato", reg_signs,
             "calmo/intermedio/agitato " + "; ".join(txt), F30, "coef (levels, A6*, H_VOL)")
    chk_cond(51, "H_VOL_d in livelli negativo nel regime calmo, positivo in quello agitato",
             "negativo calmo, positivo agitato", reg_signs,
             "; ".join(txt), F30, "coef (levels, A6*, H_VOL)")
    log_rows = d30[(d30.form == "log") & d30.specification.isin(["A6", "A6_ann", "A6_ses"])
                   & d30.term.isin(["H_VOL_t", "H_VOL_t_indep"])]
    if len(log_rows) != 20:
        raise RuntimeError(f"[30] attese 20 righe log di H_VOL (4 celle x 5 campioni), trovate {len(log_rows)}")
    chk_cond(51, "H_VOL_d in log negativo in tutti i campioni provati, non sempre significativo",
             "negativo ovunque, non sempre significativo",
             bool((log_rows.coef < 0).all() and (log_rows.p_value >= ALPHA).any()),
             f"{len(log_rows)} celle, max coef {g(log_rows.coef.max())}, "
             f"p >= 0.05 in {int((log_rows.p_value >= ALPHA).sum())}", F30, "coef, p_value (log)")

    # Troncamento (d27, d28) e griglie del gamma (d33).
    d28_005, d28_002, d33 = read("d28_005"), read("d28_002"), read("d33")
    F28 = f"{label('d28_005')} ; {label('d28_002')}"
    trunc = {}
    for src, term in [("CBOE", "H_VOL_t"), ("indep", "H_VOL_t_indep")]:
        trunc[src] = [val(one(df, "27/28", specification="A6", form="log", gamma_source=src, term=term),
                          "coef", "27/28") for df in (d27, d28_005, d28_002)]
    trunc_p = [val(one(df, "27/28", specification="A6", form="log", gamma_source="CBOE", term="H_VOL_t"),
                    "p_value", "27/28") for df in (d27, d28_005, d28_002)]
    grid = {}
    for src, base in [("CBOE", "H_VOL_t"), ("indep", "H_VOL_t_indep")]:
        rows = [one(d33, "33", specification="A6", form="log", gamma_source=src,
                    term=f"{base}_k{k}", k_minutes=k) for k in GAMMA_K]
        grid[src] = ([val(r, "coef", "33") for r in rows], [val(r, "p_value", "33") for r in rows])
        for k, (c, p) in zip(GAMMA_K, zip(*grid[src])):
            det("A4 log H_VOL per griglia del gamma", f"{src} k={k}", c, label("d33"), "coef")
    falls = {s: all(abs(a) > abs(b) for a, b in zip(grid[s][0][:-1], grid[s][0][1:])) for s in grid}
    weak_trunc = all(abs(trunc[s][0]) > abs(trunc[s][1]) > abs(trunc[s][2]) for s in trunc)
    chk_cond(48, "A4 log: il coefficiente di H_VOL varia con il troncamento e si indebolisce sulle griglie piu' rade, due fonti",
             "varia con il troncamento; cala con la griglia", weak_trunc and all(falls.values()),
             "troncamento bid/0.05/0.02 " + "; ".join(f"{s} " + "/".join(g(x) for x in trunc[s]) for s in trunc)
             + " | griglia 1/5/15/30 " + "; ".join(f"{s} " + "/".join(g(x) for x in grid[s][0]) for s in grid),
             f"{F27} ; {F28} ; {label('d33')}", "coef (A6, log, H_VOL)")

    # =====================================================
    # SEZ. 5.4 INTRADAY (pp. 46-49), TAB. 5.2 (p. 39)
    # =====================================================

    lev_tot = {(s, h): c45_intra("MFIV", "levels", s, h, "H_VOL") for s in SOURCES_45 for h in HORIZONS}
    eff_tot = [-val(r, "effect_pct", "45") for r in lev_tot.values()]
    for (s, h), r in lev_tot.items():
        det("I3 livelli MFIV totale, H_VOL, riduzione per sd", f"{s} h={h}", -val(r, "effect_pct", "45"),
            F45, "effect_pct")
    for pag, voce in [(39, "Tab. 5.2, H_VOL I3 livelli con MFIV totale: riduzione per sd (due fonti)"),
                      (47, "I3 livelli con MFIV(t) totale: riduzione della varianza per sd, due fonti")]:
        chk_range(pag, voce, "9", "13", eff_tot, F45, "effect_pct (intraday, levels, MFIV, H_VOL)", unit="%")
    chk_cond(47, "I3 livelli con MFIV totale: H_VOL significativo a ogni orizzonte, due fonti",
             "significativo", all(val(r, "p_value", "45") < ALPHA for r in lev_tot.values()),
             "max p " + g(max(val(r, "p_value", "45") for r in lev_tot.values())), F45, "p_value")
    lev_ann = [c45_intra("MFIV_ann", "levels", s, h, "H_VOL") for s in SOURCES_45 for h in HORIZONS]
    for pag, voce in [(39, "Tab. 5.2, H_VOL I3 livelli con MFIV annualizzata: non significativo"),
                      (47, "I3 livelli con MFIV^ann(t): nessun orizzonte significativo")]:
        chk_cond(pag, voce, "non significativo", all(val(r, "p_value", "45") >= ALPHA for r in lev_ann),
                 "min p " + g(min(val(r, "p_value", "45") for r in lev_ann)), F45, "p_value")

    hoi_log = [c45_intra("MFIV_ann", "log", s, h, "H_OI") for s in SOURCES_45 for h in HORIZONS]
    lo_env = min(val(r, "ci_low_pct", "45") for r in hoi_log)
    hi_env = max(val(r, "ci_high_pct", "45") for r in hoi_log)
    det("I3 log H_OI, inviluppo IC 95%", "minimo ci_low", lo_env, F45, "ci_low_pct")
    det("I3 log H_OI, inviluppo IC 95%", "massimo ci_high", hi_env, F45, "ci_high_pct")
    add(47, "I3 log: inviluppo degli IC al 95% di H_OI per sd, due fonti", "-4.5% - +2.6%",
        f"{g(lo_env)}% - {g(hi_env)}%", "estremi (min, max) alla cifra stampata",
        printed_match(lo_env, "-4.5") and printed_match(hi_env, "2.6"), F45,
        "ci_low_pct, ci_high_pct (intraday, log, MFIV_ann, H_OI)")
    sig_v = [h for h in HORIZONS if val(c45_intra("MFIV_ann", "log", "cboe", h, "H_OI"), "p_value", "45") < ALPHA]
    sig_b = [h for h in HORIZONS if val(c45_intra("MFIV_ann", "log", "indep", h, "H_OI"), "p_value", "45") < ALPHA]
    chk_cond(47, "I3 log H_OI: una sola cella significativa con il gamma vendor, non confermata da Black '76",
             "una cella vendor isolata, Black '76 no", len(sig_v) == 1 and not set(sig_v) & set(sig_b),
             f"vendor significativo a h = {sig_v}, Black '76 a h = {sig_b}", F45, "p_value")
    hvol_log = [c45_intra("MFIV_ann", "log", s, h, "H_VOL") for s in SOURCES_45 for h in HORIZONS]
    eff_hvol_log = [-val(r, "effect_pct", "45") for r in hvol_log]
    for r, e in zip(hvol_log, eff_hvol_log):
        det("I3 log H_VOL, riduzione per sd", f"{r['source']} h={int(r['horizon'])}", e, F45, "effect_pct")
    chk_cond(47, "I3 log H_VOL: negativo e significativo a ogni orizzonte, due fonti", "negativo, significativo",
             all(val(r, "coef", "45") < 0 and val(r, "p_value", "45") < ALPHA for r in hvol_log),
             "max p " + g(max(val(r, "p_value", "45") for r in hvol_log)), F45, "coef, p_value")
    for pag, voce in [(39, "Tab. 5.2, H_VOL I3 log: riduzione per sd"),
                      (47, "I3 log H_VOL: riduzione della varianza per sd, due fonti")]:
        chk_range(pag, voce, "1.7", "3.0", eff_hvol_log, F45, "effect_pct (intraday, log, H_VOL)", unit="%")

    # R2 within di I3 meno I2 (i37), osservazioni.
    F37 = label("i37")
    r37 = i37[i37.vcov == "giorno"]
    diffs = {"levels": [], "log": []}
    nobs_ok, nobs_txt = True, []
    for reg in ["MFIV", "MFIV_ann"]:
        for form in ["levels", "log"]:
            for h in HORIZONS:
                s2 = r37[(r37.regressor == reg) & (r37.specification == "IH2") & (r37.form == form)
                         & (r37.horizon == h)]
                if s2.r2_within.nunique() != 1 or s2.n.nunique() != 1:
                    raise RuntimeError(f"[37] I2 {reg} {form} h{h}: R2 o n non unici")
                for src in SOURCES_45:
                    s3 = r37[(r37.regressor == reg) & (r37.specification == "IH3") & (r37.form == form)
                             & (r37.horizon == h) & (r37.gamma_source == src)]
                    if s3.r2_within.nunique() != 1 or s3.n.nunique() != 1:
                        raise RuntimeError(f"[37] I3 {reg} {form} h{h} {src}: R2 o n non unici")
                    dr = float(s3.r2_within.iloc[0]) - float(s2.r2_within.iloc[0])
                    diffs[form].append(dr)
                    det("R2 within I3 - I2", f"{reg} {form} h={h} {src}", dr, F37, "r2_within")
                    dn = int(s2.n.iloc[0]) - int(s3.n.iloc[0])
                    nobs_ok &= dn == 4
                    nobs_txt.append(dn)
    chk_bound(48, "R2 within di I3 meno I2, livelli, MFIV totale e annualizzata, due fonti", "<=", "0.002",
              diffs["levels"], F37, "r2_within (IH3 - IH2, vcov giorno)", rounded=True)
    chk_bound(48, "R2 within di I3 meno I2, log, due fonti", "<=", "0.004",
              diffs["log"], F37, "r2_within (IH3 - IH2, vcov giorno)", rounded=True)
    chk_cond(48, "I3 ha quattro osservazioni in meno di I2 (ogni orizzonte)", "4",
             nobs_ok, f"differenze trovate: {sorted(set(nobs_txt))}", F37, "n")

    # Aumento di R2 da A2 ad A4 (d27).
    for form, tesi in [("levels", "0.010"), ("log", "0.014")]:
        a2 = val(one(d27, "27", specification="A2", form=form, term="RV_lag1" if form == "levels" else "log_RV_lag1"),
                 "r_squared", "27")
        a4v = val(one(d27, "27", specification="A6", form=form, gamma_source="CBOE", term="H_VOL_t"), "r_squared", "27")
        a4b = val(one(d27, "27", specification="A6", form=form, gamma_source="indep", term="H_VOL_t_indep"),
                  "r_squared", "27")
        det("R2 A4 - A2", f"{form} vendor", a4v - a2, F27, "r_squared")
        det("R2 A4 - A2", f"{form} Black '76", a4b - a2, F27, "r_squared")
        chk_value(48, f"Aumento di R2 aggiungendo gli indici ad A2, {form} (vendor)", tesi, a4v - a2, F27,
                  "r_squared (A6 - A2)", nota=f"Black '76: {g(a4b - a2)}")

    # =====================================================
    # CONTROLLI DI AMPIEZZA E QUANTITA' (b47): pp. 48-49, 73, 74
    # =====================================================

    F47I, F47D, F47V = label("b47_i"), label("b47_d"), label("b47_v")

    # Quota della variazione di H_VOL spiegata dai controlli: riga DIAGNOSTIC del file verdicts.
    text = path("b47_v").read_text()
    share = {}
    for src in SOURCES_46:
        m = re.search(rf"^\s*{src}\s+logN\s+([0-9.]+)\s*\|\s*logG\s+([0-9.]+)\s*\|\s*logN\+logG\s+([0-9.]+)",
                      text, flags=re.M)
        if not m:
            raise RuntimeError(f"[47 verdicts] riga DIAGNOSTIC per {src} non trovata")
        share[src] = 100.0 * float(m.group(3))
        det("Quota within di H_VOL spiegata da logN + logG (h=5)", src, share[src], F47V, "DIAGNOSTIC logN+logG")
    chk_approx(48, "Entro gli effetti fissi i due controlli spiegano circa il 15% della variazione di H_VOL",
               "15", [share[s] for s in SOURCES_46], 1.0, F47V, "DIAGNOSTIC logN+logG (vendor, black76)", unit="%",
               nota="solo 3 decimali nel file (0.154, 0.145)")

    def ratios(spec, controls, hs=HORIZONS, sources=SOURCES_46):
        out = []
        for src in sources:
            for h in hs:
                base = val(c47i(spec, h, src, "base"), "coef", "47")
                alt = c47i(spec, h, src, controls)
                out.append((src, h, val(alt, "coef", "47") / base, val(alt, "p_value", "47")))
                det(f"Quota del coefficiente di H_VOL ({controls}/base)", f"{spec} {src} h={h}",
                    out[-1][2], F47I, "coef")
        return out

    log_ng = ratios("log_MFIV_ann", "NG")
    log_n = ratios("log_MFIV_ann", "N")
    log_g = ratios("log_MFIV_ann", "G")
    lev_ng = ratios("levels_MFIV", "NG")
    lev_g = ratios("levels_MFIV", "G")

    for pag in [49, 73]:
        chk_range(pag, "Log, entrambi i controlli: quota del coefficiente di H_VOL, due fonti", "89", "95",
                  [100 * r for _, _, r, _ in log_ng], F47I, "coef NG / coef base (log_MFIV_ann)", unit="%")
    chk_range(49, "Log, entrambi i controlli: p, due fonti", "0.006", "0.047", [p for *_, p in log_ng],
              F47I, "p_value NG (log_MFIV_ann)")
    chk_bound(73, "Log, entrambi i controlli: significativo a ogni orizzonte, due fonti", "<=", "0.047",
              [p for *_, p in log_ng], F47I, "p_value NG (log_MFIV_ann)")
    chk_range(49, "Log, solo numero di strike: quota del coefficiente, due fonti", "75", "87",
              [100 * r for _, _, r, _ in log_n], F47I, "coef N / coef base (log_MFIV_ann)", unit="%")
    chk_range(49, "Log, solo numero di strike: p a 5 e 60 minuti, due fonti", "0.058", "0.089",
              [p for _, h, _, p in log_n if h in (5, 60)], F47I, "p_value N (log_MFIV_ann)")
    chk_cond(49, "Log, solo numero di strike: a 15 e 30 minuti resta significativo (il p sale solo a 5 e 60)",
             "p < 0.05 a 15 e 30", all(p < ALPHA for _, h, _, p in log_n if h in (15, 30)),
             ", ".join(f"{s} h{h} {g(p, 3)}" for s, h, _, p in log_n if h in (15, 30)), F47I, "p_value N")
    lev1530 = [x for x in lev_ng if x[1] in (15, 30)]
    for pag in [49, 73]:
        chk_range(pag, "Livelli MFIV totale, entrambi i controlli: quota a 15 e 30 minuti, due fonti", "11", "24",
                  [100 * r for _, _, r, _ in lev1530], F47I, "coef NG / coef base (levels_MFIV)", unit="%")
    chk_cond(49, "Livelli MFIV totale, entrambi i controlli: non piu' significativo a 15 e 30 minuti",
             "non significativo", all(p >= ALPHA for *_, p in lev1530),
             "min p " + g(min(p for *_, p in lev1530), 3), F47I, "p_value NG (levels_MFIV)")
    lev60 = [x for x in lev_ng if x[1] == 60]
    for pag in [49, 73]:
        chk_approx(pag, "Livelli MFIV totale, entrambi i controlli: quota a 60 minuti", "42",
                   [100 * r for _, _, r, _ in lev60], 2.0, F47I, "coef NG / coef base (levels_MFIV, h=60)",
                   unit="%", nota="vendor da solo 43.4% (la Sez. 5.4 dichiara il vendor come fonte di default)")
    chk_list(49, "Livelli MFIV totale, entrambi i controlli: p a 60 minuti (vendor, Black '76)",
             ["0.026", "0.040"], [p for *_, p in lev60], F47I, "p_value NG (levels_MFIV, h=60)")
    lev5 = [x for x in lev_ng if x[1] == 5]
    for pag in [49, 73]:
        chk_approx(pag, "Livelli MFIV totale, entrambi i controlli: a 5 minuti resta circa la meta'", "50",
                   [100 * r for _, _, r, _ in lev5], 5.0, F47I, "coef NG / coef base (levels_MFIV, h=5)", unit="%")
    chk_list(49, "Livelli MFIV totale, entrambi i controlli: p a 5 minuti (vendor, Black '76)",
             ["0.22", "0.15"], [p for *_, p in lev5], F47I, "p_value NG (levels_MFIV, h=5)")
    gmin = min(r for _, _, r, _ in log_g + lev_g)
    chk_cond(49, "Il volume gamma da solo non cambia nessuna delle due forme (quota del coefficiente)",
             "quasi invariato", gmin >= 0.95, f"min G/base {g(gmin, 3)}", F47I, "coef G / coef base")
    base_ann = [val(c47i("levels_MFIV_ann", h, s, "base"), "p_value", "47") for s in SOURCES_46 for h in HORIZONS]
    chk_cond(73, "Livelli con MFIV^ann(t): nessuna associazione da spiegare (base non significativa)",
             "base non significativa", all(p >= ALPHA for p in base_ann),
             "min p " + g(min(base_ann), 3), F47I, "p_value base (levels_MFIV_ann)")

    # Diagnostica giornaliera A4.
    dd = {s: (val(c47d("MFIV_t", "log", s, "NG"), "coef", "47") / val(c47d("MFIV_t", "log", s, "base"), "coef", "47"),
              val(c47d("MFIV_t", "log", s, "NG"), "p_value", "47")) for s in SOURCES_46}
    for s in SOURCES_46:
        det("A4 log MFIV_d, quota di H_VOL con entrambi i controlli", s, dd[s][0], F47D, "coef NG / coef base")
    chk_range(73, "A4 giornaliero (diagnostica), log con MFIV_d: quota rimasta, due fonti", "76", "81",
              [100 * dd[s][0] for s in SOURCES_46], F47D, "coef NG / coef base (MFIV_t, log)", unit="%")
    chk_list(73, "A4 giornaliero (diagnostica): p (vendor, Black '76)", ["0.019", "0.030"],
             [dd[s][1] for s in SOURCES_46], F47D, "p_value NG (MFIV_t, log)")

    # Tab. D.2: tutte le celle (vendor) e la nota Black '76.
    TABLE_D2 = {
        ("log_MFIV_ann", 5): [("-1.16", "0.535", "**"), ("-0.869", "0.509", "*"), ("-1.12", "0.517", "**"), ("-1.05", "0.513", "**")],
        ("log_MFIV_ann", 15): [("-1.13", "0.411", "***"), ("-0.906", "0.383", "**"), ("-1.08", "0.397", "***"), ("-1.03", "0.388", "***")],
        ("log_MFIV_ann", 30): [("-0.928", "0.381", "**"), ("-0.781", "0.351", "**"), ("-0.894", "0.369", "**"), ("-0.869", "0.353", "**")],
        ("log_MFIV_ann", 60): [("-0.735", "0.367", "**"), ("-0.639", "0.347", "*"), ("-0.720", "0.364", "**"), ("-0.695", "0.348", "**")],
        ("levels_MFIV", 5): [("-4.42", "1.59", "***"), ("-1.60", "2.06", ""), ("-4.39", "1.57", "***"), ("-2.35", "1.92", "")],
        ("levels_MFIV", 15): [("-11.0", "3.07", "***"), ("0.751", "6.28", ""), ("-10.8", "3.01", "***"), ("-1.18", "4.84", "")],
        ("levels_MFIV", 30): [("-26.8", "6.15", "***"), ("-5.73", "10.6", ""), ("-26.4", "6.03", "***"), ("-6.52", "8.75", "")],
        ("levels_MFIV", 60): [("-58.4", "14.3", "***"), ("-28.8", "9.89", "***"), ("-58.1", "14.0", "***"), ("-25.3", "11.3", "**")],
    }
    COLS_D2 = [("base", "Base"), ("N", "Strikes, log N"), ("G", "Gamma volume, log G"), ("NG", "Both")]
    for (spec, h), cells in TABLE_D2.items():
        scale = 1e6 if spec.startswith("levels") else 1.0
        panel = "A (log, MFIV^ann)" if spec.startswith("log") else "B (livelli, MFIV totale, x1e6)"
        for (ctrl, colname), (tc, tse, tst) in zip(COLS_D2, cells):
            r = c47i(spec, h, "vendor", ctrl)
            c, se, p = val(r, "coef", "47") * scale, val(r, "se", "47") * scale, val(r, "p_value", "47")
            ok = printed_match(c, tc) and printed_match(se, tse) and stars(p) == tst
            add(74, f"Tab. D.2 pannello {panel}, h = {h}, {colname}", f"{tc}{tst} ({tse})",
                f"{g(c, 5)}{stars(p)} ({g(se, 4)}); p {g(p, 3)}",
                "coefficiente ed errore standard alla cifra stampata; stelle", ok, F47I,
                f"coef, se, p_value ({spec}, vendor, {ctrl})")
    b_log = [x for x in log_ng if x[0] == "black76"]
    b_lev = [x for x in lev_ng if x[0] == "black76"]
    chk_range(74, "Nota Tab. D.2: Black '76, log, entrambi i controlli / base", "0.89", "0.94",
              [r for _, _, r, _ in b_log], F47I, "coef NG / coef base (log_MFIV_ann, black76)")
    chk_range(74, "Nota Tab. D.2: Black '76, log, p", "0.006", "0.043", [p for *_, p in b_log],
              F47I, "p_value NG (log_MFIV_ann, black76)")
    chk_range(74, "Nota Tab. D.2: Black '76, livelli, entrambi i controlli / base", "0.12", "0.53",
              [r for _, _, r, _ in b_lev], F47I, "coef NG / coef base (levels_MFIV, black76)")
    chk_list(74, "Nota Tab. D.2: Black '76, livelli, p a h = 5, 15, 30, 60", ["0.15", "0.76", "0.50", "0.040"],
             [p for *_, p in b_lev], F47I, "p_value NG (levels_MFIV, black76)")

    # =====================================================
    # ESPLORATIVI (p. 49): 4 APRILE ED EFFETTI FISSI LEGGERI
    # =====================================================

    i40 = read("i40_lev")
    F40 = label("i40_lev")
    expl_ok, expl_txt = True, []
    for src in SOURCES_45:
        for h in [30, 60]:
            r = one(i40, "40", regressor="MFIV", specification="IH3", gamma_source=src, horizon=h, term=f"H_OI_{src}")
            c0, c1 = val(r, "coef", "40"), val(r, "coef_without_top_day", "40")
            p0, p1 = val(r, "p_day", "40"), val(r, "p_day_without_top_day", "40")
            expl_ok &= (str(r["top_day_for_coef"]) == EXPLORATORY_DAY and c0 < 0 and abs(c1) > abs(c0)
                        and p0 >= ALPHA and p1 < ALPHA)
            expl_txt.append(f"{src} h{h}: {c0:.2e} (p {p0:.3f}) -> {c1:.2e} (p {p1:.3f}), giorno {r['top_day_for_coef']}")
    h15 = [one(i40, "40", regressor="MFIV", specification="IH3", gamma_source=s, horizon=15, term=f"H_OI_{s}")
           for s in SOURCES_45]
    chk_cond(49, "Esplorativo: H_OI in livelli con MFIV totale, negativo, tirato verso zero dal 4 aprile; "
             "senza quella sessione significativo a 30 e 60 minuti sulle due fonti",
             "significativo a 30 e 60, due fonti", expl_ok, "; ".join(expl_txt), F40,
             "coef, p_day, coef_without_top_day, p_day_without_top_day, top_day_for_coef",
             nota="a 15 minuti: " + ", ".join(f"{r['gamma_source']} p {float(r['p_day_without_top_day']):.3f}" for r in h15))
    light_ok, light_txt = True, []
    for src in SOURCES_45:
        for h in [30, 60]:
            r = one(i34, "34", specification="IH3", horizon=h, form="log", gamma_source=src,
                    fe_scheme="date_x_time", term=f"H_OI_{src}")
            light_ok &= val(r, "p_value", "34") < ALPHA
            light_txt.append(f"{src} h{h} p {val(r, 'p_value', '34'):.4f}")
    chk_cond(49, "Esplorativo: con effetti fissi leggeri H_OI in log significativo a 30 e 60 minuti, due fonti",
             "significativo (nominale)", light_ok, "; ".join(light_txt), label("i34"),
             "p_value (IH3, log, date_x_time, H_OI)", nota="le correzioni di Holm e BH sono di 16_Intraday_robustness (parte 7)")

    # =====================================================
    # SEZ. 5.5 (pp. 50-52): GRUPPO DI STRESS, RV_{d-1}
    # =====================================================

    s41, g41 = read("i41_sum"), read("i41_diag")
    lev41 = one(s41, "41", disegno="intraday", forma="levels", variante="insieme_completo")
    log41 = one(s41, "41", disegno="intraday", forma="log", variante="insieme_completo")
    chk_cond(50, "Gruppo di stress fisso: i livelli restano sensibili ad aprile, i log sono stabili",
             "livelli sensibili, log stabili",
             int(lev41["celle_che_attraversano_5pct"]) > 0 and int(log41["celle_che_attraversano_5pct"]) == 0,
             f"livelli: {int(lev41['celle_che_attraversano_5pct'])}/{int(lev41['celle_totali'])} celle attraversano il 5% "
             f"(mediana {float(lev41['variazione_mediana_pct']):.1f}%); log: {int(log41['celle_che_attraversano_5pct'])}/"
             f"{int(log41['celle_totali'])} (mediana {float(log41['variazione_mediana_pct']):.1f}%)",
             label("i41_sum"), "celle_che_attraversano_5pct, variazione_mediana_pct")
    nq = g41.loc[g41.variant == "solo_scadenze_trimestrali", "n_days_excluded"].unique()
    if len(nq) != 1:
        raise RuntimeError("[41] n_days_excluded non unico per le scadenze trimestrali")
    chk_value(50, "Scadenze trimestrali esaminate", "2", float(nq[0]), label("i41_diag"),
              "n_days_excluded (solo_scadenze_trimestrali)")

    rv_noapr = [val(c46(reg, "levels", src, "no_april", "RV_lag1"), "p_value", "46")
                for reg in DAILY_REGRESSORS for src in SOURCES_46]
    for pag in [52, 72]:
        chk_bound(pag, "A4 livelli senza aprile: RV_{d-1} indistinguibile da zero", ">", "0.18", rv_noapr,
                  F46, "p_value (levels, no_april, RV_lag1)")

    # INFO (b): B1 senza aprile, griglia di d31.
    d31 = read("d31")
    F31 = label("d31")
    need31 = ["dipendente", "controllo_apertura", "RV_lag1", "campione", "n", "coef_MFIV_0931", "p_MFIV_0931",
              "coef_SpotRV_open", "p_SpotRV_open", "r_squared"]
    miss = [c for c in need31 if c not in d31.columns]
    if miss or len(d31) != 16:
        raise RuntimeError(f"[31] griglia attesa 16 righe con {need31}; mancano {miss}, righe {len(d31)}")
    b1 = d31[need31].copy()
    b1["MFIV_0931_significativo_5pct"] = b1["p_MFIV_0931"] < ALPHA
    b1["SpotRV_open_significativo_5pct"] = b1["p_SpotRV_open"] < ALPHA
    b1.to_csv(OUTPUT_DIR / OUT_B1, index=False)
    base_na = one(d31, "31", dipendente="1m", controllo_apertura="1m", RV_lag1="1m", campione="senza aprile")
    base_full = one(d31, "31", dipendente="1m", controllo_apertura="1m", RV_lag1="1m", campione="intero")
    na5 = d31[(d31.campione == "senza aprile") & (d31.controllo_apertura == "5m")]
    na1 = d31[(d31.campione == "senza aprile") & (d31.controllo_apertura == "1m")]
    add(52, "INFO (b) B1 senza aprile: varianza della finestra di apertura (SpotRV_open)",
        "'not decomposed and not attributed to April' (p. 52)",
        f"intero {g(val(base_full, 'coef_SpotRV_open', '31'))} (p {g(val(base_full, 'p_SpotRV_open', '31'), 3)}); "
        f"senza aprile, controllo 1m: {g(val(base_na, 'coef_SpotRV_open', '31'))} (p {val(base_na, 'p_SpotRV_open', '31'):.1e}); "
        f"controllo 1m, p max {na1.p_SpotRV_open.max():.1e}; controllo 5m, p {na5.p_SpotRV_open.min():.2f}-{na5.p_SpotRV_open.max():.2f}",
        "informativa", True, F31, "coef_SpotRV_open, p_SpotRV_open",
        nota="14_Daily_MZ_robustness (parte 3) la scompone: senza aprile diventa significativa con il controllo di apertura a 1 minuto, "
             "resta non significativa con quello a 5 minuti; griglia in " + OUT_B1, info=True)

    # =====================================================
    # APPENDICE C (pp. 70-71)
    # =====================================================

    lags27 = d27.nw_lag_L.unique()
    chk_value(70, "Ritardi di Newey-West sul campione intero", "4", float(lags27[0]) if len(lags27) == 1 else -1,
              F27, "nw_lag_L", nota=f"valori trovati {sorted(lags27.tolist())}")
    lag_ok, lag_txt = True, []
    for name, df, ncol, lcol in [("27", d27, "n", "nw_lag_L"), ("30", d30, "n", "nw_lag_L"), ("46", d46, "n", "nw_lag")]:
        pairs = df[[ncol, lcol]].drop_duplicates()
        lag_ok &= all(nw_lag(int(n)) == int(L) for n, L in pairs.itertuples(index=False))
        lag_txt.append(f"{name}: " + ", ".join(f"n {int(n)} -> {int(L)}" for n, L in pairs.sort_values(ncol).itertuples(index=False)))
    chk_cond(70, "Ritardo floor(4 (T/100)^(2/9)) ricalcolato per ogni sottocampione", "regola rispettata",
             lag_ok, " | ".join(lag_txt), f"{F27} ; {F30} ; {F46}", "n, nw_lag_L / nw_lag")

    i43 = read("i43")
    F43 = label("i43")
    if len(i43) != 2:
        raise RuntimeError("[43] attese 2 righe (I1 h=5 livelli e log)")
    rel = (i43.coef_abs_diff / i43.coef_pyfixest.abs()).max()
    chk_cond(70, "Replica di I1 a 5 minuti con dummy esplicite, livelli e log: coefficienti a precisione di macchina",
             "precisione di macchina", rel < 1e-12, f"max differenza relativa {rel:.1e}", F43, "coef_abs_diff / coef_pyfixest")
    mult_nn = i43.correction_nonnested_K.unique()
    mult_full = i43.correction_full_K.unique()
    if len(mult_nn) != 1 or len(mult_full) != 1:
        raise RuntimeError("[43] moltiplicatori non unici fra le due righe")
    chk_value(70, "Moltiplicatore degli errori standard, effetti fissi annidati non contati", "1.13067",
              float(mult_nn[0]), F43, "correction_nonnested_K", d=7)
    chk_value(70, "Moltiplicatore contando gli effetti fissi (statsmodels)", "1.15002", float(mult_full[0]), F43,
              "correction_full_K", d=7)
    se_up = 100.0 * float(i43.se_rel_diff_naive.max())
    cross("43: rapporto dei moltiplicatori vs se_rel_diff_naive", float(mult_full[0]) / float(mult_nn[0]) - 1.0,
          se_up / 100.0, rtol=1e-6)
    chk_value(70, "Errori standard piu' grandi contando gli effetti fissi", "1.7", se_up, F43,
              "se_rel_diff_naive", shown="1.7%")
    vif = read("i44").vif_true.astype(float)
    chk_range(70, "VIF dei regressori di I3 (MFIV totale, pooled, senza effetti fissi)", "1.33", "1.85", vif.values,
              label("i44"), "vif_true")

    i42 = read("i42")
    F42 = label("i42")
    if len(i42) != 64:
        raise RuntimeError(f"[42] attese 64 celle, trovate {len(i42)}")
    Bs = i42.B.unique()
    chk_value(70, "Estrazioni di Rademacher per cella", "9999", float(Bs[0]) if len(Bs) == 1 else -1, F42, "B",
              shown="9,999", d=6)
    n_deg = int((i42.t_star_abs_max < DEGENERATE_MAX_ABS_T).sum())
    n_flag = int(i42.boot_degenerate.astype(str).str.lower().eq("true").sum())
    for pag in [71, 73]:
        chk_cond(pag, "Celle degeneri del bootstrap (|t*| massimo sotto 2.5)", "10 di 64",
                 n_deg == 10 and n_flag == 10 and len(i42) == 64,
                 f"{n_deg} di {len(i42)} (colonna boot_degenerate: {n_flag})", F42, "t_star_abs_max, boot_degenerate")
    iso = pd.Timestamp(APRIL_PEAK_DAY).isocalendar()
    week_label = f"{iso[0]}-W{int(iso[1]):02d}"
    iv = i42[i42.term.isin(["MFIV", "MFIV_ann"])]
    if len(iv) != 32:
        raise RuntimeError(f"[42] attese 32 celle di varianza implicita, trovate {len(iv)}")
    thr = 0.948 - half_unit("94.8") / 100.0
    cells31 = iv[(iv.top_week_restricted == week_label) & (iv.share_top1_restricted >= thr)]
    rest = iv.drop(cells31.index)
    for _, r in iv.iterrows():
        det("Quota della settimana principale (ristretta), celle IV", f"{r['regressor']} {r['specification']} h={r['horizon']}",
            100 * float(r["share_top1_restricted"]), F42, "share_top1_restricted")
    for pag in [71, 73]:
        chk_cond(pag, f"Celle di varianza implicita in cui la settimana del 7 aprile ({week_label}) pesa almeno il 94.8%",
                 "31 di 32", len(cells31) == 31,
                 f"{len(cells31)} di 32; la restante: " + ", ".join(
                     f"{r['regressor']} {r['specification']} h={r['horizon']} {100 * float(r['share_top1_restricted']):.1f}%"
                     for _, r in rest.iterrows()), F42, "top_week_restricted, share_top1_restricted")
    chk_range(71, "Quota della settimana del 7 aprile nelle 31 celle", "94.8", "99.96",
              100 * cells31.share_top1_restricted.values, F42, "share_top1_restricted", unit="%")

    # =====================================================
    # TAB. D.1 (pp. 72-73)
    # =====================================================

    chk_value(72, "Troncamento |k| <= 0.02: B1 in livelli, p di MFIV(09:31)", "0.074",
              val(one(d28_002, "28", specification="B-short", form="levels", term="MFIV_0931"), "p_value", "28"),
              label("d28_002"), "p_value (B-short, levels, MFIV_0931)")
    chk_list(72, "A4 log con MFIV totale: p di H_VOL con le regole bid, 0.05, 0.02", ["0.003", "0.007", "0.070"],
             trunc_p, f"{F27} ; {F28}", "p_value (A6, log, CBOE, H_VOL_t)")
    for reg, tc, tp in [("MFIV_ann_t", "-5.42", ["0.027", "0.028"]), ("MFIV_session_t", "-4.37", ["0.057", "0.060"])]:
        cs = [coef_full_log[(reg, s)] for s in SOURCES_46]
        add(72, f"Due fonti: A4 log, H_VOL con {reg} (coefficiente, entrambe le fonti)", tc,
            ", ".join(g(c) for c in cs), "arrotondamento alla cifra stampata", all(printed_match(c, tc) for c in cs),
            F46, "coef (log, full, H_VOL)")
        chk_list(72, f"Due fonti: A4 log, p di H_VOL con {reg} (vendor, Black '76)", tp,
                 [val(c46(reg, "log", s, "full", "H_VOL"), "p_value", "46") for s in SOURCES_46], F46,
                 "p_value (log, full, H_VOL)")
    chk_list(72, "Senza aprile: H_VOL in livelli con MFIV_d, MFIV^ann_d, MFIV^ses_d (vendor)",
             ["-1.77e-4", "-1.52e-4", "-1.48e-4"], [val(lev_noapr[(r, "vendor")], "coef", "46") for r in DAILY_REGRESSORS],
             F46, "coef (levels, no_april, vendor)")
    chk_bound(72, "Senza aprile: p di H_VOL in livelli, tre regressori, due fonti", "<=", "0.004", p_lev_noapr,
              F46, "p_value (levels, no_april)")
    chk_list(72, "Senza aprile: riduzione di RV_d per sd (vendor)", ["17.3", "14.9", "14.5"], eff_noapr_v, F46,
             "effect_pct_per_sd (levels, no_april, vendor)")
    chk_list(72, "Senza aprile: H_VOL in log (vendor)", ["-9.38", "-10.6", "-8.94"],
             [coef_noapr_log[(r, "vendor")] for r in DAILY_REGRESSORS], F46, "coef (log, no_april, vendor)")
    chk_bound(72, "Senza aprile: p di H_VOL in log, tre regressori, due fonti", "<", "0.001", p_noapr_log, F46,
              "p_value (log, no_april)")
    hoi_noapr = [val(c46(reg, f, s, "no_april", "H_OI"), "p_value", "46")
                 for reg in DAILY_REGRESSORS for f in ["levels", "log"] for s in SOURCES_46]
    chk_bound(72, "Senza aprile: H_OI non significativo in nessuna forma", ">", "0.37", hoi_noapr, F46,
              "p_value (no_april, H_OI)")
    n_noapr = d46.loc[d46["sample"] == "no_april", "n"].unique()
    chk_value(72, "Sessioni di A4 senza aprile", "228", float(n_noapr[0]) if len(n_noapr) == 1 else -1, F46,
              "n (no_april)", nota="249 osservazioni di A4 (serve RV_{d-1}) meno 21 sessioni di aprile", d=4)

    # INFO (a): I1 in livelli a h = 5 con le bande alternative.
    i1 = {}
    for key, tag in [("i34", "bid"), ("i39_005", "|k|<=0.05"), ("i39_002", "|k|<=0.02")]:
        i1[tag] = val(one(read(key), key, specification="IH1", horizon=5, form="levels", gamma_source="none",
                          fe_scheme="date_x_weektime", term="MFIV_ann"), "p_value", key)
        det("I1 livelli h=5, p di MFIV^ann per regola di troncamento", tag, i1[tag], label(key), "p_value")
    add(72, "INFO (a) Troncamento: I1 in livelli a h = 5 (l'unica cella I1 significativa in livelli)",
        "non menzionato ('Main implied-variance results persist, but B1 in levels weakens')",
        ", ".join(f"{k} p {v:.3f}" for k, v in i1.items()), "informativa", True,
        f"{label('i34')} ; {label('i39_005')} ; {label('i39_002')}", "p_value (IH1, h=5, levels, MFIV_ann)",
        nota="con |k| <= 0.02 la cella I1 h=5 in livelli perde la significativita' (Tab. 5.2: p = 0.027)", info=True)

    # Frequenza del gamma.
    fall_tot = {s: 100.0 * (1.0 - abs(grid[s][0][-1]) / abs(grid[s][0][0])) for s in grid}
    for s in grid:
        det("A4 log H_VOL: caduta da 1 a 30 minuti (%)", s, fall_tot[s], label("d33"), "coef")
    chk_cond(73, "Griglie del gamma 1, 5, 15, 30 minuti: il coefficiente assoluto cala a ogni passo, due fonti",
             "cala a ogni passo", all(falls.values()),
             "; ".join(f"{s} " + " > ".join(g(abs(x)) for x in grid[s][0]) for s in grid), label("d33"),
             "coef (A6, log, H_VOL_k)")
    add(73, "Griglie del gamma: caduta complessiva", "circa 10%",
        ", ".join(f"{s} {fall_tot[s]:.1f}%" for s in grid), "arrotondamento all'intero (cifra stampata)",
        all(printed_match(v, "10") for v in fall_tot.values()), label("d33"), "coef k=1 vs k=30")
    chk_cond(73, "Griglie del gamma: la significativita' resta", "p < 0.05",
             all(max(grid[s][1]) < ALPHA for s in grid),
             "max p " + g(max(max(grid[s][1]) for s in grid), 3), label("d33"), "p_value")

    # =====================================================
    # SCRITTURA E VERDETTO
    # =====================================================

    detail_df = pd.DataFrame(DETAIL)
    chk = pd.DataFrame(CHECKS, columns=["pagina", "voce", "valore_tesi", "valore_calcolato", "precisione", "esito",
                                        "nota", "file_fonte", "colonna_fonte"])
    detail_df.to_csv(OUTPUT_DIR / OUT_DETAIL, index=False)
    chk.to_csv(OUTPUT_DIR / OUT_THESIS, index=False)

    n_ok = int((chk.esito == "OK").sum())
    n_bad = int((chk.esito == "DIVERSO").sum())
    n_info = int((chk.esito == "INFO").sum())
    print()
    print(f"CONFRONTO CON LA TESI: {len(chk)} voci, {n_ok} OK, {n_bad} DIVERSO, {n_info} INFO")
    with pd.option_context("display.width", 250, "display.max_colwidth", 120, "display.max_rows", 500):
        if n_bad:
            print(chk[chk.esito == "DIVERSO"].to_string(index=False))
        print()
        print(chk[chk.esito == "INFO"][["pagina", "voce", "valore_calcolato", "nota"]].to_string(index=False))
    print()
    print(f"Scritti in {OUTPUT_DIR}:")
    for f in [OUT_DETAIL, OUT_THESIS, OUT_B1]:
        print(f"  {f}")
    print(f"Tempo: {time.time() - t_start:.1f} s")
    if n_bad == 0:
        print(f"VERDETTO: OK. Tutti i {n_ok} valori stampati controllati coincidono; {n_info} voci INFO da leggere.")
    else:
        print(f"VERDETTO: DA CONTROLLARE. {n_bad} valori diversi dalla tesi (elencati sopra); {n_info} voci INFO.")


def main():
    text_derived_numbers()


if __name__ == "__main__":
    main()
