"""
06_SPY_minute_bars.py: barre al minuto dello SPY e varianza realizzata, usate solo per il
confronto della volatility signature con l'indice (Appendice B, Tab. B.1).

Parte 1: barre a 1 e 10 minuti dagli Excel Bloomberg mensili.
Parte 2: varianza realizzata forward a 30 e 60 minuti.
Parte 3: pulizia dei minuti doppi delle esportazioni sovrapposte; rendimenti e RV ricalcolati.

Input:  CBOE/Data/00_inputs/SPY_excel/ (25_01.xlsx ... 25_12.xlsx)
Output: CBOE/Data/06_spy_minute_bars/
"""

import re

import numpy as np
import pandas as pd

from common import INPUT_DIR, output_dir

OUTPUT_DIR = output_dir("06_spy_minute_bars")

SPY_EXCEL_DIR = INPUT_DIR / "SPY_excel"
EXCEL_FILES = [SPY_EXCEL_DIR / f"25_{i:02d}.xlsx" for i in range(1, 13)]
SHEET_NAME = "Worksheet"

OUT_1M = OUTPUT_DIR / "06_SPY_2025_1min.csv"
OUT_10M = OUTPUT_DIR / "06_SPY_2025_10min.csv"
OUT_1M_RV = OUTPUT_DIR / "06_SPY_2025_1min_with_RV.csv"
OUT_10M_RV = OUTPUT_DIR / "06_SPY_2025_10min_with_RV.csv"
OUT_CLEAN = OUTPUT_DIR / "06_SPY_2025_1min_clean_RV.csv"

BAR_COLUMNS = [
    "Date",
    "Time Interval",
    "Close",
    "Net Chg",
    "Open",
    "High",
    "Low",
    "Tick Count",
    "Volume",
    "Log Return",
]


# =====================================================================
# PARTE 1: BARRE A 1 E 10 MINUTI DAGLI EXCEL
# =====================================================================
DATE_MARKER_RE = re.compile(r"^\d{2}[A-Z]{3}\d{4}_\d{2}:\d{2}:\d{2}\.\d{6}$")


def parse_date_marker(s: str) -> pd.Timestamp:
    d = s.split("_", 1)[0]  # "02JAN2025"
    return pd.to_datetime(d, format="%d%b%Y")


def is_time_interval(s: str) -> bool:
    return isinstance(s, str) and re.match(r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}$", s.strip()) is not None


def start_time_from_interval(s: str) -> str:
    return s.split("-")[0].strip()


def build_minute_bars():
    all_rows = []

    for f in EXCEL_FILES:
        if not f.exists():
            raise FileNotFoundError(f"File non trovato: {f}")

        df = pd.read_excel(f, sheet_name=SHEET_NAME)
        df.columns = [str(c).strip() for c in df.columns]

        # rimuove "Summary"
        df = df[df["Time Interval"].astype(str).str.strip().ne("Summary")].copy()

        current_date = None

        for _, row in df.iterrows():
            ti = row["Time Interval"]
            if not isinstance(ti, str):
                continue
            ti = ti.strip()

            # marker di data
            if DATE_MARKER_RE.match(ti):
                current_date = parse_date_marker(ti)
                continue

            # riga intraday
            if current_date is not None and is_time_interval(ti):
                st = start_time_from_interval(ti)  # HH:MM
                dt = pd.to_datetime(f"{current_date.date()} {st}:00")
                all_rows.append({
                    "dt": dt,
                    "Close": row["Close"],
                    "Open": row["Open"],
                    "High": row["High"],
                    "Low": row["Low"],
                    "Tick Count": row["Tick Count"],
                    "Volume": row["Volume"],
                })

    raw = pd.DataFrame(all_rows).dropna(subset=["dt"]).sort_values("dt").set_index("dt")

    # 1 minuto: come gli Excel, piu' Date e Log Return (azzerati ogni giorno)
    bars_1m = raw.copy()

    bars_1m["Date"] = bars_1m.index.date.astype(str)
    start_str_1m = bars_1m.index.strftime("%H:%M")
    end_str_1m = (bars_1m.index + pd.Timedelta(minutes=1)).strftime("%H:%M")
    bars_1m["Time Interval"] = start_str_1m + " - " + end_str_1m

    bars_1m["Net Chg"] = bars_1m.groupby("Date")["Close"].diff()
    bars_1m["Log Return"] = bars_1m.groupby("Date")["Close"].transform(lambda s: np.log(s).diff())

    bars_1m = bars_1m[BAR_COLUMNS]
    bars_1m.to_csv(OUT_1M, index=False)
    print(f"Creato: {OUT_1M}")
    print(bars_1m.head())
    print(bars_1m.tail())

    # 10 minuti: ricampionamento delle barre al minuto
    bars_10m = raw.resample("10min").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Tick Count": "sum",
        "Volume": "sum",
    }).dropna(subset=["Close"])

    bars_10m["Date"] = bars_10m.index.date.astype(str)
    start_str_10m = bars_10m.index.strftime("%H:%M")
    end_str_10m = (bars_10m.index + pd.Timedelta(minutes=10)).strftime("%H:%M")
    bars_10m["Time Interval"] = start_str_10m + " - " + end_str_10m

    bars_10m["Net Chg"] = bars_10m.groupby("Date")["Close"].diff()
    bars_10m["Log Return"] = bars_10m.groupby("Date")["Close"].transform(lambda s: np.log(s).diff())

    bars_10m = bars_10m[BAR_COLUMNS]
    bars_10m.to_csv(OUT_10M, index=False)
    print(f"\nCreato: {OUT_10M}")
    print(bars_10m.head())
    print(bars_10m.tail())


# =====================================================================
# PARTE 2: VARIANZA REALIZZATA FORWARD A 30 E 60 MINUTI
# =====================================================================
RV_COLUMNS = BAR_COLUMNS + [
    "RV30", "RV60", "RV30_fwd_excl", "RV60_fwd_excl",
    "Vol_RV30", "Vol_RV60", "Vol_RV30_fwd_excl", "Vol_RV60_fwd_excl",
]


def forward_rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """
    Somma in avanti allineata a t: sum_{j=0..window-1} series[t+j],
    calcolata con una somma mobile sulla serie rovesciata.
    """
    return series.iloc[::-1].rolling(window=window, min_periods=window).sum().iloc[::-1]


def realized_variance():
    df = pd.read_csv(OUT_1M)

    # Data + ora iniziale di "Time Interval" ("HH:MM - HH:MM")
    start_time = df["Time Interval"].astype(str).str.split("-").str[0].str.strip()
    df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + start_time + ":00", errors="coerce")

    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    df["r"] = pd.to_numeric(df["Log Return"], errors="coerce")
    df["r2"] = df["r"] ** 2

    # RV per giornata (azzerata ogni giorno)
    g = df.groupby("Date", sort=False, group_keys=False)

    # minuto corrente incluso
    df["RV30"] = g["r2"].apply(lambda s: forward_rolling_sum(s.fillna(0.0), 30))
    df["RV60"] = g["r2"].apply(lambda s: forward_rolling_sum(s.fillna(0.0), 60))

    # minuto corrente escluso (orizzonte puramente in avanti)
    df["RV30_fwd_excl"] = g["r2"].apply(lambda s: forward_rolling_sum(s.shift(-1).fillna(0.0), 30))
    df["RV60_fwd_excl"] = g["r2"].apply(lambda s: forward_rolling_sum(s.shift(-1).fillna(0.0), 60))

    # volatilita' (radice)
    df["Vol_RV30"] = np.sqrt(df["RV30"])
    df["Vol_RV60"] = np.sqrt(df["RV60"])
    df["Vol_RV30_fwd_excl"] = np.sqrt(df["RV30_fwd_excl"])
    df["Vol_RV60_fwd_excl"] = np.sqrt(df["RV60_fwd_excl"])

    df[RV_COLUMNS].to_csv(OUT_1M_RV, index=False)
    print("Saved 1-min:", OUT_1M_RV)

    # 10 minuti: ultima osservazione di ogni intervallo di 10 minuti
    df["bin10"] = df["datetime"].dt.floor("10min")
    df10 = df.sort_values("datetime").groupby(["Date", "bin10"], as_index=False).tail(1)

    df10[RV_COLUMNS].to_csv(OUT_10M_RV, index=False)
    print("Saved 10-min:", OUT_10M_RV)

    print("Rows 1-min:", len(df), "Rows 10-min:", len(df10))


# =====================================================================
# PARTE 3: PULIZIA
# =====================================================================
def forward_sum(series, window):
    """
    Somma il valore corrente e i successivi window-1 valori (t, t+1, ..., t+29
    per window=30). Se non ci sono abbastanza righe fino alla fine della
    giornata restituisce NaN.
    """
    return (
        series
        .iloc[::-1]
        .rolling(
            window=window,
            min_periods=window
        )
        .sum()
        .iloc[::-1]
    )


def clean():
    df = pd.read_csv(OUT_1M_RV, low_memory=False)

    print(f"Righe iniziali: {len(df):,}")

    # 1. Duplicati delle esportazioni mensili sovrapposte (31 gennaio e 31 ottobre):
    #    stessa Date e Time Interval, si tiene la prima occorrenza.
    df = df.drop_duplicates(
        subset=["Date", "Time Interval"],
        keep="first"
    ).copy()

    print(f"Righe dopo rimozione duplicati: {len(df):,}")

    # 2. Ordine per data e ora iniziale ("15:30 - 15:31" -> "15:30")
    df["_date_sort"] = pd.to_datetime(
        df["Date"],
        errors="raise"
    )

    start_time = df["Time Interval"].str.extract(
        r"^\s*(\d{1,2}:\d{2})"
    )[0]

    df["_time_sort"] = pd.to_timedelta(
        start_time + ":00"
    )

    df = (
        df
        .sort_values(
            ["_date_sort", "_time_sort"],
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    # 3. Close numerico
    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    # 4. Log Return_t = ln(Close_t / Close_t-1), separatamente per ogni giorno:
    #    il primo minuto di ogni giornata resta NaN.
    df["Log Return"] = (
        df
        .groupby("Date", sort=False)["Close"]
        .transform(
            lambda x: np.log(x / x.shift(1))
        )
    )

    # 5. Rendimenti al quadrato. Il primo di ogni giornata (NaN) vale 0 nella RV,
    #    come nella parte 2.
    ret2 = df["Log Return"] ** 2

    first_row_day = (
        df
        .groupby("Date", sort=False)
        .cumcount()
        .eq(0)
    )

    ret2_for_rv = ret2.copy()

    ret2_for_rv.loc[
        first_row_day & ret2_for_rv.isna()
    ] = 0.0

    # 6. RV30 e RV60 (minuto corrente incluso)
    for window in [30, 60]:

        rv_col = f"RV{window}"

        df[rv_col] = (
            ret2_for_rv
            .groupby(df["Date"], sort=False)
            .transform(
                lambda x: forward_sum(x, window)
            )
        )

    # 7. RV30_fwd_excl e RV60_fwd_excl: Return^2(t+1) + ... + Return^2(t+30 o t+60).
    #    Lo shift lascia NaN nell'ultima riga della giornata: vale 0, come nella
    #    parte 2.
    ret2_next = (
        ret2
        .groupby(df["Date"], sort=False)
        .shift(-1)
    )

    last_row_day = (
        df
        .groupby("Date", sort=False)
        .cumcount(ascending=False)
        .eq(0)
    )

    ret2_next.loc[
        last_row_day & ret2_next.isna()
    ] = 0.0

    for window in [30, 60]:

        col = f"RV{window}_fwd_excl"

        df[col] = (
            ret2_next
            .groupby(df["Date"], sort=False)
            .transform(
                lambda x: forward_sum(x, window)
            )
        )

    # 8. Volatilita' = radice della RV
    df["Vol_RV30"] = np.sqrt(df["RV30"])
    df["Vol_RV60"] = np.sqrt(df["RV60"])

    df["Vol_RV30_fwd_excl"] = np.sqrt(
        df["RV30_fwd_excl"]
    )

    df["Vol_RV60_fwd_excl"] = np.sqrt(
        df["RV60_fwd_excl"]
    )

    # 9. Net Chg ricalcolato dopo aver tolto i duplicati
    df["Net Chg"] = (
        df
        .groupby("Date", sort=False)["Close"]
        .diff()
    )

    df = df.drop(
        columns=[
            "_date_sort",
            "_time_sort"
        ]
    )

    df.to_csv(
        OUT_CLEAN,
        index=False
    )

    print()
    print("Operazione completata.")
    print(f"File salvato come: {OUT_CLEAN}")
    print()
    print(
        "Duplicati rimasti:",
        df.duplicated(
            subset=["Date", "Time Interval"]
        ).sum()
    )
    print()
    print(
        df[
            [
                "Date", "Time Interval", "Close", "Net Chg", "Log Return",
                "RV30", "RV60", "RV30_fwd_excl", "RV60_fwd_excl",
                "Vol_RV30", "Vol_RV60", "Vol_RV30_fwd_excl", "Vol_RV60_fwd_excl",
            ]
        ].head(10)
    )


def main():
    # senza gli Excel Bloomberg la parte 1 si salta, se il CSV a 1 minuto esiste gia'
    if all(f.exists() for f in EXCEL_FILES):
        build_minute_bars()
    elif OUT_1M.exists():
        missing = [f.name for f in EXCEL_FILES if not f.exists()]
        print(f"AVVISO: mancano {len(missing)} Excel SPY in {SPY_EXCEL_DIR} ({', '.join(missing[:3])}...).")
        print(f"        Parte 1 saltata: le parti 2 e 3 ripartono dal file gia' presente {OUT_1M.name}.")
    else:
        raise FileNotFoundError(
            f"Mancano gli Excel SPY in {SPY_EXCEL_DIR} e non c'e' {OUT_1M.name}: impossibile proseguire."
        )
    realized_variance()
    clean()


if __name__ == "__main__":
    main()
