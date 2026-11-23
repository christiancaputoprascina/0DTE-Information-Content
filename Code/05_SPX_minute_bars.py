"""
05_SPX_minute_bars.py: barre al minuto dell'S&P 500 (SPX) e varianza realizzata giornaliera.

Per ogni Excel Bloomberg mensile: orario da Roma a New York con etichetta a fine intervallo,
filtro del mese e della chiusura per giornata (13:00 nelle chiusure anticipate), rendimenti
fra chiusure consecutive e RV giornaliera ripetuta su ogni riga. E' l'unica implementazione
della varianza realizzata nella pipeline.

Input:  CBOE/Data/00_inputs/SPX_excel/ (Jan.xlsx ... Dec.xlsx)
Output: CBOE/Data/05_spx_minute_bars/ (05_SPX_01_RV.csv ... 05_SPX_12_RV.csv)
"""

import re
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from common import DATA_DIR, EARLY_CLOSE_TIMES_HHMM, INPUT_DIR, REGULAR_CLOSE, output_dir

# =====================================================================
# PARAMETRI
# =====================================================================
UNDERLYING = "SPX"

SPX_EXCEL_DIR = INPUT_DIR / "SPX_excel"
OUTPUT_DIR = output_dir("05_spx_minute_bars")
OUTPUT_FILE_PREFIX = "05_"

# Gli orari degli Excel sono in ora di Roma (la prima barra della giornata e' "15:30 - 15:31",
# cioe' 09:30 ET). ZoneInfo gestisce anche le settimane in cui lo scarto Roma-New York e' di
# cinque ore anziche' sei, per il diverso calendario dell'ora legale.
SOURCE_TZ = "Europe/Rome"

# Output e filtro in orario americano
TARGET_TZ = "America/New_York"

# =====================================================================
# CHIUSURA DI MERCATO, PER GIORNATA
# =====================================================================
# 16:00, oppure 13:00 nelle sessioni a chiusura anticipata, dove le barre successive alla
# chiusura vanno escluse (EARLY_CLOSE_TIMES_HHMM in common.py, formato "HH:MM").
DEFAULT_MARKET_CLOSE = REGULAR_CLOSE[:5]  # "16:00"

# =====================================================================
# CONVENZIONE DI ETICHETTATURA DELL'ORARIO
# =====================================================================
# Il Close della barra Bloomberg "15:30 - 15:31" e' il prezzo a fine intervallo: la barra
# si etichetta all'orario finale, come i timestamp della MFIV (390 barre, 09:31-16:00).
# Non cambia la RV giornaliera, ma evita di spostare di un minuto le finestre orarie.
LABEL_AT_INTERVAL_END = True

# Modalita' RV:
# "daily"      = RV giornaliera ripetuta su ogni riga del giorno
# "interval"   = contributo RV di ogni intervallo
# "cumulative" = RV cumulata durante la giornata
RV_MODE = "daily"


# =====================================================================
# BARRE SPX AL MINUTO E RV DAGLI EXCEL MENSILI
# =====================================================================
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "SEPT": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def get_month_from_filename(input_file):
    stem = input_file.stem.strip().upper()

    if stem not in MONTHS:
        raise ValueError(f"Mese non riconosciuto dal nome file: {input_file.name}")

    return MONTHS[stem]


def get_output_filename(month_number):
    return f"{OUTPUT_FILE_PREFIX}SPX_{month_number:02d}_RV.csv"


def clean_column_name(col):
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
    )


def find_column(df, possible_names):
    cleaned_columns = {
        clean_column_name(c): c
        for c in df.columns
    }

    for name in possible_names:
        key = clean_column_name(name)

        if key in cleaned_columns:
            return cleaned_columns[key]

    raise ValueError(f"Colonna non trovata. Cercavo una tra: {possible_names}")


def is_date_marker(value):
    """
    Riconosce righe tipo:
    02JAN2025_00:00:00.000000
    """
    if pd.isna(value):
        return False

    s = str(value).strip().upper()

    pattern = r"^\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)\d{4}_"

    return re.match(pattern, s) is not None


def parse_date_marker(value):
    """
    Converte:
    02JAN2025_00:00:00.000000
    in data Python.
    """
    s = str(value).strip().upper()
    date_part = s.split("_")[0]

    match = re.match(
        r"^(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)(\d{4})$",
        date_part
    )

    if match is None:
        raise ValueError(f"Formato data non riconosciuto: {value}")

    day = int(match.group(1))
    month_text = match.group(2)
    year = int(match.group(3))

    month = MONTHS[month_text]

    return datetime(year, month, day).date()


def parse_interval_time(value):
    """
    Estrae l'orario dall'intervallo Bloomberg "15:30 - 15:31": quello finale
    (15:31) con LABEL_AT_INTERVAL_END = True, quello iniziale (15:30) altrimenti.
    """
    if pd.isna(value):
        return None

    if isinstance(value, time):
        return value.strftime("%H:%M")

    if isinstance(value, pd.Timestamp):
        return value.strftime("%H:%M")

    s = str(value).strip()

    matches = re.findall(r"(\d{1,2}):(\d{2})", s)

    if not matches:
        return None

    if LABEL_AT_INTERVAL_END and len(matches) >= 2:
        hour, minute = matches[1]
    else:
        hour, minute = matches[0]

    return f"{int(hour):02d}:{int(minute):02d}"


def convert_to_target_time(date_value, time_value):
    """
    Combina data + ora nel fuso SOURCE_TZ
    e converte nel fuso TARGET_TZ.
    """
    hour, minute = map(int, time_value.split(":"))

    dt_source = datetime(
        date_value.year,
        date_value.month,
        date_value.day,
        hour,
        minute,
        tzinfo=ZoneInfo(SOURCE_TZ)
    )

    dt_target = dt_source.astimezone(ZoneInfo(TARGET_TZ))

    return dt_target.date(), dt_target.strftime("%H:%M")


def time_to_minutes(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    match = re.search(r"(\d{1,2}):(\d{2})", s)

    if match is None:
        return np.nan

    hour = int(match.group(1))
    minute = int(match.group(2))

    return hour * 60 + minute


def to_numeric_clean(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan}),
        errors="coerce"
    )


def convert_single_file(input_file):
    expected_month = get_month_from_filename(input_file)

    print(f"\nConverto: {input_file.name}")
    print(f"Mese atteso: {expected_month:02d}")

    raw = pd.read_excel(input_file, sheet_name=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    time_col = find_column(raw, ["Time Interval", "TimeInterval", "Time"])
    open_col = find_column(raw, ["Open"])
    close_col = find_column(raw, ["Close"])
    high_col = find_column(raw, ["High"])
    low_col = find_column(raw, ["Low"])
    net_change_col = find_column(raw, ["Net Chg", "Net Change", "NetChange"])
    tick_count_col = find_column(raw, ["Tick Count", "TickCount"])
    volume_col = find_column(raw, ["Volume", "Volumr"])

    rows = []
    current_date = None

    for _, row in raw.iterrows():
        time_value = row[time_col]

        if is_date_marker(time_value):
            current_date = parse_date_marker(time_value)
            continue

        if pd.isna(time_value):
            continue

        if str(time_value).strip().lower() == "summary":
            continue

        if current_date is None:
            continue

        interval_time = parse_interval_time(time_value)

        if interval_time is None:
            continue

        if pd.isna(row[close_col]):
            continue

        target_date, target_time = convert_to_target_time(
            current_date,
            interval_time
        )

        rows.append({
            "Underlying": UNDERLYING,
            "Date": target_date,
            "time": target_time,
            "open": row[open_col],
            "close": row[close_col],
            "high": row[high_col],
            "low": row[low_col],
            "net change": row[net_change_col],
            "tick count": row[tick_count_col],
            "volume": row[volume_col],
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"ATTENZIONE: nessun dato valido trovato in {input_file.name}")
        return

    rows_initial = len(df)

    # FILTRO MESE: tiene le date che, dopo la conversione a New York, cadono nel
    # mese del file; la barra di mezzanotte italiana del primo del mese e' ancora
    # il pomeriggio del giorno prima a New York.
    df["Date"] = pd.to_datetime(df["Date"])

    df = df[df["Date"].dt.month == expected_month].copy()

    rows_after_month_filter = len(df)
    rows_removed_wrong_month = rows_initial - rows_after_month_filter

    if df.empty:
        print(f"ATTENZIONE: dopo il filtro mese non resta nessun dato in {input_file.name}")
        return

    # FILTRO ORARIO, CHIUSURA PER GIORNATA
    df["_date_str"] = df["Date"].dt.strftime("%Y-%m-%d")

    df["_close_minutes"] = df["_date_str"].map(
        lambda d: time_to_minutes(EARLY_CLOSE_TIMES_HHMM.get(d, DEFAULT_MARKET_CLOSE))
    )

    df["_time_minutes"] = df["time"].apply(time_to_minutes)

    rows_before_time_filter = len(df)

    early_rows_before = int(df["_date_str"].isin(EARLY_CLOSE_TIMES_HHMM).sum())

    df = df[df["_time_minutes"] <= df["_close_minutes"]].copy()

    rows_after_time_filter = len(df)
    rows_removed_after_close = rows_before_time_filter - rows_after_time_filter

    early_rows_after = int(df["_date_str"].isin(EARLY_CLOSE_TIMES_HHMM).sum())

    df = df.drop(columns=["_time_minutes", "_close_minutes", "_date_str"])

    if df.empty:
        print(f"ATTENZIONE: dopo il filtro di chiusura non resta nessun dato in {input_file.name}")
        return

    # PULIZIA NUMERI
    numeric_cols = [
        "open",
        "close",
        "high",
        "low",
        "net change",
        "tick count",
        "volume",
    ]

    for col in numeric_cols:
        df[col] = to_numeric_clean(df[col])

    df = df.dropna(subset=["close"]).copy()

    # prezzi non positivi (il logaritmo darebbe -inf): segnalati ed eliminati
    non_positive = int((df["close"] <= 0).sum())

    if non_positive > 0:
        print(f"ATTENZIONE: {non_positive} righe con close <= 0, eliminate")
        df = df[df["close"] > 0].copy()

    if df.empty:
        print(f"ATTENZIONE: dopo pulizia numerica non resta nessun dato in {input_file.name}")
        return

    # ORDINAMENTO
    df = df.sort_values(
        ["Underlying", "Date", "time"]
    ).reset_index(drop=True)

    # REALIZED VARIANCE, calcolata dopo il filtro mese e il filtro di chiusura.
    df["log_return"] = (
        np.log(df["close"])
        - np.log(df.groupby(["Underlying", "Date"])["close"].shift(1))
    )

    df["rv_interval"] = df["log_return"] ** 2

    daily_rv = (
        df.groupby(["Underlying", "Date"], as_index=False)["rv_interval"]
        .sum()
        .rename(columns={"rv_interval": "daily_RV"})
    )

    df = df.merge(
        daily_rv,
        on=["Underlying", "Date"],
        how="left"
    )

    if RV_MODE == "daily":
        df["RV"] = df["daily_RV"]

    elif RV_MODE == "interval":
        df["RV"] = df["rv_interval"]

    elif RV_MODE == "cumulative":
        df["RV"] = (
            df.groupby(["Underlying", "Date"])["rv_interval"]
            .cumsum()
        )

    else:
        raise ValueError("RV_MODE deve essere: 'daily', 'interval' oppure 'cumulative'")

    # OUTPUT FINALE
    final_df = df[
        [
            "Underlying",
            "Date",
            "time",
            "open",
            "close",
            "high",
            "low",
            "net change",
            "tick count",
            "volume",
            "RV",
        ]
    ].copy()

    final_df["Date"] = final_df["Date"].dt.strftime("%Y-%m-%d")

    output_file = OUTPUT_DIR / get_output_filename(expected_month)

    final_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # diagnostica per giornata: osservazioni e ultimo orario tenuto
    per_day = (
        final_df.groupby("Date")
        .agg(n=("close", "size"), last=("time", "max"))
    )

    print(f"Creato: {output_file.name}")
    print(f"Righe iniziali valide:                {rows_initial}")
    print(f"Eliminate perche' fuori mese:         {rows_removed_wrong_month}")
    print(f"Eliminate perche' dopo la chiusura:   {rows_removed_after_close}")
    print(f"Righe finali:                         {len(final_df)}")
    print(f"Osservazioni per giornata: min {per_day['n'].min()} max {per_day['n'].max()}")
    print(f"Ultimo orario tenuto: {sorted(per_day['last'].unique())}")

    if early_rows_before > 0:
        print(f"  early close in questo mese: {early_rows_before} righe prima, "
              f"{early_rows_after} dopo il filtro")


def spx_minute_bars():
    if not SPX_EXCEL_DIR.exists():
        raise FileNotFoundError(
            f"Cartella degli Excel SPX non trovata: {SPX_EXCEL_DIR}"
        )

    print(f"Cartella dati:       {DATA_DIR}")
    print(f"Cerco file Excel in: {SPX_EXCEL_DIR}")
    print(f"Scrivo i CSV in:     {OUTPUT_DIR}")
    print(f"Chiusura di default: {DEFAULT_MARKET_CLOSE}")
    print(f"Early close: {EARLY_CLOSE_TIMES_HHMM}")
    print(f"Etichetta a fine intervallo: {LABEL_AT_INTERVAL_END}")

    excel_files = []

    for file in SPX_EXCEL_DIR.iterdir():
        if not file.is_file():
            continue

        if file.name.startswith("~$"):
            continue

        if file.suffix.lower() not in [".xlsx", ".xls"]:
            continue

        try:
            get_month_from_filename(file)
            excel_files.append(file)
        except ValueError:
            continue

    excel_files = sorted(
        excel_files,
        key=lambda x: get_month_from_filename(x)
    )

    print("\nFile Excel trovati:")

    for file in excel_files:
        print(f"- {file.name}")

    if not excel_files:
        raise FileNotFoundError(
            f"Nessun file Excel mensile trovato in: {SPX_EXCEL_DIR}. "
            "I file devono chiamarsi Jan.xlsx, Feb.xlsx, ..., Dec.xlsx"
        )

    for file in excel_files:
        convert_single_file(file)

    print("\nConversione completata.")


def main():
    spx_minute_bars()


if __name__ == "__main__":
    main()
