"""
EI Fraud Risk  (Anomaly Detection) — Canada

What this does:
1) Downloads Statistics Canada EI claims data (monthly) as a ZIP "full table download"
2) Builds a tidy time series by province/territory
3) Runs Isolation Forest per region to flag statistically unusual months
4) Plots a time-series with anomaly points highlighted

Data source:
- Statistics Canada Table 14-10-0005-01 (Claims received by province/territory, monthly)
Downloaded via StatCan "full table download (CSV) ZIP" pattern.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# ---- StatCan Full Table Download (CSV ZIP) ----
# Pattern documented by Statistics Canada developer guidance:
# https://www150.statcan.gc.ca/n1/en/tbl/csv/<8digits>-eng.zip
#
# For Table 14-10-0005-01, the download id is commonly published as 14100005.
DEFAULT_ZIP_ID = "14100005"
DEFAULT_LANG = "eng"


@dataclass
class SeriesConfig:
    zip_id: str = DEFAULT_ZIP_ID
    lang: str = DEFAULT_LANG
    contamination: float = 0.03
    random_state: int = 42
    min_points: int = 36  # don't model tiny series


def download_statcan_zip(zip_id: str, lang: str = "eng", timeout: int = 60) -> bytes:
    url = f"https://www150.statcan.gc.ca/n1/en/tbl/csv/{zip_id}-{lang}.zip"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def read_first_data_csv_from_zip(zip_bytes: bytes) -> pd.DataFrame:
    """
    StatCan full-table ZIP typically contains:
    - one main data CSV
    - one or more metadata CSVs (often smaller)
    We'll pick the largest .csv as the data table.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV files found in ZIP.")

        # choose largest csv entry as main data
        infos = [(n, z.getinfo(n).file_size) for n in csv_names]
        infos.sort(key=lambda t: t[1], reverse=True)
        data_name = infos[0][0]

        with z.open(data_name) as f:
            df = pd.read_csv(f)

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to normalize likely StatCan column names.
    Common columns include:
      - REF_DATE (date), GEO (region), VALUE (numeric)
    """
    cols = {c.upper(): c for c in df.columns}

    def pick(*cands: str) -> Optional[str]:
        for cand in cands:
            if cand in cols:
                return cols[cand]
        return None

    ref = pick("REF_DATE", "REFDATE", "DATE")
    geo = pick("GEO", "GEOGRAPHY", "GEO_NAME")
    val = pick("VALUE", "VAL", "OBS_VALUE")

    if ref is None or geo is None or val is None:
        raise RuntimeError(
            "Couldn't find expected columns. "
            f"Have columns: {list(df.columns)}"
        )

    out = df[[ref, geo, val]].rename(columns={ref: "ref_date", geo: "geo", val: "value"}).copy()

    # Parse date: StatCan often uses YYYY-MM or YYYY-MM-DD strings.
    out["ref_date"] = pd.to_datetime(out["ref_date"], errors="coerce")
    out = out.dropna(subset=["ref_date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])

    # Remove aggregate "Canada" if user wants strictly provinces, but keep it available.
    out["geo"] = out["geo"].astype(str)

    return out.sort_values(["geo", "ref_date"]).reset_index(drop=True)


def add_features(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Create lightweight anomaly-detection features from a single region time series.
    """
    ts = ts.sort_values("ref_date").copy()
    ts["month"] = ts["ref_date"].dt.month.astype(int)

    ts["diff_1"] = ts["value"].diff(1)
    ts["pct_change_1"] = ts["value"].pct_change(1).replace([np.inf, -np.inf], np.nan)

    # Rolling stats for context (avoid lookahead: rolling includes current point, OK for offline detection)
    win = 6
    ts["roll_mean_6"] = ts["value"].rolling(win, min_periods=3).mean()
    ts["roll_std_6"] = ts["value"].rolling(win, min_periods=3).std()

    ts["z_roll_6"] = (ts["value"] - ts["roll_mean_6"]) / ts["roll_std_6"]
    ts["z_roll_6"] = ts["z_roll_6"].replace([np.inf, -np.inf], np.nan)

    # Fill NA conservatively
    ts = ts.fillna(0.0)
    return ts


def detect_anomalies_isoforest(ts: pd.DataFrame, cfg: SeriesConfig) -> pd.DataFrame:
    ts = add_features(ts)

    feature_cols = ["value", "diff_1", "pct_change_1", "z_roll_6", "month"]
    X = ts[feature_cols].to_numpy(dtype=float)

    if len(ts) < cfg.min_points:
        ts["anomaly"] = False
        ts["anomaly_score"] = np.nan
        return ts

    model = IsolationForest(
        n_estimators=300,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
    )
    model.fit(X)

    # decision_function: higher means more normal; lower means more anomalous
    scores = model.decision_function(X)
    preds = model.predict(X)  # -1 anomaly, +1 normal

    ts["anomaly_score"] = scores
    ts["anomaly"] = preds == -1
    return ts


def plot_region(ts: pd.DataFrame, region: str, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(ts["ref_date"], ts["value"], linewidth=1.5)

    anom = ts[ts["anomaly"]]
    if not anom.empty:
        ax.scatter(anom["ref_date"], anom["value"], s=35)

        # label a few most extreme anomalies (lowest scores)
        label_n = min(6, len(anom))
        worst = anom.nsmallest(label_n, "anomaly_score")
        for _, r in worst.iterrows():
            ax.annotate(
                r["ref_date"].strftime("%Y-%m"),
                (r["ref_date"], r["value"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )

    ax.set_title(f"EI Claims Anomaly Signal — {region}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Claims (seasonally adjusted)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--zip-id", default=DEFAULT_ZIP_ID, help="StatCan full-table ZIP id (default: 14100005)")
    p.add_argument("--lang", default=DEFAULT_LANG, choices=["eng", "fra"], help="ZIP language (eng/fra)")
    p.add_argument("--region", default="Canada", help='Region/province label to plot (e.g., "Ontario")')
    p.add_argument("--contamination", type=float, default=0.03, help="Expected anomaly fraction (e.g., 0.02–0.08)")
    p.add_argument("--out", default="ei_anomaly.png", help="Output image path")
    p.add_argument("--export-csv", default="ei_anomaly_flags.csv", help="Export anomaly flags to CSV")
    args = p.parse_args()

    cfg = SeriesConfig(
        zip_id=args.zip_id,
        lang=args.lang,
        contamination=args.contamination,
    )

    print(f"Downloading StatCan ZIP {cfg.zip_id}-{cfg.lang}.zip ...")
    zip_bytes = download_statcan_zip(cfg.zip_id, cfg.lang)

    print("Reading table CSV from ZIP ...")
    raw = read_first_data_csv_from_zip(zip_bytes)

    print("Standardizing columns ...")
    df = standardize_columns(raw)

    # Run anomaly detection per region
    regions = sorted(df["geo"].unique())
    out_frames: List[pd.DataFrame] = []
    for g in regions:
        ts = df[df["geo"] == g][["ref_date", "geo", "value"]].copy()
        ts2 = detect_anomalies_isoforest(ts, cfg)
        out_frames.append(ts2)

    scored = pd.concat(out_frames, ignore_index=True)

    # Export a tidy file you can audit / share
    scored_out = scored[["ref_date", "geo", "value", "anomaly", "anomaly_score"]].copy()
    scored_out.to_csv(args.export_csv, index=False)
    print(f"Wrote CSV: {args.export_csv}")

    # Plot chosen region
    region = args.region
    if region not in set(scored["geo"].unique()):
        # fallback: try case-insensitive contains
        candidates = [g for g in regions if region.lower() in g.lower()]
        if candidates:
            region = candidates[0]
            print(f'Region not found exactly; using "{region}"')
        else:
            raise SystemExit(f'Region "{args.region}" not found. Example regions: {regions[:10]} ...')

    ts_plot = scored[scored["geo"] == region].sort_values("ref_date").copy()

    print(f'Plotting "{region}" anomalies -> {args.out}')
    plot_region(ts_plot, region=region, outpath=args.out)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
