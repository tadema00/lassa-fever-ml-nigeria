"""
================================================================================
Predicting Lassa Fever Outbreaks Across Distinct Ecological Zones in Nigeria:
An 8-Year Spatiotemporal Analysis Using Machine Learning and
Time-Lagged Climate Drivers (2018–2025)
================================================================================

Analysis pipeline:
  Section 1  — Data loading, integration & quality control
  Section 2  — Lag imputation (forward-fill)
  Section 3  — Descriptive statistics & epidemiological summary
  Section 4  — Exploratory visualisations (Figs 1–5)
  Section 5  — STL seasonal decomposition (Fig 6)
  Section 6  — Spearman lagged cross-correlation analysis (Figs 7–8)
  Section 7  — Negative binomial GLM with IRR forest plot (Fig 9)
  Section 8  — Predictive modelling: Random Forest + LightGBM/GBM
  Section 9  — SHAP interpretability analysis (Figs 10–12)
  Section 10 — Bootstrap confidence intervals (Fig 13)
  Section 11 — Granger causality tests (Fig 14)
  Section 12 — SARIMA baseline & head-to-head comparison (Figs 15–16)
  Section 13 — Outbreak detection — early warning system (Fig 17)
  Section 14 — Manuscript tables export

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
                scipy shap lightgbm
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not installed — GradientBoostingRegressor will be used instead.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed — install with: pip install shap")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    from statsmodels.tsa.seasonal import STL
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("statsmodels not installed — install with: pip install statsmodels")

# ── Output directories ────────────────────────────────────────────────────────
FIGURES_DIR = "figures"
RESULTS_DIR = "results"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Publication style ─────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1A3A5C",
    "secondary": "#C0392B",
    "accent":    "#F39C12",
    "neutral":   "#7F8C8D",
    "light_bg":  "#F4F6F9",
    "grid":      "#D5D8DC",
}
STATE_COLORS = {
    "Bauchi":  "#E74C3C", "Ebonyi": "#9B59B6",
    "Edo":     "#2980B9", "Ondo":   "#27AE60",
    "Plateau": "#F39C12", "Taraba": "#16A085",
}
ECOLOGICAL_ZONES = {
    "Southern (Rainforest)":            ["Ondo", "Edo"],
    "Northern/Central (Savanna/Plateau)": ["Bauchi", "Taraba", "Ebonyi", "Plateau"],
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.6,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

def savefig(name, subdir=FIGURES_DIR):
    path = os.path.join(subdir, name)
    plt.savefig(path)
    plt.close()
    print(f"  ✓ Saved: {path}")

# ── Helper metrics ────────────────────────────────────────────────────────────
def willmott_d(obs, sim):
    obs_mean = np.mean(obs)
    denom = np.sum((np.abs(sim - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    return 1 - np.sum((obs - sim) ** 2) / denom if denom > 0 else np.nan

def pbias(obs, sim):
    return 100 * np.sum(obs - sim) / np.sum(obs) if np.sum(obs) != 0 else np.nan

def mape(obs, sim):
    mask = obs > 0
    return 100 * np.mean(np.abs((obs[mask] - sim[mask]) / obs[mask])) if mask.sum() > 0 else np.nan

def eval_metrics(y_true, y_pred, label=""):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).clip(0)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mp   = mape(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    d    = willmott_d(y_true, y_pred)
    pb   = pbias(y_true, y_pred)
    if label:
        print(f"  {label:28s}  R²={r2:.3f}  RMSE={rmse:.2f}  "
              f"MAE={mae:.2f}  MAPE={mp:.1f}%  d={d:.3f}  PBIAS={pb:.1f}%")
    return {"R2": round(r2,3), "RMSE": round(rmse,3), "MAE": round(mae,3),
            "MAPE_%": round(mp,2), "Pearson_r": round(r,3),
            "Willmott_d": round(d,3), "PBIAS_%": round(pb,2)}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING, INTEGRATION & QUALITY CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 1: Data Loading & Integration")
print("="*70)

# ── Only accept a pre-merged file if it actually contains a 'cases' column ──
MERGED_CANDIDATES = [
    "/content/results/Lassa_Final_Merged_Dataset.csv",
    "results/Lassa_Final_Merged_Dataset.csv",
    "Lassa_Final_Merged_Dataset.csv",
]
FEATURES_CANDIDATES = [
    "/content/Lassa_climate_features_engineered.csv",
    "Lassa_climate_features_engineered.csv",
]
CASES_CANDIDATES = [
    "/content/Cases_rainfal_data.csv",
    "Cases_rainfal_data.csv",
]

df = None
for p in MERGED_CANDIDATES:
    if os.path.exists(p):
        _tmp = pd.read_csv(p, nrows=2)
        if "cases" in _tmp.columns:
            df = pd.read_csv(p)
            print(f"Loaded merged dataset : {p}")
            print(f"Shape                 : {df.shape}")
            break
        else:
            print(f"Skipped (no 'cases' column): {p}")

if df is None:
    print("Building merged dataset from raw source files...")
    features_path = next((p for p in FEATURES_CANDIDATES if os.path.exists(p)), None)
    cases_path    = next((p for p in CASES_CANDIDATES    if os.path.exists(p)), None)

    if not features_path:
        raise FileNotFoundError(
            "Cannot find Lassa_climate_features_engineered.csv\n"
            "Upload it to /content/ in Colab."
        )
    if not cases_path:
        raise FileNotFoundError(
            "Cannot find Cases_rainfal_data.csv\n"
            "Upload it to /content/ in Colab."
        )

    print(f"  Features file : {features_path}")
    print(f"  Cases file    : {cases_path}")

    df_features   = pd.read_csv(features_path)
    df_cases_wide = pd.read_csv(cases_path)

    print(f"  Features shape: {df_features.shape}")
    print(f"  Cases shape   : {df_cases_wide.shape}")

    case_cols = (["years", "weeks"] +
                 [c for c in df_cases_wide.columns if c.endswith("_cases")])
    df_cases_long = (
        df_cases_wide[case_cols]
        .melt(id_vars=["years", "weeks"], var_name="state_raw", value_name="cases")
    )
    df_cases_long["state"] = (
        df_cases_long["state_raw"]
        .str.replace("_cases", "", regex=False)
        .str.capitalize()
    )
    df_cases_long = df_cases_long.drop(columns=["state_raw"])

    if "state" in df_features.columns:
        df_features["state"] = df_features["state"].str.capitalize()

    # Identify week column in features file
    feat_week_col = next(
        (c for c in df_features.columns if c.lower() in ("week_number", "week", "weeks")),
        None,
    )
    if feat_week_col is None:
        raise ValueError(
            f"No week column found in features file. "
            f"Columns: {list(df_features.columns)}"
        )

    df = pd.merge(
        df_features, df_cases_long,
        left_on=["state", feat_week_col],
        right_on=["state", "weeks"],
        how="inner",
    ).drop(columns=["weeks"], errors="ignore")

    print(f"  Merged shape  : {df.shape}")
    if "cases" not in df.columns:
        raise ValueError(
            "Merge produced no 'cases' column. "
            "Check that state names and week numbers align between files."
        )

# ── Standardise column names & types ─────────────────────────────────────────
if "state" in df.columns:
    df["state"] = df["state"].astype(str).str.capitalize()

year_col = next((c for c in df.columns if c.lower() in ("years", "year")), None)
week_col = next((c for c in df.columns if c.lower() in ("week_number", "weeks", "week")), None)

if "date" not in df.columns:
    def yw_to_date(row):
        try:
            return pd.to_datetime(
                f"{int(row[year_col])}-W{int(row[week_col]):02d}-1",
                format="%G-W%V-%u",
            )
        except Exception:
            return pd.NaT
    df["date"] = df.apply(yw_to_date, axis=1)
else:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

if "week_of_year" not in df.columns:
    df["week_of_year"] = ((df[week_col] - 1) % 52) + 1
if "month" not in df.columns:
    df["month"] = df["date"].dt.month
if "season" not in df.columns:
    df["season"] = df["week_of_year"].apply(
        lambda w: "Dry" if w <= 14 or w >= 45 else "Wet"
    )
if "log_cases" not in df.columns:
    df["log_cases"] = np.log1p(df["cases"])

df = df.sort_values(["state", "date"]).reset_index(drop=True)

print(f"States              : {sorted(df['state'].unique())}")
print(f"Date range          : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Total cases         : {df['cases'].sum():,}")
print(f"Columns             : {list(df.columns)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LAG IMPUTATION (FORWARD-FILL)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 2: Lag Imputation")
print("="*70)

lag_cols = [c for c in df.columns if "lag" in c.lower() or "roll" in c.lower()]

diag_rows = []
for col in lag_cols:
    diag_rows.append({
        "column":       col,
        "zeros_before": (df[col] == 0).sum(),
        "NaN_before":   df[col].isna().sum(),
    })

df = df.sort_values(["state", "date"])
for col in lag_cols:
    df[col] = df.groupby("state")[col].transform(lambda x: x.ffill())

rows_before = len(df)
df          = df.dropna(subset=lag_cols).reset_index(drop=True)
rows_after  = len(df)

for row in diag_rows:
    col = row["column"]
    row["zeros_after"] = (df[col] == 0).sum()
    row["NaN_after"]   = df[col].isna().sum()

diag_df = pd.DataFrame(diag_rows)
diag_df.to_csv(os.path.join(RESULTS_DIR, "imputation_diagnostic.csv"), index=False)

print(f"  Rows before : {rows_before}  |  Dropped : {rows_before - rows_after}  "
      f"|  Remaining : {rows_after}")
print(diag_df[["column", "zeros_before", "NaN_before",
               "zeros_after", "NaN_after"]].to_string(index=False))

df.to_csv(os.path.join(RESULTS_DIR, "Lassa_Final_Merged_Dataset.csv"), index=False)
print("  ✓ Cleaned dataset saved → results/Lassa_Final_Merged_Dataset.csv")

states = sorted(df["state"].dropna().unique())

EXCLUDE_COLS = {
    "state", "week_number", "weeks", "years", "year", "date",
    "cases", "log_cases", "month", "quarter", "is_dry_season",
    "state_enc", "ecological_zone", "month_name", "local_rain_mm",
    "week_of_year", "season",
}
feature_cols = [
    c for c in df.columns
    if c not in EXCLUDE_COLS
    and df[c].dtype in [float, "float64", int, "int64"]
]

CLIMATE_DRIVERS = {
    "weekly_soil_moisture": "Soil Moisture",
    "rel_humidity":         "Relative Humidity",
    "weekly_rainfall":      "Rainfall",
    "weekly_temp":          "Temperature",
    "weekly_ndvi":          "NDVI",
}
CLIMATE_DRIVERS = {k: v for k, v in CLIMATE_DRIVERS.items() if k in df.columns}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 3: Descriptive Statistics")
print("="*70)

state_summary = (
    df.groupby("state").agg(
        Total_Cases=("cases", "sum"),
        Mean_Weekly=("cases", "mean"),
        Median_Weekly=("cases", "median"),
        Max_Weekly=("cases", "max"),
        Std_Weekly=("cases", "std"),
        Zero_weeks=("cases", lambda x: (x == 0).sum()),
        Weeks_with_cases=("cases", lambda x: (x > 0).sum()),
    ).round(2).sort_values("Total_Cases", ascending=False)
)
state_summary["Zero_weeks_%"] = (
    state_summary["Zero_weeks"] / (state_summary["Zero_weeks"] + state_summary["Weeks_with_cases"]) * 100
).round(1)
state_summary.to_csv(os.path.join(RESULTS_DIR, "Table1_Descriptive_Stats.csv"))
print(state_summary.to_string())

climate_summary = (
    df.groupby("state")[list(CLIMATE_DRIVERS.keys())].mean().round(3)
)
climate_summary.to_csv(os.path.join(RESULTS_DIR, "state_climate_summary.csv"))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPLORATORY VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 4: Exploratory Visualisations")
print("="*70)

# ── Fig 1: Weekly epidemic curves ────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

for ax, state in zip(axes.flat, states):
    sub   = df[df["state"] == state].sort_values("date")
    color = STATE_COLORS.get(state, "steelblue")
    ax.bar(sub["date"], sub["cases"], color=color, alpha=0.35, width=6, label="Weekly cases")
    roll  = sub["cases"].rolling(4, center=True).mean()
    ax.plot(sub["date"], roll, color=color, lw=2, label="4-wk average")
    for yr in range(2018, 2026):
        ax.axvspan(pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr}-04-07"),
                   color=PALETTE["accent"], alpha=0.08)
        ax.axvspan(pd.Timestamp(f"{yr}-11-03"), pd.Timestamp(f"{yr}-12-31"),
                   color=PALETTE["accent"], alpha=0.08)
    ax.set_title(f"{state}  (Total: {int(sub['cases'].sum()):,} cases)",
                 fontweight="bold", color=color)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Cases")
    ax.legend(fontsize=8)
    ax.set_facecolor(PALETTE["light_bg"])
plt.tight_layout()
savefig("Fig1_Epidemic_Curves.png")

# ── Fig 2: Annual trends ──────────────────────────────────────────────────────
annual = df.groupby(["years", "state"])["cases"].sum().reset_index()
fig, ax = plt.subplots(figsize=(11, 6))
for state in states:
    d = annual[annual["state"] == state]
    ax.plot(d["years"], d["cases"], marker="o",
            label=state, color=STATE_COLORS.get(state, "gray"), linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Annual Cases")

ax.legend(title="State", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_facecolor(PALETTE["light_bg"])
plt.tight_layout()
savefig("Fig2_Annual_Trends.png")

# ── Fig 3: Seasonal heatmap (week × year) ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, state in zip(axes.flat, states):
    d = df[df["state"] == state]
    pivot = d.pivot_table(index="week_of_year", columns="years",
                          values="cases", aggfunc="sum")
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"label": "Cases"}, yticklabels=4)
    ax.set_title(state, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Epidemic Week")

plt.tight_layout()
savefig("Fig3_Seasonal_Heatmap.png")

# ── Fig 4: Ecological zone comparison ────────────────────────────────────────
df["ecological_zone"] = df["state"].map(
    {s: z for z, sl in ECOLOGICAL_ZONES.items() for s in sl}
)
zone_weekly = df.groupby(["date", "ecological_zone"])["cases"].sum().reset_index()
fig, ax = plt.subplots(figsize=(14, 6))
for zone, color in zip(ECOLOGICAL_ZONES.keys(), [PALETTE["secondary"], PALETTE["primary"]]):
    d    = zone_weekly[zone_weekly["ecological_zone"] == zone]
    roll = d["cases"].rolling(8, min_periods=1).mean()
    ax.plot(d["date"], d["cases"], color=color, linewidth=0.8, alpha=0.5)
    ax.plot(d["date"], roll, color=color, linewidth=2.5, label=f"{zone} (8-wk avg)")
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Lassa Fever Cases")

ax.legend()
ax.set_facecolor(PALETTE["light_bg"])
plt.tight_layout()
savefig("Fig4_Ecological_Zones.png")

# ── Fig 5: Seasonal stratification ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
woy_mean = df.groupby(["state", "week_of_year"])["cases"].mean().reset_index()
ax = axes[0]
for state in states:
    sub = woy_mean[woy_mean["state"] == state]
    ax.plot(sub["week_of_year"], sub["cases"].rolling(3, center=True).mean(),
            lw=2, label=state, color=STATE_COLORS.get(state, "gray"))
ax.axvspan(1, 14, color=PALETTE["accent"], alpha=0.12, label="Dry Season")
ax.axvspan(45, 52, color=PALETTE["accent"], alpha=0.12)
ax.set_xlabel("Week of Year")
ax.set_ylabel("Mean Weekly Cases")

ax.legend(loc="upper right")
ax.set_facecolor(PALETTE["light_bg"])

ax = axes[1]
season_data = [
    df[df["season"] == "Dry"]["cases"].values,
    df[df["season"] == "Wet"]["cases"].values,
]
ax.boxplot(season_data,
           labels=["Dry Season\n(Wks 1–14 & 45–52)", "Wet Season\n(Wks 15–44)"],
           patch_artist=True, notch=True,
           boxprops=dict(facecolor=PALETTE["accent"], alpha=0.6),
           medianprops=dict(color="black", lw=2))
# Median values reported in axis label via annotation below
dry_med = np.median(season_data[0])
wet_med = np.median(season_data[1])
ax.text(1, dry_med + 0.1, f"Median: {dry_med:.1f}", ha="center",
        va="bottom", fontsize=9, color="black", fontweight="bold")
ax.text(2, wet_med + 0.1, f"Median: {wet_med:.1f}", ha="center",
        va="bottom", fontsize=9, color="black", fontweight="bold")
ax.set_ylabel("Weekly Cases")
ax.set_facecolor(PALETTE["light_bg"])

plt.tight_layout()
savefig("Fig5_Seasonal_Analysis.png")

print("  ✓ Figures 1–5 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STL SEASONAL DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 5: STL Seasonal Decomposition")
print("="*70)

if HAS_SM:
    stl_states = ["Edo", "Ondo"] if "Edo" in states and "Ondo" in states else states[:2]
    fig, axes  = plt.subplots(4, 2, figsize=(18, 14))

    comp_labels = ["Observed", "Trend", "Seasonal", "Residual"]
    comp_colors = [PALETTE["primary"], PALETTE["secondary"], "#27AE60", PALETTE["neutral"]]

    for col_idx, state in enumerate(stl_states):
        sub    = df[df["state"] == state].sort_values("date")
        series = sub["cases"].values
        res    = STL(series, period=52, robust=True).fit()
        comps  = [series, res.trend, res.seasonal, res.resid]
        for row_idx, (comp, label, color) in enumerate(
                zip(comps, comp_labels, comp_colors)):
            ax = axes[row_idx, col_idx]
            if label == "Residual":
                ax.bar(range(len(comp)), comp, color=color, alpha=0.6, width=1.0)
            else:
                ax.plot(comp, color=color, lw=1.5)
            ax.set_ylabel(label, fontweight="bold", color=color)
            if row_idx == 0:
                ax.set_title(f"{state} State", fontsize=13, fontweight="bold",
                             color=STATE_COLORS.get(state, "black"))
            ax.set_facecolor(PALETTE["light_bg"])
            if row_idx == 3:
                ax.set_xlabel("Week Number")
            ax.axhline(0, color="black", lw=0.5, ls="--")
    plt.tight_layout()
    savefig("Fig6_STL_Decomposition.png")
    print("  ✓ Figure 6 saved.")
else:
    print("  Skipped — statsmodels not available.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SPEARMAN LAGGED CROSS-CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 6: Spearman Lagged Cross-Correlation Analysis")
print("="*70)

LAGS = [1, 2, 4, 8]
lag_records = []

for state in states:
    sub = df[df["state"] == state].sort_values("date").copy()
    for col, label in CLIMATE_DRIVERS.items():
        for lag in LAGS:
            lagged = sub[col].shift(lag)
            mask   = lagged.notna()
            if mask.sum() < 30:
                continue
            rho, pval = spearmanr(sub["cases"][mask], lagged[mask])
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            lag_records.append({
                "State": state, "Driver": label,
                "Lag": lag, "Rho": round(rho, 4),
                "PValue": pval, "Sig": sig,
            })

df_lag = pd.DataFrame(lag_records)
df_lag.to_csv(os.path.join(RESULTS_DIR, "Table2_Spearman_Lag_Results.csv"), index=False)

best_lag = (
    df_lag.loc[df_lag.groupby(["State", "Driver"])["Rho"]
               .apply(lambda x: x.abs().idxmax())]
    .sort_values(["Driver", "Rho"])
)
best_lag.to_csv(os.path.join(RESULTS_DIR, "Table2b_Best_Lag_Summary.csv"), index=False)
print("\n  Peak lag-response (best lag per state–driver pair):")
print(best_lag[["State", "Driver", "Lag", "Rho", "PValue", "Sig"]].to_string(index=False))

# ── Fig 7: Lag-response heatmap ───────────────────────────────────────────────
fig, axes_row = plt.subplots(1, len(CLIMATE_DRIVERS), figsize=(22, 7), sharey=True)

for ax_idx, (col, label) in enumerate(CLIMATE_DRIVERS.items()):
    pivot = (
        df_lag[df_lag["Driver"] == label]
        .pivot(index="State", columns="Lag", values="Rho")
    )
    pivot.columns = [f"Lag {l}w" for l in pivot.columns]
    vabs = max(abs(pivot.values.min()), abs(pivot.values.max()))
    sns.heatmap(
        pivot, ax=axes_row[ax_idx], annot=True, fmt=".3f",
        cmap="RdBu_r", center=0, vmin=-vabs, vmax=vabs,
        linewidths=0.5, linecolor="white",
        cbar=(ax_idx == len(CLIMATE_DRIVERS) - 1),
        annot_kws={"size": 8},
    )
    axes_row[ax_idx].set_title(label, fontweight="bold", fontsize=11)
    axes_row[ax_idx].set_xlabel("Lag (weeks)")
    if ax_idx > 0:
        axes_row[ax_idx].set_ylabel("")
plt.tight_layout()
savefig("Fig7_Lag_Response_Heatmap.png")

# ── Fig 8: Spatial correlation bar chart (Lag 2) ──────────────────────────────
lag2        = df_lag[df_lag["Lag"] == 2].copy()
lag2_pivot  = lag2.pivot(index="State", columns="Driver", values="Rho")
lag2_pivot  = lag2_pivot.reindex(
    sorted(lag2_pivot.index, key=lambda s: lag2_pivot.loc[s, "Soil Moisture"])
)
bar_colors  = ["#2980B9", "#C0392B", "#27AE60", "#F39C12", "#8E44AD"]
fig, ax     = plt.subplots(figsize=(14, 7))
x, width    = np.arange(len(lag2_pivot)), 0.15

for i, (driver, color) in enumerate(zip(lag2_pivot.columns, bar_colors)):
    vals = lag2_pivot[driver].values
    bars = ax.bar(x + i * width, vals, width, label=driver,
                  color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        if abs(v) > 0.2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.01 if v >= 0 else -0.03),
                    f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7, fontweight="bold", color=color)

ax.axhline(0, color="black", lw=0.8)
ax.axhline(-0.3, color=PALETTE["neutral"], lw=0.8, ls="--", alpha=0.5)
ax.axhline( 0.3, color=PALETTE["neutral"], lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(lag2_pivot.index, fontsize=11)
ax.set_xlabel("State", fontweight="bold")
ax.set_ylabel("Spearman's ρ (Lag 2 weeks)", fontweight="bold")

ax.legend(title="Climatic Driver", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.set_facecolor(PALETTE["light_bg"])
plt.tight_layout()
savefig("Fig8_Spatial_Correlation_Chart.png")
print("  ✓ Figures 7–8 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — NEGATIVE BINOMIAL GLM WITH FOREST PLOT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 7: Negative Binomial GLM")
print("="*70)

glm_records = []

if HAS_SM:
    glm_vars = [c for c in feature_cols if df[c].std() > 0 and df[c].nunique() > 5][:8]

    for state in states:
        d = df[df["state"] == state].dropna(subset=glm_vars + ["cases"])
        if len(d) < 30:
            continue
        X_glm = sm.add_constant(d[glm_vars])
        y_glm = d["cases"].astype(int)
        try:
            nb = sm.GLM(y_glm, X_glm, family=sm.families.NegativeBinomial()).fit()
            coef_df = pd.DataFrame({
                "state":       state,
                "variable":    nb.params.index,
                "coefficient": nb.params.values.round(4),
                "std_err":     nb.bse.values.round(4),
                "z_value":     nb.tvalues.values.round(4),
                "p_value":     nb.pvalues.values.round(4),
                "IRR":         np.exp(nb.params.values).round(4),
                "IRR_CI_low":  np.exp(nb.conf_int()[0]).values.round(4),
                "IRR_CI_high": np.exp(nb.conf_int()[1]).values.round(4),
            })
            glm_records.append(coef_df)
            pseudo_r2 = 1 - nb.deviance / nb.null_deviance
            print(f"  {state}: AIC={nb.aic:.1f}  Pseudo-R²={pseudo_r2:.3f}")
        except Exception as e:
            print(f"  GLM failed for {state}: {e}")

    if glm_records:
        glm_df = pd.concat(glm_records, ignore_index=True)
        glm_df.to_csv(os.path.join(RESULTS_DIR, "Table_NB_Regression.csv"), index=False)

        sig = glm_df[(glm_df["p_value"] < 0.05) & (glm_df["variable"] != "const")].copy()
        if not sig.empty:
            sig["label"] = sig["state"] + " — " + sig["variable"]
            sig = sig.sort_values("IRR")
            fig, ax = plt.subplots(figsize=(10, max(6, len(sig) * 0.35)))
            y_pos = range(len(sig))
            ax.scatter(sig["IRR"], y_pos, color="navy", zorder=3, s=40)
            ax.hlines(y_pos, sig["IRR_CI_low"], sig["IRR_CI_high"],
                      color="navy", linewidth=1.2, alpha=0.7)
            ax.axvline(1, color="red", linestyle="--", linewidth=1)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(sig["label"].tolist(), fontsize=8)
            ax.set_xlabel("Incidence Rate Ratio (IRR) with 95% CI")
            ax.set_xlabel("Incidence Rate Ratio (IRR) with 95% CI")
            ax.axvline(1, color="red", linestyle="--", linewidth=1, label="IRR = 1 (null)")
            savefig("Fig9_Forest_Plot_GLM.png")
            print("  ✓ Figure 9 saved.")
else:
    print("  Skipped — statsmodels not available.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — PREDICTIVE MODELLING (STATE-STRATIFIED RF + LightGBM/GBM)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 8: Predictive Modelling")
print("="*70)

MODEL_LABEL = "LightGBM" if HAS_LGBM else "Gradient Boosting"

def build_model(lgbm=False):
    if lgbm and HAS_LGBM:
        return lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=6, random_state=42, verbose=-1,
        )
    return GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42,
    )

model_outputs = {}
metrics_rows  = []

for state in states:
    sub   = df[df["state"] == state].sort_values("date").copy()
    feats = [f for f in feature_cols if f in sub.columns]
    X     = sub[feats].fillna(0)
    y_log = sub["log_cases"]
    y_raw = sub["cases"]

    if len(X) < 40:
        print(f"  {state}: skipped (insufficient data)")
        continue

    tscv     = TimeSeriesSplit(n_splits=5)
    cv_rows  = []

    for tr_idx, te_idx in tscv.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_log.iloc[tr_idx], y_log.iloc[te_idx]
        y_raw_te   = y_raw.iloc[te_idx].values

        rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                                   min_samples_leaf=4, random_state=42)
        rf.fit(X_tr, y_tr)
        pred_rf = np.expm1(rf.predict(X_te))

        gb = build_model(lgbm=True)
        gb.fit(X_tr, y_tr)
        pred_gb = np.expm1(gb.predict(X_te))

        cv_rows.append({
            "RF_R2":   r2_score(y_te, rf.predict(X_te)),
            "RF_MAE":  mean_absolute_error(y_raw_te, pred_rf),
            "RF_RMSE": np.sqrt(mean_squared_error(y_raw_te, pred_rf)),
            "RF_d":    willmott_d(y_raw_te, pred_rf),
            "RF_PBIAS":pbias(y_raw_te, pred_rf),
            "RF_MAPE": mape(y_raw_te, pred_rf),
            "GB_R2":   r2_score(y_te, gb.predict(X_te)),
            "GB_MAE":  mean_absolute_error(y_raw_te, pred_gb),
            "GB_RMSE": np.sqrt(mean_squared_error(y_raw_te, pred_gb)),
            "GB_d":    willmott_d(y_raw_te, pred_gb),
            "GB_PBIAS":pbias(y_raw_te, pred_gb),
            "GB_MAPE": mape(y_raw_te, pred_gb),
        })

    cv_mean = pd.DataFrame(cv_rows).mean().round(4)

    split   = len(sub) - 52
    X_tr_f, X_te_f = X.iloc[:split], X.iloc[split:]
    y_tr_f, y_te_f = y_log.iloc[:split], y_log.iloc[split:]
    y_raw_f = y_raw.iloc[split:]

    rf_f = RandomForestRegressor(n_estimators=300, max_depth=8,
                                 min_samples_leaf=4, random_state=42)
    rf_f.fit(X_tr_f, y_tr_f)
    pred_rf_f = np.expm1(rf_f.predict(X_te_f))

    gb_f = build_model(lgbm=True)
    gb_f.fit(X_tr_f, y_tr_f)
    pred_gb_f = np.expm1(gb_f.predict(X_te_f))

    obs_f = y_raw_f.values
    wk_f  = sub["date"].iloc[split:].values

    model_outputs[state] = {
        "rf": rf_f, "gb": gb_f, "features": feats,
        "X_train": X_tr_f, "X_test": X_te_f,
        "y_test_raw": obs_f, "holdout_wk": wk_f,
        "holdout_rf": pred_rf_f, "holdout_gb": pred_gb_f,
        "cv_metrics": cv_mean,
    }

    metrics_rows.append({
        "State":       state,
        "RF_CV_R2":    cv_mean["RF_R2"],   "RF_CV_MAE":  cv_mean["RF_MAE"],
        "RF_CV_RMSE":  cv_mean["RF_RMSE"], "RF_CV_d":    cv_mean["RF_d"],
        "RF_CV_PBIAS": cv_mean["RF_PBIAS"],"RF_CV_MAPE": cv_mean["RF_MAPE"],
        "GB_CV_R2":    cv_mean["GB_R2"],   "GB_CV_MAE":  cv_mean["GB_MAE"],
        "GB_CV_RMSE":  cv_mean["GB_RMSE"], "GB_CV_d":    cv_mean["GB_d"],
        "GB_CV_PBIAS": cv_mean["GB_PBIAS"],"GB_CV_MAPE": cv_mean["GB_MAPE"],
        "Holdout_RF_d":    willmott_d(obs_f, pred_rf_f),
        "Holdout_GB_d":    willmott_d(obs_f, pred_gb_f),
        "Holdout_RF_PBIAS":pbias(obs_f, pred_rf_f),
        "Holdout_GB_PBIAS":pbias(obs_f, pred_gb_f),
    })
    print(f"  {state}: RF R²={cv_mean['RF_R2']:.3f}  "
          f"RF d={cv_mean['RF_d']:.3f}  RF PBIAS={cv_mean['RF_PBIAS']:.1f}%  "
          f"| {MODEL_LABEL} R²={cv_mean['GB_R2']:.3f}")

df_metrics = pd.DataFrame(metrics_rows)
df_metrics.to_csv(os.path.join(RESULTS_DIR, "Table3_Model_Performance.csv"), index=False)

# ── Fig 10: Model performance dashboard ───────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
metric_pairs = [
    ("RF_CV_R2",    "GB_CV_R2",    "R² (Log Scale)",    "higher ↑"),
    ("RF_CV_MAE",   "GB_CV_MAE",   "MAE (cases)",        "lower ↓"),
    ("RF_CV_RMSE",  "GB_CV_RMSE",  "RMSE (cases)",       "lower ↓"),
    ("RF_CV_d",     "GB_CV_d",     "Willmott Index (d)", "higher ↑"),
    ("RF_CV_PBIAS", "GB_CV_PBIAS", "PBIAS (%)",          "±25% = Satisfactory"),
    ("RF_CV_MAPE",  "GB_CV_MAPE",  "MAPE (%)",           "lower ↓"),
]
for idx, (rf_col, gb_col, metric_label, note) in enumerate(metric_pairs):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    x  = np.arange(len(df_metrics))
    w  = 0.35
    ax.bar(x - w/2, df_metrics[rf_col], w, label="Random Forest",
           color=PALETTE["primary"], alpha=0.85)
    ax.bar(x + w/2, df_metrics[gb_col], w, label=MODEL_LABEL,
           color=PALETTE["secondary"], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics["State"], fontsize=9)
    ax.set_title(f"{metric_label}\n({note})", fontweight="bold")
    ax.set_ylabel(metric_label)
    if "PBIAS" in rf_col:
        ax.axhline(25,  ls="--", lw=1, color=PALETTE["accent"], alpha=0.7)
        ax.axhline(-25, ls="--", lw=1, color=PALETTE["accent"], alpha=0.7)
        ax.axhline(0,   ls="-",  lw=0.8, color="black")
    ax.legend(fontsize=8)
    ax.set_facecolor(PALETTE["light_bg"])

savefig("Fig10_Model_Performance_Dashboard.png")

# ── Fig 11: Holdout — predicted vs observed ───────────────────────────────────
plot_states = list(model_outputs.keys())[:3]
fig, axes   = plt.subplots(len(plot_states), 1, figsize=(16, 5 * len(plot_states)))
if len(plot_states) == 1:
    axes = [axes]

for ax, state in zip(axes, plot_states):
    out = model_outputs[state]
    ax.plot(out["holdout_wk"], out["y_test_raw"], color="black",
            lw=2.0, label="Observed", zorder=5)
    ax.plot(out["holdout_wk"], out["holdout_rf"], color=PALETTE["primary"],
            lw=1.8, ls="--", label="Random Forest")
    ax.plot(out["holdout_wk"], out["holdout_gb"], color=PALETTE["secondary"],
            lw=1.8, ls=":", label=MODEL_LABEL)
    ax.fill_between(out["holdout_wk"], out["holdout_rf"], out["holdout_gb"],
                    alpha=0.10, color="purple", label="Model spread")
    ax.set_title(
        f"{state}  |  RF: d={willmott_d(out['y_test_raw'], out['holdout_rf']):.3f}, "
        f"PBIAS={pbias(out['y_test_raw'], out['holdout_rf']):.1f}%",
        fontweight="bold", color=STATE_COLORS.get(state, "black"),
    )
    ax.set_ylabel("Weekly Cases")
    ax.legend()
    ax.set_facecolor(PALETTE["light_bg"])
axes[-1].set_xlabel("Date")
plt.tight_layout()
savefig("Fig11_Predicted_vs_Observed.png")
print("  ✓ Figures 10–11 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SHAP INTERPRETABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 9: SHAP Interpretability Analysis")
print("="*70)

if HAS_SHAP and model_outputs:

    # ── 9a. Train a single pooled RF on all states for global SHAP ────────────
    pool_df    = df.copy()
    pool_df["state_enc"] = pool_df["state"].map(
        {s: i for i, s in enumerate(states)}
    )
    pool_feats = [c for c in feature_cols if c in pool_df.columns] + ["state_enc"]
    pool_X = pool_df[pool_feats].fillna(0)
    pool_y = pool_df["log_cases"]

    split_pool = int(len(pool_X) * 0.8)
    rf_pool    = RandomForestRegressor(
        n_estimators=300, max_depth=10,
        min_samples_leaf=3, random_state=42, n_jobs=-1,
    )
    rf_pool.fit(pool_X.iloc[:split_pool], pool_y.iloc[:split_pool])

    sample_n    = min(600, len(pool_X) - split_pool)
    X_shap_pool = pool_X.iloc[split_pool: split_pool + sample_n]

    print(f"  Computing global SHAP values ({sample_n} samples)...")
    explainer_pool = shap.TreeExplainer(rf_pool)
    shap_pool      = explainer_pool.shap_values(X_shap_pool)

    # ── Fig 12a: Global SHAP beeswarm ─────────────────────────────────────────
    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_pool, X_shap_pool,
        feature_names=pool_feats,
        plot_type="dot", show=False, max_display=15,
    )

    plt.tight_layout()
    savefig("Fig12a_SHAP_Beeswarm_Global.png")

    # ── Fig 12b: Global SHAP mean |SHAP| bar ──────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_pool, X_shap_pool,
        feature_names=pool_feats,
        plot_type="bar", show=False, max_display=15,
    )

    plt.tight_layout()
    savefig("Fig12b_SHAP_Bar_Global.png")

    # ── SHAP importance table ─────────────────────────────────────────────────
    mean_abs_shap_global = pd.DataFrame({
        "feature":        pool_feats,
        "mean_abs_SHAP":  np.abs(shap_pool).mean(axis=0).round(5),
    }).sort_values("mean_abs_SHAP", ascending=False)
    mean_abs_shap_global.to_csv(
        os.path.join(RESULTS_DIR, "Table_SHAP_Global_Importance.csv"), index=False
    )
    print("\n  Global SHAP importance (top 10):")
    print(mean_abs_shap_global.head(10).to_string(index=False))

    # ── 9b. State-stratified SHAP beeswarms ───────────────────────────────────
    fig, axes_shap = plt.subplots(2, 3, figsize=(22, 14))
    axes_shap = axes_shap.flatten()
    state_shap_records = []

    for ax_idx, state in enumerate(states):
        out = model_outputs.get(state)
        if out is None:
            axes_shap[ax_idx].set_visible(False)
            continue

        X_te_state = out["X_test"].fillna(0)
        if len(X_te_state) < 5:
            axes_shap[ax_idx].set_visible(False)
            continue

        rf_state    = out["rf"]
        feats_state = out["features"]

        explainer_s = shap.TreeExplainer(rf_state)
        shap_s      = explainer_s.shap_values(X_te_state)

        mean_abs_s = pd.DataFrame({
            "feature":       feats_state,
            "mean_abs_SHAP": np.abs(shap_s).mean(axis=0).round(5),
            "state":         state,
        }).sort_values("mean_abs_SHAP", ascending=False)
        state_shap_records.append(mean_abs_s)

        top_n   = min(10, len(feats_state))
        top_idx = np.argsort(np.abs(shap_s).mean(axis=0))[::-1][:top_n]
        shap_top = shap_s[:, top_idx]
        X_top    = X_te_state.iloc[:, top_idx]

        ax = axes_shap[ax_idx]
        plt.sca(ax)
        shap.summary_plot(
            shap_top, X_top,
            feature_names=[feats_state[i] for i in top_idx],
            plot_type="dot", show=False, max_display=top_n,
            color_bar=False,
        )
        ax.set_title(f"{state} — Top {top_n} SHAP Features",
                     fontweight="bold", color=STATE_COLORS.get(state, "black"),
                     fontsize=11)
        ax.set_xlabel("SHAP value (impact on log-cases)")


    plt.tight_layout()
    savefig("Fig12c_SHAP_State_Stratified.png")

    if state_shap_records:
        all_state_shap = pd.concat(state_shap_records, ignore_index=True)
        all_state_shap.to_csv(
            os.path.join(RESULTS_DIR, "Table_SHAP_State_Stratified.csv"), index=False
        )

    # ── 9c. SHAP partial-dependence plots for top 4 global drivers ────────────
    top4_features = mean_abs_shap_global["feature"].head(4).tolist()
    fig, axes_pd  = plt.subplots(2, 2, figsize=(14, 10))
    axes_pd       = axes_pd.flatten()

    for ax_idx, feat in enumerate(top4_features):
        if feat not in pool_feats:
            continue
        feat_idx = pool_feats.index(feat)
        ax = axes_pd[ax_idx]
        plt.sca(ax)
        shap.dependence_plot(
            feat_idx, shap_pool, X_shap_pool,
            feature_names=pool_feats,
            ax=ax, show=False,
            dot_size=8, alpha=0.6,
        )
        ax.set_title(f"SHAP Dependence — {feat}", fontweight="bold")
        ax.set_facecolor(PALETTE["light_bg"])


    plt.tight_layout()
    savefig("Fig12d_SHAP_Partial_Dependence.png")

    # ── 9d. SHAP waterfall for highest-burden week (Ondo) ─────────────────────
    ondo_key = next((s for s in model_outputs if "Ondo" in s or "ondo" in s), None)
    if ondo_key:
        out_o       = model_outputs[ondo_key]
        X_te_o      = out_o["X_test"].fillna(0)
        obs_o       = out_o["y_test_raw"]
        peak_idx    = int(np.argmax(obs_o))
        exp_o       = shap.TreeExplainer(out_o["rf"])
        sv_peak     = exp_o(X_te_o.iloc[[peak_idx]])

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(sv_peak[0], max_display=12, show=False)

        plt.tight_layout()
        savefig("Fig12e_SHAP_Waterfall_Peak_Week.png")

    print("  ✓ Figures 12a–12e saved.")

    # ── 9e. Feature importance (classic RF) as fallback comparison ────────────
    fig, axes_fi = plt.subplots(2, 3, figsize=(20, 12))
    axes_fi = axes_fi.flatten()
    agg_imp = {}

    for ax_idx, state in enumerate(states):
        out = model_outputs.get(state)
        if out is None:
            axes_fi[ax_idx].set_visible(False)
            continue
        rf    = out["rf"]
        feats = out["features"]
        imps  = rf.feature_importances_
        df_imp = (
            pd.DataFrame({"Feature": feats, "Importance": imps})
            .sort_values("Importance", ascending=True).tail(10)
        )
        color = STATE_COLORS.get(state, "steelblue")
        bars  = axes_fi[ax_idx].barh(df_imp["Feature"], df_imp["Importance"],
                                     color=color, alpha=0.80)
        for bar, val in zip(bars, df_imp["Importance"]):
            axes_fi[ax_idx].text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=color, fontweight="bold",
            )
        axes_fi[ax_idx].set_title(f"{state} — Top 10 Features",
                                  fontweight="bold", color=color)
        axes_fi[ax_idx].set_xlabel("RF Impurity Importance")
        axes_fi[ax_idx].set_facecolor(PALETTE["light_bg"])
        for feat, imp in zip(feats, imps):
            agg_imp[feat] = agg_imp.get(feat, []) + [imp]


    plt.tight_layout()
    savefig("Fig12f_RF_Feature_Importance.png")

    agg_df = pd.DataFrame({
        "feature": list(agg_imp.keys()),
        "mean_RF_importance": [np.mean(v) for v in agg_imp.values()],
    }).sort_values("mean_RF_importance", ascending=False)
    agg_df.to_csv(os.path.join(RESULTS_DIR, "Table_RF_Aggregate_Importance.csv"),
                  index=False)
    print("  ✓ Figure 12f and feature importance tables saved.")

else:
    if not HAS_SHAP:
        print("  SHAP not available — install with: pip install shap")
    elif not model_outputs:
        print("  No model outputs to analyse.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 10: Bootstrap Confidence Intervals (B=500)")
print("="*70)

B = 500
np.random.seed(42)
bootstrap_records = []

fig, axes_boot = plt.subplots(3, 2, figsize=(16, 14))
axes_boot = axes_boot.flatten()

for ax_idx, state in enumerate(states):
    out = model_outputs.get(state)
    if out is None:
        axes_boot[ax_idx].set_visible(False)
        continue

    X_tr = out["X_train"].fillna(0).values
    X_te = out["X_test"].fillna(0).values
    y_tr = df[df["state"] == state].sort_values("date")["log_cases"].values[:len(X_tr)]
    y_te = out["y_test_raw"]

    point_pred = out["holdout_rf"]

    boot_preds    = np.zeros((B, len(X_te)))
    boot_r2       = np.zeros(B)
    boot_mae      = np.zeros(B)
    boot_pearsonr = np.zeros(B)

    for b in range(B):
        idx  = np.random.choice(len(X_tr), size=len(X_tr), replace=True)
        m    = RandomForestRegressor(n_estimators=100, max_depth=10,
                                     min_samples_leaf=2, random_state=b, n_jobs=1)
        m.fit(X_tr[idx], y_tr[idx])
        pred_b = np.expm1(m.predict(X_te)).clip(0)
        boot_preds[b]    = pred_b
        boot_r2[b]       = r2_score(y_te, pred_b)
        boot_mae[b]      = mean_absolute_error(y_te, pred_b)
        boot_pearsonr[b] = pearsonr(y_te, pred_b)[0]

    def ci(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    pred_lower = np.percentile(boot_preds, 2.5,  axis=0)
    pred_upper = np.percentile(boot_preds, 97.5, axis=0)

    r2_lo, r2_hi   = ci(boot_r2)
    mae_lo, mae_hi = ci(boot_mae)

    bootstrap_records.append({
        "state":             state,
        "R2_point":          round(r2_score(y_te, point_pred), 3),
        "R2_CI_low":         round(r2_lo, 3),
        "R2_CI_high":        round(r2_hi, 3),
        "MAE_point":         round(mean_absolute_error(y_te, point_pred), 2),
        "MAE_CI_low":        round(mae_lo, 2),
        "MAE_CI_high":       round(mae_hi, 2),
        "Pearson_r_point":   round(pearsonr(y_te, point_pred)[0], 3),
        "Pearson_r_CI_low":  round(ci(boot_pearsonr)[0], 3),
        "Pearson_r_CI_high": round(ci(boot_pearsonr)[1], 3),
    })

    ax = axes_boot[ax_idx]
    dates_te = out["holdout_wk"]
    ax.plot(dates_te, y_te, label="Observed", color="black", lw=1.5, zorder=3)
    ax.plot(dates_te, point_pred, label="Predicted (RF)",
            color=STATE_COLORS.get(state, "steelblue"), lw=1.2, ls="--", zorder=2)
    ax.fill_between(dates_te, pred_lower, pred_upper, alpha=0.25,
                    color=STATE_COLORS.get(state, "steelblue"),
                    label="95% Prediction Interval")
    ax.set_xlabel("Date")  # state label via legend
    ax.annotate(state, xy=(0.02, 0.96), xycoords="axes fraction",
                fontsize=9, fontweight="bold",
                color=STATE_COLORS.get(state, "black"),
                va="top")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Cases")
    ax.legend(fontsize=7)
    ax.set_facecolor(PALETTE["light_bg"])


plt.tight_layout()
savefig("Fig13_Bootstrap_Prediction_Intervals.png")

boot_df = pd.DataFrame(bootstrap_records)
boot_df.to_csv(os.path.join(RESULTS_DIR,
               "Table4_Bootstrap_CI.csv"), index=False)
print(f"\n  Bootstrap (B={B}) 95% CI results:")
print(boot_df[["state", "R2_point", "R2_CI_low", "R2_CI_high",
               "MAE_point", "MAE_CI_low", "MAE_CI_high"]].to_string(index=False))
print("  ✓ Figure 13 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — GRANGER CAUSALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 11: Granger Causality Tests")
print("="*70)

granger_df = pd.DataFrame()

if HAS_SM:
    granger_records = []
    gc_candidates   = [c for c in feature_cols
                       if any(kw in c.lower()
                              for kw in ["rain", "ndvi", "temp", "moisture", "humidity"])][:5]

    for state in states:
        d = df[df["state"] == state].sort_values("date").copy()
        d_valid = d[["cases"] + gc_candidates].dropna()
        if len(d_valid) < 52:
            continue

        adf_stat, adf_p, *_ = adfuller(d_valid["cases"].values, autolag="AIC")
        is_stationary = adf_p < 0.05
        if not is_stationary:
            d_valid = d_valid.diff().dropna()

        for var in gc_candidates:
            try:
                test_data = d_valid[["cases", var]].dropna()
                if len(test_data) < 30:
                    continue
                gc_res   = grangercausalitytests(test_data.values, maxlag=8, verbose=False)
                min_p    = min(gc_res[lag][0]["ssr_ftest"][1] for lag in range(1, 9))
                best_lag = min(gc_res, key=lambda lag: gc_res[lag][0]["ssr_ftest"][1])
                granger_records.append({
                    "state": state, "climate_var": var,
                    "best_lag_weeks": best_lag,
                    "F_statistic":    round(gc_res[best_lag][0]["ssr_ftest"][0], 3),
                    "min_p_value":    round(min_p, 4),
                    "significant":    min_p < 0.05,
                    "series_differenced": not is_stationary,
                    "interpretation": (
                        f"{var} Granger-preceded cases at lag {best_lag}w"
                        if min_p < 0.05 else "No significant temporal precedence"
                    ),
                })
            except Exception as e:
                print(f"    Granger failed {state}/{var}: {e}")

    granger_df = pd.DataFrame(granger_records)
    if not granger_df.empty:
        granger_df.to_csv(os.path.join(RESULTS_DIR,
                          "Table5_Granger_Causality.csv"), index=False)

        pivot_f = granger_df.pivot_table(
            index="state", columns="climate_var",
            values="F_statistic", aggfunc="max",
        )
        pivot_sig = granger_df.pivot_table(
            index="state", columns="climate_var",
            values="significant", aggfunc="max",
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot_f, ax=ax, cmap="YlOrRd", annot=True,
                    fmt=".2f", linewidths=0.5,
                    cbar_kws={"label": "F-statistic"})
        for r_idx, row_name in enumerate(pivot_f.index):
            for c_idx, col_name in enumerate(pivot_f.columns):
                val = pivot_sig.loc[row_name, col_name] \
                      if row_name in pivot_sig.index and col_name in pivot_sig.columns \
                      else False
                if val:
                    ax.text(c_idx + 0.5, r_idx + 0.2, "*",
                            ha="center", va="center",
                            fontsize=16, color="white", fontweight="bold")
        ax.set_xlabel("Climate Variable\n* p < 0.05")
        plt.xticks(rotation=30, ha="right")
        savefig("Fig14_Granger_Causality.png")
        print("  ✓ Figure 14 saved.")

        sig_gc = granger_df[granger_df["significant"]]
        if not sig_gc.empty:
            print("\n  Significant Granger pairs:")
            print(sig_gc[["state", "climate_var", "best_lag_weeks",
                           "F_statistic", "min_p_value"]].to_string(index=False))
else:
    print("  Skipped — statsmodels not available.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SARIMA BASELINE & HEAD-TO-HEAD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 12: SARIMA Baseline Model")
print("="*70)

sarima_records = []

if HAS_SM:
    fig_sar, axes_sar = plt.subplots(3, 2, figsize=(16, 14))
    axes_sar = axes_sar.flatten()

    for ax_idx, state in enumerate(states):
        d = df[df["state"] == state].sort_values("date").reset_index(drop=True)
        y = d["cases"].values.astype(float)
        if len(y) < 52:
            print(f"  {state}: insufficient data, skipped.")
            continue

        split  = int(len(y) * 0.8)
        y_tr, y_te = y[:split], y[split:]
        n_test = len(y_te)

        best_aic, best_order, best_sorder = np.inf, (1, 1, 1), (1, 1, 1, 52)
        for p in range(3):
            for d_val in range(2):
                for q in range(3):
                    for P in range(2):
                        for D in range(2):
                            for Q in range(2):
                                try:
                                    m = SARIMAX(
                                        y_tr,
                                        order=(p, d_val, q),
                                        seasonal_order=(P, D, Q, 52),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,
                                    ).fit(disp=False)
                                    if m.aic < best_aic:
                                        best_aic    = m.aic
                                        best_order  = (p, d_val, q)
                                        best_sorder = (P, D, Q, 52)
                                except Exception:
                                    pass

        try:
            best_model = SARIMAX(
                y_tr, order=best_order, seasonal_order=best_sorder,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)

            fc        = best_model.get_forecast(steps=n_test)
            sarima_pred = np.array(fc.predicted_mean).clip(0)
            ci_arr    = np.array(fc.conf_int(alpha=0.05))
            pi_lower  = ci_arr[:, 0].clip(0)
            pi_upper  = ci_arr[:, 1]

            s_r2   = r2_score(y_te, sarima_pred)
            s_mae  = mean_absolute_error(y_te, sarima_pred)
            s_rmse = np.sqrt(mean_squared_error(y_te, sarima_pred))
            s_mape_v = mape(y_te, sarima_pred)

            rf_row = next((p for p in metrics_rows if p["State"] == state), {})

            sarima_records.append({
                "state":          state,
                "SARIMA_order":   str(best_order),
                "SARIMA_seasonal":str(best_sorder),
                "Best_AIC":       round(best_aic, 1),
                "SARIMA_R2":      round(s_r2,   3),
                "SARIMA_MAE":     round(s_mae,  2),
                "SARIMA_RMSE":    round(s_rmse, 2),
                "SARIMA_MAPE_%":  round(s_mape_v, 2),
                "RF_R2":          rf_row.get("RF_CV_R2", "—"),
                "RF_MAE":         rf_row.get("RF_CV_MAE", "—"),
                "Delta_R2":       round(rf_row.get("RF_CV_R2", 0) - s_r2, 3)
                                  if isinstance(rf_row.get("RF_CV_R2"), float) else "—",
            })

            dates_te = d["date"].values[split:]
            plot_len = min(len(dates_te), len(sarima_pred), len(y_te))
            ax = axes_sar[ax_idx]
            ax.plot(dates_te[:plot_len], y_te[:plot_len],
                    label="Observed", color="black", lw=1.5, zorder=3)
            ax.plot(dates_te[:plot_len], sarima_pred[:plot_len],
                    label=f"SARIMA{best_order}×{best_sorder}",
                    color=PALETTE["secondary"], lw=1.2, ls="--")
            ax.fill_between(dates_te[:plot_len],
                            pi_lower[:plot_len], pi_upper[:plot_len],
                            alpha=0.2, color=PALETTE["secondary"],
                            label="95% Prediction Interval")
            delta = sarima_records[-1]["Delta_R2"]
            ax.annotate(
                f"{state}  |  SARIMA R²={s_r2:.3f}  vs RF R²={rf_row.get('RF_CV_R2','—')}  ΔR²={delta}",
                xy=(0.02, 0.96), xycoords="axes fraction",
                fontsize=7, fontweight="bold", va="top",
                color=STATE_COLORS.get(state, "black"),
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Weekly Cases")
            ax.legend(fontsize=7)
            ax.set_facecolor(PALETTE["light_bg"])
            print(f"  {state}: SARIMA{best_order}×{best_sorder}  "
                  f"AIC={best_aic:.1f}  R²={s_r2:.3f}  MAE={s_mae:.1f}  "
                  f"ΔR²(RF−SARIMA)={delta}")

        except Exception as e:
            print(f"  SARIMA fitting failed for {state}: {e}")


    plt.tight_layout()
    savefig("Fig15_SARIMA_Forecasts.png")

    if sarima_records:
        sarima_df = pd.DataFrame(sarima_records)
        sarima_df.to_csv(os.path.join(RESULTS_DIR,
                         "Table6_SARIMA_vs_RF.csv"), index=False)

        valid_cmp = sarima_df.dropna(subset=["SARIMA_R2", "RF_R2"])
        if not valid_cmp.empty:
            x, w   = np.arange(len(valid_cmp)), 0.35
            fig, ax = plt.subplots(figsize=(11, 6))
            b1 = ax.bar(x - w/2, pd.to_numeric(valid_cmp["SARIMA_R2"], errors="coerce"),
                        w, label="SARIMA (climate-naive)",
                        color=PALETTE["secondary"], alpha=0.85, edgecolor="white")
            b2 = ax.bar(x + w/2, pd.to_numeric(valid_cmp["RF_R2"], errors="coerce"),
                        w, label="Random Forest (climate-driven)",
                        color=PALETTE["primary"], alpha=0.85, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(valid_cmp["state"].tolist())
            ax.set_ylabel("R² on Test Set")
            ax.axhline(0, color="black", lw=0.8)

            ax.legend()
            ax.set_facecolor(PALETTE["light_bg"])
            for bar in [*b1, *b2]:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            savefig("Fig16_SARIMA_vs_RF_Comparison.png")
        print("  ✓ Figures 15–16 saved.")
else:
    print("  Skipped — statsmodels not available.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — OUTBREAK DETECTION (EARLY WARNING SYSTEM)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 13: Outbreak Detection — Early Warning System")
print("="*70)

outbreak_records = []
fig, axes_ew = plt.subplots(3, 2, figsize=(16, 14))
axes_ew = axes_ew.flatten()

for ax_idx, state in enumerate(states):
    d = df[df["state"] == state].sort_values("date").copy()
    roll_mean = d["cases"].rolling(window=52, min_periods=26).mean()
    roll_std  = d["cases"].rolling(window=52, min_periods=26).std()
    threshold = roll_mean + 1.5 * roll_std
    d["alert"] = (d["cases"] > threshold).astype(int)

    alerts = d[d["alert"] == 1]
    outbreak_records.append({
        "state":              state,
        "total_alert_weeks":  len(alerts),
        "alert_rate_%":       round(len(alerts) / len(d) * 100, 1),
        "peak_date":          str(d.loc[d["cases"].idxmax(), "date"])[:10],
        "peak_cases":         int(d["cases"].max()),
    })

    ax = axes_ew[ax_idx]
    ax.fill_between(d["date"], d["cases"], alpha=0.45,
                    color=STATE_COLORS.get(state, "steelblue"), label="Weekly Cases")
    ax.plot(d["date"], threshold, color="red", lw=1.5,
            ls="--", label="Alert Threshold (mean + 1.5 SD)")
    ax.scatter(alerts["date"], alerts["cases"],
               color="red", zorder=5, s=18, label="Outbreak Alert")
    ax.annotate(f"{state}  ({len(alerts)} alert weeks)",
                xy=(0.02, 0.96), xycoords="axes fraction",
                fontsize=9, fontweight="bold",
                color=STATE_COLORS.get(state, "steelblue"), va="top")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cases")
    ax.legend(fontsize=7)
    ax.set_facecolor(PALETTE["light_bg"])


plt.tight_layout()
savefig("Fig17_Outbreak_Detection.png")

outbreak_df = pd.DataFrame(outbreak_records)
outbreak_df.to_csv(os.path.join(RESULTS_DIR,
                   "Table7_Outbreak_Alerts.csv"), index=False)
print("\n  Outbreak alert summary:")
print(outbreak_df.to_string(index=False))
print("  ✓ Figure 17 saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — MANUSCRIPT TABLES EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 14: Manuscript Tables Export")
print("="*70)

# ── Consolidated summary table (Table 8) ──────────────────────────────────────
summary_rows = []
for row in bootstrap_records:
    state = row["state"]
    perf  = next((p for p in metrics_rows if p["State"] == state), {})
    gc_sig = (granger_df[granger_df["state"] == state]["significant"].any()
              if not granger_df.empty else "N/A")
    total_cases = int(df[df["state"] == state]["cases"].sum())
    mean_wk     = round(df[df["state"] == state]["cases"].mean(), 1)
    peak_wk     = int(df[df["state"] == state]["cases"].max())

    naive_rmse = None
    for out_state in model_outputs:
        if out_state == state:
            obs = model_outputs[out_state]["y_test_raw"]
            rf_pred = model_outputs[out_state]["holdout_rf"]
            naive_rmse_val = np.sqrt(mean_squared_error(
                obs,
                np.full_like(obs, float(df[df["state"] == state]["cases"].mean()))
            ))
            rf_rmse_val = np.sqrt(mean_squared_error(obs, rf_pred))
            skill = round((naive_rmse_val - rf_rmse_val) / naive_rmse_val * 100, 1) \
                    if naive_rmse_val > 0 else "—"
            break
    else:
        skill = "—"

    summary_rows.append({
        "State":                  state,
        "Total Cases (2018–25)":  f"{total_cases:,}",
        "Mean Cases/Week":        mean_wk,
        "Peak Week Cases":        peak_wk,
        "RF R² [95% CI]":
            f"{row['R2_point']} [{row['R2_CI_low']}, {row['R2_CI_high']}]",
        "RF MAE [95% CI]":
            f"{row['MAE_point']} [{row['MAE_CI_low']}, {row['MAE_CI_high']}]",
        "RF Willmott d":          perf.get("RF_CV_d", "—"),
        "RF PBIAS (%)":           perf.get("RF_CV_PBIAS", "—"),
        "Skill vs Naïve (%)":     skill,
        "Granger Sig. (p<0.05)":  "Yes†" if gc_sig is True else ("No" if gc_sig is False else gc_sig),
    })

summary_table = pd.DataFrame(summary_rows)
summary_table.to_csv(os.path.join(RESULTS_DIR,
                     "Table8_Consolidated_Summary.csv"), index=False)
print("  ✓ Table 8 — Consolidated summary saved.")

print(f"""
  Tables saved to ./{RESULTS_DIR}/
    Table1_Descriptive_Stats.csv
    Table2_Spearman_Lag_Results.csv
    Table2b_Best_Lag_Summary.csv
    Table3_Model_Performance.csv
    Table4_Bootstrap_CI.csv
    Table5_Granger_Causality.csv
    Table6_SARIMA_vs_RF.csv
    Table7_Outbreak_Alerts.csv
    Table8_Consolidated_Summary.csv
    Table_NB_Regression.csv
    Table_SHAP_Global_Importance.csv
    Table_SHAP_State_Stratified.csv
    Table_RF_Aggregate_Importance.csv
""")




# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — ELEVATION SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 15: Elevation Sensitivity Analysis")
print("="*70)

elev_col = next((c for c in df.columns if "elev" in c.lower()), None)

if elev_col is None:
    print(f"  No elevation column found — skipping sensitivity analysis.")
    print(f"  (Checked columns: {[c for c in df.columns if c not in EXCLUDE_COLS]})")
else:
    print(f"  Elevation column identified: '{elev_col}'")

    feature_cols_no_elev = [c for c in feature_cols if c != elev_col]
    elev_sensitivity     = []
    shap_comparison      = []

    for state in states:
        sub   = df[df["state"] == state].sort_values("date").copy()

        # ── WITH elevation ────────────────────────────────────────────────────
        feats_with = [f for f in feature_cols     if f in sub.columns]
        feats_sans = [f for f in feature_cols_no_elev if f in sub.columns]

        for label, feats in [("With Elevation", feats_with),
                              ("Without Elevation", feats_sans)]:
            X = sub[feats].fillna(0)
            y = sub["log_cases"]
            if len(X) < 40:
                continue

            split   = int(len(X) * 0.8)
            X_tr, X_te = X.iloc[:split], X.iloc[split:]
            y_tr, y_te = y.iloc[:split], y.iloc[split:]
            y_raw_te   = sub["cases"].iloc[split:].values

            rf = RandomForestRegressor(
                n_estimators=300, max_depth=10,
                min_samples_leaf=3, random_state=42, n_jobs=-1,
            )
            rf.fit(X_tr, y_tr)
            pred_log = rf.predict(X_te)
            pred_raw = np.expm1(pred_log).clip(0)

            r2_log  = r2_score(y_te, pred_log)
            mae_raw = mean_absolute_error(y_raw_te, pred_raw)
            rmse_raw= np.sqrt(mean_squared_error(y_raw_te, pred_raw))
            d_val   = willmott_d(y_raw_te, pred_raw)
            pb      = pbias(y_raw_te, pred_raw)

            elev_sensitivity.append({
                "state":   state,
                "model":   label,
                "R2_log":  round(r2_log,  3),
                "MAE":     round(mae_raw, 2),
                "RMSE":    round(rmse_raw,2),
                "Willmott_d": round(d_val, 3),
                "PBIAS_%": round(pb,       2),
            })

            # SHAP importance for this config
            if HAS_SHAP and len(X_te) >= 5:
                exp  = shap.TreeExplainer(rf)
                sv   = exp.shap_values(X_te)
                mean_abs = np.abs(sv).mean(axis=0)
                top_feat = feats[int(np.argmax(mean_abs))]
                top_shap = round(float(mean_abs.max()), 5)
                shap_comparison.append({
                    "state":         state,
                    "model":         label,
                    "top_feature":   top_feat,
                    "top_mean_SHAP": top_shap,
                })

        print(f"  {state}: done")

    sens_df = pd.DataFrame(elev_sensitivity)
    sens_df.to_csv(
        os.path.join(RESULTS_DIR, "Table_Elevation_Sensitivity.csv"), index=False
    )

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n  Performance comparison — With vs. Without Elevation:")
    pivot_r2 = sens_df.pivot_table(
        index="state", columns="model", values="R2_log"
    )
    pivot_r2["Delta_R2 (with − without)"] = (
        pivot_r2.get("With Elevation", np.nan) -
        pivot_r2.get("Without Elevation", np.nan)
    )
    print(pivot_r2.round(3).to_string())

    if shap_comparison:
        shap_comp_df = pd.DataFrame(shap_comparison)
        shap_comp_df.to_csv(
            os.path.join(RESULTS_DIR, "Table_Elevation_SHAP_Comparison.csv"),
            index=False,
        )
        print("\n  Top SHAP feature — With vs. Without Elevation:")
        print(shap_comp_df.pivot_table(
            index="state", columns="model",
            values="top_feature", aggfunc="first",
        ).to_string())

    # ── Fig 18: Side-by-side bar chart — R² with vs without elevation ────────
    states_in   = sens_df["state"].unique().tolist()
    r2_with     = [sens_df[(sens_df["state"]==s) &
                            (sens_df["model"]=="With Elevation")]["R2_log"].values
                   for s in states_in]
    r2_without  = [sens_df[(sens_df["state"]==s) &
                            (sens_df["model"]=="Without Elevation")]["R2_log"].values
                   for s in states_in]
    r2_with    = [v[0] if len(v) else np.nan for v in r2_with]
    r2_without = [v[0] if len(v) else np.nan for v in r2_without]

    x, w   = np.arange(len(states_in)), 0.35
    fig, axes_sens = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: R² comparison
    ax = axes_sens[0]
    ax.bar(x - w/2, r2_with,    w, label="With Elevation",
           color=PALETTE["primary"], alpha=0.85, edgecolor="white")
    ax.bar(x + w/2, r2_without, w, label="Without Elevation",
           color=PALETTE["secondary"], alpha=0.85, edgecolor="white")
    for xi, (vw, vwo) in enumerate(zip(r2_with, r2_without)):
        if not np.isnan(vw):
            ax.text(xi - w/2, vw + 0.005, f"{vw:.3f}",
                    ha="center", va="bottom", fontsize=8)
        if not np.isnan(vwo):
            ax.text(xi + w/2, vwo + 0.005, f"{vwo:.3f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(states_in)
    ax.set_ylabel("R² (log-scale)")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.set_facecolor(PALETTE["light_bg"])

    # Panel B: delta R² (effect of removing elevation)
    ax = axes_sens[1]
    delta_r2 = [w - wo for w, wo in zip(r2_with, r2_without)]
    bar_cols  = [PALETTE["primary"] if d >= 0 else PALETTE["secondary"]
                 for d in delta_r2]
    bars = ax.bar(states_in, delta_r2, color=bar_cols, alpha=0.85,
                  edgecolor="white")
    for bar, val in zip(bars, delta_r2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.002 if val >= 0 else -0.005),
                f"{val:+.3f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("ΔR² (With − Without Elevation)")
    ax.set_xlabel("State")
    ax.annotate("Positive = elevation improves model  |  "
                "Negative = elevation not needed",
                xy=(0.5, -0.13), xycoords="axes fraction",
                ha="center", fontsize=8, style="italic",
                color=PALETTE["neutral"])
    ax.set_facecolor(PALETTE["light_bg"])

    plt.tight_layout()
    savefig("Fig18_Elevation_Sensitivity.png")
    print("  ✓ Figure 18 saved.")
    print("  ✓ Tables saved → Table_Elevation_Sensitivity.csv, "
          "Table_Elevation_SHAP_Comparison.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PIPELINE COMPLETE")
print("="*70)

all_figs = sorted(f for f in os.listdir(FIGURES_DIR) if f.endswith(".png"))
all_tabs = sorted(f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv"))

print(f"\n  {len(all_figs)} figures in ./{FIGURES_DIR}/")
for f in all_figs:
    sz = os.path.getsize(os.path.join(FIGURES_DIR, f))
    print(f"    {f:55s}  {sz/1024:>6.0f} KB")

print(f"\n  {len(all_tabs)} tables in ./{RESULTS_DIR}/")
for f in all_tabs:
    sz = os.path.getsize(os.path.join(RESULTS_DIR, f))
    print(f"    {f:55s}  {sz/1024:>6.0f} KB")

print("""
  Sections completed:
    1   Data loading & integration
    2   Lag imputation (forward-fill + leading-row drop)
    3   Descriptive statistics
    4   Exploratory visualisations          (Figs 1–5)
    5   STL seasonal decomposition          (Fig  6)
    6   Spearman lagged cross-correlation   (Figs 7–8)
    7   Negative binomial GLM + forest plot (Fig  9)
    8   Random Forest + LightGBM/GBM        (Figs 10–11)
    9   SHAP interpretability analysis      (Figs 12a–12f)
   10   Bootstrap confidence intervals      (Fig  13)
   11   Granger causality tests             (Fig  14)
   12   SARIMA baseline comparison          (Figs 15–16)
   13   Outbreak early warning detection    (Fig  17)
   14   Manuscript tables export
   15   Elevation sensitivity analysis      (Fig  18)
""")
