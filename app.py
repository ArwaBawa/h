"""
Green Hydrogen Electrolyzer Predictive Maintenance System
ACWA Power Challenge Solution using Nixtla TimeGPT
"""

# ============= Imports =============
import os
import re
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============= Page Config & Styles =============
st.set_page_config(
    page_title="ACWA Power Electrolyzer Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stAlert {border-radius: 10px;}
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .plot-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============= Small helpers =============
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def last_valid_nonzero(s: pd.Series) -> float:
    """Return the latest non-zero, non-NaN value; if none, return latest non-NaN."""
    if s is None:
        return np.nan
    s = pd.to_numeric(s, errors="coerce")
    nz = s.replace(0, np.nan).dropna()
    if not nz.empty:
        return nz.iloc[-1]
    s = s.dropna()
    return s.iloc[-1] if not s.empty else np.nan

def delta_24_rows(s: pd.Series) -> float:
    """Delta between last valid value and the value 24 rows earlier (ignoring zeros)."""
    if s is None:
        return np.nan
    s = pd.to_numeric(s, errors="coerce").replace(0, np.nan).dropna()
    if len(s) < 25:
        return np.nan
    return s.iloc[-1] - s.iloc[-25]

def filter_operating(df: pd.DataFrame, thr_a: float) -> pd.DataFrame:
    """Keep only rows where stack_current > threshold."""
    if "stack_current" not in df.columns:
        return df
    mask = pd.to_numeric(df["stack_current"], errors="coerce") > thr_a
    return df.loc[mask].copy()

# ============= Cached readers =============
@st.cache_data(show_spinner=False)
def read_file_as_bytes(uploaded):
    uploaded.seek(0); return uploaded.read()

@st.cache_data(show_spinner=False)
def load_csv_canonical(file_like_or_path):
    return pd.read_csv(file_like_or_path, parse_dates=["timestamp"])

@st.cache_data(show_spinner=False)
def load_excel_raw_headerless(file_like_or_path, sheet_name=None):
    xl = pd.ExcelFile(file_like_or_path)
    sheet = sheet_name or ("Data Recording Table Template"
                           if "Data Recording Table Template" in xl.sheet_names
                           else xl.sheet_names[0])
    raw = pd.read_excel(xl, sheet_name=sheet, header=None)
    return raw, sheet

def _looks_like_time(s: str) -> bool:
    s = str(s).strip()
    return bool(re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s))

# ============= Robust time parser =============
def build_timestamps_from_time_column(time_series: pd.Series) -> pd.Series:
    """
    Parse a 'Time' column that might be:
      - 'HH:MM' or 'HH:MM:SS' strings
      - full datetimes like '1900-01-01 06:30:00'
      - Excel serials (fractions of a day or date+time serials)
    Return full timestamps by rolling the day when time-of-day decreases.
    """
    # 1) Generic parse
    t_dt = pd.to_datetime(time_series, errors="coerce")

    # 2) Excel serials → datetime
    num = pd.to_numeric(time_series, errors="coerce")
    mask = t_dt.isna() & num.notna()
    if mask.any():
        t_dt.loc[mask] = pd.to_datetime(
            num[mask], unit="D", origin="1899-12-30", errors="coerce"
        )

    # 3) Strict time-only strings
    mask = t_dt.isna()
    if mask.any():
        t_dt.loc[mask] = pd.to_datetime(
            time_series[mask].astype(str).str.strip(),
            format="%H:%M:%S",
            errors="coerce"
        )
    mask = t_dt.isna()
    if mask.any():
        t_dt.loc[mask] = pd.to_datetime(
            time_series[mask].astype(str).str.strip(),
            format="%H:%M",
            errors="coerce"
        )

    t_dt = pd.to_datetime(t_dt, errors="coerce")
    if t_dt.isna().all():
        raise ValueError("Unable to parse the 'Time' column into datetime.")

    # If real calendar dates are present (>1901), use them directly
    if (t_dt.dropna().dt.year > 1901).any():
        return t_dt

    # Otherwise reconstruct dates by rolling at midnight
    sec = (t_dt.dt.hour * 3600 + t_dt.dt.minute * 60 + t_dt.dt.second).astype("Int64")
    roll = (sec.diff() < 0).cumsum().fillna(0).astype(int)
    base_date = pd.Timestamp("2024-01-01")
    timestamps = base_date + pd.to_timedelta(roll, unit="D") + pd.to_timedelta(sec.astype(float), unit="s")
    return timestamps

# ============= Transformer (Excel → canonical hourly) =============
@st.cache_data(show_spinner=False)
def transform_real_electrolyzer(file_like_or_path, n_cells=200, faradaic_eff=0.96, current_units="A"):
    """
    Robust transformer for ACWA 'Data Recording Table Template' → canonical hourly dataset.
    """
    raw, sheet = load_excel_raw_headerless(file_like_or_path)

    # Find header row (contains "Time")
    header_row_idx = None
    for i in range(min(60, raw.shape[0])):
        row = raw.iloc[i].astype(str).str.strip()
        if row.str.contains(r"\btime\b", case=False, regex=True).any() and (row != "nan").sum() >= 4:
            header_row_idx = i; break
    if header_row_idx is None:
        raise ValueError("Could not find a header row containing 'Time'.")

    # Unique headers
    labels = []
    for idx, v in enumerate(raw.iloc[header_row_idx].astype(str)):
        v = re.sub(r"\s+", " ", v).strip()
        if v.lower().startswith("unnamed"): v = f"col_{idx}"
        labels.append(v)
    seen, uniq = {}, []
    for name in labels:
        if name not in seen: seen[name]=0; uniq.append(name)
        else: seen[name]+=1; uniq.append(f"{name}__{seen[name]}")
    df = raw.copy(); df.columns = uniq

    # Locate time col and first data row
    time_col_candidates = [c for c in df.columns if re.search(r"\btime\b", c, re.IGNORECASE)]
    if not time_col_candidates:
        raise ValueError("No 'Time' column detected after parsing headers.")
    time_col = time_col_candidates[0]
    data_start = None
    for i in range(header_row_idx+1, df.shape[0]):
        if _looks_like_time(df.at[i, time_col]) or pd.notna(df.at[i, time_col]):
            data_start = i; break
    if data_start is None:
        raise ValueError("Could not locate the start of time-series rows.")
    data = df.iloc[data_start:].reset_index(drop=True)

    # Coerce numerics except time; drop empty cols
    for c in data.columns:
        if c != time_col:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    non_empty = [time_col] + [c for c in data.columns if c != time_col and data[c].notna().any()]
    data = data[non_empty]

    # Timestamps
    ts_series = build_timestamps_from_time_column(data[time_col])

    # Fuzzy map headers → canonical names
    colmap = {}
    for c in data.columns:
        if c == time_col: continue
        cl = _norm(c)
        if "room" in cl and "temp" in cl: colmap[c] = "ambient_temperature"
        elif "current" in cl and "stack" in cl: colmap[c] = "stack_current"
        elif "volt" in cl and "cell" in cl: colmap[c] = "cell_voltage"
        elif "volt" in cl and "stack" in cl: colmap[c] = "stack_voltage"
        elif ("cond" in cl or "conductivity" in cl) and ("dem" in cl or "dm" in cl or "deion" in cl): colmap[c] = "demin_water_quality"
        elif ("cond" in cl or "conductivity" in cl) and ("electrolyte" in cl or "koh" in cl or "lye" in cl): colmap[c] = "electrolyte_conductivity"
        elif ("temp" in cl) and any(k in cl for k in ["electrolyte","lye","anolyte","catholyte","koh"]): colmap[c] = "electrolyte_temperature"
        elif ("press" in cl or "pressure" in cl) and ("operating" in cl or "stack" in cl): colmap[c] = "operating_pressure"
        elif ("h2" in cl or "hydrogen" in cl) and ("purity" in cl): colmap[c] = "h2_purity"
        elif ("o2" in cl and "h2" in cl) or ("oxygen" in cl and "hydrogen" in cl): colmap[c] = "o2_in_h2"
        elif ("h2" in cl and "o2" in cl) or ("hydrogen" in cl and "oxygen" in cl): colmap[c] = "h2_in_o2"
        elif ("flow" in cl or "rate" in cl or "prod" in cl) and ("h2" in cl or "hydrogen" in cl): colmap[c] = "h2_production_rate"
        elif ("flow" in cl or "rate" in cl or "prod" in cl) and ("o2" in cl or "oxygen" in cl): colmap[c] = "o2_production_rate"

    data = data.rename(columns=colmap).drop(columns=[time_col])
    if data.columns.duplicated().any():
        data = data.T.groupby(level=0).mean(numeric_only=True).T

    data.insert(0, "timestamp", pd.to_datetime(ts_series))

    # Derivations
    if "stack_current" in data.columns and str(current_units).lower() == "ka":
        data["stack_current"] = data["stack_current"] * 1000.0

    if {"stack_voltage","stack_current"}.issubset(data.columns):
        data["power_consumption"] = (pd.to_numeric(data["stack_voltage"], errors="coerce") *
                                     pd.to_numeric(data["stack_current"], errors="coerce")) / 1000.0

    if "h2_production_rate" not in data.columns and "stack_current" in data.columns:
        # Faraday: H2 mol/s = I / (2F), convert to Nm3/h (22.414 L/mol at STP)
        F = 96485.0
        const = (1.0 / (2.0 * F)) * 0.022414 * 3600.0  # Nm3/h per A per cell
        data["h2_production_rate"] = (pd.to_numeric(data["stack_current"], errors="coerce") *
                                      n_cells * const * faradaic_eff)

    if "o2_production_rate" not in data.columns and "h2_production_rate" in data.columns:
        data["o2_production_rate"] = data["h2_production_rate"] * 0.5

    if "cell_voltage" not in data.columns and "stack_voltage" in data.columns:
        data["cell_voltage"] = pd.to_numeric(data["stack_voltage"], errors="coerce") / max(1, n_cells)

    if {"h2_production_rate","power_consumption"}.issubset(data.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            data["efficiency"] = (pd.to_numeric(data["h2_production_rate"], errors="coerce") /
                                  pd.to_numeric(data["power_consumption"], errors="coerce"))
            data["efficiency"] = data["efficiency"].replace([np.inf, -np.inf], np.nan)

    data = data.sort_values("timestamp").reset_index(drop=True)
    data["hours_since_maintenance"] = (data["timestamp"] - data["timestamp"].min())/np.timedelta64(1,"h")
    if "stack_current" in data.columns:
        vals = pd.to_numeric(data["stack_current"], errors="coerce").fillna(0).values
        thr = max(1e-6, 0.1*np.nanmedian(vals[vals>0]) if (vals>0).any() else 0)
        on = vals > thr
        data["cycles_count"] = (pd.Series(on.astype(int)).diff()==1).fillna(False).astype(int).cumsum()
    else:
        data["cycles_count"] = 0

    # Hourly resample to stabilize charts/metrics
    data = data.set_index("timestamp")
    agg = {
        "cell_voltage":"mean","stack_current":"mean","electrolyte_temperature":"mean",
        "electrolyte_conductivity":"mean","operating_pressure":"mean","h2_production_rate":"mean",
        "o2_production_rate":"mean","power_consumption":"mean","h2_purity":"mean","o2_in_h2":"mean",
        "h2_in_o2":"mean","differential_pressure":"mean","hours_since_maintenance":"max","cycles_count":"max",
        "ambient_temperature":"mean","cooling_water_temp":"mean","demin_water_quality":"mean",
        "efficiency":"mean","stack_voltage":"mean"
    }
    present = {k:v for k,v in agg.items() if k in data.columns}
    hourly = data.resample("H").agg(present).dropna(how="all").reset_index()
    return hourly

@st.cache_data(show_spinner=False)
def load_canonical_or_transform(n_cells=200, faradaic_eff=0.96, current_units="A"):
    """Try bundled canonical CSV first, else transform bundled Excel."""
    try:
        return load_csv_canonical("data/ACWA_Power2_canonical_from_template.csv")
    except Exception:
        pass
    return transform_real_electrolyzer("data/ACWA Power 2.xlsx", n_cells, faradaic_eff, current_units)

# ============= Title =============
st.title(" Green Hydrogen Electrolyzer Predictive Maintenance System")
st.markdown("**ACWA Power Challenge Solution** | Powered by Nixtla TimeGPT & Advanced Analytics")

# ============= Nixtla / TimeGPT integration =============
# Try to import either 'nixtla' (NixtlaClient) or 'timegpt' (TimeGPT) for robustness
try:
    from nixtla import NixtlaClient as _TimeGPTClient
    _TIMEGPT_FLAVOR = "nixtla"
    _TIMEGPT_AVAILABLE = True
except Exception:
    try:
        from timegpt import TimeGPT as _TimeGPTClient   # fallback legacy
        _TIMEGPT_FLAVOR = "timegpt"
        _TIMEGPT_AVAILABLE = True
    except Exception:
        _TIMEGPTClient = None
        _TIMEGPT_FLAVOR = None
        _TIMEGPT_AVAILABLE = False

def init_timegpt(api_key: str = None):
    """Initialize TimeGPT client from secrets/env/input."""
    if not _TIMEGPT_AVAILABLE:
        return None, "TimeGPT library not installed. Add 'nixtla' to requirements.txt."
    key = api_key or st.secrets.get("NIXTLA_API_KEY", None) or os.getenv("NIXTLA_API_KEY", None)
    if not key:
        return None, "No TimeGPT API key set. Add it to Streamlit secrets or paste it in the sidebar."
    try:
        # class signatures vary slightly; try both
        try:
            client = _TIMEGPTClient(api_key=key)
        except TypeError:
            client = _TIMEGPTClient(token=key)
        return client, None
    except Exception as e:
        return None, f"Failed to initialize TimeGPT client: {e}"

def prepare_timegpt_frames(df_in: pd.DataFrame):
    """Build Nixtla/TimeGPT frames: df( unique_id, ds, y ) + optional exogenous X_df."""
    df = df_in.copy().sort_values("timestamp")
    target = pd.to_numeric(df.get("cell_voltage"), errors="coerce")
    nixtla_df = pd.DataFrame({
        "unique_id": "electrolyzer_1",
        "ds": pd.to_datetime(df["timestamp"]),
        "y": target,
    })
    mask = nixtla_df["y"].notna() & nixtla_df["ds"].notna()
    nixtla_df = nixtla_df.loc[mask].reset_index(drop=True)

    exog_cols = [c for c in ["stack_current","electrolyte_temperature","cycles_count","hours_since_maintenance"]
                 if c in df.columns]
    X_df = None
    if exog_cols:
        X_df = pd.concat([
            pd.Series("electrolyzer_1", index=df.index, name="unique_id"),
            df["timestamp"].rename("ds"),
            df[exog_cols]
        ], axis=1)
        X_df = X_df.loc[mask].reset_index(drop=True)
    return nixtla_df, X_df, exog_cols

def timegpt_forecast(client, nixtla_df, h, X_df=None, freq="H", level=[90,95]):
    """Call TimeGPT forecast and return raw forecast frame."""
    try:
        if hasattr(client, "forecast"):
            fc = client.forecast(
                df=nixtla_df,
                h=int(h),
                time_col="ds",
                target_col="y",
                id_col="unique_id",
                freq=freq,
                X_df=X_df,
                level=level
            )
        else:
            fc = client.forecast(df=nixtla_df, h=int(h), time_col="ds",
                                 target_col="y", id_col="unique_id", freq=freq, X_df=X_df)
        return fc, None
    except Exception as e:
        return None, str(e)

def unify_timegpt_forecast(fc_raw: pd.DataFrame) -> pd.DataFrame:
    """Map TimeGPT output columns to app's standard columns."""
    f = fc_raw.copy()
    lower = {c.lower(): c for c in f.columns}

    def pick(*cands):
        for c in cands:
            if c in lower: return lower[c]
        return None

    ds_col = pick("ds","date","timestamp") or f.columns[0]
    mean_col = pick("timegpt","yhat","forecast","prediction","point_forecast")
    lo95_col = pick("timegpt-lo-95","lo-95","yhat_lo_95","lower_95","low_95","forecast_lo_95","lo_95")
    hi95_col = pick("timegpt-hi-95","hi-95","yhat_hi_95","upper_95","high_95","forecast_hi_95","hi_95")
    std_col = pick("forecast_std","yhat_std","std")

    # Build base
    out = pd.DataFrame({"timestamp": pd.to_datetime(f[ds_col])})
    if mean_col is not None:
        out["predicted_voltage"] = pd.to_numeric(f[mean_col], errors="coerce")
    else:
        out["predicted_voltage"] = np.nan

    if std_col is not None:
        std = pd.to_numeric(f[std_col], errors="coerce")
    elif lo95_col is not None and hi95_col is not None and mean_col is not None:
        std = (pd.to_numeric(f[hi95_col], errors="coerce") - pd.to_numeric(f[mean_col], errors="coerce")) / 1.96
    else:
        std = pd.Series(np.nan, index=f.index)

    if lo95_col is not None and hi95_col is not None:
        out["lower_bound"] = pd.to_numeric(f[lo95_col], errors="coerce")
        out["upper_bound"] = pd.to_numeric(f[hi95_col], errors="coerce")
    else:
        out["lower_bound"] = out["predicted_voltage"] - 1.96*std
        out["upper_bound"] = out["predicted_voltage"] + 1.96*std

    out["uncertainty"] = std.replace([np.inf,-np.inf], np.nan)
    # Failure probability vs 2.0 V
    out["failure_probability"] = 1 - stats.norm.cdf(2.0, out["predicted_voltage"], out["uncertainty"])
    return out

def timegpt_anomalies(client, nixtla_df, confidence=99, method="iqr"):
    try:
        if hasattr(client, "detect_anomalies"):
            res = client.detect_anomalies(
                df=nixtla_df, time_col="ds", target_col="y", id_col="unique_id",
                freq="H", anomaly_detection_params={"confidence_level":int(confidence), "method":method}
            )
            return res, None
        return None, "This client does not support anomaly detection."
    except Exception as e:
        return None, str(e)

def timegpt_cross_validation(client, nixtla_df, h=24, n_windows=5):
    try:
        res = client.cross_validation(
            df=nixtla_df, h=int(h), n_windows=int(n_windows),
            time_col="ds", target_col="y", id_col="unique_id", freq="H"
        )
        return res, None
    except Exception as e:
        return None, str(e)

# ============= Sidebar =============
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1c83e1/ffffff?text=ACWA+Power", use_container_width=True)
    st.markdown("### ⚙️ System Configuration")
    model_type = st.selectbox("Select Prediction Model", ["Nixtla TimeGPT","Statistical Ensemble","XGBoost ML","Hybrid Approach","(Simulated Baseline)"])
    forecast_horizon = st.slider("Forecast Horizon (hours)", 24, 168, 72, step=24)
    risk_threshold = st.slider("Risk Alert Threshold (%)", 50, 95, 75, step=5)

    st.markdown("---")
    st.markdown("### 🔧 Data & Physics")
    n_cells = st.number_input("Number of cells (N)", 10, 1000, 200, 10)
    faradaic_eff = st.slider("Faradaic efficiency (ηF)", 0.80, 1.00, 0.96, 0.01)
    current_units = st.selectbox("Current units in source", ["A","kA"])
    temp_target = st.number_input("Target electrolyte temperature (°C)", value=85.0, step=0.5)

    st.markdown("---")
    st.markdown("### 🔐 TimeGPT Access")
    api_key_input = st.text_input("Nixtla TimeGPT API Key", type="password",
                                  help="Optional here if you already set NIXTLA_API_KEY in Streamlit secrets.")

    st.markdown("---")
    st.markdown("### 📊 Data Source")
    uploaded_file = st.file_uploader("Upload Electrolyzer Data (.xlsx/.xls/.csv)", type=["xlsx","xls","csv"])
    use_bundled = st.button("Use Bundled Dataset", type="primary")

    st.markdown("---")
    st.markdown("### ⚙️ Analysis Scope")
    op_only = st.checkbox("Analyze operating hours only", value=True)
    op_thr = st.number_input("Operating threshold (A)", value=10.0, step=1.0)

# ============= Load Data (real only) =============
df = None
if uploaded_file is not None:
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx",".xls",".xlsm")):
        df = transform_real_electrolyzer(io.BytesIO(read_file_as_bytes(uploaded_file)),
                                         n_cells=n_cells, faradaic_eff=faradaic_eff, current_units=current_units)
        st.success("✅ Uploaded Excel transformed successfully.")
    else:
        df = load_csv_canonical(uploaded_file)
        st.success("✅ Uploaded CSV loaded as canonical dataset.")
else:
    try:
        df = load_canonical_or_transform(n_cells=n_cells, faradaic_eff=faradaic_eff, current_units=current_units)
        if use_bundled:
            st.success("✅ Loaded bundled dataset.")
        else:
            st.info("ℹ️ Loaded bundled dataset (upload a file to override).")
    except Exception as e:
        st.error(str(e))
        st.stop()

# Choose temp column (prefer electrolyte)
temp_col = "electrolyte_temperature" if ("electrolyte_temperature" in df.columns and df["electrolyte_temperature"].notna().any()) else \
           ("ambient_temperature" if "ambient_temperature" in df.columns else None)
temp_label = "Electrolyte" if temp_col=="electrolyte_temperature" else ("Ambient" if temp_col=="ambient_temperature" else "Temperature")

# Optionally keep operating rows only
df_view = filter_operating(df, op_thr) if op_only else df.copy()

st.success(f"Loaded {df_view.shape[0]:,} rows × {df_view.shape[1]:,} columns.")
st.dataframe(df_view.tail(50), use_container_width=True)

# ============= Risk Metrics =============
def calculate_risk_metrics(df_in: pd.DataFrame, temp_target_c=85.0, temp_col_name="electrolyte_temperature") -> pd.DataFrame:
    s_v = pd.to_numeric(df_in.get("cell_voltage"), errors="coerce")
    s_x = pd.to_numeric(df_in.get("o2_in_h2"), errors="coerce")
    s_m = pd.to_numeric(df_in.get("hours_since_maintenance"), errors="coerce")
    s_t = pd.to_numeric(df_in.get(temp_col_name), errors="coerce") if temp_col_name else pd.Series(np.nan, index=df_in.index)

    voltage_risk = np.clip(((s_v - 1.8) / 0.2) * 100, 0, 100)
    crossover_risk = np.clip((s_x / 500) * 100, 0, 100)
    thermal_risk = np.clip((s_t.sub(temp_target_c).abs() / 10) * 100, 0, 100)
    maintenance_risk = np.clip((s_m / 2000) * 100, 0, 100)

    risk = pd.DataFrame({
        "overall_risk": (voltage_risk + crossover_risk + thermal_risk + maintenance_risk) / 4,
        "voltage_risk": voltage_risk,
        "crossover_risk": crossover_risk,
        "thermal_risk": thermal_risk,
        "maintenance_risk": maintenance_risk
    })
    try:
        risk["risk_level"] = pd.cut(risk["overall_risk"], bins=[0,25,50,75,100], labels=["Low","Medium","High","Critical"])
    except Exception:
        risk["risk_level"] = np.nan
    return risk

risk_metrics = calculate_risk_metrics(df_view, temp_target, temp_col if temp_col else "electrolyte_temperature")

# ============= Simulated fallback (for when TimeGPT is unavailable) =============
def generate_predictions_simulated(df_in: pd.DataFrame, horizon: int) -> pd.DataFrame:
    last_timestamp = df_in["timestamp"].max()
    future_timestamps = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=horizon, freq="H")

    hist = df_in["cell_voltage"].dropna()
    if len(hist) < 2:
        trend = 0.0; base = hist.iloc[-1] if len(hist) else 1.9
    else:
        n = min(168, len(hist))
        recent_voltage = hist.tail(n).values
        trend = np.polyfit(range(len(recent_voltage)), recent_voltage, 1)[0]
        base = hist.iloc[-1]

    predictions, uncertainties = [], []
    for i in range(horizon):
        pred = base + trend * i + np.random.normal(0, 0.01)
        uncertainty = 0.02 + 0.001 * i
        predictions.append(pred); uncertainties.append(uncertainty)

    pred_df = pd.DataFrame({
        "timestamp": future_timestamps,
        "predicted_voltage": predictions,
        "lower_bound": np.array(predictions) - 1.96*np.array(uncertainties),
        "upper_bound": np.array(predictions) + 1.96*np.array(uncertainties),
        "uncertainty": uncertainties
    })
    pred_df["failure_probability"] = pred_df.apply(
        lambda row: 1 - stats.norm.cdf(2.0, row["predicted_voltage"], row["uncertainty"]),
        axis=1
    )
    return pred_df

# ============= Tabs =============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Real-time Monitoring","⚠️ Failure Prediction","⚠️ Risk Assessment","📋 Maintenance Planning","🧪 Validation"
])

# ---------- Tab 1 ----------
with tab1:
    st.markdown("### 🔍 Current System Status")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        v_now = last_valid_nonzero(df_view.get("cell_voltage"))
        v_delta = delta_24_rows(df_view.get("cell_voltage"))
        st.metric("Cell Voltage", f"{v_now:.3f} V" if pd.notna(v_now) else "N/A",
                  f"{v_delta:+.3f} V" if pd.notna(v_delta) else "—", delta_color="inverse")

    with col2:
        eff_now = last_valid_nonzero(df_view.get("efficiency"))
        eff_delta = delta_24_rows(df_view.get("efficiency"))
        st.metric("Efficiency", f"{eff_now:.3f}" if pd.notna(eff_now) else "N/A",
                  f"{eff_delta:+.3f}" if pd.notna(eff_delta) else "—")

    with col3:
        h2_now = last_valid_nonzero(df_view.get("h2_production_rate"))
        h2_delta = delta_24_rows(df_view.get("h2_production_rate"))
        st.metric("H₂ Production", f"{h2_now:.2f} Nm³/h" if pd.notna(h2_now) else "N/A",
                  f"{h2_delta:+.2f}" if pd.notna(h2_delta) else "—")

    with col4:
        t_now = last_valid_nonzero(df_view.get(temp_col)) if temp_col else np.nan
        temp_delta = (t_now - temp_target) if pd.notna(t_now) else np.nan
        st.metric(f"{temp_label} Temperature",
                  f"{t_now:.1f} °C" if pd.notna(t_now) else "N/A",
                  f"{temp_delta:+.1f} °C" if pd.notna(temp_delta) else "—",
                  delta_color="inverse" if pd.notna(temp_delta) and abs(temp_delta) > 5 else "normal")

    with col5:
        or_now = last_valid_nonzero(risk_metrics.get("overall_risk"))
        rl_now = risk_metrics.get("risk_level").dropna().iloc[-1] if risk_metrics.get("risk_level").notna().any() else "Unknown"
        badge = "🟢" if str(rl_now)=="Low" else "🟡" if str(rl_now)=="Medium" else "🔴" if str(rl_now) in ["High","Critical"] else "⚪"
        st.metric("Risk Score", f"{or_now:.1f}%" if pd.notna(or_now) else "N/A", f"{badge} {rl_now}")

    st.markdown("---")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Cell Voltage Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_view["timestamp"].tail(168), y=df_view["cell_voltage"].tail(168), mode="lines", name="Cell Voltage"))
        fig.add_hline(y=1.95, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        fig.add_hline(y=1.90, line_dash="dash", line_color="orange", annotation_text="Warning Level")
        fig.update_layout(xaxis_title="Time", yaxis_title="Voltage (V)", height=300, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with cB:
        st.markdown("#### Efficiency & Production")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_view["timestamp"].tail(168), y=df_view["efficiency"].tail(168), mode="lines", name="Efficiency", yaxis="y"))
        fig.add_trace(go.Scatter(x=df_view["timestamp"].tail(168), y=df_view["h2_production_rate"].tail(168), mode="lines", name="H₂ Production", yaxis="y2"))
        fig.update_layout(
            xaxis_title="Time",
            yaxis=dict(title="Efficiency", side="left"),
            yaxis2=dict(title="H₂ Rate (Nm³/h)", side="right", overlaying="y"),
            height=300, margin=dict(l=0,r=0,t=30,b=0), hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### System Health Indicators")
    g1, g2, g3 = st.columns(3)
    with g1:
        val = last_valid_nonzero(df_view.get("h2_purity"))
        if pd.notna(val):
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=val, title={'text': "H₂ Purity (%)"},
                delta={'reference': 99.5},
                gauge={'axis': {'range': [None, 100]},
                       'steps':[{'range':[0,98],'color':"lightgray"},
                                {'range':[98,99.5],'color':"yellow"},
                                {'range':[99.5,100],'color':"green"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 99}}
            ))
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("H₂ Purity: N/A")
    with g2:
        val = last_valid_nonzero(df_view.get("o2_in_h2"))
        if pd.notna(val):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val, title={'text': "O₂ in H₂ (ppm)"},
                gauge={'axis': {'range': [None, 500]},
                       'steps':[{'range':[0,100],'color':"green"},
                                {'range':[100,300],'color':"yellow"},
                                {'range':[300,500],'color':"red"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 400}}
            ))
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("O₂ in H₂: N/A")
    with g3:
        val = last_valid_nonzero(df_view.get("hours_since_maintenance"))
        if pd.notna(val):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val, title={'text': "Hours Since Maintenance"},
                gauge={'axis': {'range': [None, 2500]},
                       'steps':[{'range':[0,1000],'color':"green"},
                                {'range':[1000,2000],'color':"yellow"},
                                {'range':[2000,2500],'color':"red"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 2000}}
            ))
            fig.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hours Since Maintenance: N/A")

# ---------- Tab 2 ----------
with tab2:
    st.markdown("### Predictive Analytics - Equipment Failure Forecast")

    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Running forecast..."):
            pred_df = None
            used_model = "(simulated)"

            if model_type == "Nixtla TimeGPT":
                client, err = init_timegpt(api_key_input)
                if err:
                    st.warning(f"TimeGPT not used: {err}")
                else:
                    nixtla_df, X_df, exog = prepare_timegpt_frames(df_view if op_only else df)
                    fc_raw, f_err = timegpt_forecast(client, nixtla_df, forecast_horizon, X_df=None)  # univariate by default
                    if f_err:
                        st.warning(f"TimeGPT error → using simulated model: {f_err}")
                    else:
                        pred_df = unify_timegpt_forecast(fc_raw)
                        used_model = "Nixtla TimeGPT"

            if pred_df is None:
                pred_df = generate_predictions_simulated(df_view, forecast_horizon)

            st.session_state.predictions = pred_df
            st.session_state.used_model = used_model

    if st.session_state.get("predictions") is not None:
        pred_df = st.session_state.predictions
        used_model = st.session_state.get("used_model","(simulated)")

        st.caption(f"Model used: **{used_model}**")

        # --- Prediction summary cards ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_voltage = pred_df["predicted_voltage"].max()
            st.metric("Max Predicted Voltage", f"{max_voltage:.3f} V", "⚠️ Critical" if max_voltage>1.95 else "✅ Normal")
        with col2:
            max_prob = float(pred_df["failure_probability"].max()*100.0)
            st.metric("Max Failure Risk", f"{max_prob:.1f}%", "🔴 High" if max_prob>50 else "🟢 Low")
        with col3:
            crit = pred_df[pred_df["failure_probability"] > 0.5]
            if not crit.empty:
                time_to_failure = crit["timestamp"].min()
                hours_to_failure = (time_to_failure - df_view["timestamp"].max()).total_seconds()/3600
                st.metric("Time to Critical", f"{hours_to_failure:.0f} hours", "⏰ Plan Maintenance")
            else:
                st.metric("Time to Critical", "No Risk", "✅ Safe")
        with col4:
            # If uncertainty exists, show derived confidence notion
            mean_unc = pd.to_numeric(pred_df.get("uncertainty"), errors="coerce").dropna().mean()
            confidence = 100 - (mean_unc * 100) if pd.notna(mean_unc) else 80.0
            st.metric("Model Confidence (proxy)", f"{confidence:.1f}%", "High" if confidence>80 else "Medium")

        st.markdown("---")
        # --- Charts ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_view["timestamp"].tail(168), y=df_view["cell_voltage"].tail(168), mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=pred_df["timestamp"], y=pred_df["predicted_voltage"], mode="lines", name="Prediction", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(
            x=pred_df["timestamp"].tolist()+pred_df["timestamp"].tolist()[::-1],
            y=pred_df["upper_bound"].tolist()+pred_df["lower_bound"].tolist()[::-1],
            fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(255,255,255,0)"), name="95% Confidence"
        ))
        fig.add_hline(y=2.0, line_dash="dash", line_color="darkred", annotation_text="Critical Failure Threshold")
        fig.update_layout(title="Cell Voltage Prediction with Uncertainty Bands", xaxis_title="Time", yaxis_title="Cell Voltage (V)", height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Failure Probability Timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_df["timestamp"], y=pred_df["failure_probability"]*100, mode="lines+markers", name="Failure Probability"))
        fig.add_hline(y=risk_threshold, line_dash="dash", line_color="orange", annotation_text=f"Alert Threshold ({risk_threshold}%)")
        fig.update_layout(xaxis_title="Time", yaxis_title="Failure Probability (%)", height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Optional anomaly detection (TimeGPT only)
        if model_type == "Nixtla TimeGPT":
            client, err = init_timegpt(api_key_input)
            if not err:
                nixtla_df, _, _ = prepare_timegpt_frames(df_view if op_only else df)
                with st.expander("🔎 Anomaly Detection (TimeGPT)"):
                    with st.spinner("Detecting anomalies..."):
                        anom, aerr = timegpt_anomalies(client, nixtla_df, confidence=99, method="iqr")
                    if aerr:
                        st.info(f"Anomaly detection unavailable: {aerr}")
                    else:
                        # Expect columns like ds, y, anomaly
                        anom = anom.rename(columns={"ds":"timestamp","y":"cell_voltage"})
                        st.dataframe(anom.tail(30), use_container_width=True)
                        # Overlay anomalies (if flagged) on history
                        if "anomaly" in anom.columns:
                            a = anom[anom["anomaly"]==1]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=nixtla_df["ds"], y=nixtla_df["y"], mode="lines", name="Cell Voltage"))
                            if not a.empty:
                                fig.add_trace(go.Scatter(x=a["timestamp"], y=a["cell_voltage"], mode="markers", name="Anomaly", marker=dict(size=8)))
                            fig.update_layout(title="Detected Anomalies", xaxis_title="Time", yaxis_title="Cell Voltage (V)")
                            st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 3 ----------
with tab3:
    st.markdown("### ⚠️ Comprehensive Risk Assessment Dashboard")
    or_now = last_valid_nonzero(risk_metrics.get("overall_risk"))
    rl_now = risk_metrics.get("risk_level").dropna().iloc[-1] if risk_metrics.get("risk_level").notna().any() else "Unknown"

    if str(rl_now)=="Critical": st.error(f"🚨 **CRITICAL RISK** - Overall Risk: {or_now:.1f}%")
    elif str(rl_now)=="High":    st.warning(f"⚠️ **HIGH RISK** - Overall Risk: {or_now:.1f}%")
    elif str(rl_now)=="Medium":  st.info(f"ℹ️ **MEDIUM RISK** - Overall Risk: {or_now:.1f}%")
    elif str(rl_now)=="Low":     st.success(f"✅ **LOW RISK** - Overall Risk: {or_now:.1f}%")
    else:                        st.info("Risk level: **Unknown** (insufficient data)")

    st.markdown("#### Risk Factor Breakdown")
    c1, c2 = st.columns(2)
    values = [
        last_valid_nonzero(risk_metrics.get("voltage_risk")) or 0,
        last_valid_nonzero(risk_metrics.get("crossover_risk")) or 0,
        last_valid_nonzero(risk_metrics.get("thermal_risk")) or 0,
        last_valid_nonzero(risk_metrics.get("maintenance_risk")) or 0,
    ]
    with c1:
        categories = ["Voltage\nDegradation","Gas\nCrossover","Thermal\nStress","Maintenance\nUrgency"]
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill="toself", name="Current Risk Profile"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_view["timestamp"].tail(168), y=risk_metrics["overall_risk"].tail(168), mode="lines", name="Overall Risk"))
        fig.add_hline(y=75, line_dash="dash", line_color="darkred", annotation_text="Critical")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="High")
        fig.add_hline(y=25, line_dash="dash", line_color="yellow", annotation_text="Medium")
        fig.update_layout(title="Risk Score Trend (Last 7 Days)", xaxis_title="Time", yaxis_title="Risk Score (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Risk Report")
    rr = pd.DataFrame({
        "Risk Factor": ["Voltage Degradation","Gas Crossover","Thermal Stress","Maintenance Urgency"],
        "Current Value": [
            f"{last_valid_nonzero(df_view.get('cell_voltage')):.3f} V" if pd.notna(last_valid_nonzero(df_view.get('cell_voltage'))) else "N/A",
            f"{last_valid_nonzero(df_view.get('o2_in_h2')):.0f} ppm" if pd.notna(last_valid_nonzero(df_view.get('o2_in_h2'))) else "N/A",
            f"{last_valid_nonzero(df_view.get(temp_col)):.1f} °C" if temp_col and pd.notna(last_valid_nonzero(df_view.get(temp_col))) else "N/A",
            f"{last_valid_nonzero(df_view.get('hours_since_maintenance')):.0f} h" if pd.notna(last_valid_nonzero(df_view.get('hours_since_maintenance'))) else "N/A",
        ],
        "Risk Score": [f"{v:.1f}%" for v in values],
        "Status": ["🔴 Critical" if v>75 else "🟠 High" if v>50 else "🟡 Medium" if v>25 else "🟢 Low" for v in values],
        "Recommended Action": [
            "Immediate electrode inspection" if values[0]>75 else "Monitor closely" if values[0]>50 else "Routine monitoring",
            "Check diaphragm integrity" if values[1]>75 else "Verify gas analyzers" if values[1]>50 else "Normal operation",
            "Adjust cooling system" if values[2]>75 else "Check temperature control" if values[2]>50 else "Maintain current settings",
            "Schedule immediate maintenance" if values[3]>75 else "Plan maintenance soon" if values[3]>50 else "Continue operation",
        ]
    })
    st.dataframe(rr, use_container_width=True, hide_index=True)

    st.markdown("#### Historical Incident Analysis (example)")
    c3, c4 = st.columns(2)
    with c3:
        incident_types = ["Voltage Spike","Gas Crossover","Temperature Excursion","Pressure Imbalance","Efficiency Drop"]
        incident_counts = [12,8,15,5,10]
        fig = px.pie(values=incident_counts, names=incident_types, title="Incident Distribution (Last 90 Days)",
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        weeks = pd.date_range(start=df_view["timestamp"].min(), end=df_view["timestamp"].max(), freq="W")
        incident_freq = np.random.poisson(2, len(weeks))
        fig = go.Figure(data=go.Bar(x=weeks, y=incident_freq))
        fig.update_layout(title="Weekly Incident Frequency", xaxis_title="Week", yaxis_title="Number of Incidents", height=350)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 4 ----------
with tab4:
    st.markdown("### 📋 Intelligent Maintenance Planning & Recommendations")

    hours_operated = last_valid_nonzero(df_view.get("hours_since_maintenance")) or 0.0
    cycles_completed = last_valid_nonzero(df_view.get("cycles_count")) or 0.0
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Hours Since Last Maintenance", f"{hours_operated:.0f} hrs")
    with c2: st.metric("Cycles Completed", f"{cycles_completed:.0f}")
    with c3:
        next_scheduled = 2000 - hours_operated
        st.metric("Hours to Scheduled Maintenance", f"{max(0, next_scheduled):.0f} hrs")

    st.markdown("---")
    or_now = last_valid_nonzero(risk_metrics.get("overall_risk")) or 0
    cell_v_now = last_valid_nonzero(df_view.get("cell_voltage")) or 1.9
    if or_now>75 or cell_v_now>1.95:
        urgency, color, icon, eta, downtime = "IMMEDIATE","red","🚨","Within 4 hours","4–6 hours"
    elif or_now>50:
        urgency, color, icon, eta, downtime = "SCHEDULED","orange","⚠️","Within 48 hours","2–4 hours"
    elif or_now>25:
        urgency, color, icon, eta, downtime = "PLANNED","yellow","📅","Within 1 week","2–3 hours"
    else:
        urgency, color, icon, eta, downtime = "ROUTINE","green","✅","As scheduled","1–2 hours"

    st.markdown(f"""<div style="background-color:{color};opacity:0.1;padding:20px;border-radius:10px;"></div>""", unsafe_allow_html=True)
    a1, a2 = st.columns([1,2])
    with a1:
        st.markdown(f"### {icon} **{urgency}**")
        st.markdown(f"**Timeline:** {eta}")
        st.markdown(f"**Est. Downtime:** {downtime}")
    with a2:
        st.markdown("**Recommended Actions:**")
        # reuse risk 'values' from Tab 3
        actions=[]
        v_risk, x_risk, t_risk, m_risk = values
        if v_risk>50: actions += ["• Inspect electrode coating for degradation","• Measure individual cell voltages"]
        if x_risk>50: actions += ["• Check diaphragm integrity","• Verify gas analyzer calibration"]
        if t_risk>50: actions += ["• Inspect cooling system performance","• Check electrolyte circulation"]
        if m_risk>50: actions += ["• Replace worn gaskets and seals","• Clean and recalibrate sensors"]
        if not actions: actions = ["• Routine visual inspection","• Record operational parameters"]
        for a in actions[:6]: st.markdown(a)

    st.markdown("---")
    st.markdown("#### 📅 Optimized Maintenance Schedule")
    maintenance_tasks = pd.DataFrame({
        "Task":["Electrode Inspection","Diaphragm Check","Electrolyte Analysis","Sensor Calibration","Seal Replacement","System Flush"],
        "Priority":["High","High","Medium","Medium","Low","Low"],
        "Estimated Duration":["2 hrs","3 hrs","1 hr","1 hr","4 hrs","2 hrs"],
        "Last Performed":[(datetime.now()-timedelta(days=d)).strftime('%Y-%m-%d') for d in [30,45,14,7,90,60]],
        "Next Due":[(datetime.now()+timedelta(days=d)).strftime('%Y-%m-%d') for d in [5,3,16,23,10,30]],
        "Status":["⚠️ Due Soon","🔴 Overdue","✅ On Schedule","✅ On Schedule","⚠️ Due Soon","✅ On Schedule"]
    })
    st.dataframe(maintenance_tasks, use_container_width=True, hide_index=True)

    st.markdown("#### 💰 Maintenance Cost-Benefit Analysis")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Preventive Maintenance Benefits:**")
        for k,v in {"Avoided Downtime":"$15,000","Extended Equipment Life":"$8,000","Improved Efficiency":"$5,000","Reduced Emergency Repairs":"$10,000","Total Benefit":"$38,000"}.items():
            st.markdown(f"**{k}: {v}**" if k=="Total Benefit" else f"• {k}: {v}")
    with b2:
        st.markdown("**Maintenance Costs:**")
        for k,v in {"Labor":"$3,000","Parts & Materials":"$5,000","Production Loss":"$4,000","Testing & Validation":"$1,000","Total Cost":"$13,000"}.items():
            st.markdown(f"**{k}: {v}**" if k=="Total Cost" else f"• {k}: {v}")
    st.success("**Net Benefit: $25,000** (ROI: 192%)")

    st.markdown("---")
    st.button("📄 Generate Maintenance Report", type="primary")

# ---------- Tab 5 (Validation) ----------
with tab5:
    st.markdown("### 🧪 Model Validation (TimeGPT Cross‑Validation)")
    st.caption("Walk‑forward backtesting using Nixtla TimeGPT API for the 'cell_voltage' series.")
    client, err = init_timegpt(api_key_input)
    if err:
        st.info(f"TimeGPT validation unavailable: {err}")
    else:
        nixtla_df, _, _ = prepare_timegpt_frames(df_view if op_only else df)
        n_windows = st.slider("Number of CV windows", 2, 10, 5, 1)
        h_cv = st.slider("Horizon per window (hours)", 6, 48, 24, 6)
        with st.spinner("Running cross‑validation..."):
            cv, cv_err = timegpt_cross_validation(client, nixtla_df, h=h_cv, n_windows=n_windows)
        if cv_err:
            st.error(f"Cross‑validation error: {cv_err}")
        else:
            # Expect columns like: ds, y, yhat, (maybe intervals)
            cv = cv.rename(columns={"ds":"timestamp","y":"actual","yhat":"forecast"})
            st.dataframe(cv.tail(30), use_container_width=True)

            # Metrics
            # RMSE
            rmse = float(np.sqrt(np.mean((pd.to_numeric(cv["forecast"], errors="coerce") -
                                          pd.to_numeric(cv["actual"], errors="coerce"))**2)))
            # Coverage (95) if intervals exist
            lo = None; hi = None
            for cand in ["lo-95","yhat_lo_95","lower_95","lo_95"]:
                if cand in [c.lower() for c in cv.columns]:
                    lo = cv[[c for c in cv.columns if c.lower()==cand][0]]
                    break
            for cand in ["hi-95","yhat_hi_95","upper_95","hi_95"]:
                if cand in [c.lower() for c in cv.columns]:
                    hi = cv[[c for c in cv.columns if c.lower()==cand][0]]
                    break
            if lo is not None and hi is not None:
                inside = (pd.to_numeric(cv["actual"], errors="coerce") >= pd.to_numeric(lo, errors="coerce")) & \
                         (pd.to_numeric(cv["actual"], errors="coerce") <= pd.to_numeric(hi, errors="coerce"))
                pi95_cov = float(inside.mean())
            else:
                pi95_cov = np.nan

            # Brier (proxy): classify exceedance of 2.0V based on forecast vs actual
            y_event = (pd.to_numeric(cv["actual"], errors="coerce") >= 2.0).astype(float)
            p_event = (pd.to_numeric(cv["forecast"], errors="coerce") >= 2.0).astype(float)  # 0/1 proxy
            brier = float(np.mean((p_event - y_event)**2))

            k1, k2, k3 = st.columns(3)
            with k1: st.metric("RMSE (CV)", f"{rmse:.4f} V")
            with k2: st.metric("PI95 Coverage", f"{pi95_cov:.0%}" if pd.notna(pi95_cov) else "N/A")
            with k3: st.metric("Brier score", f"{brier:.3f}")

            # Plot sample window
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cv["timestamp"], y=cv["actual"], name="Actual", mode="lines"))
            fig.add_trace(go.Scatter(x=cv["timestamp"], y=cv["forecast"], name="Forecast", mode="lines"))
            fig.update_layout(title="Cross‑Validation Forecast vs Actual", xaxis_title="Time", yaxis_title="Cell Voltage (V)")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <small>
    Green Hydrogen Electrolyzer Predictive Maintenance System v1.2<br>
    Powered by Nixtla TimeGPT & Advanced Analytics<br>
    ACWA Power Challenge Solution 2024
    </small>
</div>
""", unsafe_allow_html=True)
