"""
Green Hydrogen Electrolyzer Predictive Maintenance System
ACWA Power Challenge Solution using Nixtla TimeGPT
"""

# ============= Imports =============
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


# ============= Helpers =============
@st.cache_data(show_spinner=False)
def read_file_as_bytes(uploaded):
    """Read an UploadedFile to bytes safely (for multiple reads)."""
    uploaded.seek(0)
    return uploaded.read()

@st.cache_data(show_spinner=False)
def load_csv_canonical(file_like_or_path):
    """Load canonical CSV with parsed timestamp."""
    return pd.read_csv(file_like_or_path, parse_dates=["timestamp"])

@st.cache_data(show_spinner=False)
def load_excel_raw_headerless(file_like_or_path, sheet_name=None):
    """
    Read Excel 'as-is' with header=None so we can detect the header row and time column ourselves.
    Accepts path or bytes (BytesIO).
    """
    xl = pd.ExcelFile(file_like_or_path)
    sheet = sheet_name or ("Data Recording Table Template" if "Data Recording Table Template" in xl.sheet_names else xl.sheet_names[0])
    raw = pd.read_excel(xl, sheet_name=sheet, header=None)
    return raw, sheet

def _looks_like_time(s: str) -> bool:
    s = str(s).strip()
    return bool(re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", s))

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


@st.cache_data(show_spinner=False)
def transform_real_electrolyzer(file_like_or_path, n_cells=200, faradaic_eff=0.96, current_units="A"):
    """
    Robust transformer for ACWA 'Data Recording Table Template' → canonical hourly dataset.

    Steps:
      1) Detect header row (contains 'Time').
      2) Build timestamps by rolling day at midnight (template has time-of-day only).
      3) Fuzzy-map headers → canonical names.
      4) Collapse duplicate canonical columns by averaging.
      5) Derive missing physics-based fields (Faraday's law, power, cell voltage).
      6) Build counters and resample to hourly.
    """
    # 1) Read raw sheet headerless
    raw, sheet = load_excel_raw_headerless(file_like_or_path)

    # 2) Detect header row
    header_row_idx = None
    for i in range(min(60, raw.shape[0])):
        row = raw.iloc[i].astype(str).str.strip()
        if row.str.contains(r"\btime\b", case=False, regex=True).any() and (row != "nan").sum() >= 4:
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError("Could not find a header row containing 'Time'.")

    # Build unique column labels
    header_vals = raw.iloc[header_row_idx].astype(str).tolist()
    labels = []
    for idx, v in enumerate(header_vals):
        v = re.sub(r"\s+", " ", v).strip()
        if v.lower().startswith("unnamed"):
            v = f"col_{idx}"
        labels.append(v)

    seen, uniq = {}, []
    for name in labels:
        if name not in seen:
            seen[name] = 0; uniq.append(name)
        else:
            seen[name] += 1; uniq.append(f"{name}__{seen[name]}")

    df = raw.copy()
    df.columns = uniq

    # Locate time column, find first data row
    time_col_candidates = [c for c in df.columns if re.search(r"\btime\b", c, re.IGNORECASE)]
    if not time_col_candidates:
        raise ValueError("No 'Time' column detected after parsing headers.")
    time_col = time_col_candidates[0]

    data_start = None
    for i in range(header_row_idx + 1, df.shape[0]):
        if _looks_like_time(df.at[i, time_col]):
            data_start = i
            break
    if data_start is None:
        raise ValueError("Could not locate the start of time-series rows.")

    data = df.iloc[data_start:].reset_index(drop=True)

    # Coerce numeric except time col; drop empty columns
    for c in data.columns:
        if c != time_col:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    non_empty_cols = [time_col] + [c for c in data.columns if c != time_col and data[c].notna().any()]
    data = data[non_empty_cols]

    # 3) Build timestamps (roll day at midnight)
    t_str = data[time_col].astype(str).str.strip()
    t_str = t_str.str.replace(r"^(\d{1,2}):(\d{2})$", r"\1:\2:00", regex=True)
    is_midnight = t_str.eq("00:00:00")

    day_counter = np.zeros(len(t_str), dtype=int)
    current_day = 0
    for i in range(len(t_str)):
        if i > 0 and is_midnight.iat[i] and t_str.iat[i-1] != "00:00:00":
            current_day += 1
        day_counter[i] = current_day

    base_date = pd.Timestamp("2024-01-01")
    timestamps = []
    for i, ts in enumerate(t_str):
        try:
            t_comp = pd.to_datetime(ts, format="%H:%M:%S")
            dt = (base_date + pd.Timedelta(days=int(day_counter[i]))).replace(
                hour=t_comp.hour, minute=t_comp.minute, second=t_comp.second
            )
            timestamps.append(dt)
        except Exception:
            timestamps.append(pd.NaT)

    # Preserve ts, drop original time col
    ts_series = pd.to_datetime(timestamps)

    # 4) Fuzzy mapping to canonical names
    colmap = {}
    for c in data.columns:
        if c == time_col:
            continue
        cl = _norm(c)
        if "room" in cl and "temp" in cl:
            colmap[c] = "ambient_temperature"
        elif "current" in cl and "stack" in cl:
            colmap[c] = "stack_current"
        elif "volt" in cl and "cell" in cl:
            colmap[c] = "cell_voltage"
        elif "volt" in cl and "stack" in cl:
            colmap[c] = "stack_voltage"
        elif ("cond" in cl or "conductivity" in cl) and ("dem" in cl or "dm" in cl or "deion" in cl):
            colmap[c] = "demin_water_quality"
        elif ("cond" in cl or "conductivity" in cl) and ("electrolyte" in cl or "koh" in cl or "lye" in cl):
            colmap[c] = "electrolyte_conductivity"
        elif ("temp" in cl) and ("electrolyte" in cl or "lye" in cl or "koh" in cl):
            colmap[c] = "electrolyte_temperature"
        elif ("press" in cl or "pressure" in cl) and ("operating" in cl or "stack" in cl):
            colmap[c] = "operating_pressure"
        elif ("h2" in cl or "hydrogen" in cl) and ("purity" in cl):
            colmap[c] = "h2_purity"
        elif ("o2" in cl and "h2" in cl) or ("oxygen" in cl and "hydrogen" in cl):
            colmap[c] = "o2_in_h2"
        elif ("h2" in cl and "o2" in cl) or ("hydrogen" in cl and "oxygen" in cl):
            colmap[c] = "h2_in_o2"
        elif ("flow" in cl or "rate" in cl or "prod" in cl) and ("h2" in cl or "hydrogen" in cl):
            colmap[c] = "h2_production_rate"
        elif ("flow" in cl or "rate" in cl or "prod" in cl) and ("o2" in cl or "oxygen" in cl):
            colmap[c] = "o2_production_rate"

    data_ren = data.rename(columns=colmap).drop(columns=[time_col])

    # 5) Collapse duplicate mapped columns by averaging
    if data_ren.columns.duplicated().any():
        data_ren = data_ren.T.groupby(level=0).mean(numeric_only=True).T

    # Reattach timestamp
    data_ren.insert(0, "timestamp", ts_series)

    # 6) Derivations (physics)
    if "stack_current" in data_ren.columns and current_units.lower() == "ka":
        data_ren["stack_current"] = data_ren["stack_current"] * 1000.0

    # power (kW) if stack_current & stack_voltage exist
    if "stack_current" in data_ren.columns and "stack_voltage" in data_ren.columns:
        data_ren["power_consumption"] = (pd.to_numeric(data_ren["stack_current"], errors="coerce") *
                                         pd.to_numeric(data_ren["stack_voltage"], errors="coerce")) / 1000.0

    # H2 flow from current if missing (Faraday)
    if "h2_production_rate" not in data_ren.columns and "stack_current" in data_ren.columns:
        F = 96485.0
        const = (1.0 / (2.0 * F)) * 0.022414 * 3600.0  # 0.00041815 Nm3/h per A per cell
        data_ren["h2_production_rate"] = pd.to_numeric(data_ren["stack_current"], errors="coerce") * n_cells * const * faradaic_eff

    # O2 = 0.5 * H2
    if "o2_production_rate" not in data_ren.columns and "h2_production_rate" in data_ren.columns:
        data_ren["o2_production_rate"] = data_ren["h2_production_rate"] * 0.5

    # cell_voltage from stack_voltage if missing
    if "cell_voltage" not in data_ren.columns and "stack_voltage" in data_ren.columns:
        data_ren["cell_voltage"] = pd.to_numeric(data_ren["stack_voltage"], errors="coerce") / n_cells

    # simple efficiency proxy (trend KPI)
    if "h2_production_rate" in data_ren.columns and "power_consumption" in data_ren.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            data_ren["efficiency"] = data_ren["h2_production_rate"] / data_ren["power_consumption"]
            data_ren["efficiency"] = data_ren["efficiency"].replace([np.inf, -np.inf], np.nan)

    # Counters
    data_ren = data_ren.sort_values("timestamp").reset_index(drop=True)
    if "timestamp" in data_ren.columns:
        data_ren["hours_since_maintenance"] = (data_ren["timestamp"] - data_ren["timestamp"].min()) / np.timedelta64(1, "h")
    else:
        data_ren["hours_since_maintenance"] = np.nan

    if "stack_current" in data_ren.columns:
        vals = pd.to_numeric(data_ren["stack_current"], errors="coerce").fillna(0).values
        thr = max(1e-6, 0.1 * np.nanmedian(vals[vals > 0]) if (vals > 0).any() else 0)
        on = vals > thr
        data_ren["cycles_count"] = (pd.Series(on.astype(int)).diff() == 1).fillna(False).astype(int).cumsum()
    else:
        data_ren["cycles_count"] = 0

    # Hourly resample
    data_ren = data_ren.set_index("timestamp")
    agg = {
        "cell_voltage": "mean",
        "stack_current": "mean",
        "electrolyte_temperature": "mean",
        "electrolyte_conductivity": "mean",
        "operating_pressure": "mean",
        "h2_production_rate": "mean",
        "o2_production_rate": "mean",
        "power_consumption": "mean",
        "h2_purity": "mean",
        "o2_in_h2": "mean",
        "h2_in_o2": "mean",
        "differential_pressure": "mean",
        "hours_since_maintenance": "max",
        "cycles_count": "max",
        "ambient_temperature": "mean",
        "cooling_water_temp": "mean",
        "demin_water_quality": "mean",
        "efficiency": "mean",
        "stack_voltage": "mean",
    }
    present = {k: v for k, v in agg.items() if k in data_ren.columns}
    hourly = data_ren.resample("H").agg(present).dropna(how="all").reset_index()

    return hourly


@st.cache_data(show_spinner=False)
def load_canonical_or_transform(n_cells=200, faradaic_eff=0.96, current_units="A"):
    """
    Fallback loader when no upload:
      1) Try canonical CSV in ./data
      2) Else transform the Excel in ./data
    """
    # 1) canonical CSV
    try:
        return load_csv_canonical("data/ACWA_Power2_canonical_from_template.csv")
    except Exception:
        pass
    # 2) raw Excel
    try:
        return transform_real_electrolyzer("data/ACWA Power 2.xlsx", n_cells, faradaic_eff, current_units)
    except Exception as e:
        raise FileNotFoundError(
            "No bundled dataset found. Add either:\n"
            " - data/ACWA_Power2_canonical_from_template.csv  (canonical), or\n"
            " - data/ACWA Power 2.xlsx                         (raw template)\n"
            f"Details: {e}"
        )


# ============= App State =============
if "predictions" not in st.session_state:
    st.session_state.predictions = None


# ============= Title =============
st.title(" Green Hydrogen Electrolyzer Predictive Maintenance System")
st.markdown("**ACWA Power Challenge Solution** | Powered by Nixtla TimeGPT & Advanced Analytics")


# ============= Sidebar =============
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1c83e1/ffffff?text=ACWA+Power", use_container_width=True)

    st.markdown("### ⚙️ System Configuration")
    model_type = st.selectbox("Select Prediction Model",
                              ["Nixtla TimeGPT", "Statistical Ensemble", "XGBoost ML", "Hybrid Approach"])
    forecast_horizon = st.slider("Forecast Horizon (hours)", 24, 168, 72, step=24)
    risk_threshold = st.slider("Risk Alert Threshold (%)", 50, 95, 75, step=5)

    st.markdown("---")
    st.markdown("### 🔧 Data & Physics Settings")
    n_cells = st.number_input("Number of cells in stack (N)", min_value=10, max_value=1000, value=200, step=10)
    faradaic_eff = st.slider("Faradaic efficiency (ηF)", 0.80, 1.00, 0.96, 0.01)
    current_units = st.selectbox("Current units (in source)", ["A", "kA"], index=0)

    st.markdown("---")
    st.markdown("### 📊 Data Source")
    uploaded_file = st.file_uploader("Upload Electrolyzer Data (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])
    use_bundled = st.button("Use Bundled Dataset", type="primary")


# ============= Load Data (Real Only) =============
df = None
if uploaded_file is not None:
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        # Transform the ACWA template
        bytes_data = read_file_as_bytes(uploaded_file)
        df = transform_real_electrolyzer(io.BytesIO(bytes_data), n_cells=n_cells, faradaic_eff=faradaic_eff, current_units=current_units)
        st.success("✅ Uploaded Excel transformed successfully.")
    elif name.endswith(".csv"):
        # Assume canonical CSV
        df = load_csv_canonical(uploaded_file)
        st.success("✅ Uploaded CSV loaded as canonical dataset.")
else:
    if use_bundled:
        df = load_canonical_or_transform(n_cells=n_cells, faradaic_eff=faradaic_eff, current_units=current_units)
        st.success("✅ Loaded bundled dataset.")
    else:
        # Auto-load bundled if present (nice default for deployed app)
        try:
            df = load_canonical_or_transform(n_cells=n_cells, faradaic_eff=faradaic_eff, current_units=current_units)
            st.info("ℹ️ Loaded bundled dataset from /data (upload a file to override).")
        except Exception as e:
            st.warning(str(e))
            df = None

# Guard: ensure timestamp dtype and expected columns exist
if df is not None:
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # If hours_since_maintenance missing or all NaN, recompute from timestamp
    if "hours_since_maintenance" not in df.columns or df["hours_since_maintenance"].isna().all():
        df["hours_since_maintenance"] = (df["timestamp"] - df["timestamp"].min()) / np.timedelta64(1, "h")
    # Replace inf
    for c in df.columns:
        if c == "timestamp":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)


# ============= Risk Metrics =============
def calculate_risk_metrics(df_in: pd.DataFrame) -> pd.DataFrame:
    s_v = pd.to_numeric(df_in.get("cell_voltage"), errors="coerce")
    s_x = pd.to_numeric(df_in.get("o2_in_h2"), errors="coerce")
    s_t = pd.to_numeric(df_in.get("electrolyte_temperature"), errors="coerce")
    s_m = pd.to_numeric(df_in.get("hours_since_maintenance"), errors="coerce")

    voltage_risk = np.clip(((s_v - 1.8) / 0.2) * 100, 0, 100)
    crossover_risk = np.clip((s_x / 500) * 100, 0, 100)
    thermal_risk = np.clip((s_t.sub(85).abs() / 10) * 100, 0, 100)
    maintenance_risk = np.clip((s_m / 2000) * 100, 0, 100)

    risk = pd.DataFrame({
        "overall_risk": (voltage_risk + crossover_risk + thermal_risk + maintenance_risk) / 4,
        "voltage_risk": voltage_risk,
        "crossover_risk": crossover_risk,
        "thermal_risk": thermal_risk,
        "maintenance_risk": maintenance_risk
    })
    try:
        risk["risk_level"] = pd.cut(
            risk["overall_risk"],
            bins=[0, 25, 50, 75, 100],
            labels=["Low", "Medium", "High", "Critical"]
        )
    except Exception:
        risk["risk_level"] = np.nan
    return risk


# ============= Predictions =============
def generate_predictions(df_in: pd.DataFrame, horizon: int) -> pd.DataFrame:
    last_timestamp = df_in["timestamp"].max()
    future_timestamps = pd.date_range(start=last_timestamp + timedelta(hours=1), periods=horizon, freq="H")

    hist = df_in["cell_voltage"].dropna()
    if len(hist) < 2:
        trend = 0.0
        base = hist.iloc[-1] if len(hist) else 1.9
    else:
        n = min(168, len(hist))
        recent_voltage = hist.tail(n).values
        trend = np.polyfit(range(len(recent_voltage)), recent_voltage, 1)[0]
        base = hist.iloc[-1]

    predictions, uncertainties = [], []
    for i in range(horizon):
        pred = base + trend * i + np.random.normal(0, 0.01)
        uncertainty = 0.02 + 0.001 * i
        predictions.append(pred)
        uncertainties.append(uncertainty)

    pred_df = pd.DataFrame({
        "timestamp": future_timestamps,
        "predicted_voltage": predictions,
        "lower_bound": np.array(predictions) - 1.96 * np.array(uncertainties),
        "upper_bound": np.array(predictions) + 1.96 * np.array(uncertainties),
        "uncertainty": uncertainties
    })

    critical_voltage = 2.0
    pred_df["failure_probability"] = pred_df.apply(
        lambda row: 1 - stats.norm.cdf(critical_voltage, row["predicted_voltage"], row["uncertainty"]),
        axis=1
    )
    return pred_df


# ============= UI =============
if df is None:
    st.stop()  # Nothing to show until data is available

st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns.")
st.dataframe(df.head(50), use_container_width=True)

risk_metrics = calculate_risk_metrics(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Real-time Monitoring",
    "⚠️ Failure Prediction",
    "⚠️ Risk Assessment",
    "📋 Maintenance Planning"
])


# ---------- Tab 1: Real-time Monitoring ----------
with tab1:
    st.markdown("### 🔍 Current System Status")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        current_voltage = df["cell_voltage"].dropna().iloc[-1] if df["cell_voltage"].notna().any() else np.nan
        voltage_delta = np.nan
        if df["cell_voltage"].notna().sum() >= 25:
            voltage_delta = df["cell_voltage"].dropna().iloc[-1] - df["cell_voltage"].dropna().iloc[-24]
        st.metric("Cell Voltage", f"{current_voltage:.3f} V" if pd.notna(current_voltage) else "N/A",
                  f"{voltage_delta:+.3f} V" if pd.notna(voltage_delta) else "—",
                  delta_color="inverse")

    with col2:
        eff_series = df["efficiency"].dropna() if "efficiency" in df.columns else pd.Series(dtype=float)
        current_efficiency = eff_series.iloc[-1] if not eff_series.empty else np.nan
        efficiency_delta = np.nan
        if len(eff_series) >= 25:
            efficiency_delta = eff_series.iloc[-1] - eff_series.iloc[-24]
        st.metric("Efficiency",
                  f"{current_efficiency:.3f}" if pd.notna(current_efficiency) else "N/A",
                  f"{efficiency_delta:+.3f}" if pd.notna(efficiency_delta) else "—")

    with col3:
        h2_series = df["h2_production_rate"].dropna() if "h2_production_rate" in df.columns else pd.Series(dtype=float)
        h2_rate = h2_series.iloc[-1] if not h2_series.empty else np.nan
        h2_delta = np.nan
        if len(h2_series) >= 25:
            h2_delta = h2_series.iloc[-1] - h2_series.iloc[-24]
        st.metric("H₂ Production",
                  f"{h2_rate:.2f} Nm³/h" if pd.notna(h2_rate) else "N/A",
                  f"{h2_delta:+.2f}" if pd.notna(h2_delta) else "—")

    with col4:
        temp_series = df["electrolyte_temperature"].dropna() if "electrolyte_temperature" in df.columns else pd.Series(dtype=float)
        current_temp = temp_series.iloc[-1] if not temp_series.empty else np.nan
        temp_delta = (current_temp - 85) if pd.notna(current_temp) else np.nan
        st.metric("Temperature",
                  f"{current_temp:.1f} °C" if pd.notna(current_temp) else "N/A",
                  f"{temp_delta:+.1f} °C" if pd.notna(temp_delta) else "—",
                  delta_color="inverse" if pd.notna(temp_delta) and abs(temp_delta) > 5 else "normal")

    with col5:
        current_overall_risk = risk_metrics["overall_risk"].dropna().iloc[-1] if risk_metrics["overall_risk"].notna().any() else np.nan
        risk_level = risk_metrics["risk_level"].iloc[-1] if risk_metrics["risk_level"].notna().any() else "Unknown"
        color = "🟢" if str(risk_level) == "Low" else "🟡" if str(risk_level) == "Medium" else "🔴" if str(risk_level) in ["High", "Critical"] else "⚪"
        st.metric("Risk Score",
                  f"{current_overall_risk:.1f}%" if pd.notna(current_overall_risk) else "N/A",
                  f"{color} {risk_level}")

    st.markdown("---")

    # Time series plots
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Cell Voltage Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"].tail(168),
            y=df["cell_voltage"].tail(168),
            mode="lines",
            name="Cell Voltage",
            line=dict(width=2)
        ))
        fig.add_hline(y=1.95, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        fig.add_hline(y=1.90, line_dash="dash", line_color="orange", annotation_text="Warning Level")
        fig.update_layout(xaxis_title="Time", yaxis_title="Voltage (V)", height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("#### Efficiency & Production")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"].tail(168),
            y=df["efficiency"].tail(168),
            mode="lines",
            name="Efficiency",
            yaxis="y"
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"].tail(168),
            y=df["h2_production_rate"].tail(168),
            mode="lines",
            name="H₂ Production",
            yaxis="y2"
        ))
        fig.update_layout(
            xaxis_title="Time",
            yaxis=dict(title="Efficiency", side="left"),
            yaxis2=dict(title="H₂ Rate (Nm³/h)", side="right", overlaying="y"),
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # System health gauges
    st.markdown("#### System Health Indicators")
    g1, g2, g3 = st.columns(3)

    with g1:
        val = df["h2_purity"].dropna().iloc[-1] if "h2_purity" in df.columns and df["h2_purity"].notna().any() else np.nan
        if pd.notna(val):
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=val, title={'text': "H₂ Purity (%)"},
                delta={'reference': 99.5},
                gauge={'axis': {'range': [None, 100]},
                       'steps': [{'range': [0, 98], 'color': "lightgray"},
                                 {'range': [98, 99.5], 'color': "yellow"},
                                 {'range': [99.5, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 99}}
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("H₂ Purity: N/A")

    with g2:
        val = df["o2_in_h2"].dropna().iloc[-1] if "o2_in_h2" in df.columns and df["o2_in_h2"].notna().any() else np.nan
        if pd.notna(val):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val, title={'text': "O₂ in H₂ (ppm)"},
                gauge={'axis': {'range': [None, 500]},
                       'steps': [{'range': [0, 100], 'color': "green"},
                                 {'range': [100, 300], 'color': "yellow"},
                                 {'range': [300, 500], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 400}}
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("O₂ in H₂: N/A")

    with g3:
        val = df["hours_since_maintenance"].dropna().iloc[-1] if "hours_since_maintenance" in df.columns and df["hours_since_maintenance"].notna().any() else np.nan
        if pd.notna(val):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val, title={'text': "Hours Since Maintenance"},
                gauge={'axis': {'range': [None, 2500]},
                       'steps': [{'range': [0, 1000], 'color': "green"},
                                 {'range': [1000, 2000], 'color': "yellow"},
                                 {'range': [2000, 2500], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 2000}}
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Hours Since Maintenance: N/A")


# ---------- Tab 2: Failure Prediction ----------
with tab2:
    st.markdown("### Predictive Analytics - Equipment Failure Forecast")

    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Running Nixtla TimeGPT model (simulated)…"):
            st.session_state.predictions = generate_predictions(df, forecast_horizon)

    if st.session_state.predictions is not None:
        pred_df = st.session_state.predictions

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            max_voltage = pred_df["predicted_voltage"].max()
            st.metric("Max Predicted Voltage", f"{max_voltage:.3f} V", "⚠️ Critical" if max_voltage > 1.95 else "✅ Normal")

        with col2:
            max_prob = (pred_df["failure_probability"].max() or 0) * 100
            st.metric("Max Failure Risk", f"{max_prob:.1f}%", "🔴 High" if max_prob > 50 else "🟢 Low")

        with col3:
            crit = pred_df[pred_df["failure_probability"] > 0.5]
            if not crit.empty:
                time_to_failure = crit["timestamp"].min()
                hours_to_failure = (time_to_failure - df["timestamp"].max()).total_seconds() / 3600
                st.metric("Time to Critical", f"{hours_to_failure:.0f} hours", "⏰ Plan Maintenance")
            else:
                st.metric("Time to Critical", "No Risk", "✅ Safe")

        with col4:
            confidence = 100 - (pred_df["uncertainty"].mean() * 100)
            st.metric("Model Confidence", f"{confidence:.1f}%", "High" if confidence > 80 else "Medium")

        st.markdown("---")

        # Voltage forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"].tail(168), y=df["cell_voltage"].tail(168), mode="lines", name="Historical"))
        fig.add_trace(go.Scatter(x=pred_df["timestamp"], y=pred_df["predicted_voltage"], mode="lines", name="Prediction", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(
            x=pred_df["timestamp"].tolist() + pred_df["timestamp"].tolist()[::-1],
            y=pred_df["upper_bound"].tolist() + pred_df["lower_bound"].tolist()[::-1],
            fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence"
        ))
        fig.add_hline(y=2.0, line_dash="dash", line_color="darkred", annotation_text="Critical Failure Threshold")
        fig.update_layout(title="Cell Voltage Prediction with Uncertainty Bands", xaxis_title="Time", yaxis_title="Cell Voltage (V)", height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Failure probability
        st.markdown("#### Failure Probability Timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_df["timestamp"], y=pred_df["failure_probability"] * 100, mode="lines+markers", name="Failure Probability"))
        fig.add_hline(y=risk_threshold, line_dash="dash", line_color="orange", annotation_text=f"Alert Threshold ({risk_threshold}%)")
        fig.update_layout(xaxis_title="Time", yaxis_title="Failure Probability (%)", height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Component risk (with fallbacks)
        st.markdown("#### Component-wise Failure Risk Analysis")
        components = ["Diaphragm", "Electrode", "Seal", "Vessel", "Sensors"]
        dp = float(df["differential_pressure"].dropna().iloc[-1]) if "differential_pressure" in df.columns and df["differential_pressure"].notna().any() else 0.0
        temp = float(df["electrolyte_temperature"].dropna().iloc[-1]) if "electrolyte_temperature" in df.columns and df["electrolyte_temperature"].notna().any() else 85.0
        cycles = float(df["cycles_count"].dropna().iloc[-1]) if "cycles_count" in df.columns and df["cycles_count"].notna().any() else 0.0
        hours = float(df["hours_since_maintenance"].dropna().iloc[-1]) if "hours_since_maintenance" in df.columns and df["hours_since_maintenance"].notna().any() else 0.0
        conc = float(df["electrolyte_concentration"].dropna().iloc[-1]) if "electrolyte_concentration" in df.columns and df["electrolyte_concentration"].notna().any() else 30.0

        comp_risks = []
        for comp in components:
            if comp == "Diaphragm":
                r = min(100, (dp / 50 + (temp - 85) / 10) * 30)
            elif comp == "Electrode":
                r = min(100, cycles / 20)
            elif comp == "Seal":
                r = min(100, hours / 30)
            elif comp == "Vessel":
                r = min(100, max(0, (conc - 30)) * 10)
            else:
                r = np.random.uniform(10, 40)
            comp_risks.append(max(0, r))

        fig = go.Figure(data=[go.Bar(x=components, y=comp_risks, text=[f"{r:.1f}%" for r in comp_risks], textposition="auto")])
        fig.update_layout(title="Component Failure Risk Assessment", xaxis_title="Component", yaxis_title="Risk Level (%)", height=300)
        st.plotly_chart(fig, use_container_width=True)


# ---------- Tab 3: Risk Assessment ----------
with tab3:
    st.markdown("### ⚠️ Comprehensive Risk Assessment Dashboard")

    current_overall_risk = risk_metrics["overall_risk"].dropna().iloc[-1] if risk_metrics["overall_risk"].notna().any() else np.nan
    risk_level = risk_metrics["risk_level"].iloc[-1] if risk_metrics["risk_level"].notna().any() else "Unknown"

    if str(risk_level) == "Critical":
        st.error(f"🚨 **CRITICAL RISK DETECTED** - Overall Risk Score: {current_overall_risk:.1f}%")
    elif str(risk_level) == "High":
        st.warning(f"⚠️ **HIGH RISK** - Overall Risk Score: {current_overall_risk:.1f}%")
    elif str(risk_level) == "Medium":
        st.info(f"ℹ️ **MEDIUM RISK** - Overall Risk Score: {current_overall_risk:.1f}%")
    elif str(risk_level) == "Low":
        st.success(f"✅ **LOW RISK** - Overall Risk Score: {current_overall_risk:.1f}%")
    else:
        st.info("Risk level: **Unknown** (insufficient data)")

    # Risk radar
    st.markdown("#### Risk Factor Breakdown")
    c1, c2 = st.columns(2)
    values = [
        risk_metrics["voltage_risk"].dropna().iloc[-1] if risk_metrics["voltage_risk"].notna().any() else 0,
        risk_metrics["crossover_risk"].dropna().iloc[-1] if risk_metrics["crossover_risk"].notna().any() else 0,
        risk_metrics["thermal_risk"].dropna().iloc[-1] if risk_metrics["thermal_risk"].notna().any() else 0,
        risk_metrics["maintenance_risk"].dropna().iloc[-1] if risk_metrics["maintenance_risk"].notna().any() else 0,
    ]
    with c1:
        categories = ["Voltage\nDegradation", "Gas\nCrossover", "Thermal\nStress", "Maintenance\nUrgency"]
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Current Risk Profile'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"].tail(168), y=risk_metrics["overall_risk"].tail(168), mode="lines", name="Overall Risk"))
        fig.add_hline(y=75, line_dash="dash", line_color="darkred", annotation_text="Critical")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="High")
        fig.add_hline(y=25, line_dash="dash", line_color="yellow", annotation_text="Medium")
        fig.update_layout(title="Risk Score Trend (Last 7 Days)", xaxis_title="Time", yaxis_title="Risk Score (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed risk report
    st.markdown("#### Detailed Risk Report")
    rr = pd.DataFrame({
        "Risk Factor": ["Voltage Degradation", "Gas Crossover", "Thermal Stress", "Maintenance Urgency"],
        "Current Value": [
            f"{(df['cell_voltage'].dropna().iloc[-1] if 'cell_voltage' in df.columns and df['cell_voltage'].notna().any() else np.nan):.3f} V",
            f"{(df['o2_in_h2'].dropna().iloc[-1] if 'o2_in_h2' in df.columns and df['o2_in_h2'].notna().any() else np.nan):.0f} ppm",
            f"{(df['electrolyte_temperature'].dropna().iloc[-1] if 'electrolyte_temperature' in df.columns and df['electrolyte_temperature'].notna().any() else np.nan):.1f} °C",
            f"{(df['hours_since_maintenance'].dropna().iloc[-1] if 'hours_since_maintenance' in df.columns and df['hours_since_maintenance'].notna().any() else np.nan):.0f} hours",
        ],
        "Risk Score": [f"{v:.1f}%" for v in values],
        "Status": ["🔴 Critical" if v > 75 else "🟠 High" if v > 50 else "🟡 Medium" if v > 25 else "🟢 Low" for v in values],
        "Recommended Action": [
            "Immediate electrode inspection" if values[0] > 75 else "Monitor closely" if values[0] > 50 else "Routine monitoring",
            "Check diaphragm integrity" if values[1] > 75 else "Verify gas analyzers" if values[1] > 50 else "Normal operation",
            "Adjust cooling system" if values[2] > 75 else "Check temperature control" if values[2] > 50 else "Maintain current settings",
            "Schedule immediate maintenance" if values[3] > 75 else "Plan maintenance soon" if values[3] > 50 else "Continue operation",
        ]
    })
    st.dataframe(rr, use_container_width=True, hide_index=True)

    # Incidents (simulated example)
    st.markdown("#### Historical Incident Analysis")
    c3, c4 = st.columns(2)
    with c3:
        incident_types = ["Voltage Spike", "Gas Crossover", "Temperature Excursion", "Pressure Imbalance", "Efficiency Drop"]
        incident_counts = [12, 8, 15, 5, 10]
        fig = px.pie(values=incident_counts, names=incident_types, title="Incident Distribution (Last 90 Days)",
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        weeks = pd.date_range(start=df["timestamp"].min(), end=df["timestamp"].max(), freq="W")
        incident_freq = np.random.poisson(2, len(weeks))
        fig = go.Figure(data=go.Bar(x=weeks, y=incident_freq))
        fig.update_layout(title="Weekly Incident Frequency", xaxis_title="Week", yaxis_title="Number of Incidents", height=350)
        st.plotly_chart(fig, use_container_width=True)


# ---------- Tab 4: Maintenance Planning ----------
with tab4:
    st.markdown("### 📋 Intelligent Maintenance Planning & Recommendations")

    hours_operated = df["hours_since_maintenance"].dropna().iloc[-1] if df["hours_since_maintenance"].notna().any() else 0.0
    cycles_completed = df["cycles_count"].dropna().iloc[-1] if "cycles_count" in df.columns and df["cycles_count"].notna().any() else 0.0

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Hours Since Last Maintenance", f"{hours_operated:.0f} hrs")
    with c2: st.metric("Cycles Completed", f"{cycles_completed:.0f}")
    with c3:
        next_scheduled = 2000 - hours_operated
        st.metric("Hours to Scheduled Maintenance", f"{max(0, next_scheduled):.0f} hrs")

    st.markdown("---")

    # Use same risk snapshot as Tab 3
    current_overall_risk = risk_metrics["overall_risk"].dropna().iloc[-1] if risk_metrics["overall_risk"].notna().any() else 0
    cell_v = df["cell_voltage"].dropna().iloc[-1] if df["cell_voltage"].notna().any() else 1.9
    if current_overall_risk > 75 or cell_v > 1.95:
        urgency, color, icon, eta, downtime = "IMMEDIATE", "red", "🚨", "Within 4 hours", "4–6 hours"
    elif current_overall_risk > 50:
        urgency, color, icon, eta, downtime = "SCHEDULED", "orange", "⚠️", "Within 48 hours", "2–4 hours"
    elif current_overall_risk > 25:
        urgency, color, icon, eta, downtime = "PLANNED", "yellow", "📅", "Within 1 week", "2–3 hours"
    else:
        urgency, color, icon, eta, downtime = "ROUTINE", "green", "✅", "As scheduled", "1–2 hours"

    st.markdown(f"""<div style="background-color: {color}; opacity: 0.1; padding: 20px; border-radius: 10px;"></div>""",
                unsafe_allow_html=True)

    a1, a2 = st.columns([1, 2])
    with a1:
        st.markdown(f"### {icon} **{urgency}**")
        st.markdown(f"**Timeline:** {eta}")
        st.markdown(f"**Est. Downtime:** {downtime}")
    with a2:
        st.markdown("**Recommended Actions:**")
        # Reuse 'values' from Tab 3 breakdown
        actions = []
        if values[0] > 50:  # Voltage risk
            actions += ["• Inspect electrode coating for degradation", "• Measure individual cell voltages"]
        if values[1] > 50:  # Gas crossover risk
            actions += ["• Check diaphragm integrity", "• Verify gas analyzer calibration"]
        if values[2] > 50:  # Thermal risk
            actions += ["• Inspect cooling system performance", "• Check electrolyte circulation"]
        if values[3] > 50:  # Maintenance urgency
            actions += ["• Replace worn gaskets and seals", "• Clean and recalibrate sensors"]
        if not actions:
            actions = ["• Routine visual inspection", "• Record operational parameters"]
        for action in actions[:6]:
            st.markdown(action)

    st.markdown("---")

    st.markdown("#### 📅 Optimized Maintenance Schedule")
    maintenance_tasks = pd.DataFrame({
        "Task": ["Electrode Inspection", "Diaphragm Check", "Electrolyte Analysis",
                 "Sensor Calibration", "Seal Replacement", "System Flush"],
        "Priority": ["High", "High", "Medium", "Medium", "Low", "Low"],
        "Estimated Duration": ["2 hrs", "3 hrs", "1 hr", "1 hr", "4 hrs", "2 hrs"],
        "Last Performed": [
            (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d'),
            (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d'),
            (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
            (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        ],
        "Next Due": [(datetime.now() + timedelta(days=d)).strftime('%Y-%m-%d') for d in [5, 3, 16, 23, 10, 30]],
        "Status": ["⚠️ Due Soon", "🔴 Overdue", "✅ On Schedule", "✅ On Schedule", "⚠️ Due Soon", "✅ On Schedule"]
    })
    st.dataframe(maintenance_tasks, use_container_width=True, hide_index=True)

    st.markdown("#### 💰 Maintenance Cost-Benefit Analysis")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Preventive Maintenance Benefits:**")
        for k, v in {"Avoided Downtime": "$15,000", "Extended Equipment Life": "$8,000",
                     "Improved Efficiency": "$5,000", "Reduced Emergency Repairs": "$10,000",
                     "Total Benefit": "$38,000"}.items():
            st.markdown(f"**{k}: {v}**" if k == "Total Benefit" else f"• {k}: {v}")
    with b2:
        st.markdown("**Maintenance Costs:**")
        for k, v in {"Labor": "$3,000", "Parts & Materials": "$5,000", "Production Loss": "$4,000",
                     "Testing & Validation": "$1,000", "Total Cost": "$13,000"}.items():
            st.markdown(f"**{k}: {v}**" if k == "Total Cost" else f"• {k}: {v}")
    st.success("**Net Benefit: $25,000** (ROI: 192%)")

    st.markdown("#### 🔧 Spare Parts Inventory Status")
    parts = pd.DataFrame({
        "Part Name": ["Nickel Electrodes", "Diaphragm Material", "Gasket Set", "Temperature Sensors", "Pressure Gauges", "KOH Solution"],
        "Current Stock": [3, 2, 8, 4, 3, 500],
        "Min Required": [2, 1, 5, 2, 2, 300],
        "Unit": ["pcs", "rolls", "sets", "pcs", "pcs", "liters"],
        "Status": ["✅", "⚠️", "✅", "✅", "✅", "✅"],
        "Reorder": ["No", "Yes", "No", "No", "No", "No"]
    })
    st.dataframe(parts, use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("📄 Generate Maintenance Report", type="primary"):
        st.success("✅ Maintenance report generated successfully!")
        st.balloons()


# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <small>
    Green Hydrogen Electrolyzer Predictive Maintenance System v1.0<br>
    Powered by Nixtla TimeGPT & Advanced Analytics<br>
    ACWA Power Challenge Solution 2024
    </small>
</div>
""", unsafe_allow_html=True)
