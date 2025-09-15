# -------------------------------
# Green Hydrogen Electrolyzer Predictive Maintenance System
# ACWA Power Challenge Solution using Nixtla TimeGPT
# -------------------------------

import os
import io
import math
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ============= Optional model libs (robust fallbacks) =============
# Try Nixtla SDK
try:
    from nixtla import NixtlaClient
    HAS_NIXTLA = True
    NIXTLA_IMPORT_ERROR = None
except Exception as _e:
    HAS_NIXTLA = False
    NIXTLA_IMPORT_ERROR = str(_e)

# Try XGBoost, else fall back to scikit-learn’s GradientBoosting
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except Exception:
        GradientBoostingRegressor = None

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

# ============= Session State =============
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None

# ============= Small helpers =============
def safe_last(series: pd.Series, default=np.nan):
    s = pd.to_numeric(series, errors="coerce")
    return s.iloc[-1] if len(s) else default

def safe_delta(series: pd.Series, k: int, default=np.nan):
    s = pd.to_numeric(series, errors="coerce")
    if len(s) > k:
        return s.iloc[-1] - s.iloc[-1 - k]
    return default

def _get_api_key() -> Optional[str]:
    # prefer Streamlit secrets, fall back to env var
    key = None
    try:
        key = st.secrets.get("NIXTLA_API_KEY", None)  # type: ignore[attr-defined]
    except Exception:
        pass
    return key or os.getenv("NIXTLA_API_KEY")

def _get_nixtla_client() -> Optional["NixtlaClient"]:
    if not HAS_NIXTLA:
        st.warning(f"Nixtla SDK not installed: {NIXTLA_IMPORT_ERROR}")
        return None
    key = _get_api_key()
    if not key:
        st.warning("NIXTLA_API_KEY not found in .streamlit/secrets.toml or environment.")
        return None
    try:
        return NixtlaClient(api_key=key)
    except Exception as e:
        st.warning(f"Failed to init NixtlaClient: {e}")
        return None

def _ensure_schema_for_rt(df: pd.DataFrame) -> list[str]:
    """Check that Real-time Monitoring has what it needs; return list of missing columns."""
    required = [
        "timestamp", "cell_voltage", "efficiency", "h2_production_rate",
        "electrolyte_temperature", "o2_in_h2", "hours_since_maintenance"
    ]
    return [c for c in required if c not in df.columns]

def _std_from_pi95(upper: pd.Series, mean: pd.Series) -> pd.Series:
    # approximate sigma from 95% band; avoid divide-by-zero
    sig = (upper - mean) / 1.96
    sig = sig.replace([np.inf, -np.inf], np.nan)
    med = np.nanmedian(sig) if np.isfinite(sig).any() else 0.03
    sig = sig.fillna(med)
    sig = sig.clip(lower=1e-6)
    return sig

def _standardize_forecast_output(ts_index: pd.DatetimeIndex,
                                 yhat: np.ndarray,
                                 lo95: Optional[np.ndarray]=None,
                                 hi95: Optional[np.ndarray]=None,
                                 sigma: Optional[np.ndarray]=None) -> pd.DataFrame:
    """Return a unified predictions DataFrame used by the app UI."""
    dfp = pd.DataFrame({"timestamp": ts_index, "predicted_voltage": yhat})
    if hi95 is not None and lo95 is not None:
        dfp["lower_bound"] = lo95
        dfp["upper_bound"] = hi95
        if sigma is None:
            sigma = ((dfp["upper_bound"] - dfp["predicted_voltage"]) / 1.96).to_numpy()
    else:
        if sigma is None:
            sigma = np.linspace(0.02, 0.02 + 0.001*(len(ts_index)-1), len(ts_index))
        dfp["lower_bound"] = yhat - 1.96*sigma
        dfp["upper_bound"] = yhat + 1.96*sigma
    dfp["uncertainty"] = sigma
    dfp["failure_probability"] = 1 - stats.norm.cdf(
        2.0,
        dfp["predicted_voltage"],
        np.clip(dfp["uncertainty"], 1e-6, None),
    )
    return dfp

# ============= File loader (Excel/CSV with sniffing) =============
@st.cache_data(show_spinner=False)
def load_table(uploaded) -> Optional[pd.DataFrame]:
    if uploaded is None:
        return None

    name = (uploaded.name or "").lower()

    # try by extension first
    try:
        if name.endswith((".xlsx", ".xlsm")):
            return pd.read_excel(uploaded, engine="openpyxl")
        elif name.endswith(".xls"):
            return pd.read_excel(uploaded, engine="xlrd")
        elif name.endswith(".csv"):
            return pd.read_csv(uploaded)
    except Exception as e:
        st.warning(f"Tried by extension but failed: {e}. Falling back to sniffing.")

    # fallback: sniff by content
    uploaded.seek(0)
    raw = uploaded.read()

    # try as Excel first
    try:
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        return df
    except Exception:
        pass

    # try as CSV
    try:
        uploaded.seek(0)
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(
            "Could not parse the uploaded file as Excel or CSV. "
            f"Original error: {e}"
        )

# ============= Demo data generator (realistic synthetic) =============
@st.cache_data
def generate_demo_data(n_points: int = 2000) -> pd.DataFrame:
    """Generate realistic synthetic electrolyzer data"""
    np.random.seed(42)
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(hours=n_points-1),
        periods=n_points,
        freq="H"
    )
    t = np.arange(n_points)

    # cell voltage with slow degradation + daily cycle
    base_voltage = 1.8
    degradation_rate = 0.00005
    voltage = (base_voltage
               + degradation_rate * t
               + 0.05 * np.sin(2*np.pi*t/24)
               + np.random.normal(0, 0.02, n_points))

    # current with daily pattern
    current = 1000 + 200 * np.sin(2*np.pi*t/24) + np.random.normal(0, 50, n_points)
    current = np.clip(current, 500, 1500)

    temp = 85 + 5 * np.sin(2*np.pi*t/168) + np.random.normal(0, 2, n_points)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "cell_voltage": voltage,
        "stack_current": current,
        "electrolyte_temperature": temp,
        "electrolyte_concentration": 30 + np.random.normal(0, 1, n_points),
        "electrolyte_conductivity": 500 - 0.01 * t + np.random.normal(0, 10, n_points),
        "operating_pressure": 10 + np.random.normal(0, 0.5, n_points),
        "h2_production_rate": current * 0.08 + np.random.normal(0, 5, n_points),
        "o2_production_rate": current * 0.04 + np.random.normal(0, 2, n_points),
        "power_consumption": current * voltage * 0.001,
        "h2_purity": 99.5 + np.random.normal(0, 0.1, n_points),
        "o2_in_h2": 100 + 0.05 * t + np.random.normal(0, 20, n_points),
        "h2_in_o2": 50 + 0.02 * t + np.random.normal(0, 10, n_points),
        "differential_pressure": np.random.normal(0, 10, n_points),
        "hours_since_maintenance": np.arange(n_points) % 2000,
        "cycles_count": np.cumsum(np.random.binomial(1, 0.02, n_points)),
        "ambient_temperature": 25 + 10 * np.sin(2*np.pi*t/24) + np.random.normal(0, 2, n_points),
        "cooling_water_temp": 20 + 5 * np.sin(2*np.pi*t/24) + np.random.normal(0, 1, n_points),
        "demin_water_quality": 0.1 + np.random.normal(0, 0.01, n_points),
    })
    df["efficiency"] = df["h2_production_rate"] / df["power_consumption"]
    df["failure_risk"] = 0
    df.loc[df["cell_voltage"] > 1.95, "failure_risk"] = 1
    df.loc[df["o2_in_h2"] > 200, "failure_risk"] = 1
    df.loc[df["hours_since_maintenance"] > 1800, "failure_risk"] = 1
    return df

# ============= Risk metrics =============
def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive risk metrics"""
    risk_scores = pd.DataFrame(index=df.index)
    voltage_risk = np.clip((df["cell_voltage"] - 1.8) / 0.2 * 100, 0, 100)
    crossover_risk = np.clip(df["o2_in_h2"] / 500 * 100, 0, 100)
    thermal_risk = np.clip(np.abs(df["electrolyte_temperature"] - 85) / 10 * 100, 0, 100)
    maintenance_risk = np.clip(df["hours_since_maintenance"] / 2000 * 100, 0, 100)

    risk_scores["overall_risk"] = (voltage_risk + crossover_risk + thermal_risk + maintenance_risk) / 4
    risk_scores["voltage_risk"] = voltage_risk
    risk_scores["crossover_risk"] = crossover_risk
    risk_scores["thermal_risk"] = thermal_risk
    risk_scores["maintenance_risk"] = maintenance_risk
    risk_scores["risk_level"] = pd.cut(
        risk_scores["overall_risk"], bins=[0, 25, 50, 75, 100],
        labels=["Low", "Medium", "High", "Critical"]
    )
    return risk_scores

# ============= Forecast models (4 options) =============
def forecast_timegpt(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Nixtla TimeGPT forecast for 'cell_voltage', with graceful fallback."""
    client = _get_nixtla_client()
    if client is None:
        st.warning("Falling back to Statistical Ensemble (TimeGPT not available).")
        return forecast_stat_ensemble(df, horizon)

    hist = df[["timestamp", "cell_voltage"]].dropna().copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist = hist.sort_values("timestamp")

    try:
        fcst = client.forecast(
            df=hist.rename(columns={"cell_voltage": "value"}),
            h=int(horizon),
            freq="H",
            time_col="timestamp",
            target_col="value",
            level=[95],
        )
    except Exception as e:
        st.warning(f"TimeGPT call failed: {e}. Falling back to Statistical Ensemble.")
        return forecast_stat_ensemble(df, horizon)

    # Flexible column pickers (SDK versions may differ slightly in names)
    def _first_match(cands):
        for c in cands:
            if c in fcst.columns:
                return c
        # try case-insensitive
        low = {c.lower(): c for c in fcst.columns}
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return None

    ts_col = _first_match(["timestamp", "ds", "time"])
    y_col  = _first_match(["TimeGPT", "yhat", "forecast", "value"])
    lo_col = _first_match(["TimeGPT-lo-95", "lo-95", "yhat_lower", "lower"])
    hi_col = _first_match(["TimeGPT-hi-95", "hi-95", "yhat_upper", "upper"])

    if ts_col is None or y_col is None:
        st.warning("TimeGPT response missing expected columns; using Statistical Ensemble.")
        return forecast_stat_ensemble(df, horizon)

    ts = pd.to_datetime(fcst[ts_col])
    yhat = pd.to_numeric(fcst[y_col], errors="coerce").to_numpy()

    lo95 = pd.to_numeric(fcst[lo_col], errors="coerce").to_numpy() if lo_col else None
    hi95 = pd.to_numeric(fcst[hi_col], errors="coerce").to_numpy() if hi_col else None
    sigma = None
    if lo95 is not None and hi95 is not None:
        sigma = _std_from_pi95(pd.Series(hi95), pd.Series(yhat)).to_numpy()

    return _standardize_forecast_output(ts, yhat, lo95, hi95, sigma)

def forecast_stat_ensemble(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple, dependency-light ensemble: Naive + 24h mean + short linear trend."""
    y = pd.to_numeric(df["cell_voltage"], errors="coerce").dropna().to_numpy()
    ts_last = pd.to_datetime(df["timestamp"]).max()

    if len(y) < 32:
        base = y[-1] if len(y) else 1.9
        yhat = base + np.zeros(horizon)
        ts = pd.date_range(ts_last + pd.Timedelta(hours=1), periods=horizon, freq="H")
        return _standardize_forecast_output(ts, yhat)

    naive = np.full(horizon, y[-1])
    ma = np.full(horizon, y[-24:].mean() if len(y) >= 24 else y.mean())

    nfit = min(168, len(y))
    slope = np.polyfit(np.arange(nfit), y[-nfit:], 1)[0]
    trend = y[-1] + slope * np.arange(1, horizon + 1)

    yhat = (naive + ma + trend) / 3.0
    ts = pd.date_range(ts_last + pd.Timedelta(hours=1), periods=horizon, freq="H")

    if nfit > 2:
     y_series = pd.Series(y).dropna().to_numpy()
     y_lag = pd.Series(y).shift(1).dropna().to_numpy()
     min_len = min(len(y_series), len(y_lag))
     resid_std = np.std(y_series[-min_len:] - y_lag[-min_len:])
    else:
     resid_std = np.std(y[-nfit:])

    sigma = np.clip(resid_std if not math.isnan(resid_std) else 0.03, 0.01, 0.12)
    sigma_vec = np.linspace(sigma, sigma * 1.5, horizon)
    return _standardize_forecast_output(ts, yhat, sigma=sigma_vec)

def _make_features(ts):
    ts = pd.Series(ts)   # convert DatetimeIndex → Series
    return pd.DataFrame({
        "year": ts.dt.year,
        "month": ts.dt.month,
        "day": ts.dt.day,
        "dayofweek": ts.dt.dayofweek,
        "hour": ts.dt.hour,
    })

def forecast_xgboost(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    XGBoost (or GradientBoosting fallback) with lag features and calendar effects.
    Recursive multi-step forecast.
    """
    y = pd.to_numeric(df["cell_voltage"], errors="coerce")
    ts = pd.to_datetime(df["timestamp"])
    data = pd.DataFrame({"y": y.values}, index=ts).dropna()

    if len(data) < 200:
        st.info("Not enough history for ML model; falling back to Statistical Ensemble.")
        return forecast_stat_ensemble(df, horizon)

    # lags
    for k in [1, 2, 3, 24]:
        data[f"lag_{k}"] = data["y"].shift(k)

    feats = _make_features(data.index)
    data = pd.concat([data, feats], axis=1).dropna()

    X = data.drop(columns=["y"])
    Y = data["y"]

    # choose regressor
    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    elif GradientBoostingRegressor is not None:
        model = GradientBoostingRegressor(random_state=42)
    else:
        return forecast_stat_ensemble(df, horizon)

    model.fit(X, Y)

    # recursive forecast
    future_index = pd.date_range(data.index.max() + pd.Timedelta(hours=1), periods=horizon, freq="H")
    history = data.copy()

    preds = []
    for h in range(horizon):
        t = future_index[h]
        row = _make_features(pd.Series([t]))
        lag1 = history["y"].iloc[-1]
        lag2 = history["y"].iloc[-2] if len(history) > 1 else lag1
        lag3 = history["y"].iloc[-3] if len(history) > 2 else lag2
        lag24 = history["y"].iloc[-24] if len(history) >= 24 else history["y"].iloc[0]
        row["lag_1"] = lag1
        row["lag_2"] = lag2
        row["lag_3"] = lag3
        row["lag_24"] = lag24
        yhat = float(model.predict(row)[0])
        preds.append(yhat)
        # append for next-step lags
        nx = pd.Series(index=history.columns, dtype=float)
        history.loc[t, "y"] = yhat

    preds = np.array(preds)
    # uncertainty ~ training residuals
    try:
        residuals = Y - model.predict(X)
        sigma = np.clip(residuals.std(), 0.01, 0.12)
    except Exception:
        sigma = 0.04
    sigma_vec = np.linspace(sigma, sigma * 1.5, horizon)
    return _standardize_forecast_output(future_index, preds, sigma=sigma_vec)

def forecast_hybrid(data, horizon):
    try:
        # Try Nixtla TimeGPT forecast first
        pgpt = forecast_timegpt(data, horizon)
    except Exception as e:
        st.warning(f"TimeGPT unavailable, falling back: {e}")
        pgpt = forecast_stat_ensemble(data, horizon)

    try:
        # Then run XGBoost forecast
        pml = forecast_xgboost(data, horizon)
    except Exception as e:
        st.warning(f"XGBoost forecast failed: {e}")
        return pgpt  # fallback only to TimeGPT/ensemble

    # Align outputs on timestamp
    merged = pd.merge(pgpt, pml, on="timestamp", suffixes=("_gpt", "_xgb"))

    # Average predictions
    merged["predicted_voltage"] = (merged["predicted_voltage_gpt"] + merged["predicted_voltage_xgb"]) / 2
    merged["lower_bound"] = (merged["lower_bound_gpt"] + merged["lower_bound_xgb"]) / 2
    merged["upper_bound"] = (merged["upper_bound_gpt"] + merged["upper_bound_xgb"]) / 2
    merged["uncertainty"] = (merged["uncertainty_gpt"] + merged["uncertainty_xgb"]) / 2
    merged["failure_probability"] = (merged["failure_probability_gpt"] + merged["failure_probability_xgb"]) / 2

    # Keep unified schema
    pred_df = merged[["timestamp", "predicted_voltage", "lower_bound", "upper_bound", "uncertainty", "failure_probability"]]
    return pred_df


# ============= Title & Sidebar =============
st.title(" Green Hydrogen Electrolyzer Predictive Maintenance System")
st.markdown("**ACWA Power Challenge Solution** | Powered by Nixtla TimeGPT & Advanced Analytics")

with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1c83e1/ffffff?text=ACWA+Power", use_container_width=True)
    st.markdown("---")

    st.markdown("### ⚙️ System Configuration")
    model_type = st.selectbox(
        "Select Prediction Model",
        ["Nixtla TimeGPT", "Statistical Ensemble", "XGBoost ML", "Hybrid Approach"]
    )
    forecast_horizon = st.slider("Forecast Horizon (hours)", 24, 168, 72, step=24)
    risk_threshold = st.slider("Risk Alert Threshold (%)", 50, 95, 75, step=5)

    st.markdown("---")
    st.markdown("### 📊 Data Source")

    uploaded_file = st.file_uploader("Upload Electrolyzer Data (.xlsx/.xls/.csv)", type=["xlsx", "xls", "csv"])

    if st.button("Use Demo Data", type="primary"):
        st.session_state.data_loaded = True

# ============= Load or generate data =============
if st.session_state.data_loaded or uploaded_file is not None:
    if uploaded_file is not None:
        df = load_table(uploaded_file)
        # Try to enforce timestamp presence if user upload has it
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            st.info("Uploaded file has no 'timestamp' column. Using demo data instead for testing.")
            df = generate_demo_data()

        # If efficiency missing but needed columns exist, derive it
        if "efficiency" not in df.columns and {"h2_production_rate", "power_consumption"}.issubset(df.columns):
            with np.errstate(divide="ignore", invalid="ignore"):
                df["efficiency"] = df["h2_production_rate"] / df["power_consumption"]
                df["efficiency"] = df["efficiency"].replace([np.inf, -np.inf], np.nan)
    else:
        df = generate_demo_data()

    # Real-time schema self-test
    missing_cols = _ensure_schema_for_rt(df)
    if missing_cols:
        st.error(f"Real-time Monitoring is missing required columns: {missing_cols}")
    else:
        st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns. Real-time Monitoring schema ✅ OK")

    # Enforce dtype & sort
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Risk metrics
    risk_metrics = calculate_risk_metrics(df)

    # ============= Tabs =============
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
            v_now = safe_last(df["cell_voltage"])
            v_delta = safe_delta(df["cell_voltage"], 24)
            st.metric("Cell Voltage",
                      f"{v_now:.3f} V" if pd.notna(v_now) else "N/A",
                      f"{v_delta:+.3f} V" if pd.notna(v_delta) else "—",
                      delta_color="inverse")

        with col2:
            eff_now = safe_last(df["efficiency"])
            eff_delta = safe_delta(df["efficiency"], 24)
            st.metric("Efficiency",
                      f"{eff_now:.3f}" if pd.notna(eff_now) else "N/A",
                      f"{eff_delta:+.3f}" if pd.notna(eff_delta) else "—")

        with col3:
            h2_now = safe_last(df["h2_production_rate"])
            h2_delta = safe_delta(df["h2_production_rate"], 24)
            st.metric("H₂ Production",
                      f"{h2_now:.1f} Nm³/h" if pd.notna(h2_now) else "N/A",
                      f"{h2_delta:+.1f}" if pd.notna(h2_delta) else "—")

        with col4:
            t_now = safe_last(df["electrolyte_temperature"])
            temp_delta = t_now - 85 if pd.notna(t_now) else np.nan
            st.metric("Temperature",
                      f"{t_now:.1f} °C" if pd.notna(t_now) else "N/A",
                      f"{temp_delta:+.1f} °C" if pd.notna(temp_delta) else "—",
                      delta_color="inverse" if pd.notna(temp_delta) and abs(temp_delta) > 5 else "normal")

        with col5:
            or_now = safe_last(risk_metrics["overall_risk"])
            rl_now = risk_metrics["risk_level"].iloc[-1]
            badge = "🟢" if str(rl_now) == "Low" else "🟡" if str(rl_now) == "Medium" else "🔴" if str(rl_now) in ["High", "Critical"] else "⚪"
            st.metric("Risk Score", f"{or_now:.1f}%" if pd.notna(or_now) else "N/A", f"{badge} {rl_now}")

        st.markdown("---")

        cA, cB = st.columns(2)
        with cA:
            st.markdown("#### Cell Voltage Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"].tail(168),
                                     y=df["cell_voltage"].tail(168),
                                     mode="lines", name="Cell Voltage"))
            fig.add_hline(y=1.95, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
            fig.add_hline(y=1.90, line_dash="dash", line_color="orange", annotation_text="Warning Level")
            fig.update_layout(xaxis_title="Time", yaxis_title="Voltage (V)",
                              height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with cB:
            st.markdown("#### Efficiency & Production")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"].tail(168),
                                     y=df["efficiency"].tail(168),
                                     mode="lines", name="Efficiency", yaxis="y"))
            fig.add_trace(go.Scatter(x=df["timestamp"].tail(168),
                                     y=df["h2_production_rate"].tail(168),
                                     mode="lines", name="H₂ Production", yaxis="y2"))
            fig.update_layout(
                xaxis_title="Time",
                yaxis=dict(title="Efficiency", side="left"),
                yaxis2=dict(title="H₂ Rate (Nm³/h)", side="right", overlaying="y"),
                height=300, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### System Health Indicators")
        g1, g2, g3 = st.columns(3)
        with g1:
            if "h2_purity" in df.columns and pd.notna(df["h2_purity"].iloc[-1]):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=float(df["h2_purity"].iloc[-1]),
                    title={'text': "H₂ Purity (%)"},
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
            if "o2_in_h2" in df.columns and pd.notna(df["o2_in_h2"].iloc[-1]):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=float(df["o2_in_h2"].iloc[-1]),
                    title={'text': "O₂ in H₂ (ppm)"},
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
            if "hours_since_maintenance" in df.columns and pd.notna(df["hours_since_maintenance"].iloc[-1]):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=float(df["hours_since_maintenance"].iloc[-1]),
                    title={'text': "Hours Since Maintenance"},
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

        # Quick diagnostics for Nixtla & XGBoost integration
        with st.expander("Integration diagnostics"):
            st.write({
                "nixtla_sdk_installed": HAS_NIXTLA,
                "api_key_loaded": bool(_get_api_key()),
                "xgboost_available": HAS_XGB,
                "rows_in_df": int(df.shape[0])
            })

        if st.button("Generate Predictions", type="primary"):
            with st.spinner(f"Running {model_type}..."):
                if model_type == "Nixtla TimeGPT":
                    st.session_state.predictions = forecast_timegpt(df, forecast_horizon)
                elif model_type == "Statistical Ensemble":
                    st.session_state.predictions = forecast_stat_ensemble(df, forecast_horizon)
                elif model_type == "XGBoost ML":
                    st.session_state.predictions = forecast_xgboost(df, forecast_horizon)
                else:  # Hybrid Approach
                    st.session_state.predictions = forecast_hybrid(df, forecast_horizon)

        if st.session_state.predictions is not None:
            pred_df = st.session_state.predictions

            # --- Prediction summary cards ---
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                max_voltage = float(np.nanmax(pred_df["predicted_voltage"]))
                st.metric("Max Predicted Voltage", f"{max_voltage:.3f} V",
                          "⚠️ Critical" if max_voltage > 1.95 else "✅ Normal")
            with col2:
                max_prob = float(np.nanmax(pred_df["failure_probability"]) * 100.0)
                st.metric("Max Failure Risk", f"{max_prob:.1f}%",
                          "🔴 High" if max_prob > 50 else "🟢 Low")
            with col3:
                crit = pred_df[pred_df["failure_probability"] > 0.5]
                if not crit.empty:
                    time_to_failure = pd.to_datetime(crit["timestamp"]).min()
                    hours_to_failure = (time_to_failure - pd.to_datetime(df["timestamp"]).max()).total_seconds() / 3600
                    st.metric("Time to Critical", f"{hours_to_failure:.0f} hours", "⏰ Plan Maintenance")
                else:
                    st.metric("Time to Critical", "No Risk", "✅ Safe")
            with col4:
                conf = 100 - (np.nanmean(pred_df["uncertainty"]) * 100)
                st.metric("Model Confidence", f"{conf:.1f}%", "High" if conf > 80 else "Medium")

            st.markdown("---")

            # --- Charts ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"].tail(168), y=df["cell_voltage"].tail(168),
                                     mode="lines", name="Historical"))
            fig.add_trace(go.Scatter(x=pred_df["timestamp"], y=pred_df["predicted_voltage"],
                                     mode="lines", name="Prediction", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(
                x=pred_df["timestamp"].tolist() + pred_df["timestamp"].tolist()[::-1],
                y=pred_df["upper_bound"].tolist() + pred_df["lower_bound"].tolist()[::-1],
                fill="toself", fillcolor="rgba(255,0,0,0.2)",
                line=dict(color="rgba(255,255,255,0)"), name="95% Confidence"
            ))
            fig.add_hline(y=2.0, line_dash="dash", line_color="darkred", annotation_text="Critical Failure Threshold")
            fig.update_layout(title="Cell Voltage Prediction with Uncertainty Bands",
                              xaxis_title="Time", yaxis_title="Cell Voltage (V)",
                              height=400, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Failure Probability Timeline")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_df["timestamp"], y=pred_df["failure_probability"] * 100,
                                     mode="lines+markers", name="Failure Probability"))
            fig.add_hline(y=risk_threshold, line_dash="dash", line_color="orange",
                          annotation_text=f"Alert Threshold ({risk_threshold}%)")
            fig.update_layout(xaxis_title="Time", yaxis_title="Failure Probability (%)", height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Component-wise Failure Risk Analysis")
            components = ["Diaphragm", "Electrode", "Seal", "Vessel", "Sensors"]
            dp = float(df["differential_pressure"].iloc[-1]) if "differential_pressure" in df.columns else 0.0
            temp_now = float(df["electrolyte_temperature"].iloc[-1]) if "electrolyte_temperature" in df.columns else 85.0
            cycles = float(df["cycles_count"].iloc[-1]) if "cycles_count" in df.columns else 0.0
            hours = float(df["hours_since_maintenance"].iloc[-1]) if "hours_since_maintenance" in df.columns else 0.0
            conc = float(df["electrolyte_concentration"].iloc[-1]) if "electrolyte_concentration" in df.columns else 30.0
            comp_risks = []
            for comp in components:
                if comp == "Diaphragm": r = min(100, (dp/50 + (temp_now-85)/10)*30)
                elif comp == "Electrode": r = min(100, cycles/20)
                elif comp == "Seal": r = min(100, hours/30)
                elif comp == "Vessel": r = min(100, max(0, (conc-30))*10)
                else: r = np.random.uniform(10, 40)
                comp_risks.append(max(0, r))
            fig = go.Figure(data=[go.Bar(x=components, y=comp_risks,
                                         text=[f"{r:.1f}%" for r in comp_risks], textposition="auto")])
            fig.update_layout(title="Component Failure Risk Assessment",
                              xaxis_title="Component", yaxis_title="Risk Level (%)", height=300)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Tab 3: Risk Assessment ----------
    with tab3:
        st.markdown("### ⚠️ Comprehensive Risk Assessment Dashboard")
        current_overall_risk = float(risk_metrics["overall_risk"].iloc[-1])
        risk_level = risk_metrics["risk_level"].iloc[-1]

        if str(risk_level) == "Critical":
            st.error(f"🚨 **CRITICAL RISK** - Overall Risk: {current_overall_risk:.1f}%")
        elif str(risk_level) == "High":
            st.warning(f"⚠️ **HIGH RISK** - Overall Risk: {current_overall_risk:.1f}%")
        elif str(risk_level) == "Medium":
            st.info(f"ℹ️ **MEDIUM RISK** - Overall Risk: {current_overall_risk:.1f}%")
        else:
            st.success(f"✅ **LOW RISK** - Overall Risk: {current_overall_risk:.1f}%")

        st.markdown("#### Risk Factor Breakdown")
        c1, c2 = st.columns(2)
        values = [
            float(risk_metrics["voltage_risk"].iloc[-1]),
            float(risk_metrics["crossover_risk"].iloc[-1]),
            float(risk_metrics["thermal_risk"].iloc[-1]),
            float(risk_metrics["maintenance_risk"].iloc[-1]),
        ]
        with c1:
            categories = ["Voltage\nDegradation", "Gas\nCrossover", "Thermal\nStress", "Maintenance\nUrgency"]
            fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill="toself", name="Current Risk Profile"))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                              showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"].tail(168), y=risk_metrics["overall_risk"].tail(168),
                                     mode="lines", name="Overall Risk"))
            fig.add_hline(y=75, line_dash="dash", line_color="darkred", annotation_text="Critical")
            fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="High")
            fig.add_hline(y=25, line_dash="dash", line_color="yellow", annotation_text="Medium")
            fig.update_layout(title="Risk Score Trend (Last 7 Days)", xaxis_title="Time", yaxis_title="Risk Score (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Detailed Risk Report")
        risk_report = pd.DataFrame({
            "Risk Factor": ["Voltage Degradation", "Gas Crossover", "Thermal Stress", "Maintenance Urgency"],
            "Current Value": [
                f"{df['cell_voltage'].iloc[-1]:.3f} V",
                f"{df['o2_in_h2'].iloc[-1]:.0f} ppm",
                f"{df['electrolyte_temperature'].iloc[-1]:.1f} °C",
                f"{df['hours_since_maintenance'].iloc[-1]:.0f} h",
            ],
            "Risk Score": [f"{v:.1f}%" for v in values],
            "Status": ["🔴 Critical" if v > 75 else "🟠 High" if v > 50 else "🟡 Medium" if v > 25 else "🟢 Low" for v in values],
            "Recommended Action": [
                "Immediate electrode inspection" if values[0] > 75 else "Monitor closely" if values[0] > 50 else "Routine monitoring",
                "Check diaphragm integrity" if values[1] > 75 else "Verify gas analyzers" if values[1] > 50 else "Normal operation",
                "Adjust cooling system" if values[2] > 75 else "Check temperature control" if values[2] > 50 else "Maintain current settings",
                "Schedule immediate maintenance" if values[3] > 75 else "Plan maintenance soon" if values[3] > 50 else "Continue operation",
            ],
        })
        st.dataframe(risk_report, use_container_width=True, hide_index=True)

        st.markdown("#### Historical Incident Analysis (example)")
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

        hours_operated = float(df["hours_since_maintenance"].iloc[-1]) if "hours_since_maintenance" in df.columns else 0.0
        cycles_completed = float(df["cycles_count"].iloc[-1]) if "cycles_count" in df.columns else 0.0
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Hours Since Last Maintenance", f"{hours_operated:.0f} hrs")
        with c2: st.metric("Cycles Completed", f"{cycles_completed:.0f}")
        with c3:
            next_scheduled = 2000 - hours_operated
            st.metric("Hours to Scheduled Maintenance", f"{max(0, next_scheduled):.0f} hrs")

        st.markdown("---")
        current_overall_risk = float(risk_metrics["overall_risk"].iloc[-1])
        cell_v_now = float(df["cell_voltage"].iloc[-1])
        if current_overall_risk > 75 or cell_v_now > 1.95:
            urgency, color, icon, eta, downtime = "IMMEDIATE", "red", "🚨", "Within 4 hours", "4–6 hours"
        elif current_overall_risk > 50:
            urgency, color, icon, eta, downtime = "SCHEDULED", "orange", "⚠️", "Within 48 hours", "2–4 hours"
        elif current_overall_risk > 25:
            urgency, color, icon, eta, downtime = "PLANNED", "yellow", "📅", "Within 1 week", "2–3 hours"
        else:
            urgency, color, icon, eta, downtime = "ROUTINE", "green", "✅", "As scheduled", "1–2 hours"

        st.markdown(f"""<div style="background-color:{color};opacity:0.1;padding:20px;border-radius:10px;"></div>""",
                    unsafe_allow_html=True)
        a1, a2 = st.columns([1, 2])
        with a1:
            st.markdown(f"### {icon} **{urgency}**")
            st.markdown(f"**Timeline:** {eta}")
            st.markdown(f"**Est. Downtime:** {downtime}")
        with a2:
            st.markdown("**Recommended Actions:**")
            v_risk = float(risk_metrics["voltage_risk"].iloc[-1])
            x_risk = float(risk_metrics["crossover_risk"].iloc[-1])
            t_risk = float(risk_metrics["thermal_risk"].iloc[-1])
            m_risk = float(risk_metrics["maintenance_risk"].iloc[-1])
            actions = []
            if v_risk > 50: actions += ["• Inspect electrode coating for degradation", "• Measure individual cell voltages"]
            if x_risk > 50: actions += ["• Check diaphragm integrity", "• Verify gas analyzer calibration"]
            if t_risk > 50: actions += ["• Inspect cooling system performance", "• Check electrolyte circulation"]
            if m_risk > 50: actions += ["• Replace worn gaskets and seals", "• Clean and recalibrate sensors"]
            if not actions: actions = ["• Routine visual inspection", "• Record operational parameters"]
            for a in actions[:6]: st.markdown(a)

        st.markdown("---")
        st.markdown("#### 📅 Optimized Maintenance Schedule")
        maintenance_tasks = pd.DataFrame({
            "Task": ["Electrode Inspection", "Diaphragm Check", "Electrolyte Analysis", "Sensor Calibration", "Seal Replacement", "System Flush"],
            "Priority": ["High", "High", "Medium", "Medium", "Low", "Low"],
            "Estimated Duration": ["2 hrs", "3 hrs", "1 hr", "1 hr", "4 hrs", "2 hrs"],
            "Last Performed": [(datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d") for d in [30, 45, 14, 7, 90, 60]],
            "Next Due": [(datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d") for d in [5, 3, 16, 23, 10, 30]],
            "Status": ["⚠️ Due Soon", "🔴 Overdue", "✅ On Schedule", "✅ On Schedule", "⚠️ Due Soon", "✅ On Schedule"],
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
            for k, v in {"Labor": "$3,000", "Parts & Materials": "$5,000",
                         "Production Loss": "$4,000", "Testing & Validation": "$1,000",
                         "Total Cost": "$13,000"}.items():
                st.markdown(f"**{k}: {v}**" if k == "Total Cost" else f"• {k}: {v}")
        st.success("**Net Benefit: $25,000** (ROI: 192%)")

        st.markdown("---")
        st.button("📄 Generate Maintenance Report", type="primary")

else:
    # Landing page when no data is loaded
    st.info("👆 Please upload electrolyzer data or click **Use Demo Data** in the sidebar to begin.")
    st.markdown("---")
    st.markdown("### 🎯 System Capabilities")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🔮 Failure Prediction**
        - TimeGPT & ML forecasting
        - Component-wise risk assessment
        - 24–168 hour horizon
        - Uncertainty & alerting
        """)
    with col2:
        st.markdown("""
        **⚠️ Risk Assessment**
        - Multi-factor risk scoring
        - Real-time anomaly signals
        - Incident analytics
        - Automated alerts
        """)
    with col3:
        st.markdown("""
        **📋 Maintenance Planning**
        - Optimized scheduling
        - Cost–benefit analysis
        - Spare parts overview
        - Downtime minimization
        """)

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <small>
    Green Hydrogen Electrolyzer Predictive Maintenance System v1.1<br>
    Powered by Nixtla TimeGPT & Advanced Analytics<br>
    ACWA Power Challenge Solution 2024
    </small>
</div>
""", unsafe_allow_html=True)
