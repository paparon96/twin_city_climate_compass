import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="Twin City Climate Compass", layout="wide")
st.title("Twin City Climate Compass")
st.write(f"A peer group comparison dashboard for climate initiatives of cities")

# Real-world cities — U.S. and Europe
all_cities = ["New York City", "Los Angeles", "Chicago", "London", "Paris", "Berlin", "Madrid"]

# Sidebar configuration
st.sidebar.header("Settings")
base_city = st.sidebar.selectbox("Select Your City", all_cities, index=0)
peer_cities = st.sidebar.multiselect(
    "Select Peer Cities (Controls)",
    [c for c in all_cities if c != base_city],
    default=[c for c in all_cities if c != base_city][:2]
)
use_synthetic = st.sidebar.checkbox("Include Synthetic Control (average of peers)", value=True)

# Time series setup
n_months = 36
dates = pd.date_range(end=pd.Timestamp.today(), periods=n_months, freq="M")

def generate_city_data(city):
    np.random.seed(abs(hash(city)) % (2**32))
    air_pollution = np.cumsum(np.random.randn(n_months) * 2 + 1) + 50
    well_being = np.cumsum(np.random.randn(n_months) * 1.5 + 0.5) + 70
    temperature = 15 + np.sin(np.linspace(0, 2 * np.pi, n_months)) * 10 + np.random.randn(n_months)
    climate_deaths = np.abs(np.random.randn(n_months) * 3 + 15)
    return pd.DataFrame({
        "city": city,
        "date": dates,
        "air_pollution": air_pollution,
        "well_being": well_being,
        "temperature": temperature,
        "climate_deaths": climate_deaths
    })

# Assemble data
all_data = pd.concat([generate_city_data(c) for c in [base_city] + peer_cities], ignore_index=True)

# Synthetic Control
if use_synthetic and peer_cities:
    peers = all_data[all_data["city"].isin(peer_cities)]
    synth = peers.groupby("date").mean(numeric_only=True).reset_index().assign(city="Synthetic Control")
    all_data = pd.concat([all_data, synth], ignore_index=True)

# Metric selection
metric = st.selectbox(
    "Select Metric",
    ["air_pollution", "well_being", "temperature", "climate_deaths"],
    format_func=lambda x: {
        "air_pollution": "Air Pollution",
        "well_being": "Well-Being",
        "temperature": "Temperature (°C)",
        "climate_deaths": "Climate-Related Deaths"
    }[x]
)

# Add trend lines via moving average
all_data['trend'] = all_data.groupby("city")[metric].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

# Anomaly detection via STL decomposition (residual thresholding)
anomalies = pd.DataFrame()
for city in all_data['city'].unique():
    df_city = all_data[all_data['city'] == city].sort_values('date')
    stl = STL(df_city[metric], period=12, robust=True)
    res = stl.fit()
    thr = res.resid.std() * 2  # threshold at ±2 std dev
    df_city['anomaly'] = np.abs(res.resid) > thr
    anomalies = pd.concat([anomalies, df_city[df_city['anomaly']]], ignore_index=True)
all_data = all_data.merge(anomalies[['city','date','anomaly']], on=['city','date'], how='left').fillna({'anomaly': False})

# Charting with Altair
chart = alt.Chart(all_data).mark_line().encode(
    x="date:T",
    y=alt.Y(f"{metric}:Q", title=metric.replace("_"," ").title()),
    color="city:N"
)

trend_line = alt.Chart(all_data).mark_line(strokeDash=[5,5]).encode(
    x="date:T",
    y=alt.Y("trend:Q", title="Trend"),
    color="city:N"
)

anomaly_points = alt.Chart(all_data[all_data['anomaly']]).mark_circle(size=100, color="red").encode(
    x="date:T", y=alt.Y(f"{metric}:Q"), tooltip=["city","date", f"{metric}:Q"]
)

st.altair_chart((chart + trend_line + anomaly_points).interactive(), use_container_width=True)

# Latest values comparison
latest = (
    all_data.sort_values("date").groupby("city").last().reset_index()[["city", metric]]
)
st.subheader("Latest Metric Values")
st.table(latest.set_index("city"))

# Difference from base
base_val = latest.loc[latest['city']==base_city, metric].item()
latest['difference'] = latest[metric] - base_val
st.subheader("Difference from selected base city (Latest)")
st.table(latest.set_index("city")[["difference"]])

# Difference-in-Differences (DiD)
pre_period = dates[:12]
post_period = dates[-12:]
base_pre = all_data[(all_data.city == base_city) & (all_data.date.isin(pre_period))][metric].mean()
base_post = all_data[(all_data.city == base_city) & (all_data.date.isin(post_period))][metric].mean()
peer_pre = all_data[(all_data.city.isin(peer_cities)) & (all_data.date.isin(pre_period))].groupby("city")[metric].mean().mean()
peer_post = all_data[(all_data.city.isin(peer_cities)) & (all_data.date.isin(post_period))].groupby("city")[metric].mean().mean()

did = (base_post - base_pre) - (peer_post - peer_pre)
st.subheader("Difference-in-Differences Estimate")
st.write(f"The Difference-in-Differences estimate of the latest climate action in **{base_city}** for **{metric}** is **{did:.2f}**.")
