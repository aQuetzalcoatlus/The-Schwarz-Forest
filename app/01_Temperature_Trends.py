from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Black Forest Temperature Trends (1990-2024)", layout="wide"
)

DATA_PATH = Path(__file__).parents[1] / "data" / "processed" / "dwd_bw_annual.parquet"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    # Precompute 5-year rolling mean for display
    df = df.sort_values(["station", "year"])
    df["roll5"] = df.groupby("station")["tmean_c"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    return df


st.title("Temperature Trends in the Black Forest (1990-2024)")
st.caption(
    "Data: Deutscher Wetterdienst (DWD) Climate Data Center, monthly KL → annual mean."
)

df = load_data()
year_min, year_max = int(df.year.min()), int(df.year.max())

with st.sidebar:
    stations = st.multiselect(
        "Stations",
        options=sorted(df.station.unique()),
        default=sorted(df.station.unique()),
    )
    years = st.slider("Year range", year_min, year_max, (max(1990, year_min), year_max))
    show_roll = st.checkbox("Show 5-year rolling mean", value=True)
    show_raw = st.checkbox("Overlay raw annual means", value=False)

sub = df[(df["station"].isin(stations)) & (df["year"].between(*years))].copy()

if sub.empty:
    st.warning("No data for the current selection.")
    st.stop()

# Prepare wide table for chart
if show_roll:
    y = "roll5"
else:
    y = "tmean_c"

wide = sub.pivot(index="year", columns="station", values=y).sort_index()
st.line_chart(wide)

# Optional overlay raw annual means as a table below chart (keeps chart light)
if show_roll and show_raw:
    st.caption("Raw annual means (for reference)")
    st.line_chart(
        sub.pivot(index="year", columns="station", values="tmean_c").sort_index()
    )


# Compute simple OLS slope in °C/decade for the selected window
def slope_c_per_decade(g: pd.DataFrame) -> float:
    m = np.polyfit(g["year"], g["tmean_c"], 1)[0]  # °C per year
    return m * 10


slopes = (
    sub.groupby("station")
    .apply(slope_c_per_decade)
    .rename("slope_°C_per_decade")
    .reset_index()
    .sort_values("slope_°C_per_decade", ascending=False)
)

st.subheader("Trend summary (OLS on selected years)")
st.dataframe(slopes, use_container_width=True)

# Download filtered data
csv_bytes = sub[["station", "year", "tmean_c", "roll5"]].to_csv(index=False).encode()
st.download_button(
    "Download current selection (CSV)",
    data=csv_bytes,
    file_name="bf_temp_selection.csv",
)

st.markdown("---")
st.markdown(
    "Notes: Annual mean is the mean of monthly KL 'TT' (air temperature, °C). "
    "Years with fewer than 10 valid months are excluded. Linear trends shown are simple OLS fits; "
    "interpretation is illustrative."
)
