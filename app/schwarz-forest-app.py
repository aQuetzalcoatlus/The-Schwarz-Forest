from pathlib import Path

import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
import streamlit as st
from matplotlib.colors import ListedColormap
from rasterio.mask import mask
from rasterio.plot import show as rio_show
from rasterio.windows import from_bounds
from shapely.geometry import Point
from shapely.ops import unary_union

# ---------------------------
# CONFIG
# ---------------------------

CWD = Path.cwd()
print(CWD)
DATA_DIR = CWD / "data" / "gfc"
# TREE_PATH = DATA_DIR / "Hansen_GFC-2024-v1.12_treecover2000_50N_000E.tif"
# LOSS_PATH = DATA_DIR / "Hansen_GFC-2024-v1.12_lossyear_50N_000E.tif"
# The above two original files were used to get the "clipped" tif files below. They are too large to be uploaded to git.
# Locally, this was done in the `test-notebook.ipynb`
TREE_PATH = DATA_DIR / "treecover_schwarzwald_noDS.tif"
LOSS_PATH = DATA_DIR / "lossyear_schwarzwald_noDS.tif"

LANDMARKS_DICT = {
    "Feldberg": "Feldberg, Baden-Württemberg, Germany",
    "Schauinsland": "Schauinsland, Freiburg im Breisgau, Germany",
    "Titisee": "Titisee, Titisee-Neustadt, Germany",
    "Schluchsee": "Schluchsee, Baden-Württemberg, Germany",
    "Freiburg im Breisgau": "Freiburg im Breisgau, Germany",
    "Waldkirch": "Waldkirch, Baden-Württemberg, Germany",
    "Kirchzarten": "Kirchzarten, Baden-Württemberg, Germany",
    "Triberg": "Triberg im Schwarzwald, Germany",
}


# ---------------------------
# DATA LOADING HELPERS
# ---------------------------


@st.cache_resource(show_spinner=True)
def load_rasters() -> tuple:
    tree_src = rasterio.open(TREE_PATH)
    loss_src = rasterio.open(LOSS_PATH)
    return tree_src, loss_src


@st.cache_data(show_spinner=True)
def load_schwarzwald_boundary(_crs):
    sw = ox.geocode_to_gdf("Schwarzwald, Germany")
    sw = sw.to_crs(_crs)
    return sw


@st.cache_data(show_spinner=True)
def load_landmarks(_crs):
    """
    Load landmarks from LANDMARKS_DICT using name-based geocoding.

    Returns a GeoDataFrame with columns: ['label', 'geometry'] in _crs.
    """
    rows: list[gpd.GeoDataFrame] = []

    for label, query in LANDMARKS_DICT.items():
        try:
            # ox.geocode returns (lat, lon) tuple
            lat, lon = ox.geocode(query)

            gdf = gpd.GeoDataFrame(
                {"label": [label]},
                geometry=[Point(lon, lat)],  # shapely wants (x=lon, y=lat)
                crs="EPSG:4326",
            ).to_crs(_crs)

            rows.append(gdf[["label", "geometry"]])
        except Exception as e:
            st.write(f"Geocoding FAILED for {label!r} with query {query!r}: {e}")
            continue

    if not rows:
        return gpd.GeoDataFrame(columns=["label", "geometry"], crs=_crs)

    landmarks = gpd.GeoDataFrame(
        pd.concat(rows, ignore_index=True),
        crs=_crs,
    )
    return landmarks


def compute_schwarzwald_window(tree_src, schwarzwald_gdf):
    sw_bounds = schwarzwald_gdf.total_bounds
    window = from_bounds(
        sw_bounds[0],
        sw_bounds[1],
        sw_bounds[2],
        sw_bounds[3],
        transform=tree_src.transform,
    )
    transform = tree_src.window_transform(window)
    tree = tree_src.read(1, window=window)
    return window, transform, tree


def read_loss_in_window(loss_src, window):
    loss = loss_src.read(1, window=window)
    return loss


# ---------------------------
# ANALYSIS HELPERS
# ---------------------------


def convert_units(value_ha, unit_key: str):
    """Convert a value in hectares to the selected unit."""
    if unit_key == "km²":
        return value_ha / 100.0  # 1 km² = 100 ha
    return value_ha  # "ha" case


def unit_label(unit_key: str) -> str:
    return "ha" if unit_key == "ha" else "km²"


def compute_masks(tree, loss, threshold):
    forest_2000 = tree >= threshold
    forest_2024 = (tree >= threshold) & (loss == 0)
    lost_mask = forest_2000 & ~forest_2024
    persistent_mask = forest_2024
    return forest_2000, forest_2024, lost_mask, persistent_mask


def make_category_map(forest_2000, forest_2024):
    lost_mask = forest_2000 & ~forest_2024
    persistent_mask = forest_2024
    category = np.zeros_like(forest_2000, dtype=np.uint8)
    category[persistent_mask] = 1
    category[lost_mask] = 2
    cmap = ListedColormap(["white", "green", "red"])
    return category, cmap


def compute_area_stats(forest_2000, forest_2024):
    pixel_area_ha = 0.09  # 30m x 30m
    n2000 = np.sum(forest_2000)
    n2024 = np.sum(forest_2024)
    area2000 = n2000 * pixel_area_ha
    area2024 = n2024 * pixel_area_ha
    lost = (n2000 - n2024) * pixel_area_ha
    lost_pct = 100 * (n2000 - n2024) / n2000 if n2000 > 0 else 0
    return area2000, area2024, lost, lost_pct


def compute_loss_timeseries(loss, forest_2000):
    # years 1..24 correspond to 2001..2024
    years_index = np.arange(1, 25)
    loss_per_year = []
    pixel_area_ha = 0.09
    for y in years_index:
        count = np.sum((loss == y) & forest_2000)
        loss_per_year.append(count * pixel_area_ha)
    years = 2000 + years_index
    return years, np.array(loss_per_year)


# ---------------------------
# PLOTTING HELPERS
# ---------------------------


def plot_schwarzwald_map(category_map, transform, schwarzwald_gdf, landmarks):
    fig, ax = plt.subplots(figsize=(3 * 1.5, 4 * 1.5), dpi=200)
    rio_show(
        category_map,
        transform=transform,
        ax=ax,
        cmap=ListedColormap(["white", "green", "red"]),
    )
    schwarzwald_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

    if (landmarks is not None) and (len(landmarks) > 0):
        landmark_points = landmarks.copy()  # no centroid line

        landmark_points.plot(ax=ax, color="purple", marker="x", markersize=10, zorder=4)

        for _, row in landmark_points.iterrows():
            pt = row.geometry
            ax.annotate(
                row["label"],
                xy=(pt.x, pt.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=4,
                color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5),
                zorder=5,
            )

    # lock view to Schwarzwald
    minx, miny, maxx, maxy = schwarzwald_gdf.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    ax.set_axis_off()
    ax.set_title(
        "Forest change in the Black Forest (Schwarzwald)\n"
        "Green = persistent forest, Red = forest lost (2001–2024)",
        fontsize=10,
    )
    plt.tight_layout()
    return fig


def plot_area_bar_interactive(area2000, area2024, unit_key: str):
    # Convert units
    area2000_u = convert_units(area2000, unit_key)
    area2024_u = convert_units(area2024, unit_key)
    label = unit_label(unit_key)

    df = pd.DataFrame(
        {
            "year": ["2000", "2024"],
            "area": [area2000_u, area2024_u],
        }
    )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("area:Q", title=f"Forest area [{label}]"),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("area:Q", title=f"Forest area ({label})", format=",.2f"),
            ],
            color=alt.Color(
                "year:N", scale=alt.Scale(range=["darkgreen", "limegreen"])
            ),
        )
        .properties(
            title="Forest area in the Black Forest (≥ canopy threshold)",
            height=300,
        )
        .interactive()
    )

    return chart


def plot_loss_timeseries_interactive(years, loss_ha, unit_key: str):
    loss_u = convert_units(loss_ha, unit_key)
    label = unit_label(unit_key)

    df = pd.DataFrame(
        {
            "year": years,
            "loss": loss_u,
        }
    )

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("loss:Q", title=f"Forest loss [{label}]"),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("loss:Q", title=f"Loss ({label})", format=",.2f"),
            ],
        )
        .properties(
            title="Annual forest loss in Schwarzwald (Hansen GFC)",
            height=300,
        )
        .interactive()
    )

    return chart


# ---------------------------
# STREAMLIT APP
# ---------------------------


def main():
    st.set_page_config(page_title="Schwarzwald Forest Change Explorer", layout="wide")

    st.title("Schwarzwald Forest Change Explorer (2000-2024)")

    st.markdown(
        """
        ## Data source

        The forest maps in this app are based on the **Hansen Global Forest Change (GFC 2024 v1.12)** dataset.
        Each dataset is stored as a **GeoTIFF raster**, which works like a huge grid laid over the landscape:

        - Each **pixel corresponds to a 30×30 metre area** on the ground (900 m² = 0.09 ha).
        - Pixel values encode properties of that exact piece of land.

        To reduce file size, the original global tiles were clipped locally to the Schwarzwald region before use in this app.
        This clipping is *lossless*: **pixel values, resolution, and CRS remain identical to the original data**.

        We use two raster files for the tile covering the Black Forest (Schwarzwald):

        ### 1. `treecover2000`: Tree canopy cover in the year 2000

        Each pixel contains a value between **0 and 100**, representing:

        **Percentage of tree canopy cover in 2000**

        * `0` → no tree cover
        * `25` → 25% canopy
        * `80` → dense forest

        A pixel is classified as “forest” if its canopy cover is **above the selected threshold**.


        ### 2. `lossyear`: Year of forest loss (2001-2024)

        Each pixel contains an integer between **0 and 24**:

        * `0` → no forest loss detected
        * `1` → loss in **2001**
        * `2` → loss in **2002**
        * …
        * `24` → loss in **2024**

        A pixel is counted as **forest lost** only if:

        1. It was forest in 2000 (above threshold), **and**
        2. `lossyear` is non-zero.

        ---

        ## Pixel area and unit conversions

        Because each pixel represents **0.09 hectares**, the app can compute total forest area and loss by simply counting pixels and converting them into:

        * **hectares (ha)** — standard in forestry
        * **square kilometres (km²)** — SI-derived unit (1 km² = 100 ha)

        The user can switch between these units in the sidebar.

        ---
        """
    )

    st.markdown(
        """
        ## Processing steps (overview)

        To create the forest change maps and analyses, the app performs the following steps:

        1. **Load the Schwarzwald boundary** from OpenStreetMap and project it to match the rasters.
        2. **Load the clipped Hansen rasters** (identical to the original data, just spatially trimmed).
        3. **Classify forest in the year 2000** by applying a user-selected canopy threshold.
        4. **Determine forest persistence or loss** by comparing treecover2000 with lossyear
        (0 = no loss, 1-24 = year of loss from 2001-2024).
        5. **Calculate forest area and area lost** by counting pixels and converting to ha or km².
        6. Generate visualizations.
        """
    )

    st.sidebar.header("Map controls")

    threshold = st.sidebar.slider(
        "Canopy threshold for defining forest (%)",
        min_value=10,
        max_value=80,
        value=30,
        step=5,
        help="Pixels with tree cover above this value in 2000 are treated as forest.",
    )

    unit: str = st.sidebar.radio(
        "Display units",
        options=["ha", "km²"],
        index=0,
        format_func=lambda u: "hectares (ha)"
        if u == "ha"
        else "square kilometers (km²)",
        help="Choose whether to show forest area in hectares or square kilometers.",
    )

    tree_src, loss_src = load_rasters()
    schwarzwald = load_schwarzwald_boundary(tree_src.crs)
    window, transform, tree = compute_schwarzwald_window(tree_src, schwarzwald)
    loss = read_loss_in_window(loss_src, window)
    landmarks = load_landmarks(tree_src.crs)
    # st.write("Landmarks loaded:", landmarks)

    forest_2000, forest_2024, lost_mask, persistent_mask = compute_masks(
        tree, loss, threshold
    )
    category_map, cmap = make_category_map(forest_2000, forest_2024)
    area2000, area2024, lost_area, lost_pct = compute_area_stats(
        forest_2000, forest_2024
    )
    years, loss_ha = compute_loss_timeseries(loss, forest_2000)

    # Year with the greatest loss
    correct_year = int(years[np.argmax(loss_ha)])

    st.warning(
        "⚠️ If you don't see a map below, please wait a few seconds for it to get loaded."
    )
    # ---------------- Map ----------------
    st.markdown("## Map: forest change in the Black Forest")
    st.write(
        f"Canopy threshold: **{threshold}%**. "
        f"Estimated forest loss: **{lost_area:,.0f} ha** "
        f"({lost_pct:.1f}% of 2000 forest)."
    )

    fig_map = plot_schwarzwald_map(category_map, transform, schwarzwald, landmarks)
    st.pyplot(fig_map, width="stretch")

    # ---------------- Analyses ----------------
    st.markdown("## From the data:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Forest area 2000 vs 2024")
        chart_area = plot_area_bar_interactive(area2000, area2024, unit)
        st.altair_chart(chart_area, use_container_width=True)

        label = unit_label(unit)

        area2000_u = convert_units(area2000, unit)
        area2024_u = convert_units(area2024, unit)
        lost_area_u = convert_units(lost_area, unit)

        st.write(
            f"- 2000 forest area: **{area2000_u:,.2f} {label}**  \n"
            f"- 2024 forest area: **{area2024_u:,.2f} {label}**  \n"
            f"- Forest lost: **{lost_area_u:,.2f} {label}** "
            f"(**{lost_pct:.1f}%** of 2000 forest)"
        )

    with col2:
        st.markdown("### Annual forest loss (2001-2024)")
        chart_ts = plot_loss_timeseries_interactive(years, loss_ha, unit)
        st.altair_chart(chart_ts, use_container_width=True)

        st.markdown("### Mini quiz")

        selected_year = st.slider(
            "From the above chart, identify the year with the greatest forest loss",
            min_value=int(years[0]),
            max_value=int(years[-1]),
            value=int(years[0]),
            step=1,
            key="quiz_year",
        )

        if st.button("Submit answer", key="quiz_submit"):
            if selected_year == correct_year:
                st.balloons()
                st.success(
                    f"✅ Correct! The highest forest loss in this dataset is in **{correct_year}**. This matches with the results described in this study: https://kommunikation.uni-freiburg.de/pm-en/press-releases-2023/tree-mortality-in-the-black-forest-on-the-rise-climate-change-a-key-driver"
                )
            else:
                st.error(
                    f"Not quite. Try again! (Hint: Hover over each bar in the chart to see the forest loss that year.)"
                )

    st.markdown(
        """
        The time-series bar chart highlights how forest loss changes over time.
        Peaks typically correspond to major disturbance years
        (storms, drought, bark beetle outbreaks, or large clear-cuts).
        """
    )


if __name__ == "__main__":
    # small import needed for concat inside cached function

    main()
