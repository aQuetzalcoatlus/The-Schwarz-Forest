from pathlib import Path

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
from shapely.ops import unary_union

# ---------------------------
# CONFIG
# ---------------------------

CWD = Path.cwd()
print(CWD)
DATA_DIR = CWD / "data" / "gfc"
TREE_PATH = DATA_DIR / "Hansen_GFC-2024-v1.12_treecover2000_50N_000E.tif"
LOSS_PATH = DATA_DIR / "Hansen_GFC-2024-v1.12_lossyear_50N_000E.tif"

LANDMARKS_DICT = {
    # label           # geocoding query string
    "Feldberg": "Feldberg peak, Schwarzwald, Baden-Württemberg, Germany",
    "Schauinsland": "Schauinsland, Baden-Württemberg, Germany",
    "Titisee": "Titisee lake, Baden-Württemberg, Germanyy",
    "Schluchsee": "Schluchsee lake, Black Forest, Germany",
    "Freiburg im Breisgau": "Freiburg im Breisgau, Germany",
    "Waldkirch": "Waldkirch, Baden-Württemberg, Germany",
    "Kirchzarten": "Kirchzarten, Baden-Württemberg, Germany",
    "Triberg": "Triberg, Black Forest, Germany",
}


# LANDMARKS_DICT = {
#     # Major peaks of the Black Forest
#     "Feldberg": ("node", 240908624),  # Feldberg peak, highest in Schwarzwald
#     "Schauinsland": ("node", 223671757),  # Schauinsland peak
#     # Lakes (large, iconic)
#     "Titisee": ("node", 3980758466),  # Titisee lake
#     "Schluchsee": ("node", 100158442),  # Schluchsee lake
#     # Towns / settlements
#     "Freiburg im Breisgau": ("node", 21769883),
#     "Waldkirch": ("node", 13069744919),
#     "Kirchzarten": ("node", 2394885847),
#     # Tourist locality
#     "Triberg": ("node", 3330068297),  # famous waterfalls town
# }


def geocode_osm_id(osm_type: str, osm_id: int, target_crs) -> gpd.GeoDataFrame:
    """
    Fetch a single OSM feature by type/id and project to target_crs.

    osm_type: 'node', 'way', or 'relation'
    osm_id:  OSM integer ID
    """
    gdf = ox.geocode_to_gdf(f"{osm_type}/{osm_id}")
    return gdf.to_crs(target_crs)


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


# @st.cache_data(show_spinner=True)
# def load_landmarks(_crs):
#     rows = []
#     for name in LANDMARK_NAMES:
#         try:
#             gdf = ox.geocode_to_gdf(name)
#             gdf = gdf.to_crs(_crs)
#             gdf["label"] = name.split(",")[0]
#             rows.append(gdf[["label", "geometry"]])
#         except Exception:
#             continue
#     if not rows:
#         return gpd.GeoDataFrame(columns=["label", "geometry"], crs=_crs)
#     landmarks = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), crs=rows[0].crs)
#     return landmarks


@st.cache_data(show_spinner=True)
def load_landmarks(_crs):
    """
    Load landmarks from LANDMARKS_DICT using name-based geocoding.

    Returns a GeoDataFrame with columns: ['label', 'geometry'] in _crs.
    """
    rows: list[gpd.GeoDataFrame] = []

    for label, query in LANDMARKS_DICT.items():
        try:
            gdf = ox.geocode_to_gdf(query)
            gdf = gdf.to_crs(_crs)

            # keep just one geometry (first result) and attach our short label
            gdf = gdf.iloc[[0]][["geometry"]].copy()
            gdf["label"] = label
            gdf["geometry"] = gdf.geometry.centroid
            rows.append(gdf[["label", "geometry"]])
        except Exception as e:
            # you can log this once while debugging:
            # st.write(f"Failed to geocode {label}: {e}")
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
    fig, ax = plt.subplots(figsize=(6, 8), dpi=200)
    rio_show(
        category_map,
        transform=transform,
        ax=ax,
        cmap=ListedColormap(["white", "green", "red"]),
    )
    schwarzwald_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)

    if (landmarks is not None) and (len(landmarks) > 0):
        # sw_union = schwarzwald_gdf.unary_union
        # landmarks_in = landmarks[landmarks.geometry.intersects(sw_union)]

        if not landmarks.empty:
            # 👉 convert everything to point centroids BEFORE plotting
            # landmark_points = landmarks_in.copy()
            landmark_points = landmarks.copy()
            landmark_points["geometry"] = landmark_points.geometry.centroid

            # small point markers, no polygons
            landmark_points.plot(
                ax=ax, color="purple", marker="x", markersize=2, zorder=4
            )

            # labels at the same centroids
            for _, row in landmark_points.iterrows():
                pt = row.geometry
                ax.annotate(
                    row["label"],
                    xy=(pt.x, pt.y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
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


def plot_area_bar(area2000, area2024):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
    ax.bar(["2000", "2024"], [area2000, area2024], color=["darkgreen", "limegreen"])
    ax.set_ylabel("Forest area [hectares]")
    ax.set_title("Forest area in Schwarzwald\n(≥ canopy threshold)")
    return fig


def plot_loss_timeseries(years, loss_ha):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=200)
    ax.bar(years, loss_ha)
    ax.set_xlabel("Year")
    ax.set_ylabel("Forest loss [hectares]")
    ax.set_title("Annual forest loss in Schwarzwald (Hansen GFC)")
    ax.grid(axis="y", alpha=0.3)
    return fig


# ---------------------------
# STREAMLIT APP
# ---------------------------


def main():
    st.set_page_config(page_title="Schwarzwald Forest Change Explorer", layout="wide")

    st.title("Schwarzwald Forest Change Explorer (2000-2024)")

    st.markdown(
        """
        ## Data source

        This app uses the **Hansen Global Forest Change (GFC 2024 v1.12)** dataset,
        derived from Landsat satellite imagery at 30 m resolution.

        We use two raster files for the tile covering the Black Forest (Schwarzwald):

        - `treecover2000` - tree canopy cover (%) in the year 2000  
        - `lossyear` - year of forest loss, where  
          - 0 = no loss detected  
          - 1 = loss in 2001, 2 = loss in 2002, ..., 24 = loss in 2024  

        A pixel is considered **forest** if its canopy cover is above a chosen threshold.
        Forest is considered **lost** if it was forest in 2000 and has a non-zero loss year.
        """
    )

    st.markdown(
        """
        ## Processing steps (overview)

        To create the map and statistics, the app:

        1. Downloads the **Schwarzwald boundary** from OpenStreetMap.  
        2. Crops the Hansen rasters to the Schwarzwald area.  
        3. Applies a **canopy threshold** to define forest in 2000.  
        4. Marks pixels that stayed forest (no loss) vs pixels that lost forest (2001-2024).  
        5. Calculates forest area and yearly loss, and visualizes them in maps and charts.  
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

    tree_src, loss_src = load_rasters()
    schwarzwald = load_schwarzwald_boundary(tree_src.crs)
    window, transform, tree = compute_schwarzwald_window(tree_src, schwarzwald)
    loss = read_loss_in_window(loss_src, window)
    landmarks = load_landmarks(tree_src.crs)
    st.write("Landmarks loaded:", landmarks)

    forest_2000, forest_2024, lost_mask, persistent_mask = compute_masks(
        tree, loss, threshold
    )
    category_map, cmap = make_category_map(forest_2000, forest_2024)
    area2000, area2024, lost_area, lost_pct = compute_area_stats(
        forest_2000, forest_2024
    )
    years, loss_ha = compute_loss_timeseries(loss, forest_2000)

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
    st.markdown("## Simple analyses")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Forest area 2000 vs 2024")
        fig_area = plot_area_bar(area2000, area2024)
        st.pyplot(fig_area, width="stretch")
        st.write(
            f"- 2000 forest area: **{area2000:,.0f} ha**  \n"
            f"- 2024 forest area: **{area2024:,.0f} ha**  \n"
            f"- Forest lost: **{lost_area:,.0f} ha** (**{lost_pct:.1f}%**)"
        )

    with col2:
        st.markdown("### Annual forest loss (2001-2024)")
        fig_ts = plot_loss_timeseries(years, loss_ha)
        st.pyplot(fig_ts, width="stretch")

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
