# Schwarzwald Forest Change Explorer

*A Streamlit application for visualizing forest dynamics in Germany’s Black Forest (2000-2024)*

This project provides an interactive dashboard for exploring long-term forest change in the **Schwarzwald (Black Forest)** using the [Hansen Global Forest Change (GFC 2024 v1.12) dataset](https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html).
The app visualizes forest cover and forest loss, shows simple statistics describing forest dynamics over the last two decades.

<div style="background-color:#e8f4ff; padding:10px 12px; border-radius:6px; border:1px solid #c4ddff; margin-bottom:16px;">
  <strong>Live app:</strong>
  <a href="https://the-schwarz-forest.streamlit.app/" target="_blank">
    https://the-schwarz-forest.streamlit.app/
  </a>
</div>
---

## 🚀 Features

* **Interactive forest change map**
  Visualizes persistent forest, forest lost between 2001–2024, and key landmarks across the Schwarzwald.

* **Flexible canopy threshold**
  A slider lets users choose what counts as “forest” based on canopy cover (%).

* **Dual-unit support (ha / km²)**
  Users can toggle between hectares and square kilometres for all area calculations.

* **Yearly forest loss time series**
  Interactive Altair chart with hover-based tooltips.


---

## 📁 Data workflow

The Hansen dataset provides global 30 m resolution `.tif` tiles.
To keep file sizes manageable in this repository:

* The full 10° × 10° source rasters were **clipped locally** to the Schwarzwald region.
* Clipping is **lossless**:

  * Pixel values are unchanged
  * Resolution remains 30 m
  * No resampling or reprojection
  * CRS and datatypes are preserved
* Compression uses DEFLATE with tiling for efficient reading.

The repo stores the clipped versions:

```
data/gfc/treecover_schwarzwald_noDS.tif
data/gfc/lossyear_schwarzwald_noDS.tif
```

Clipping was performed using `rasterio.mask.mask()`.

---

## 🧱 Technical stack

* **Streamlit** - application framework
* **Rasterio** - raster I/O and spatial masking
* **GeoPandas** - handling geospatial vector layers
* **OSMnx** - geocoding Schwarzwald boundary + landmark lookup
* **Altair** - interactive charts
* **Matplotlib** - base map rendering
* **Shapely** - geometric operations

---

## ▶️ Running the app

### 1. Clone the repo

```bash
git clone https://github.com/aQuetzalcoatlus/The-Schwarz-Forest
cd The-Schwarz-Forest
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

If you use `uv`, you can do (to make use of the `pyproject.toml` file):

```bash
uv sync
```

### 3. Run the Streamlit app

```bash
streamlit run app/schwarz-forest-app.py
```

and using `uv`:

```bash
uv run streamlit run app/schwarz-forest-app.py
```

The app will open automatically in your browser.

---

## 📘 Notes on reproducibility

* All clipped GeoTIFFs were produced from official GFC 2024 v1.12 sources.
* Landmarks are geocoded dynamically via OpenStreetMap; exact coordinates may change over time.
* Schwarzwald boundary uses OSM’s named region ("Schwarzwald, Germany"), which corresponds to a geographic, not administrative, boundary.

---
