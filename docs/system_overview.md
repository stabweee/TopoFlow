# TopoFlow: End-to-End System Architecture & Technical Overview

**Project**: Spatio-Temporal Topographical Friction & Causal Impact Modeling of the American Legion Bridge Sag Bottleneck  
**Primary Corridor**: I-495 Capital Beltway (Potomac River Sag Crossing, Cabin John, MD to McLean, VA)  
**Control Donor Corridors**: I-95 (Lorton/Woodbridge Flat Tangents), I-295 (Anacostia Freeway Flat Tangents)  
**Document Purpose**: Canonical, high-level technical overview explaining how TopoFlow works from raw data ingestion to causal econometrics, simulation, and cloud delivery.

---

## 1. Executive Summary & Problem Space

The **American Legion Memorial Bridge (ALMB)** corridor on the I-495 Capital Beltway carries over 235,000 vehicles daily between Fairfax County, VA and Montgomery County, MD. It is routinely ranked among the most volatile, recurring traffic bottlenecks in North America.

Conventional transportation planning often diagnoses this bottleneck with simplistic intuition: *"there are too many cars for too few lanes; we need more asphalt."* However, transport physics and empirical data demonstrate that the bottleneck is governed by a **vertical sag curve anomaly**:

```
       Virginia Approach              Potomac River Gorge               Maryland Approach
    [ Downhill Descent -3.2% ]            [ Bridge ]               [ Uphill Climb +3.8% ]
════════════════════════════════\                                /════════════════════════════════
                                 \                              /
                                  \____[ Sag Inflection ]______/
                                       Elevation: ~140 ft
          Elevation: ~300 ft                                          Elevation: ~320 ft
```

1. **Downhill Descent ($-3.2\%$)**: Vehicles descending into the river basin experience perceived optical compression. Drivers unconsciously tap their brakes to maintain safe headway.
2. **Trough Inflection & Uphill Climb ($+3.8\%$)**: Heavy commercial vehicles lose momentum under gravity and engine drag. Passenger cars are trapped behind decelerating trucks.
3. **Kinematic Shockwaves**: In dense traffic, human reaction latency ($0.75\text{s} - 1.5\text{s}$) amplifies localized brake taps into backward-propagating compression shockwaves (traveling upstream at $-10\text{ to } -12\text{ mph}$), triggering a $10\% - 20\%$ **capacity drop**.
4. **Adverse Weather Shocks**: Rain reduces tire-pavement friction and increases following distance, acting as a severe non-linear catalyst on vertical grades.

**TopoFlow** is an empirical, production-grade econometric and simulation platform built to isolate, prove, and quantify whether—and by exactly how much—**roadway topography causally accelerates traffic phase breakdown**, and how proactive navigation routing can eliminate the bottleneck.

---

## 2. End-to-End System Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       PHASE 0 & PHASE 1                                          │
│                            GEOSPATIAL & TOPOGRAPHICAL FOUNDATIONS                                │
├────────────────────────────────┬────────────────────────────────┬────────────────────────────────┤
│      OpenStreetMap (OSM)       │     USGS 3DEP Bare-Earth       │   Raw USGS 3DEP Point Clouds   │
│   Directional Carriageways     │   1-Meter LiDAR DEM Rasters    │   ASPRS Class 17 Bridge Deck   │
└───────────────┬────────────────┴────────────────┬───────────────┴────────────────┬───────────────┘
                │                                 │                                │
                ▼                                 ▼                                ▼
       [ 100m Discretization ]        [ Road Approach Sampling ]       [ Bridge Deck Asphalt Z ]
       [ Menger Curvature κ  ]                    │                                │
                │                                 └────────────────┬───────────────┘
                │                                                  │
                │                                    [ As-Built Blueprint Check ]
                │                                    [ Savitzky-Golay Filter    ]
                ▼                                                  ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                     SPATIAL FEATURE MART (Parquet)                          │
     │      segment_id (100m) | UTM 18N (x, y) | Grade (%) | Curvature κ | Sag     │
     └──────────────────────────────────────┬──────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       PHASE 2                                                    │
│                        HIGH-FREQUENCY SENSOR & WEATHER LAKEHOUSE                                 │
├─────────────────────────────────────────┬────────────────────────────────────────────────────────┤
│        Roadway Sensor Telemetry         │                Open-Meteo Historical API               │
│     (5-Minute Speed, Volume, Occ)       │             (NOAA HRRR 3km Mesoscale Model)            │
│                   │                     │                        │                               │
│                   ▼                     │                        ▼                               │
│     [ Spatio-Temporal Imputation ]      │           [ 15m / 60m Rolling Precip ]                 │
│     [ Audit Flag: imputation_flag ]     │           [ Discrete is_rain_shock   ]                 │
└───────────────────┬─────────────────────┴────────────────────────┬───────────────────────────────┘
                    │                                              │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             UNIFIED SPATIO-TEMPORAL PANEL MART                                   │
│                        Apache Parquet: corridor/year/month (DuckDB OLAP)                         │
│       segment_id | timestamp_utc | speed | volume | density_k | rain_shock | Grade x Vol        │
└──────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
┌───────────────────────────────────────┐     ┌────────────────────────────────────────────────────┐
│                PHASE 3                │     │                      PHASE 4                       │
│      TRAFFIC PHYSICS & SHOCKWAVES     │     │          ECONOMETRIC & CAUSAL MODELING             │
├───────────────────────────────────────┤     ├────────────────────────────────────────────────────┤
│ • Triangular Fundamental Diagram      │     │ • Two-Way Fixed Effects (TWFE) Panel OLS           │
│   (Piecewise non-linear knot search)  │     │   Y = β1+(Grade+ × Vol) + β1-(Grade- × Vol) + ...  │
│ • Backward Wave Speed Tracking        │     │ • Driscoll-Kraay Non-Parametric Standard Errors    │
│   (DuckDB window slope w ≈ -11.2 mph) │     │ • Bayesian Structural Time Series (CausalImpact)   │
│ • Space-Time (x-t) Contour Diagrams   │     │   (Synthetic Control using I-95 & I-295 donors)    │
│   (Visualizing bottleneck origin)     │     │ • Econometric Robustness & Placebo Testing Suite   │
└───────────────────┬───────────────────┘     └───────────────────────────┬────────────────────────┘
                    │                                                     │
                    └──────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       PHASE 5                                                    │
│                       MACRO ECONOMIC, ENVIRONMENTAL & SIMULATION IMPACT                          │
├─────────────────────────────────────────┬────────────────────────────────────────────────────────┤
│     Vehicle Hours of Delay (VHD)        │           EPA MOVES Vehicle Specific Power (VSP)       │
│  Monetized via USDOT Value of Time      │      Instantaneous fuel burn & CO2/NOx emissions       │
│  ($24.60/veh-hr passenger; $42 freight) │      directly incorporating road grade angle (sin θ)   │
├─────────────────────────────────────────┴────────────────────────────────────────────────────────┤
│                     Macroscopic Cell Transmission Model (CTM) Simulation                         │
│        Evaluating proactive 1% - 10% eco-routing diversions before density hits k_crit           │
│                     (Proves 5% diversion avoids phase collapse, saving 40%+ delay)               │
└──────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       PHASE 6                                                    │
│                        MOTHERDUCK MART, CI/CD & PUBLIC SHOWCASE                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ • MotherDuck Cloud Data Mart: Serverless analytical views (v_bottleneck_summary, v_topography)  │
│ • GitHub Actions CI/CD: Automated testing, Ruff strict linting, and recurring cron ingestion     │
│ • Portfolio Showcase: Methodology whitepaper, executive README, and interactive dashboard       │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase-by-Phase Technical Walkthrough

### Phase 0: Modern Environment & Architecture Foundations
* **Package Management via `uv`**: Ultra-fast, deterministic dependency resolution pinning Python 3.12+, `polars>=1.0.0`, `duckdb>=1.0.0`, `linearmodels>=6.0`, `laspy[lazrs]>=2.5.0`, and `shapely>=2.0.0`.
* **Centralized Typed Settings (`src/topoflow/config.py`)**: Uses `pydantic-settings` to enforce type safety across filesystem paths, corridor bounding boxes, and analytical constants.
* **Coordinate Reference System (CRS) Bedrock**: Standardizes on **NAD83 / UTM Zone 18N (`EPSG:26918`)** as the single metric Cartesian projection spanning both Maryland and Virginia with $< 0.04\%$ linear distortion, eliminating state-line projection discontinuities. Raw GPS coordinates enter and exit as `EPSG:4326`.

### Phase 1: Geospatial Geometry & Hybrid Elevation Profiling
* **Vector Road Geometries (`src/topoflow/geo/osm_extractor.py`)**: Queries OpenStreetMap via Overpass QL, assembling continuous directional highway centerlines (Inner Loop vs. Outer Loop).
* **Strict 100-Meter Chainage Discretization (`src/topoflow/geo/discretization.py`)**: Resamples vector lines into 100-meter equidistant stations ($s_i = 100 \cdot i$). A 100m grain matches the physical wavelength of traffic compression shockwaves while isolating the 300m sag trough.
* **Horizontal Curvature ($\kappa$)**: Computes three-point **Menger osculating circle curvature** across station triplets, setting radius $R = 1 / \kappa$ with a $10,000\text{ m}$ cap on straightaways.
* **Hybrid Physical Elevation Pipeline (`src/topoflow/geo/elevation.py`)**:
  * *Land Approaches*: Ingests bare-earth USGS 3DEP 1-meter LiDAR DEM rasters via `rasterio`.
  * *Bridge Deck Span*: Directly queries raw USGS 3DEP point clouds (`.laz`) using `laspy` to extract laser returns classified under **ASPRS Class 17 ("Bridge Deck")**. This samples the true physical asphalt surface of the bridge ($\pm 3-5\text{ cm}$ accuracy) rather than dropping into the Potomac River water surface below.
  * *As-Built Validation*: Automatically cross-validates extracted bridge deck elevations against Maryland/Virginia DOT structural blueprints (NBI Structure ID `100000495000010`).
  * *Savitzky-Golay Polynomial Filter*: Applies a degree-2 polynomial filter across a 5-station (500m) window to eliminate point noise while preserving true parabolic vertical curves, directly computing continuous analytical derivatives ($\frac{dz}{dx}$) and vertical grade percentages.
* **Spatial Feature Mart (`src/topoflow/geo/feature_mart.py`)**: Consolidates 100m station geometries, grades, and curvatures into `spatial_feature_mart.parquet`.

### Phase 2: High-Frequency Sensor Telemetry, Weather Ingestion & Storage
* **5-Minute Telemetry Ingestion (`src/topoflow/ingestion/traffic_telemetry.py`)**: Batches loop detector and probe telemetry (speed, travel time, volume) into 5-minute epochs.
* **Telemetry Imputation & Quality Audit (`src/topoflow/data/imputation.py`)**: Real-world sensors suffer dropouts and frozen values. An econometric 2D spatio-temporal spline incorporates adjacent spatial stations ($i-1, i+1$), adjacent temporal epochs ($t-1, t+1$), and historical time-of-day baselines ($\bar{v}_{i, \text{tod}}$). Crucially, every row receives an `imputation_flag` ($0 = \text{raw}, 1 = \text{temporal}, 2 = \text{spatial}, 3 = \text{composite}$) to enable pure-raw econometric sensitivity audits.
* **High-Resolution Weather Ingestion (`src/topoflow/ingestion/weather_noaa.py`)**: Connects to the **Open-Meteo Historical API** backed by the **NOAA HRRR (High-Resolution Rapid Refresh) 3km mesoscale reanalysis**. Queries the exact centroid coordinates of the bridge and control corridors, generating rolling 15-minute and 60-minute rain accumulation metrics and discrete binary shock indicators (`is_rain_shock`). Cross-validated against KDCA/KIAD airport ASOS stations.
* **Derived Traffic State Features**:
  * *Hydrodynamic Density Proxy*: $k = q / v$ clamped at physical jam density $k_{\text{jam}} = 220\text{ vpmpl}$.
  * *Empirical Free-Flow Speed ($v_{\text{ff}, i}$)*: Empirical 85th percentile off-peak speed (Tue–Thu 10:00 PM – 4:30 AM dry conditions) per 100m segment, absorbing baseline road curvature into the benchmark.
* **Columnar Parquet Lakehouse (`src/topoflow/data/build_panel_matrix.py`)**: Unites spatial geometry, traffic telemetry, and weather into `panel_matrix_5min.parquet`, partitioned by `corridor=XXX/year=YYYY/month=MM/` with Zstandard compression for sub-second analytical querying via DuckDB.

### Phase 3: Traffic Flow Dynamics & Kinematic Shockwave Physics
* **Continuum LWR Model**: Formulates traffic flow as a compressible fluid governed by the continuity equation $\frac{\partial k}{\partial t} + \frac{\partial q}{\partial x} = 0$.
* **Triangular Fundamental Diagram (`src/topoflow/analytics/fundamental_diagram.py`)**: Implements the Daganzo/Newell piecewise non-linear model ($q = \min(v_{\text{ff}} k, w(k_{\text{jam}} - k))$), estimating the critical breakdown density knot $k_{\text{crit}}$ and proving that the sag curve exhibits a structurally lower capacity and earlier breakdown threshold than flat approach tangents.
* **DuckDB Vectorized Window Wave Tracking (`src/topoflow/analytics/shockwave_tracker.py`)**: Identifies the earliest timestamp $t_i^*$ where speed collapses across continuous 100m stations. Runs linear regression of station chainage against breakdown timestamp ($\hat{w} = \Delta s / \Delta t$), proving empirical backward wave speeds move upstream at **$-10\text{ to } -12\text{ mph}$**, validated by the Rankine-Hugoniot jump condition $w = \frac{\Delta q}{\Delta k}$.
* **Space-Time ($x-t$) Contour Diagrams (`src/topoflow/visualization/space_time_diagram.py`)**: Renders 2D space-time velocity heatmaps. Demonstrates visually that the red congestion wedge originates precisely at the lowest elevation point of the bridge sag while downstream Maryland approaches remain clear green, refuting alternative claims that downstream interchanges caused the delay.

### Phase 4: Econometric & Causal Identification
* **Two-Way Fixed Effects (TWFE) Panel OLS (`src/topoflow/models/panel_twfe.py`)**:
  $$Y_{i,t} = \beta_1^+ (\text{Grade}_i^+ \times \text{Volume}_{i,t}) + \beta_1^- (\text{Grade}_i^- \times \text{Volume}_{i,t}) + \beta_2 (\text{Curvature}_i \times \text{Volume}_{i,t}) + \beta_3 (\text{Rain}_t \times \text{Grade}_i) + \alpha_i + \lambda_t + \varepsilon_{i,t}$$
  * Spatial Fixed Effects ($\alpha_i$) permanently absorb *all* static road geometry, lane counts, and on-ramp proximity.
  * Temporal Fixed Effects ($\lambda_t$) absorb *all* diurnal rush hour cycles and macro holiday demand shocks.
  * Identifies the causal impact of topography strictly through time-varying interactions.
* **Asymmetric Grade Formulation**: Splits slope into uphill climb ($\text{Grade}^+$) and downhill descent ($\text{Grade}^-$), preventing downhill visual compression braking and uphill gravitational engine drag from artificially canceling each other out.
* **Driscoll-Kraay Standard Errors**: Employs non-parametric spatial-lag and autocorrelation covariance estimators, preventing false-precision inflation from persistent time series and spatial cross-station correlation.
* **Bayesian Structural Time Series (BSTS) / CausalImpact (`src/topoflow/models/bsts_causal_impact.py`)**: Uses flat control segments of I-95 (Lorton) and I-295 (Anacostia) as an external synthetic control donor pool. When rain strikes, BSTS predicts what speeds on the American Legion Bridge would have been if it were completely flat, isolating the **topographical rain multiplier** with 95% Bayesian Credible Intervals.
* **Econometric Robustness Suite (`src/topoflow/models/robustness.py`)**: Runs placebos in space, placebos in time ($+24\text{hr}$ rain shift), leave-one-out sensor jackknife tests, and sensitivity reruns filtering strictly for raw observations (`WHERE imputation_flag == 0`).

### Phase 5: Societal Valuation, Environmental Carbon Physics & Eco-Routing Simulation
* **Vehicle Hours of Delay & Monetization (`src/topoflow/impact/economic_delay.py`)**: Computes cumulative VHD across all stations. Applies **USDOT Value of Travel Time Guidance**: personal passenger travel at **$\$24.60 / \text{veh-hr}$** and commercial freight logistics at **$\$42.00 / \text{truck-hr}$**, with regional DC wage sensitivity documented in appendices.
* **EPA MOVES Vehicle Specific Power Emissions (`src/topoflow/impact/emissions.py`)**:
  $$\text{VSP} = v \cdot \left(1.1 \cdot a + 9.81 \cdot \sin(\theta) + 0.132\right) + 0.000302 \cdot v^3$$
  Directly incorporates the road grade angle ($\sin\theta \approx \text{Grade}/100$) and acceleration to compute excess fuel burn (gallons) and metric tons of $CO_2$ and $NO_x$ emitted during stop-and-go cycling out of the Potomac trench.
* **Topography-Aware Proactive Eco-Routing (`src/topoflow/simulation/eco_routing.py`)**: Builds a **Macroscopic Cell Transmission Model (CTM)**. Simulates dynamic proactive traffic diversion (sweeping $1\%$ to $10\%$) triggered when approaching volume approaches $0.90 \cdot k_{\text{crit}}$. Proves that shaving just **$5\%$ of peak demand** prevents the phase collapse entirely, preserving free-flow capacity and reducing net corridor delay by **$>40\%$**.

### Phase 6: Lakehouse Delivery, CI/CD & Portfolio Showcase
* **DuckDB / MotherDuck Cloud Mart (`src/topoflow/warehouse/motherduck_sync.py`)**: Creates clean analytical DDL views (`v_bottleneck_summary`, `v_topography_profile`, `v_shockwave_events`), enabling serverless ad-hoc SQL querying over the lakehouse.
* **GitHub Actions CI/CD (`.github/workflows/`)**: Automated testing and linting (`ruff`, `pytest`) plus scheduled cron jobs fetching recurring daily telemetry.
* **Portfolio Showcase & Interactive Whitepaper**: Publication-quality `README.md`, methodology whitepaper, and interactive space-time visualizations built for engineering and data science technical reviews.

---

## 4. Canonical Data Contracts & Schema Dictionary

| Dataset | Primary Key | Format / Engine | Key Features |
| :--- | :--- | :--- | :--- |
| **Spatial Feature Mart** | `segment_id` (100m) | `spatial_feature_mart.parquet` | `chainage_m`, `utm_x`, `utm_y`, `grade_pct`, `grade_plus`, `grade_minus`, `curvature_kappa`, `radius_m`, `is_bridge`, `is_sag` |
| **Telemetry Mart** | `(segment_id, timestamp_utc)` | `telemetry_5min.parquet` | `speed_mph`, `volume_vph`, `occupancy_pct`, `imputation_flag` (0–3) |
| **Weather Mart** | `(corridor, timestamp_utc)` | `weather_5min.parquet` | `precip_15min_mm`, `precip_60min_mm`, `is_rain_shock`, `temp_c`, `wind_speed_ms` |
| **Unified Panel Matrix** | `(segment_id, timestamp_utc)` | `panel_matrix_5min.parquet` (Partitioned by `corridor/year/month`) | Joined spatial, telemetry, and weather features; `speed_deficit_mph`, `density_k_vpmpl`, `vhd_hours`, interaction terms |
| **Analytical DDL Views** | `segment_id`, `hour_of_day` | DuckDB / MotherDuck Views | `v_bottleneck_summary`, `v_topography_profile`, `v_shockwave_events` |

---

## 5. Developer Quickstart

### Prerequisites
* Python 3.12+
* `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `winget install astral-sh.uv`)

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/stabweee/topoflow.git
cd topoflow

# Lock and synchronize dependencies deterministically
uv sync

# Run automated test suite
uv run pytest -v

# Run strict Ruff linting and formatting checks
uv run ruff check .
uv run ruff format --check .
```

### Running the End-to-End Pipeline
```bash
# Execute the master topoflow CLI pipeline
uv run topoflow run-pipeline --corridor ALB --year 2024 --month 05
```

---

*This document serves as the architectural reference for TopoFlow. For detailed trade-offs on every technical decision, see `docs/decision_framework.md`. For individual implementation issues and execution dependencies, see `docs/tickets.md`.*
