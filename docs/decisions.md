# TopoFlow Architectural Decision Record (ADR)

**Document Purpose**: High-level executive log of all key architectural, engineering, statistical, and econometric decisions across the TopoFlow project lifecycle.  
**Full Justification Catalog**: [`docs/private/decision_framework.md`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/docs/private/decision_framework.md)

---

## 1. Decision Governance Framework

To balance engineering velocity with scientific rigor, every architectural decision in TopoFlow is governed by a three-tier execution hierarchy:

```
+---------------------------------------------------------------------------------------------------+
| TIER 1: Foundational Bedrock ("One-Way Doors")     --> MUST DECIDE UPFRONT                        |
| Irreversible structural choices. Changing them later breaks primary keys, invalidates schemas,    |
| or requires rebuilding data lakehouse partitions.                                                 |
+---------------------------------------------------------------------------------------------------+
                                              |
                                              v
+---------------------------------------------------------------------------------------------------+
| TIER 2: Pipeline Defaults ("Two-Way Doors")        --> LOCK IN A MODULAR DEFAULT NOW              |
| Implementation choices requiring a sensible default to unblock development, but engineered with   |
| swappable functions, configuration parameters, or audit flags for flexibility.                    |
+---------------------------------------------------------------------------------------------------+
                                              |
                                              v
+---------------------------------------------------------------------------------------------------+
| TIER 3: Empirical Discoveries & Sensitivities      --> EXPLORE & VERIFY ALONG THE WAY             |
| Scientific hypotheses and parameter estimations that cannot be decided a priori; they must be     |
| estimated from empirical data and validated via the planned robustness testing suite.             |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Phase 1: Geospatial & Topographical Foundations

### Choice 1.1: Coordinate Reference System (CRS) & Projection Strategy
* **Decision Tier**: **Tier 1 (One-Way Door - Locked Upfront)**
* **Context**: Project spans Virginia (Fairfax County) and Maryland (Montgomery County). Need Euclidean metric units ($\text{meters}$) where linear distortion is minimal to compute true physical grades ($\Delta z / \Delta x$).
* **Decision**: **NAD83 / UTM Zone 18N (`EPSG:26918`)**.
* **Rationale**: Covers the entire DC/VA/MD metropolitan area in a single metric coordinate frame without state boundary distortion. Linear distortion along the central meridian is $< 0.04\%$ (millimeter accuracy). Unprojected `EPSG:4326` is reserved exclusively for raw data ingestion and GeoJSON visualization.
* **Rejected**: Web Mercator `EPSG:3857` (inflates distances by $\approx 28\%$ at latitude $39^\circ\text{N}$, destroying vertical slope calculations); Maryland/Virginia State Plane (splits across state border).

### Choice 1.2: Spatial Discretization Interval (Chainage Grid)
* **Decision Tier**: **Tier 1 (One-Way Door - Locked Upfront)**
* **Context**: Linear linestrings must be partitioned into discrete segments to join elevation, curvature, and traffic telemetry.
* **Decision**: **Strict 100-Meter Equidistant Discretization ($\Delta x = 100\text{m}$)**.
* **Rationale**: 100 meters represents the standard AASHTO highway design stationing length. At $60\text{ mph}$, a vehicle traverses $100\text{m}$ in $\approx 3.6\text{ seconds}$, closely aligning with the physical propagation of kinematic compression waves. Avoids over-aggregating the localized $300\text{m}$ Potomac gorge trough while avoiding micro-scale sensor noise.
* **Rejected**: 500m coarse segments (would blend downhill and uphill slopes, masking the sag); 50m micro segments (would cause artificial spatial autocorrelation and over-interpolate detector sensors).

### Choice 1.3: Vector Road Geometry Source
* **Decision Tier**: **Tier 1 (One-Way Door - Locked Upfront)**
* **Context**: Need accurate directional carriageways capturing real S-curves and bridge spans.
* **Decision**: **OpenStreetMap Overpass QL with Authenticated Offline GeoJSON Fixtures**.
* **Rationale**: Provides high-resolution vector geometry capturing the Scott's Run S-curves and Clara Barton interchange with explicit bridge span tags (`bridge=yes`). Offline fixtures in `tests/fixtures/` ensure 100% deterministic, hermetic testing without live network dependence.

### Choice 1.4: Elevation Ingestion & Bridge Deck Strategy
* **Decision Tier**: **Tier 1 (One-Way Door - Locked Upfront)**
* **Context**: Standard USGS bare-earth DEM rasters are hydro-enforced, plummeting $30\text{m}$ into the Potomac River basin.
* **Decision**: **Dual-Source Hybrid Elevation: USGS 3DEP 1m EPQS Bare-Earth + USGS 3DEP LiDAR ASPRS Class 17 Bridge Deck Returns**.
* **Rationale**: Roadway approach slopes use authentic USGS bare-earth 1m queries, while the 450m bridge span uses 1,583 Class 17 laser returns (~35.75m–46.14m NAVD88) validated against official VDOT/MDOT SHA As-Built drawings ($\le 0.005\text{m}$ error).
* **Rejected**: Satellite SRTM (30m resolution too coarse); raw bare-earth alone (creates artificial $+15\%$ river cliff).

### Choice 1.5: Elevation Smoothing & Derivative Formulation
* **Decision Tier**: **Tier 2 (Two-Way Door - Modular Default)**
* **Context**: Raw discrete elevation queries contain micro-scale discretization jitter that creates artificial gradient spikes when differenced.
* **Decision**: **Quadratic Savitzky-Golay Polynomial Filter ($w=5$ stations / 500m, $p=2$) with Analytical Derivative Evaluation**.
* **Rationale**: Computes continuous, smooth analytical 1st and 2nd derivatives directly from polynomial coefficients. Reduces total variation by $30.6\%$, suppresses unphysical jerk energy by $6.75\times$, and cuts RMSE by $1.95\times$ without blunting the true sag inflection point.
* **Rejected**: Raw two-point finite differences (extreme noise amplification); simple moving average (flattens true sag peaks).

### Choice 1.6: Spatial Feature Mart Schema & Storage Grain
* **Decision Tier**: **Tier 1 (One-Way Door - Locked Upfront)**
* **Context**: Downstream econometric regressions, CTM simulations, and dashboards need a unified dimensional geometry table.
* **Decision**: **33-column Zstandard-compressed Apache Parquet partitioned by `corridor_id` with Zero-Copy DuckDB View Registration**.
* **Rationale**: Canonical table of 406 stations with zero nulls/NaNs. DuckDB registration (`register_spatial_feature_mart`) enables instant zero-copy in-process SQL querying and analytical joins with 5-minute telemetry tables.

---

## 3. Phase 2: Telemetry & Weather Lakehouse (Pipeline Defaults)

### Choice 2.1: Telemetry Temporal Aggregation Grain
* **Decision Tier**: **Tier 2 (Two-Way Door - Modular Default)**
* **Decision**: **5-Minute Harmonized Windows**.
* **Rationale**: Standard operational grain for VDOT / MDOT SHA sensor loops and INRIX probe feeds. Smooths high-frequency vehicle headway variance while resolving backward-propagating shockwaves (which propagate at $\approx -11.2\text{ mph}$, traveling 1.5 km in 5 minutes).

### Choice 2.2: Missing Telemetry & Sensor Outage Imputation
* **Decision Tier**: **Tier 2 (Two-Way Door - Modular Default)**
* **Decision**: **Linear Spatio-Temporal Interpolation with Explicit `imputation_flag` Boolean Column**.
* **Rationale**: Preserves econometric auditability. Prevents unobserved detector outages from corrupting fixed-effect regressions by allowing researchers to filter `WHERE imputation_flag = FALSE` in robustness checks.

### Choice 2.3: Meteorological Data Source & Rain Shock Feature
* **Decision Tier**: **Tier 2 (Two-Way Door - Modular Default)**
* **Decision**: **Open-Meteo Historical API (NOAA HRRR 3km Mesoscale Model)**.
* **Rationale**: 1-hour native resolution interpolated to 5-minute corridor windows. Generates a discrete `is_rain_shock` binary flag ($\ge 2.5\text{ mm/hr}$) and rolling 15m/60m precipitation intensity features.

---

## 4. Phase 3 & 4: Traffic Physics & Causal Tournaments

### Choice 3.1: Traffic Phase Breakdown Definition
* **Decision Tier**: **Tier 3 (Empirical Discovery - Sensitivity Tested)**
* **Baseline Choice**: **Speed $< 45\text{ mph}$ for $\ge 15$ continuous minutes** accompanied by a sustained drop in discharge flow.
* **Sensitivity Suite**: Evaluated across alternative speed thresholds ($40\text{ mph}, 45\text{ mph}, 50\text{ mph}$) and persistence windows ($10\text{m}, 15\text{m}, 20\text{m}$) to prove causal robustness.

### Choice 4.1: Econometric Identification Strategy
* **Decision Tier**: **Tier 1 (One-Way Door - Locked Upfront)**
* **Decision**: **Synthetic Difference-in-Differences (SDID) & Two-Way Fixed Effects (TWFE) with Spatial Panel GMM**.
* **Rationale**: Controls for unobserved time-invariant spatial confounders (lane widths, sight distance) via segment fixed effects ($\alpha_i$) and macroeconomic time shocks via time fixed effects ($\gamma_t$).
* **Causal Model**:
  $$Y_{i,t} = \beta_1^+ (\text{Grade}^+_i \times \text{Vol}_{i,t}) + \beta_1^- (\text{Grade}^-_i \times \text{Vol}_{i,t}) + \beta_2 (\kappa_i \times \text{Vol}_{i,t}) + \beta_3 (\text{Rain}_{i,t} \times \text{Grade}_i) + \alpha_i + \gamma_t + \epsilon_{i,t}$$

### Choice 4.2: Control Corridor Selection (Donor Pool)
* **Decision Tier**: **Tier 1 (One-Way Door - Locked Upfront)**
* **Decision**: **I-95 Lorton-Woodbridge & I-295 Anacostia Freeway Flat Control Donors**.
* **Rationale**: Selected because they share the same DC-area macroeconomic commuter patterns, weather events, and federal holiday schedules, but have flat/mild topography ($\le \pm 2.0\%$). Allows synthetic matching to isolate the pure treatment effect of the Potomac sag curve.

### Choice 4.3: Standard Error Corrections
* **Decision Tier**: **Tier 1 (One-Way Door - Locked Upfront)**
* **Decision**: **Driscoll-Kraay Non-Parametric Standard Errors**.
* **Rationale**: Standard OLS standard errors collapse in spatial panel data due to spatial cross-sectional dependence and temporal autocorrelation. Driscoll-Kraay guarantees unbiased hypothesis testing and valid confidence intervals.

---

## 5. Phase 5 & 6: Simulation & Analytical Delivery

### Choice 5.1: Simulation Platform
* **Decision Tier**: **Tier 2 (Two-Way Door - Modular Default)**
* **Decision**: **Macroscopic Cell Transmission Model (CTM)**.
* **Rationale**: Solves first-order hydrodynamic Lighthill-Whitham-Richards (LWR) conservation equations across 100m cells. Capable of evaluating system-level proactive eco-routing diversions ($1\% - 10\%$) in seconds, unlike microscopic simulators (SUMO, VISSIM) which require hours of compute.

### Choice 6.1: Serverless Cloud Delivery
* **Decision Tier**: **Tier 2 (Two-Way Door - Modular Default)**
* **Decision**: **MotherDuck Serverless Cloud OLAP**.
* **Rationale**: Zero-infrastructure, high-performance analytical SQL queries directly against compressed Parquet marts, powering interactive portfolio dashboards and automated recurring CI/CD pipelines.
