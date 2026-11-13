# TopoFlow Algorithmic & Methodological Manual

**Document Purpose**: High-level architectural specification of the mathematical, spatial, and econometric methods powering TopoFlow.

---

## 1. Vector Road Geometry Ingestion & Topology

* **Implementation**: [`src/topoflow/geo/osm_extractor.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/osm_extractor.py)
* **Data Source**: OpenStreetMap (OSM) via Overpass QL API.

### Method Overview
1. **Overpass QL Bounding Extraction**: Queries highway geometries within corridor bounding boxes defined in `configs/corridors.yaml`, filtering for `highway~"motorway|motorway_link"`.
2. **Directional Separation**: Distinguishes dual carriageways using OSM tags (`oneway=yes`, `name`, `ref`) to separate directional flows (Inner Loop / Northbound vs Outer Loop / Southbound).
3. **Topological Line Sorting & Stitching**: High-density highway links are assembled into continuous directional `LineString` centerlines through endpoint proximity matching and topological order sorting.
4. **Offline Fixture Fallback**: When live Overpass endpoints are unreachable, extractor automatically falls back to authenticated static fixtures in `tests/fixtures/osm_corridors_sample.geojson`.

---

## 2. 100-Meter Equidistant Spatial Discretization

* **Implementation**: [`src/topoflow/geo/discretization.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/discretization.py)
* **Projection**: NAD83 / UTM Zone 18N (`EPSG:26918`).

### Method Overview
1. **Metric Projection**: Transforms geographic coordinates (`EPSG:4326`) into Euclidean meters (`EPSG:26918`), where scale distortion across Virginia and Maryland is $< 0.04\%$.
2. **Arc-Length Resampling**: Computes total 2D line length $L$ and resamples equidistant stations at exactly $\Delta x = 100.0\text{ m}$ intervals using Shapely linear interpolation:
   $$P(x_i) = \text{interpolate}(i \times 100.0), \quad i \in \left\{0, 1, \dots, \lfloor L / 100 \rfloor\right\}$$
3. **Dual-Layer Anti-Loopback Defense**: Resolves reversible median managed facilities (such as the I-95 Express Lanes loopback) by:
   - Enforcing strictly monotonic directional bearing constraints.
   - Rejecting reverse-bearing segment vectors whose interior angle exceeds $90^\circ$ relative to the corridor's macro azimuth.
4. **Canonical Primary Key Generation**: Every discretized station receives a composite unique primary key:
   $$\text{segment\_id} = \text{corridor\_id} + \text{"\_"} + \text{direction} + \text{"\_"} + \text{station\_index}$$

---

## 3. Three-Point Menger Osculating Circle Curvature

* **Implementation**: [`src/topoflow/geo/discretization.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/discretization.py)

### Mathematical Formulation
For three consecutive 100m spatial stations $P_{i-1}, P_i, P_{i+1}$, the circumscribed osculating circle defines the horizontal road curvature:
$$\kappa_i = \frac{4 \cdot \text{Area}(P_{i-1}, P_i, P_{i+1})}{\|P_{i-1} - P_i\| \cdot \|P_i - P_{i+1}\| \cdot \|P_{i+1} - P_{i-1}\|}$$
where the triangle area is evaluated via the 2D cross product:
$$\text{Area} = \frac{1}{2} \left| (x_i - x_{i-1})(y_{i+1} - y_{i-1}) - (y_i - y_{i-1})(x_{i+1} - x_{i-1}) \right|$$

### Numerical Safeguards
1. **Collinear Zero-Guard**: When three stations are collinear ($\text{Area} \to 0$), numerical division by zero is prevented with a float guard ($\epsilon = 10^{-9}$), returning $\kappa = 0.0\text{ m}^{-1}$.
2. **Tangent Inversion Capping**: The horizontal radius $R_i = 1 / \kappa_i$ is capped at $R_{\max} = 10,000\text{ m}$ on straight tangents.
3. **Endpoint Boundary Handling**: Boundary stations ($i=0$ and $i=N$) replicate their adjacent interior curvature to maintain structural vector lengths.

---

## 4. Hybrid Bare-Earth & LiDAR Bridge Deck Elevation

* **Implementation**: [`src/topoflow/ingestion/elevation_usgs.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/ingestion/elevation_usgs.py), [`src/topoflow/ingestion/lidar_usgs.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/ingestion/lidar_usgs.py), [`src/topoflow/geo/elevation.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/elevation.py)
* **Data Sources**: USGS 3DEP Elevation Point Query Service (EPQS) & USGS 3DEP EPT LiDAR on AWS Open Data (`USGS_LPC_VA_Fairfax_County_2018`).

### The Dual-Source Problem
Standard USGS bare-earth DEM rasters are "hydro-enforced": over water bodies, elevations represent the river bottom or water surface, not the roadway bridge deck. For the American Legion Bridge, bare-earth DEM plummets $\approx 30\text{m}$ into the Potomac River gorge, creating an artificial $+15\%$ grade cliff.

### Hybrid Resolution Protocol
1. **Bare-Earth Roadway Sampling**: Queries the USGS 1-meter 3DEP EPQS service for all 406 station coordinates across Maryland and Virginia.
2. **LiDAR Point Cloud Traversal**: Traverses the Entwine Point Tile (EPT) octree over Fairfax County, streaming and filtering $90,828$ authentic laser returns.
3. **ASPRS Class 17 Filtering**: Extracts ASPRS Class 17 (Bridge Deck) returns within a 35m spatial search buffer of the bridge centerline.
4. **Pavement Elevation Isolation**: Fits a high-confidence asphalt deck elevation ($Z \approx 35.75\text{m} - 46.14\text{m}$ NAVD88) across Stations 37–41 (450m Potomac span).
5. **As-Built Blueprint Verification**: Compares extracted deck elevations against official VDOT/MDOT SHA As-Built reference profiles (`configs/bridge_profile.csv`), passing with maximum error $\le 0.0050\text{m}$ (well within the $0.20\text{m}$ engineering tolerance).

---

## 5. Roadway Continuity & Water-Crossing Restoration

* **Implementation**: [`src/topoflow/geo/elevation.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/elevation.py)

### Method Overview
1. **River Trench Detection**: Detects bare-earth drops over unbridged water-crossings where DEM heights drop into sea-level tidal waters ($Z < 2\text{m}$) accompanied by unphysical grade spikes ($> \pm 7.5\%$).
2. **Abutment-to-Abutment Continuity**: Replaces hydro-enforced trench dips with synthetic linear roadway continuity between physical bridge abutments:
   - *Occoquan River Crossing (I-95)*: Restored Stations 21–22 (NB) and 55–56 (SB), eliminating artificial $\pm 7.8\%$ spikes and restoring interstate Piedmont grades ($\le \pm 2.5\%$).
   - *Underpass Cut Smoothing (I-295)*: Smoothed intermediate underpass cuts (Oxon Run, Suitland Pkwy, Howard Rd).
   - *Maryland Approach Continuity (I-495)*: Eliminated underpass trench dips between Chainage 4,100m and 4,650m.

---

## 6. Savitzky-Golay Polynomial Differentiation

* **Implementation**: [`src/topoflow/geo/elevation.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/elevation.py)

### Mathematical Formulation
To eliminate high-frequency raster jitter while preserving sharp vertical curve inflection points, elevation profiles are smoothed using a quadratic Savitzky-Golay convolution filter with window length $w = 5$ stations ($500\text{m}$) and polynomial degree $p = 2$:
$$z_{\text{smooth}}(x) = c_0 + c_1 (x - x_0) + c_2 (x - x_0)^2$$

### Analytical Derivative Evaluation
Rather than applying noisy discrete differences to smoothed data, the 1st and 2nd derivatives are evaluated **analytically from the polynomial coefficients**:
$$\text{Grade}_i = 100 \times \left.\frac{dz}{dx}\right\|_i = 100 \times c_1$$
$$\left.\frac{d^2z}{dx^2}\right\|_i = 2 \cdot c_2$$

### Empirical Superiority
Comparing Savitzky-Golay analytical differentiation against raw two-point finite differences:
* **Total Variation (TV)**: Reduced by $30.6\%$ (from $37.53\%$ raw down to $26.04\%$).
* **Jerk Energy**: Attenuated by $6.75\times$ (from $8.64 \times 10^{-9}$ down to $1.28 \times 10^{-9}\text{ m}^{-2}$).
* **Estimation RMSE**: Error against synthetic ground-truth profiles decreased by $1.95\times$ ($0.275\%$ raw down to $0.141\%$).

---

## 7. Geospatial Feature Mart Integration & DuckDB Registration

* **Implementation**: [`src/topoflow/geo/feature_mart.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/feature_mart.py)

### Method Overview
1. **Vectorized Columnar Merge**: Combines 100m discretization stations and smoothed elevation profiles in Polars via `segment_id`.
2. **Domain Feature Synthesis**: Computes asymmetric grades ($\text{Grade}^+, \text{Grade}^-$), AASHTO grade regimes, the Topographical Friction Index ($\text{TFI}$), and emissions slope angles ($\sin\theta$).
3. **Compressed Parquet Export**: Writes canonical mart to `data/processed/spatial_feature_mart.parquet` using Zstandard compression partitioned by `corridor_id`.
4. **Zero-Copy DuckDB View**: `register_spatial_feature_mart` registers an in-memory zero-copy DuckDB table view (`spatial_feature_mart`) enabling instant SQL analytics and downstream panel joins.
