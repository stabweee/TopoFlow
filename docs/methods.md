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
4. **Pavement Elevation Isolation**: Takes the median Class 17 elevation within 35 m of each station, giving measured ASPRS Class 17 deck returns on ALB inner-loop stations 37–40 (44.2 m → 35.9 m NAVD88; OSM bridge way ≈ chainage 3,661–4,088 m, ≈427 m).
5. **Independent Validation (pending)**: `validate_bridge_as_built` compares deck elevations against an independent reference profile (`configs/bridge_asbuilt_reference.csv`, every row must cite a `source`, tolerance 0.20 m). Deck elevations are currently not yet validated against an independent reference (no cited as-built profile is in the repo; the former `configs/bridge_profile.csv` was derived from this pipeline's own output and was removed as circular).

---

## 5. Bridge Stations Without LiDAR & Elevation Provenance

* **Implementation**: [`src/topoflow/geo/elevation.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/elevation.py), [`src/topoflow/geo/osm_extractor.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/osm_extractor.py)

### Method Overview
1. **Per-Station Bridge Tags**: Each OSM way is located along its stitched carriageway as a chainage interval (EPSG:26918). A station is a bridge station only if it lies on a `bridge`-tagged way.
2. **Measured Sources Only**: Every station takes USGS 3DEP EPQS bare-earth elevation, overridden by LiDAR Class 17 deck returns where they exist. A station with no measurement raises an error; nothing is invented.
3. **Flagged Interpolation**: Bare-earth DEMs sample the ground beneath bridge decks (Occoquan River on I-95; MacArthur Blvd / Clara Barton Pkwy overpasses on the ALB Maryland approach; Oxon Run, Suitland Pkwy, Howard Rd on I-295). Bridge stations without LiDAR are linearly interpolated between the nearest measured stations and tagged `elevation_source = "interpolated"` (ALB 5, I-95 9, I-295 9 stations). All other stations keep their measured values unchanged.
4. **Edge Flag**: The first and last two stations of each carriageway are tagged `grade_edge_effect` (truncated Savitzky-Golay window); exclude them from grade-sensitive analysis.

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
* **Total Variation (TV)**: Reduced by $29.4\%$ (from $36.31\%$ raw down to $25.62\%$) on the measured ALB inner-loop profile.
* **Jerk Energy**: Attenuated by $7.1\times$ (from $7.74 \times 10^{-9}$ down to $1.09 \times 10^{-9}\text{ m}^{-2}$).
* **Synthetic illustration only**: on a known synthetic parabola with σ = 5 cm noise, grade RMSE falls $1.95\times$ ($0.275\%$ → $0.141\%$). This illustrates the filter; it is **not** a measurement of accuracy on real data.

---

## 7. Geospatial Feature Mart Integration & DuckDB Registration

* **Implementation**: [`src/topoflow/geo/feature_mart.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/geo/feature_mart.py)

### Method Overview
1. **Vectorized Columnar Merge**: Combines 100m discretization stations and smoothed elevation profiles in Polars via `segment_id`.
2. **Domain Feature Synthesis**: Computes asymmetric grades ($\text{Grade}^+, \text{Grade}^-$), AASHTO grade regimes, the Topographical Friction Index ($\text{TFI}$), and emissions slope angles ($\sin\theta$).
3. **Compressed Parquet Export**: Writes canonical mart to `data/processed/spatial_feature_mart.parquet` using Zstandard compression partitioned by `corridor_id`.
4. **Zero-Copy DuckDB View**: `register_spatial_feature_mart` registers an in-memory zero-copy DuckDB table view (`spatial_feature_mart`) enabling instant SQL analytics and downstream panel joins.
