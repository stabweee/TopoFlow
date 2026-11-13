# TopoFlow Visualizations Catalog

**Repository Location**: `outputs/figures/` (organized by project phase)  
**Document Purpose**: Canonical visual reference catalog documenting every figure, map, and diagnostic chart generated across the TopoFlow project lifecycle, explaining their technical intent, visual composition, and empirical takeaways.

---

## Directory Organization

```
outputs/figures/
├── phase1/                               # Phase 1: Geospatial, Curvature & Elevation
│   ├── corridors_map.html                # Interactive Folium map of corridor footprints
│   ├── p1_corridor_centerlines.png       # Plan-view centerline alignment across corridors
│   ├── p1_horizontal_curvature.png       # Menger curvature and radius-of-curvature profiles
│   ├── p1_elevation_grade_profile.png    # Longitudinal elevation and vertical % grade profiles
│   ├── p1_savgol_vs_raw_differences.png  # Savitzky-Golay filtering performance & jerk diagnostics
│   └── p1_spatial_feature_mart.png       # Integrated 4-panel spatial dimension mart dashboard
├── phase2/                               # (Upcoming) Telemetry, missingness & weather heatmaps
├── phase3/                               # (Upcoming) Fundamental diagrams & space-time (x-t) shockwaves
├── phase4/                               # (Upcoming) Econometric tournament forest plots & SDID event studies
└── phase5/                               # (Upcoming) CTM simulation space-time trajectories & delay curves
```

---

## Phase 1 Visual Artifacts (Geospatial & Topographical Foundations)

### 1. Interactive Multi-Corridor Exploration Map
* **Artifact File**: [`outputs/figures/phase1/corridors_map.html`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/outputs/figures/phase1/corridors_map.html)
* **Generating Module**: [`src/topoflow/visualization/corridor_map.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/visualization/corridor_map.py)
* **Technical Intent**: Provide an interactive web-based map (Leaflet / Folium) for inspecting bounding boxes, directional carriageways, and bridge structures across the study corridor and flat control corridors.
* **Key Features**:
  - Full coordinate overlay in WGS84 (`EPSG:4326`) with layer controls for each corridor.
  - Directional vector carriageways (Outer Loop / Inner Loop, Northbound / Southbound).
  - Explicit bridge span highlighting over the Potomac River (American Legion Bridge) and Occoquan River (I-95).
* **Empirical Takeaway**: Confirms complete geographic coverage without topological fragmentation or boundary clipping across Virginia and Maryland borders.

---

### 2. Multi-Corridor Plan-View Centerline Layout
* **Artifact File**: [`outputs/figures/phase1/p1_corridor_centerlines.png`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/outputs/figures/phase1/p1_corridor_centerlines.png)
* **Generating Module**: [`src/topoflow/visualization/corridor_map.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/visualization/corridor_map.py)
* **Technical Intent**: Publication-grade comparative plan-view visualization of the roadway geometries projected into metric UTM Zone 18N (`EPSG:26918`).
* **Visual Composition**:
  - **Panel 1 (I-495 American Legion Bridge)**: Displays the primary study corridor spanning McLean, VA to Cabin John, MD, highlighting the Potomac River bridge span.
  - **Panel 2 (I-95 Lorton-Woodbridge)**: Flat control corridor along the Occoquan River basin.
  - **Panel 3 (I-295 Anacostia Freeway)**: Flat control corridor along the urban Anacostia River waterfront.
* **Empirical Takeaway**: Validates metric spatial projection and confirms directional carriageway separation across all three corridors.

---

### 3. Horizontal Curvature Profiling & Plan-View Scatter
* **Artifact File**: [`outputs/figures/phase1/p1_horizontal_curvature.png`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/outputs/figures/phase1/p1_horizontal_curvature.png)
* **Generating Module**: [`src/topoflow/visualization/curvature_plot.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/visualization/curvature_plot.py)
* **Technical Intent**: Diagnose horizontal geometric resistance and identify tight horizontal curves using three-point Menger osculating circle curvature ($\kappa, \text{m}^{-1}$) and radius of curvature ($R = 1/\kappa$).
* **Visual Composition**:
  - **Top-Left (Plan-View Curvature Map)**: ALMB corridor stations color-coded by curvature ($\text{km}^{-1}$), highlighting geometric stress zones.
  - **Top-Right (Curvature Profile vs Chainage)**: Longitudinal plot of $\kappa$ along the centerline, displaying acute peaks at Scott's Run Bluffs and the Clara Barton interchange.
  - **Bottom-Left (Radius of Curvature $R$)**: Inverted radius profile with an AASHTO minimum horizontal curve reference threshold line ($R_{\min} = 450\text{m}$).
  - **Bottom-Right (Cross-Corridor Curvature Boxplots)**: Comparative distributions demonstrating that I-495 ALMB has substantially higher curve volatility than the I-95 and I-295 control corridors.
* **Empirical Takeaway**: Confirms that horizontal curves on the Virginia and Maryland approaches compound the vertical sag inflection, while control corridors remain geometric tangents ($R > 1,500\text{m}$).

---

### 4. Hybrid Longitudinal Elevation & Grade Profile
* **Artifact File**: [`outputs/figures/phase1/p1_elevation_grade_profile.png`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/outputs/figures/phase1/p1_elevation_grade_profile.png)
* **Generating Module**: [`src/topoflow/visualization/elevation_plot.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/visualization/elevation_plot.py)
* **Technical Intent**: Present the definitive vertical elevation profile ($Z$) and analytical slope ($\text{Grade} = 100 \cdot dz/dx$) for the American Legion Bridge sag curve, juxtaposed against control corridors.
* **Visual Composition**:
  - **Panel 1 (ALMB Absolute Elevation Profile)**: Bare-Earth DEM vs. Savitzky-Golay smoothed elevation with ASPRS Class 17 LiDAR bridge deck laser returns overlaid across the 450m Potomac gorge span (~35.75m–46.14m NAVD88).
  - **Panel 2 (ALMB Longitudinal Vertical Grade %)**: First derivative grade profile with staggered geographic landmark annotations:
    - *Tysons Corner Plateau* (~120m NAVD88)
    - *McLean Crest Summit* (~105m NAVD88)
    - *Exit 44 Georgetown Pike*
    - *Scott's Run Bluffs descent* ($-3.2\%$ grade)
    - *Potomac River Gorge Sag Inflection* (~35m NAVD88)
    - *Clara Barton Parkway climb* ($+3.8\%$ grade)
    - *River Road (MD-190) Summit* (~95m NAVD88)
  - **Panels 3 & 4 (Control Corridor Profiles - I-95 & I-295)**: Elevation and grade profiles of control corridors confirming authentic rolling Piedmont topography ($\le \pm 2.5\%$) with unclipped dynamic scaling, free from bare-earth river trench artifacts.
* **Empirical Takeaway**: Conclusively demonstrates the $-3.2\%$ downhill gravity descent into the Potomac gorge followed immediately by the $+3.8\%$ uphill climb, establishing the physical bedrock for sag bottleneck deceleration.

---

### 5. Savitzky-Golay Filter Diagnostics & Total Variation
* **Artifact File**: [`outputs/figures/phase1/p1_savgol_vs_raw_differences.png`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/outputs/figures/phase1/p1_savgol_vs_raw_differences.png)
* **Generating Module**: [`src/topoflow/visualization/elevation_plot.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/visualization/elevation_plot.py)
* **Technical Intent**: Empirically prove that the quadratic Savitzky-Golay convolution filter ($w=5, p=2$) eliminates discrete raw sampling jitter without blunting true vertical inflection points.
* **Visual Composition**:
  - **Top-Left (Raw vs Smoothed Grade Overlay)**: Direct comparison showing raw two-point finite difference noise spikes vs the smooth analytical derivative.
  - **Top-Right (Residual Distribution)**: Histogram and KDE of elevation residuals $(\Delta z = z_{\text{raw}} - z_{\text{savgol}})$ showing zero-centered Gaussian distribution ($\mu \approx 0.00\text{m}, \sigma < 0.15\text{m}$).
  - **Bottom-Left (Jerk Energy & Roughness Reduction)**: Second derivative $(d^2z/dx^2)$ plot demonstrating a $6.75\times$ attenuation in unphysical vertical jerk energy (from $8.64 \times 10^{-9}$ down to $1.28 \times 10^{-9}\text{ m}^{-2}$).
  - **Bottom-Right (Cumulative Total Variation Metric)**: Total Variation curve proving a $30.6\%$ reduction in cumulative gradient oscillation (from $37.53\%$ raw down to $26.04\%$ filtered).
* **Empirical Takeaway**: Justifies Decision 1.5, proving the Savitzky-Golay filter preserves macro physical highway curvature while suppressing micro-scale LiDAR and DEM raster discretization noise.

---

### 6. Geospatial Feature Mart Health Dashboard
* **Artifact File**: [`outputs/figures/phase1/p1_spatial_feature_mart.png`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/outputs/figures/phase1/p1_spatial_feature_mart.png)
* **Generating Module**: [`src/topoflow/visualization/feature_mart_plot.py`](file:///c:/Users/hsu_s/Projects/Personal/TopoFlow/src/topoflow/visualization/feature_mart_plot.py)
* **Technical Intent**: Full audit and verification of the consolidated 33-column spatial feature mart (`data/processed/spatial_feature_mart.parquet`) across all 406 stations prior to Phase 2 sensor integration.
* **Visual Composition**:
  - **Panel 1 (Topographical Friction Index Distribution)**: Empirical KDE and boxplot distributions demonstrating that the American Legion Bridge corridor exhibits ~40% higher composite topographical resistance ($\mu_{\text{TFI}} = 0.591$) compared to control corridors (I-95 $\mu = 0.414$, I-295 $\mu = 0.438$).
  - **Panel 2 (Correlation Matrix Heatmap)**: Pearson correlation heatmap validating physical collinearity structures (e.g., $|\text{Grade}|$ with $\text{TFI}$ $r \approx 0.92$, elevation $z$ with chainage $x$).
  - **Panel 3 (AASHTO Geometric Grade Regimes)**: Stacked bar chart showing the breakdown across 5 AASHTO regimes (`steep_downgrade`, `moderate_downgrade`, `flat_tangent`, `moderate_upgrade`, `steep_upgrade`). Shows ALMB has $38\%$ of its length in steep regimes ($\ge \pm 2.0\%$), whereas control corridors are dominated by flat tangents.
  - **Panel 4 (Bivariate Geometric Phase Space)**: Scatter plot of $|\text{Grade}|$ vs. Curvature $\kappa$ with iso-TFI friction contours, illustrating that ALMB occupies the extreme upper-right quadrant of geometric friction.
* **Empirical Takeaway**: Confirms that the feature mart is mathematically clean (0 nulls, 0 NaNs), physically sound, and ready for ingestion into downstream panel econometric regressions and traffic shockwave models.
