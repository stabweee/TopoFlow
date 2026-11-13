# TopoFlow Quantitative Metrics Dictionary

**Document Purpose**: Definitive reference catalog of all mathematical definitions, spatial attributes, engineering indices, and econometric estimators used across the TopoFlow project.

---

## 1. Spatial & Geometric Metrics

| Metric Name | Symbol / Code | Unit | Mathematical Definition | Domain Interpretation & Bounds |
| :--- | :---: | :---: | :--- | :--- |
| **Linear Chainage** | $x_i$ / `chainage_m` | meters ($\text{m}$) | $x_i = i \times 100.0$ along centerline | Arc-length distance from the upstream corridor origin. Equidistant $100\text{m}$ grid. |
| **Station Elevation** | $z_i$ / `elevation_m` | meters ($\text{m}$) NAVD88 | $z_i = z_{\text{bare\_earth}}$ (roadway) or $z_{\text{lidar\_class17}}$ (bridge deck) | Absolute surface height. Validated against official VDOT/MDOT SHA As-Built drawings ($\le 0.005\text{m}$ error on bridge span). |
| **Menger Curvature** | $\kappa_i$ / `curvature_kappa` | $\text{m}^{-1}$ (or $\text{km}^{-1}$) | $\kappa = \frac{4 A}{\|p_1 - p_2\| \|p_2 - p_3\| \|p_3 - p_1\|}$ | Reciprocal of the radius of the osculating circumcircle through stations $(i-1, i, i+1)$. Includes $10^{-9}$ numerical zero-guard. |
| **Radius of Curvature** | $R_i$ / `radius_m` | meters ($\text{m}$) | $R_i = \min\left(\frac{1}{\kappa_i}, 10000.0\right)$ | Radius of horizontal curve. AASHTO interstate guideline $R \ge 450\text{m}$ at $60\text{ mph}$. Capped at $10,000\text{m}$ on tangents. |
| **Projected Coordinates** | $(X_i, Y_i)$ / `geometry` | meters ($\text{m}$) | NAD83 / UTM Zone 18N (`EPSG:26918`) | Metric Cartesian plane coordinates. Distortions $< 0.04\%$ across VA and MD. |

---

## 2. Vertical Slope & Geometric Derivatives

| Metric Name | Symbol / Code | Unit | Mathematical Formulation | Domain Interpretation & Implementation |
| :--- | :---: | :---: | :--- | :--- |
| **Longitudinal Grade** | $\text{Grade}_i$ / `grade_pct` | percent ($\%$) | $\text{Grade}_i = 100 \times \left.\frac{dz}{dx}\right\|_i$ | Tangent slope of roadway. Calculated analytically from the 1st derivative of a quadratic Savitzky-Golay polynomial ($w=5, p=2$). |
| **Raw Two-Point Grade** | $\text{Grade}_{\text{raw}, i}$ | percent ($\%$) | $\text{Grade}_{\text{raw}, i} = 100 \times \frac{z_{i+1} - z_{i-1}}{x_{i+1} - x_{i-1}}$ | Unfiltered finite difference benchmark used for filter residual and total variation audits. |
| **Vertical Acceleration** | $z''_i$ / `d2z_dx2` | $\text{m}^{-1}$ | $z''_i = \left.\frac{d^2z}{dx^2}\right\|_i$ | Rate of change of grade per unit distance. Evaluated analytically via 2nd derivative of Savitzky-Golay polynomial. |
| **Vertical Curve Rate** | $K$ | $\text{m} / \%$ | $K = \frac{L}{\|G_2 - G_1\|}$ | AASHTO sag/crest design parameter. Sag curves require $K \ge 40\text{ m}/\%$ for headlight sight distance at $60\text{ mph}$. |
| **Jerk Roughness Energy** | $\mathcal{E}_{\text{jerk}}$ | $\text{m}^{-2}$ | $\mathcal{E} = \frac{1}{N} \sum_{i=1}^N (z''_i)^2$ | Measure of unphysical vertical oscillation. Savitzky-Golay filtering reduces jerk by $6.75\times$ over raw EPQS queries. |
| **Total Variation (Grade)** | $\text{TV}(\text{Grade})$ | percent ($\%$) | $\text{TV} = \sum_{i=1}^{N-1} \|\text{Grade}_{i+1} - \text{Grade}_i\|$ | Cumulative gradient fluctuation. Filtered profile achieves $30.6\%$ total variation reduction ($26.04\%$ vs. $37.53\%$ raw). |

---

## 3. Domain Behavioral Features & Friction Indices

### Asymmetric Grade Decomposition
Vehicle kinematics and driver behavior respond asymmetrically to uphill vs downhill vertical grades:
* **Positive Climbing Grade ($\text{Grade}^+$)**:
  $$\text{Grade}^+_i = \max(0, \text{Grade}_i)$$
  *Captures gravitational deceleration and mechanical engine load on heavy commercial trucks ($+3.8\%$ on Maryland climb).*
* **Negative Gravitational Grade ($\text{Grade}^-$)**:
  $$\text{Grade}^-_i = \min(0, \text{Grade}_i)$$
  *Captures optical compression, perceived downward acceleration, and driver precautionary tap-braking ($-3.2\%$ on Virginia descent).*

### AASHTO Geometric Grade Regimes (`grade_regime`)
Discretizes continuous grades into standard interstate engineering thresholds:
$$\text{Regime}_i = \begin{cases} 
\text{steep\_downgrade} & \text{Grade}_i \le -2.0\% \\
\text{moderate\_downgrade} & -2.0\% < \text{Grade}_i \le -0.5\% \\
\text{flat\_tangent} & -0.5\% < \text{Grade}_i < +0.5\% \\
\text{moderate\_upgrade} & +0.5\% \le \text{Grade}_i < +2.0\% \\
\text{steep\_upgrade} & \text{Grade}_i \ge +2.0\%
\end{cases}$$

### Topographical Friction Index ($\text{TFI}$)
Composite multi-attribute impedance metric quantifying total geometric resistance to free vehicle flow:
$$\text{TFI}_i = w_g \frac{|\text{Grade}_i|}{\text{Grade}_{\text{norm}}} + w_\kappa \frac{\kappa_i}{\kappa_{\text{norm}}}$$
* **Default Parameters**:
  - Grade weight: $w_g = 0.7$
  - Curvature weight: $w_\kappa = 0.3$
  - Grade normalizing factor: $\text{Grade}_{\text{norm}} = 3.0\%$ (design limit for interstate rolling terrain)
  - Curvature normalizing factor: $\kappa_{\text{norm}} = 0.002\text{ m}^{-1}$ (corresponding to $R = 500\text{m}$)
* **Observed Corridor Distributions**:
  - **I-495 American Legion Bridge**: $\mu_{\text{TFI}} = 0.591$ (high friction shoulder, max $1.26$)
  - **I-95 Lorton-Woodbridge (Control)**: $\mu_{\text{TFI}} = 0.414$ (tangent-dominated)
  - **I-295 Anacostia (Control)**: $\mu_{\text{TFI}} = 0.438$ (urban flatland)

### Vehicle Emissions Slope Parameter
$$\text{Slope}_i = \sin(\theta_i) \approx \frac{\text{Grade}_i}{100.0}$$
Pre-computed for direct integration into EPA MOVES Vehicle Specific Power (VSP) equations in Phase 5:
$$\text{VSP} = v \cdot (1.1 a + 9.81 \sin\theta + 0.132) + 0.000302 v^3$$

---

## 4. Downstream Metric Specifications (Phases 2–5)

| Metric Category | Metric Symbol | Unit | Definition / Purpose |
| :--- | :---: | :---: | :--- |
| **Traffic States** | $v_{i,t}$ / $q_{i,t}$ / $k_{i,t}$ | $\text{mph}, \text{vphpl}, \text{vpm}$ | 5-minute harmonized space-mean speed, flow rate, and density ($k = q/v$). |
| **Density Thresholds** | $k_{\text{crit}}$ | $\text{veh}/\text{mile}/\text{lane}$ | Critical density at capacity inflection point on the Triangular Fundamental Diagram. |
| **Phase Breakdown** | $P(\text{breakdown})$ | probability $[0, 1]$ | Conditional probability of speed dropping below $45\text{ mph}$ for $\ge 15$ continuous minutes. |
| **Capacity Drop** | $\Delta Q$ | percent ($\%$) | Empirical loss in discharge capacity: $\Delta Q = \frac{Q_{\text{free}} - Q_{\text{queue}}}{Q_{\text{free}}} \times 100\%$. Typically $10\% - 20\%$. |
| **Shockwave Speed** | $w$ | $\text{mph}$ | Backward-propagating congestion wave speed: $w = \frac{\Delta q}{\Delta k} \approx -11.2\text{ mph}$. |
| **Econometric ATT** | $\tau_{\text{ATT}}$ | $\Delta \text{mph}$ or $\Delta \text{VHD}$ | Average Treatment Effect on the Treated estimated via Synthetic Difference-in-Differences (SDID). |
| **Travel Time Index** | $\text{TTI}$ | ratio | $\text{TTI} = \frac{\text{Travel Time}_{\text{actual}}}{\text{Travel Time}_{\text{free\_flow}}}$. Peak vs off-peak ratio. |
| **Buffer Time Index** | $\text{BTI}$ | ratio | $\text{BTI} = \frac{t_{95\text{th}} - t_{\text{avg}}}{t_{\text{avg}}}$. Quantifies travel time unreliability. |
| **Vehicle Hours of Delay** | $\text{VHD}$ | veh-hours | $\text{VHD} = \sum q_{i,t} \cdot \max\left(0, \frac{L_i}{v_{i,t}} - \frac{L_i}{v_{\text{free\_flow}}}\right)$. |
| **Monetized Delay Cost** | $\mathcal{C}_{\text{delay}}$ | $\text{USD} (\$$) | Monetized using USDOT Value of Time (\$24.60/hr passenger; \$42.00/hr commercial). |
