# Control Corridor Assessment (Phase F, data provenance review)

**Status**: Analysis only. No code or corridor configuration was changed. Decision pending.

## Why this review

The study design describes I-95 (Lorton/Woodbridge) and I-295 (Anacostia) as **flat** controls
(`docs/decisions.md`, Choice 4.2: "flat/mild topography, ≤ ±2.0%"). Measured USGS 3DEP elevations
contradict that. The comparison below uses the rebuilt `spatial_feature_mart.parquet`, where every
elevation is measured (USGS EPQS bare-earth or LiDAR Class 17) or explicitly flagged as interpolated
on an OSM-tagged bridge. Stations flagged `grade_edge_effect` are excluded.

## Measured comparison

| Corridor / direction | Peak descent | Peak climb | Mean \|grade\| | % stations \|grade\| ≥ 2.5% | Longest ≥2.5% descent | Longest ≥2.5% climb | Descent→sag→climb sequence* |
|---|---|---|---|---|---|---|---|
| ALB inner loop | −3.26% | +3.49% | 2.07% | 36.1% | 800 m | 900 m | **1,700 m** |
| ALB outer loop | −3.41% | +3.49% | 2.11% | 46.7% | 1,000 m | 800 m | **1,800 m** |
| I-95 northbound | −3.28% | +3.03% | 1.53% | 17.6% | 500 m | 800 m | 0 m |
| I-95 southbound | −3.03% | +3.60% | 1.61% | 21.1% | 800 m | 600 m | 0 m |
| I-295 northbound | −3.72% | +2.85% | 1.43% | 21.1% | 400 m | 600 m | 0 m |
| I-295 southbound | −2.91% | +3.67% | 1.52% | 22.0% | 400 m | 400 m | 0 m |

\* A run of ≥ 3 stations (≥ 300 m) at ≤ −2.5% followed within 1 km by a run at ≥ +2.5%, i.e. a
sustained descent into a sag and straight back out (combined length of the two runs).

Mean Topographical Friction Index: ALB 0.598, I-95 0.426, I-295 0.438.

## Findings

1. **The controls are not flat.** Their peak smoothed grades (about ±3.0–3.7%) match or exceed ALB's.
   I-295 has more sag-classified stations (51) than ALB (32). Any design or write-up that treats
   them as "flat" or "≈0% grade" is not supported by the data.
2. **What distinguishes ALB is the sustained descent–sag–climb sequence, not peak grade.** ALB is the
   only corridor with a ≥ 2.5% descent of 800–1,000 m followed almost immediately by a ≥ 2.5% climb
   (1.7–1.8 km combined). No control carriageway has such a sequence. ALB also spends roughly twice
   as much of its length at ≥ 2.5% grade (36–47% vs 18–22%).
3. **The current controls can remain usable** if the treatment is defined as "long sag sequence"
   rather than "sloped vs flat". Local grade then becomes a covariate measured on all corridors,
   which is how the asymmetric grade terms of TF-023 already treat it.

## Options

| Option | What changes | Trade-off |
|---|---|---|
| **A. Reframe treatment (recommended)** | Keep I-95/I-295; define treatment as the descent→sag→climb sequence; control for station grade, curvature and grade regime in TF-041–TF-043. | Least rework; matches the data. Requires updating TF-012's "Grade ≈ 0%" test and Decision 4.2 wording. |
| B. Match on grade regime | Keep corridors; compare ALB sag stations only with control stations in the same AASHTO grade regime. | Fewer matched observations; still the same corridors. |
| C. Choose new controls | Search the DC region for interstate segments with mean \|grade\| < 1% and no sag sequence. | Requires re-running TF-010–TF-013 and new telemetry sourcing; strongest "flat" contrast. |

## Caveats

- Interpolated bridge stations (ALB 5, I-95 9, I-295 9) carry no direct measurement. They are
  included in the table above; downstream models should report results with and without them.
- ALB deck elevations come from one LiDAR collection (USGS_LPC_VA_Fairfax_County_2018) and are not
  yet validated against an independent reference.
