"""Visualization module for highway elevation profiles and Savitzky-Golay vs raw discrete difference diagnostics.

Generates publication-ready figures for Phase 1 milestone visual verification:
1. outputs/figures/phase1/p1_elevation_grade_profile.png (Corridor elevation, vertical grade, curvature, and control flatness)
2. outputs/figures/phase1/p1_savgol_vs_raw_differences.png (Empirical performance comparison of Savitzky-Golay vs raw finite differences)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import savgol_filter

from topoflow.config import Settings, get_settings

logger = logging.getLogger(__name__)


# (elevation_source, color, marker size, legend label)
_SOURCE_STYLES: tuple[tuple[str, str, int, str], ...] = (
    ("usgs_epqs", "#2ca02c", 18, "USGS 3DEP bare-earth (EPQS)"),
    ("usgs_dem", "#17becf", 18, "USGS 3DEP bare-earth (DEM)"),
    ("lidar_c17", "#d62728", 35, "LiDAR Class 17 bridge deck"),
    ("interpolated", "#ff7f0e", 35, "Interpolated (bridge, no LiDAR)"),
    ("bare_earth_under_bridge", "#7f7f7f", 35, "Bare earth under bridge (deck unknown)"),
    ("unknown", "#bcbd22", 18, "Unknown source"),
)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Return (first, last) indices of the longest contiguous True run, or None."""
    best: tuple[int, int] | None = None
    start: int | None = None
    for i, flag in enumerate(np.append(np.asarray(mask, dtype=bool), False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or (i - 1 - start) > (best[1] - best[0]):
                best = (start, i - 1)
            start = None
    return best


def _edge_mask(df: pl.DataFrame) -> np.ndarray:
    """Stations flagged as Savitzky-Golay edge-effect (all False when column absent)."""
    if "grade_edge_effect" in df.columns:
        return df["grade_edge_effect"].to_numpy().astype(bool)
    return np.zeros(len(df), dtype=bool)


def plot_corridor_elevation_profile(
    df_elevation: pl.DataFrame,
    output_path: Path | None = None,
    settings: Settings | None = None,
    dpi: int = 300,
) -> Path:
    """Generate multi-panel figure displaying elevation, grade, curvature, and control corridors.

    Args:
        df_elevation: Polars DataFrame from process_corridor_stations.
        output_path: Optional destination path for the generated PNG figure.
        settings: Optional Settings instance for path resolution.
        dpi: Output image resolution (default 300).

    Returns:
        Path to the saved figure.

    Raises:
        ValueError: If df_elevation is empty.
    """
    if len(df_elevation) == 0:
        raise ValueError("Cannot plot elevation profile: DataFrame is empty.")

    active_settings = settings or get_settings()
    if output_path is None:
        output_path = (
            active_settings.project_root
            / "outputs"
            / "figures"
            / "phase1"
            / "p1_elevation_grade_profile.png"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # 1. Study Corridor Elevation Profile (ALB Inner Loop)
    ax_elev = fig.add_subplot(gs[0, :])
    df_alb_inner = df_elevation.filter(
        (pl.col("corridor_id") == "alb_i495") & (pl.col("direction") == "inner_loop")
    ).sort("chainage_m")

    if len(df_alb_inner) > 0:
        km = df_alb_inner["chainage_m"].to_numpy() / 1000.0
        z_smooth = df_alb_inner["elevation_smooth_m"].to_numpy()
        z_raw = df_alb_inner["elevation_raw_m"].to_numpy()
        is_bridge = df_alb_inner["is_bridge_deck"].to_numpy().astype(bool)
        sources = (
            np.asarray(df_alb_inner["elevation_source"].to_list(), dtype=object)
            if "elevation_source" in df_alb_inner.columns
            else np.full(len(df_alb_inner), "unknown", dtype=object)
        )

        ax_elev.plot(
            km,
            z_smooth,
            color="#1f77b4",
            lw=2.5,
            label="Roadway Centerline (Savitzky-Golay)",
        )
        for src, color, size, label in _SOURCE_STYLES:
            mask = sources == src
            if np.any(mask):
                ax_elev.scatter(
                    km[mask],
                    z_raw[mask],
                    color=color,
                    s=size,
                    zorder=5,
                    label=f"{label} (n={int(mask.sum())})",
                )

        # Main bridge span = longest contiguous run of OSM bridge-tagged stations
        span = _longest_true_run(is_bridge)
        if span is not None:
            i0, i1 = span
            span_len_m = (km[i1] - km[i0]) * 1000.0
            ax_elev.axvspan(
                km[i0],
                km[i1],
                color="#fee08b",
                alpha=0.40,
                label=f"Longest OSM bridge run (stations {i0}-{i1}, {span_len_m:.0f} m)",
            )

        # Data-derived annotations (no hard-coded landmark values)
        valid = ~_edge_mask(df_alb_inner)
        grade = df_alb_inner["grade_pct"].to_numpy()
        i_low = int(np.argmin(np.where(valid, z_smooth, np.inf)))
        i_desc = int(np.argmin(np.where(valid, grade, np.inf)))
        i_climb = int(np.argmax(np.where(valid, grade, -np.inf)))
        y_top = float(np.nanmax(z_raw)) + 6.0
        annotations = [
            (km[i_desc], f"Steepest descent\n{grade[i_desc]:+.2f}%"),
            (km[i_low], f"Sag low point\n{z_smooth[i_low]:.1f} m"),
            (km[i_climb], f"Steepest climb\n{grade[i_climb]:+.2f}%"),
        ]
        for k, (x_pos, label_text) in enumerate(annotations):
            ax_elev.axvline(x_pos, color="#888888", ls=":", lw=0.9, alpha=0.7)
            ax_elev.text(
                x_pos,
                y_top + (11.0 if k % 2 else 0.0),  # stagger to avoid label overlap
                label_text,
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="semibold",
                color="#222222",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="#f8f9fa",
                    edgecolor="#cccccc",
                    alpha=0.9,
                ),
            )
        ax_elev.set_ylim(max(0.0, float(np.nanmin(z_raw)) - 10.0), y_top + 26.0)

        ax_elev.set_title(
            "I-495 American Legion Bridge Inner Loop: Measured Elevation Profile "
            "(USGS 3DEP EPQS + LiDAR Class 17)",
            fontsize=13,
            fontweight="bold",
        )
        ax_elev.set_xlabel("Chainage Distance (km)", fontsize=11)
        ax_elev.set_ylabel("Elevation NAVD88 (m)", fontsize=11)
        ax_elev.grid(True, linestyle=":", alpha=0.6)
        ax_elev.legend(loc="lower left", frameon=True, fontsize=8)

    # 2. Vertical Grade Profile (Grade %)
    ax_grade = fig.add_subplot(gs[1, 0])
    if len(df_alb_inner) > 0:
        grade = df_alb_inner["grade_pct"].to_numpy()
        ax_grade.plot(km, grade, color="#ff7f0e", lw=2.2, label="Savitzky-Golay Grade (%)")
        ax_grade.axhline(0.0, color="black", lw=0.8, ls="--")
        valid_g = ~_edge_mask(df_alb_inner)
        g_min = float(np.min(grade[valid_g])) if np.any(valid_g) else float(np.min(grade))
        g_max = float(np.max(grade[valid_g])) if np.any(valid_g) else float(np.max(grade))
        ax_grade.axhline(
            g_min, color="#d62728", lw=1.0, ls=":", label=f"Observed max descent ({g_min:+.2f}%)"
        )
        ax_grade.axhline(
            g_max, color="#2ca02c", lw=1.0, ls=":", label=f"Observed max climb ({g_max:+.2f}%)"
        )
        ax_grade.fill_between(
            km,
            grade,
            0.0,
            where=(grade < 0),
            color="#d62728",
            alpha=0.15,
            label="Descent Zone",
        )
        ax_grade.fill_between(
            km,
            grade,
            0.0,
            where=(grade > 0),
            color="#2ca02c",
            alpha=0.15,
            label="Ascent Zone",
        )

        ax_grade.set_title(
            "Vertical Percent Grade Profile (% Slope)", fontsize=12, fontweight="bold"
        )
        ax_grade.set_xlabel("Chainage Distance (km)", fontsize=10)
        ax_grade.set_ylabel("Vertical Grade (%)", fontsize=10)
        ax_grade.grid(True, linestyle=":", alpha=0.6)
        ax_grade.legend(loc="lower right", frameon=True, fontsize=8)

    # 3. Vertical Curvature & Sag Classification (d^2z/dx^2)
    ax_curv = fig.add_subplot(gs[1, 1])
    if len(df_alb_inner) > 0:
        curv = (
            df_alb_inner["vertical_curvature"].to_numpy() * 10000.0
        )  # Scale for readability (per 10km)
        is_sag = df_alb_inner["is_sag"].to_numpy()
        ax_curv.plot(km, curv, color="#9467bd", lw=2.0, label="Curvature (d²z/dx² × 10⁴)")
        ax_curv.scatter(
            km[is_sag],
            curv[is_sag],
            color="#8c564b",
            s=30,
            label="Sag Inflection Trough",
        )
        ax_curv.axhline(0.0, color="black", lw=0.8, ls="--")

        ax_curv.set_title(
            "AASHTO Vertical Curvature & Sag Inflection", fontsize=12, fontweight="bold"
        )
        ax_curv.set_xlabel("Chainage Distance (km)", fontsize=10)
        ax_curv.set_ylabel("Curvature (× 10⁻⁴ m⁻¹)", fontsize=10)
        ax_curv.grid(True, linestyle=":", alpha=0.6)
        ax_curv.legend(loc="upper left", frameon=True, fontsize=8)

    # 4. Control Corridor 1: I-95 (Lorton / Woodbridge)
    ax_c1 = fig.add_subplot(gs[2, 0])
    df_i95 = df_elevation.filter(
        (pl.col("corridor_id") == "i95_lorton_woodbridge") & (pl.col("direction") == "northbound")
    ).sort("chainage_m")

    if len(df_i95) > 0:
        km_i95 = df_i95["chainage_m"].to_numpy() / 1000.0
        grade_i95 = df_i95["grade_pct"].to_numpy()
        ax_c1.plot(km_i95, grade_i95, color="#17becf", lw=1.8, label="I-95 NB Grade (%)")
        ax_c1.axhline(0.0, color="black", lw=0.8, ls="--")
        max_g1 = max(3.5, float(np.nanmax(np.abs(grade_i95))) + 0.5)
        ax_c1.set_ylim(-max_g1, max_g1)
        ax_c1.set_title(
            f"Control 1: I-95 Lorton/Woodbridge NB (Mean: {np.mean(grade_i95):.2f}% ± {np.std(grade_i95):.2f}%)",
            fontsize=11,
            fontweight="bold",
        )
        ax_c1.set_xlabel("Chainage Distance (km)", fontsize=10)
        ax_c1.set_ylabel("Grade (%)", fontsize=10)
        ax_c1.grid(True, linestyle=":", alpha=0.6)
        ax_c1.legend(loc="lower right", frameon=True, fontsize=8)

    # 5. Control Corridor 2: I-295 (Anacostia Freeway)
    ax_c2 = fig.add_subplot(gs[2, 1])
    df_i295 = df_elevation.filter(
        (pl.col("corridor_id") == "i295_anacostia") & (pl.col("direction") == "northbound")
    ).sort("chainage_m")

    if len(df_i295) > 0:
        km_i295 = df_i295["chainage_m"].to_numpy() / 1000.0
        grade_i295 = df_i295["grade_pct"].to_numpy()
        ax_c2.plot(km_i295, grade_i295, color="#bcbd22", lw=1.8, label="I-295 NB Grade (%)")
        ax_c2.axhline(0.0, color="black", lw=0.8, ls="--")
        max_g2 = max(3.5, float(np.nanmax(np.abs(grade_i295))) + 0.5)
        ax_c2.set_ylim(-max_g2, max_g2)
        ax_c2.set_title(
            f"Control 2: I-295 Anacostia NB (Mean: {np.mean(grade_i295):.2f}% ± {np.std(grade_i295):.2f}%)",
            fontsize=11,
            fontweight="bold",
        )
        ax_c2.set_xlabel("Chainage Distance (km)", fontsize=10)
        ax_c2.set_ylabel("Grade (%)", fontsize=10)
        ax_c2.grid(True, linestyle=":", alpha=0.6)
        ax_c2.legend(loc="lower right", frameon=True, fontsize=8)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved corridor elevation profile figure to %s", output_path)
    return output_path


def plot_savgol_vs_raw_differences(
    df_elevation: pl.DataFrame,
    metrics: dict[str, float] | None = None,
    output_path: Path | None = None,
    settings: Settings | None = None,
    dpi: int = 300,
) -> Path:
    """Generate comparative diagnostic figure proving Savitzky-Golay performance superiority.

    Args:
        df_elevation: Polars DataFrame containing both grade_pct (S-G) and grade_raw_pct.
        metrics: Optional dictionary of noise reduction metrics.
        output_path: Optional destination path for the generated PNG figure.
        settings: Optional Settings instance for path resolution.
        dpi: Output image resolution (default 300).

    Returns:
        Path to the saved figure.

    Raises:
        ValueError: If df_elevation is empty.
    """
    if len(df_elevation) == 0:
        raise ValueError("Cannot plot S-G diagnostics: DataFrame is empty.")

    active_settings = settings or get_settings()
    if output_path is None:
        output_path = (
            active_settings.project_root
            / "outputs"
            / "figures"
            / "phase1"
            / "p1_savgol_vs_raw_differences.png"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(16, 11), dpi=dpi)

    df_alb = df_elevation.filter(
        (pl.col("corridor_id") == "alb_i495") & (pl.col("direction") == "inner_loop")
    ).sort("chainage_m")

    if len(df_alb) == 0:
        df_alb = df_elevation.sort("chainage_m")

    km = df_alb["chainage_m"].to_numpy() / 1000.0
    z_raw = df_alb["elevation_raw_m"].to_numpy()
    z_sg = df_alb["elevation_smooth_m"].to_numpy()
    grade_raw = df_alb["grade_raw_pct"].to_numpy()
    grade_sg = df_alb["grade_pct"].to_numpy()

    # Panel A: Elevation Profile (Raw vs Savitzky-Golay Smoothed)
    axs[0, 0].scatter(
        km, z_raw, color="#7f7f7f", alpha=0.65, s=20, label="Raw Sampled Elevations (z)"
    )
    axs[0, 0].plot(
        km,
        z_sg,
        color="#1f77b4",
        lw=2.4,
        label="Savitzky-Golay Filtered Surface (w=5, p=2)",
    )
    axs[0, 0].set_title(
        "Panel A: Elevation Profile Surface Filtering", fontsize=12, fontweight="bold"
    )
    axs[0, 0].set_xlabel("Chainage (km)", fontsize=10)
    axs[0, 0].set_ylabel("Elevation NAVD88 (m)", fontsize=10)
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)
    axs[0, 0].legend(loc="upper right", frameon=True, fontsize=9)

    # Panel B: Grade Comparison (% Slope): Raw Discrete Differences vs Savitzky-Golay
    axs[0, 1].plot(
        km,
        grade_raw,
        color="#d62728",
        lw=1.2,
        alpha=0.75,
        ls="--",
        label="Raw Central Differences (dz/dx)",
    )
    axs[0, 1].plot(km, grade_sg, color="#1b9e77", lw=2.4, label="Savitzky-Golay Analytical Grade")
    axs[0, 1].axhline(0.0, color="black", lw=0.8, ls=":")
    axs[0, 1].set_title(
        "Panel B: Vertical Grade Estimates (% Slope)", fontsize=12, fontweight="bold"
    )
    axs[0, 1].set_xlabel("Chainage (km)", fontsize=10)
    axs[0, 1].set_ylabel("Vertical Grade (%)", fontsize=10)
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)
    axs[0, 1].legend(loc="lower right", frameon=True, fontsize=9)

    # Panel C: Station-to-Station Grade Jitter |Δ Grade| (Total Variation Roughness)
    jitter_raw = np.abs(np.diff(grade_raw))
    jitter_sg = np.abs(np.diff(grade_sg))
    km_mid = km[:-1] + 0.05

    tv_raw = (
        float(np.sum(jitter_raw))
        if metrics is None
        else metrics.get("total_variation_raw", float(np.sum(jitter_raw)))
    )
    tv_sg = (
        float(np.sum(jitter_sg))
        if metrics is None
        else metrics.get("total_variation_savgol", float(np.sum(jitter_sg)))
    )
    tv_red = float((tv_raw - tv_sg) / tv_raw * 100.0) if tv_raw > 0 else 0.0

    axs[1, 0].bar(
        km_mid - 0.015,
        jitter_raw,
        width=0.03,
        color="#e41a1c",
        alpha=0.6,
        label=f"Raw Jitter (TV: {tv_raw:.2f}%)",
    )
    axs[1, 0].bar(
        km_mid + 0.015,
        jitter_sg,
        width=0.03,
        color="#377eb8",
        alpha=0.8,
        label=f"S-G Jitter (TV: {tv_sg:.2f}%)",
    )
    axs[1, 0].set_title(
        f"Panel C: Step-to-Step Jitter |ΔGrade| (Total Variation Reduction: {tv_red:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axs[1, 0].set_xlabel("Chainage (km)", fontsize=10)
    axs[1, 0].set_ylabel("|Grade(i+1) - Grade(i)| (%)", fontsize=10)
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)
    axs[1, 0].legend(loc="upper right", frameon=True, fontsize=9)

    # Panel D: SYNTHETIC benchmark. Known parabola + noise; illustrates the filter, not measured data
    x_bench = np.linspace(0, 1000, 101)
    g_true = -3.2 + (7.0 / 1000.0) * x_bench
    rng = np.random.default_rng(42)
    z_bench = 50.0 - 0.032 * x_bench + (0.07 / 2000.0) * (x_bench**2) + rng.normal(0, 0.05, 101)

    raw_g_bench = np.gradient(z_bench, x_bench) * 100.0
    sg_g_bench = savgol_filter(z_bench, window_length=5, polyorder=2, deriv=1, delta=10.0) * 100.0

    err_raw = raw_g_bench - g_true
    err_sg = sg_g_bench - g_true

    rmse_raw = float(np.sqrt(np.mean(err_raw**2)))
    rmse_sg = float(np.sqrt(np.mean(err_sg**2)))

    bins = np.linspace(-0.8, 0.8, 25)
    axs[1, 1].hist(
        err_raw,
        bins=bins,
        color="#e41a1c",
        alpha=0.45,
        density=True,
        label=f"Raw Discrete Diff (RMSE = {rmse_raw:.3f}%)",
    )
    axs[1, 1].hist(
        err_sg,
        bins=bins,
        color="#4daf4a",
        alpha=0.65,
        density=True,
        label=f"Savitzky-Golay (RMSE = {rmse_sg:.3f}%)",
    )
    axs[1, 1].axvline(0.0, color="black", lw=1.0, ls="--")

    axs[1, 1].set_title(
        f"Panel D: SYNTHETIC Benchmark, not measured data (RMSE ratio {rmse_raw / rmse_sg:.2f}×)",
        fontsize=12,
        fontweight="bold",
    )
    axs[1, 1].set_xlabel(
        "Error vs synthetic ground truth (%; parabola + σ=5 cm noise)", fontsize=10
    )
    axs[1, 1].set_ylabel("Probability Density", fontsize=10)
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)
    axs[1, 1].legend(loc="upper right", frameon=True, fontsize=9)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved S-G vs raw discrete differences comparison to %s", output_path)
    return output_path
