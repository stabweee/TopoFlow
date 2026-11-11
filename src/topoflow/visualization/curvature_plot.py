"""Horizontal curvature and station discretization visualization utilities.

Generates publication-quality figures depicting 100-meter chainage station locations,
horizontal curvature profiles, and radius of curvature distributions across study
and control corridors for physical plausibility verification.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from topoflow.config import Settings, get_settings

logger = logging.getLogger(__name__)


def plot_horizontal_curvature(
    stations_gdf: gpd.GeoDataFrame,
    output_path: Path | None = None,
    settings: Settings | None = None,
    dpi: int = 300,
) -> Path:
    """Render publication-quality horizontal curvature profiles and plan-view maps.

    Args:
        stations_gdf: GeoDataFrame containing 100m station records with curvature metrics.
        output_path: Destination file path (defaults to outputs/figures/p1_horizontal_curvature.png).
        settings: Optional Settings instance.
        dpi: Image resolution.

    Returns:
        Path to the saved PNG image.

    Raises:
        ValueError: If stations_gdf is empty or missing required columns.
    """
    if stations_gdf.empty:
        raise ValueError("GeoDataFrame contains no station records to plot.")

    required_cols = {
        "segment_id",
        "corridor_id",
        "direction",
        "chainage_m",
        "curvature_kappa",
        "radius_m",
    }
    missing = required_cols - set(stations_gdf.columns)
    if missing:
        raise ValueError(f"stations_gdf missing required columns: {missing}")

    cfg = settings or get_settings()
    target_path = output_path or (cfg.outputs_dir / "figures" / "p1_horizontal_curvature.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    # Panel 1 (Top Left): Plan View of ALMB Study Corridor with Stations colored by Curvature
    ax_map = fig.add_subplot(gs[0, 0])
    alb_df = stations_gdf[stations_gdf["corridor_id"] == "alb_i495"].copy()

    if not alb_df.empty:
        # Convert curvature to 1/km for readable color mapping
        alb_df["kappa_per_km"] = alb_df["curvature_kappa"] * 1000.0

        scatter = ax_map.scatter(
            alb_df["longitude"],
            alb_df["latitude"],
            c=alb_df["kappa_per_km"],
            cmap="plasma",
            s=28,
            edgecolors="none",
            alpha=0.9,
            zorder=3,
        )
        cbar = plt.colorbar(scatter, ax=ax_map, pad=0.02, shrink=0.85)
        cbar.set_label("Curvature $\\kappa$ ($\\mathrm{km}^{-1}$)", fontsize=10)

        # Highlight bridge crossing latitude band (Potomac River ~ 38.972 - 38.976)
        ax_map.axhspan(
            38.968, 38.977, color="lightblue", alpha=0.3, zorder=1, label="Potomac River Span"
        )
        ax_map.set_title(
            "I-495 American Legion Bridge (100m Stations)", fontsize=12, fontweight="bold"
        )
        ax_map.set_xlabel("Longitude (deg WGS84)", fontsize=10)
        ax_map.set_ylabel("Latitude (deg WGS84)", fontsize=10)
        ax_map.grid(True, linestyle="--", alpha=0.5)
        ax_map.legend(loc="upper left", fontsize=9)
    else:
        ax_map.text(0.5, 0.5, "No alb_i495 stations found", ha="center", va="center")

    # Panel 2 (Top Right): Curvature Distribution & Boxplots Across Corridors
    ax_box = fig.add_subplot(gs[0, 1])
    corridors = list(stations_gdf["corridor_id"].unique())
    box_data = []
    labels = []

    for cid in corridors:
        sub = stations_gdf[stations_gdf["corridor_id"] == cid]
        # Filter finite non-capped radii
        curved_radii = sub[sub["radius_m"] < 9999.0]["radius_m"]
        if not curved_radii.empty:
            box_data.append(curved_radii)
            labels.append(cid.replace("_", "\n"))

    if box_data:
        bp = ax_box.boxplot(box_data, tick_labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4A90E2")
            patch.set_alpha(0.7)
        ax_box.set_title(
            "Horizontal Curve Radius Distribution (Curved Segments)", fontsize=12, fontweight="bold"
        )
        ax_box.set_ylabel("Radius of Curvature $R$ (meters)", fontsize=10)
        ax_box.axhline(500, color="red", linestyle=":", label="AASHTO 55mph Min R (~500m)")
        ax_box.legend(loc="upper right", fontsize=9)
        ax_box.grid(True, linestyle="--", alpha=0.5)
    else:
        ax_box.text(0.5, 0.5, "All segments straight tangents", ha="center", va="center")

    # Panel 3 (Bottom Left): I-495 ALMB Chainage vs Curvature Profile
    ax_prof = fig.add_subplot(gs[1, 0])
    if not alb_df.empty:
        directions = alb_df["direction"].unique()
        colors = {"inner_loop": "#D9534F", "outer_loop": "#0275D8"}

        for d in directions:
            d_sub = alb_df[alb_df["direction"] == d].sort_values("chainage_m")
            chainage_km = d_sub["chainage_m"] / 1000.0
            kappa_km = d_sub["curvature_kappa"] * 1000.0
            ax_prof.plot(
                chainage_km,
                kappa_km,
                marker="o",
                markersize=3,
                label=f"{d.replace('_', ' ').title()}",
                color=colors.get(d, "gray"),
                linewidth=1.5,
                alpha=0.85,
            )

        ax_prof.set_title(
            "I-495 ALMB Curvature vs. Chainage Profile", fontsize=12, fontweight="bold"
        )
        ax_prof.set_xlabel("Chainage Distance along Corridor (km)", fontsize=10)
        ax_prof.set_ylabel("Curvature $\\kappa$ ($\\mathrm{km}^{-1}$)", fontsize=10)
        ax_prof.grid(True, linestyle="--", alpha=0.5)
        ax_prof.legend(loc="upper right", fontsize=9)

    # Panel 4 (Bottom Right): Control Corridors Curvature Profiles
    ax_ctrl = fig.add_subplot(gs[1, 1])
    ctrl_df = stations_gdf[stations_gdf["corridor_id"] != "alb_i495"]

    if not ctrl_df.empty:
        ctrl_corridors = ctrl_df["corridor_id"].unique()
        ctrl_colors = ["#5CB85C", "#F0AD4E", "#6F42C1"]

        for idx, cid in enumerate(ctrl_corridors):
            c_sub = ctrl_df[ctrl_df["corridor_id"] == cid]
            # Plot dominant direction
            main_dir = c_sub["direction"].iloc[0]
            d_sub = c_sub[c_sub["direction"] == main_dir].sort_values("chainage_m")
            ax_ctrl.plot(
                d_sub["chainage_m"] / 1000.0,
                d_sub["curvature_kappa"] * 1000.0,
                label=f"{cid} ({main_dir})",
                color=ctrl_colors[idx % len(ctrl_colors)],
                linewidth=1.3,
                alpha=0.8,
            )

        ax_ctrl.set_title("Control Corridors Curvature Profiles", fontsize=12, fontweight="bold")
        ax_ctrl.set_xlabel("Chainage Distance along Corridor (km)", fontsize=10)
        ax_ctrl.set_ylabel("Curvature $\\kappa$ ($\\mathrm{km}^{-1}$)", fontsize=10)
        ax_ctrl.grid(True, linestyle="--", alpha=0.5)
        ax_ctrl.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "TopoFlow Phase 1: 100-Meter Spatial Discretization & Horizontal Curvature Profiling",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.savefig(target_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved horizontal curvature profile figure to %s", target_path)
    return target_path
