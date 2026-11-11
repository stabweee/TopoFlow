"""Corridor centerline and road geometry visualization utilities.

Renders plan-view highway geometries, directional carriageways, and bridge structures
for study and control corridors to verify topological continuity and physical plausibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

from topoflow.config import Settings, get_settings

logger = logging.getLogger(__name__)


def plot_corridors(
    gdf: gpd.GeoDataFrame,
    output_path: Path | None = None,
    settings: Settings | None = None,
    dpi: int = 300,
) -> Path:
    """Render publication-quality plan-view map of highway corridors.

    Generates a multi-panel figure displaying directional centerlines and bridge structures
    for the I-495 American Legion Bridge study corridor and I-95 / I-295 control corridors.

    Args:
        gdf: GeoDataFrame containing corridor geometries (from OSMExtractor).
        output_path: Target image file path (defaults to outputs/figures/p1_corridor_centerlines.png).
        settings: Optional Settings instance.
        dpi: Resolution of saved figure.

    Returns:
        Path to the saved figure image.
    """
    cfg = settings or get_settings()
    target_path = output_path or (cfg.outputs_dir / "figures" / "p1_corridor_centerlines.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    corridor_ids = list(gdf["corridor_id"].unique())
    n_corridors = len(corridor_ids)
    if n_corridors == 0:
        raise ValueError("GeoDataFrame contains no corridor records to plot.")

    # Create multi-panel figure
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_corridors,
        figsize=(5.5 * n_corridors, 5.0),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    colors = {
        "inner_loop": "#0ea5e9",  # Sky blue
        "outer_loop": "#f43f5e",  # Rose
        "northbound": "#0ea5e9",  # Sky blue
        "southbound": "#f43f5e",  # Rose
    }

    labels = {
        "inner_loop": "Inner Loop (VA → MD)",
        "outer_loop": "Outer Loop (MD → VA)",
        "northbound": "Northbound",
        "southbound": "Southbound",
    }

    for idx, corridor_id in enumerate(corridor_ids):
        ax = axes_flat[idx]
        subset = gdf[gdf["corridor_id"] == corridor_id]
        name = subset["name"].iloc[0] if "name" in subset.columns else corridor_id

        for _, row in subset.iterrows():
            geom = row.geometry
            direction = row.get("direction", "unknown")
            color = colors.get(direction, "#64748b")
            label = labels.get(direction, direction.replace("_", " ").title())

            if geom is not None and not geom.is_empty:
                x, y = geom.xy
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=2.8,
                    solid_capstyle="round",
                    label=label,
                    alpha=0.9,
                )
                # Station vertices
                ax.scatter(x, y, color=color, s=14, zorder=4, alpha=0.8)

                # Bridge highlight if flagged
                if row.get("has_bridge"):
                    # Find mid-point to annotate bridge
                    mid_idx = len(x) // 2
                    ax.scatter(
                        [x[mid_idx]],
                        [y[mid_idx]],
                        color="#f59e0b",
                        s=80,
                        marker="^",
                        zorder=5,
                        edgecolors="#78350f",
                        linewidth=1.2,
                        label="Bridge Span" if idx == 0 and direction == "inner_loop" else None,
                    )

        ax.set_title(f"{name}", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Longitude (deg W)", fontsize=9)
        ax.set_ylabel("Latitude (deg N)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "TopoFlow — Highway Vector Corridors & Directional Centerlines (Phase 1)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(target_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved corridor visualization figure to %s", target_path)
    return target_path
