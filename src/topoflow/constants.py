"""Centralized domain constants for TopoFlow.

Defines Coordinate Reference Systems (CRS), spatial discretization parameters,
kinematic bounds, meteorological shock thresholds, and economic valuation constants.
"""

# =============================================================================
# Coordinate Reference Systems (CRS)
# =============================================================================

# Standard unprojected geodetic CRS for GPS ingestion, OpenStreetMap, and GeoJSON export.
CRS_WGS84: str = "EPSG:4326"

# Master metric analytical projection: NAD83 / UTM Zone 18N.
# Spans Maryland and Virginia without state-line projection discontinuities (<0.04% distortion).
# Used for all 100m station discretization, curvature, buffers, and vertical grades.
CRS_METRIC: str = "EPSG:26918"


# =============================================================================
# Spatial Discretization & Geometry
# =============================================================================

# Discretization interval for highway centerline chainage (meters).
SPATIAL_RESOLUTION_METERS: float = 100.0

# Curvature threshold cap for straight highway tangents (meters).
MAX_STRAIGHT_RADIUS_METERS: float = 10000.0


# =============================================================================
# Traffic & Kinematic Constraints
# =============================================================================

# Maximum theoretical jam density cap (vehicles per mile per lane).
JAM_DENSITY_VPMPL: float = 220.0

# Default highway free-flow speed baseline (miles per hour).
DEFAULT_FREE_FLOW_SPEED_MPH: float = 65.0


# =============================================================================
# Weather & Meteorological Shocks
# =============================================================================

# 15-minute precipitation accumulation threshold defining a rain shock (millimeters).
RAIN_SHOCK_THRESHOLD_MM: float = 1.25


# =============================================================================
# Economic Valuation & Impact Constants (USDOT Guidance)
# =============================================================================

# Revised USDOT standard value of travel time for personal vehicles ($/veh-hour).
VOT_PERSONAL_PER_HOUR: float = 24.60

# Revised USDOT standard value of travel time for commercial freight ($/truck-hour).
VOT_COMMERCIAL_PER_HOUR: float = 42.00

# Regional wage-adjusted value of travel time for DC/Northern Virginia ($/hour).
VOT_REGIONAL_WAGE_PER_HOUR: float = 38.00
