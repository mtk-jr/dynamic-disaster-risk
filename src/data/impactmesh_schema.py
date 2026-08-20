REMOTE_FEATURES = [
    # Sentinel-1
    "s1_t0_vh_mean",
    "s1_t0_vh_std",
    "s1_t0_vv_mean",
    "s1_t0_vv_std",
    "s1_t0_vv_vh_difference",

    "s1_t1_vh_mean",
    "s1_t1_vh_std",
    "s1_t1_vv_mean",
    "s1_t1_vv_std",
    "s1_t1_vv_vh_difference",

    "s1_t2_vh_mean",
    "s1_t2_vh_std",
    "s1_t2_vv_mean",
    "s1_t2_vv_std",
    "s1_t2_vv_vh_difference",

    "s1_t3_vh_mean",
    "s1_t3_vh_std",
    "s1_t3_vv_mean",
    "s1_t3_vv_std",
    "s1_t3_vv_vh_difference",

    "s1_vh_change",
    "s1_vv_change",

    # Sentinel-2 changes
    "s2_mndwi_change",
    "s2_ndvi_change",
    "s2_ndwi_change",

    # Sentinel-2 t0
    "s2_t0_mndwi_mean",
    "s2_t0_mndwi_std",
    "s2_t0_ndvi_mean",
    "s2_t0_ndwi_mean",
    "s2_t0_ndwi_std",
    "s2_t0_water_ratio",

    # Sentinel-2 t1
    "s2_t1_mndwi_mean",
    "s2_t1_mndwi_std",
    "s2_t1_ndvi_mean",
    "s2_t1_ndwi_mean",
    "s2_t1_ndwi_std",
    "s2_t1_water_ratio",

    # Sentinel-2 t2
    "s2_t2_mndwi_mean",
    "s2_t2_mndwi_std",
    "s2_t2_ndvi_mean",
    "s2_t2_ndwi_mean",
    "s2_t2_ndwi_std",
    "s2_t2_water_ratio",

    # Sentinel-2 t3
    "s2_t3_mndwi_mean",
    "s2_t3_mndwi_std",
    "s2_t3_ndvi_mean",
    "s2_t3_ndwi_mean",
    "s2_t3_ndwi_std",
    "s2_t3_water_ratio",

    "s2_water_ratio_change",
]


GIS_FEATURES = [
    "dem_max",
    "dem_mean",
    "dem_median",
    "dem_min",
    "dem_p25",
    "dem_p75",
    "dem_range",
    "dem_std",
]


TARGET = "flood_ratio"


ID_COLUMNS = [
    "sample_id",
]