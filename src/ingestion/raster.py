import numpy as np
import rasterio

def raster_statistics(path):
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "median": float(np.nanmedian(arr)),
    }
