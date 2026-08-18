import numpy as np

def normalized_difference(a, b, eps=1e-6):
    return (a - b) / (a + b + eps)

def sentinel2_indices(b03, b04, b08, b11):
    ndvi = normalized_difference(b08, b04)
    ndwi = normalized_difference(b03, b08)
    mndwi = normalized_difference(b03, b11)
    return {
        "ndvi": float(np.nanmean(ndvi)),
        "ndwi": float(np.nanmean(ndwi)),
        "mndwi": float(np.nanmean(mndwi)),
    }
