import geopandas as gpd

def load_vector(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("Input GIS layer has no CRS.")
    return gdf
