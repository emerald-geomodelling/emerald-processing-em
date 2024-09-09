import geopandas as gpd
import mercantile
import pandas as pd
import shapely.geometry
import requests
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import mapbox_vector_tile

class Maptiler(object):
    def __init__(self, url):
        self.url = url
        self.config = requests.get(url).json()

    def tiles(self, west, south, east, north, zoom = None):
        for data, config in self.raw_tiles(west, south, east, north, zoom):
            yield self._scale_tile(data, config)

    def raw_tiles(self, west, south, east, north, zoom = None):
        for tile, url in self.tile_configs(west, south, east, north, zoom):
            yield self._download_tile(url), tile
            
    def tile_configs(self, west, south, east, north, zoom = None):
        if zoom is None: zoom = self.config["maxzoom"]
        for tile in mercantile.tiles(west, south, east, north, zoom):
            yield tile, self._tile_url(tile)

    @classmethod
    def _scale_tile(cls, gdf, tile):
        bounds = mercantile.xy_bounds(tile)
        geometry = gdf.geometry.scale(
            (bounds.right - bounds.left)/4096., (bounds.top - bounds.bottom)/4096.,
            origin=(0,0,0)
        ).translate(bounds.left, bounds.bottom)
        return gpd.GeoDataFrame(gdf, geometry=geometry)
            
    def _tile_url(self, tile):
        url = self.config["tiles"][0]
        for k in ("x", "y", "z"):
            url = url.replace("{%s}" % k, "%s" % getattr(tile, k))
        return url
            
    @classmethod
    def _all_features(cls, t):
        for layer_name, layer in t.items():
            for feature in layer["features"]:
                yield layer_name, feature

    @classmethod
    def _download_tile(cls, url):
        with requests.get(url) as r:
            t = mapbox_vector_tile.decode(r.content)

        return gpd.GeoDataFrame(
            [{"layer": layer_name, **feature["properties"]}
             for layer_name, feature in cls._all_features(t)],
            geometry=[shapely.geometry.shape(feature["geometry"]) for layer_name, feature in cls._all_features(t)])
