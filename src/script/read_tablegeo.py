import geopandas as gpd

CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"

import fiona

couches = fiona.listlayers(CHEMIN_GPKG)
print("Couches disponibles dans le fichier : ")
for c in couches:
    print(f"  - {c}")


gdf = gpd.read_file(CHEMIN_GPKG,layer=couches[0])

print(f"\nNombre de lignes : {len(gdf)}")
print(f"\nColonnes : {list(gdf.columns)}")
print(f"\nSysteme de coordonnees (CRS) : {gdf.crs}")
print(f"\nType de geometrie : {gdf.geom_type.unique()}")
print(f"\nApercu des 5 premieres lignes : ")
print(gdf.head())