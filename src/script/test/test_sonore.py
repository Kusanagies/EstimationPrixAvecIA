import geopandas as gpd

# Si c'est un fichier physique sur votre ordinateur :
chemin_fichier = "chemin/vers/votre_fichier.gml" 
gdf = gpd.read_file(chemin_fichier)

# Si c'est une URL de service WFS (Web Feature Service) :
# (Remplacez l'URL par l'adresse du vrai service WFS, pas celle du XSD)
url_wfs = "http://serveur-carto.com/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=mon_calque&outputFormat=gml3"
gdf = gpd.read_file(url_wfs)

print(gdf.head())