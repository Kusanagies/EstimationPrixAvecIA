import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import seaborn as sns
import matplotlib.pyplot as plt

##### UN RALENTI BEAUCOUP SUR MON PC PORTABLE REDUIRE LES LIGNES A LIRE
##### Chargement des données #####


maisons = pd.DataFrame(
    pd.read_csv('./data/dvf.csv',nrows = 1000)
)

stations = pd.DataFrame(
    pd.read_csv('./data/gtfs-stops-france-export-2026-01-13.csv',nrows = 1000)
)
# On supprime toutes les ventes qui n'ont pas de lat et lon
maisons = maisons.dropna(subset=['latitude', 'longitude']).copy()

# On supprime toutes les stations qui n'ont pas de lat et lon
stations = stations.dropna(subset=['stop_lat', 'stop_lon']).copy()
##### Calcul des distances #####

# Rayon de la terre en mètre
RAYON_TERRE = 6371000

# On convertit les latitudes et les longitudes des stations et de degrée en radiant
stations_rad = np.deg2rad(stations[['stop_lat','stop_lon']])

# On crée l'arbre spatial avec la métrique "haversine" (qui prend en compte la courbure de la terre)
arbre_stations = BallTree(stations_rad,metric = 'haversine')

# On convertit les coordonnées des maisons en radians
maisons_rad = np.deg2rad(maisons[['latitude','longitude']])

# On interroge l'arbre pour qu'il donne la station la plus proche (k=1) pour chaque maison
# distance_rad contiendra la distance en rad, indices contiendra la ligne de la station correspondante
distances_rad, indices = arbre_stations.query(maisons_rad,k=1)

# On convertit la distance en mètre et on l'ajoute au tableau de maisons
maisons['distance_station_m'] = distances_rad * RAYON_TERRE

##### Création des variables pour la corrélation #####

# Calcul du prix au mètre carré
maisons['prix_m2'] = maisons['valeur_fonciere']/maisons['surface_reelle_bati']

# On isole uniquement les colonnes mathématiquement pertinentes
donnees_etude = maisons[['prix_m2','distance_station_m','surface_reelle_bati']]

print("Aperçu des données pour l'analyse : ")
print(donnees_etude.head())

##### Affichage de la matrice de corrélation #####

matrice_corr = donnees_etude.corr()

plt.figure(figsize=(6,5))
sns.heatmap(matrice_corr, annot=True, cmap='coolwarm',vmin=-1,vmax=1,fmt=".2f")
plt.title('Corrélation : Prix de l\'immobilier par rapport aux transports')
plt.tight_layout()
plt.show()
