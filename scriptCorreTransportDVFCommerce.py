import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import seaborn as sns
import matplotlib.pyplot as plt

##### Chargement et nettoyage de base #####

maisons = pd.read_csv('./data/dvf.csv',dtype={'code_departement': str,'code_commune':str})
stations = pd.read_csv('./data/gtfs_donnees.csv')
commerces = pd.read_csv('commerces_cimmunes.csv',sep=';',dtype={'departement_commune':str})

# On supprime les lignes sans coordonnées GPS 
maisons = maisons.dropna(subset=['latitude','longitude'])
stations = stations.dropna(subset=['stop_lat','stop_lon'])

##### Filtrage pour IDF #####
# Filtrer l'immobilier (Départements : 75, 77, 78, 91, 92, 93, 94, 95)
departements_idf = ['75','77','78','91','92','93','94','95']
maisons_idf = maisons[maisons['code_departement'].isint(departements_idf)].copy()

# Filtrer les stations géographiquement (Cadre GPS approximatif de l'IDF)
# Latitudes entre 48.1 et 49.3 / Longitudes entre 1.4 et 3.6
stations_idf = stations[
    (stations['stop_lat'] >= 48.1) & (stations['stop_lat'] <= 49.3) & 
    (stations['stop_lat'] >= 1.4) & (stations['stop_lon'] <= 3.6)
].copy()

##### Calcul de la distance aux transports (BallTree)

# Rayon de la terre en mètre
RAYON_TERRE = 6371000

stations_rad = np.deg2rad(stations_idf[['stop_lat','stop_lon']])
arbre_stations = BallTree(stations_rad,metric='haversine')

maisons_rad = np.deg2rad(maisons_idf[['latitude','longitude']])
distances_rad, _ = arbre_stations.query(maisons_rad,k=1)

maisons_idf['distance_station_m'] = distances_rad * RAYON_TERRE

##### Fusion des données (Immo et Commerces) #####

# On fusionne la table des maisons avec celle des commerces en utilisant le code INSEE de la commune
# DVF l'appelle 'code_commune', la table commerce l'appelle 'departement_commune
donnees_completes = pd.merge(
    maisons_idf,
    commerces,
    left_on='code_commune',
    right_on='departement_commune',
    how='inner'
)

##### Préparation des variables et corrélation #####
# Calcule du prix au m²
donnees_completes['prix_m2'] = donnees_completes['valeur_fonciere']/donnees_completes['surface_reelle_bati']

colonnes_etude = [
    'prix_m2','distance_station_m','surface_reelle_bati','boulangerie','supermarche','superette','boucherie_charcuterie',
    'population_2010'
]

# On filtre la donnée finale et on gère les valeurs infinies ou manquqntes
donnees_finales = donnees_completes[colonnes_etude].replace([np.inf,-np.inf],np.nan).dropna()
matrice_corr = donnees_finales.corr()

plt.figure(figsize=(10,8))
sns.heatmap(matrice_corr,annot=True,cmap='coolwarm',vmin=-1,vmax=1,fmt=".2f",linewidths=0.5)
plt.title('Corrélation IDF : Immobilier, Transports et commerces', fontsize=14)
plt.xticks(rotations=45,ha ='rigth')
plt.tight_layout()
plt.show
