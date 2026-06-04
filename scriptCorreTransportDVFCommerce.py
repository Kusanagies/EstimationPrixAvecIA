import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

##### Param #####

USER = 'root'
PASSWORD = '1618'
HOTE = 'localhost'
PORT = '3306'
DB = 'EstimationIA'

##### Creation du moteur #####
# On construit la chaine de connexion au format attendu par SQLAlchemy
chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"

# On crée le moteur
moteur = create_engine(chaine_connexion)

print("Connexion à Mysql")

requete_dvf = "SELECT * FROM  valeurs_foncieres_idf;"
maisons_idf = pd.read_sql(requete_dvf,con=moteur)

requete_transports = "SELECT * FROM donnees_transport_idf;"
stations_idf = pd.read_sql(requete_transports,con=moteur)

requete_commerces = "SELECT * FROM commerces_communes;"
commerces = pd.read_sql(requete_commerces,con=moteur)

print(f"Succès ! {len(maisons_idf)} maisons, {len(stations_idf)} stations et {len(commerces)} communes chargées.")

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
plt.xticks(rotation=45,ha ='right')
plt.tight_layout()
print("Affichage du graphe")
plt.show()
