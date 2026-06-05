import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# ==========================================
# 1. CONNEXION MYSQL ET RÉCUPÉRATION
# ==========================================
USER = 'root'
PASSWORD = '1618'
HOTE = 'localhost'
PORT = '3306'
DB = 'EstimationIA'

chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"
moteur = create_engine(chaine_connexion)

print("Téléchargement des données en cours...")

requete_dvf = """
SELECT code_commune, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, surface_reelle_bati
FROM valeurs_foncieres_idf
WHERE latitude IS NOT NULL AND surface_reelle_bati > 9;
"""
maisons = pd.read_sql(requete_dvf, con=moteur)

requete_transports = "SELECT stop_lat, stop_lon, type_station FROM donnees_transport_idf WHERE type_station IS NOT NULL;"
stations = pd.read_sql(requete_transports, con=moteur)

requete_commerces = "SELECT * FROM commerces_communes;"
commerces = pd.read_sql(requete_commerces, con=moteur)

# ==========================================
# 2. CRÉATION DU SCORE GLOBAL + CONSERVATION DU DÉTAIL
# ==========================================
print("Analyse du dynamisme commercial...")

colonnes_vie_quotidienne = [
    'supermarche', 'superette', 'epicerie', 'boulangerie', 
    'boucherie_charcuterie', 'poissonnerie', 'pharmacie', 'librairie_papeterie_journaux'
]

colonnes_existantes = [col for col in colonnes_vie_quotidienne if col in commerces.columns]

# 1. On calcule le total
commerces['total_commerces_proximite'] = commerces[colonnes_existantes].sum(axis=1)

# 2. LA MODIFICATION EST ICI : On garde la commune, la population, le total ET le détail de chaque commerce
colonnes_a_garder = ['departement_commune', 'population_2010', 'total_commerces_proximite'] + colonnes_existantes
commerces_detail = commerces[colonnes_a_garder]

# ==========================================
# 3. CALCUL DES DISTANCES PAR TYPE DE TRANSPORT
# ==========================================
print("Calcul des distances géospatiales par type de station...")
RAYON_TERRE_METRES = 6371000
maisons_rad = np.deg2rad(maisons[['latitude', 'longitude']])
types_de_transport = stations['type_station'].unique()

for type_transport in types_de_transport:
    stations_du_type = stations[stations['type_station'] == type_transport]
    
    if len(stations_du_type) > 0:
        stations_rad = np.deg2rad(stations_du_type[['stop_lat', 'stop_lon']])
        arbre = BallTree(stations_rad, metric='haversine')
        distances_rad, _ = arbre.query(maisons_rad, k=1)
        
        nom_colonne = f"dist_{type_transport}_m"
        maisons[nom_colonne] = distances_rad * RAYON_TERRE_METRES

# ==========================================
# 4. FUSION DES DONNÉES
# ==========================================
donnees_completes = pd.merge(
    maisons,
    commerces_detail, # On utilise notre nouvelle table détaillée
    left_on='code_commune',
    right_on='departement_commune',
    how='inner'
)

# Filtre anti-outliers
donnees_propres = donnees_completes[
    (donnees_completes['prix_m2'] >= 1000) & 
    (donnees_completes['prix_m2'] <= 25000)
]

# ==========================================
# 5. MATRICE DE CORRÉLATION (VERSION XL)
# ==========================================
print("Génération de la matrice de corrélation détaillée...")

colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]

# On intègre TOUT : le prix, la surface, le total, le détail des commerces, et les transports
colonnes_etude = ['prix_m2', 'surface_reelle_bati', 'total_commerces_proximite'] + colonnes_existantes + colonnes_dist

donnees_finales = donnees_propres[colonnes_etude].dropna()
matrice_corr = donnees_finales.corr()

# Graphique plus grand (14x11) pour accommoder toutes les nouvelles lignes/colonnes
plt.figure(figsize=(14, 11))
sns.heatmap(
    matrice_corr, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1, 
    fmt=".2f", 
    linewidths=0.5,
    cbar_kws={'label': 'Coefficient de Corrélation de Pearson'},
    annot_kws={"size": 8} # On réduit un peu la taille des chiffres pour ne pas surcharger les cases
)

plt.title("Corrélation Détaillée : Immobilier vs Transports vs Types de Commerces", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

print("Affichage du graphe...")
plt.show()