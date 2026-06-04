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

# Immobilier (On récupère aussi le code_commune pour la fusion avec les commerces)
requete_dvf = """
SELECT code_commune, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, surface_reelle_bati
FROM valeurs_foncieres_idf
WHERE latitude IS NOT NULL 
  AND surface_reelle_bati > 9;
"""
maisons = pd.read_sql(requete_dvf, con=moteur)

# Transports
requete_transports = "SELECT stop_lat, stop_lon, type_station FROM donnees_transport_idf WHERE type_station IS NOT NULL;"
stations = pd.read_sql(requete_transports, con=moteur)

# Commerces
requete_commerces = "SELECT * FROM commerces_communes;"
commerces = pd.read_sql(requete_commerces, con=moteur)

# ==========================================
# 2. CRÉATION DU "SCORE COMMERCES"
# ==========================================
print("Calcul du dynamisme commercial...")

# On sélectionne les commerces qui ont un vrai impact sur la vie de quartier (vous pouvez modifier cette liste)
colonnes_vie_quotidienne = [
    'supermarche', 'superette', 'epicerie', 'boulangerie', 
    'boucherie_charcuterie', 'poissonnerie', 'pharmacie', 'librairie_papeterie_journaux'
]

# On s'assure que seules les colonnes existantes dans votre table sont utilisées pour éviter les erreurs
colonnes_existantes = [col for col in colonnes_vie_quotidienne if col in commerces.columns]

# On crée une nouvelle colonne qui est la SOMME de tous ces commerces par commune
commerces['total_commerces_proximite'] = commerces[colonnes_existantes].sum(axis=1)

# On allège la table des commerces avant la fusion
commerces_light = commerces[['departement_commune', 'population_2010', 'total_commerces_proximite']]

# ==========================================
# 3. CALCUL DES DISTANCES PAR TYPE DE TRANSPORT
# ==========================================
print("Calcul des distances géospatiales par type de station...")

RAYON_TERRE_METRES = 6371000
maisons_rad = np.deg2rad(maisons[['latitude', 'longitude']])

# On récupère la liste des types de stations uniques (ex: ['tram', 'bus', 'metro', 'rer'])
types_de_transport = stations['type_station'].unique()

# On boucle sur chaque type de transport !
for type_transport in types_de_transport:
    # 1. On isole uniquement les stations de ce type
    stations_du_type = stations[stations['type_station'] == type_transport]
    
    if len(stations_du_type) > 0:
        # 2. On crée l'arbre spatial uniquement pour ce type
        stations_rad = np.deg2rad(stations_du_type[['stop_lat', 'stop_lon']])
        arbre = BallTree(stations_rad, metric='haversine')
        
        # 3. On calcule la distance
        distances_rad, _ = arbre.query(maisons_rad, k=1)
        
        # 4. On crée une NOUVELLE colonne dynamique (ex: 'dist_metro_m', 'dist_bus_m')
        nom_colonne = f"dist_{type_transport}_m"
        maisons[nom_colonne] = distances_rad * RAYON_TERRE_METRES

# ==========================================
# 4. FUSION DES DONNÉES
# ==========================================
donnees_completes = pd.merge(
    maisons,
    commerces_light,
    left_on='code_commune',
    right_on='departement_commune',
    how='inner'
)

# Nettoyage des valeurs aberrantes (Outliers) pour ne pas fausser l'algorithme
# On exclut les prix irréalistes (ex: < 1000€/m2 ou > 25000€/m2 en IDF)
donnees_propres = donnees_completes[
    (donnees_completes['prix_m2'] >= 1000) & 
    (donnees_completes['prix_m2'] <= 25000)
]

# ==========================================
# 5. MATRICE DE CORRÉLATION ET AFFICHAGE
# ==========================================
print("Génération de la matrice de corrélation...")

# On sélectionne toutes les colonnes qui commencent par 'dist_' + nos autres variables clés
colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
colonnes_etude = ['prix_m2', 'surface_reelle_bati', 'total_commerces_proximite'] + colonnes_dist

donnees_finales = donnees_propres[colonnes_etude].dropna()

# Calcul de la matrice
matrice_corr = donnees_finales.corr()

# Configuration du graphique
plt.figure(figsize=(12, 9))
sns.heatmap(
    matrice_corr, 
    annot=True, 
    cmap='coolwarm', 
    vmin=-1, 
    vmax=1, 
    fmt=".2f", 
    linewidths=0.5,
    cbar_kws={'label': 'Coefficient de Corrélation de Pearson'}
)
plt.title("Impact des Transports et des Commerces sur le Prix de l'Immobilier (IDF)", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

print("Affichage du graphe...")
plt.show()