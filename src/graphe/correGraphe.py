import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# ==========================================
# 1. CONNEXION MYSQL
# ==========================================
USER = 'root'
PASSWORD = '1618'
HOTE = 'localhost'
PORT = '3306'
DB = 'EstimationIA'

chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"
moteur = create_engine(chaine_connexion)

print("📥 Téléchargement des données depuis les 6 tables SQL...")

# 1. IMMOBILIER (DVF)
requete_dvf = """
SELECT code_commune, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, surface_reelle_bati
FROM valeurs_foncieres_idf
WHERE latitude IS NOT NULL AND surface_reelle_bati > 9;
"""
maisons = pd.read_sql(requete_dvf, con=moteur)

# 2. DPE (Agrégé par commune avec conversion A=7, G=1)
requete_dpe = """
SELECT code_insee_ban, 
       AVG(CASE 
           WHEN etiquette_dpe = 'A' THEN 7 
           WHEN etiquette_dpe = 'B' THEN 6 
           WHEN etiquette_dpe = 'C' THEN 5 
           WHEN etiquette_dpe = 'D' THEN 4 
           WHEN etiquette_dpe = 'E' THEN 3 
           WHEN etiquette_dpe = 'F' THEN 2 
           WHEN etiquette_dpe = 'G' THEN 1 
           ELSE NULL END) AS score_dpe_moyen
FROM dpe_logements_france
WHERE etiquette_dpe IN ('A','B','C','D','E','F','G')
GROUP BY code_insee_ban;
"""
dpe = pd.read_sql(requete_dpe, con=moteur)

# 3. COMMERCES
requete_commerces = "SELECT * FROM commerces_communes;"
commerces = pd.read_sql(requete_commerces, con=moteur)

# 4. TRANSPORTS
requete_transports = "SELECT stop_lat, stop_lon, type_station FROM donnees_transport WHERE type_station IS NOT NULL;"
stations = pd.read_sql(requete_transports, con=moteur)

# 5. MONUMENTS HISTORIQUES
requete_monuments = "SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND longitude IS NOT NULL;"
monuments = pd.read_sql(requete_monuments, con=moteur)

# 6. HÔPITAUX
requete_hopitaux = "SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND longitude IS NOT NULL;"
hopitaux = pd.read_sql(requete_hopitaux, con=moteur)

# ==========================================
# 2. FUSION DES DONNÉES COMMUNALES (DVF + Commerces + DPE)
# ==========================================
print("🔗 Fusion des données communales (Immobilier, DPE, Commerces)...")

# On calcule le total des commerces
colonnes_commerces = ['supermarche', 'boulangerie', 'boucherie_charcuterie', 'pharmacie', 'librairie_papeterie_journaux']
colonnes_existantes = [col for col in colonnes_commerces if col in commerces.columns]
commerces['total_commerces'] = commerces[colonnes_existantes].sum(axis=1)

# Fusion DVF <-> Commerces
donnees = pd.merge(maisons, commerces[['departement_commune', 'total_commerces'] + colonnes_existantes], 
                   left_on='code_commune', right_on='departement_commune', how='inner')

# Fusion DVF <-> DPE
donnees = pd.merge(donnees, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')

# ==========================================
# 3. CALCULS DES DISTANCES SPATIALES (BallTree)
# ==========================================
print("🌍 Calcul géospatial des distances (Transports, Monuments, Hôpitaux)...")
RAYON_TERRE_METRES = 6371000
maisons_rad = np.deg2rad(donnees[['latitude', 'longitude']])

# --- A. Les Transports ---
for type_transport in stations['type_station'].unique():
    stations_du_type = stations[stations['type_station'] == type_transport]
    if len(stations_du_type) > 0:
        stations_rad = np.deg2rad(stations_du_type[['stop_lat', 'stop_lon']])
        arbre = BallTree(stations_rad, metric='haversine')
        distances_rad, _ = arbre.query(maisons_rad, k=1)
        donnees[f"dist_{type_transport}_m"] = distances_rad * RAYON_TERRE_METRES

# --- B. Les Monuments Historiques ---
if len(monuments) > 0:
    monuments_rad = np.deg2rad(monuments[['latitude', 'longitude']])
    arbre_monu = BallTree(monuments_rad, metric='haversine')
    distances_rad, _ = arbre_monu.query(maisons_rad, k=1)
    donnees['dist_monument_m'] = distances_rad * RAYON_TERRE_METRES

# --- C. Les Hôpitaux ---
if len(hopitaux) > 0:
    hopitaux_rad = np.deg2rad(hopitaux[['latitude', 'longitude']])
    arbre_hopi = BallTree(hopitaux_rad, metric='haversine')
    distances_rad, _ = arbre_hopi.query(maisons_rad, k=1)
    donnees['dist_hopital_m'] = distances_rad * RAYON_TERRE_METRES

# ==========================================
# 4. NETTOYAGE ET MATRICE DE CORRÉLATION
# ==========================================
print("📈 Génération de la matrice de corrélation finale...")

# On retire les prix extrêmes pour ne pas fausser les calculs
donnees_propres = donnees[(donnees['prix_m2'] >= 1000) & (donnees['prix_m2'] <= 25000)]

# On sélectionne toutes les colonnes numériques qui nous intéressent
colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
colonnes_finales = ['prix_m2', 'surface_reelle_bati', 'score_dpe_moyen', 'total_commerces'] + colonnes_existantes + colonnes_dist

# On calcule la matrice
matrice_corr = donnees_propres[colonnes_finales].dropna().corr()

# ==========================================
# 5. AFFICHAGE DU GRAPHIQUE
# ==========================================
plt.figure(figsize=(16, 12)) # Format XXL pour tout voir

# On crée un masque pour cacher la moitié supérieure du tableau (redondante)
masque = np.triu(np.ones_like(matrice_corr, dtype=bool))

sns.heatmap(
    matrice_corr, 
    mask=masque,
    annot=True, 
    cmap='RdYlGn', # Rouge pour corrélation négative, Vert pour positive
    vmin=-1, 
    vmax=1, 
    fmt=".2f", 
    linewidths=0.5,
    annot_kws={"size": 9}
)

plt.title("Master Dataset : Corrélation Prix au m² vs. Infrastructures & DPE", fontsize=18, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

print("✅ Affichage du graphe...")
plt.show()