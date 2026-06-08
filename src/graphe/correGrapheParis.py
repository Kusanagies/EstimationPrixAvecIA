import time
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# ==========================================
# 1. CONNEXION MYSQL ET TÉLÉCHARGEMENT
# ==========================================
print("\n📥 DÉBUT DU TÉLÉCHARGEMENT DES DONNÉES (CIBLE : PARIS 75)...")
temps_total_debut = time.time()

moteur = create_engine("mysql+pymysql://root:1618@localhost:3306/EstimationIA")

# 1. IMMOBILIER - Uniquement Paris (75)
maisons = pd.read_sql("""
    SELECT code_commune, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, surface_reelle_bati
    FROM valeurs_foncieres
    WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
      AND LEFT(code_commune, 2) = '75'; -- Strictement Paris
""", con=moteur)

# 2. DPE - Uniquement Paris (75)
dpe = pd.read_sql("""
    SELECT code_insee_ban, 
           (SUM(CASE WHEN etiquette_dpe = 'A' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_A,
           (SUM(CASE WHEN etiquette_dpe = 'B' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_B,
           (SUM(CASE WHEN etiquette_dpe = 'C' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_C,
           (SUM(CASE WHEN etiquette_dpe = 'D' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_D,
           (SUM(CASE WHEN etiquette_dpe = 'E' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_E,
           (SUM(CASE WHEN etiquette_dpe = 'F' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_F,
           (SUM(CASE WHEN etiquette_dpe = 'G' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_G
    FROM dpe_logements_france
    WHERE etiquette_dpe IN ('A','B','C','D','E','F','G')
      AND LEFT(code_insee_ban, 2) = '75' -- Strictement Paris
    GROUP BY code_insee_ban;
""", con=moteur)

# 3. COMMERCES - Uniquement Paris (75)
commerces = pd.read_sql("""
    SELECT * FROM commerces_communes 
    WHERE LEFT(departement_commune, 2) = '75'; -- Strictement Paris
""", con=moteur)

# 4. TRANSPORTS - On filtre par une boîte géographique (Bounding Box) pour ne garder que Paris et optimiser le calcul
stations = pd.read_sql("""
    SELECT stop_lat, stop_lon FROM donnees_transport 
    WHERE stop_lat BETWEEN 48.81 AND 48.91 
      AND stop_lon BETWEEN 2.22 AND 2.47;
""", con=moteur)

# 5. MONUMENTS HISTORIQUES - Uniquement Paris (75)
monuments = pd.read_sql("""
    SELECT latitude, longitude FROM monuments_historiques 
    WHERE latitude IS NOT NULL 
      AND LEFT(code_insee, 2) = '75';
""", con=moteur)

# 6. HÔPITAUX - Uniquement Paris (75)
hopitaux = pd.read_sql("""
    SELECT latitude, longitude FROM infrastructures_hopitaux 
    WHERE latitude IS NOT NULL 
      AND LEFT(code_postal, 2) = '75';
""", con=moteur)

# 7. UNIVERSITÉS - Uniquement Paris (75)
universites = pd.read_sql("""
    SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites 
    WHERE latitude IS NOT NULL 
      AND LEFT(code_insee, 2) = '75';
""", con=moteur)

# 8. ANOMALIES (DansMaRue - Déjà 100% Paris)
anomalies = pd.read_sql("SELECT latitude, longitude FROM urbanisme_anomalies_paris WHERE latitude IS NOT NULL;", con=moteur)

print(f"✔️ Données Parisiennes extraites en {time.time() - temps_total_debut:.2f} secondes.")

# ==========================================
# 2. FUSION DES DONNÉES COMMUNALES
# ==========================================
print("🔗 Fusion des données des arrondissements de Paris...")
colonnes_commerces = ['supermarche', 'boulangerie', 'boucherie_charcuterie', 'pharmacie', 'librairie_papeterie_journaux']
colonnes_existantes = [col for col in colonnes_commerces if col in commerces.columns]
commerces['total_commerces'] = commerces[colonnes_existantes].sum(axis=1)

donnees = pd.merge(maisons, commerces[['departement_commune', 'total_commerces'] + colonnes_existantes], left_on='code_commune', right_on='departement_commune', how='inner')
donnees = pd.merge(donnees, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')

# ==========================================
# 3. CALCULS SPATIAUX (BallTree)
# ==========================================
print("🌍 Calculs géospatiaux sur la topologie de Paris...")
t_geo = time.time()
RAYON_TERRE_METRES = 6371000
maisons_rad = np.deg2rad(donnees[['latitude', 'longitude']])

def calculer_distance_min(df_points, nom_colonne):
    if len(df_points) > 0:
        points_rad = np.deg2rad(df_points.iloc[:, 0:2])
        arbre = BallTree(points_rad, metric='haversine')
        dist_rad, _ = arbre.query(maisons_rad, k=1)
        donnees[nom_colonne] = dist_rad * RAYON_TERRE_METRES

# Calcul des distances intra-muros
calculer_distance_min(stations, 'dist_transport_m')
calculer_distance_min(monuments, 'dist_monument_m')
calculer_distance_min(hopitaux, 'dist_hopital_m')

# Distance Université
if len(universites) > 0:
    univ_rad = np.deg2rad(universites[['latitude', 'longitude']])
    arbre_univ = BallTree(univ_rad, metric='haversine')
    dist_rad, idx_univ = arbre_univ.query(maisons_rad, k=1)
    donnees['dist_universite_m'] = dist_rad * RAYON_TERRE_METRES
    donnees['volume_etudiants_proche'] = universites.iloc[idx_univ.flatten()]['nombre_etudiants'].values

# Densité des anomalies à 100 mètres à Paris
if len(anomalies) > 0:
    anom_rad = np.deg2rad(anomalies[['latitude', 'longitude']])
    arbre_dmr = BallTree(anom_rad, metric='haversine')
    rayon_100m_rad = 100 / RAYON_TERRE_METRES
    voisins = arbre_dmr.query_radius(maisons_rad, r=rayon_100m_rad)
    donnees['densite_anomalies_100m'] = [len(v) for v in voisins]

print(f"✔️ Calculs géospatiaux parisiens terminés en {time.time() - t_geo:.2f} secondes.")

# ==========================================
# 4. NETTOYAGE ET NORMALISATION
# ==========================================
print("🧹 Nettoyage et mise à l'échelle (Scaling)...")

donnees['total_commerces'] = donnees.get('total_commerces', 0).fillna(0)
donnees['volume_etudiants_proche'] = donnees.get('volume_etudiants_proche', 0).fillna(0)
donnees['densite_anomalies_100m'] = donnees.get('densite_anomalies_100m', 0).fillna(0)

colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
for col in colonnes_dpe:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

# Outliers adaptés au marché parisien (Prix max relevé à 25000€/m²)
donnees_propres = donnees[
    (donnees['prix_m2'] >= 4000) & (donnees['prix_m2'] <= 25000) & 
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

# Transformations mathématiques
donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])

colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
if colonnes_dist:
    donnees_propres[colonnes_dist] = MinMaxScaler().fit_transform(donnees_propres[colonnes_dist])

colonnes_standard = ['surface_reelle_bati', 'total_commerces', 'volume_etudiants_proche', 'densite_anomalies_100m']
donnees_propres[colonnes_standard] = StandardScaler().fit_transform(donnees_propres[colonnes_standard])

# ==========================================
# 5. MATRICE DE CORRÉLATION ET AFFICHAGE
# ==========================================
print("📈 Génération de la Matrice Parisienne...")

colonnes_finales = ['log_prix_m2', 'surface_reelle_bati'] + colonnes_dpe + colonnes_standard + colonnes_dist
matrice_corr = donnees_propres[colonnes_finales].corr()

plt.figure(figsize=(18, 14))
masque = np.triu(np.ones_like(matrice_corr, dtype=bool))

sns.heatmap(matrice_corr, mask=masque, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, annot_kws={"size": 8})

plt.title("Master Dataset Paris (75) : Corrélations Prix, DPE et Incivilités Urbaines", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

print("-" * 50)
print(f"🏁 PROCESSUS COMPLET PARIS TERMINÉ EN : {time.time() - temps_total_debut:.2f} secondes.")
print("-" * 50)

plt.show()