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
print("\n📥 DÉBUT DU TÉLÉCHARGEMENT DES DONNÉES (CIBLE : 49)...")
temps_total_debut = time.time()

moteur = create_engine("mysql+pymysql://root:1618@localhost:3306/EstimationIA")

# 1. IMMOBILIER (DVF) - Uniquement Hérault (49)
maisons = pd.read_sql("""
    SELECT code_commune, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, surface_reelle_bati
    FROM valeurs_foncieres
    WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
      AND LEFT(code_commune, 2) = '49';
""", con=moteur)

# 2. DPE - Uniquement Hérault (49)
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
      AND LEFT(code_insee_ban, 2) = '49'
    GROUP BY code_insee_ban;
""", con=moteur)

# 3. TRANSPORTS - Boîte géographique (Bounding Box) englobant 
# TRANSPORTS - Boîte géographique pour le Maine-et-Loire (49)
stations = pd.read_sql("""
    SELECT latitude, longitude FROM donnees_transport 
    WHERE latitude BETWEEN 46.9 AND 47.9 
      AND longitude BETWEEN -1.4 AND 0.2;
""", con=moteur)

# 4. MONUMENTS HISTORIQUES - Uniquement  (49)
monuments = pd.read_sql("""
    SELECT latitude, longitude FROM monuments_historiques 
    WHERE latitude IS NOT NULL 
      AND LEFT(code_insee, 2) = '49';
""", con=moteur)

# 5. HÔPITAUX - Uniquement Hérault (49)
hopitaux = pd.read_sql("""
    SELECT latitude, longitude FROM infrastructures_hopitaux 
    WHERE latitude IS NOT NULL 
      AND LEFT(code_postal, 2) = '49';
""", con=moteur)

# 6. UNIVERSITÉS - Uniquement Hérault (49
universites = pd.read_sql("""
    SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites 
    WHERE latitude IS NOT NULL 
      AND LEFT(code_insee, 2) = '49';
""", con=moteur)

print(f"✔️ Données extraites en {time.time() - temps_total_debut:.2f} secondes.")

# ==========================================
# 2. FUSION DES DONNÉES COMMUNALES
# ==========================================
print("🔗 Fusion des données DVF et DPE...")
donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')

# ==========================================
# 3. CALCULS SPATIAUX (BallTree)
# ==========================================
print("🌍 Calculs géospatiaux (Topologie de l'Hérault)...")
t_geo = time.time()
RAYON_TERRE_METRES = 6371000
maisons_rad = np.deg2rad(donnees[['latitude', 'longitude']])

def calculer_distance_min(df_points, nom_colonne):
    if len(df_points) > 0:
        points_rad = np.deg2rad(df_points.iloc[:, 0:2])
        arbre = BallTree(points_rad, metric='haversine')
        dist_rad, _ = arbre.query(maisons_rad, k=1)
        # Utilisation de .flatten() pour garantir le format de la colonne
        donnees[nom_colonne] = dist_rad.flatten() * RAYON_TERRE_METRES

# Remplacer l'ancien appel par celui-ci :
calculer_distance_min(stations[['latitude', 'longitude']], 'dist_transport_m')
calculer_distance_min(monuments, 'dist_monument_m')
calculer_distance_min(hopitaux, 'dist_hopital_m')

if len(universites) > 0:
    univ_rad = np.deg2rad(universites[['latitude', 'longitude']])
    arbre_univ = BallTree(univ_rad, metric='haversine')
    dist_rad, idx_univ = arbre_univ.query(maisons_rad, k=1)
    donnees['dist_universite_m'] = dist_rad.flatten() * RAYON_TERRE_METRES
    donnees['volume_etudiants_proche'] = universites.iloc[idx_univ.flatten()]['nombre_etudiants'].values

print(f"✔️ Calculs géospatiaux terminés en {time.time() - t_geo:.2f} secondes.")

# ==========================================
# 4. NETTOYAGE ET NORMALISATION (Scaling)
# ==========================================
print("🧹 Nettoyage et mise à l'échelle...")

# A. Valeurs manquantes et sécurisation des colonnes absentes
if 'volume_etudiants_proche' not in donnees.columns:
    # Si la colonne n'existe pas (ex: base SQL vide pour ce département), on la crée avec des 0
    donnees['volume_etudiants_proche'] = 0
else:
    # Si elle existe, on remplace juste les vides par 0
    donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)

colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
for col in colonnes_dpe:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

# B. Outliers (Adaptés au marché du Sud : 500€ à 10 000€ le m²)
donnees_propres = donnees[
    (donnees['prix_m2'] >= 500) & (donnees['prix_m2'] <= 10000) & 
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

# C. Transformation Logarithmique du Prix
donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])

# D. Min-Max Scaling (Toutes les distances entre 0 et 1)
colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
if colonnes_dist:
    donnees_propres[colonnes_dist] = MinMaxScaler().fit_transform(donnees_propres[colonnes_dist])

# E. Standardisation Z-Score (Surfaces et Volumes étudiants)
colonnes_standard = ['surface_reelle_bati', 'volume_etudiants_proche']
if colonnes_standard:
    donnees_propres[colonnes_standard] = StandardScaler().fit_transform(donnees_propres[colonnes_standard])

# ==========================================
# 5. MATRICE DE CORRÉLATION ET AFFICHAGE
# ==========================================
print("📈 Génération de la Matrice 49...")

# On assemble proprement la liste finale sans doublons
colonnes_finales = ['log_prix_m2'] + colonnes_dpe + colonnes_standard + colonnes_dist
matrice_corr = donnees_propres[colonnes_finales].corr()

plt.figure(figsize=(16, 12))
masque = np.triu(np.ones_like(matrice_corr, dtype=bool))

sns.heatmap(matrice_corr, mask=masque, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, annot_kws={"size": 9})

plt.title("Master Dataset (49) : Immobilier, DPE et Infrastructures", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

print("-" * 50)
print(f"🏁 PROCESSUS COMPLET TERMINÉ EN : {time.time() - temps_total_debut:.2f} secondes.")
print("-" * 50)

# ==========================================
# 6. EXPORT ET AFFICHAGE TEXTE DES RÉSULTATS
# ==========================================
print("\n" + "="*50)
print("📊 TOP DES CORRÉLATIONS AVEC LE PRIX AU M²")
print("="*50)

# On isole la colonne du prix, on supprime la corrélation avec elle-même (1.0) et on trie
correlations_prix = matrice_corr['log_prix_m2'].drop('log_prix_m2').sort_values(ascending=False)

print("\n📈 IMPACTS POSITIFS (Font monter le prix) :")
for index, valeur in correlations_prix[correlations_prix > 0].items():
    print(f"🔹 {index.ljust(25)} : {valeur:+.3f}")

print("\n📉 IMPACTS NÉGATIFS (Font baisser le prix) :")
for index, valeur in correlations_prix[correlations_prix < 0].items():
    print(f"🔻 {index.ljust(25)} : {valeur:+.3f}")

print("\n" + "="*50)

# OPTIONNEL : Si vous voulez sauvegarder toute la matrice dans un fichier Excel/CSV
matrice_corr.to_csv("/home/sylvain-huang/Documents/EstimationIA/resultats_correlation.csv", sep=";", decimal=",")
print("💾 Matrice complète sauvegardée dans 'resultats_correlation.csv'")

plt.show()