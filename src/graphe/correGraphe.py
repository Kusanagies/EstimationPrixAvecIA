import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import BallTree
import seaborn as sns
import matplotlib.pyplot as plt
import time
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

print("📥 Téléchargement des données depuis les 7 tables SQL...")

# 1. IMMOBILIER (DVF)
# Modifier les requetes pour faire une recherche sur d'autre ville/région
requete_dvf = """
SELECT code_commune, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, surface_reelle_bati
FROM valeurs_foncieres
WHERE latitude IS NOT NULL 
  AND surface_reelle_bati > 9
  AND LEFT(code_commune, 2) IN ('75', '77', '78', '91', '92', '93', '94', '95');
"""
maisons = pd.read_sql(requete_dvf, con=moteur)

# 2. DPE (Agrégé par commune avec conversion A=7, G=1)
requete_dpe = """
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
  AND LEFT(code_insee_ban, 2) IN ('75', '77', '78', '91', '92', '93', '94', '95')
GROUP BY code_insee_ban;
"""
dpe = pd.read_sql(requete_dpe, con=moteur)

# 3. COMMERCES
requete_commerces = """
SELECT * FROM commerces_communes
WHERE LEFT(departement_commune, 2) IN ('75', '77', '78', '91', '92', '93', '94', '95');
"""
commerces = pd.read_sql(requete_commerces, con=moteur)

# 4. TRANSPORTS (Version simplifiée sans type_station)
requete_transports = "SELECT stop_lat, stop_lon FROM donnees_transport WHERE stop_lat IS NOT NULL;"
stations = pd.read_sql(requete_transports, con=moteur)

# 5. MONUMENTS HISTORIQUES
requete_monuments = "SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND longitude IS NOT NULL;"
monuments = pd.read_sql(requete_monuments, con=moteur)

# 6. HÔPITAUX
requete_hopitaux = "SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND longitude IS NOT NULL;"
hopitaux = pd.read_sql(requete_hopitaux, con=moteur)

# 7. UNIVERSITÉS
requete_univ = "SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;"
universites = pd.read_sql(requete_univ, con=moteur)

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

# --- A. Les Transports (Global) ---
if len(stations) > 0:
    stations_rad = np.deg2rad(stations[['stop_lat', 'stop_lon']])
    arbre_transport = BallTree(stations_rad, metric='haversine')
    distances_rad, _ = arbre_transport.query(maisons_rad, k=1)
    donnees["dist_transport_m"] = distances_rad * RAYON_TERRE_METRES
    
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

# --- D. Les Universités ---
if len(universites) > 0:
    univ_rad = np.deg2rad(universites[['latitude', 'longitude']])
    arbre_univ = BallTree(univ_rad, metric='haversine')
    
    # On cherche l'université la plus proche et on récupère la distance ET l'index
    distances_rad, index_univ = arbre_univ.query(maisons_rad, k=1)
    
    donnees['dist_universite_m'] = distances_rad * RAYON_TERRE_METRES
    
    # ASTUCE PRO : On récupère aussi la taille de cette université !
    # Une fac de 30 000 étudiants n'a pas le même impact sur le quartier qu'une annexe de 200 étudiants
    donnees['volume_etudiants_proche'] = universites.iloc[index_univ.flatten()]['nombre_etudiants'].values

# ==========================================
# 4. DATA CLEANSING ET NORMALISATION (Avec Profilage)
# ==========================================
print("\n🧹 DÉBUT DU TRAITEMENT DES DONNÉES...")
temps_total_debut = time.time()

# --- Étape A : GESTION DES VALEURS MANQUANTES ---
t_debut_etape = time.time()

colonnes_a_remplir_zero = ['total_commerces'] + colonnes_existantes
donnees[colonnes_a_remplir_zero] = donnees[colonnes_a_remplir_zero].fillna(0)

colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
for col in colonnes_dpe:
    donnees[col] = donnees[col].fillna(donnees[col].median())

t_fin_etape = time.time()
print(f"✔️ Étape A (Valeurs manquantes) terminée en : {t_fin_etape - t_debut_etape:.4f} sec")

# --- Étape B : FILTRAGE DES OUTLIERS ---
t_debut_etape = time.time()

donnees_propres = donnees[
    (donnees['prix_m2'] >= 1000) & (donnees['prix_m2'] <= 25000) & 
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

t_fin_etape = time.time()
print(f"✔️ Étape B (Filtrage Outliers) terminée en  : {t_fin_etape - t_debut_etape:.4f} sec")

# --- Étape C : TRANSFORMATION LOGARITHMIQUE ---
t_debut_etape = time.time()

donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])

t_fin_etape = time.time()
print(f"✔️ Étape C (Transformation Log) terminée en : {t_fin_etape - t_debut_etape:.4f} sec")

# --- Étape D : MIN-MAX SCALING ---
t_debut_etape = time.time()

colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
scaler_minmax = MinMaxScaler()
donnees_propres[colonnes_dist] = scaler_minmax.fit_transform(donnees_propres[colonnes_dist])

t_fin_etape = time.time()
print(f"✔️ Étape D (Min-Max Scaling) terminée en    : {t_fin_etape - t_debut_etape:.4f} sec")

# --- Étape E : STANDARDISATION Z-SCORE ---
t_debut_etape = time.time()

colonnes_a_standardiser = ['surface_reelle_bati', 'total_commerces']
scaler_standard = StandardScaler()
donnees_propres[colonnes_a_standardiser] = scaler_standard.fit_transform(donnees_propres[colonnes_a_standardiser])

t_fin_etape = time.time()
print(f"✔️ Étape E (Standardisation) terminée en    : {t_fin_etape - t_debut_etape:.4f} sec")

# --- Étape F : MATRICE DE CORRÉLATION ---
t_debut_etape = time.time()

colonnes_finales = ['log_prix_m2', 'surface_reelle_bati'] + colonnes_dpe + ['total_commerces'] + colonnes_dist 
matrice_corr = donnees_propres[colonnes_finales].corr()

t_fin_etape = time.time()
print(f"✔️ Étape F (Calcul Matrice) terminée en     : {t_fin_etape - t_debut_etape:.4f} sec")

# --- BILAN ---
temps_total_fin = time.time()
print("-" * 50)
print(f"🏁 TEMPS TOTAL DU PROCESSUS : {temps_total_fin - temps_total_debut:.4f} secondes.")
print("-" * 50)

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