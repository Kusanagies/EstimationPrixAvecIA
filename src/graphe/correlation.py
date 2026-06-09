import time
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sqlalchemy import create_engine

# ==========================================
# 0. CONNEXION INITIALE ET MENU INTERACTIF
# ==========================================
print("-" * 50)
print("INITIALISATION DU MOTEUR D'ESTIMATION IMMOBILIERE")
print("-" * 50)

# 1. Connexion precoce a la base de donnees pour le menu
try:
    moteur = create_engine("mysql+pymysql://root:1618@localhost:3306/EstimationIA")
    connexion_test = moteur.connect()
    connexion_test.close()
except Exception as e:
    print("Erreur de connexion a la base MySQL. Verifiez que le serveur est allume.")
    sys.exit()

# 2. Saisie du departement
departement = input("\nVeuillez saisir le numero du departement (ex: 34, 75, 17) : ").strip()

if len(departement) < 2:
    print("Erreur : Le format du departement est invalide.")
    sys.exit()

# 3. Interrogation des communes/arrondissements disponibles
print(f"Recherche des secteurs disponibles pour le {departement}...")
query_communes = f"""
    SELECT code_commune, MAX(nom_commune) as nom_commune, COUNT(*) as volume_ventes
    FROM valeurs_foncieres
    WHERE LEFT(code_commune, 2) = '{departement}'
      AND type_local IN ('Maison', 'Appartement')
    GROUP BY code_commune
    ORDER BY volume_ventes DESC
    LIMIT 15;
"""
df_communes = pd.read_sql(query_communes, con=moteur)

if len(df_communes) == 0:
    print(f"Erreur : Aucune donnee immobiliere trouvee pour le departement {departement}.")
    sys.exit()

# 4. Affichage du Top 15 pour aider l'utilisateur
print(f"\nVoici les secteurs avec le plus de donnees dans le {departement} :")
for index, row in df_communes.iterrows():
    # Formatage propre pour que les colonnes soient alignees dans le terminal
    nom = str(row['nom_commune']).ljust(25)[:25] 
    print(f"  - {row['code_commune']} : {nom} ({row['volume_ventes']} ventes)")

print("  - ... (et autres communes)")

# 5. Saisie du choix local
choix_local = input("\nSaisissez le code INSEE d'un secteur precis (ou tapez 'TOUS' pour le departement complet) : ").strip().upper()

# 6. Configuration des filtres SQL dynamiques
if choix_local == 'TOUS':
    filtre_dvf = f"LEFT(code_commune, 2) = '{departement}'"
    filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
    nom_zone = f"Departement {departement}"
else:
    filtre_dvf = f"code_commune = '{choix_local}'"
    filtre_dpe = f"code_insee_ban = '{choix_local}'"
    nom_zone = f"Secteur {choix_local}"

print(f"\nLancement de l'apprentissage pour : {nom_zone}")
print("-" * 50)
temps_total_debut = time.time()

# ==========================================
# 1. TELECHARGEMENT DES DONNEES FILTREES
# ==========================================
print("Etape 1/6 : Extraction des donnees depuis SQL...")

# 1. Immobilier (DVF) - Filtrage dynamique
maisons = pd.read_sql(f"""
    SELECT code_commune, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, surface_reelle_bati, type_local
    FROM valeurs_foncieres
    WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
      AND {filtre_dvf}
      AND type_local IN ('Maison', 'Appartement');
""", con=moteur)

if len(maisons) == 0:
    print(f"Erreur : Le code INSEE {choix_local} n'existe pas ou ne contient aucune donnee.")
    sys.exit()

# 2. DPE - Filtrage dynamique
dpe = pd.read_sql(f"""
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
      AND {filtre_dpe}
    GROUP BY code_insee_ban;
""", con=moteur)

# 3. Infrastructures (On charge large pour le departement, le BallTree fera le tri local)
stations = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur)
monuments = pd.read_sql(f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{departement}';", con=moteur)
hopitaux = pd.read_sql(f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal, 2) = '{departement}';", con=moteur)
universites = pd.read_sql(f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{departement}';", con=moteur)

# ==========================================
# 2. FUSION DES DONNEES
# ==========================================
print("Etape 2/6 : Fusion des tables...")
donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')

# ==========================================
# 3. CALCULS SPATIAUX (BallTree)
# ==========================================
print("Etape 3/6 : Calcul des matrices de distance geospatiales...")
RAYON_TERRE_METRES = 6371000
maisons_rad = np.deg2rad(donnees[['latitude', 'longitude']])

def calculer_distance_min(df_points, nom_colonne):
    if len(df_points) > 0:
        points_rad = np.deg2rad(df_points.iloc[:, 0:2])
        arbre = BallTree(points_rad, metric='haversine')
        dist_rad, _ = arbre.query(maisons_rad, k=1)
        donnees[nom_colonne] = dist_rad.flatten() * RAYON_TERRE_METRES
    else:
        donnees[nom_colonne] = 999999

calculer_distance_min(stations, 'dist_transport_m')
calculer_distance_min(monuments, 'dist_monument_m')
calculer_distance_min(hopitaux, 'dist_hopital_m')

if len(universites) > 0:
    univ_rad = np.deg2rad(universites[['latitude', 'longitude']])
    arbre_univ = BallTree(univ_rad, metric='haversine')
    dist_rad, idx_univ = arbre_univ.query(maisons_rad, k=1)
    donnees['dist_universite_m'] = dist_rad.flatten() * RAYON_TERRE_METRES
    donnees['volume_etudiants_proche'] = universites.iloc[idx_univ.flatten()]['nombre_etudiants'].values
else:
    donnees['dist_universite_m'] = 999999
    donnees['volume_etudiants_proche'] = 0

# ==========================================
# 4. NETTOYAGE ET FEATURE ENGINEERING
# ==========================================
print("Etape 4/6 : Nettoyage et normalisation du dataset...")

colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
for col in colonnes_dpe:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)

# Filtrage securise pour eviter les valeurs extremes
donnees_propres = donnees[
    (donnees['prix_m2'] >= 500) & (donnees['prix_m2'] <= 25000) & 
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])
donnees_propres['est_maison'] = (donnees_propres['type_local'] == 'Maison').astype(int)

# Normalisation (Gestion des erreurs si variance = 0 dans une petite commune)
colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
if colonnes_dist:
    try:
        donnees_propres[colonnes_dist] = MinMaxScaler().fit_transform(donnees_propres[colonnes_dist])
    except ValueError:
        pass

colonnes_standard = ['surface_reelle_bati']
if donnees_propres['volume_etudiants_proche'].nunique() > 1:
    colonnes_standard.append('volume_etudiants_proche')

try:
    donnees_propres[colonnes_standard] = StandardScaler().fit_transform(donnees_propres[colonnes_standard])
except ValueError:
    pass

# ==========================================
# 5. PREPARATION DES MATRICES POUR L'IA
# ==========================================
print("Etape 5/6 : Separation des donnees (Train/Test Split)...")

features = ['est_maison'] + colonnes_dpe + colonnes_standard + colonnes_dist
X = donnees_propres[features]
y = donnees_propres['log_prix_m2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 6. ENTRAINEMENT ET EVALUATION DE XGBOOST
# ==========================================
print("Etape 6/6 : Entrainement de l'algorithme XGBoost...")

modele_xgb = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

modele_xgb.fit(X_train, y_train)

predictions_log = modele_xgb.predict(X_test)
prix_reels_euros = np.exp(y_test)
prix_predits_euros = np.exp(predictions_log)

mae = mean_absolute_error(prix_reels_euros, prix_predits_euros)
r2 = r2_score(y_test, predictions_log)

# Affichage du rapport
print("\n" + "="*50)
print(f"RAPPORT DE PERFORMANCE XGBOOST - {nom_zone.upper()}")
print("="*50)
print(f"Nombre de logements pour l'apprentissage : {len(X_train)}")
print(f"Nombre de logements pour la validation   : {len(X_test)}")
print("-" * 50)
print(f"Coefficient de determination (R2)        : {r2 * 100:.2f} %")
print(f"Erreur absolue moyenne (MAE)             : {mae:.2f} EUR / m2")
print("="*50)
print(f"Temps de traitement global : {time.time() - temps_total_debut:.2f} secondes.\n")