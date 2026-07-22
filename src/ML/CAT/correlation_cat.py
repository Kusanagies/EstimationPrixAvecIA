import time
import sys
import shap
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sqlalchemy import create_engine
import os
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import geopandas as gpd

# Enlever si on veut reactiver Optuna 
""" 
import optuna
def perte_pinball(y_vrai, y_pred, alpha=0.5):
    """""" Perte pinball (quantile loss). Pour alpha=0.5, c'est l'erreur absolue / 2.""""""
    erreur = y_vrai - y_pred
    return np.mean(np.maximum(alpha * erreur, (alpha - 1) * erreur))
def tuner_hyperparametres(X_train, y_train, n_essais=100):
    # Lance l'optimisation Optuna sur le modele median CatBoost.
    # Retourne le dictionnaire des meilleurs hyperparametres.
    # Sous-split train / validation pour evaluer chaque essai
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    def objectif(trial):
        # Espace de recherche : les hyperparametres CatBoost a explorer
        params = {
            'loss_function': 'Quantile:alpha=0.5',
            'iterations': 4000,             # plafond eleve, l'early stopping coupe
            'random_seed': 42,
            'verbose': False,
            'early_stopping_rounds': 50,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
        }
        modele = CatBoostRegressor(**params)
        modele.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

        # On evalue la perte pinball sur la validation (metrique adaptee au quantile)
        pred_val = modele.predict(X_val)
        return perte_pinball(y_val.values, pred_val, alpha=0.5)

    # 'minimize' car on minimise la perte pinball
    etude = optuna.create_study(direction='minimize')
    etude.optimize(objectif, n_trials=n_essais, show_progress_bar=True)
    print("\n" + "=" * 50)
    print("MEILLEURS HYPERPARAMETRES TROUVES (CatBoost)")
    print("=" * 50)
    for nom, val in etude.best_params.items():
        print(f"  {nom:22s} : {val}")
    print(f"\nMeilleure perte pinball : {etude.best_value:.5f}")
    print("=" * 50)
    return etude.best_params

# ==========================================
# EXEMPLE D'UTILISATION
# ==========================================
# Dans ton pipeline, une fois X_train / y_train prets :
#
#   meilleurs = tuner_hyperparametres(X_train, y_train, n_essais=100)
#
# Puis tu entraines les 3 quantiles avec ces hyperparametres,
# en changeant seulement alpha dans loss_function :
#
#   quantiles = {'bas': 0.025, 'median': 0.50, 'haut': 0.975}
#   modeles_q = {}
#   for nom_q, alpha in quantiles.items():
#       m = CatBoostRegressor(
#           loss_function=f'Quantile:alpha={alpha}',
#           iterations=4000, random_seed=42, verbose=False,
#           early_stopping_rounds=50,
#           **meilleurs   # les hyperparametres tunes sur le median
#       )
#       m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
#       modeles_q[nom_q] = m
"""

# ==========================================
# 0. CONNEXION INITIALE ET MENU INTERACTIF
# ==========================================
print("-" * 50)
print("INITIALISATION DU MOTEUR D'ESTIMATION IMMOBILIERE (EVALUATION - CATBOOST)")
print("-" * 50)

CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"
gdf_littoral = gpd.read_file(CHEMIN_GPKG)

RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    connexion_test = moteur.connect()
    connexion_test.close()
except KeyError:
    print("Erreur : variable DB_PASS introuvable. Verifiez votre fichier .env à la racine")
    sys.exit()
except Exception as e:
    print("Erreur de connexion a la base MySQL. Verifiez que le serveur est allume.")
    sys.exit()

# ==========================================
# 0bis. CHOIX INTERACTIF DES FEATURES
# ==========================================
def demander(question, defaut=True):
    """Pose une question oui/non. Entree vide = defaut. Redemande si invalide."""
    ind = "O/n" if defaut else "o/N"
    while True:
        r = input(f"  {question} ({ind}) : ").strip().lower()
        if r == "":
            return defaut
        if r.startswith('o'):
            return True
        if r.startswith('n'):
            return False
        print("    Tapez 'o' (oui) ou 'n' (non).")

print("\n" + "=" * 50)
print("CHOIX DES FEATURES POUR L'ENTRAINEMENT")
print("(Entree = valeur par defaut en majuscule)")
print("=" * 50)

FA = {}
print("\n--- Caracteristiques du bien ---")
FA['geo_base']    = demander("Latitude / longitude ?", True)
FA['surface']     = demander("Surface (+ log, par piece) ?", True)
FA['pieces']      = demander("Nombre de pieces ?", True)
FA['terrain']     = demander("Terrain (surface, log, a_terrain) ?", True)
FA['date']        = demander("Date (annee, mois de vente) ?", True)
print("\n--- Enrichissements communaux ---")
FA['dpe']         = demander("Profil DPE ?", True)
FA['chauffage']   = demander("Profil chauffage ?", True)
FA['revenus']     = demander("Revenus / Gini / minima ?", True)
print("\n--- Distances ---")
FA['dist_transport']  = demander("Distance gare ?", True)
FA['dist_monument']   = demander("Distance monument ?", True)
FA['dist_hopital']    = demander("Distance hopital ?", True)
FA['dist_universite'] = demander("Distance universite (+ etudiants) ?", True)
FA['dist_littoral']   = demander("Distance mer/lac/estuaire ?", True)
print("\n--- Features spatiales locales ---")
FA['voisins']     = demander("Prix des voisins ?", True)
FA['densite']     = demander("Densite ventes 1km ?", True)
FA['section']     = demander("Prix par section ?", True)
print("\n--- Features experimentales ---")
FA['potentiel_urbain'] = demander("Potentiel urbain ?", True)
FA['chomage']          = demander("Taux de chomage departemental ?", False)
FA['taux_credit']      = demander("Taux de credit immobilier ?", False)
FA['taux_inflation']   = demander("Taux d'inflation ?", False)
FA['pib']              = demander("PIB national ?", False)

departement = input("\nVeuillez saisir le numero du departement (ex: 34, 75) ou 'FRANCE' : ").strip().upper()

if len(departement) < 2:
    print("Erreur : Le format de saisie est invalide.")
    sys.exit()

print(f"Recherche des secteurs disponibles pour : {departement}...")

if departement == 'FRANCE':
    condition_dep = "1=1"
else:
    condition_dep = f"code_departement = '{departement}'"

query_communes = f"""
    SELECT code_commune, MAX(nom_commune) as nom_commune, COUNT(*) as volume_ventes
    FROM valeurs_foncieres
    WHERE {condition_dep}
      AND type_local IN ('Maison', 'Appartement')
    GROUP BY code_commune
    ORDER BY volume_ventes DESC
    LIMIT 15;
"""
df_communes = pd.read_sql(query_communes, con=moteur)

if len(df_communes) == 0:
    print(f"Erreur : Aucune donnee trouvee pour le secteur {departement}.")
    sys.exit()

print(f"\nVoici les secteurs avec le plus de donnees pour {departement} :")
for index, row in df_communes.iterrows():
    nom = str(row['nom_commune']).ljust(25)[:25]
    print(f"  - {row['code_commune']} : {nom} ({row['volume_ventes']} ventes)")

print("  - ... (et autres communes)")

choix_local = input("\nSaisissez le code INSEE d'un secteur precis (ou tapez 'TOUS' pour le choix initial complet) : ").strip().upper()

if departement == 'FRANCE':
    if choix_local == 'TOUS':
        filtre_dvf = "1=1"
        filtre_dpe = "1=1"
        dep_infra = "FRANCE"
        nom_zone = "France Entiere"
    else:
        filtre_dvf = f"code_commune = '{choix_local}'"
        filtre_dpe = f"code_insee_ban = '{choix_local}'"
        dep_infra = choix_local[:2]
        nom_zone = f"Secteur {choix_local}"
else:
    if choix_local == 'TOUS':
        filtre_dvf = f"code_departement = '{departement}'"
        filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
        dep_infra = departement
        nom_zone = f"Departement {departement}"
    else:
        filtre_dvf = f"code_commune = '{choix_local}'"
        filtre_dpe = f"code_insee_ban = '{choix_local}'"
        dep_infra = departement
        nom_zone = f"Secteur {choix_local}"

print(f"\nLancement de l'apprentissage pour : {nom_zone}")

DOSSIER_OUT = RACINE_PROJET / "out"

if departement == 'FRANCE':
    dossier_graphes = DOSSIER_OUT / "FRANCE" if choix_local == 'TOUS' else DOSSIER_OUT / choix_local[:2] / choix_local
else:
    dossier_graphes = DOSSIER_OUT / departement if choix_local == 'TOUS' else DOSSIER_OUT / departement / choix_local

dossier_graphes.mkdir(parents=True, exist_ok=True)

print("-" * 50)
temps_total_debut = time.time()

# ==========================================
# 1. TELECHARGEMENT DES DONNEES FILTREES
# ==========================================
print("Etape 1/4 : Extraction des donnees depuis SQL...")

maisons = pd.read_sql(f"""
    SELECT code_commune, id_parcelle, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2,
           surface_reelle_bati, type_local, nombre_pieces_principales,
           surface_terrain, YEAR(date_mutation) AS annee_vente, MONTH(date_mutation) AS mois_vente
    FROM valeurs_foncieres
    WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
        AND nature_mutation = 'Vente' AND nombre_lots <= 3 AND nombre_pieces_principales > 0
        AND {filtre_dvf} AND type_local IN ('Maison', 'Appartement')
        AND id_mutation IN (
            SELECT id_mutation FROM valeurs_foncieres
            WHERE surface_reelle_bati > 0
            GROUP BY id_mutation HAVING COUNT(*) = 1);
""", con=moteur)
maisons = maisons.drop_duplicates(subset=['id_parcelle','prix_m2','surface_reelle_bati'])

if len(maisons) == 0:
    print(f"Erreur : Aucune donnee trouvée.")
    sys.exit()

dpe = pd.read_sql(f"""
    SELECT code_insee_ban,
           (SUM(CASE WHEN etiquette_dpe = 'A' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_A,
           (SUM(CASE WHEN etiquette_dpe = 'B' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_B,
           (SUM(CASE WHEN etiquette_dpe = 'C' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_C,
           (SUM(CASE WHEN etiquette_dpe = 'D' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_D,
           (SUM(CASE WHEN etiquette_dpe = 'E' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_E,
           (SUM(CASE WHEN etiquette_dpe = 'F' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_F,
           (SUM(CASE WHEN etiquette_dpe = 'G' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_G,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Électricité%%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_chauffage_elec,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Gaz%%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_chauffage_gaz,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Fioul%%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_chauffage_fioul,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Réseau de Chaleur%%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_chauffage_urbain
    FROM dpe_logements_france
    WHERE etiquette_dpe IN ('A','B','C','D','E','F','G') AND {filtre_dpe}
    GROUP BY code_insee_ban;
""", con=moteur)

stations = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur)

monuments = pd.read_sql("SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL;", con=moteur)
hopitaux = pd.read_sql("SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL;", con=moteur)
universites = pd.read_sql("SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;", con=moteur)

filtre_rev = "1=1" if dep_infra == 'FRANCE' else (f"code_commune = '{choix_local}'" if choix_local != 'TOUS' and len(choix_local) == 5 else f"LEFT(code_commune,2) = '{dep_infra}'")

revenus = pd.read_sql(f"SELECT code_commune, median_revenu_disponible,indice_gini,pct_minima_sociaux FROM demographie_communes WHERE {filtre_rev};", con=moteur)

for col in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']:
    revenus[col] = pd.to_numeric(revenus[col], errors='coerce')

# Poles urbains - charge seulement si actif
poles = None
if FA['potentiel_urbain']:
    poles = pd.read_sql("""
        SELECT aav_nom, AVG(latitude) AS latitude, AVG(longitude) AS longitude, COUNT(*) AS poids_aire
        FROM referentiel_communes
        WHERE aav_nom IS NOT NULL AND aav_nom != 'SO' AND latitude IS NOT NULL
        GROUP BY aav_nom HAVING poids_aire >= 10
    """, con=moteur)
    poles_etrangers = pd.DataFrame([
        {'aav_nom': 'Genève',    'latitude': 46.2044, 'longitude': 6.1432, 'poids_aire': 250},
        {'aav_nom': 'Lausanne',  'latitude': 46.5197, 'longitude': 6.6323, 'poids_aire': 90},
        {'aav_nom': 'Bâle',      'latitude': 47.5596, 'longitude': 7.5886, 'poids_aire': 150},
        {'aav_nom': 'Neuchâtel', 'latitude': 46.9925, 'longitude': 6.9310, 'poids_aire': 40},
        {'aav_nom': 'Luxembourg','latitude': 49.6116, 'longitude': 6.1319, 'poids_aire': 200},
        {'aav_nom': 'Sarrebruck','latitude': 49.2402, 'longitude': 6.9969, 'poids_aire': 120},
        {'aav_nom': 'Karlsruhe', 'latitude': 49.0069, 'longitude': 8.4037, 'poids_aire': 130},
        {'aav_nom': 'Fribourg-en-Brisgau', 'latitude': 47.9990, 'longitude': 7.8421, 'poids_aire': 110},
        {'aav_nom': 'Bruxelles', 'latitude': 50.8503, 'longitude': 4.3517, 'poids_aire': 300},
        {'aav_nom': 'Charleroi', 'latitude': 50.4114, 'longitude': 4.4446, 'poids_aire': 90},
        {'aav_nom': 'Liège',     'latitude': 50.6326, 'longitude': 5.5797, 'poids_aire': 100},
        {'aav_nom': 'Mons',      'latitude': 50.4542, 'longitude': 3.9563, 'poids_aire': 50},
        {'aav_nom': 'Turin',     'latitude': 45.0703, 'longitude': 7.6869, 'poids_aire': 250},
        {'aav_nom': 'Vintimille','latitude': 43.7900, 'longitude': 7.6083, 'poids_aire': 30},
        {'aav_nom': 'Monaco',    'latitude': 43.7384, 'longitude': 7.4246, 'poids_aire': 120},
        {'aav_nom': 'Barcelone', 'latitude': 41.3874, 'longitude': 2.1686, 'poids_aire': 300},
        {'aav_nom': 'Saint-Sébastien', 'latitude': 43.3183, 'longitude': -1.9812, 'poids_aire': 70},
        {'aav_nom': 'Gérone',    'latitude': 41.9794, 'longitude': 2.8214, 'poids_aire': 40},
        {'aav_nom': 'Andorre-la-Vieille', 'latitude': 42.5063, 'longitude': 1.5218, 'poids_aire': 25},
    ])
    poles = pd.concat([poles, poles_etrangers], ignore_index=True)

# Sources economiques - chargees seulement si actives
pib = None
if FA['pib']:
    pib = pd.read_sql("SELECT annee, pib_national FROM pib_national", con=moteur)

chomage = None
if FA['chomage']:
    chomage = pd.read_sql("SELECT code_departement, annee, trimestre, taux_chomage FROM chomage_departements", con=moteur)

taux = None
if FA['taux_credit'] or FA['taux_inflation']:
    taux = pd.read_sql("SELECT annee, mois, taux_credit_immo_fixe, taux_inflation FROM taux_macro", con=moteur)

# ==========================================
# 2. FUSION ET DISTANCES GLOBALES
# ==========================================
print("Etape 2/4 : Fusion et calculs spatiaux partagés...")

donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees, revenus, on='code_commune', how='left')

# Cles departement + trimestre (utiles pour chomage)
donnees['code_departement'] = donnees['code_commune'].str[:2]
donnees['trimestre'] = (donnees['mois_vente'] - 1) // 3 + 1

# Jointure PIB (si actif)
if pib is not None:
    donnees = pd.merge(donnees, pib, left_on='annee_vente', right_on='annee', how='left')
    donnees = donnees.drop(columns=['annee'], errors='ignore')

# Jointure chomage (si actif)
if chomage is not None:
    donnees = pd.merge(donnees, chomage,
                       left_on=['code_departement', 'annee_vente', 'trimestre'],
                       right_on=['code_departement', 'annee', 'trimestre'], how='left')
    donnees = donnees.drop(columns=['annee'], errors='ignore')

# Jointure taux (si actif)
if taux is not None:
    donnees = pd.merge(donnees, taux,
                       left_on=['annee_vente', 'mois_vente'],
                       right_on=['annee', 'mois'], how='left')
    donnees = donnees.drop(columns=['annee', 'mois'], errors='ignore')

RAYON_TERRE_METRES = 6371000
points_rad = np.deg2rad(donnees[['latitude', 'longitude']])

def calculer_distance_min(df_points, nom_colonne):
    if len(df_points) > 0:
        infra_rad = np.deg2rad(df_points.iloc[:, 0:2])
        arbre = BallTree(infra_rad, metric='haversine')
        dist_rad, _ = arbre.query(points_rad, k=1)
        donnees[nom_colonne] = dist_rad.flatten() * RAYON_TERRE_METRES
    else:
        donnees[nom_colonne] = 999999

# Distances - calculees seulement si le groupe correspondant est actif
if FA['dist_transport']:
    calculer_distance_min(stations, 'dist_transport_m')
if FA['dist_monument']:
    calculer_distance_min(monuments, 'dist_monument_m')
if FA['dist_hopital']:
    calculer_distance_min(hopitaux, 'dist_hopital_m')

if FA['dist_universite']:
    if len(universites) > 0:
        univ_rad = np.deg2rad(universites[['latitude', 'longitude']])
        arbre_univ = BallTree(univ_rad, metric='haversine')
        dist_rad, idx_univ = arbre_univ.query(points_rad, k=1)
        donnees['dist_universite_m'] = dist_rad.flatten() * RAYON_TERRE_METRES
        donnees['volume_etudiants_proche'] = universites.iloc[idx_univ.flatten()]['nombre_etudiants'].values
    else:
        donnees['dist_universite_m'] = 999999
        donnees['volume_etudiants_proche'] = 0
else:
    donnees['volume_etudiants_proche'] = 0

def extraire_points_contour(sous_gdf):
    points = []
    for geom in sous_gdf.geometry :
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                points.extend(list(poly.exterior.coords))
        else :
            points.extend(list(geom.exterior.coords))
    if not points:
        return pd.DataFrame(columns=['latitude','longitude'])
    pts = np.array(points)
    return pd.DataFrame(pts[:,[1,0]], columns=['latitude','longitude'])

if FA['dist_littoral']:
    classements = {'Mer':'dist_mer_m','Lac':'dist_lac_m','Estuaire':'dist_estuaire_m'}
    for classement, nom_colonne in classements.items():
        sous = gdf_littoral[gdf_littoral['CLASSEMENT'] == classement]
        df_points = extraire_points_contour(sous)
        calculer_distance_min(df_points,nom_colonne)

# Potentiel urbain (si actif)
if FA['potentiel_urbain'] and poles is not None and len(poles) > 0:
    poles_rad = np.deg2rad(poles[['latitude', 'longitude']].values)
    poids_poles = poles['poids_aire'].values.astype(float)
    arbre_poles = BallTree(poles_rad, metric='haversine')
    k_poles = min(20, len(poles))
    dist_rad_p, idx_p = arbre_poles.query(points_rad, k=k_poles)
    dist_m_p = dist_rad_p * RAYON_TERRE_METRES
    donnees['potentiel_urbain'] = np.sum(poids_poles[idx_p] / (dist_m_p + 5000), axis=1)

# ==========================================
# 3. NETTOYAGE GLOBAL ET CORRECTION DVF
# ==========================================
print("Etape 3/4 : Nettoyage global et correction des biais...")

colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
colonnes_chauffage = ['pct_chauffage_elec', 'pct_chauffage_gaz', 'pct_chauffage_fioul', 'pct_chauffage_urbain']
colonnes_revenus = ['median_revenu_disponible','indice_gini','pct_minima_sociaux']

donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)

# Remplissage des NaN des features economiques (si presentes)
for col in ['pib_national', 'taux_chomage', 'taux_credit_immo_fixe', 'taux_inflation', 'potentiel_urbain']:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

donnees_propres = donnees[
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

mask_appart = donnees_propres['type_local'] == 'Appartement'
donnees_propres.loc[mask_appart, 'surface_terrain'] = 0

# ==========================================
# FEATURE ENGINEERING : creation des variables derivees
# ==========================================
donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])
donnees_propres['log_surface'] = np.log(donnees_propres['surface_reelle_bati'])
donnees_propres['surface_par_piece'] = donnees_propres['surface_reelle_bati'] / donnees_propres['nombre_pieces_principales']
donnees_propres['a_terrain'] = (donnees_propres['surface_terrain'] > 0).astype(int)
donnees_propres['log_terrain'] = np.log1p(donnees_propres['surface_terrain'])
donnees_propres['code_section'] = donnees_propres['id_parcelle'].str[:10]

# ==========================================
# CONSTRUCTION DYNAMIQUE DE LA LISTE DES FEATURES (selon les interrupteurs)
# ==========================================
def construire_features_base():
    f = []
    if FA['geo_base']:   f += ['latitude', 'longitude']
    if FA['pieces']:     f += ['nombre_pieces_principales']
    if FA['date']:       f += ['annee_vente', 'mois_vente']
    if FA['terrain']:    f += ['a_terrain', 'surface_terrain', 'log_terrain']
    if FA['surface']:    f += ['surface_reelle_bati', 'log_surface', 'surface_par_piece']
    if FA['dist_universite']: f += ['volume_etudiants_proche']
    if FA['revenus']:    f += colonnes_revenus
    if FA['potentiel_urbain'] and 'potentiel_urbain' in donnees_propres.columns:
        f += ['potentiel_urbain']
    if FA['dpe']:        f += colonnes_dpe
    if FA['chauffage']:  f += colonnes_chauffage
    # Distances (selon interrupteurs individuels)
    if FA['dist_transport']:  f += ['dist_transport_m']
    if FA['dist_monument']:   f += ['dist_monument_m']
    if FA['dist_hopital']:    f += ['dist_hopital_m']
    if FA['dist_universite']: f += ['dist_universite_m']
    if FA['dist_littoral']:   f += ['dist_mer_m', 'dist_lac_m', 'dist_estuaire_m']
    # Features economiques (selon interrupteurs)
    if FA['pib'] and 'pib_national' in donnees_propres.columns:
        f += ['pib_national']
    if FA['chomage'] and 'taux_chomage' in donnees_propres.columns:
        f += ['taux_chomage']
    if FA['taux_credit'] and 'taux_credit_immo_fixe' in donnees_propres.columns:
        f += ['taux_credit_immo_fixe']
    if FA['taux_inflation'] and 'taux_inflation' in donnees_propres.columns:
        f += ['taux_inflation']
    # Dedoublonne en gardant l'ordre
    return list(dict.fromkeys(f))

features_base = construire_features_base()
print(f"\n{len(features_base)} features de base actives : {features_base}")

# ==========================================
# 4. BOUCLE D'EVALUATION (MAISONS / APPARTEMENTS)
# ==========================================
print("Etape 4/4 : Evaluation specifique par type de bien (CatBoost)...")

datasets = {
    'maisons': donnees_propres[donnees_propres['type_local'] == 'Maison'].copy(),
    'appartements': donnees_propres[donnees_propres['type_local'] == 'Appartement'].copy()
}

for type_bien, df_bien in datasets.items():
    if len(df_bien) < 50:
        print(f"\n--- IGNORÉ : Pas assez de donnees pour le type {type_bien} ---")
        continue

    plancher = max(df_bien['prix_m2'].quantile(0.01), 500)
    plafond = min(df_bien['prix_m2'].quantile(0.99), 20000)

    df_bien = df_bien[
        (df_bien['prix_m2'] >= plancher) & (df_bien['prix_m2'] <= plafond)
    ].copy()
    print(f"\n{type_bien} : {len(df_bien)} biens apres filtrage")

    print("\n" + "=" * 50)
    print(f"ANALYSE DU FLUX : {type_bien.upper()} ({len(df_bien)} biens)")
    print("=" * 50)

    X = df_bien[features_base].copy()
    y = df_bien['log_prix_m2']

    annee_max = df_bien['annee_vente'].max()
    train_mask = df_bien['annee_vente'] < annee_max
    test_mask = df_bien['annee_vente'] == annee_max

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    if type_bien == 'appartements':
        print(f"\n=== DIAGNOSTIC APPARTEMENTS (correlation_cat) ===")
        print(f"Nombre total apparts : {len(df_bien)}")
        print(f"Train = {len(X_train)} | Test : {len(X_test)}")
        print(f"Anne de test : {annee_max}")
        print(f"Prix m2 - Median : {df_bien['prix_m2'].median():.0f}")
        print(f"Annees train : {sorted(df_bien.loc[X_train.index, 'annee_vente'].unique())}")
        print(f"Annee test : {sorted(df_bien.loc[X_test.index,'annee_vente'].unique())}")

    coords_train = np.deg2rad(df_bien.loc[X_train.index, ['latitude', 'longitude']])
    arbre_voisins = BallTree(coords_train, metric='haversine')

    # ---- Indice de marche (train uniquement) : actualisation des prix ----
    annees_train_serie = df_bien.loc[X_train.index, 'annee_vente']
    idx_marche = df_bien.loc[X_train.index].groupby('annee_vente')['prix_m2'].median()
    ref_marche = idx_marche.loc[idx_marche.index.max()]
    coef_marche = (ref_marche / idx_marche).to_dict()

    prix_train = df_bien.loc[X_train.index, 'prix_m2'].values
    prix_train_actu = prix_train * annees_train_serie.map(coef_marche).values
    surface_train = df_bien.loc[X_train.index, 'surface_reelle_bati'].values

    features_finales = list(features_base)

    # --- Features de voisinage (si actives) ---
    if FA['voisins']:
        def voisins_surface_ponderes(distances_rad, indices, surface_bien, idx_self=None):
            if idx_self is not None:
                garder = indices != idx_self
                distances_rad, indices = distances_rad[garder], indices[garder]
            if len(indices) == 0:
                return np.nan
            dist_m = distances_rad * RAYON_TERRE_METRES
            prix_v = prix_train_actu[indices]
            surf_v = surface_train[indices]
            borne_bas, borne_haut = surface_bien * 0.6, surface_bien * 1.4
            masque = (surf_v >= borne_bas) & (surf_v <= borne_haut)
            if masque.sum() >= 3:
                d, p = dist_m[masque], prix_v[masque]
            else:
                d, p = dist_m, prix_v
            poids = 1.0 / (d + 50.0)
            return np.sum(poids * p) / np.sum(poids)

        k_train = min(41, len(coords_train))
        dist_tr, idx_tr = arbre_voisins.query(coords_train, k=k_train)
        surface_bien_train = X_train['surface_reelle_bati'].values if 'surface_reelle_bati' in X_train.columns else df_bien.loc[X_train.index, 'surface_reelle_bati'].values
        voisins_train = [
            voisins_surface_ponderes(dist_tr[i], idx_tr[i], surface_bien_train[i], idx_self=i)
            for i in range(len(idx_tr))
        ]
        k_test = min(40, len(coords_train))
        coords_test = np.deg2rad(df_bien.loc[X_test.index, ['latitude', 'longitude']])
        dist_te, idx_te = arbre_voisins.query(coords_test, k=k_test)
        surface_bien_test = X_test['surface_reelle_bati'].values if 'surface_reelle_bati' in X_test.columns else df_bien.loc[X_test.index, 'surface_reelle_bati'].values
        voisins_test = [
            voisins_surface_ponderes(dist_te[i], idx_te[i], surface_bien_test[i])
            for i in range(len(idx_te))
        ]
        X_train['prix_m2_voisins'] = voisins_train
        X_test['prix_m2_voisins'] = voisins_test
        features_finales += ['prix_m2_voisins']

    # --- Densite de ventes (si active) ---
    if FA['densite']:
        rayon_rad = 1000 / RAYON_TERRE_METRES
        coords_test_d = np.deg2rad(df_bien.loc[X_test.index, ['latitude', 'longitude']])
        X_train['densite_ventes_1km'] = arbre_voisins.query_radius(coords_train, r=rayon_rad, count_only=True)
        X_test['densite_ventes_1km'] = arbre_voisins.query_radius(coords_test_d, r=rayon_rad, count_only=True)
        features_finales += ['densite_ventes_1km']

    # --- Prix par section (si actif) : encodage out-of-fold anti-leakage ---
    if FA['section']:
        df_tr = df_bien.loc[X_train.index].copy()
        df_tr['prix_m2_actu'] = df_tr['prix_m2'].values * df_tr['annee_vente'].map(coef_marche).values
        med_commune = df_tr.groupby('code_commune')['prix_m2_actu'].median()
        med_globale = df_tr['prix_m2_actu'].median()

        vals_train = pd.Series(np.nan, index=X_train.index)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for pos_fit, pos_oof in kf.split(df_tr):
            med_s = df_tr.iloc[pos_fit].groupby('code_section')['prix_m2_actu'].median()
            med_c = df_tr.iloc[pos_fit].groupby('code_commune')['prix_m2_actu'].median()
            sous = df_tr.iloc[pos_oof]
            v = sous['code_section'].map(med_s)
            v = v.fillna(sous['code_commune'].map(med_c))
            v = v.fillna(df_tr.iloc[pos_fit]['prix_m2_actu'].median())
            vals_train.iloc[pos_oof] = v.values
        X_train['prix_m2_section'] = vals_train.values

        med_section = df_tr.groupby('code_section')['prix_m2_actu'].median()
        sec_te = df_bien.loc[X_test.index, 'code_section'].map(med_section)
        com_te = df_bien.loc[X_test.index, 'code_commune'].map(med_commune)
        X_test['prix_m2_section'] = sec_te.fillna(com_te).fillna(med_globale).values

        nb_ventes_section = df_tr.groupby('code_section').size()
        X_train['nb_ventes_section'] = df_bien.loc[X_train.index, 'code_section'].map(nb_ventes_section).fillna(0).values
        X_test['nb_ventes_section'] = df_bien.loc[X_test.index, 'code_section'].map(nb_ventes_section).fillna(0).values
        features_finales += ['prix_m2_section', 'nb_ventes_section']

    features_finales = list(dict.fromkeys(features_finales))
    X_train = X_train[features_finales]
    X_test = X_test[features_finales]

    # ==========================================
    # MODELE CATBOOST
    # ==========================================
    mask_val = (annees_train_serie == annees_train_serie.max()).values
    if mask_val.sum() < 50 or (~mask_val).sum() < 200:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    else:
        X_tr, y_tr = X_train[~mask_val], y_train[~mask_val]
        X_val, y_val = X_train[mask_val], y_train[mask_val]

    quantiles = {'bas':0.025,'median':0.50,'haut':0.975}
    modeles_q = {}
    for nom_q, alpha in quantiles.items():
        m = CatBoostRegressor(
            loss_function=f'Quantile:alpha={alpha}',
            iterations = 2000, learning_rate=0.04,depth=8,
            random_seed=42,early_stopping_rounds=50,verbose=False
        )
        m.fit(X_tr,y_tr,eval_set=(X_val,y_val),use_best_model=True)
        modeles_q[nom_q] = m

    modele_cat = modeles_q['median']
    best_iter = modele_cat.get_best_iteration()

    print(f"\nArbres construits jusqu'a   : {modele_cat.get_param('iterations')} (plafond)")
    print(f"Meilleur arbre (arret)      : {best_iter}")
    print(f"Arbres reellement utilises  : {best_iter + 1}")

    pred_bas_log = modeles_q['bas'].predict(X_test)
    pred_med_log = modeles_q['median'].predict(X_test)
    pred_haut_log = modeles_q['haut'].predict(X_test)

    prix_reels_euros = np.exp(y_test)
    prix_bas = np.exp(pred_bas_log)
    prix_predits_euros = np.exp(pred_med_log)
    prix_haut = np.exp(pred_haut_log)

    prix_bas, prix_haut = np.minimum(prix_bas, prix_haut), np.maximum(prix_bas, prix_haut)

    mae = mean_absolute_error(prix_reels_euros, prix_predits_euros)
    mape = np.mean(np.abs((prix_reels_euros - prix_predits_euros) / prix_reels_euros)) * 100
    erreur_mediane = np.median(np.abs(prix_reels_euros - prix_predits_euros))
    rmse = np.sqrt(np.mean((prix_reels_euros.values - prix_predits_euros) ** 2))
    erreur_rel = np.abs(prix_reels_euros.values - prix_predits_euros) / prix_reels_euros
    r2_euros = r2_score(prix_reels_euros, prix_predits_euros)

    dans_intervalle = (prix_reels_euros.values >= prix_bas) & (prix_reels_euros.values <= prix_haut)
    couverture = np.mean(dans_intervalle) * 100
    largeur_moyenne = np.mean(prix_haut - prix_bas)

    print(f"Calcul des valeurs SHAP pour {type_bien}...")
    explainer = shap.TreeExplainer(modele_cat)
    shap_values = explainer.shap_values(X_test)

    importance_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
    print(f"\nTop 5 SHAP ({type_bien}) :")
    for nom, val in importance_shap.head(5).items():
        print(f"  - {nom.ljust(25)} : {val:.4f}")

    nom_fichier_base = nom_zone.replace(' ', '_')

    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title(f"Impact des variables (CatBoost) - {nom_zone} ({type_bien})", fontsize=13)
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"shap_summary_CAT_{nom_fichier_base}_{type_bien}.png", dpi=150, bbox_inches='tight')
    plt.close()

    evals = modele_cat.get_evals_result()
    nom_metrique = list(evals['learn'].keys())[0]
    courbe_train = evals['learn'][nom_metrique]
    courbe_val = evals['validation'][nom_metrique]

    plt.figure(figsize=(10, 6))
    plt.plot(courbe_train,label='Entrainement',color='steelblue')
    plt.plot(courbe_val, label='Validation', color='darkorange')
    plt.axvline(best_iter, color='green', linestyle='--', label=f'Arret optimal (arbre {best_iter})')
    plt.xlabel("Nombre d'arbres")
    plt.ylabel(f"Erreur quantile({nom_metrique})")
    plt.title(f"Courbe d'apprentissage (CatBoost) - {nom_zone} ({type_bien})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"courbe_apprentissage_CAT_{nom_fichier_base}_{type_bien}.png")
    plt.close()
    print("Courbe d'apprentissage enregistree")

    variables_a_tracer = ['prix_m2_section', 'surface_reelle_bati', 'prix_m2_voisins', 'median_revenu_disponible']
    if type_bien == 'maisons':
        variables_a_tracer.append('surface_terrain')

    for var in variables_a_tracer:
        if var not in X_test.columns:
            continue
        if X_test[var].nunique() <= 1:
            continue
        plt.figure()
        shap.dependence_plot(var, shap_values, X_test, interaction_index=None, show=False)
        plt.title(f"Effet de {var} (CatBoost) - {nom_zone} ({type_bien})", fontsize=11)
        plt.tight_layout()
        plt.savefig(dossier_graphes / f"shap_dep_CAT_{var}_{nom_fichier_base}_{type_bien}.png", dpi=150, bbox_inches='tight')
        plt.close()
    print(f"Dependence plots enregistres pour {type_bien}")

    plt.figure(figsize=(8, 8))
    plt.scatter(prix_reels_euros, prix_predits_euros, alpha=0.3, s=10)
    lims = [min(prix_reels_euros.min(), prix_predits_euros.min()), max(prix_reels_euros.max(), prix_predits_euros.max())]
    plt.plot(lims, lims, 'r--', linewidth=2, label='Prédiction parfaite')
    plt.axhline(np.mean(prix_predits_euros), color='blue', linestyle=':', linewidth=2, label=f'Moyenne prédite : {np.mean(prix_predits_euros):.0f} EUR/m²')
    plt.xlabel("Prix réel (EUR/m²)")
    plt.ylabel("Prix prédit (EUR/m²)")
    plt.title(f"Prédictions vs réalité (CatBoost) - {nom_zone} ({type_bien})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"pred_vs_reels_CAT_{nom_fichier_base}_{type_bien}.png", dpi=150)
    plt.close()

    residus = prix_reels_euros.values - prix_predits_euros
    plt.figure(figsize=(9, 5))
    plt.hist(residus, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='Erreur nulle')
    plt.axvline(np.mean(residus), color='orange', linestyle='--', label=f'Biais moyen : {np.mean(residus):.0f} EUR/m²')
    plt.xlabel("Résidu (réel - prédit) en EUR/m²")
    plt.ylabel("Nombre de biens")
    plt.title(f"Distribution des erreurs (CatBoost) - {nom_zone} ({type_bien})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"residu_CAT_{nom_fichier_base}_{type_bien}.png", dpi=150)
    plt.close()

    ordre = np.argsort(prix_predits_euros)
    ech = ordre[::max(1, len(ordre)//300)]

    plt.figure(figsize=(11, 6))
    x_axis = range(len(ech))
    plt.fill_between(x_axis, prix_bas[ech], prix_haut[ech], alpha=0.3,
                     color='steelblue', label='Intervalle 90%')
    plt.plot(x_axis, prix_predits_euros[ech], color='navy', linewidth=1, label='Prediction (median)')
    plt.scatter(x_axis, prix_reels_euros.values[ech], color='red', s=8, label='Prix reel')
    plt.xlabel("Biens (tries par prix predit)")
    plt.ylabel("Prix (EUR/m²)")
    plt.title(f"Intervalles de confiance - {nom_zone} ({type_bien})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"intervalles_CAT_{nom_fichier_base}_{type_bien}.png", dpi=150)
    plt.close()

    print("\n" + "-" * 50)
    print(f"RAPPORT CATBOOST - {type_bien.upper()}")
    print("-" * 50)
    print(f"Apprentissage : {len(X_train)} biens | Validation : {len(X_test)} biens")
    print(f"R2 (euros / m2) : {r2_euros * 100:.2f} %")
    print(f"MAE             : {mae:.2f} EUR / m2")
    print(f"MAPE            : {mape:.1f} %")
    print(f"Erreur mediane  : {erreur_mediane:.0f} EUR/m2")
    print(f"RMSE            : {rmse:.0f} EUR/m2")
    print(f"Dans les +/- 10%: {np.mean(erreur_rel <= 0.10) * 100:.1f} %")
    print(f"Dans les +/- 20%: {np.mean(erreur_rel <= 0.20) * 100:.1f} %")
    print(f"Couverture intervalle 95% : {couverture:.1f} %")
    print(f"Largeur moyenne intervalle : {largeur_moyenne:.0f} EUR/m2")

print("\n" + "=" * 50)
print(f"Temps de traitement global : {time.time() - temps_total_debut:.2f} secondes.")