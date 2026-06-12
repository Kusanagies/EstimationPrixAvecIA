"""
PHASE 2 - ENTRAINEMENT ET SAUVEGARDE POUR LA PRODUCTION
========================================================
Ce script reprend le pipeline d'evaluation mais :
  - n'utilise PAS de split temporel (entraine sur TOUTES les donnees)
  - sauvegarde les 3 modeles quantiles, l'ordre des features, et le
    contexte d'enrichissement necessaire pour estimer un bien depuis une adresse.

A lancer une fois (ou a chaque mise a jour des donnees). Le fichier
d'evaluation reste separe et sert a mesurer la fiabilite du modele.
"""

import sys
import os
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ==========================================
# 0. CONNEXION + CHOIX DE LA ZONE
# ==========================================
print("ENTRAINEMENT DU MODELE DE PRODUCTION")
print("-" * 50)

RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")

# Dossier ou seront ecrits les artefacts (modeles, contexte...)
DOSSIER_MODELE = RACINE_PROJET / "modele_production"
DOSSIER_MODELE.mkdir(exist_ok=True)

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except KeyError:
    print("Erreur : variable DB_PASS introuvable.")
    sys.exit()
except Exception:
    print("Erreur de connexion a la base MySQL.")
    sys.exit()

departement = input("Departement (ex: 34, 75) ou 'FRANCE' : ").strip().upper()
if len(departement) < 2:
    print("Format invalide.")
    sys.exit()

# Pour la production, on entraine sur le departement entier (ou la France)
if departement == 'FRANCE':
    filtre_dvf = "1=1"
    filtre_dpe = "1=1"
    filtre_rev = "1=1"
    dep_infra = "FRANCE"
    nom_zone = "France"
else:
    filtre_dvf = f"LEFT(code_commune, 2) = '{departement}'"
    filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
    filtre_rev = f"LEFT(code_commune, 2) = '{departement}'"
    dep_infra = departement
    nom_zone = f"Departement {departement}"

print(f"\nEntrainement pour : {nom_zone}")
print("-" * 50)

# ==========================================
# 1. EXTRACTION DES DONNEES
# ==========================================
print("Etape 1 : Extraction SQL...")

maisons = pd.read_sql(f"""
    SELECT code_commune, id_parcelle, latitude, longitude,
           (valeur_fonciere / surface_reelle_bati) AS prix_m2,
           surface_reelle_bati, type_local, nombre_pieces_principales,
           surface_terrain,
           YEAR(date_mutation) AS annee_vente,
           MONTH(date_mutation) AS mois_vente
    FROM valeurs_foncieres
    WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
      AND nature_mutation = 'Vente'
      AND nombre_lots <= 1
      AND nombre_pieces_principales > 0
      AND {filtre_dvf}
      AND type_local IN ('Maison', 'Appartement');
""", con=moteur)

if len(maisons) == 0:
    print("Aucune donnee.")
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
    WHERE etiquette_dpe IN ('A','B','C','D','E','F','G')
      AND {filtre_dpe}
    GROUP BY code_insee_ban;
""", con=moteur)

stations = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur)

if dep_infra == 'FRANCE':
    q_mon = "SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL;"
    q_hop = "SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL;"
    q_uni = "SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;"
else:
    q_mon = f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{dep_infra}';"
    q_hop = f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal, 2) = '{dep_infra}';"
    q_uni = f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{dep_infra}';"

monuments = pd.read_sql(q_mon, con=moteur)
hopitaux = pd.read_sql(q_hop, con=moteur)
universites = pd.read_sql(q_uni, con=moteur)

revenus = pd.read_sql(f"""
    SELECT code_commune, median_revenu_disponible, indice_gini, pct_minima_sociaux
    FROM demographie_communes
    WHERE {filtre_rev};
""", con=moteur)
for col in ['median_revenu_disponible', 'indice_gini', 'pct_minima_sociaux']:
    revenus[col] = pd.to_numeric(revenus[col], errors='coerce')

# ==========================================
# 2. FUSION
# ==========================================
print("Etape 2 : Fusion...")
donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees, revenus, on='code_commune', how='left')

# ==========================================
# 3. DISTANCES SPATIALES
# ==========================================
print("Etape 3 : Distances...")
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
# 4. NETTOYAGE + FEATURES
# ==========================================
print("Etape 4 : Features...")

colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
colonnes_chauffage = ['pct_chauffage_elec', 'pct_chauffage_gaz', 'pct_chauffage_fioul', 'pct_chauffage_urbain']
colonnes_revenus = ['median_revenu_disponible', 'indice_gini', 'pct_minima_sociaux']

for col in colonnes_dpe + colonnes_chauffage + colonnes_revenus:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)

donnees_propres = donnees[
    (donnees['prix_m2'] >= 500) & (donnees['prix_m2'] <= 25000) &
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])
donnees_propres['est_maison'] = (donnees_propres['type_local'] == 'Maison').astype(int)
donnees_propres['log_surface'] = np.log(donnees_propres['surface_reelle_bati'])
donnees_propres['surface_par_piece'] = donnees_propres['surface_reelle_bati'] / donnees_propres['nombre_pieces_principales']
donnees_propres['a_terrain'] = (donnees_propres['surface_terrain'] > 0).astype(int)
donnees_propres['log_terrain'] = np.log1p(donnees_propres['surface_terrain'])
donnees_propres['code_section'] = donnees_propres['id_parcelle'].str[:10]

colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
colonnes_standard = ['surface_reelle_bati', 'volume_etudiants_proche',
                     'log_surface', 'surface_par_piece',
                     'surface_terrain', 'log_terrain'] + colonnes_revenus

# ==========================================
# 5. FEATURES SPATIALES (sur TOUT le dataset, pas de split)
# ==========================================
print("Etape 5 : Features de voisinage (sur tout le dataset)...")

features = ['est_maison', 'latitude', 'longitude', 'nombre_pieces_principales',
            'annee_vente', 'mois_vente', 'a_terrain'] \
           + colonnes_dpe + colonnes_chauffage + colonnes_standard + colonnes_dist

X = donnees_propres[features].copy()
y = donnees_propres['log_prix_m2']

# En production : voisins calcules sur TOUTES les donnees (pas de leakage ici,
# car le but n'est plus d'evaluer mais de produire les meilleures features).
coords_all = np.deg2rad(donnees_propres[['latitude', 'longitude']])
prix_all = donnees_propres['prix_m2'].values
arbre_voisins = BallTree(coords_all, metric='haversine')

k = min(16, len(coords_all))
_, idx_v = arbre_voisins.query(coords_all, k=k)
voisins_prix = prix_all[idx_v[:,1:]]
X['prix_m2_voisins'] = np.median(voisins_prix,axis=1)

rayon_rad = 1000 / RAYON_TERRE_METRES
X['densite_ventes_1km'] = arbre_voisins.query_radius(coords_all, r=rayon_rad, count_only=True)

med_section = donnees_propres.groupby('code_section')['prix_m2'].median()
med_commune = donnees_propres.groupby('code_commune')['prix_m2'].median()
med_globale = donnees_propres['prix_m2'].median()

sec = donnees_propres['code_section']
com = donnees_propres['code_commune']

prix_sec = sec.map(med_section)
prix_sec = prix_sec.fillna(com.map(med_commune))
prix_sec = prix_sec.fillna(med_globale)
X['prix_m2_section'] = prix_sec.values

nb_ventes_section = donnees_propres.groupby('code_section').size()
X['nb_ventes_section'] = donnees_propres['code_section'].map(nb_ventes_section).fillna(0).values

features = features + ['prix_m2_voisins', 'densite_ventes_1km', 'prix_m2_section', 'nb_ventes_section']
features = list(dict.fromkeys(features))
X = X[features]

# ==========================================
# 6. ENTRAINEMENT DES 3 MODELES QUANTILES
# ==========================================
print("Etape 6 : Entrainement des modeles quantiles...")

# On garde un petit jeu de validation juste pour l'early stopping
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

quantiles = {'bas': 0.05, 'median': 0.50, 'haut': 0.95}
modeles = {}
for nom, alpha in quantiles.items():
    m = xgb.XGBRegressor(
        objective='reg:quantileerror', quantile_alpha=alpha,
        n_estimators=2000, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=1.0,
        early_stopping_rounds=50, random_state=42, n_jobs=-1
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    modeles[nom] = m
    print(f"  Modele '{nom}' entraine ({m.best_iteration + 1} arbres).")

# ==========================================
# 7. SAUVEGARDE DES ARTEFACTS
# ==========================================
print("Etape 7 : Sauvegarde...")

for nom, m in modeles.items():
    m.save_model(str(DOSSIER_MODELE / f"modele_{nom}.json"))

with open(DOSSIER_MODELE / "features.json", "w") as f:
    json.dump(features, f)

# Contexte d'enrichissement : tout ce qu'il faut pour reconstruire les features
# d'un bien a partir de son adresse (voir estimer.py).
contexte = {
    'arbre_voisins_data': coords_all.values,   # coords (radians) pour reconstruire l'arbre
    'prix_all': prix_all,
    'med_section': med_section.to_dict(),
    'med_commune': med_commune.to_dict(),
    'med_globale': float(med_globale),
    'nb_ventes_section': nb_ventes_section.to_dict(),
    'stations': stations,
    'monuments': monuments,
    'hopitaux': hopitaux,
    'universites': universites,
    # Profils communaux (DPE, chauffage, revenus) indexes par code commune
    'profils_communes': donnees.drop_duplicates('code_commune').set_index('code_commune')[
        colonnes_dpe + colonnes_chauffage + colonnes_revenus
    ].to_dict('index'),
    'medianes_globales': {c: float(donnees[c].median()) for c in
                          colonnes_dpe + colonnes_chauffage + colonnes_revenus},
    'rayon_terre': RAYON_TERRE_METRES,
}
with open(DOSSIER_MODELE / "contexte.pkl", "wb") as f:
    pickle.dump(contexte, f)

print("-" * 50)
print(f"Artefacts sauvegardes dans : {DOSSIER_MODELE}")
print("  - modele_bas.json / modele_median.json / modele_haut.json")
print("  - features.json")
print("  - contexte.pkl")
print("Pret pour l'estimation (Phase 3).")