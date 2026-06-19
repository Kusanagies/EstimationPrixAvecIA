"""
PHASE 2 - ENTRAINEMENT ET SAUVEGARDE POUR LA PRODUCTION (MAISONS & APPARTEMENTS)
================================================================================
Ce script reprend le pipeline d'evaluation mais :
  - n'utilise PAS de split temporel (entraine sur TOUTES les donnees)
  - separe completement les flux Maisons et Appartements.
  - sauvegarde 6 modeles quantiles (3 par type), l'ordre des features, et les
    contextes d'enrichissement necessaires pour la production.
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
import geopandas as gpd

# ==========================================
# 0. CONNEXION + CHOIX DE LA ZONE
# ==========================================
print("ENTRAINEMENT DU MODELE DE PRODUCTION (MAISONS & APPARTEMENTS)")
print("-" * 50)

RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"
gdf_littoral = gpd.read_file(CHEMIN_GPKG)

if gdf_littoral.crs is not None and gdf_littoral.crs.to_epsg() != 4326:
    gdf_littoral = gdf_littoral.to_crs(epsg=4326)

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

maisons_apparts = pd.read_sql(f"""
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

if len(maisons_apparts) == 0:
    print("Aucune donnee immobiliere trouvee.")
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
# 2. FUSION GLOBALE
# ==========================================
print("Etape 2 : Fusion...")
donnees = pd.merge(maisons_apparts, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees, revenus, on='code_commune', how='left')

# ==========================================
# 3. DISTANCES SPATIALES (Communes aux deux types)
# ==========================================
print("Etape 3 : Calcul des distances aux infrastructures...")
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

calculer_distance_min(stations, 'dist_transport_m')
calculer_distance_min(monuments, 'dist_monument_m')
calculer_distance_min(hopitaux, 'dist_hopital_m')

if len(universites) > 0:
    univ_rad = np.deg2rad(universites[['latitude', 'longitude']])
    arbre_univ = BallTree(univ_rad, metric='haversine')
    dist_rad, idx_univ = arbre_univ.query(points_rad, k=1)
    donnees['dist_universite_m'] = dist_rad.flatten() * RAYON_TERRE_METRES
    donnees['volume_etudiants_proche'] = universites.iloc[idx_univ.flatten()]['nombre_etudiants'].values
else:
    donnees['dist_universite_m'] = 999999
    donnees['volume_etudiants_proche'] = 0

def extraire_points_contour(sous_gdf):
    points = []
    for geom in sous_gdf.geometry:
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                points.extend(list(poly.exterior.coords))
        else:
            points.extend(list(geom.exterior.coords))
    if not points:
        return pd.DataFrame(columns=['latitude', 'longitude'])
    pts = np.array(points)
    return pd.DataFrame(pts[:, [1, 0]], columns=['latitude', 'longitude'])

classements = {'Mer': 'dist_mer_m', 'Lac': 'dist_lac_m', 'Estuaire': 'dist_estuaire_m'}
for classement, nom_colonne in classements.items():
    sous = gdf_littoral[gdf_littoral['CLASSEMENT'] == classement]
    df_points = extraire_points_contour(sous)
    calculer_distance_min(df_points, nom_colonne)
# ==========================================
# 4. NETTOYAGE + FEATURES
# ==========================================
print("Etape 4 : Nettoyage et creation des variables...")

colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
colonnes_chauffage = ['pct_chauffage_elec', 'pct_chauffage_gaz', 'pct_chauffage_fioul', 'pct_chauffage_urbain']
colonnes_revenus = ['median_revenu_disponible', 'indice_gini', 'pct_minima_sociaux']

for col in colonnes_dpe + colonnes_chauffage + colonnes_revenus:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)

plancher = max(donnees['prix_m2'].quantile(0.01),800)
plafond = min(donnees['prix_m2'].quantile(0.99),15000)
# Filtrage securise pour eviter les valeurs extremes
donnees_propres = donnees[
    (donnees['prix_m2'] >= plancher) & (donnees['prix_m2'] <= plafond) & 
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])
donnees_propres['log_surface'] = np.log(donnees_propres['surface_reelle_bati'])
donnees_propres['surface_par_piece'] = donnees_propres['surface_reelle_bati'] / donnees_propres['nombre_pieces_principales']
donnees_propres['a_terrain'] = (donnees_propres['surface_terrain'] > 0).astype(int)
donnees_propres['log_terrain'] = np.log1p(donnees_propres['surface_terrain'])
donnees_propres['code_section'] = donnees_propres['id_parcelle'].str[:10]

colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
colonnes_standard = ['surface_reelle_bati', 'volume_etudiants_proche',
                     'log_surface', 'surface_par_piece',
                     'surface_terrain', 'log_terrain'] + colonnes_revenus

# Note : On retire 'est_maison' des features car chaque modele n'aura qu'un seul type
features_base = ['latitude', 'longitude', 'nombre_pieces_principales',
                 'annee_vente', 'mois_vente', 'a_terrain'] \
                + colonnes_dpe + colonnes_chauffage + colonnes_standard + colonnes_dist

# ==========================================
# 5. BIFURCATION : MAISONS VS APPARTEMENTS
# ==========================================
print("Etape 5 : Separation des flux (Maisons / Appartements)...")

datasets = {
    'maisons': donnees_propres[donnees_propres['type_local'] == 'Maison'].copy(),
    'appartements': donnees_propres[donnees_propres['type_local'] == 'Appartement'].copy()
}

# On prepare la boucle de traitement
for type_bien, df_bien in datasets.items():
    if len(df_bien) < 50:
        print(f"\n--- IGNORÉ : Pas assez de donnees pour le type {type_bien} ---")
        continue

    print(f"\n" + "="*50)
    print(f"TRAITEMENT DU FLUX : {type_bien.upper()} ({len(df_bien)} biens)")
    print("="*50)

    # 5.1 Variables Spatiales Locales (Entre pairs)
    print("  -> Calcul des voisinages spécifiques au type de bien...")
    
    X = df_bien[features_base].copy()
    y = df_bien['log_prix_m2']

    coords_all = np.deg2rad(df_bien[['latitude', 'longitude']])
    prix_all = df_bien['prix_m2'].values
    arbre_voisins = BallTree(coords_all, metric='haversine')

    k_voisins = min(16, len(coords_all))
    _, idx_v = arbre_voisins.query(coords_all, k=k_voisins)
    
    # Prudence si k=1 (le bien est seul dans le secteur)
    if k_voisins > 1:
        voisins_prix = prix_all[idx_v[:, 1:]]
        X['prix_m2_voisins'] = np.median(voisins_prix, axis=1)
    else:
        X['prix_m2_voisins'] = prix_all

    rayon_rad = 1000 / RAYON_TERRE_METRES
    X['densite_ventes_1km'] = arbre_voisins.query_radius(coords_all, r=rayon_rad, count_only=True)

    med_section = df_bien.groupby('code_section')['prix_m2'].median()
    med_commune = df_bien.groupby('code_commune')['prix_m2'].median()
    med_globale = df_bien['prix_m2'].median()

    sec = df_bien['code_section']
    com = df_bien['code_commune']

    prix_sec = sec.map(med_section)
    prix_sec = prix_sec.fillna(com.map(med_commune))
    prix_sec = prix_sec.fillna(med_globale)
    X['prix_m2_section'] = prix_sec.values

    nb_ventes_section = df_bien.groupby('code_section').size()
    X['nb_ventes_section'] = df_bien['code_section'].map(nb_ventes_section).fillna(0).values

    features_finales = features_base + ['prix_m2_voisins', 'densite_ventes_1km', 'prix_m2_section', 'nb_ventes_section']
    features_finales = list(dict.fromkeys(features_finales))
    X = X[features_finales]

    # 6. ENTRAINEMENT DES 3 MODELES QUANTILES
    print("  -> Entrainement des modèles quantiles...")
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
        print(f"     * Modele '{nom}' entraine ({m.best_iteration + 1} arbres).")

    # 7. SAUVEGARDE DES ARTEFACTS SPECIFIQUES
    print("  -> Sauvegarde des artefacts...")
    
    # Modeles
    for nom, m in modeles.items():
        nom_fichier = f"modele_{type_bien}_{nom}.json"
        m.save_model(str(DOSSIER_MODELE / nom_fichier))

    # Features (identiques pour les deux, mais sauvegarde ecrasée/actualisée)
    with open(DOSSIER_MODELE / "features.json", "w") as f:
        json.dump(features_finales, f)

    # Contexte
    contexte = {
        'arbre_voisins_data': coords_all.values,
        'prix_all': prix_all,
        'med_section': med_section.to_dict(),
        'med_commune': med_commune.to_dict(),
        'med_globale': float(med_globale),
        'nb_ventes_section': nb_ventes_section.to_dict(),
        'stations': stations,
        'monuments': monuments,
        'hopitaux': hopitaux,
        'universites': universites,
        'profils_communes': donnees.drop_duplicates('code_commune').set_index('code_commune')[
            colonnes_dpe + colonnes_chauffage + colonnes_revenus
        ].to_dict('index'),
        'medianes_globales': {c: float(donnees[c].median()) for c in
                              colonnes_dpe + colonnes_chauffage + colonnes_revenus},
        'rayon_terre': RAYON_TERRE_METRES,
        'points_mec':extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT'] == 'Mer']),
        'points_lac':extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT'] == 'Lac']),
        'points_estuaire':extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT'] == 'Estuaire'])
    }
    
    with open(DOSSIER_MODELE / f"contexte_{type_bien}.pkl", "wb") as f:
        pickle.dump(contexte, f)

print("\n" + "="*50)
print(f"TERMINÉ. Artefacts sauvegardes dans : {DOSSIER_MODELE}")
print("Vos fichiers de production sont prêts à être intégrés dans votre API d'estimation !")