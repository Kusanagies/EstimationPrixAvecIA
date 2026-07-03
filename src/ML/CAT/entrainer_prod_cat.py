"""
PHASE 2 - ENTRAINEMENT ET SAUVEGARDE POUR LA PRODUCTION (MAISONS & APPARTEMENTS)
================================================================================
Version CatBoost.
Ce script reprend le pipeline d'evaluation mais :
  - n'utilise PAS de split temporel (entraine sur TOUTES les donnees, normal en prod)
  - separe completement les flux Maisons et Appartements
  - filtre les aberrations de prix PAR TYPE (coherent avec correlation.py)
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
from catboost import CatBoostRegressor
from sqlalchemy import create_engine
from dotenv import load_dotenv
import geopandas as gpd

# ==========================================
# 0. CONNEXION + CHOIX DE LA ZONE
# ==========================================
print("ENTRAINEMENT DU MODELE DE PRODUCTION - CATBOOST (MAISONS & APPARTEMENTS)")
print("-" * 50)

RACINE_PROJET = Path(__file__).resolve().parents[3]
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
    filtre_dvf = f"code_departement = '{departement}'"
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
      AND nombre_lots <= 3
      AND nombre_pieces_principales > 0
      AND {filtre_dvf}
      AND type_local IN ('Maison', 'Appartement');
""", con=moteur)

maisons_apparts = maisons_apparts.drop_duplicates(
    subset=['id_parcelle', 'prix_m2', 'surface_reelle_bati']
)

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

# Poles urbains (aires d'attraction) - national, pour le potentiel urbain
poles = pd.read_sql("""
    SELECT aav_nom, AVG(latitude) AS latitude, AVG(longitude) AS longitude, COUNT(*) AS poids_aire
    FROM referentiel_communes
    WHERE aav_nom IS NOT NULL AND aav_nom != 'SO' AND latitude IS NOT NULL
    GROUP BY aav_nom HAVING poids_aire >= 10
""", con=moteur)
poles_etrangers = pd.DataFrame([
    # --- Suisse (frontière est : Ain, Haute-Savoie, Doubs, Jura, Territoire de Belfort) ---
    {'aav_nom': 'Genève',    'latitude': 46.2044, 'longitude': 6.1432, 'poids_aire': 250},
    {'aav_nom': 'Lausanne',  'latitude': 46.5197, 'longitude': 6.6323, 'poids_aire': 90},
    {'aav_nom': 'Bâle',      'latitude': 47.5596, 'longitude': 7.5886, 'poids_aire': 150},
    {'aav_nom': 'Neuchâtel', 'latitude': 46.9925, 'longitude': 6.9310, 'poids_aire': 40},

    # --- Luxembourg (frontière nord-est : Moselle, Meurthe-et-Moselle) ---
    {'aav_nom': 'Luxembourg','latitude': 49.6116, 'longitude': 6.1319, 'poids_aire': 200},

    # --- Allemagne (frontière est : Bas-Rhin, Haut-Rhin, Moselle) ---
    {'aav_nom': 'Sarrebruck','latitude': 49.2402, 'longitude': 6.9969, 'poids_aire': 120},
    {'aav_nom': 'Karlsruhe', 'latitude': 49.0069, 'longitude': 8.4037, 'poids_aire': 130},
    {'aav_nom': 'Fribourg-en-Brisgau', 'latitude': 47.9990, 'longitude': 7.8421, 'poids_aire': 110},

    # --- Belgique (frontière nord : Nord, Ardennes, etc.) ---
    {'aav_nom': 'Bruxelles', 'latitude': 50.8503, 'longitude': 4.3517, 'poids_aire': 300},
    {'aav_nom': 'Charleroi', 'latitude': 50.4114, 'longitude': 4.4446, 'poids_aire': 90},
    {'aav_nom': 'Liège',     'latitude': 50.6326, 'longitude': 5.5797, 'poids_aire': 100},
    {'aav_nom': 'Mons',      'latitude': 50.4542, 'longitude': 3.9563, 'poids_aire': 50},

    # --- Italie (frontière sud-est : Alpes-Maritimes, Haute-Savoie, Savoie) ---
    {'aav_nom': 'Turin',     'latitude': 45.0703, 'longitude': 7.6869, 'poids_aire': 250},
    {'aav_nom': 'Vintimille','latitude': 43.7900, 'longitude': 7.6083, 'poids_aire': 30},

    # --- Monaco (frontière sud-est : Alpes-Maritimes) ---
    {'aav_nom': 'Monaco',    'latitude': 43.7384, 'longitude': 7.4246, 'poids_aire': 120},

    # --- Espagne (frontière sud-ouest : Pyrénées-Atlantiques, Pyrénées-Orientales) ---
    {'aav_nom': 'Barcelone', 'latitude': 41.3874, 'longitude': 2.1686, 'poids_aire': 300},
    {'aav_nom': 'Saint-Sébastien', 'latitude': 43.3183, 'longitude': -1.9812, 'poids_aire': 70},
    {'aav_nom': 'Gérone',    'latitude': 41.9794, 'longitude': 2.8214, 'poids_aire': 40},

    # --- Andorre (frontière sud : Ariège, Pyrénées-Orientales) ---
    {'aav_nom': 'Andorre-la-Vieille', 'latitude': 42.5063, 'longitude': 1.5218, 'poids_aire': 25},
])
poles = pd.concat([poles, poles_etrangers], ignore_index=True)

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

# Potentiel urbain (gravite : somme ponderee de l'influence des poles)
poles_rad = np.deg2rad(poles[['latitude', 'longitude']].values)
poids_poles = poles['poids_aire'].values.astype(float)
arbre_poles = BallTree(poles_rad, metric='haversine')
k_poles = min(20, len(poles))
dist_rad_p, idx_p = arbre_poles.query(points_rad, k=k_poles)
dist_m_p = dist_rad_p * RAYON_TERRE_METRES
donnees['potentiel_urbain'] = np.sum(poids_poles[idx_p] / (dist_m_p + 5000), axis=1)

# ==========================================
# 4. NETTOYAGE + FEATURES (filtrage prix repousse dans la boucle par type)
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

# Filtrage de SURFACE uniquement ici (le prix sera filtre par type dans la boucle)
donnees_propres = donnees[
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

# Correction du biais DVF pour les appartements (surface_terrain = 0)
mask_appart = donnees_propres['type_local'] == 'Appartement'
donnees_propres.loc[mask_appart, 'surface_terrain'] = 0

donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])
donnees_propres['log_surface'] = np.log(donnees_propres['surface_reelle_bati'])
donnees_propres['surface_par_piece'] = donnees_propres['surface_reelle_bati'] / donnees_propres['nombre_pieces_principales']
donnees_propres['a_terrain'] = (donnees_propres['surface_terrain'] > 0).astype(int)
donnees_propres['log_terrain'] = np.log1p(donnees_propres['surface_terrain'])
donnees_propres['code_section'] = donnees_propres['id_parcelle'].str[:10]

colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
colonnes_standard = ['surface_reelle_bati', 'volume_etudiants_proche',
                     'log_surface', 'surface_par_piece',
                     'surface_terrain', 'log_terrain', 'potentiel_urbain'] + colonnes_revenus

# Pas de 'est_maison' : chaque modele n'a qu'un seul type
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

for type_bien, df_bien in datasets.items():
    if len(df_bien) < 50:
        print(f"\n--- IGNORÉ : Pas assez de donnees pour le type {type_bien} ---")
        continue

    # Filtrage des aberrations de prix PAR TYPE (coherent avec correlation.py)
    plancher = max(df_bien['prix_m2'].quantile(0.01), 800)
    plafond = min(df_bien['prix_m2'].quantile(0.99), 15000)
    df_bien = df_bien[
        (df_bien['prix_m2'] >= plancher) & (df_bien['prix_m2'] <= plafond)
    ].copy()

    print("\n" + "=" * 50)
    print(f"TRAITEMENT DU FLUX : {type_bien.upper()} ({len(df_bien)} biens)")
    print("=" * 50)

    # 5.1 Variables spatiales locales (entre pairs du meme type)
    print("  -> Calcul des voisinages specifiques au type de bien...")

    X = df_bien[features_base].copy()
    y = df_bien['log_prix_m2']

    coords_all = np.deg2rad(df_bien[['latitude', 'longitude']])
    prix_all = df_bien['prix_m2'].values
    surface_all = df_bien['surface_reelle_bati'].values
    arbre_voisins = BallTree(coords_all, metric='haversine')

    # prix_m2_voisins : pondere par distance ET filtre par surface comparable (version B)
    def voisins_surface_ponderes(distances, indices, surface_bien):
        # On exclut le 1er voisin (le bien lui-meme, present dans coords_all)
        distances = distances[1:]
        indices = indices[1:]
        if len(indices) == 0:
            return np.nan
        prix_v = prix_all[indices]
        surf_v = surface_all[indices]
        borne_bas, borne_haut = surface_bien * 0.6, surface_bien * 1.4
        masque = (surf_v >= borne_bas) & (surf_v <= borne_haut)
        if masque.sum() >= 3:
            d, p = distances[masque], prix_v[masque]
        else:
            d, p = distances, prix_v
        poids = 1.0 / (d + 1e-9)
        return np.sum(poids * p) / np.sum(poids)

    k_voisins = min(41, len(coords_all))
    dist_v, idx_v = arbre_voisins.query(coords_all, k=k_voisins)
    if k_voisins > 1:
        X['prix_m2_voisins'] = [
            voisins_surface_ponderes(dist_v[i], idx_v[i], surface_all[i])
            for i in range(len(idx_v))
        ]
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

    # 6. ENTRAINEMENT DES 3 MODELES QUANTILES (CatBoost)
    print("  -> Entrainement des modeles quantiles (CatBoost)...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    quantiles = {'bas': 0.025, 'median': 0.50, 'haut': 0.975}
    modeles = {}

    for nom, alpha in quantiles.items():
        m = CatBoostRegressor(
            loss_function=f'Quantile:alpha={alpha}',
            iterations=1000, learning_rate=0.04, depth=8,
            l2_leaf_reg=3.0,
            random_seed=42, early_stopping_rounds=50, verbose=False
        )
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        modeles[nom] = m
        print(f"     * Modele '{nom}' entraine ({m.get_best_iteration() + 1} arbres).")

    # 7. SAUVEGARDE DES ARTEFACTS SPECIFIQUES
    print("  -> Sauvegarde des artefacts...")

    for nom, m in modeles.items():
        nom_fichier = f"modele_{type_bien}_{nom}.cbm"
        m.save_model(str(DOSSIER_MODELE / nom_fichier))

    with open(DOSSIER_MODELE / "features.json", "w") as f:
        json.dump(features_finales, f)

    contexte = {
        'arbre_voisins_data': coords_all.values,
        'prix_all': prix_all,
        'surface_all': surface_all,
        'poles_urbains': poles[['latitude', 'longitude', 'poids_aire']].values,
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
        'points_mer': extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT'] == 'Mer']),
        'points_lac': extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT'] == 'Lac']),
        'points_estuaire': extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT'] == 'Estuaire'])
    }

    with open(DOSSIER_MODELE / f"contexte_{type_bien}.pkl", "wb") as f:
        pickle.dump(contexte, f)

print("\n" + "=" * 50)
print(f"TERMINE. Artefacts sauvegardes dans : {DOSSIER_MODELE}")
print("Vos fichiers de production sont prets a etre integres dans votre API d'estimation !")