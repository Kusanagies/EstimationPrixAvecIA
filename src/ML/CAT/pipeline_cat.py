import time
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sqlalchemy import create_engine
import os
from pathlib import Path
from dotenv import load_dotenv
import pickle
import geopandas as gpd

# ==========================================
# 0. CONNEXION INITIALE ET MENU INTERACTIF
# ==========================================
print("-" * 50)
print("INITIALISATION DU MOTEUR D'ESTIMATION IMMOBILIERE")
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

# Saisie du departement ou de la France entiere
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

# ==========================================
# Configuration des filtres SQL (communs aux deux types)
# ==========================================
if departement == 'FRANCE':
    if choix_local == 'TOUS':
        filtre_dvf = "1=1"
        filtre_dpe = "1=1"
        dep_infra = "FRANCE"
        nom_zone_base = "France Entiere"
    else:
        filtre_dvf = f"code_commune = '{choix_local}'"
        filtre_dpe = f"code_insee_ban = '{choix_local}'"
        dep_infra = choix_local[:2]
        nom_zone_base = f"Secteur {choix_local}"
else:
    if choix_local == 'TOUS':
        filtre_dvf = f"code_departement = '{departement}'"
        filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
        dep_infra = departement
        nom_zone_base = f"Departement {departement}"
    else:
        filtre_dvf = f"code_commune = '{choix_local}'"
        filtre_dpe = f"code_insee_ban = '{choix_local}'"
        dep_infra = departement
        nom_zone_base = f"Secteur {choix_local}"

DOSSIER_OUT = RACINE_PROJET / "out"
if departement == 'FRANCE':
    if choix_local == 'TOUS':
        dossier_base = DOSSIER_OUT / "FRANCE"
    else:
        dossier_base = DOSSIER_OUT / choix_local[:2] / choix_local
else:
    if choix_local == 'TOUS':
        dossier_base = DOSSIER_OUT / departement
    else:
        dossier_base = DOSSIER_OUT / departement / choix_local

# ==========================================
# TELECHARGEMENT DES DONNEES COMMUNES (une seule fois)
# ==========================================
print("-" * 50)
print("Extraction des donnees communales (DPE, infra, revenus)...")
temps_total_debut = time.time()

# DPE
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

# Chargement NATIONAL (evite l'effet de bordure de departement)
monuments = pd.read_sql("SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL;", con=moteur)
hopitaux = pd.read_sql("SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL;", con=moteur)
universites = pd.read_sql("SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;", con=moteur)

if dep_infra == 'FRANCE':
    filtre_rev = "1=1"
elif choix_local != 'TOUS' and len(choix_local) == 5:
    filtre_rev = f"code_commune = '{choix_local}'"
else:
    filtre_rev = f"LEFT(code_commune,2) = '{dep_infra}'"

revenus = pd.read_sql(f"""
                        SELECT code_commune, median_revenu_disponible,indice_gini,pct_minima_sociaux
                        FROM demographie_communes
                        WHERE {filtre_rev};
                    """, con=moteur)
for col in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']:
    revenus[col] = pd.to_numeric(revenus[col], errors='coerce')

# Taux macro (mensuel, national) : credit immobilier fixe + inflation
taux = pd.read_sql("SELECT annee, mois, taux_credit_immo_fixe, taux_inflation FROM taux_macro", con=moteur) 

RAYON_TERRE_METRES = 6371000

def extraire_points_contour(sous_gdf):
    points = []
    for geom in sous_gdf.geometry:
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                points.extend(list(poly.exterior.coords))
        else:
            points.extend(list(geom.exterior.coords))
    if not points:
        return pd.DataFrame(columns=['latitude','longitude'])
    pts = np.array(points)
    return pd.DataFrame(pts[:, [1, 0]], columns=['latitude','longitude'])

# ==========================================
# FONCTION DE TRAITEMENT D'UN TYPE DE BIEN
# ==========================================
def traiter_type(filtre_type, suffixe_type):
    nom_zone = f"{nom_zone_base} ({suffixe_type})"
    print("\n" + "#" * 60)
    print(f"#  TRAITEMENT : {nom_zone}")
    print("#" * 60)

    # --- Extraction DVF (filtre <= 2 lots, decision validee) ---
    maisons = pd.read_sql(f"""
        SELECT code_commune, id_parcelle, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2,
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
          AND {filtre_type}
          AND id_mutation IN (
              SELECT id_mutation FROM valeurs_foncieres
              WHERE surface_reelle_bati > 0
              GROUP BY id_mutation HAVING COUNT(*) = 1)
    """, con=moteur)

    maisons = maisons.drop_duplicates(subset=['id_parcelle','prix_m2','surface_reelle_bati'])
    if len(maisons) == 0:
        print(f"  Aucune donnee pour {suffixe_type}, on passe.")
        return

    # Correction biais DVF : terrain des appartements force a 0
    if suffixe_type == 'appartements':
        maisons['surface_terrain'] = 0

    # --- Fusion ---
    donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
    donnees = pd.merge(donnees, revenus, on='code_commune', how='left')

    # --- Fusion ---
    donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
    donnees = pd.merge(donnees, revenus, on='code_commune', how='left')

    # Jointure des taux macro par annee + mois de la vente
    donnees = pd.merge(donnees, taux,
                       left_on=['annee_vente', 'mois_vente'],
                       right_on=['annee', 'mois'], how='left')
    donnees = donnees.drop(columns=['annee', 'mois'], errors='ignore')

    # --- Distances spatiales ---
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

    classements = {'Mer':'dist_mer_m', 'Lac':'dist_lac_m', 'Estuaire':'dist_estuaire_m'}
    for classement, nom_colonne in classements.items():
        sous = gdf_littoral[gdf_littoral['CLASSEMENT'] == classement]
        df_points = extraire_points_contour(sous)
        calculer_distance_min(df_points, nom_colonne)

    # Potentiel urbain (gravite : influence ponderee des poles)
    poles_rad = np.deg2rad(poles[['latitude', 'longitude']].values)
    poids_poles = poles['poids_aire'].values.astype(float)
    arbre_poles = BallTree(poles_rad, metric='haversine')
    k_poles = min(20, len(poles))
    dist_rad_p, idx_p = arbre_poles.query(maisons_rad, k=k_poles)
    dist_m_p = dist_rad_p * RAYON_TERRE_METRES
    donnees['potentiel_urbain'] = np.sum(poids_poles[idx_p] / (dist_m_p + 5000), axis=1)

    # --- Nettoyage et feature engineering ---
    colonnes_dpe = ['pct_dpe_A','pct_dpe_B','pct_dpe_C','pct_dpe_D','pct_dpe_E','pct_dpe_F','pct_dpe_G']
    colonnes_chauffage = ['pct_chauffage_elec','pct_chauffage_gaz','pct_chauffage_fioul','pct_chauffage_urbain']

    donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
    donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)

    # plancher = max(donnees['prix_m2'].quantile(0.01), 800)
    # plafond = min(donnees['prix_m2'].quantile(0.99), 15000)

    plancher = donnees['prix_m2'].quantile(0.01)
    plafond = donnees['prix_m2'].quantile(0.99)
    
    donnees_propres = donnees[
        (donnees['prix_m2'] >= plancher) & (donnees['prix_m2'] <= plafond) &
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
    colonnes_standard = ['surface_reelle_bati','volume_etudiants_proche',
                         'log_surface','surface_par_piece',
                         'surface_terrain','log_terrain','potentiel_urbain',
                         'median_revenu_disponible','indice_gini','pct_minima_sociaux']

    features = ['est_maison','latitude','longitude','nombre_pieces_principales','annee_vente','mois_vente','a_terrain',
                'taux_credit_immo_fixe','taux_inflation'] \
               + colonnes_dpe + colonnes_chauffage + colonnes_standard + colonnes_dist

    X = donnees_propres[features]
    y = donnees_propres['log_prix_m2']

    annee_max = donnees_propres['annee_vente'].max()
    train_mask = donnees_propres['annee_vente'] < annee_max
    test_mask = donnees_propres['annee_vente'] == annee_max

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("  (une seule annee disponible -> repli sur split aleatoire)")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    # ---- Indice de marche (train uniquement) : actualisation des prix ----
    annees_train_serie = donnees_propres.loc[X_train.index, 'annee_vente']
    idx_marche = donnees_propres.loc[X_train.index].groupby('annee_vente')['prix_m2'].median()
    ref_marche = idx_marche.loc[idx_marche.index.max()]
    coef_marche = (ref_marche / idx_marche).to_dict()

    coords_train = np.deg2rad(donnees_propres.loc[X_train.index, ['latitude','longitude']])
    prix_train = donnees_propres.loc[X_train.index, 'prix_m2'].values
    prix_train_actu = prix_train * annees_train_serie.map(coef_marche).values
    surface_train = donnees_propres.loc[X_train.index, 'surface_reelle_bati'].values
    arbre_voisins = BallTree(coords_train, metric='haversine')

    def voisins_surface_ponderes(distances_rad, indices, surface_bien, idx_self=None):
        # Exclusion du bien lui-meme par identite d'indice (et non "le premier")
        if idx_self is not None:
            garder = indices != idx_self
            distances_rad, indices = distances_rad[garder], indices[garder]
        if len(indices) == 0:
            return np.nan
        dist_m = distances_rad * RAYON_TERRE_METRES   # radians -> metres
        prix_v = prix_train_actu[indices]
        surf_v = surface_train[indices]
        borne_bas, borne_haut = surface_bien * 0.6, surface_bien * 1.4
        masque = (surf_v >= borne_bas) & (surf_v <= borne_haut)
        if masque.sum() >= 3:
            d, p = dist_m[masque], prix_v[masque]
        else:
            d, p = dist_m, prix_v
        poids = 1.0 / (d + 50.0)   # plancher 50 m
        return np.sum(poids * p) / np.sum(poids)

    k_train = min(41, len(coords_train))
    dist_tr, idx_tr = arbre_voisins.query(coords_train, k=k_train)
    surface_bien_train = X_train['surface_reelle_bati'].values
    voisins_train = [
        voisins_surface_ponderes(dist_tr[i], idx_tr[i], surface_bien_train[i], idx_self=i)
        for i in range(len(idx_tr))
    ]

    k_test = min(40, len(coords_train))
    coords_test = np.deg2rad(donnees_propres.loc[X_test.index, ['latitude','longitude']])
    dist_te, idx_te = arbre_voisins.query(coords_test, k=k_test)
    surface_bien_test = X_test['surface_reelle_bati'].values
    voisins_test = [
        voisins_surface_ponderes(dist_te[i], idx_te[i], surface_bien_test[i])
        for i in range(len(idx_te))
    ]

    rayon_rad = 1000 / RAYON_TERRE_METRES
    dens_train = arbre_voisins.query_radius(coords_train, r=rayon_rad, count_only=True)
    dens_test = arbre_voisins.query_radius(coords_test, r=rayon_rad, count_only=True)

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['densite_ventes_1km'] = dens_train
    X_test['densite_ventes_1km'] = dens_test
    X_train['prix_m2_voisins'] = voisins_train
    X_test['prix_m2_voisins'] = voisins_test

    df_tr = donnees_propres.loc[X_train.index].copy()
    df_tr['prix_m2_actu'] = df_tr['prix_m2'].values * df_tr['annee_vente'].map(coef_marche).values

    med_commune = df_tr.groupby('code_commune')['prix_m2_actu'].median()
    med_globale = df_tr['prix_m2_actu'].median()

    # --- TRAIN : encodage out-of-fold (une ligne ne voit jamais son propre prix) ---
    vals_train = pd.Series(np.nan, index=X_train.index)
    kf_enc = KFold(n_splits=5, shuffle=True, random_state=42)
    for pos_fit, pos_oof in kf_enc.split(df_tr):
        med_s = df_tr.iloc[pos_fit].groupby('code_section')['prix_m2_actu'].median()
        med_c = df_tr.iloc[pos_fit].groupby('code_commune')['prix_m2_actu'].median()
        sous = df_tr.iloc[pos_oof]
        v = sous['code_section'].map(med_s)
        v = v.fillna(sous['code_commune'].map(med_c))
        v = v.fillna(df_tr.iloc[pos_fit]['prix_m2_actu'].median())
        vals_train.iloc[pos_oof] = v.values
    X_train['prix_m2_section'] = vals_train.values

    # --- TEST : medianes sur tout le train ---
    med_section = df_tr.groupby('code_section')['prix_m2_actu'].median()
    sec_te = donnees_propres.loc[X_test.index, 'code_section'].map(med_section)
    com_te = donnees_propres.loc[X_test.index, 'code_commune'].map(med_commune)
    X_test['prix_m2_section'] = sec_te.fillna(com_te).fillna(med_globale).values

    nb_ventes_section = df_tr.groupby('code_section').size()
    X_train['nb_ventes_section'] = donnees_propres.loc[X_train.index, 'code_section'].map(nb_ventes_section).fillna(0).values
    X_test['nb_ventes_section'] = donnees_propres.loc[X_test.index, 'code_section'].map(nb_ventes_section).fillna(0).values
    features = features + ['densite_ventes_1km', 'prix_m2_voisins', 'prix_m2_section', 'nb_ventes_section']
    features = list(dict.fromkeys(features))
    X_train = X_train[features]
    X_test = X_test[features]

    # --- Validation croisee ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    modele_cv = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=1.0, random_seed=42, verbose=False)
    cv = cross_validate(modele_cv, X_train, y_train, cv=kf, scoring='r2', return_train_score=True)

    print("\n" + "=" * 50)
    print(f"VALIDATION CROISEE (5 plis) - {suffixe_type}")
    print("=" * 50)
    print(f"R2 train moyen      : {cv['train_score'].mean():.3f}")
    print(f"R2 validation moyen : {cv['test_score'].mean():.3f}")
    print(f"Ecart train-valid   : {(cv['train_score'].mean() - cv['test_score'].mean()):.3f}")
    print(f"Stabilite (ecart-type valid)    : {cv['test_score'].std():.3f}")
    print("=" * 50)

    mask_val = (annees_train_serie == annees_train_serie.max()).values
    if mask_val.sum() < 50 or (~mask_val).sum() < 200:
        # repli si trop peu de donnees pour un split temporel
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    else:
        X_tr, y_tr = X_train[~mask_val], y_train[~mask_val]
        X_val, y_val = X_train[mask_val], y_train[mask_val]
    quantiles = {'bas': 0.025, 'median': 0.50, 'haut': 0.975}

    modeles_q = {}
    for nom_q, alpha in quantiles.items():
        m = CatBoostRegressor(
            loss_function=f'Quantile:alpha={alpha}',
            iterations=1000, learning_rate=0.04, depth=8, l2_leaf_reg=1.0,
            early_stopping_rounds=50, random_seed=42, verbose=False
        )
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        modeles_q[nom_q] = m

    # Le modele median sert de reference (SHAP, courbe, arret)
    modele = modeles_q['median']

    pred_bas_log = modeles_q['bas'].predict(X_test)
    pred_med_log = modeles_q['median'].predict(X_test)
    pred_haut_log = modeles_q['haut'].predict(X_test)

    prix_reels_euros = np.exp(y_test)
    prix_bas = np.exp(pred_bas_log)
    prix_predits_euros = np.exp(pred_med_log)
    prix_haut = np.exp(pred_haut_log)

    b = np.minimum(prix_bas, prix_haut)
    h = np.maximum(prix_bas, prix_haut)
    prix_bas, prix_haut = b, h


    # --- Sauvegarde resultats.pkl (un par type) ---
    dossier_graphes = dossier_base / suffixe_type
    dossier_graphes.mkdir(parents=True, exist_ok=True)

    resultats = {
        'modeles_q': modeles_q,
        'modele': modele,
        'X_test': X_test,
        'y_test': y_test,
        'prix_reels_euros': prix_reels_euros,
        'prix_predits_euros': prix_predits_euros,
        'prix_bas': prix_bas,
        'prix_haut': prix_haut,
        'features': features,
        'nom_zone': nom_zone,
        'type_bien': suffixe_type,
        'dossier_graphes': str(dossier_graphes),
        'profil_test': donnees_propres.loc[X_test.index, [
            'code_commune', 'prix_m2', 'surface_reelle_bati', 'surface_terrain',
            'type_local', 'latitude', 'longitude'
        ]],
    }
    with open(dossier_graphes / "resultats.pkl", "wb") as f:
        pickle.dump(resultats, f)
    print(f"Resultats sauvegardes dans {dossier_graphes / 'resultats.pkl'}")

    # --- Metriques ---
    mae = mean_absolute_error(prix_reels_euros, prix_predits_euros)
    mape = np.mean(np.abs((prix_reels_euros - prix_predits_euros) / prix_reels_euros)) * 100
    erreur_mediane = np.median(np.abs(prix_reels_euros - prix_predits_euros))
    rmse = np.sqrt(np.mean((prix_reels_euros.values - prix_predits_euros) ** 2))
    erreur_rel = np.abs(prix_reels_euros.values - prix_predits_euros) / prix_reels_euros
    pct_10 = np.mean(erreur_rel <= 0.10) * 100
    pct_20 = np.mean(erreur_rel <= 0.20) * 100
    r2_log = r2_score(y_test,pred_med_log)
    r2_euros = r2_score(prix_reels_euros, prix_predits_euros)

    dans_intervalle = (prix_reels_euros.values >= prix_bas) & (prix_reels_euros.values <= prix_haut)
    couverture = np.mean(dans_intervalle) * 100
    largeur_moyenne = np.mean(prix_haut - prix_bas)

    print("\n" + "=" * 50)
    print(f"RAPPORT DE PERFORMANCE CatBoost - {nom_zone.upper()}")
    print("=" * 50)
    print(f"Nombre de logements pour l'apprentissage : {len(X_train)}")
    print(f"Nombre de logements pour la validation   : {len(X_test)}")
    print("-" * 50)
    print(f"R2 (espace log)                    : {r2_log * 100:.2f} %")
    print(f"R2 (euros / m2)                    : {r2_euros * 100:.2f} %")
    print(f"Erreur absolue moyenne (MAE)             : {mae:.2f} EUR / m2")
    print(f"*******************************************************")
    print(f"MAPE (erreur moyenne %)     : {mape:.1f}%")
    print(f"Erreur médiane              : {erreur_mediane:.0f} EUR/m²")
    print(f"RMSE                        : {rmse:.0f} EUR/m²")
    print(f"Prédictions à plus ou moins 10 % du réel : {pct_10:.1f} %")
    print(f"Prédiction à plus ou moins 20 % du réel  : {pct_20:.1f} %")
    print(f"Couverture intervalle 95% : {couverture:.1f} %")
    print(f"Largeur moyenne intervalle : {largeur_moyenne:.0f} EUR/m2")
    print("=" * 50)


# ==========================================
# BOUCLE SUR LES DEUX TYPES
# ==========================================
traiter_type("type_local = 'Maison'", "maisons")
traiter_type("type_local = 'Appartement'", "appartements")

print(f"\nTemps de traitement global : {time.time() - temps_total_debut:.2f} secondes.")