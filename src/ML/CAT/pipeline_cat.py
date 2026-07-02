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
    {'aav_nom': 'Genève', 'latitude': 46.2044, 'longitude': 6.1432, 'poids_aire': 200},
    {'aav_nom': 'Lausanne', 'latitude': 46.5197, 'longitude': 6.6323, 'poids_aire': 80},
])
poles = pd.concat([poles, poles_etrangers], ignore_index=True)

if dep_infra == 'FRANCE':
    query_monuments = "SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL;"
    query_hopitaux = "SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL;"
    query_universites = "SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;"
else:
    query_monuments = f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{dep_infra}';"
    query_hopitaux = f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal, 2) = '{dep_infra}';"
    query_universites = f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{dep_infra}';"

monuments = pd.read_sql(query_monuments, con=moteur)
hopitaux = pd.read_sql(query_hopitaux, con=moteur)
universites = pd.read_sql(query_universites, con=moteur)

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
    for col in colonnes_dpe:
        if col in donnees.columns:
            donnees[col] = donnees[col].fillna(donnees[col].median())
    colonnes_chauffage = ['pct_chauffage_elec','pct_chauffage_gaz','pct_chauffage_fioul','pct_chauffage_urbain']
    for col in colonnes_chauffage:
        if col in donnees.columns:
            donnees[col] = donnees[col].fillna(donnees[col].median())

    donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
    donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)

    plancher = max(donnees['prix_m2'].quantile(0.01), 800)
    plafond = min(donnees['prix_m2'].quantile(0.99), 15000)
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

    colonnes_revenus = ['median_revenu_disponible','indice_gini','pct_minima_sociaux']
    for col in colonnes_revenus:
        if col in donnees_propres.columns:
            donnees_propres[col] = donnees_propres[col].fillna(donnees_propres[col].median())

    colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
    colonnes_standard = ['surface_reelle_bati','volume_etudiants_proche',
                         'log_surface','surface_par_piece',
                         'surface_terrain','log_terrain','potentiel_urbain',
                         'median_revenu_disponible','indice_gini','pct_minima_sociaux']

    features = ['est_maison','latitude','longitude','nombre_pieces_principales','annee_vente','mois_vente','a_terrain'] \
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

    # --- prix_m2_voisins : pondere par distance ET filtre par surface comparable (version B) ---
    coords_train = np.deg2rad(donnees_propres.loc[X_train.index, ['latitude','longitude']])
    prix_train = donnees_propres.loc[X_train.index, 'prix_m2'].values
    surface_train = donnees_propres.loc[X_train.index, 'surface_reelle_bati'].values
    arbre_voisins = BallTree(coords_train, metric='haversine')

    def voisins_surface_ponderes(distances, indices, surface_bien, exclure_premier=False):
        if exclure_premier:
            distances = distances[1:]
            indices = indices[1:]
        if len(indices) == 0:
            return np.nan
        prix_v = prix_train[indices]
        surf_v = surface_train[indices]
        borne_bas, borne_haut = surface_bien * 0.6, surface_bien * 1.4
        masque = (surf_v >= borne_bas) & (surf_v <= borne_haut)
        if masque.sum() >= 3:
            d, p = distances[masque], prix_v[masque]
        else:
            d, p = distances, prix_v
        poids = 1.0 / (d + 1e-9)
        return np.sum(poids * p) / np.sum(poids)

    k_train = min(41, len(coords_train))
    dist_tr, idx_tr = arbre_voisins.query(coords_train, k=k_train)
    surface_bien_train = X_train['surface_reelle_bati'].values
    voisins_train = [
        voisins_surface_ponderes(dist_tr[i], idx_tr[i], surface_bien_train[i], exclure_premier=True)
        for i in range(len(idx_tr))
    ]

    k_test = min(40, len(coords_train))
    coords_test = np.deg2rad(donnees_propres.loc[X_test.index, ['latitude','longitude']])
    dist_te, idx_te = arbre_voisins.query(coords_test, k=k_test)
    surface_bien_test = X_test['surface_reelle_bati'].values
    voisins_test = [
        voisins_surface_ponderes(dist_te[i], idx_te[i], surface_bien_test[i], exclure_premier=False)
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

    df_tr = donnees_propres.loc[X_train.index]
    med_section = df_tr.groupby('code_section')['prix_m2'].median()
    med_commune = df_tr.groupby('code_commune')['prix_m2'].median()
    med_globale = df_tr['prix_m2'].median()

    def prix_section(idx):
        sec = donnees_propres.loc[idx, 'code_section']
        com = donnees_propres.loc[idx, 'code_commune']
        if sec in med_section.index:
            return med_section[sec]
        elif com in med_commune.index:
            return med_commune[com]
        else:
            return med_globale

    X_train['prix_m2_section'] = [prix_section(i) for i in X_train.index]
    X_test['prix_m2_section'] = [prix_section(i) for i in X_test.index]

    nb_ventes_section = df_tr.groupby('code_section').size()
    X_train['nb_ventes_section'] = [nb_ventes_section.get(donnees_propres.loc[i, 'code_section'], 0) for i in X_train.index]
    X_test['nb_ventes_section'] = [nb_ventes_section.get(donnees_propres.loc[i, 'code_section'], 0) for i in X_test.index]

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

    # --- Entrainement final + Duan (modele sur la moyenne) ---
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    modele = CatBoostRegressor(n_estimators=4000, learning_rate=0.05, max_depth=6, l2_leaf_reg=1.0,
                               early_stopping_rounds=50, random_state=42, verbose=False)
    modele.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    pred_val_log = modele.predict(X_val)
    residus_val = y_val.values - pred_val_log
    facteur_duan = np.mean(np.exp(residus_val))
    print(f"\nFacteur de correction de duan : {facteur_duan:.4f}")

    predictions_log = modele.predict(X_test)
    prix_reels_euros = np.exp(y_test)
    prix_predits_euros = np.exp(predictions_log) * facteur_duan

    # --- Sauvegarde resultats.pkl (un par type) ---
    dossier_graphes = dossier_base / suffixe_type
    dossier_graphes.mkdir(parents=True, exist_ok=True)

    resultats = {
        'modele': modele,
        'X_test': X_test,
        'y_test': y_test,
        'prix_reels_euros': prix_reels_euros,
        'prix_predits_euros': prix_predits_euros,
        'facteur_duan': facteur_duan,
        'features': features,
        'nom_zone': nom_zone,
        'type_bien': suffixe_type,
        'dossier_graphes': str(dossier_graphes),
        'profil_test': donnees_propres.loc[X_test.index, [
            'code_commune','prix_m2','surface_reelle_bati','surface_terrain',
            'type_local','latitude','longitude'
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
    r2_log = r2_score(y_test, predictions_log)
    r2_euros = r2_score(prix_reels_euros, prix_predits_euros)

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
    print("=" * 50)


# ==========================================
# BOUCLE SUR LES DEUX TYPES
# ==========================================
traiter_type("type_local = 'Maison'", "maisons")
traiter_type("type_local = 'Appartement'", "appartements")

print(f"\nTemps de traitement global : {time.time() - temps_total_debut:.2f} secondes.")