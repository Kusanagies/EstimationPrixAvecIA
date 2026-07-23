"""
EVALUATION CATBOOST - VERSION BASE etalab_dvf (table synthese)
===============================================================
Adaptation de correlation_cat.py pour la nouvelle source de donnees :
  - Les VENTES viennent de etalab_dvf.synthese (pre-agregee, propre, 2014-2025)
  - Les ENRICHISSEMENTS (DPE, revenus, infra, taux...) restent dans EstimationIA
  - PAS de filtre lots (le test a montre qu'il est inutile : synthese a deja
    demultiplexe les ventes complexes ; A/B identiques, C degrade)

Mapping des colonnes synthese -> ancien schema :
  lat/lng -> latitude/longitude | typebien -> type_local
  communes_code -> code_commune | parcelles_code -> id_parcelle
  surface -> surface_reelle_bati | nb_pieces -> nombre_pieces_principales
  prix_m2 deja calcule | date -> annee_vente/mois_vente
"""

import time, sys, os, shap
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import geopandas as gpd

# ==========================================
# 0. CONNEXIONS (DEUX BASES) ET MENU
# ==========================================
print("-" * 50)
print("EVALUATION CATBOOST - SOURCE etalab_dvf.synthese")
print("-" * 50)

CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"
gdf_littoral = gpd.read_file(CHEMIN_GPKG)

RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
    # Base des VENTES (nouvelle)
    moteur_dvf = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    # Base des ENRICHISSEMENTS (ancienne)
    moteur_enr = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur_dvf.connect().close()
    moteur_enr.connect().close()
except KeyError:
    print("Erreur : DB_PASS introuvable dans .env"); sys.exit()
except Exception as e:
    print(f"Erreur de connexion : {e}"); sys.exit()

# ==========================================
# 0bis. CHOIX INTERACTIF DES FEATURES
# ==========================================
def demander(question, defaut=True):
    ind = "O/n" if defaut else "o/N"
    while True:
        r = input(f"  {question} ({ind}) : ").strip().lower()
        if r == "": return defaut
        if r.startswith('o'): return True
        if r.startswith('n'): return False
        print("    Tapez 'o' ou 'n'.")

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
FA['date']        = demander("Date (annee, mois) ?", True)
print("\n--- Enrichissements communaux ---")
FA['dpe']         = demander("Profil DPE ?", True)
FA['chauffage']   = demander("Profil chauffage ?", True)
FA['revenus']     = demander("Revenus / Gini / minima ?", True)
print("\n--- Distances ---")
FA['dist_transport']  = demander("Distance gare ?", True)
FA['dist_monument']   = demander("Distance monument ?", True)
FA['dist_hopital']    = demander("Distance hopital ?", True)
FA['dist_universite'] = demander("Distance universite ?", True)
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

departement = input("\nDepartement (ex: 34, 75) ou 'FRANCE' : ").strip().upper()
if len(departement) < 2:
    print("Format invalide."); sys.exit()

if departement == 'FRANCE':
    filtre_dvf = "1=1"; filtre_dpe = "1=1"; dep_infra = "FRANCE"; nom_zone = "France"
else:
    filtre_dvf = f"departements_code = '{departement}'"
    filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
    dep_infra = departement; nom_zone = f"Departement {departement}"

DOSSIER_OUT = RACINE_PROJET / "out"
dossier_graphes = DOSSIER_OUT / (departement if departement != 'FRANCE' else 'FRANCE') / "synthese"
dossier_graphes.mkdir(parents=True, exist_ok=True)

print(f"\nZone : {nom_zone}")
temps_debut = time.time()

# ==========================================
# 1. EXTRACTION DES VENTES (depuis synthese)
# ==========================================
print("Etape 1/4 : Extraction des ventes (synthese)...")

# PAS de filtre lots (inutile d'apres le test). Filtres : surface, prix, pieces.
maisons = pd.read_sql(f"""
    SELECT id,
           communes_code AS code_commune,
           parcelles_code AS id_parcelle,
           lat AS latitude,
           lng AS longitude,
           prix_m2,
           surface AS surface_reelle_bati,
           typebien AS type_local,
           nb_pieces AS nombre_pieces_principales,
           surface_terrain,
           nb_dependances,
           valeur_fonciere,
           YEAR(date) AS annee_vente,
           MONTH(date) AS mois_vente
    FROM synthese
    WHERE {filtre_dvf}
      AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
      AND typebien IN ('maison', 'appartement');
""", con=moteur_dvf)

# Conversion explicite des types (synthese renvoie du 'object' a cause des NULL SQL)
colonnes_num = ['prix_m2', 'surface_reelle_bati', 'surface_terrain',
                'nombre_pieces_principales', 'nb_dependances', 'valeur_fonciere',
                'latitude', 'longitude', 'annee_vente', 'mois_vente']
for col in colonnes_num:
    maisons[col] = pd.to_numeric(maisons[col], errors='coerce')
maisons = maisons.dropna(subset=['prix_m2', 'surface_reelle_bati', 'nombre_pieces_principales',
                                 'latitude', 'longitude'])

# Harmonisation du type (synthese : 'maison'/'appartement' -> 'Maison'/'Appartement')
maisons['type_local'] = maisons['type_local'].str.capitalize()

if len(maisons) == 0:
    print("Aucune donnee."); sys.exit()
print(f"  {len(maisons):,} ventes extraites.")

# ==========================================
# 1bis. ENRICHISSEMENTS (depuis EstimationIA)
# ==========================================
print("Etape 1bis/4 : Chargement des enrichissements (EstimationIA)...")

dpe = pd.read_sql(f"""
    SELECT code_insee_ban,
           (SUM(CASE WHEN etiquette_dpe='A' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_dpe_A,
           (SUM(CASE WHEN etiquette_dpe='B' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_dpe_B,
           (SUM(CASE WHEN etiquette_dpe='C' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_dpe_C,
           (SUM(CASE WHEN etiquette_dpe='D' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_dpe_D,
           (SUM(CASE WHEN etiquette_dpe='E' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_dpe_E,
           (SUM(CASE WHEN etiquette_dpe='F' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_dpe_F,
           (SUM(CASE WHEN etiquette_dpe='G' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_dpe_G,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Électricité%%' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_chauffage_elec,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Gaz%%' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_chauffage_gaz,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Fioul%%' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_chauffage_fioul,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Réseau de Chaleur%%' THEN 1 ELSE 0 END)/COUNT(*))*100 AS pct_chauffage_urbain
    FROM dpe_logements_france
    WHERE etiquette_dpe IN ('A','B','C','D','E','F','G') AND {filtre_dpe}
    GROUP BY code_insee_ban;
""", con=moteur_enr)

stations = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur_enr)
if dep_infra == 'FRANCE':
    q_mon = "SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL;"
    q_hop = "SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL;"
    q_uni = "SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;"
else:
    q_mon = f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{dep_infra}';"
    q_hop = f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal,2)='{dep_infra}';"
    q_uni = f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{dep_infra}';"
monuments = pd.read_sql(q_mon, con=moteur_enr)
hopitaux = pd.read_sql(q_hop, con=moteur_enr)
universites = pd.read_sql(q_uni, con=moteur_enr)

filtre_rev = "1=1" if dep_infra == 'FRANCE' else f"LEFT(code_commune,2)='{dep_infra}'"
revenus = pd.read_sql(f"SELECT code_commune, median_revenu_disponible, indice_gini, pct_minima_sociaux FROM demographie_communes WHERE {filtre_rev};", con=moteur_enr)
for col in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']:
    revenus[col] = pd.to_numeric(revenus[col], errors='coerce')

poles = None
if FA['potentiel_urbain']:
    poles = pd.read_sql("""
        SELECT aav_nom, AVG(latitude) AS latitude, AVG(longitude) AS longitude, COUNT(*) AS poids_aire
        FROM referentiel_communes
        WHERE aav_nom IS NOT NULL AND aav_nom != 'SO' AND latitude IS NOT NULL
        GROUP BY aav_nom HAVING poids_aire >= 10
    """, con=moteur_enr)
    poles_etrangers = pd.DataFrame([
        {'aav_nom':'Genève','latitude':46.2044,'longitude':6.1432,'poids_aire':250},
        {'aav_nom':'Lausanne','latitude':46.5197,'longitude':6.6323,'poids_aire':90},
        {'aav_nom':'Bâle','latitude':47.5596,'longitude':7.5886,'poids_aire':150},
        {'aav_nom':'Luxembourg','latitude':49.6116,'longitude':6.1319,'poids_aire':200},
        {'aav_nom':'Bruxelles','latitude':50.8503,'longitude':4.3517,'poids_aire':300},
        {'aav_nom':'Monaco','latitude':43.7384,'longitude':7.4246,'poids_aire':120},
        {'aav_nom':'Turin','latitude':45.0703,'longitude':7.6869,'poids_aire':250},
        {'aav_nom':'Barcelone','latitude':41.3874,'longitude':2.1686,'poids_aire':300},
    ])
    poles = pd.concat([poles, poles_etrangers], ignore_index=True)
    for c in ['latitude','longitude','poids_aire']:
        poles[c] = pd.to_numeric(poles[c], errors='coerce')
    poles = poles.dropna(subset=['latitude','longitude','poids_aire'])

pib = None
if FA['pib']:
    pib = pd.read_sql("SELECT annee, pib_national FROM pib_national", con=moteur_enr)
chomage = None
if FA['chomage']:
    chomage = pd.read_sql("SELECT code_departement, annee, trimestre, taux_chomage FROM chomage_departements", con=moteur_enr)
taux = None
if FA['taux_credit'] or FA['taux_inflation']:
    taux = pd.read_sql("SELECT annee, mois, taux_credit_immo_fixe, taux_inflation FROM taux_macro", con=moteur_enr)

# ==========================================
# 2. FUSION ET DISTANCES
# ==========================================
print("Etape 2/4 : Fusion et distances...")
donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees, revenus, on='code_commune', how='left')

donnees['code_departement'] = donnees['code_commune'].str[:2]
donnees['trimestre'] = (donnees['mois_vente'] - 1) // 3 + 1

if pib is not None:
    donnees = pd.merge(donnees, pib, left_on='annee_vente', right_on='annee', how='left').drop(columns=['annee'], errors='ignore')
if chomage is not None:
    donnees = pd.merge(donnees, chomage, left_on=['code_departement','annee_vente','trimestre'],
                       right_on=['code_departement','annee','trimestre'], how='left').drop(columns=['annee'], errors='ignore')
if taux is not None:
    donnees = pd.merge(donnees, taux, left_on=['annee_vente','mois_vente'],
                       right_on=['annee','mois'], how='left').drop(columns=['annee','mois'], errors='ignore')

RAYON = 6371000
points_rad = np.deg2rad(donnees[['latitude','longitude']])

def dist_min(df_points, col):
    if len(df_points) > 0:
        arbre = BallTree(np.deg2rad(df_points.iloc[:,0:2]), metric='haversine')
        d,_ = arbre.query(points_rad, k=1)
        donnees[col] = d.flatten()*RAYON
    else:
        donnees[col] = 999999

if FA['dist_transport']: dist_min(stations, 'dist_transport_m')
if FA['dist_monument']:  dist_min(monuments, 'dist_monument_m')
if FA['dist_hopital']:   dist_min(hopitaux, 'dist_hopital_m')

if FA['dist_universite']:
    if len(universites) > 0:
        au = BallTree(np.deg2rad(universites[['latitude','longitude']]), metric='haversine')
        d, iu = au.query(points_rad, k=1)
        donnees['dist_universite_m'] = d.flatten()*RAYON
        donnees['volume_etudiants_proche'] = universites.iloc[iu.flatten()]['nombre_etudiants'].values
    else:
        donnees['dist_universite_m'] = 999999; donnees['volume_etudiants_proche'] = 0
else:
    donnees['volume_etudiants_proche'] = 0

def contour(sg):
    pts=[]
    for g in sg.geometry:
        if g.geom_type=='MultiPolygon':
            for p in g.geoms: pts.extend(list(p.exterior.coords))
        else: pts.extend(list(g.exterior.coords))
    if not pts: return pd.DataFrame(columns=['latitude','longitude'])
    a=np.array(pts); return pd.DataFrame(a[:,[1,0]], columns=['latitude','longitude'])

if FA['dist_littoral']:
    for cl,col in {'Mer':'dist_mer_m','Lac':'dist_lac_m','Estuaire':'dist_estuaire_m'}.items():
        dist_min(contour(gdf_littoral[gdf_littoral['CLASSEMENT']==cl]), col)

if FA['potentiel_urbain'] and poles is not None and len(poles) > 0:
    ap = BallTree(np.deg2rad(poles[['latitude','longitude']].values), metric='haversine')
    pp = poles['poids_aire'].values.astype(float)
    drp, ip = ap.query(points_rad, k=min(20,len(poles)))
    donnees['potentiel_urbain'] = np.sum(pp[ip]/(drp*RAYON+5000), axis=1)

# ==========================================
# 3. NETTOYAGE + FEATURES
# ==========================================
print("Etape 3/4 : Nettoyage et feature engineering...")
colonnes_dpe = ['pct_dpe_A','pct_dpe_B','pct_dpe_C','pct_dpe_D','pct_dpe_E','pct_dpe_F','pct_dpe_G']
colonnes_chauffage = ['pct_chauffage_elec','pct_chauffage_gaz','pct_chauffage_fioul','pct_chauffage_urbain']
colonnes_revenus = ['median_revenu_disponible','indice_gini','pct_minima_sociaux']

for col in colonnes_dpe + colonnes_chauffage + colonnes_revenus:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())
donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)
for col in ['pib_national','taux_chomage','taux_credit_immo_fixe','taux_inflation','potentiel_urbain']:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

dp = donnees[(donnees['surface_reelle_bati']>=9) & (donnees['surface_reelle_bati']<=300)].copy()
dp.loc[dp['type_local']=='Appartement', 'surface_terrain'] = 0
dp['log_prix_m2'] = np.log(dp['prix_m2'])
# CIBLE = PRIX TOTAL (le modele predit directement des euros, pas du EUR/m2)
dp['prix_total'] = dp['prix_m2'] * dp['surface_reelle_bati']
dp['log_prix_total'] = np.log(dp['prix_total'])
dp['log_surface'] = np.log(dp['surface_reelle_bati'])
dp['surface_par_piece'] = dp['surface_reelle_bati'] / dp['nombre_pieces_principales']
dp['a_terrain'] = (dp['surface_terrain']>0).astype(int)
dp['log_terrain'] = np.log1p(dp['surface_terrain'])
dp['code_section'] = dp['id_parcelle'].str[:10]

def construire_features_base():
    f = []
    if FA['geo_base']:   f += ['latitude','longitude']
    if FA['pieces']:     f += ['nombre_pieces_principales']
    if FA['date']:       f += ['annee_vente','mois_vente']
    if FA['terrain']:    f += ['a_terrain','surface_terrain','log_terrain']
    if FA['surface']:    f += ['surface_reelle_bati','log_surface','surface_par_piece']
    if FA['dist_universite']: f += ['volume_etudiants_proche']
    if FA['revenus']:    f += colonnes_revenus
    if FA['potentiel_urbain'] and 'potentiel_urbain' in dp.columns: f += ['potentiel_urbain']
    if FA['dpe']:        f += colonnes_dpe
    if FA['chauffage']:  f += colonnes_chauffage
    if FA['dist_transport']:  f += ['dist_transport_m']
    if FA['dist_monument']:   f += ['dist_monument_m']
    if FA['dist_hopital']:    f += ['dist_hopital_m']
    if FA['dist_universite']: f += ['dist_universite_m']
    if FA['dist_littoral']:   f += ['dist_mer_m','dist_lac_m','dist_estuaire_m']
    if FA['pib'] and 'pib_national' in dp.columns: f += ['pib_national']
    if FA['chomage'] and 'taux_chomage' in dp.columns: f += ['taux_chomage']
    if FA['taux_credit'] and 'taux_credit_immo_fixe' in dp.columns: f += ['taux_credit_immo_fixe']
    if FA['taux_inflation'] and 'taux_inflation' in dp.columns: f += ['taux_inflation']
    return list(dict.fromkeys(f))

features_base = construire_features_base()
print(f"  {len(features_base)} features de base : {features_base}")

# ==========================================
# 4. BOUCLE PAR TYPE
# ==========================================
print("Etape 4/4 : Entrainement et evaluation par type...")

datasets = {
    'maisons': dp[dp['type_local']=='Maison'].copy(),
    'appartements': dp[dp['type_local']=='Appartement'].copy()
}

for type_bien, df_bien in datasets.items():
    if len(df_bien) < 50:
        print(f"\n--- {type_bien} ignore (pas assez de donnees) ---"); continue

    plancher = df_bien['prix_m2'].quantile(0.01)
    plafond = df_bien['prix_m2'].quantile(0.99)
    df_bien = df_bien[(df_bien['prix_m2']>=plancher) & (df_bien['prix_m2']<=plafond)].copy()

    # --- Filtre de coherence marche : elimine les transactions hors marche ---
    # (ventes familiales sous-evaluees, parts indivises, nue-propriete) que les
    # quantiles departementaux laissent passer. Compare chaque bien au prix
    # median de SA commune ; garde ceux entre 40% et 250% de cette reference.
    stats_com = df_bien.groupby('code_commune')['prix_m2'].agg(['median', 'size'])
    ref_com = df_bien['code_commune'].map(stats_com['median'])
    n_com = df_bien['code_commune'].map(stats_com['size'])
    ref_com = ref_com.where(n_com >= 10, df_bien['prix_m2'].median())  # repli si commune trop petite
    ratio = df_bien['prix_m2'] / ref_com
    nb_avant = len(df_bien)
    df_bien = df_bien[ratio.between(0.40, 2.50)].copy()
    print(f"  Filtre coherence marche : {nb_avant - len(df_bien)} biens retires "
          f"({(nb_avant - len(df_bien)) / nb_avant * 100:.1f} %)")

    print("\n" + "=" * 50)
    print(f"FLUX : {type_bien.upper()} ({len(df_bien)} biens)")
    print("=" * 50)

    X = df_bien[features_base].copy()
    y = df_bien['log_prix_total']

    annee_max = df_bien['annee_vente'].max()
    train_mask = df_bien['annee_vente'] < annee_max
    test_mask = df_bien['annee_vente'] == annee_max
    if train_mask.sum()==0 or test_mask.sum()==0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    coords_train = np.deg2rad(df_bien.loc[X_train.index, ['latitude','longitude']])
    arbre_v = BallTree(coords_train, metric='haversine')

    annees_train = df_bien.loc[X_train.index, 'annee_vente']
    idx_marche = df_bien.loc[X_train.index].groupby('annee_vente')['prix_m2'].median()
    ref_marche = idx_marche.loc[idx_marche.index.max()]
    coef_marche = (ref_marche / idx_marche).to_dict()
    prix_train_actu = df_bien.loc[X_train.index, 'prix_m2'].values * annees_train.map(coef_marche).values
    surface_train = df_bien.loc[X_train.index, 'surface_reelle_bati'].values

    features_finales = list(features_base)

    if FA['voisins']:
        def voisins(dist_rad, idx, sb, self_i=None):
            if self_i is not None:
                keep = idx != self_i; dist_rad, idx = dist_rad[keep], idx[keep]
            if len(idx)==0: return np.nan
            dm = dist_rad*RAYON; pv, sv = prix_train_actu[idx], surface_train[idx]
            m = (sv>=sb*0.6)&(sv<=sb*1.4)
            if m.sum()>=3: d,p = dm[m], pv[m]
            else: d,p = dm, pv
            w = 1.0/(d+50.0); return np.sum(w*p)/np.sum(w)
        k = min(41, len(coords_train))
        dtr, itr = arbre_v.query(coords_train, k=k)
        sb_tr = df_bien.loc[X_train.index, 'surface_reelle_bati'].values
        X_train = X_train.copy(); X_test = X_test.copy()
        X_train['prix_m2_voisins'] = [voisins(dtr[i], itr[i], sb_tr[i], self_i=i) for i in range(len(itr))]
        coords_test = np.deg2rad(df_bien.loc[X_test.index, ['latitude','longitude']])
        dte, ite = arbre_v.query(coords_test, k=min(40,len(coords_train)))
        sb_te = df_bien.loc[X_test.index, 'surface_reelle_bati'].values
        X_test['prix_m2_voisins'] = [voisins(dte[i], ite[i], sb_te[i]) for i in range(len(ite))]
        features_finales += ['prix_m2_voisins']

    if FA['densite']:
        rr = 1000/RAYON
        coords_test_d = np.deg2rad(df_bien.loc[X_test.index, ['latitude','longitude']])
        X_train = X_train.copy(); X_test = X_test.copy()
        X_train['densite_ventes_1km'] = arbre_v.query_radius(coords_train, r=rr, count_only=True)
        X_test['densite_ventes_1km'] = arbre_v.query_radius(coords_test_d, r=rr, count_only=True)
        features_finales += ['densite_ventes_1km']

    if FA['section']:
        df_tr = df_bien.loc[X_train.index].copy()
        df_tr['prix_actu'] = df_tr['prix_m2'].values * df_tr['annee_vente'].map(coef_marche).values
        med_commune = df_tr.groupby('code_commune')['prix_actu'].median()
        med_globale = df_tr['prix_actu'].median()
        vals_train = pd.Series(np.nan, index=X_train.index)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for pf, po in kf.split(df_tr):
            ms = df_tr.iloc[pf].groupby('code_section')['prix_actu'].median()
            mc = df_tr.iloc[pf].groupby('code_commune')['prix_actu'].median()
            sous = df_tr.iloc[po]
            v = sous['code_section'].map(ms).fillna(sous['code_commune'].map(mc)).fillna(df_tr.iloc[pf]['prix_actu'].median())
            vals_train.iloc[po] = v.values
        X_train = X_train.copy(); X_test = X_test.copy()
        X_train['prix_m2_section'] = vals_train.values
        med_section = df_tr.groupby('code_section')['prix_actu'].median()
        sec_te = df_bien.loc[X_test.index, 'code_section'].map(med_section)
        com_te = df_bien.loc[X_test.index, 'code_commune'].map(med_commune)
        X_test['prix_m2_section'] = sec_te.fillna(com_te).fillna(med_globale).values
        nb_vs = df_tr.groupby('code_section').size()
        X_train['nb_ventes_section'] = df_bien.loc[X_train.index, 'code_section'].map(nb_vs).fillna(0).values
        X_test['nb_ventes_section'] = df_bien.loc[X_test.index, 'code_section'].map(nb_vs).fillna(0).values
        features_finales += ['prix_m2_section','nb_ventes_section']

    features_finales = list(dict.fromkeys(features_finales))
    X_train = X_train[features_finales]; X_test = X_test[features_finales]

    mask_val = (annees_train == annees_train.max()).values
    if mask_val.sum() < 50 or (~mask_val).sum() < 200:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    else:
        X_tr, y_tr = X_train[~mask_val], y_train[~mask_val]
        X_val, y_val = X_train[mask_val], y_train[mask_val]

    modeles_q = {}
    for nom_q, alpha in {'bas':0.025,'median':0.50,'haut':0.975}.items():
        m = CatBoostRegressor(loss_function=f'Quantile:alpha={alpha}', iterations=1000,
                              learning_rate=0.04, depth=8, random_seed=42,
                              early_stopping_rounds=50, verbose=False)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        modeles_q[nom_q] = m
    modele = modeles_q['median']

    # Le modele predit maintenant directement le PRIX TOTAL (en log)
    total_reel = np.exp(y_test).values
    total_pred = np.exp(modele.predict(X_test))
    total_bas = np.exp(modeles_q['bas'].predict(X_test))
    total_haut = np.exp(modeles_q['haut'].predict(X_test))
    total_bas, total_haut = np.minimum(total_bas, total_haut), np.maximum(total_bas, total_haut)

    # Metriques sur le PRIX TOTAL
    mae = mean_absolute_error(total_reel, total_pred)
    mape = np.mean(np.abs((total_reel - total_pred) / total_reel)) * 100
    err_med = np.median(np.abs(total_reel - total_pred))
    rmse = np.sqrt(np.mean((total_reel - total_pred) ** 2))
    # RMSLE : erreur quadratique dans l'espace log (erreur RELATIVE).
    rmsle = np.sqrt(np.mean((np.log1p(total_pred) - np.log1p(total_reel)) ** 2))
    err_rel = np.abs(total_reel - total_pred) / total_reel
    r2 = r2_score(total_reel, total_pred)
    couv = np.mean((total_reel >= total_bas) & (total_reel <= total_haut)) * 100
    largeur = np.mean(total_haut - total_bas)

    print(f"SHAP {type_bien}...")
    shap_vals = shap.TreeExplainer(modele).shap_values(X_test)
    imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
    print(f"Top 5 SHAP ({type_bien}) :")
    for nom, val in imp.head(5).items():
        print(f"  - {nom.ljust(25)} : {val:.4f}")

    base = f"{nom_zone.replace(' ','_')}_{type_bien}_synthese"
    shap.summary_plot(shap_vals, X_test, show=False, max_display=15)
    plt.title(f"SHAP - {nom_zone} ({type_bien}) [synthese]", fontsize=12)
    plt.tight_layout(); plt.savefig(dossier_graphes / f"shap_{base}.png", dpi=150, bbox_inches='tight'); plt.close()

    print("\n" + "-"*50)
    print(f"RAPPORT {type_bien.upper()} (source synthese) - PRIX TOTAL")
    print("-"*50)
    print(f"Apprentissage : {len(X_train)} | Test : {len(X_test)}")
    print(f"R2              : {r2*100:.2f} %")
    print(f"MAE             : {mae:,.0f} EUR")
    print(f"MAPE            : {mape:.1f} %")
    print(f"Erreur mediane  : {err_med:,.0f} EUR")
    print(f"RMSE            : {rmse:,.0f} EUR")
    print(f"RMSLE           : {rmsle:.4f} (erreur relative, espace log)")
    print(f"PE10 / PE20     : {np.mean(err_rel<=0.10)*100:.1f} % / {np.mean(err_rel<=0.20)*100:.1f} %")
    print(f"Couverture 95%  : {couv:.1f} %")
    print(f"Largeur moyenne : {largeur:,.0f} EUR")

print(f"\nTemps total : {time.time()-temps_debut:.2f}s")
print(f"Graphes : {dossier_graphes}")