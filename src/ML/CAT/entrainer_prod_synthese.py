"""
ENTRAINEMENT PRODUCTION - SOURCE etalab_dvf.synthese + CIBLE PRIX TOTAL
=======================================================================
Version production coherente avec correlation_synthese.py :
  - VENTES depuis etalab_dvf.synthese (propre, 2014-2025)
  - ENRICHISSEMENTS depuis EstimationIA (DPE, revenus, infra, taux...)
  - CIBLE = log(prix_total) : le modele predit directement des EUROS
  - PAS de filtre lots (inutile d'apres le test A/B)
  - entraine sur TOUTES les donnees (pas de split), sauvegarde les artefacts

IMPORTANT : les features de prix (voisins, section) restent en EUR/m2
(comparables entre biens), seule la CIBLE est le prix total.
estimer.py devra : predire le prix total directement (plus de x surface),
et lire features.json pour reconstruire les bonnes features.
"""

import sys, os, json, pickle
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
# 0. CHOIX DES FEATURES
# ==========================================
def demander(question, defaut=True):
    ind = "O/n" if defaut else "o/N"
    while True:
        r = input(f"  {question} ({ind}) : ").strip().lower()
        if r == "": return defaut
        if r.startswith('o'): return True
        if r.startswith('n'): return False
        print("    Tapez 'o' ou 'n'.")

print("ENTRAINEMENT PRODUCTION - synthese + PRIX TOTAL")
print("=" * 60)
FA = {}
print("\n--- Caracteristiques du bien ---")
FA['geo_base']    = demander("Latitude / longitude ?", True)
FA['surface']     = demander("Surface (+ log, par piece) ?", True)
FA['pieces']      = demander("Nombre de pieces ?", True)
FA['terrain']     = demander("Terrain ?", True)
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
print("\n--- Feature experimentale ---")
FA['potentiel_urbain'] = demander("Potentiel urbain ?", True)

# ==========================================
# 1. CONNEXIONS ET ZONE
# ==========================================
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
    moteur_dvf = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur_enr = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur_dvf.connect().close(); moteur_enr.connect().close()
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

departement = input("\nDepartement (ex: 34) ou 'FRANCE' : ").strip().upper()
if len(departement) < 2:
    print("Format invalide."); sys.exit()

if departement == 'FRANCE':
    filtre_dvf = "1=1"; filtre_dpe = "1=1"; filtre_rev = "1=1"; dep_infra = "FRANCE"; nom_zone = "France"
else:
    filtre_dvf = f"departements_code = '{departement}'"
    filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
    filtre_rev = f"LEFT(code_commune, 2) = '{departement}'"
    dep_infra = departement; nom_zone = f"Departement {departement}"

print(f"\nEntrainement pour : {nom_zone}")

# ==========================================
# 1. EXTRACTION VENTES (synthese)
# ==========================================
print("Etape 1 : Extraction des ventes (synthese)...")
maisons_apparts = pd.read_sql(f"""
    SELECT id,
           communes_code AS code_commune, parcelles_code AS id_parcelle,
           lat AS latitude, lng AS longitude, prix_m2,
           surface AS surface_reelle_bati, typebien AS type_local,
           nb_pieces AS nombre_pieces_principales, surface_terrain, nb_dependances,
           valeur_fonciere, YEAR(date) AS annee_vente, MONTH(date) AS mois_vente
    FROM synthese
    WHERE {filtre_dvf}
      AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
      AND typebien IN ('maison', 'appartement');
""", con=moteur_dvf)

colonnes_num = ['prix_m2','surface_reelle_bati','surface_terrain','nombre_pieces_principales',
                'nb_dependances','valeur_fonciere','latitude','longitude','annee_vente','mois_vente']
for col in colonnes_num:
    maisons_apparts[col] = pd.to_numeric(maisons_apparts[col], errors='coerce')
maisons_apparts = maisons_apparts.dropna(subset=['prix_m2','surface_reelle_bati',
                                                 'nombre_pieces_principales','latitude','longitude'])
maisons_apparts['type_local'] = maisons_apparts['type_local'].str.capitalize()

if len(maisons_apparts) == 0:
    print("Aucune donnee."); sys.exit()
print(f"  {len(maisons_apparts):,} ventes extraites.")

# ==========================================
# 1bis. ENRICHISSEMENTS (EstimationIA)
# ==========================================
print("Etape 1bis : Enrichissements (EstimationIA)...")
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

revenus = pd.read_sql(f"SELECT code_commune, median_revenu_disponible, indice_gini, pct_minima_sociaux FROM demographie_communes WHERE {filtre_rev};", con=moteur_enr)
for col in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']:
    revenus[col] = pd.to_numeric(revenus[col], errors='coerce')

poles = None
if FA['potentiel_urbain']:
    poles = pd.read_sql("""
        SELECT aav_nom, AVG(latitude) AS latitude, AVG(longitude) AS longitude, COUNT(*) AS poids_aire
        FROM referentiel_communes WHERE aav_nom IS NOT NULL AND aav_nom != 'SO' AND latitude IS NOT NULL
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

# ==========================================
# 2. FUSION ET DISTANCES
# ==========================================
print("Etape 2 : Fusion et distances...")
donnees = pd.merge(maisons_apparts, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees, revenus, on='code_commune', how='left')

RAYON_TERRE_METRES = 6371000
points_rad = np.deg2rad(donnees[['latitude','longitude']])

def calculer_distance_min(df_points, nom_colonne):
    if len(df_points) > 0:
        arbre = BallTree(np.deg2rad(df_points.iloc[:,0:2]), metric='haversine')
        dist_rad, _ = arbre.query(points_rad, k=1)
        donnees[nom_colonne] = dist_rad.flatten() * RAYON_TERRE_METRES
    else:
        donnees[nom_colonne] = 999999

if FA['dist_transport']: calculer_distance_min(stations, 'dist_transport_m')
if FA['dist_monument']:  calculer_distance_min(monuments, 'dist_monument_m')
if FA['dist_hopital']:   calculer_distance_min(hopitaux, 'dist_hopital_m')

if FA['dist_universite']:
    if len(universites) > 0:
        au = BallTree(np.deg2rad(universites[['latitude','longitude']]), metric='haversine')
        d, iu = au.query(points_rad, k=1)
        donnees['dist_universite_m'] = d.flatten()*RAYON_TERRE_METRES
        donnees['volume_etudiants_proche'] = universites.iloc[iu.flatten()]['nombre_etudiants'].values
    else:
        donnees['dist_universite_m'] = 999999; donnees['volume_etudiants_proche'] = 0
else:
    donnees['volume_etudiants_proche'] = 0

def extraire_points_contour(sous_gdf):
    points = []
    for geom in sous_gdf.geometry:
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms: points.extend(list(poly.exterior.coords))
        else: points.extend(list(geom.exterior.coords))
    if not points: return pd.DataFrame(columns=['latitude','longitude'])
    pts = np.array(points)
    return pd.DataFrame(pts[:,[1,0]], columns=['latitude','longitude'])

if FA['dist_littoral']:
    for classement, nom_col in {'Mer':'dist_mer_m','Lac':'dist_lac_m','Estuaire':'dist_estuaire_m'}.items():
        calculer_distance_min(extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT']==classement]), nom_col)

if FA['potentiel_urbain'] and poles is not None and len(poles) > 0:
    poles_rad = np.deg2rad(poles[['latitude','longitude']].values)
    poids_poles = poles['poids_aire'].values.astype(float)
    arbre_poles = BallTree(poles_rad, metric='haversine')
    drp, ip = arbre_poles.query(points_rad, k=min(20,len(poles)))
    donnees['potentiel_urbain'] = np.sum(poids_poles[ip]/(drp*RAYON_TERRE_METRES+5000), axis=1)

# ==========================================
# 3. NETTOYAGE + FEATURES + CIBLE PRIX TOTAL
# ==========================================
print("Etape 3 : Nettoyage et feature engineering...")
colonnes_dpe = ['pct_dpe_A','pct_dpe_B','pct_dpe_C','pct_dpe_D','pct_dpe_E','pct_dpe_F','pct_dpe_G']
colonnes_chauffage = ['pct_chauffage_elec','pct_chauffage_gaz','pct_chauffage_fioul','pct_chauffage_urbain']
colonnes_revenus = ['median_revenu_disponible','indice_gini','pct_minima_sociaux']

for col in colonnes_dpe + colonnes_chauffage + colonnes_revenus:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())
donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)
if 'potentiel_urbain' in donnees.columns:
    donnees['potentiel_urbain'] = donnees['potentiel_urbain'].fillna(donnees['potentiel_urbain'].median())

donnees_propres = donnees[(donnees['surface_reelle_bati']>=9) & (donnees['surface_reelle_bati']<=300)].copy()
mask_appart = donnees_propres['type_local'] == 'Appartement'
donnees_propres.loc[mask_appart, 'surface_terrain'] = 0

# CIBLE = PRIX TOTAL (le modele predit directement des euros)
donnees_propres['prix_total'] = donnees_propres['prix_m2'] * donnees_propres['surface_reelle_bati']
donnees_propres['log_prix_total'] = np.log(donnees_propres['prix_total'])
donnees_propres['log_surface'] = np.log(donnees_propres['surface_reelle_bati'])
donnees_propres['surface_par_piece'] = donnees_propres['surface_reelle_bati'] / donnees_propres['nombre_pieces_principales']
donnees_propres['a_terrain'] = (donnees_propres['surface_terrain']>0).astype(int)
donnees_propres['log_terrain'] = np.log1p(donnees_propres['surface_terrain'])
donnees_propres['code_section'] = donnees_propres['id_parcelle'].str[:10]

def build_features_base():
    f = []
    if FA['geo_base']:   f += ['latitude','longitude']
    if FA['pieces']:     f += ['nombre_pieces_principales']
    if FA['date']:       f += ['annee_vente','mois_vente']
    if FA['terrain']:    f += ['a_terrain','surface_terrain','log_terrain']
    if FA['surface']:    f += ['surface_reelle_bati','log_surface','surface_par_piece']
    if FA['dist_universite']: f += ['volume_etudiants_proche']
    if FA['revenus']:    f += colonnes_revenus
    if FA['potentiel_urbain'] and 'potentiel_urbain' in donnees_propres.columns: f += ['potentiel_urbain']
    if FA['dpe']:        f += colonnes_dpe
    if FA['chauffage']:  f += colonnes_chauffage
    if FA['dist_transport']:  f += ['dist_transport_m']
    if FA['dist_monument']:   f += ['dist_monument_m']
    if FA['dist_hopital']:    f += ['dist_hopital_m']
    if FA['dist_universite']: f += ['dist_universite_m']
    if FA['dist_littoral']:   f += ['dist_mer_m','dist_lac_m','dist_estuaire_m']
    return list(dict.fromkeys(f))

features_base = build_features_base()
print(f"  {len(features_base)} features de base actives.")

# ==========================================
# 4. ENTRAINEMENT PAR TYPE (sur TOUTES les donnees)
# ==========================================
datasets = {
    'maisons': donnees_propres[donnees_propres['type_local']=='Maison'].copy(),
    'appartements': donnees_propres[donnees_propres['type_local']=='Appartement'].copy()
}

for type_bien, df_bien in datasets.items():
    if len(df_bien) < 50:
        print(f"\n--- {type_bien} ignore (pas assez de donnees) ---"); continue

    plancher = df_bien['prix_m2'].quantile(0.01)
    plafond = df_bien['prix_m2'].quantile(0.99)
    df_bien = df_bien[(df_bien['prix_m2']>=plancher) & (df_bien['prix_m2']<=plafond)].copy()

    # --- Filtre de coherence marche (identique a l'evaluation) ---
    # Elimine les transactions hors marche (ventes familiales sous-evaluees,
    # parts indivises) en comparant chaque bien au prix median de SA commune.
    # COHERENCE : ce meme filtre est applique dans correlation_cat/synthese.
    stats_com = df_bien.groupby('code_commune')['prix_m2'].agg(['median', 'size'])
    ref_com = df_bien['code_commune'].map(stats_com['median'])
    n_com = df_bien['code_commune'].map(stats_com['size'])
    ref_com = ref_com.where(n_com >= 10, df_bien['prix_m2'].median())
    ratio = df_bien['prix_m2'] / ref_com
    nb_avant = len(df_bien)
    df_bien = df_bien[ratio.between(0.40, 2.50)].copy()
    print(f"  Filtre coherence marche : {nb_avant - len(df_bien)} biens retires "
          f"({(nb_avant - len(df_bien)) / nb_avant * 100:.1f} %)")

    print("\n" + "=" * 50)
    print(f"FLUX : {type_bien.upper()} ({len(df_bien)} biens)")
    print("=" * 50)

    X = df_bien[features_base].copy()
    y = df_bien['log_prix_total']  # CIBLE = prix total

    coords_all = np.deg2rad(df_bien[['latitude','longitude']])
    prix_all = df_bien['prix_m2'].values          # features en EUR/m2 (comparables)
    surface_all = df_bien['surface_reelle_bati'].values
    arbre_voisins = BallTree(coords_all, metric='haversine')

    features_finales = list(features_base)

    if FA['voisins']:
        def voisins_pond(distances_rad, indices, surface_bien):
            distances_rad = distances_rad[1:]; indices = indices[1:]
            if len(indices)==0: return np.nan
            dm = distances_rad*RAYON_TERRE_METRES
            pv, sv = prix_all[indices], surface_all[indices]
            m = (sv>=surface_bien*0.6)&(sv<=surface_bien*1.4)
            if m.sum()>=3: d,p = dm[m], pv[m]
            else: d,p = dm, pv
            w = 1.0/(d+50.0); return np.sum(w*p)/np.sum(w)
        k = min(41, len(coords_all))
        dv, iv = arbre_voisins.query(coords_all, k=k)
        if k > 1:
            X['prix_m2_voisins'] = [voisins_pond(dv[i], iv[i], surface_all[i]) for i in range(len(iv))]
        else:
            X['prix_m2_voisins'] = prix_all
        features_finales += ['prix_m2_voisins']

    if FA['densite']:
        rayon_rad = 1000 / RAYON_TERRE_METRES
        X['densite_ventes_1km'] = arbre_voisins.query_radius(coords_all, r=rayon_rad, count_only=True)
        features_finales += ['densite_ventes_1km']

    med_section = med_commune = None
    med_globale = float(df_bien['prix_m2'].median())
    nb_ventes_section = None
    if FA['section']:
        med_section = df_bien.groupby('code_section')['prix_m2'].median()
        med_commune = df_bien.groupby('code_commune')['prix_m2'].median()
        sec = df_bien['code_section']; com = df_bien['code_commune']
        X['prix_m2_section'] = sec.map(med_section).fillna(com.map(med_commune)).fillna(med_globale).values
        nb_ventes_section = df_bien.groupby('code_section').size()
        X['nb_ventes_section'] = df_bien['code_section'].map(nb_ventes_section).fillna(0).values
        features_finales += ['prix_m2_section','nb_ventes_section']

    features_finales = list(dict.fromkeys(features_finales))
    X = X[features_finales]

    print("  -> Entrainement des modeles quantiles...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    modeles = {}
    for nom, alpha in {'bas':0.025,'median':0.50,'haut':0.975}.items():
        m = CatBoostRegressor(loss_function=f'Quantile:alpha={alpha}', iterations=1000,
                              learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
                              random_seed=42, early_stopping_rounds=50, verbose=False)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        modeles[nom] = m
        print(f"     * '{nom}' entraine ({m.get_best_iteration()+1} arbres).")

    print("  -> Sauvegarde...")
    for nom, m in modeles.items():
        m.save_model(str(DOSSIER_MODELE / f"modele_{type_bien}_{nom}.cbm"))
    with open(DOSSIER_MODELE / f"features_{type_bien}.json", "w") as f:
        json.dump(features_finales, f)
    with open(DOSSIER_MODELE / "features.json", "w") as f:
        json.dump(features_finales, f)

    contexte = {
        'cible': 'prix_total',  # NEW : signale a estimer.py que la sortie est un prix total
        'features_actives': FA,
        'arbre_voisins_data': coords_all.values,
        'prix_all': prix_all,
        'surface_all': surface_all,
        'med_section': med_section.to_dict() if med_section is not None else {},
        'med_commune': med_commune.to_dict() if med_commune is not None else {},
        'med_globale': med_globale,
        'nb_ventes_section': nb_ventes_section.to_dict() if nb_ventes_section is not None else {},
        'stations': stations, 'monuments': monuments, 'hopitaux': hopitaux, 'universites': universites,
        'profils_communes': donnees.drop_duplicates('code_commune').set_index('code_commune')[
            colonnes_dpe + colonnes_chauffage + colonnes_revenus].to_dict('index'),
        'medianes_globales': {c: float(donnees[c].median()) for c in colonnes_dpe+colonnes_chauffage+colonnes_revenus},
        'rayon_terre': RAYON_TERRE_METRES,
        'points_mer': extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT']=='Mer']),
        'points_lac': extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT']=='Lac']),
        'points_estuaire': extraire_points_contour(gdf_littoral[gdf_littoral['CLASSEMENT']=='Estuaire'])
    }
    if poles is not None and len(poles) > 0:
        contexte['poles_urbains'] = poles[['latitude','longitude','poids_aire']].values

    with open(DOSSIER_MODELE / f"contexte_{type_bien}.pkl", "wb") as f:
        pickle.dump(contexte, f)

print("\n" + "=" * 50)
print(f"TERMINE. Artefacts dans : {DOSSIER_MODELE}")
print("CIBLE = prix total : estimer.py doit predire directement des euros")
print("(plus de multiplication par la surface). Le contexte contient cible='prix_total'.")