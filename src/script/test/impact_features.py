"""
IMPACT DES FEATURES SUR LA BASELINE
====================================
Mesure la contribution de CHAQUE feature ajoutee individuellement a la baseline.

  BASELINE = groupes 'geo_base' (lat/lon) + 'date' (annee, mois) + 'surface'
             (surface_reelle_bati, log_surface, surface_par_piece)
  Pour chaque GROUPE de features du menu correlation_synthese.py :
  BASELINE + ce groupe -> on mesure le RMSLE. L'ecart vs baseline = son apport.

Saisie UNE SEULE FOIS (departement), puis tout s'enchaine automatiquement.
Affiche uniquement le RMSLE (maisons + appartements) + un tableau trie par impact.

Source : etalab_dvf.synthese | Cible : prix total (log) | Split : aleatoire 70/30
"""

import os, sys, time
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostRegressor
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"

try:
    db_pass = os.environ["DB_PASS"]
    moteur_dvf = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur_enr = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur_dvf.connect().close()
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

# ======= SAISIE UNIQUE =======
departement = input("Departement a tester (ex: 34) : ").strip()
filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
RAYON = 6371000

print("\nChargement des donnees (une seule fois)...")
gdf_littoral = gpd.read_file(CHEMIN_GPKG)
if gdf_littoral.crs is not None and gdf_littoral.crs.to_epsg() != 4326:
    gdf_littoral = gdf_littoral.to_crs(epsg=4326)

# ======= EXTRACTION VENTES =======
maisons_apparts = pd.read_sql(f"""
    SELECT communes_code AS code_commune, parcelles_code AS id_parcelle,
           geo_iris_id,
           lat AS latitude, lng AS longitude, prix_m2,
           surface AS surface_reelle_bati, typebien AS type_local,
           nb_pieces AS nombre_pieces_principales, surface_terrain,
           YEAR(date) AS annee_vente, MONTH(date) AS mois_vente
    FROM synthese
    WHERE departements_code = '{departement}'
      AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
      AND typebien IN ('maison', 'appartement');
""", con=moteur_dvf)
for col in ['prix_m2','surface_reelle_bati','surface_terrain','nombre_pieces_principales',
            'latitude','longitude','annee_vente','mois_vente']:
    maisons_apparts[col] = pd.to_numeric(maisons_apparts[col], errors='coerce')
maisons_apparts = maisons_apparts.dropna(subset=['prix_m2','surface_reelle_bati',
                                                 'nombre_pieces_principales','latitude','longitude'])
maisons_apparts['type_local'] = maisons_apparts['type_local'].str.capitalize()

# ======= ENRICHISSEMENTS =======
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
revenus = pd.read_sql(f"SELECT code_commune, median_revenu_disponible, indice_gini, pct_minima_sociaux FROM demographie_communes WHERE LEFT(code_commune,2)='{departement}';", con=moteur_enr)
densite_pop = pd.read_sql("""
    SELECT d.code_commune, d.densite_population
    FROM densite_population d
    INNER JOIN (SELECT code_commune, MAX(annee) AS a FROM densite_population GROUP BY code_commune) m
      ON d.code_commune = m.code_commune AND d.annee = m.a
""", con=moteur_enr)
densite_pop['densite_population'] = pd.to_numeric(densite_pop['densite_population'], errors='coerce')
for c in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']:
    revenus[c] = pd.to_numeric(revenus[c], errors='coerce')
stations = pd.read_sql(f"SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur_enr)
hopitaux = pd.read_sql(f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal,2)='{departement}';", con=moteur_enr)
monuments = pd.read_sql(f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{departement}';", con=moteur_enr)
universites = pd.read_sql(f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{departement}';", con=moteur_enr)

# Features economiques experimentales (temporelles - redondantes avec l'annee sur un seul dep)
pib = pd.read_sql("SELECT annee, pib_national FROM pib_national", con=moteur_enr)
chomage = pd.read_sql("SELECT code_departement, annee, trimestre, taux_chomage FROM chomage_departements", con=moteur_enr)
taux = pd.read_sql("SELECT annee, mois, taux_credit_immo_fixe, taux_inflation FROM taux_macro", con=moteur_enr)

# ======= FUSION + DISTANCES (calculees une fois pour toutes) =======
donnees = pd.merge(maisons_apparts, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees, revenus, on='code_commune', how='left')
donnees = pd.merge(donnees, densite_pop, on='code_commune', how='left')

# Jointures economiques (cles departement + trimestre pour le chomage)
donnees['code_departement'] = donnees['code_commune'].str[:2]
donnees['trimestre'] = (donnees['mois_vente'] - 1) // 3 + 1
donnees = pd.merge(donnees, pib, left_on='annee_vente', right_on='annee', how='left').drop(columns=['annee'], errors='ignore')
donnees = pd.merge(donnees, chomage, left_on=['code_departement','annee_vente','trimestre'],
                   right_on=['code_departement','annee','trimestre'], how='left').drop(columns=['annee'], errors='ignore')
donnees = pd.merge(donnees, taux, left_on=['annee_vente','mois_vente'],
                   right_on=['annee','mois'], how='left').drop(columns=['annee','mois'], errors='ignore')
points_rad = np.deg2rad(donnees[['latitude','longitude']])

def dmin(dfp, col):
    if len(dfp) > 0:
        arbre = BallTree(np.deg2rad(dfp.iloc[:,0:2]), metric='haversine')
        d,_ = arbre.query(points_rad, k=1); donnees[col] = d.flatten()*RAYON
    else:
        donnees[col] = 999999

dmin(stations, 'dist_transport_m')
dmin(monuments, 'dist_monument_m')
dmin(hopitaux, 'dist_hopital_m')
if len(universites) > 0:
    au = BallTree(np.deg2rad(universites[['latitude','longitude']]), metric='haversine')
    d, iu = au.query(points_rad, k=1)
    donnees['dist_universite_m'] = d.flatten()*RAYON
    donnees['volume_etudiants_proche'] = universites.iloc[iu.flatten()]['nombre_etudiants'].values
else:
    donnees['dist_universite_m'] = 999999; donnees['volume_etudiants_proche'] = 0

def contour(sg):
    pts=[]
    for g in sg.geometry:
        if g.geom_type=='MultiPolygon':
            for p in g.geoms: pts.extend(list(p.exterior.coords))
        else: pts.extend(list(g.exterior.coords))
    if not pts: return pd.DataFrame(columns=['latitude','longitude'])
    a=np.array(pts); return pd.DataFrame(a[:,[1,0]], columns=['latitude','longitude'])
for cl,col in {'Mer':'dist_mer_m','Lac':'dist_lac_m','Estuaire':'dist_estuaire_m'}.items():
    dmin(contour(gdf_littoral[gdf_littoral['CLASSEMENT']==cl]), col)

# Potentiel urbain (modele de gravite sur les poles d'attraction)
poles = pd.read_sql("""
    SELECT aav_nom, AVG(latitude) AS latitude, AVG(longitude) AS longitude, COUNT(*) AS poids_aire
    FROM referentiel_communes
    WHERE aav_nom IS NOT NULL AND aav_nom != 'SO' AND latitude IS NOT NULL
    GROUP BY aav_nom HAVING poids_aire >= 10
""", con=moteur_enr)
poles_etrangers = pd.DataFrame([
    {'aav_nom':'Genève','latitude':46.2044,'longitude':6.1432,'poids_aire':250},
    {'aav_nom':'Barcelone','latitude':41.3874,'longitude':2.1686,'poids_aire':300},
    {'aav_nom':'Monaco','latitude':43.7384,'longitude':7.4246,'poids_aire':120},
    {'aav_nom':'Turin','latitude':45.0703,'longitude':7.6869,'poids_aire':250},
])
poles = pd.concat([poles, poles_etrangers], ignore_index=True)
for c in ['latitude','longitude','poids_aire']:
    poles[c] = pd.to_numeric(poles[c], errors='coerce')
poles = poles.dropna(subset=['latitude','longitude','poids_aire'])
if len(poles) > 0:
    ap = BallTree(np.deg2rad(poles[['latitude','longitude']].values), metric='haversine')
    pp = poles['poids_aire'].values.astype(float)
    drp, ip = ap.query(points_rad, k=min(20,len(poles)))
    donnees['potentiel_urbain'] = np.sum(pp[ip]/(drp*RAYON+5000), axis=1)
else:
    donnees['potentiel_urbain'] = 0

# Remplissage NaN + feature engineering
colonnes_dpe = ['pct_dpe_A','pct_dpe_B','pct_dpe_C','pct_dpe_D','pct_dpe_E','pct_dpe_F','pct_dpe_G']
colonnes_chauffage = ['pct_chauffage_elec','pct_chauffage_gaz','pct_chauffage_fioul','pct_chauffage_urbain']
colonnes_revenus = ['median_revenu_disponible','indice_gini','pct_minima_sociaux']
for col in colonnes_dpe + colonnes_chauffage + colonnes_revenus:
    donnees[col] = donnees[col].fillna(donnees[col].median())
donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)
if 'densite_population' in donnees.columns:
    donnees['densite_population'] = donnees['densite_population'].fillna(donnees['densite_population'].median())
for col in ['pib_national','taux_chomage','taux_credit_immo_fixe','taux_inflation']:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())
if 'potentiel_urbain' in donnees.columns:
    donnees['potentiel_urbain'] = donnees['potentiel_urbain'].fillna(donnees['potentiel_urbain'].median())

dp = donnees[(donnees['surface_reelle_bati']>=9)&(donnees['surface_reelle_bati']<=300)].copy()
dp.loc[dp['type_local']=='Appartement','surface_terrain'] = 0
dp['prix_total'] = dp['prix_m2'] * dp['surface_reelle_bati']
dp['log_prix_total'] = np.log(dp['prix_total'])
dp['log_surface'] = np.log(dp['surface_reelle_bati'])
dp['surface_par_piece'] = dp['surface_reelle_bati'] / dp['nombre_pieces_principales']
dp['a_terrain'] = (dp['surface_terrain']>0).astype(int)
dp['log_terrain'] = np.log1p(dp['surface_terrain'])
dp['code_section'] = dp['id_parcelle'].str[:10]
# IRIS : identifiant en texte, NULL remplaces par une valeur sentinelle (repli commune)
dp['geo_iris_id'] = dp['geo_iris_id'].astype('string').fillna('__NA__')

# ======= DEFINITION BASELINE + FEATURES CANDIDATES =======
# BASELINE (definition stricte) = localisation (lat/lon) + surface du BATI (sans terrain)
#                                 + date (annee, mois). Rien d'autre.
BASELINE = ['latitude', 'longitude', 'surface_reelle_bati', 'annee_vente', 'mois_vente']

# Candidates = EXACTEMENT les groupes du menu de correlation_synthese.py, chacun teste
# EN ENTIER (pas de sous-features). On ajoute UN groupe a la fois a la baseline.
# NB : le groupe 'surface' de correlation ajoute log_surface + surface_par_piece
#      (surface_reelle_bati est deja dans la baseline).
CANDIDATES_SIMPLES = {
    'surface':         ['log_surface', 'surface_par_piece'],
    'pieces':          ['nombre_pieces_principales'],
    'terrain':         ['a_terrain', 'surface_terrain', 'log_terrain'],
    'dpe':             colonnes_dpe,
    'chauffage':       colonnes_chauffage,
    'revenus':         colonnes_revenus,
    'dist_transport':  ['dist_transport_m'],
    'dist_monument':   ['dist_monument_m'],
    'dist_hopital':    ['dist_hopital_m'],
    'dist_universite': ['dist_universite_m', 'volume_etudiants_proche'],
    'dist_littoral':   ['dist_mer_m', 'dist_lac_m', 'dist_estuaire_m'],
    'potentiel_urbain':['potentiel_urbain'],
    'densite_population':['densite_population'],
    # Features experimentales economiques (temporelles - attendu : gain ~0 sur un seul dep)
    'chomage':         ['taux_chomage'],
    'taux_credit':     ['taux_credit_immo_fixe'],
    'taux_inflation':  ['taux_inflation'],
    'pib':             ['pib_national'],
}
# Groupes spatiaux locaux (calcules apres split) = memes noms que le menu correlation
CANDIDATES_SPATIALES = ['voisins', 'densite', 'section', 'iris']

def _ajouter_une_spatiale(nom_spat, X_train, X_test, d, suffixe=''):
    """Ajoute UNE feature spatiale (colonne 'f_'+suffixe) a X_train/X_test. Anti-leakage."""
    coords_tr = np.deg2rad(d.loc[X_train.index, ['latitude','longitude']])
    arbre = BallTree(coords_tr, metric='haversine')
    col = 'f_' + (suffixe if suffixe else nom_spat)
    if nom_spat == 'voisins':
        prix_tr = d.loc[X_train.index,'prix_m2'].values
        surf_tr = d.loc[X_train.index,'surface_reelle_bati'].values
        def vois(dr, idx, sb, self_i=None):
            if self_i is not None:
                keep = idx != self_i; dr, idx = dr[keep], idx[keep]
            if len(idx)==0: return np.nan
            dm = dr*RAYON; pv, sv = prix_tr[idx], surf_tr[idx]
            m=(sv>=sb*0.6)&(sv<=sb*1.4)
            if m.sum()>=3: dd,pp=dm[m],pv[m]
            else: dd,pp=dm,pv
            w=1.0/(dd+50.0); return np.sum(w*pp)/np.sum(w)
        k=min(41,len(coords_tr)); dtr,itr=arbre.query(coords_tr,k=k)
        sb_tr = d.loc[X_train.index,'surface_reelle_bati'].values
        X_train[col]=[vois(dtr[i],itr[i],sb_tr[i],self_i=i) for i in range(len(itr))]
        ct=np.deg2rad(d.loc[X_test.index,['latitude','longitude']])
        dte,ite=arbre.query(ct,k=min(40,len(coords_tr)))
        sb_te=d.loc[X_test.index,'surface_reelle_bati'].values
        X_test[col]=[vois(dte[i],ite[i],sb_te[i]) for i in range(len(ite))]
    elif nom_spat == 'densite':
        rr=1000/RAYON
        ct=np.deg2rad(d.loc[X_test.index,['latitude','longitude']])
        X_train[col]=arbre.query_radius(coords_tr,r=rr,count_only=True)
        X_test[col]=arbre.query_radius(ct,r=rr,count_only=True)
    elif nom_spat in ('section', 'iris'):
        cle = 'code_section' if nom_spat == 'section' else 'geo_iris_id'
        dtr_=d.loc[X_train.index].copy()
        mzone=dtr_.groupby(cle)['prix_m2'].median()
        mc=dtr_.groupby('code_commune')['prix_m2'].median()
        mg=dtr_['prix_m2'].median()
        vals=pd.Series(np.nan,index=X_train.index)
        kf=KFold(n_splits=5,shuffle=True,random_state=42)
        for pf_,po_ in kf.split(dtr_):
            z=dtr_.iloc[pf_].groupby(cle)['prix_m2'].median()
            c=dtr_.iloc[pf_].groupby('code_commune')['prix_m2'].median()
            sous=dtr_.iloc[po_]
            v=sous[cle].map(z).fillna(sous['code_commune'].map(c)).fillna(dtr_.iloc[pf_]['prix_m2'].median())
            vals.iloc[po_]=v.values
        X_train[col]=vals.values
        zt=d.loc[X_test.index,cle].map(mzone)
        ce=d.loc[X_test.index,'code_commune'].map(mc)
        X_test[col]=zt.fillna(ce).fillna(mg).values
    return X_train, X_test, col


def calculer_rmsle_type(df_bien, features_simples, features_spatiales=None):
    """
    Entraine un modele median sur (baseline + features_simples + features_spatiales)
    et renvoie le RMSLE (prix total). 'features_spatiales' est une LISTE (0, 1 ou +).
    """
    if features_spatiales is None:
        features_spatiales = []
    if isinstance(features_spatiales, str):
        features_spatiales = [features_spatiales]

    pl, pf = df_bien['prix_m2'].quantile(0.01), df_bien['prix_m2'].quantile(0.99)
    d = df_bien[(df_bien['prix_m2']>=pl)&(df_bien['prix_m2']<=pf)].copy()
    stats_com = d.groupby('code_commune')['prix_m2'].agg(['median','size'])
    ref = d['code_commune'].map(stats_com['median'])
    nc = d['code_commune'].map(stats_com['size'])
    ref = ref.where(nc>=10, d['prix_m2'].median())
    d = d[(d['prix_m2']/ref).between(0.40,2.50)].copy()
    if len(d) < 300: return None

    feats = list(BASELINE) + list(features_simples)
    X = d[feats].copy()
    y = d['log_prix_total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train = X_train.copy(); X_test = X_test.copy()

    for nom_spat in features_spatiales:
        X_train, X_test, _ = _ajouter_une_spatiale(nom_spat, X_train, X_test, d)

    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1000,
                          learning_rate=0.04, depth=8, random_seed=42,
                          early_stopping_rounds=50, verbose=False)
    Xtr,Xval,ytr,yval = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
    m.fit(Xtr,ytr,eval_set=(Xval,yval),use_best_model=True)
    total_reel = np.exp(y_test).values
    total_pred = np.exp(m.predict(X_test))
    return np.sqrt(np.mean((np.log1p(total_pred)-np.log1p(total_reel))**2))

def evaluer_config(features_simples, features_spatiales=None):
    """Renvoie (rmsle_maisons, rmsle_apparts)."""
    r_m = calculer_rmsle_type(dp[dp['type_local']=='Maison'], features_simples, features_spatiales)
    r_a = calculer_rmsle_type(dp[dp['type_local']=='Appartement'], features_simples, features_spatiales)
    return r_m, r_a

# ======= OUTILLAGE COMMUN =======
# Colonnes reelles produites par les features spatiales (pour l'affichage)
COLS_SPATIALES = {
    'voisins':  ['prix_m2_voisins'],
    'densite':  ['densite_ventes_1km'],
    'section':  ['prix_m2_section'],
    'iris':     ['prix_m2_iris'],
}

def format_sous_features(cols):
    """Liste les sous-features (colonnes reelles) d'un groupe."""
    if not cols:
        return ""
    return "      -> " + ", ".join(cols)

def moyenne_gain(gm, ga):
    vals = [g for g in (gm, ga) if g is not None]
    return sum(vals)/len(vals) if vals else None

# Toutes les features candidates, avec un type ('simple' ou 'spatiale')
TOUTES_CANDIDATES = ([(nom, 'simple') for nom in CANDIDATES_SIMPLES]
                     + [(nom, 'spatiale') for nom in CANDIDATES_SPATIALES])

def eval_combinaison(simples_groupes, spatiales):
    """
    Evalue une combinaison : liste de groupes simples (noms) + liste de features
    spatiales. Renvoie (rmsle_maison, rmsle_appart).
    """
    cols_simples = []
    for g in simples_groupes:
        cols_simples += CANDIDATES_SIMPLES[g]
    rm = calculer_rmsle_type(dp[dp['type_local']=='Maison'], cols_simples, spatiales)
    ra = calculer_rmsle_type(dp[dp['type_local']=='Appartement'], cols_simples, spatiales)
    return rm, ra


# ======= MODE 1 : IMPACT SUR BASELINE =======
def mode_impact():
    print("\n" + "="*72)
    print(f"MODE 1 - IMPACT DE CHAQUE FEATURE SUR LA BASELINE - Dep {departement}")
    print(f"BASELINE = {BASELINE}")
    print("="*72)
    print(f"{'Configuration':<22} {'RMSLE maison':>14} {'RMSLE appart':>14}")
    print("-"*72)
    rb_m, rb_a = evaluer_config([])
    print(f"{'BASELINE (seule)':<22} {rb_m:>14.4f} {rb_a:>14.4f}")
    print(format_sous_features(BASELINE))
    print("-"*72)
    resultats = []
    for nom, cols in CANDIDATES_SIMPLES.items():
        rm, ra = evaluer_config(cols)
        resultats.append((f"+ {nom}", rm, ra, (rb_m-rm) if rm else None, (rb_a-ra) if ra else None, cols))
        print(f"{'+ '+nom:<22} {rm:>14.4f} {ra:>14.4f}"); print(format_sous_features(cols))
    for nom in CANDIDATES_SPATIALES:
        rm, ra = evaluer_config([], features_spatiales=[nom])
        cols = COLS_SPATIALES.get(nom, [nom])
        resultats.append((f"+ {nom}", rm, ra, (rb_m-rm) if rm else None, (rb_a-ra) if ra else None, cols))
        print(f"{'+ '+nom:<22} {rm:>14.4f} {ra:>14.4f}"); print(format_sous_features(cols))

    print("\n" + "="*72)
    print("CLASSEMENT PAR IMPACT (gain de RMSLE vs baseline, moyenne des 2 types)")
    print("="*72)
    tries = sorted(resultats, key=lambda r: moyenne_gain(r[3], r[4]) or -999, reverse=True)
    print(f"{'Feature':<22} {'gain maison':>12} {'gain appart':>12} {'gain moyen':>12}")
    print("-"*72)
    for nom, rm, ra, gm, ga, cols in tries:
        gm_s = f"{gm:+.4f}" if gm is not None else "  n/a"
        ga_s = f"{ga:+.4f}" if ga is not None else "  n/a"
        gmo = moyenne_gain(gm, ga)
        print(f"{nom:<22} {gm_s:>12} {ga_s:>12} {(gmo if gmo else 0):>+12.4f}")
        print(f"      ({', '.join(cols)})")
    print(f"\nRappel baseline : maison {rb_m:.4f} | appart {rb_a:.4f}")


# ======= MODE 2 : ABLATION (retirer chaque feature du modele COMPLET) =======
def mode_ablation():
    print("\n" + "="*72)
    print(f"MODE 2 - ABLATION : apport de chaque feature DANS le modele complet - Dep {departement}")
    print("="*72)
    tous_simples = list(CANDIDATES_SIMPLES.keys())
    tous_spatiaux = list(CANDIDATES_SPATIALES)
    rc_m, rc_a = eval_combinaison(tous_simples, tous_spatiaux)
    print(f"COMPLET : maison {rc_m:.4f} | appart {rc_a:.4f}\n")
    print(f"{'Feature retiree':<22} {'RMSLE maison':>13} {'RMSLE appart':>13} {'perte moyenne':>14}")
    print("-"*72)
    resultats = []
    for nom in tous_simples:
        s = [x for x in tous_simples if x != nom]
        rm, ra = eval_combinaison(s, tous_spatiaux)
        perte = moyenne_gain((rm-rc_m) if rm else None, (ra-rc_a) if ra else None)
        resultats.append((nom, rm, ra, perte, CANDIDATES_SIMPLES[nom]))
    for nom in tous_spatiaux:
        sp = [x for x in tous_spatiaux if x != nom]
        rm, ra = eval_combinaison(tous_simples, sp)
        perte = moyenne_gain((rm-rc_m) if rm else None, (ra-rc_a) if ra else None)
        resultats.append((nom, rm, ra, perte, COLS_SPATIALES.get(nom, [nom])))
    for nom, rm, ra, perte, cols in sorted(resultats, key=lambda r: r[3] or -999, reverse=True):
        print(f"{'- '+nom:<22} {rm:>13.4f} {ra:>13.4f} {(perte if perte else 0):>+14.4f}")
        print(f"      ({', '.join(cols)})")
    print("\nPerte positive = retirer cette feature DEGRADE -> elle est utile dans le complet.")
    print("Perte ~0 = feature redondante (le modele s'en passe sans dommage).")


# ======= MODE 3 : SELECTION GLOUTONNE (greedy) =======
def mode_greedy():
    print("\n" + "="*72)
    print(f"MODE 3 - SELECTION GLOUTONNE : meilleure combinaison pas a pas - Dep {departement}")
    print("="*72)
    seuil = input("  Gain minimal pour continuer d'ajouter (defaut 0.0005) : ").strip()
    seuil = float(seuil) if seuil else 0.0005

    rb_m, rb_a = evaluer_config([])
    rmsle_courant = moyenne_gain(rb_m, rb_a)
    print(f"\nDepart (baseline) : maison {rb_m:.4f} | appart {rb_a:.4f} | moyen {rmsle_courant:.4f}")

    simples_retenus, spatiales_retenues = [], []
    restantes = list(TOUTES_CANDIDATES)
    etape = 1
    while restantes:
        meilleure, meilleur_rmsle, meilleur_rm, meilleur_ra = None, rmsle_courant, None, None
        for nom, typ in restantes:
            if typ == 'simple':
                rm, ra = eval_combinaison(simples_retenus + [nom], spatiales_retenues)
            else:
                rm, ra = eval_combinaison(simples_retenus, spatiales_retenues + [nom])
            rmo = moyenne_gain(rm, ra)
            if rmo is not None and rmo < meilleur_rmsle:
                meilleure, meilleur_rmsle = (nom, typ), rmo
                meilleur_rm, meilleur_ra = rm, ra
        gain = rmsle_courant - meilleur_rmsle
        if meilleure is None or gain < seuil:
            print(f"\n  -> Arret : aucune feature restante n'apporte >= {seuil}")
            break
        nom, typ = meilleure
        if typ == 'simple': simples_retenus.append(nom)
        else: spatiales_retenues.append(nom)
        restantes = [c for c in restantes if c[0] != nom]
        print(f"  Etape {etape} : + {nom:<18} -> RMSLE moyen {meilleur_rmsle:.4f} "
              f"(gain {gain:+.4f})  [maison {meilleur_rm:.4f}, appart {meilleur_ra:.4f}]")
        rmsle_courant = meilleur_rmsle
        etape += 1

    print("\n" + "="*72)
    print("MEILLEURE COMBINAISON TROUVEE (ordre d'ajout = ordre d'importance)")
    print("="*72)
    print(f"  Features simples   : {simples_retenus}")
    print(f"  Features spatiales : {spatiales_retenues}")
    print(f"  RMSLE moyen final  : {rmsle_courant:.4f} (baseline : {moyenne_gain(rb_m, rb_a):.4f})")
    print(f"  Gain total         : {moyenne_gain(rb_m, rb_a) - rmsle_courant:+.4f}")


# ======= MODE 4 : COMPLEMENTARITE DE DEUX FEATURES =======
def mode_complementarite():
    print("\n" + "="*72)
    print("MODE 4 - COMPLEMENTARITE DE DEUX FEATURES")
    print("="*72)
    dispo = [n for n, _ in TOUTES_CANDIDATES]
    print(f"  Features disponibles : {', '.join(dispo)}")
    f1 = input("  Feature 1 : ").strip()
    f2 = input("  Feature 2 : ").strip()
    if f1 not in dispo or f2 not in dispo:
        print("  Nom(s) invalide(s)."); return

    def split_type(nom):
        return ([nom], []) if nom in CANDIDATES_SIMPLES else ([], [nom])

    s1, sp1 = split_type(f1)
    s2, sp2 = split_type(f2)
    rb_m, rb_a = evaluer_config([])
    r1_m, r1_a = eval_combinaison(s1, sp1)
    r2_m, r2_a = eval_combinaison(s2, sp2)
    r12_m, r12_a = eval_combinaison(s1 + s2, sp1 + sp2)

    print(f"\n{'Configuration':<28} {'maison':>10} {'appart':>10} {'gain moyen':>12}")
    print("-"*64)
    def ligne(lbl, rm, ra):
        g = moyenne_gain((rb_m-rm) if rm else None, (rb_a-ra) if ra else None)
        print(f"{lbl:<28} {rm:>10.4f} {ra:>10.4f} {(g if g else 0):>+12.4f}")
    print(f"{'baseline':<28} {rb_m:>10.4f} {rb_a:>10.4f} {0:>+12.4f}")
    ligne(f"+ {f1}", r1_m, r1_a)
    ligne(f"+ {f2}", r2_m, r2_a)
    ligne(f"+ {f1} + {f2}", r12_m, r12_a)

    g1 = moyenne_gain((rb_m-r1_m) if r1_m else None, (rb_a-r1_a) if r1_a else None) or 0
    g2 = moyenne_gain((rb_m-r2_m) if r2_m else None, (rb_a-r2_a) if r2_a else None) or 0
    g12 = moyenne_gain((rb_m-r12_m) if r12_m else None, (rb_a-r12_a) if r12_a else None) or 0
    print(f"\n  Gain {f1} seul : {g1:+.4f} | {f2} seul : {g2:+.4f} | ensemble : {g12:+.4f}")
    surplus = g12 - max(g1, g2)
    if surplus > 0.001:
        print(f"  -> COMPLEMENTAIRES : ensemble ({g12:+.4f}) > meilleur seul ({max(g1,g2):+.4f}), surplus {surplus:+.4f}")
    else:
        print(f"  -> REDONDANTES : ensemble n'apporte quasi rien de plus que le meilleur seul (surplus {surplus:+.4f})")


# ======= MENU =======
print("\n" + "="*72)
print("OUTIL DE TEST DES FEATURES")
print("="*72)
print("  1 = Impact de chaque feature sur la baseline")
print("  2 = Ablation (apport de chaque feature dans le modele complet)")
print("  3 = Selection gloutonne (meilleure combinaison pas a pas)")
print("  4 = Complementarite de deux features")
choix = input("  Choix (1-4) : ").strip()

t0 = time.time()
if choix == '2':   mode_ablation()
elif choix == '3': mode_greedy()
elif choix == '4': mode_complementarite()
else:              mode_impact()
print(f"\nTemps total : {time.time()-t0:.0f}s")