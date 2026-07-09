"""
TEST TOUT-EN-UN : CHOIX FEATURES -> ENTRAINEMENT -> TEST
=========================================================
1. Demande les features a inclure (comme le labo)
2. Entraine les modeles quantiles CatBoost sur ces features
3. Tire nb_test biens reels par type (maisons + appartements)
4. Reconstruit LES MEMES features pour ces biens et predit
5. Affiche chaque test (idDVF, surface, terrain, prix reel/estime, fourchette,
   statut OK/hors marge) + decompte dans/hors marge 20%

Cohérence garantie : le meme jeu de features sert a l'entrainement et au test.
"""

import time, sys, os
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv
import geopandas as gpd

MARGE = 0.20  # marge d'erreur de reference (20%)

# =====================================================================
# SAISIE : FEATURES (o/n) + parametres
# =====================================================================
def demander(question, defaut=True):
    ind = "O/n" if defaut else "o/N"
    while True:
        r = input(f"  {question} ({ind}) : ").strip().lower()
        if r == "": return defaut
        if r.startswith('o'): return True
        if r.startswith('n'): return False
        print("    Tapez 'o' ou 'n'.")

print("=" * 55)
print("TEST TOUT-EN-UN : features -> entrainement -> test")
print("=" * 55)

print("\n--- Caracteristiques du bien ---")
FA = {}
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
print("\n--- Features experimentales ---")
FA['potentiel_urbain'] = demander("Potentiel urbain ?", True)

departement = input("\nDepartement (ex: 34) : ").strip().upper()
nb_test = int(input("Nombre de tests par type (ex: 20) : ").strip())

filtre_dvf = "1=1" if departement == 'FRANCE' else f"code_departement = '{departement}'"
filtre_dpe = "1=1" if departement == 'FRANCE' else f"LEFT(code_insee_ban,2) = '{departement}'"
dep_infra = "FRANCE" if departement == 'FRANCE' else departement

# =====================================================================
# CONNEXION
# =====================================================================
RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"
try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except Exception:
    print("Erreur connexion base."); sys.exit()

gdf_littoral = gpd.read_file(CHEMIN_GPKG)
RAYON = 6371000
temps_debut = time.time()

# =====================================================================
# EXTRACTION (tout le dataset du departement)
# =====================================================================
print("\nExtraction SQL...")
donnees = pd.read_sql(f"""
    SELECT id, code_commune, id_parcelle, latitude, longitude, valeur_fonciere,
           (valeur_fonciere / surface_reelle_bati) AS prix_m2,
           surface_reelle_bati, type_local, nombre_pieces_principales,
           surface_terrain, YEAR(date_mutation) AS annee_vente, MONTH(date_mutation) AS mois_vente
    FROM valeurs_foncieres
    WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
      AND nature_mutation = 'Vente' AND nombre_lots <= 3 AND nombre_pieces_principales > 0
      AND {filtre_dvf} AND type_local IN ('Maison','Appartement');
""", con=moteur)
donnees = donnees.drop_duplicates(subset=['id_parcelle','prix_m2','surface_reelle_bati']).reset_index(drop=True)
if len(donnees) == 0:
    print("Aucune donnee."); sys.exit()

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
    FROM dpe_logements_france WHERE etiquette_dpe IN ('A','B','C','D','E','F','G') AND {filtre_dpe}
    GROUP BY code_insee_ban;
""", con=moteur)

stations = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur)
if dep_infra == 'FRANCE':
    q_mon="SELECT latitude,longitude FROM monuments_historiques WHERE latitude IS NOT NULL;"
    q_hop="SELECT latitude,longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL;"
    q_uni="SELECT latitude,longitude,nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;"
else:
    q_mon=f"SELECT latitude,longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{dep_infra}';"
    q_hop=f"SELECT latitude,longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal,2)='{dep_infra}';"
    q_uni=f"SELECT latitude,longitude,nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{dep_infra}';"
monuments = pd.read_sql(q_mon, con=moteur)
hopitaux = pd.read_sql(q_hop, con=moteur)
universites = pd.read_sql(q_uni, con=moteur)

filtre_rev = "1=1" if dep_infra=='FRANCE' else f"LEFT(code_commune,2)='{dep_infra}'"
revenus = pd.read_sql(f"SELECT code_commune, median_revenu_disponible, indice_gini, pct_minima_sociaux FROM demographie_communes WHERE {filtre_rev};", con=moteur)
for c in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']:
    revenus[c] = pd.to_numeric(revenus[c], errors='coerce')

poles = None
if FA['potentiel_urbain']:
    poles = pd.read_sql("""SELECT aav_nom, AVG(latitude) AS latitude, AVG(longitude) AS longitude, COUNT(*) AS poids_aire
        FROM referentiel_communes WHERE aav_nom IS NOT NULL AND aav_nom!='SO' AND latitude IS NOT NULL
        GROUP BY aav_nom HAVING poids_aire>=10""", con=moteur)
    pe = pd.DataFrame([{'aav_nom':'Genève','latitude':46.2044,'longitude':6.1432,'poids_aire':250},
                       {'aav_nom':'Lausanne','latitude':46.5197,'longitude':6.6323,'poids_aire':90}])
    poles = pd.concat([poles, pe], ignore_index=True)
    for c in ['latitude','longitude','poids_aire']:
        poles[c] = pd.to_numeric(poles[c], errors='coerce')
    poles = poles.dropna(subset=['latitude','longitude','poids_aire'])

# =====================================================================
# FUSION + DISTANCES
# =====================================================================
print("Fusion et distances...")
donnees = pd.merge(donnees, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees, revenus, on='code_commune', how='left')
donnees = donnees.reset_index(drop=True)
points_rad = np.deg2rad(donnees[['latitude','longitude']])

def dist_min(df_points, col):
    if len(df_points) > 0:
        arbre = BallTree(np.deg2rad(df_points.iloc[:,0:2]), metric='haversine')
        d,_ = arbre.query(points_rad, k=1)
        donnees[col] = d.flatten()*RAYON
    else:
        donnees[col] = 999999

dist_min(stations,'dist_transport_m'); dist_min(monuments,'dist_monument_m'); dist_min(hopitaux,'dist_hopital_m')
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
    dist_min(contour(gdf_littoral[gdf_littoral['CLASSEMENT']==cl]), col)

if poles is not None and len(poles) > 0:
    ap = BallTree(np.deg2rad(poles[['latitude','longitude']].values), metric='haversine')
    pp = poles['poids_aire'].values.astype(float)
    kp = min(20, len(poles))
    drp, ip = ap.query(points_rad, k=kp)
    donnees['potentiel_urbain'] = np.sum(pp[ip]/(drp*RAYON+5000), axis=1)

# =====================================================================
# NETTOYAGE + FEATURES DERIVEES
# =====================================================================
print("Nettoyage...")
cols_dpe=['pct_dpe_A','pct_dpe_B','pct_dpe_C','pct_dpe_D','pct_dpe_E','pct_dpe_F','pct_dpe_G']
cols_chauf=['pct_chauffage_elec','pct_chauffage_gaz','pct_chauffage_fioul','pct_chauffage_urbain']
cols_rev=['median_revenu_disponible','indice_gini','pct_minima_sociaux']
for c in cols_dpe+cols_chauf+cols_rev:
    if c in donnees.columns: donnees[c]=donnees[c].fillna(donnees[c].median())
donnees['volume_etudiants_proche']=donnees['volume_etudiants_proche'].fillna(0)
donnees['surface_terrain']=donnees['surface_terrain'].fillna(0)
if 'potentiel_urbain' in donnees.columns:
    donnees['potentiel_urbain']=donnees['potentiel_urbain'].fillna(donnees['potentiel_urbain'].median())

dp = donnees[(donnees['surface_reelle_bati']>=9)&(donnees['surface_reelle_bati']<=300)].copy()
dp.loc[dp['type_local']=='Appartement','surface_terrain']=0
dp['log_prix_m2']=np.log(dp['prix_m2'])
dp['log_surface']=np.log(dp['surface_reelle_bati'])
dp['surface_par_piece']=dp['surface_reelle_bati']/dp['nombre_pieces_principales']
dp['a_terrain']=(dp['surface_terrain']>0).astype(int)
dp['log_terrain']=np.log1p(dp['surface_terrain'])
dp['code_section']=dp['id_parcelle'].str[:10]

# Liste de features selon les interrupteurs
def build_features():
    f=[]
    if FA['geo_base']: f+=['latitude','longitude']
    if FA['surface']: f+=['surface_reelle_bati','log_surface','surface_par_piece']
    if FA['pieces']: f+=['nombre_pieces_principales']
    if FA['terrain']: f+=['surface_terrain','log_terrain','a_terrain']
    if FA['date']: f+=['annee_vente','mois_vente']
    if FA['dpe']: f+=cols_dpe
    if FA['chauffage']: f+=cols_chauf
    if FA['revenus']: f+=cols_rev
    if FA['dist_transport']: f+=['dist_transport_m']
    if FA['dist_monument']: f+=['dist_monument_m']
    if FA['dist_hopital']: f+=['dist_hopital_m']
    if FA['dist_universite']: f+=['dist_universite_m','volume_etudiants_proche']
    if FA['dist_littoral']: f+=['dist_mer_m','dist_lac_m','dist_estuaire_m']
    if FA['potentiel_urbain'] and 'potentiel_urbain' in dp.columns: f+=['potentiel_urbain']
    return f
features_base = build_features()
print(f"\n{len(features_base)} features de base : {features_base}")

# =====================================================================
# SAISIE DES BORNES DE PRIX (plancher / plafond)
# =====================================================================
# On affiche les quantiles 0.01 et 0.99 du prix au m2, par type de bien,
# pour aider l'utilisateur a choisir des bornes eclairees.
print("\n" + "=" * 55)
print("BORNES DE PRIX AU M2 (filtrage des aberrations)")
print("=" * 55)
print("Quantiles observes du prix au m2 par type :")
for tb in ['Maison', 'Appartement']:
    sous = dp[dp['type_local'] == tb]
    if len(sous) > 0:
        q01 = sous['prix_m2'].quantile(0.01)
        q99 = sous['prix_m2'].quantile(0.99)
        print(f"  {tb:12s} : q01 = {q01:.0f} EUR/m2  |  q99 = {q99:.0f} EUR/m2  (median {sous['prix_m2'].median():.0f})")

def saisir_borne(question, defaut):
    rep = input(f"  {question} (Entree = {defaut:.0f}) : ").strip().replace(',', '.')
    if rep == "":
        return defaut
    try:
        return float(rep)
    except ValueError:
        print(f"    Valeur invalide, on garde {defaut:.0f}.")
        return defaut

# Valeurs par defaut : les memes garde-fous qu'avant (800 et 15000)
print("\nSaisissez les bornes a appliquer (communes aux deux types) :")
PLANCHER_PRIX = saisir_borne("Prix plancher (EUR/m2)", 800)
PLAFOND_PRIX = saisir_borne("Prix plafond (EUR/m2)", 15000)
print(f"\nBornes retenues : {PLANCHER_PRIX:.0f} - {PLAFOND_PRIX:.0f} EUR/m2")

# =====================================================================
# ENTRAINEMENT + TEST PAR TYPE
# =====================================================================
def entrainer_et_tester(type_bien):
    df_bien = dp[dp['type_local']==type_bien].copy()
    if len(df_bien) < 100:
        print(f"\n{type_bien} : pas assez de donnees."); return None

    # Bornes : on combine le choix utilisateur avec les quantiles du type
    # (on ne descend jamais sous q01 ni au-dessus de q99 pour rester coherent)
    plancher = max(df_bien['prix_m2'].quantile(0.01), PLANCHER_PRIX)
    plafond = min(df_bien['prix_m2'].quantile(0.99), PLAFOND_PRIX)
    df_bien=df_bien[(df_bien['prix_m2']>=plancher)&(df_bien['prix_m2']<=plafond)].copy()

    # Features locales (voisins/section) calculees sur TOUT df_bien (train complet en prod)
    feats = list(features_base)
    coords = np.deg2rad(df_bien[['latitude','longitude']])
    prix_all = df_bien['prix_m2'].values
    surf_all = df_bien['surface_reelle_bati'].values
    arbre_v = BallTree(coords, metric='haversine')

    if FA['voisins']:
        def vois(dist, idx, sb):
            dist,idx = dist[1:], idx[1:]
            if len(idx)==0: return np.nan
            pv,sv = prix_all[idx], surf_all[idx]
            m=(sv>=sb*0.6)&(sv<=sb*1.4)
            if m.sum()>=3: d,p=dist[m],pv[m]
            else: d,p=dist,pv
            w=1.0/(d+1e-9); return np.sum(w*p)/np.sum(w)
        k=min(41,len(coords))
        dv,iv = arbre_v.query(coords, k=k)
        df_bien['prix_m2_voisins'] = [vois(dv[i],iv[i],surf_all[i]) for i in range(len(iv))]
        feats += ['prix_m2_voisins']
    if FA['densite']:
        rr=1000/RAYON
        df_bien['densite_ventes_1km']=arbre_v.query_radius(coords, r=rr, count_only=True)
        feats += ['densite_ventes_1km']
    if FA['section']:
        ms=df_bien.groupby('code_section')['prix_m2'].median()
        mc=df_bien.groupby('code_commune')['prix_m2'].median()
        mg=df_bien['prix_m2'].median()
        df_bien['prix_m2_section']=df_bien['code_section'].map(ms).fillna(df_bien['code_commune'].map(mc)).fillna(mg)
        nvs=df_bien.groupby('code_section').size()
        df_bien['nb_ventes_section']=df_bien['code_section'].map(nvs).fillna(0)
        feats += ['prix_m2_section','nb_ventes_section']

    X = df_bien[feats]; y = df_bien['log_prix_m2']

    # Entrainement quantile
    X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=0.3,random_state=42)
    modeles={}
    for nom,alpha in {'bas':0.025,'median':0.50,'haut':0.975}.items():
        m=CatBoostRegressor(loss_function=f'Quantile:alpha={alpha}',iterations=1000,
                            learning_rate=0.04,depth=8,random_seed=42,l2_leaf_reg=3.0,
                            early_stopping_rounds=50,verbose=False)
        m.fit(X_tr,y_tr,eval_set=(X_val,y_val),use_best_model=True)
        modeles[nom]=m

    # Tirage de nb_test biens reels
    biens_test = df_bien.sample(n=min(nb_test,len(df_bien)), random_state=1)

    print("\n"+"="*100)
    print(f"TESTS - {type_bien.upper()} ({len(biens_test)} biens)")
    print("="*100)
    print(f"{'#':>3} {'idDVF':>9} {'Surf':>5} {'Terr':>6} {'PrixReel':>10} {'PrixEstime':>11} {'Bas':>10} {'Haut':>10} {'Err%':>6}  Statut")
    print("-"*100)

    res=[]
    for i,(_,b) in enumerate(biens_test.iterrows(),1):
        xb = b[feats].to_frame().T
        m2_med = float(np.exp(modeles['median'].predict(xb)[0]))
        m2_bas = float(np.exp(modeles['bas'].predict(xb)[0]))
        m2_haut = float(np.exp(modeles['haut'].predict(xb)[0]))
        m2_bas, m2_haut = min(m2_bas,m2_haut), max(m2_bas,m2_haut)
        surf=float(b['surface_reelle_bati'])
        total_reel=float(b['valeur_fonciere'])
        total_est=m2_med*surf
        total_bas=m2_bas*surf; total_haut=m2_haut*surf
        err=abs(total_est-total_reel)/total_reel
        ok = err<=MARGE
        terr = f"{b['surface_terrain']:.0f}" if b['surface_terrain']>0 else "-"
        print(f"{i:>3} {str(b['id']):>9} {surf:>5.0f} {terr:>6} {total_reel:>10.0f} {total_est:>11.0f} {total_bas:>10.0f} {total_haut:>10.0f} {err*100:>5.1f}%  {'OK' if ok else 'HORS'}")
        res.append({'err':err,'ok':ok,'dans':total_bas<=total_reel<=total_haut})

    d=pd.DataFrame(res)
    nb_ok=int(d['ok'].sum())
    print("-"*100)
    print(f"RESUME {type_bien.upper()} : {nb_ok} OK / {len(d)-nb_ok} hors marge sur {len(d)} | "
          f"MAPE {d['err'].mean()*100:.1f} % | couverture {d['dans'].mean()*100:.1f} %")
    return d

resultats_types = {}
for tb in ['Maison','Appartement']:
    resultats_types[tb] = entrainer_et_tester(tb)

# =====================================================================
# BILAN GLOBAL (les deux types ensemble)
# =====================================================================
print("\n" + "=" * 100)
print("BILAN GLOBAL - MAISONS + APPARTEMENTS")
print("=" * 100)

total_ok = 0
total_tests = 0
for tb, d in resultats_types.items():
    if d is None:
        print(f"{tb:12s} : non teste (pas assez de donnees)")
        continue
    nb_ok = int(d['ok'].sum())
    nb_hors = len(d) - nb_ok
    total_ok += nb_ok
    total_tests += len(d)
    print(f"{tb:12s} : {nb_ok} OK / {nb_hors} hors marge sur {len(d)} | "
          f"MAPE {d['err'].mean()*100:>5.1f} % | couverture {d['dans'].mean()*100:>5.1f} %")

if total_tests > 0:
    print("-" * 100)
    print(f"{'TOTAL':12s} : {total_ok} OK / {total_tests - total_ok} hors marge sur {total_tests} "
          f"({total_ok / total_tests * 100:.1f} % dans la marge de {int(MARGE*100)} %)")

print(f"\nTemps total : {time.time()-temps_debut:.2f}s | {len(features_base)} features de base")
print(f"\n{len(features_base)} features de base : {features_base}")
