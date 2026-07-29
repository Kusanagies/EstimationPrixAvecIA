"""
TEST DE PARCIMONIE : modele complet vs combinaisons ciblees
============================================================
Compare le modele COMPLET (toutes les features) a des versions simplifiees,
pour voir combien de features suffisent a atteindre (presque) la meme performance.

Base sur le classement d'impact : voisins, terrain, section = vrais contributeurs ;
distances/potentiel/densite = apport modeste ; revenus/dpe/chauffage/eco = quasi nul.

Configurations comparees :
  A. BASELINE            : lat/lon + surface + date
  B. + VOISINS           : baseline + prix_m2_voisins
  C. MINIMAL             : baseline + voisins + terrain + section  (les 3 gros gains)
  D. MINIMAL + GEO       : C + distances (littoral/transport/hopital) + potentiel + densite
  E. COMPLET SANS ECO    : toutes les features sauf economiques (chomage/taux/pib)
  F. COMPLET             : absolument toutes les features

Objectif : si C ou D egale F, on demontre qu'un modele parcimonieux suffit.

Saisie unique (departement). Split aleatoire 70/30. Cible prix total (log).
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

departement = input("Departement a tester (ex: 34) : ").strip()
filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
RAYON = 6371000

print("\nChargement (une seule fois)...")
gdf = gpd.read_file(CHEMIN_GPKG)
if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)

ventes = pd.read_sql(f"""
    SELECT communes_code AS code_commune, parcelles_code AS id_parcelle,
           lat AS latitude, lng AS longitude, prix_m2,
           surface AS surface_reelle_bati, typebien AS type_local,
           nb_pieces AS nombre_pieces_principales, surface_terrain,
           YEAR(date) AS annee_vente, MONTH(date) AS mois_vente
    FROM synthese
    WHERE departements_code = '{departement}'
      AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
      AND typebien IN ('maison', 'appartement');
""", con=moteur_dvf)
for c in ['prix_m2','surface_reelle_bati','surface_terrain','nombre_pieces_principales',
          'latitude','longitude','annee_vente','mois_vente']:
    ventes[c] = pd.to_numeric(ventes[c], errors='coerce')
ventes = ventes.dropna(subset=['prix_m2','surface_reelle_bati','nombre_pieces_principales','latitude','longitude'])
ventes['type_local'] = ventes['type_local'].str.capitalize()

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
""", con=moteur_enr)
revenus = pd.read_sql(f"SELECT code_commune, median_revenu_disponible, indice_gini, pct_minima_sociaux FROM demographie_communes WHERE LEFT(code_commune,2)='{departement}';", con=moteur_enr)
for c in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']:
    revenus[c] = pd.to_numeric(revenus[c], errors='coerce')
densite_pop = pd.read_sql("""SELECT d.code_commune, d.densite_population FROM densite_population d
    INNER JOIN (SELECT code_commune, MAX(annee) a FROM densite_population GROUP BY code_commune) m
    ON d.code_commune=m.code_commune AND d.annee=m.a""", con=moteur_enr)
densite_pop['densite_population'] = pd.to_numeric(densite_pop['densite_population'], errors='coerce')
stations = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur_enr)
hopitaux = pd.read_sql(f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal,2)='{departement}';", con=moteur_enr)
monuments = pd.read_sql(f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{departement}';", con=moteur_enr)
universites = pd.read_sql(f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{departement}';", con=moteur_enr)
poles = pd.read_sql("""SELECT aav_nom, AVG(latitude) latitude, AVG(longitude) longitude, COUNT(*) poids_aire
    FROM referentiel_communes WHERE aav_nom IS NOT NULL AND aav_nom!='SO' AND latitude IS NOT NULL
    GROUP BY aav_nom HAVING poids_aire>=10""", con=moteur_enr)
for c in ['latitude','longitude','poids_aire']:
    poles[c] = pd.to_numeric(poles[c], errors='coerce')
poles = poles.dropna()

# Fusion + distances
d = pd.merge(ventes, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
d = pd.merge(d, revenus, on='code_commune', how='left')
d = pd.merge(d, densite_pop, on='code_commune', how='left')
pr = np.deg2rad(d[['latitude','longitude']])
def dmin(dfp, col):
    if len(dfp)>0:
        a=BallTree(np.deg2rad(dfp.iloc[:,0:2]),metric='haversine'); dd,_=a.query(pr,k=1); d[col]=dd.flatten()*RAYON
    else: d[col]=999999
dmin(stations,'dist_transport_m'); dmin(monuments,'dist_monument_m'); dmin(hopitaux,'dist_hopital_m')
if len(universites)>0:
    a=BallTree(np.deg2rad(universites[['latitude','longitude']]),metric='haversine'); dd,iu=a.query(pr,k=1)
    d['dist_universite_m']=dd.flatten()*RAYON; d['volume_etudiants_proche']=universites.iloc[iu.flatten()]['nombre_etudiants'].values
else: d['dist_universite_m']=999999; d['volume_etudiants_proche']=0
def contour(sg):
    pts=[]
    for g in sg.geometry:
        if g.geom_type=='MultiPolygon':
            for p in g.geoms: pts.extend(list(p.exterior.coords))
        else: pts.extend(list(g.exterior.coords))
    if not pts: return pd.DataFrame(columns=['latitude','longitude'])
    ar=np.array(pts); return pd.DataFrame(ar[:,[1,0]],columns=['latitude','longitude'])
for cl,col in {'Mer':'dist_mer_m','Lac':'dist_lac_m','Estuaire':'dist_estuaire_m'}.items():
    dmin(contour(gdf[gdf['CLASSEMENT']==cl]),col)
if len(poles)>0:
    ap=BallTree(np.deg2rad(poles[['latitude','longitude']].values),metric='haversine')
    pp=poles['poids_aire'].values.astype(float); drp,ip=ap.query(pr,k=min(20,len(poles)))
    d['potentiel_urbain']=np.sum(pp[ip]/(drp*RAYON+5000),axis=1)
else: d['potentiel_urbain']=0

cdpe=['pct_dpe_A','pct_dpe_B','pct_dpe_C','pct_dpe_D','pct_dpe_E','pct_dpe_F','pct_dpe_G']
cchauf=['pct_chauffage_elec','pct_chauffage_gaz','pct_chauffage_fioul','pct_chauffage_urbain']
crev=['median_revenu_disponible','indice_gini','pct_minima_sociaux']
for col in cdpe+cchauf+crev+['densite_population']:
    if col in d.columns: d[col]=d[col].fillna(d[col].median())
d['volume_etudiants_proche']=d['volume_etudiants_proche'].fillna(0)
d['surface_terrain']=d['surface_terrain'].fillna(0)
d['potentiel_urbain']=d['potentiel_urbain'].fillna(d['potentiel_urbain'].median())

dp = d[(d['surface_reelle_bati']>=9)&(d['surface_reelle_bati']<=300)].copy()
dp.loc[dp['type_local']=='Appartement','surface_terrain']=0
dp['prix_total']=dp['prix_m2']*dp['surface_reelle_bati']
dp['log_prix_total']=np.log(dp['prix_total'])
dp['log_surface']=np.log(dp['surface_reelle_bati'])
dp['surface_par_piece']=dp['surface_reelle_bati']/dp['nombre_pieces_principales']
dp['a_terrain']=(dp['surface_terrain']>0).astype(int)
dp['log_terrain']=np.log1p(dp['surface_terrain'])
dp['code_section']=dp['id_parcelle'].str[:10]

# Definition des blocs de features
BASE = ['latitude','longitude','surface_reelle_bati','log_surface','surface_par_piece','annee_vente','mois_vente']
TERRAIN = ['a_terrain','surface_terrain','log_terrain']
DIST = ['dist_transport_m','dist_monument_m','dist_hopital_m','dist_universite_m','volume_etudiants_proche','dist_mer_m','dist_lac_m','dist_estuaire_m']
ENRICH = cdpe + cchauf + crev + ['densite_population','potentiel_urbain','nombre_pieces_principales']
# 'voisins' et 'section' sont calcules apres split (marqueurs)

CONFIGS = {
    'A. baseline':          {'simples': BASE, 'spatial': []},
    'B. + voisins':         {'simples': BASE, 'spatial': ['voisins']},
    'C. minimal':           {'simples': BASE + TERRAIN, 'spatial': ['voisins','section']},
    'D. minimal + geo':     {'simples': BASE + TERRAIN + DIST + ['potentiel_urbain','densite_population'], 'spatial': ['voisins','section','densite']},
    'E. complet sans eco':  {'simples': BASE + TERRAIN + DIST + ENRICH, 'spatial': ['voisins','section','densite']},
    'F. complet':           {'simples': BASE + TERRAIN + DIST + ENRICH, 'spatial': ['voisins','section','densite']},
    # (E et F identiques ici car les features eco n'etaient pas activees dans ce socle ;
    #  F sert de reference "tout ce qu'on a". Les eco ont deja montre gain nul/negatif.)
}

def ajouter_spatial(Xtr, Xte, d_sub, idx_tr, idx_te, spat):
    """Ajoute les features spatiales (voisins/section/densite) sans leakage."""
    coords_tr = np.deg2rad(d_sub.loc[idx_tr, ['latitude','longitude']])
    arbre = BallTree(coords_tr, metric='haversine')
    Xtr, Xte = Xtr.copy(), Xte.copy()
    if 'voisins' in spat:
        prix_tr = d_sub.loc[idx_tr,'prix_m2'].values; surf_tr=d_sub.loc[idx_tr,'surface_reelle_bati'].values
        def vois(dr,idx,sb,self_i=None):
            if self_i is not None:
                k=idx!=self_i; dr,idx=dr[k],idx[k]
            if len(idx)==0: return np.nan
            dm=dr*RAYON; pv,sv=prix_tr[idx],surf_tr[idx]; m=(sv>=sb*0.6)&(sv<=sb*1.4)
            if m.sum()>=3: dd,pp=dm[m],pv[m]
            else: dd,pp=dm,pv
            w=1.0/(dd+50.0); return np.sum(w*pp)/np.sum(w)
        k=min(41,len(coords_tr)); dtr,itr=arbre.query(coords_tr,k=k)
        sb_tr=d_sub.loc[idx_tr,'surface_reelle_bati'].values
        Xtr['prix_m2_voisins']=[vois(dtr[i],itr[i],sb_tr[i],self_i=i) for i in range(len(itr))]
        ct=np.deg2rad(d_sub.loc[idx_te,['latitude','longitude']]); dte,ite=arbre.query(ct,k=min(40,len(coords_tr)))
        sb_te=d_sub.loc[idx_te,'surface_reelle_bati'].values
        Xte['prix_m2_voisins']=[vois(dte[i],ite[i],sb_te[i]) for i in range(len(ite))]
    if 'densite' in spat:
        rr=1000/RAYON; ct=np.deg2rad(d_sub.loc[idx_te,['latitude','longitude']])
        Xtr['densite_ventes_1km']=arbre.query_radius(coords_tr,r=rr,count_only=True)
        Xte['densite_ventes_1km']=arbre.query_radius(ct,r=rr,count_only=True)
    if 'section' in spat:
        dtr_=d_sub.loc[idx_tr].copy()
        ms=dtr_.groupby('code_section')['prix_m2'].median(); mc=dtr_.groupby('code_commune')['prix_m2'].median(); mg=dtr_['prix_m2'].median()
        vals=pd.Series(np.nan,index=idx_tr); kf=KFold(n_splits=5,shuffle=True,random_state=42)
        arr=np.array(idx_tr)
        for pf_,po_ in kf.split(arr):
            s=dtr_.loc[arr[pf_]].groupby('code_section')['prix_m2'].median()
            c=dtr_.loc[arr[pf_]].groupby('code_commune')['prix_m2'].median()
            sous=dtr_.loc[arr[po_]]
            v=sous['code_section'].map(s).fillna(sous['code_commune'].map(c)).fillna(dtr_.loc[arr[pf_]]['prix_m2'].median())
            vals.loc[arr[po_]]=v.values
        Xtr['prix_m2_section']=vals.values
        st=d_sub.loc[idx_te,'code_section'].map(ms); ce=d_sub.loc[idx_te,'code_commune'].map(mc)
        Xte['prix_m2_section']=st.fillna(ce).fillna(mg).values
    return Xtr, Xte

def filtrer(db):
    pl,pf=db['prix_m2'].quantile(0.01),db['prix_m2'].quantile(0.99)
    dd=db[(db['prix_m2']>=pl)&(db['prix_m2']<=pf)].copy()
    sc=dd.groupby('code_commune')['prix_m2'].agg(['median','size'])
    ref=dd['code_commune'].map(sc['median']); nc=dd['code_commune'].map(sc['size'])
    ref=ref.where(nc>=10,dd['prix_m2'].median())
    return dd[(dd['prix_m2']/ref).between(0.40,2.50)].copy()

def evaluer(db, cfg):
    dd = filtrer(db)
    if len(dd) < 300: return None, 0
    X = dd[cfg['simples']].copy(); y = dd['log_prix_total']
    idx_tr, idx_te = train_test_split(dd.index, test_size=0.30, random_state=42)
    X_tr, X_te = X.loc[idx_tr], X.loc[idx_te]
    if cfg['spatial']:
        X_tr, X_te = ajouter_spatial(X_tr, X_te, dd, idx_tr, idx_te, cfg['spatial'])
    y_tr, y_te = y.loc[idx_tr], y.loc[idx_te]
    Xt, Xv, yt, yv = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1000,
                          learning_rate=0.04, depth=8, random_seed=42,
                          early_stopping_rounds=50, verbose=False)
    m.fit(Xt, yt, eval_set=(Xv, yv), use_best_model=True)
    tr=np.exp(y_te).values; prd=np.exp(m.predict(X_te))
    return np.sqrt(np.mean((np.log1p(prd)-np.log1p(tr))**2)), X_tr.shape[1]

# ======= EXECUTION =======
print("\n" + "="*74)
print(f"TEST DE PARCIMONIE - Departement {departement}")
print("="*74)
print(f"{'Configuration':<22} {'nb feat.':>9} {'RMSLE maison':>14} {'RMSLE appart':>14}")
print("-"*74)
t0=time.time()
dm=dp[dp['type_local']=='Maison']; da=dp[dp['type_local']=='Appartement']
res=[]
for nom,cfg in CONFIGS.items():
    rm,nf = evaluer(dm,cfg); ra,_ = evaluer(da,cfg)
    res.append((nom,nf,rm,ra))
    print(f"{nom:<22} {nf:>9} {rm:>14.4f} {ra:>14.4f}")

# Comparaison au complet
print("\n" + "="*74)
print("ECART AU MODELE COMPLET (F)")
print("="*74)
cf = res[-1]
print(f"{'Configuration':<22} {'nb feat.':>9} {'ecart maison':>14} {'ecart appart':>14}")
print("-"*74)
for nom,nf,rm,ra in res:
    dm_=rm-cf[2]; da_=ra-cf[3]
    print(f"{nom:<22} {nf:>9} {dm_:>+14.4f} {da_:>+14.4f}")
print("\nUn ecart proche de 0 = cette config parcimonieuse egale le modele complet.")
print(f"Temps total : {time.time()-t0:.0f}s")