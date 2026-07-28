"""
COMPARAISON DES VARIANTES D'INFLATION comme feature
====================================================
Teste differentes facons d'encoder l'inflation, sur la baseline, pour voir si
l'une bat la version glissante actuelle (ou n'apporte rien, ce qui est l'hypothese).

Contexte : la table taux_macro contient 'taux_inflation' = glissement annuel
(hausse des prix sur 12 mois), valeur mensuelle. On ne dispose PAS de l'indice
IPC brut, donc l'inflation "instantanee" est APPROCHEE a partir du glissant.

Variantes testees (ajoutees a baseline lat/lon + surface + date) :
  0. BASELINE seule (reference)
  1. INFLATION GLISSANTE    : taux_inflation (glissement 12 mois, version actuelle)
  2. INFLATION INSTANTANEE  : variation du taux glissant d'un mois a l'autre
                              (derivee = acceleration/deceleration de l'inflation)
  3. INDICE PRIX RECONSTRUIT: cumul du glissant -> niveau de prix implicite
                              (approx d'un indice base 100, facon "deflateur")

NB : l'instantanee exacte exigerait l'IPC brut (non disponible) ; la variante 2
est une approximation par la variation du glissant. Interpretation prudente.

Saisie unique (departement). Split aleatoire 70/30. Cible prix total (log).
"""

import os, sys, time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
    moteur_dvf = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur_enr = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur_dvf.connect().close()
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

departement = input("Departement a tester (ex: 34) : ").strip()

print("\nExtraction des ventes...")
df = pd.read_sql(f"""
    SELECT communes_code AS code_commune, lat AS latitude, lng AS longitude,
           prix_m2, surface AS surface_reelle_bati, typebien AS type_local,
           YEAR(date) AS annee_vente, MONTH(date) AS mois_vente
    FROM synthese
    WHERE departements_code = '{departement}'
      AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
      AND typebien IN ('maison', 'appartement');
""", con=moteur_dvf)
for c in ['prix_m2','surface_reelle_bati','latitude','longitude','annee_vente','mois_vente']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['prix_m2','surface_reelle_bati','latitude','longitude','annee_vente','mois_vente'])
df['type_local'] = df['type_local'].str.capitalize()
df['prix_total'] = df['prix_m2'] * df['surface_reelle_bati']
df['log_prix_total'] = np.log(df['prix_total'])

# --- Construction des variantes d'inflation a partir de la serie mensuelle ---
print("Construction des variantes d'inflation...")
infl = pd.read_sql("SELECT annee, mois, taux_inflation FROM taux_macro", con=moteur_enr)
infl['taux_inflation'] = pd.to_numeric(infl['taux_inflation'], errors='coerce')
infl = infl.dropna(subset=['taux_inflation']).sort_values(['annee','mois']).reset_index(drop=True)

# Variante 1 : glissante (telle quelle)
infl['infl_glissante'] = infl['taux_inflation']

# Variante 2 : instantanee approchee = variation du taux glissant d'un mois a l'autre
#   (positive = inflation qui accelere ; negative = qui ralentit)
infl['infl_instantanee'] = infl['taux_inflation'].diff().fillna(0)

# Variante 3 : indice de prix reconstruit par cumul (approx d'un niveau base 100)
#   On cumule l'inflation mensuelle implicite (glissant/12) pour approcher un indice.
#   -> capture le NIVEAU de prix cumule, facon deflateur.
infl['infl_mensuelle_approx'] = infl['taux_inflation'] / 12.0
infl['indice_prix'] = 100 * (1 + infl['infl_mensuelle_approx']/100).cumprod()

# Jointure aux ventes par annee-mois
df = pd.merge(df, infl[['annee','mois','infl_glissante','infl_instantanee','indice_prix']],
              left_on=['annee_vente','mois_vente'], right_on=['annee','mois'], how='left')
for c in ['infl_glissante','infl_instantanee','indice_prix']:
    df[c] = df[c].fillna(df[c].median())

CONFIGS = {
    '0. baseline seule':      [],
    '1. infl. glissante':     ['infl_glissante'],
    '2. infl. instantanee':   ['infl_instantanee'],
    '3. indice prix (cumul)': ['indice_prix'],
}
FIXES = ['latitude','longitude','surface_reelle_bati','annee_vente','mois_vente']

def filtrer(d0):
    pl,pf = d0['prix_m2'].quantile(0.01), d0['prix_m2'].quantile(0.99)
    d = d0[(d0['prix_m2']>=pl)&(d0['prix_m2']<=pf)].copy()
    sc = d.groupby('code_commune')['prix_m2'].agg(['median','size'])
    ref = d['code_commune'].map(sc['median']); nc=d['code_commune'].map(sc['size'])
    ref = ref.where(nc>=10, d['prix_m2'].median())
    return d[(d['prix_m2']/ref).between(0.40,2.50)].copy()

def evaluer(df_bien, cols):
    d = filtrer(df_bien)
    if len(d) < 300: return None
    feats = FIXES + cols if cols else FIXES
    X,y = d[feats], d['log_prix_total']
    Xtr_,Xte,ytr_,yte = train_test_split(X,y,test_size=0.30,random_state=42)
    Xtr,Xval,ytr,yval = train_test_split(Xtr_,ytr_,test_size=0.2,random_state=42)
    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1000,
                          learning_rate=0.04, depth=8, random_seed=42,
                          early_stopping_rounds=50, verbose=False)
    m.fit(Xtr,ytr,eval_set=(Xval,yval),use_best_model=True)
    tr=np.exp(yte).values; pr=np.exp(m.predict(Xte))
    return np.sqrt(np.mean((np.log1p(pr)-np.log1p(tr))**2))

# ======= EXECUTION =======
# Correlations des variantes avec l'annee (pour montrer la redondance)
print("\n--- Correlations avec annee_vente (redondance) ---")
for c in ['infl_glissante','infl_instantanee','indice_prix']:
    corr = df[[c,'annee_vente']].corr().iloc[0,1]
    print(f"  {c:22s} : {corr:+.3f}")

print("\n" + "="*64)
print(f"VARIANTES D'INFLATION - Departement {departement}")
print(f"Baseline = {FIXES}")
print("="*64)
print(f"{'Variante':<24} {'RMSLE maison':>13} {'RMSLE appart':>13}")
print("-"*64)
t0=time.time()
df_m = df[df['type_local']=='Maison']; df_a = df[df['type_local']=='Appartement']
ref_m=ref_a=None
for nom,cols in CONFIGS.items():
    rm=evaluer(df_m,cols); ra=evaluer(df_a,cols)
    marque=""
    if nom.startswith('0.'): ref_m,ref_a=rm,ra
    elif ref_m and rm and ra:
        d=((rm-ref_m)+(ra-ref_a))/2
        marque = f"  ({d:+.4f} vs baseline)"
    print(f"{nom:<24} {rm:>13.4f} {ra:>13.4f}{marque}")

print("\n" + "="*64)
print("Un ecart negatif vs baseline = la variante ameliore.")
print("Un ecart ~0 = aucun apport (attendu : inflation redondante avec annee).")
print(f"Temps total : {time.time()-t0:.0f}s")