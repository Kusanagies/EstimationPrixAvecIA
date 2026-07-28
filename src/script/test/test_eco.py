"""
DATE vs VARIABLES ECONOMIQUES dans la dimension TEMPS
=====================================================
Teste l'hypothese du tuteur : remplacer la feature date (annee/mois) par des
variables economiques liees au temps (taux d'interet, inflation) ameliore-t-il
le modele ?

Deux tests complementaires :

  TEST A (split aleatoire 70/30) : compare, a features geo egales, differentes
    facons d'encoder le temps. Montre s'il y a une difference "a periode connue".

  TEST B (generalisation temporelle) : entraine sur annees ANCIENNES, teste sur
    une annee RECENTE eloignee. Revele l'avantage theorique des variables eco :
    generaliser a une annee non vue via une relation causale, la ou la date
    (simple identifiant memorise) ne le peut pas.

Configurations comparees dans chaque test :
  - DATE : annee_vente + mois_vente (reference)
  - ECO  : taux_credit_immo_fixe + taux_inflation (remplace la date)
  - DATE+ECO : les deux ensemble
  - SANS TEMPS : aucune variable temporelle (pour mesurer l'apport du temps)

Features fixes (dans tous les cas) : lat/lon + surface.
Source : synthese (ventes) + EstimationIA (taux). Cible : prix total (log).
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
annee_holdout = input("Annee de test pour la generalisation (ex: 2025) : ").strip()
annee_holdout = int(annee_holdout) if annee_holdout else 2025

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

# Variables economiques (mensuelles, nationales)
taux = pd.read_sql("SELECT annee, mois, taux_credit_immo_fixe, taux_inflation FROM taux_macro", con=moteur_enr)
df = pd.merge(df, taux, left_on=['annee_vente','mois_vente'],
              right_on=['annee','mois'], how='left').drop(columns=['annee','mois'], errors='ignore')
for c in ['taux_credit_immo_fixe','taux_inflation']:
    df[c] = df[c].fillna(df[c].median())

df['prix_total'] = df['prix_m2'] * df['surface_reelle_bati']
df['log_prix_total'] = np.log(df['prix_total'])

FEATURES_FIXES = ['latitude', 'longitude', 'surface_reelle_bati']
CONFIGS = {
    'SANS TEMPS':  [],
    'DATE':        ['annee_vente', 'mois_vente'],
    'ECO':         ['taux_credit_immo_fixe', 'taux_inflation'],
    'DATE + ECO':  ['annee_vente', 'mois_vente', 'taux_credit_immo_fixe', 'taux_inflation'],
}

def filtrer(df_bien):
    pl, pf = df_bien['prix_m2'].quantile(0.01), df_bien['prix_m2'].quantile(0.99)
    d = df_bien[(df_bien['prix_m2']>=pl)&(df_bien['prix_m2']<=pf)].copy()
    sc = d.groupby('code_commune')['prix_m2'].agg(['median','size'])
    ref = d['code_commune'].map(sc['median'])
    nc = d['code_commune'].map(sc['size'])
    ref = ref.where(nc>=10, d['prix_m2'].median())
    return d[(d['prix_m2']/ref).between(0.40,2.50)].copy()

def rmsle(y_true_log, y_pred_log):
    tr = np.exp(y_true_log); pr = np.exp(y_pred_log)
    return np.sqrt(np.mean((np.log1p(pr) - np.log1p(tr))**2))

def entrainer(X_train, y_train, X_test, y_test):
    Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1000,
                          learning_rate=0.04, depth=8, random_seed=42,
                          early_stopping_rounds=50, verbose=False)
    m.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
    return rmsle(y_test.values, m.predict(X_test))

# ===================================================================
# TEST A : split aleatoire 70/30
# ===================================================================
def test_A(df_bien, cols_temps):
    d = filtrer(df_bien)
    if len(d) < 300: return None
    cols = FEATURES_FIXES + cols_temps if cols_temps else FEATURES_FIXES
    X, y = d[cols], d['log_prix_total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return entrainer(X_train, y_train, X_test, y_test)

# ===================================================================
# TEST B : generalisation (train < annee_holdout, test = annee_holdout)
# ===================================================================
def test_B(df_bien, cols_temps):
    d = filtrer(df_bien)
    train = d[d['annee_vente'] < annee_holdout]
    test = d[d['annee_vente'] == annee_holdout]
    if len(train) < 300 or len(test) < 50: return None
    cols = FEATURES_FIXES + cols_temps if cols_temps else FEATURES_FIXES
    return entrainer(train[cols], train['log_prix_total'], test[cols], test['log_prix_total'])

# ======= EXECUTION =======
df_m = df[df['type_local']=='Maison']
df_a = df[df['type_local']=='Appartement']
t0 = time.time()

for nom_test, fonction in [("A - SPLIT ALEATOIRE 70/30", test_A),
                            (f"B - GENERALISATION (test = {annee_holdout}, non vu)", test_B)]:
    print("\n" + "="*66)
    print(f"TEST {nom_test}")
    print("="*66)
    print(f"{'Configuration temps':<16} {'RMSLE maison':>14} {'RMSLE appart':>14}")
    print("-"*66)
    ref = {}
    for nom_cfg, cols in CONFIGS.items():
        rm = fonction(df_m, cols)
        ra = fonction(df_a, cols)
        rm_s = f"{rm:.4f}" if rm else "n/a"
        ra_s = f"{ra:.4f}" if ra else "n/a"
        marque = ""
        if nom_cfg == 'DATE':
            ref = {'m': rm, 'a': ra}
        elif nom_cfg == 'ECO' and rm and ra and ref.get('m') and ref.get('a'):
            delta = ((rm - ref['m']) + (ra - ref['a'])) / 2
            marque = f"  (vs DATE : {delta:+.4f})"
        print(f"{nom_cfg:<16} {rm_s:>14} {ra_s:>14}{marque}")

print("\n" + "="*66)
print("LECTURE :")
print("  TEST A : si DATE et ECO donnent le meme RMSLE -> redondants (attendu,")
print("           corr. 0.86 entre taux et annee). Confirme tes mesures.")
print("  TEST B : si ECO < DATE sur l'annee non vue -> les variables eco")
print("           GENERALISENT mieux (relation causale transferable). C'est")
print("           l'avantage theorique dont parle ton tuteur.")
print("           Si DATE reste meilleur ou egal -> la date suffit.")
print(f"\nTemps total : {time.time()-t0:.0f}s")