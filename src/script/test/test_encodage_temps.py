"""
COMPARAISON DES ENCODAGES TEMPORELS (sur baseline)
===================================================
Compare 5 facons d'encoder le TEMPS, a features egales par ailleurs.
Baseline commune = lat/lon + surface_reelle_bati (+ l'encodage temporel teste).
Seul l'encodage du temps change -> toute difference de RMSLE vient de lui.

Variantes testees :
  1. ANNEE + MOIS      : 'annee_vente' + 'mois_vente' (version actuelle, reference)
  2. DATE DECIMALE     : annee + (mois-1)/12  (une seule variable continue)
  3. MOIS CYCLIQUE     : annee + sin(2pi*mois/12) + cos(2pi*mois/12)  (saisonnalite)
  4. MOIS ECOULES      : (annee-2014)*12 + mois  (rampe temporelle lineaire)
  5. INDICE DE MARCHE  : annee remplacee par la mediane des prix/m2 de l'annee
                         (calculee SUR LE TRAIN uniquement, anti-leakage)

Saisie unique (departement). Affiche uniquement le RMSLE (maisons + apparts).
Source : synthese | Cible : prix total (log) | Split : aleatoire 70/30
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
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur.connect().close()
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

departement = input("Departement a tester (ex: 34) : ").strip()

print("\nExtraction...")
df = pd.read_sql(f"""
    SELECT communes_code AS code_commune, lat AS latitude, lng AS longitude,
           prix_m2, surface AS surface_reelle_bati, typebien AS type_local,
           nb_pieces AS nombre_pieces_principales,
           YEAR(date) AS annee_vente, MONTH(date) AS mois_vente
    FROM synthese
    WHERE departements_code = '{departement}'
      AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
      AND typebien IN ('maison', 'appartement');
""", con=moteur)
for c in ['prix_m2','surface_reelle_bati','nombre_pieces_principales','latitude','longitude','annee_vente','mois_vente']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['prix_m2','surface_reelle_bati','latitude','longitude','annee_vente','mois_vente'])
df['type_local'] = df['type_local'].str.capitalize()

# Feature engineering commun
df['prix_total'] = df['prix_m2'] * df['surface_reelle_bati']
df['log_prix_total'] = np.log(df['prix_total'])

# Pre-calcul des variantes temporelles (celles qui ne dependent pas du train)
df['date_decimale'] = df['annee_vente'] + (df['mois_vente'] - 1) / 12.0
df['mois_sin'] = np.sin(2 * np.pi * df['mois_vente'] / 12.0)
df['mois_cos'] = np.cos(2 * np.pi * df['mois_vente'] / 12.0)
annee_min = df['annee_vente'].min()
df['mois_ecoules'] = (df['annee_vente'] - annee_min) * 12 + df['mois_vente']

# Les variantes : nom -> liste des colonnes temporelles a utiliser
# ('indice_marche' est traite a part car calcule sur le train)
VARIANTES = {
    '1. annee + mois':     ['annee_vente', 'mois_vente'],
    '2. date decimale':    ['date_decimale'],
    '3. mois cyclique':    ['annee_vente', 'mois_sin', 'mois_cos'],
    '4. mois ecoules':     ['mois_ecoules'],
    '5. indice de marche': ['__indice__'],  # marqueur special
}

FEATURES_FIXES = ['latitude', 'longitude', 'surface_reelle_bati']

def evaluer_variante(df_bien, cols_temps):
    """Entraine baseline + encodage temporel donne, renvoie le RMSLE (prix total)."""
    # Filtres (quantiles + coherence marche)
    pl, pf = df_bien['prix_m2'].quantile(0.01), df_bien['prix_m2'].quantile(0.99)
    d = df_bien[(df_bien['prix_m2']>=pl)&(df_bien['prix_m2']<=pf)].copy()
    sc = d.groupby('code_commune')['prix_m2'].agg(['median','size'])
    ref = d['code_commune'].map(sc['median'])
    nc = d['code_commune'].map(sc['size'])
    ref = ref.where(nc>=10, d['prix_m2'].median())
    d = d[(d['prix_m2']/ref).between(0.40,2.50)].copy()
    if len(d) < 300: return None

    y = d['log_prix_total']
    idx_train, idx_test = train_test_split(d.index, test_size=0.30, random_state=42)

    # Cas special : indice de marche (calcule sur le TRAIN uniquement)
    if cols_temps == ['__indice__']:
        indice = d.loc[idx_train].groupby('annee_vente')['prix_m2'].median()
        indice_global = d.loc[idx_train, 'prix_m2'].median()
        d = d.copy()
        d['indice_marche'] = d['annee_vente'].map(indice).fillna(indice_global)
        cols = FEATURES_FIXES + ['indice_marche']
    else:
        cols = FEATURES_FIXES + cols_temps

    X = d[cols]
    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1000,
                          learning_rate=0.04, depth=8, random_seed=42,
                          early_stopping_rounds=50, verbose=False)
    m.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
    total_reel = np.exp(y_test).values
    total_pred = np.exp(m.predict(X_test))
    return np.sqrt(np.mean((np.log1p(total_pred) - np.log1p(total_reel))**2))

# ======= EXECUTION =======
print("\n" + "="*68)
print(f"COMPARAISON DES ENCODAGES TEMPORELS - Departement {departement}")
print(f"Features fixes : {FEATURES_FIXES} + [encodage du temps teste]")
print("="*68)
print(f"{'Encodage du temps':<24} {'RMSLE maison':>13} {'RMSLE appart':>13}")
print("-"*68)

t0 = time.time()
resultats = []
df_m = df[df['type_local']=='Maison']
df_a = df[df['type_local']=='Appartement']
for nom, cols in VARIANTES.items():
    rm = evaluer_variante(df_m, cols)
    ra = evaluer_variante(df_a, cols)
    resultats.append((nom, rm, ra))
    rm_s = f"{rm:.4f}" if rm else "n/a"
    ra_s = f"{ra:.4f}" if ra else "n/a"
    print(f"{nom:<24} {rm_s:>13} {ra_s:>13}")

# ======= CLASSEMENT =======
print("\n" + "="*68)
print("CLASSEMENT (RMSLE le plus bas = meilleur encodage du temps)")
print("="*68)
# Reference = variante 1 (annee + mois)
ref_m = resultats[0][1]
ref_a = resultats[0][2]
print(f"{'Encodage':<24} {'maison':>10} {'appart':>10} {'vs annee+mois':>16}")
print("-"*68)
for nom, rm, ra in sorted(resultats, key=lambda r: (r[1] or 9) + (r[2] or 9)):
    # Ecart vs reference (negatif = meilleur)
    if rm and ra and ref_m and ref_a:
        delta = ((rm - ref_m) + (ra - ref_a)) / 2
        delta_s = f"{delta:+.4f}"
        marque = " <-- meilleur" if delta < -0.0005 else (" (reference)" if abs(delta) < 0.0005 else "")
    else:
        delta_s, marque = "n/a", ""
    print(f"{nom:<24} {rm:>10.4f} {ra:>10.4f} {delta_s:>16}{marque}")

print("\n" + "="*68)
print("Un ecart negatif vs annee+mois = cet encodage ameliore le modele.")
print("Un ecart ~0 = equivalent (les arbres captent deja bien le temps brut).")
print(f"Temps total : {time.time()-t0:.0f}s")