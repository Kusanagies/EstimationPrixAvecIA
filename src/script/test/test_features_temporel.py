"""
COMPARAISON DE FEATURES TEMPORELLES CANDIDATES
===============================================
Cherche une feature temporelle qui ferait mieux que 'annee + mois' brut.
Toutes testees sur la meme baseline : lat/lon + surface (+ la feature testee).

Candidates :
  0. SANS TEMPS               : reference basse (aucune info temporelle)
  1. ANNEE + MOIS             : version actuelle (reference a battre)
  2. SAISONNALITE             : annee + trimestre + mois cyclique (sin/cos)
  3. VOLUME MARCHE            : annee + nb de ventes du trimestre (liquidite periode)
  4. DYNAMIQUE LOCALE RECENTE : annee + prix median commune sur 12 mois glissants
                                ANTERIEURS (anti-leakage strict), + sa tendance
  5. DYNAMIQUE + SAISONNALITE : combinaison des plus prometteuses

Anti-leakage : les features "dynamique locale" n'utilisent QUE des ventes
strictement anterieures a la date du bien courant (pas de fuite du futur).

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
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur.connect().close()
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

departement = input("Departement a tester (ex: 34) : ").strip()

print("\nExtraction...")
df = pd.read_sql(f"""
    SELECT communes_code AS code_commune, lat AS latitude, lng AS longitude,
           prix_m2, surface AS surface_reelle_bati, typebien AS type_local,
           YEAR(date) AS annee_vente, MONTH(date) AS mois_vente,
           date AS date_vente
    FROM synthese
    WHERE departements_code = '{departement}'
      AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
      AND typebien IN ('maison', 'appartement');
""", con=moteur)
for c in ['prix_m2','surface_reelle_bati','latitude','longitude','annee_vente','mois_vente']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['date_vente'] = pd.to_datetime(df['date_vente'], errors='coerce')
df = df.dropna(subset=['prix_m2','surface_reelle_bati','latitude','longitude','annee_vente','mois_vente','date_vente'])
df['type_local'] = df['type_local'].str.capitalize()
df['prix_total'] = df['prix_m2'] * df['surface_reelle_bati']
df['log_prix_total'] = np.log(df['prix_total'])

# Index temporel continu (en mois) pour les calculs glissants
df['mois_absolu'] = (df['annee_vente'] - df['annee_vente'].min()) * 12 + df['mois_vente']

# Features temporelles simples (pre-calculables sans risque de fuite)
df['trimestre'] = (df['mois_vente'] - 1) // 3 + 1
df['mois_sin'] = np.sin(2*np.pi*df['mois_vente']/12)
df['mois_cos'] = np.cos(2*np.pi*df['mois_vente']/12)

# Volume de marche : nb de ventes par (annee, trimestre) -- indicateur de liquidite
vol = df.groupby(['annee_vente','trimestre']).size().rename('volume_trimestre').reset_index()
df = pd.merge(df, vol, on=['annee_vente','trimestre'], how='left')

print("Calcul de la dynamique locale recente (anti-leakage)...")
# Pour chaque bien : prix median de SA commune sur les 12 mois ANTERIEURS.
# Anti-leakage : on ne regarde que les ventes strictement avant mois_absolu courant.
def dynamique_locale(sous_df):
    sous_df = sous_df.sort_values('mois_absolu').copy()
    prix_recent = np.full(len(sous_df), np.nan)
    tendance = np.full(len(sous_df), 0.0)
    # groupe par commune pour limiter la fenetre
    for com, grp in sous_df.groupby('code_commune'):
        idx = grp.index.to_numpy()
        mois = grp['mois_absolu'].to_numpy()
        prix = grp['prix_m2'].to_numpy()
        for i in range(len(grp)):
            m0 = mois[i]
            fenetre = (mois >= m0 - 12) & (mois < m0)  # 12 mois anterieurs, exclut le present
            if fenetre.sum() >= 5:
                prix_recent[np.where(sous_df.index == idx[i])[0][0]] = np.median(prix[fenetre])
    return prix_recent

# On calcule par type (les biens sont compares a leur propre type)
df['prix_local_recent'] = np.nan
for tl in ['Maison','Appartement']:
    m = df['type_local']==tl
    df.loc[m, 'prix_local_recent'] = dynamique_locale(df[m])
# Repli : mediane par commune puis globale quand pas d'historique
med_com = df.groupby('code_commune')['prix_m2'].transform('median')
df['prix_local_recent'] = df['prix_local_recent'].fillna(med_com).fillna(df['prix_m2'].median())

CONFIGS = {
    '0. SANS TEMPS':        [],
    '1. annee + mois':      ['annee_vente','mois_vente'],
    '2. saisonnalite':      ['annee_vente','trimestre','mois_sin','mois_cos'],
    '3. volume marche':     ['annee_vente','volume_trimestre'],
    '4. dynamique locale':  ['annee_vente','prix_local_recent'],
    '5. dynamique+saison':  ['annee_vente','prix_local_recent','trimestre','mois_sin','mois_cos'],
}
FIXES = ['latitude','longitude','surface_reelle_bati']

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
print("\n" + "="*66)
print(f"FEATURES TEMPORELLES - Departement {departement}")
print(f"Baseline fixe : {FIXES}")
print("="*66)
print(f"{'Feature temporelle':<22} {'RMSLE maison':>14} {'RMSLE appart':>14}")
print("-"*66)
t0=time.time()
df_m = df[df['type_local']=='Maison']; df_a = df[df['type_local']=='Appartement']
res=[]; ref_m=ref_a=None
for nom,cols in CONFIGS.items():
    rm=evaluer(df_m,cols); ra=evaluer(df_a,cols)
    if nom.startswith('1.'): ref_m,ref_a = rm,ra
    res.append((nom,rm,ra))
    marque=""
    if ref_m and rm and ra and not nom.startswith('1.') and not nom.startswith('0.'):
        delta=((rm-ref_m)+(ra-ref_a))/2
        if delta < -0.001: marque=f"  <-- mieux ({delta:+.4f})"
        elif delta > 0.001: marque=f"  (pire {delta:+.4f})"
        else: marque="  (equivalent)"
    print(f"{nom:<22} {rm:>14.4f} {ra:>14.4f}{marque}")

print("\n" + "="*66)
print("Reference a battre = '1. annee + mois'.")
print("Une feature 'mieux' apporte une info temporelle NON captee par l'annee.")
print(f"Temps total : {time.time()-t0:.0f}s")