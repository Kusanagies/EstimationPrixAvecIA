"""
TEST D'IMPACT DES FILTRES SUR synthese
=======================================
Compare 3 configurations de filtrage sur la table synthese, pour decider
quels filtres appliquer (l'equivalent de ton ancien 'nombre_lots <= 3').

Comme synthese n'a pas de colonne 'nombre_lots', on teste le proxy disponible :
'nb_dependances' (nombre de dependances rattachees au bien).

Configs comparees :
  A. BRUT       : aucun filtre lots/dependances (tout synthese)
  B. DEP <= 3   : nb_dependances <= 3 (esprit de ton ancien filtre)
  C. DEP = 0    : uniquement les biens sans dependance (le plus strict)

Pour chaque config : entraine un CatBoost quantile (split temporel) et compare
MAE, R2, PE20, couverture. Meme methodo que correlation_cat, allegee.
"""

import time, sys, os
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv

RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur.connect().close()
except Exception as e:
    print(f"Erreur de connexion : {e}"); sys.exit()

departement = input("Departement a tester (ex: 34) : ").strip()
type_bien = input("Type de bien (maison / appartement) : ").strip().lower()

# --- Extraction depuis synthese (une seule fois, tout le departement) ---
print("\nExtraction depuis synthese...")
df = pd.read_sql(f"""
    SELECT id, date, valeur_fonciere, typebien,
           communes_code, parcelles_code, lat, lng,
           prix_m2, surface_terrain, surface, nb_pieces, nb_dependances,
           YEAR(date) AS annee_vente
    FROM synthese
    WHERE departements_code = '{departement}'
      AND typebien = '{type_bien}'
      AND surface > 9 AND surface <= 300
      AND prix_m2 > 0 AND nb_pieces > 0
""", con=moteur)
print(f"Biens bruts extraits : {len(df):,}")

if len(df) < 500:
    print("Trop peu de biens pour un test fiable."); sys.exit()

# Feature engineering minimal commun
# Conversion explicite des colonnes numeriques (synthese renvoie parfois du 'object'
# a cause des NULL SQL, ce qui casse np.log1p / np.log)
colonnes_num = ['valeur_fonciere', 'prix_m2', 'surface_terrain', 'surface',
                'nb_pieces', 'nb_dependances', 'lat', 'lng']
for col in colonnes_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Feature engineering (maintenant que les types sont propres)
df['code_departement'] = df['communes_code'].str[:2]
df['code_section'] = df['parcelles_code'].str[:10]
df['surface_terrain'] = df['surface_terrain'].fillna(0)
df['log_prix_m2'] = np.log(df['prix_m2'])
df['log_surface'] = np.log(df['surface'])
df['surface_par_piece'] = df['surface'] / df['nb_pieces']
df['log_terrain'] = np.log1p(df['surface_terrain'])

RAYON = 6371000

def evaluer_config(df_config, nom_config):
    """Entraine et evalue un modele sur une config de filtrage donnee."""
    d = df_config.copy()
    # Filtrage aberrations de prix (quantiles, comme d'habitude)
    plancher = d['prix_m2'].quantile(0.01)
    plafond = d['prix_m2'].quantile(0.99)
    d = d[(d['prix_m2'] >= plancher) & (d['prix_m2'] <= plafond)].copy()

    if len(d) < 500:
        print(f"  {nom_config}: trop peu de biens ({len(d)}), ignore.")
        return None

    # Split temporel
    annee_max = d['annee_vente'].max()
    train = d[d['annee_vente'] < annee_max]
    test = d[d['annee_vente'] == annee_max]
    if len(train) < 200 or len(test) < 50:
        # repli aleatoire
        train, test = train_test_split(d, test_size=0.2, random_state=42)

    # Features de voisinage (sur train uniquement)
    coords_train = np.deg2rad(train[['lat', 'lng']])
    arbre = BallTree(coords_train, metric='haversine')
    prix_train = train['prix_m2'].values
    surf_train = train['surface'].values

    def voisins(dist_rad, idx, sb, self_i=None):
        if self_i is not None:
            keep = idx != self_i
            dist_rad, idx = dist_rad[keep], idx[keep]
        if len(idx) == 0: return np.nan
        dm = dist_rad * RAYON
        pv, sv = prix_train[idx], surf_train[idx]
        m = (sv >= sb*0.6) & (sv <= sb*1.4)
        if m.sum() >= 3: dd, pp = dm[m], pv[m]
        else: dd, pp = dm, pv
        w = 1.0/(dd+50.0)
        return np.sum(w*pp)/np.sum(w)

    k = min(41, len(coords_train))
    dtr, itr = arbre.query(coords_train, k=k)
    sb_tr = train['surface'].values
    train = train.copy()
    train['prix_m2_voisins'] = [voisins(dtr[i], itr[i], sb_tr[i], self_i=i) for i in range(len(itr))]

    coords_test = np.deg2rad(test[['lat', 'lng']])
    dte, ite = arbre.query(coords_test, k=min(40, len(coords_train)))
    sb_te = test['surface'].values
    test = test.copy()
    test['prix_m2_voisins'] = [voisins(dte[i], ite[i], sb_te[i]) for i in range(len(ite))]

    # Prix section (median train)
    med_sec = train.groupby('code_section')['prix_m2'].median()
    med_com = train.groupby('communes_code')['prix_m2'].median()
    med_glob = train['prix_m2'].median()
    for sous in (train, test):
        sous['prix_m2_section'] = (sous['code_section'].map(med_sec)
                                   .fillna(sous['communes_code'].map(med_com))
                                   .fillna(med_glob))

    feats = ['lat', 'lng', 'surface', 'log_surface', 'surface_par_piece',
             'nb_pieces', 'surface_terrain', 'log_terrain', 'annee_vente',
             'prix_m2_voisins', 'prix_m2_section']

    X_train, y_train = train[feats], train['log_prix_m2']
    X_test, y_test = test[feats], test['log_prix_m2']

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    modeles = {}
    for nom_q, alpha in {'bas':0.025, 'median':0.50, 'haut':0.975}.items():
        m = CatBoostRegressor(loss_function=f'Quantile:alpha={alpha}',
                              iterations=1000, learning_rate=0.04, depth=8,
                              random_seed=42, early_stopping_rounds=50, verbose=False)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        modeles[nom_q] = m

    reels = np.exp(y_test)
    pred = np.exp(modeles['median'].predict(X_test))
    bas = np.exp(modeles['bas'].predict(X_test))
    haut = np.exp(modeles['haut'].predict(X_test))
    bas, haut = np.minimum(bas, haut), np.maximum(bas, haut)

    mae = mean_absolute_error(reels, pred)
    r2 = r2_score(reels, pred)
    err_rel = np.abs(reels.values - pred) / reels
    pe20 = np.mean(err_rel <= 0.20) * 100
    couv = np.mean((reels.values >= bas) & (reels.values <= haut)) * 100

    print(f"  {nom_config:18s} | n={len(d):>6} | MAE {mae:>6.0f} | R2 {r2*100:>5.1f}% | "
          f"PE20 {pe20:>5.1f}% | couv {couv:>5.1f}%")
    return {'config': nom_config, 'n': len(d), 'mae': mae, 'r2': r2, 'pe20': pe20, 'couv': couv}

print("\n" + "=" * 80)
print(f"COMPARAISON DES FILTRES - {type_bien.upper()} (dep {departement})")
print("=" * 80)
print(f"  {'Config':18s} | {'n':>6} | {'MAE':>6} | {'R2':>6} | {'PE20':>6} | {'Couv':>6}")
print("-" * 80)

resultats = []
# Config A : brut (tout)
resultats.append(evaluer_config(df, "A. BRUT (tout)"))
# Config B : nb_dependances <= 3
resultats.append(evaluer_config(df[df['nb_dependances'] <= 3], "B. dependances<=3"))
# Config C : nb_dependances == 0
resultats.append(evaluer_config(df[df['nb_dependances'] == 0], "C. dependances=0"))

print("=" * 80)
print("\nINTERPRETATION :")
print("  - Si les 3 configs donnent un MAE proche -> le filtre lots/dependances")
print("    n'a plus d'effet (synthese a deja demultiplexe les ventes complexes).")
print("  - Si BRUT a un MAE nettement pire -> garder un filtre sur nb_dependances.")
print("  - Compare aussi le volume n : filtrer trop reduit les donnees.")