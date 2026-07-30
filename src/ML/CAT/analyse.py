"""
ANALYSE DES RESIDUS - OU LE MODELE SE TROMPE-T-IL LE PLUS ?
============================================================
Entraine le modele median (config complete) sur un departement, calcule les
erreurs sur le test 30%, puis decoupe ces erreurs par segments pour reperer
les poches d'erreur. Ne modifie RIEN au pipeline : c'est un outil de diagnostic.

Deux lectures des erreurs :
  - ABSOLUE (|reel - pred|)  -> ou le modele est IMPRECIS
  - SIGNEE  (reel - pred)    -> ou le modele est BIAISE (sur/sous-estimation)

Axes de decoupage : gamme de prix, taille de commune, anciennete de la vente,
surface, fiabilite de la section (nb de ventes dans la section).

Reutilise les modules eia_*. Config complete par defaut (features validees).
"""

import os, sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from dotenv import load_dotenv
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

from eia_chargement import (connexions, extraire_ventes, charger_enrichissements,
                            fusion_et_distances, nettoyer_et_feature_base, RAYON)
from eia_features import filtre_coherence_marche, construire_features_spatiales

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

# ---- Config features : complete (features validees), eco/ipc/socio desactives ----
FA = {k: True for k in ['geo_base','surface','pieces','terrain','date','dpe','chauffage',
                        'revenus','densite_pop','dist_transport','dist_monument','dist_hopital',
                        'dist_universite','dist_littoral','voisins','densite','section','potentiel_urbain']}
FA.update({k: False for k in ['chomage','taux_credit','taux_inflation','pib','ipc','stat_socio_eco']})

RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"

try:
    db_pass = os.environ["DB_PASS"]
    moteur_dvf, moteur_enr = connexions(db_pass)
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

departement = input("Departement (ex: 34) : ").strip()
filtre_dvf = f"departements_code = '{departement}'"
filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"

print("\nChargement des donnees...")
gdf = gpd.read_file(CHEMIN_GPKG)
if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)
maisons = extraire_ventes(moteur_dvf, filtre_dvf)
enr = charger_enrichissements(moteur_enr, FA, filtre_dpe, departement)
donnees = fusion_et_distances(maisons, enr, FA, gdf)
dp, features_base = nettoyer_et_feature_base(donnees, FA)


def analyser_type(type_local):
    """Entraine le modele median pour un type de bien et renvoie un DataFrame de residus."""
    df_bien = dp[dp['type_local'] == type_local].copy()
    if len(df_bien) < 300:
        print(f"  {type_local} : pas assez de donnees."); return None

    # Memes filtres que le pipeline
    pl, pf = df_bien['prix_m2'].quantile(0.01), df_bien['prix_m2'].quantile(0.99)
    df_bien = df_bien[(df_bien['prix_m2']>=pl)&(df_bien['prix_m2']<=pf)].copy()
    df_bien, _ = filtre_coherence_marche(df_bien)

    # Nombre de ventes par section (pour l'axe fiabilite section)
    nb_sec = df_bien.groupby('code_section').size()
    df_bien['nb_ventes_section_diag'] = df_bien['code_section'].map(nb_sec)

    X = df_bien[features_base].copy()
    y = df_bien['log_prix_total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    X_train, X_test, ajoutees = construire_features_spatiales(
        X_train, X_test, df_bien, X_train.index, X_test.index, FA, RAYON)
    feats = list(dict.fromkeys(list(features_base) + ajoutees))
    X_train, X_test = X_train[feats], X_test[feats]

    m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1000,
                          learning_rate=0.04, depth=8, l2_leaf_reg=3.0,
                          random_seed=42, early_stopping_rounds=50, verbose=False)
    Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    m.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)

    # Erreurs sur le test, en EUROS (prix total)
    reel = np.exp(y_test).values
    pred = np.exp(m.predict(X_test))
    res = pd.DataFrame(index=X_test.index)
    res['reel'] = reel
    res['pred'] = pred
    res['err_abs'] = np.abs(reel - pred)                 # imprecision
    res['err_signee'] = reel - pred                      # biais (>0 = sous-estime, <0 = surestime)
    res['ape'] = np.abs(reel - pred) / reel * 100        # erreur abs en %
    res['pe_signee'] = (reel - pred) / reel * 100        # erreur signee en %
    res['rmsle_indiv'] = np.abs(np.log1p(pred) - np.log1p(reel))

    # Colonnes de segmentation (alignees par index)
    res['prix_total']   = df_bien.loc[res.index, 'prix_total'].values
    res['surface']      = df_bien.loc[res.index, 'surface_reelle_bati'].values
    res['annee_vente']  = df_bien.loc[res.index, 'annee_vente'].values
    res['code_commune'] = df_bien.loc[res.index, 'code_commune'].values
    res['nb_sec']       = df_bien.loc[res.index, 'nb_ventes_section_diag'].values
    # Taille de commune (nb de ventes de la commune dans tout le df_bien)
    taille_com = df_bien.groupby('code_commune').size()
    res['taille_commune'] = res['code_commune'].map(taille_com).values

    rmsle_global = np.sqrt(np.mean(res['rmsle_indiv']**2))
    print(f"\n  {type_local} : {len(X_train)} train / {len(X_test)} test | "
          f"RMSLE global {rmsle_global:.4f} | MAPE {res['ape'].mean():.1f}%")
    return res


def tableau_par_segment(res, col_segment, libelle, bins=None, labels=None, est_categorie=False):
    """Affiche l'erreur moyenne par tranche d'un segment."""
    r = res.copy()
    if est_categorie:
        r['seg'] = r[col_segment]
    else:
        r['seg'] = pd.cut(r[col_segment], bins=bins, labels=labels, include_lowest=True)
    agg = r.groupby('seg', observed=True).agg(
        n=('ape', 'size'),
        mape=('ape', 'mean'),
        ape_med=('ape', 'median'),
        biais_pct=('pe_signee', 'mean'),
        err_med_eur=('err_abs', 'median'),
        part_err=('err_abs', 'sum'),
    )
    agg['part_err_%'] = agg['part_err'] / res['err_abs'].sum() * 100
    agg = agg.drop(columns='part_err')
    print(f"\n--- {libelle} ---")
    print(agg.to_string(float_format=lambda x: f"{x:.1f}"))
    return agg


def diagnostiquer(res, type_local):
    print("\n" + "="*72)
    print(f"ANALYSE DES RESIDUS - {type_local.upper()} (departement {departement})")
    print("Colonnes : n=nb biens | mape=err moy % | ape_med=err med % |")
    print("           biais_pct>0=SOUS-estime <0=SUR-estime | err_med_eur | part_err_%=part de l'erreur totale")
    print("="*72)

    # 1. GAMME DE PRIX (deciles)
    q = res['prix_total'].quantile(np.linspace(0, 1, 11)).values.copy()
    q[0], q[-1] = q[0]-1, q[-1]+1
    labels_p = [f"D{i+1}" for i in range(10)]
    tableau_par_segment(res, 'prix_total', "PAR GAMME DE PRIX (deciles, D1=moins cher)",
                        bins=q, labels=labels_p)

    # 2. TAILLE DE COMMUNE
    tableau_par_segment(res, 'taille_commune', "PAR TAILLE DE COMMUNE (nb ventes)",
                        bins=[0,50,200,500,2000,1e9],
                        labels=['<50','50-200','200-500','500-2000','>2000'])

    # 3. ANCIENNETE DE LA VENTE
    tableau_par_segment(res, 'annee_vente', "PAR ANNEE DE VENTE", est_categorie=True)

    # 4. SURFACE
    tableau_par_segment(res, 'surface', "PAR SURFACE (m2)",
                        bins=[0,30,50,80,120,300],
                        labels=['<30','30-50','50-80','80-120','>120'])

    # 5. FIABILITE DE LA SECTION
    tableau_par_segment(res, 'nb_sec', "PAR FIABILITE DE LA SECTION (nb ventes dans la section)",
                        bins=[0,3,10,30,1e9],
                        labels=['1-3 (repli)','4-10','11-30','>30'])


t0 = time.time()
for type_local in ['Maison', 'Appartement']:
    res = analyser_type(type_local)
    if res is not None:
        diagnostiquer(res, type_local)

print("\n" + "="*72)
print("LECTURE")
print("="*72)
print("- part_err_% eleve sur un segment = c'est la que se concentre l'erreur totale.")
print("- biais_pct fortement POSITIF = le modele SOUS-estime (predit trop bas) ce segment.")
print("- biais_pct fortement NEGATIF = le modele SUR-estime (predit trop haut) ce segment.")
print("- Bas de gamme (D1-D2) avec biais NEGATIF fort = regression vers la moyenne")
print("  (le modele tire les biens pas chers vers le haut).")
print("- Section '1-3 (repli)' avec MAPE eleve = la feature section s'appuie sur un repli grossier.")
print(f"\nTemps total : {time.time()-t0:.0f}s")