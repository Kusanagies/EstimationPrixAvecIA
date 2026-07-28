"""
EVALUATION CATBOOST - SOURCE etalab_dvf.synthese (VERSION MODULAIRE)
====================================================================
Orchestrateur leger. La logique est repartie dans les modules :
  - eia_chargement.py     : connexions, extraction, enrichissements, distances, features de base
  - eia_features.py       : filtre coherence marche + features spatiales (voisins/section/densite)
  - eia_cross_validation.py : cross-validation rigoureuse sur le train
  - eia_metriques.py      : calcul et affichage des metriques
  - analyse_normalite.py  : analyse de normalite des erreurs log

Cible : prix total (log). Split parametrable. CV toujours active.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from dotenv import load_dotenv
import geopandas as gpd

# Modules du projet
from eia_chargement import (connexions, extraire_ventes, charger_enrichissements,
                            fusion_et_distances, nettoyer_et_feature_base, RAYON)
from eia_features import filtre_coherence_marche, construire_features_spatiales
from eia_cross_validation import cross_validation_train, afficher_resultats_cv
from eia_metriques import calculer_metriques, afficher_rapport

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

print("-" * 50)
print("EVALUATION CATBOOST - SOURCE etalab_dvf.synthese (MODULAIRE)")
print("-" * 50)
print("\n" + "=" * 50)
print("CHOIX DES FEATURES POUR L'ENTRAINEMENT")
print("=" * 50)

FA = {}
print("\n--- Caracteristiques du bien ---")
FA['geo_base']    = demander("Latitude / longitude ?", True)
FA['surface']     = demander("Surface (+ log, par piece) ?", True)
FA['pieces']      = demander("Nombre de pieces ?", True)
FA['terrain']     = demander("Terrain (surface, log, a_terrain) ?", True)
FA['date']        = demander("Date (annee, mois) ?", True)
print("\n--- Enrichissements communaux ---")
FA['dpe']         = demander("Profil DPE ?", True)
FA['chauffage']   = demander("Profil chauffage ?", True)
FA['revenus']     = demander("Revenus / Gini / minima ?", True)
FA['densite_pop'] = demander("Densite de population ?", True)
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
FA['chomage']          = demander("Taux de chomage departemental ?", False)
FA['taux_credit']      = demander("Taux de credit immobilier ?", False)
FA['taux_inflation']   = demander("Taux d'inflation ?", False)
FA['pib']              = demander("PIB national ?", False)
FA['ipc']              = demander("Indice prix conso (IPC) ?", False)

# Mode de split
print("\n--- Mode de decoupage train/test ---")
print("  1 = Aleatoire 70/30    2 = Temporel (train < derniere annee)")
_choix = input("  Choix (1/2, defaut 1) : ").strip()
MODE_SPLIT = 'temporel' if _choix == '2' else 'aleatoire'
print(f"  -> Split {MODE_SPLIT}")
FAIRE_CV = True
N_FOLDS = 5

# ==========================================
# 1. CONNEXIONS ET ZONE
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"

try:
    db_pass = os.environ["DB_PASS"]
    moteur_dvf, moteur_enr = connexions(db_pass)
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

departement = input("\nDepartement (ex: 34, 75) ou 'FRANCE' : ").strip().upper()
if len(departement) < 2:
    print("Format invalide."); sys.exit()

if departement == 'FRANCE':
    filtre_dvf = "1=1"; filtre_dpe = "1=1"; dep_infra = "FRANCE"; nom_zone = "France"
else:
    filtre_dvf = f"departements_code = '{departement}'"
    filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
    dep_infra = departement; nom_zone = f"Departement {departement}"

dossier_graphes = RACINE_PROJET / "out" / (departement if departement != 'FRANCE' else 'FRANCE') / "synthese"
dossier_graphes.mkdir(parents=True, exist_ok=True)

print(f"\nZone : {nom_zone}")
temps_debut = time.time()

# ==========================================
# 2. CHARGEMENT (via modules)
# ==========================================
gdf_littoral = gpd.read_file(CHEMIN_GPKG)
if gdf_littoral.crs is not None and gdf_littoral.crs.to_epsg() != 4326:
    gdf_littoral = gdf_littoral.to_crs(epsg=4326)

print("Etape 1/4 : Extraction des ventes...")
maisons = extraire_ventes(moteur_dvf, filtre_dvf)
if len(maisons) == 0:
    print("Aucune donnee."); sys.exit()
print(f"  {len(maisons):,} ventes.")

print("Etape 2/4 : Enrichissements...")
enr = charger_enrichissements(moteur_enr, FA, filtre_dpe, dep_infra)

print("Etape 3/4 : Fusion et distances...")
donnees = fusion_et_distances(maisons, enr, FA, gdf_littoral)

print("Etape 4/4 : Nettoyage et features de base...")
dp, features_base = nettoyer_et_feature_base(donnees, FA)
print(f"  {len(features_base)} features de base : {features_base}")

# ==========================================
# 3. ENTRAINEMENT ET EVALUATION PAR TYPE
# ==========================================
datasets = {
    'maisons': dp[dp['type_local'] == 'Maison'].copy(),
    'appartements': dp[dp['type_local'] == 'Appartement'].copy(),
}

for type_bien, df_bien in datasets.items():
    if len(df_bien) < 50:
        print(f"\n--- {type_bien} ignore (pas assez de donnees) ---"); continue

    # Filtres aberrations
    plancher = df_bien['prix_m2'].quantile(0.01)
    plafond = df_bien['prix_m2'].quantile(0.99)
    df_bien = df_bien[(df_bien['prix_m2'] >= plancher) & (df_bien['prix_m2'] <= plafond)].copy()
    df_bien, nb_retires = filtre_coherence_marche(df_bien)
    print(f"\n  Filtre coherence marche : {nb_retires} biens retires")

    print("\n" + "=" * 50)
    print(f"FLUX : {type_bien.upper()} ({len(df_bien)} biens)")
    print("=" * 50)

    X = df_bien[features_base].copy()
    y = df_bien['log_prix_total']

    # Split
    if MODE_SPLIT == 'aleatoire':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    else:
        annee_max = df_bien['annee_vente'].max()
        tr = df_bien['annee_vente'] < annee_max
        te = df_bien['annee_vente'] == annee_max
        if tr.sum() == 0 or te.sum() == 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, y_train = X[tr], y[tr]
            X_test, y_test = X[te], y[te]

    # Features spatiales (via module)
    X_train, X_test, ajoutees = construire_features_spatiales(
        X_train, X_test, df_bien, X_train.index, X_test.index, FA, RAYON)
    features_finales = list(dict.fromkeys(list(features_base) + ajoutees))
    X_train = X_train[features_finales]
    X_test = X_test[features_finales]

    # Validation early stopping
    if MODE_SPLIT == 'aleatoire':
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    else:
        annees_train = df_bien.loc[X_train.index, 'annee_vente']
        mask_val = (annees_train == annees_train.max()).values
        if mask_val.sum() < 50 or (~mask_val).sum() < 200:
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
        else:
            X_tr, y_tr = X_train[~mask_val], y_train[~mask_val]
            X_val, y_val = X_train[mask_val], y_train[mask_val]

    # Entrainement quantile
    modeles_q = {}
    for nom_q, alpha in {'bas': 0.025, 'median': 0.50, 'haut': 0.975}.items():
        m = CatBoostRegressor(loss_function=f'Quantile:alpha={alpha}', iterations=1000,
                              learning_rate=0.04, depth=8, random_seed=42,
                              early_stopping_rounds=50, verbose=False)
        m.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        modeles_q[nom_q] = m

    # Predictions (prix total)
    total_reel = np.exp(y_test).values
    total_pred = np.exp(modeles_q['median'].predict(X_test))
    total_bas = np.exp(modeles_q['bas'].predict(X_test))
    total_haut = np.exp(modeles_q['haut'].predict(X_test))
    total_bas, total_haut = np.minimum(total_bas, total_haut), np.maximum(total_bas, total_haut)

    # Metriques (via module)
    metriques = calculer_metriques(total_reel, total_pred, total_bas, total_haut)

    # Analyse de normalite (via module)
    try:
        from analyse_normalite import analyser_normalite_log
        analyser_normalite_log(total_reel, total_pred, dossier_graphes, nom_zone, type_bien)
    except Exception as e:
        print(f"  (analyse de normalite non disponible : {e})")

    afficher_rapport(metriques, len(X_train), len(X_test), type_bien)

    # Cross-validation sur le train (via module)
    print(f"\n  Cross-validation {N_FOLDS}-fold sur le train...")
    rmsle_cv = cross_validation_train(df_bien, features_base, FA, RAYON, X_train.index, n_folds=N_FOLDS)
    afficher_resultats_cv(rmsle_cv, metriques['rmsle'])

print(f"\nTemps total : {time.time()-temps_debut:.2f}s")
print(f"Graphes : {dossier_graphes}")