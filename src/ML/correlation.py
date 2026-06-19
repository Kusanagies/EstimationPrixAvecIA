import time
import sys
import shap
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from sqlalchemy import create_engine
import os
import matplotlib.pyplot as plt 
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 0. CONNEXION INITIALE ET MENU INTERACTIF
# ==========================================
print("-" * 50)
print("INITIALISATION DU MOTEUR D'ESTIMATION IMMOBILIERE (EVALUATION)")
print("-" * 50)

RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    connexion_test = moteur.connect()
    connexion_test.close()
except KeyError:
    print("Erreur : variable DB_PASS introuvable. Verifiez votre fichier .env à la racine")
    sys.exit()
except Exception as e:
    print("Erreur de connexion a la base MySQL. Verifiez que le serveur est allume.")
    sys.exit()

departement = input("\nVeuillez saisir le numero du departement (ex: 34, 75) ou 'FRANCE' : ").strip().upper()

if len(departement) < 2:
    print("Erreur : Le format de saisie est invalide.")
    sys.exit()

print(f"Recherche des secteurs disponibles pour : {departement}...")

if departement == 'FRANCE':
    condition_dep = "1=1"
else:
    condition_dep = f"LEFT(code_commune, 2) = '{departement}'"

query_communes = f"""
    SELECT code_commune, MAX(nom_commune) as nom_commune, COUNT(*) as volume_ventes
    FROM valeurs_foncieres
    WHERE {condition_dep}
      AND type_local IN ('Maison', 'Appartement')
    GROUP BY code_commune
    ORDER BY volume_ventes DESC
    LIMIT 15;
"""
df_communes = pd.read_sql(query_communes, con=moteur)

if len(df_communes) == 0:
    print(f"Erreur : Aucune donnee trouvee pour le secteur {departement}.")
    sys.exit()

print(f"\nVoici les secteurs avec le plus de donnees pour {departement} :")
for index, row in df_communes.iterrows():
    nom = str(row['nom_commune']).ljust(25)[:25] 
    print(f"  - {row['code_commune']} : {nom} ({row['volume_ventes']} ventes)")

print("  - ... (et autres communes)")

choix_local = input("\nSaisissez le code INSEE d'un secteur precis (ou tapez 'TOUS' pour le choix initial complet) : ").strip().upper()

if departement == 'FRANCE':
    if choix_local == 'TOUS':
        filtre_dvf = "1=1"
        filtre_dpe = "1=1"
        dep_infra = "FRANCE"
        nom_zone = "France Entiere"
    else:
        filtre_dvf = f"code_commune = '{choix_local}'"
        filtre_dpe = f"code_insee_ban = '{choix_local}'"
        dep_infra = choix_local[:2]
        nom_zone = f"Secteur {choix_local}"
else:
    if choix_local == 'TOUS':
        filtre_dvf = f"LEFT(code_commune, 2) = '{departement}'"
        filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"
        dep_infra = departement
        nom_zone = f"Departement {departement}"
    else:
        filtre_dvf = f"code_commune = '{choix_local}'"
        filtre_dpe = f"code_insee_ban = '{choix_local}'"
        dep_infra = departement
        nom_zone = f"Secteur {choix_local}"

print(f"\nLancement de l'apprentissage pour : {nom_zone}")

DOSSIER_OUT = RACINE_PROJET / "out"

if departement == 'FRANCE':
    dossier_graphes = DOSSIER_OUT / "FRANCE" if choix_local == 'TOUS' else DOSSIER_OUT / choix_local[:2] / choix_local
else:
    dossier_graphes = DOSSIER_OUT / departement if choix_local == 'TOUS' else DOSSIER_OUT / departement / choix_local

dossier_graphes.mkdir(parents=True, exist_ok=True)

print("-" * 50)
temps_total_debut = time.time()

# ==========================================
# 1. TELECHARGEMENT DES DONNEES FILTREES
# ==========================================
print("Etape 1/4 : Extraction des donnees depuis SQL...")

maisons = pd.read_sql(f"""
    SELECT code_commune, id_parcelle, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, 
           surface_reelle_bati, type_local, nombre_pieces_principales, 
           surface_terrain, YEAR(date_mutation) AS annee_vente, MONTH(date_mutation) AS mois_vente
    FROM valeurs_foncieres
    WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
      AND nature_mutation = 'Vente' AND nombre_lots <= 1 AND nombre_pieces_principales > 0
      AND {filtre_dvf} AND type_local IN ('Maison', 'Appartement');
""", con=moteur)

if len(maisons) == 0:
    print(f"Erreur : Aucune donnee trouvée.")
    sys.exit()

dpe = pd.read_sql(f"""
    SELECT code_insee_ban, 
           (SUM(CASE WHEN etiquette_dpe = 'A' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_A,
           (SUM(CASE WHEN etiquette_dpe = 'B' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_B,
           (SUM(CASE WHEN etiquette_dpe = 'C' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_C,
           (SUM(CASE WHEN etiquette_dpe = 'D' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_D,
           (SUM(CASE WHEN etiquette_dpe = 'E' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_E,
           (SUM(CASE WHEN etiquette_dpe = 'F' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_F,
           (SUM(CASE WHEN etiquette_dpe = 'G' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_G,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Électricité%%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_chauffage_elec,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Gaz%%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_chauffage_gaz,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Fioul%%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_chauffage_fioul,
           (SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Réseau de Chaleur%%' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_chauffage_urbain
    FROM dpe_logements_france
    WHERE etiquette_dpe IN ('A','B','C','D','E','F','G') AND {filtre_dpe}
    GROUP BY code_insee_ban;
""", con=moteur)

stations = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur)

if dep_infra == 'FRANCE':
    query_monuments = "SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL;"
    query_hopitaux = "SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL;"
    query_universites = "SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;"
else:
    query_monuments = f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{dep_infra}';"
    query_hopitaux = f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal, 2) = '{dep_infra}';"
    query_universites = f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{dep_infra}';"

monuments = pd.read_sql(query_monuments, con=moteur)
hopitaux = pd.read_sql(query_hopitaux, con=moteur)
universites = pd.read_sql(query_universites, con=moteur)

filtre_rev = "1=1" if dep_infra == 'FRANCE' else (f"code_commune = '{choix_local}'" if choix_local != 'TOUS' and len(choix_local) == 5 else f"LEFT(code_commune,2) = '{dep_infra}'")
revenus = pd.read_sql(f"SELECT code_commune, median_revenu_disponible,indice_gini,pct_minima_sociaux FROM demographie_communes WHERE {filtre_rev};", con=moteur)

for col in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']:
    revenus[col] = pd.to_numeric(revenus[col], errors='coerce')

# ==========================================
# 2. FUSION ET DISTANCES GLOBALES
# ==========================================
print("Etape 2/4 : Fusion et calculs spatiaux partagés...")
donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees, revenus, on='code_commune', how='left')

RAYON_TERRE_METRES = 6371000
points_rad = np.deg2rad(donnees[['latitude', 'longitude']])

def calculer_distance_min(df_points, nom_colonne):
    if len(df_points) > 0:
        infra_rad = np.deg2rad(df_points.iloc[:, 0:2])
        arbre = BallTree(infra_rad, metric='haversine')
        dist_rad, _ = arbre.query(points_rad, k=1)
        donnees[nom_colonne] = dist_rad.flatten() * RAYON_TERRE_METRES
    else:
        donnees[nom_colonne] = 999999

calculer_distance_min(stations, 'dist_transport_m')
calculer_distance_min(monuments, 'dist_monument_m')
calculer_distance_min(hopitaux, 'dist_hopital_m')

if len(universites) > 0:
    univ_rad = np.deg2rad(universites[['latitude', 'longitude']])
    arbre_univ = BallTree(univ_rad, metric='haversine')
    dist_rad, idx_univ = arbre_univ.query(points_rad, k=1)
    donnees['dist_universite_m'] = dist_rad.flatten() * RAYON_TERRE_METRES
    donnees['volume_etudiants_proche'] = universites.iloc[idx_univ.flatten()]['nombre_etudiants'].values
else:
    donnees['dist_universite_m'] = 999999
    donnees['volume_etudiants_proche'] = 0

# ==========================================
# 3. NETTOYAGE GLOBAL ET CORRECTION DVF
# ==========================================
print("Etape 3/4 : Nettoyage global et correction des biais...")
colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
colonnes_chauffage = ['pct_chauffage_elec', 'pct_chauffage_gaz', 'pct_chauffage_fioul', 'pct_chauffage_urbain']
colonnes_revenus = ['median_revenu_disponible','indice_gini','pct_minima_sociaux']

for col in colonnes_dpe + colonnes_chauffage + colonnes_revenus:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)

plancher = max(donnees['prix_m2'].quantile(0.01),800)
plafond = min(donnees['prix_m2'].quantile(0.99),15000)
# Filtrage securise pour eviter les valeurs extremes
donnees_propres = donnees[
    (donnees['prix_m2'] >= plancher) & (donnees['prix_m2'] <= plafond) & 
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

# ---> CORRECTION DU BIAIS DVF POUR LES APPARTEMENTS <---
# Le notaire renseigne souvent la taille de la parcelle de la copropriété entiere
mask_appart = donnees_propres['type_local'] == 'Appartement'
donnees_propres.loc[mask_appart, 'surface_terrain'] = 0

donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])
donnees_propres['log_surface'] = np.log(donnees_propres['surface_reelle_bati'])
donnees_propres['surface_par_piece'] = donnees_propres['surface_reelle_bati'] / donnees_propres['nombre_pieces_principales']
donnees_propres['a_terrain'] = (donnees_propres['surface_terrain']>0).astype(int)
donnees_propres['log_terrain'] = np.log1p(donnees_propres['surface_terrain'])
donnees_propres['code_section'] = donnees_propres['id_parcelle'].str[:10]

colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]
colonnes_standard = ['surface_reelle_bati', 'volume_etudiants_proche', 'log_surface','surface_par_piece',
                     'surface_terrain','log_terrain', 'median_revenu_disponible','indice_gini','pct_minima_sociaux']

features_base = ['latitude', 'longitude', 'nombre_pieces_principales', 'annee_vente','mois_vente','a_terrain'] \
                + colonnes_dpe + colonnes_chauffage + colonnes_standard + colonnes_dist

# ==========================================
# 4. BOUCLE D'EVALUATION (MAISONS / APPARTEMENTS)
# ==========================================
print("Etape 4/4 : Evaluation specifique par type de bien...")

datasets = {
    'maisons': donnees_propres[donnees_propres['type_local'] == 'Maison'].copy(),
    'appartements': donnees_propres[donnees_propres['type_local'] == 'Appartement'].copy()
}

for type_bien, df_bien in datasets.items():
    if len(df_bien) < 50:
        print(f"\n--- IGNORÉ : Pas assez de donnees pour le type {type_bien} ---")
        continue

    print("\n" + "="*50)
    print(f"ANALYSE DU FLUX : {type_bien.upper()} ({len(df_bien)} biens)")
    print("="*50)

    X = df_bien[features_base].copy()
    y = df_bien['log_prix_m2']

    annee_max = df_bien['annee_vente'].max()
    train_mask = df_bien['annee_vente'] < annee_max
    test_mask = df_bien['annee_vente'] == annee_max

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

    coords_train = np.deg2rad(df_bien.loc[X_train.index, ['latitude', 'longitude']])
    prix_train = df_bien.loc[X_train.index, 'prix_m2'].values
    arbre_voisins = BallTree(coords_train, metric='haversine')

    k_train = min(16, len(coords_train))
    _, idx_tr = arbre_voisins.query(coords_train, k=k_train)
    voisins_train = [np.median(prix_train[row[1:]]) if len(row) > 1 else prix_train[row[0]] for row in idx_tr]

    k_test = min(15, len(coords_train))
    coords_test = np.deg2rad(df_bien.loc[X_test.index, ['latitude', 'longitude']])
    _, idx_te = arbre_voisins.query(coords_test, k=k_test)
    voisins_test = [np.median(prix_train[row]) for row in idx_te]

    rayon_rad = 1000 / RAYON_TERRE_METRES
    X_train['densite_ventes_1km'] = arbre_voisins.query_radius(coords_train, r=rayon_rad, count_only=True)
    X_test['densite_ventes_1km'] = arbre_voisins.query_radius(coords_test, r=rayon_rad, count_only=True)
    
    X_train['prix_m2_voisins'] = voisins_train
    X_test['prix_m2_voisins'] = voisins_test

    df_tr = df_bien.loc[X_train.index]
    med_section = df_tr.groupby('code_section')['prix_m2'].median()
    med_commune = df_tr.groupby('code_commune')['prix_m2'].median()
    med_globale = df_tr['prix_m2'].median()

    def prix_section_local(idx): 
        sec = df_bien.loc[idx, 'code_section']
        com = df_bien.loc[idx, 'code_commune']
        if sec in med_section.index: return med_section[sec]
        elif com in med_commune.index: return med_commune[com]
        else: return med_globale

    X_train['prix_m2_section'] = [prix_section_local(i) for i in X_train.index]
    X_test['prix_m2_section'] = [prix_section_local(i) for i in X_test.index]

    nb_ventes_section = df_tr.groupby('code_section').size()
    X_train['nb_ventes_section'] = [nb_ventes_section.get(df_bien.loc[i, 'code_section'], 0) for i in X_train.index]
    X_test['nb_ventes_section'] = [nb_ventes_section.get(df_bien.loc[i, 'code_section'], 0) for i in X_test.index]

    features_finales = features_base + ['densite_ventes_1km', 'prix_m2_voisins', 'prix_m2_section', 'nb_ventes_section']
    X_train = X_train[features_finales]
    X_test = X_test[features_finales]

    modele_xgb = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=1.0,
        early_stopping_rounds=50, random_state=42, n_jobs=-1
    )

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    modele_xgb.fit(X_tr, y_tr, eval_set=[(X_tr,y_tr),(X_val, y_val)], verbose=False)

    print(f"\nArbres construits jusqu'a :{modele_xgb.n_estimators}(plafond)")
    print(f"Meilleur arbre (arret)      :{modele_xgb.best_iteration}")
    print(f"Arbres reellement utilisés  :{modele_xgb.best_iteration + 1}")
    
    pred_val_log = modele_xgb.predict(X_val)
    facteur_duan = np.mean(np.exp(y_val.values - pred_val_log))

    predictions_log = modele_xgb.predict(X_test)
    prix_reels_euros = np.exp(y_test)
    prix_predits_euros = np.exp(predictions_log) * facteur_duan

    mae = mean_absolute_error(prix_reels_euros, prix_predits_euros)
    mape = np.mean(np.abs((prix_reels_euros - prix_predits_euros)/prix_reels_euros)) * 100
    erreur_mediane = np.median(np.abs(prix_reels_euros - prix_predits_euros))
    rmse = np.sqrt(np.mean((prix_reels_euros.values - prix_predits_euros)**2))
    erreur_rel = np.abs(prix_reels_euros.values - prix_predits_euros)/prix_reels_euros
    r2_euros = r2_score(prix_reels_euros, prix_predits_euros)


    print(f"Courbe d'apprentisssage enregistree")
    print(f"Calcul des valeurs SHAP pour {type_bien}...")
    explainer = shap.TreeExplainer(modele_xgb)
    shap_values = explainer.shap_values(X_test)

    importance_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
    print(f"\nTop 5 SHAP ({type_bien}) :")
    for nom, val in importance_shap.head(5).items():
        print(f"  - {nom.ljust(25)} : {val:.4f}")

    nom_fichier_base = nom_zone.replace(' ', '_')
    
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title(f"Impact des variables - {nom_zone} ({type_bien})", fontsize=13)
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"shap_summary_{nom_fichier_base}_{type_bien}.png", dpi=150, bbox_inches='tight')
    plt.close()
    # Dependence plots des variables les plus instructives
    variables_a_tracer = ['prix_m2_section', 'surface_reelle_bati',
                          'prix_m2_voisins', 'median_revenu_disponible']
    # Affichage de la courbe d'apprentissage
    resultats_eval = modele_xgb.evals_result()

    plt.figure(figsize=(10,6))
    plt.plot(resultats_eval['validation_0']['rmse'],label='Entrainement',color='steelblue')
    plt.plot(resultats_eval['validation_1']['rmse'],label='Validation',color='darkorange')
    plt.axvline(modele_xgb.best_iteration,color='green',linestyle='--',
                label=f'Arret optimal (arbre {modele_xgb.best_iteration})')
    plt.xlabel("Nombre d'arbre")
    plt.ylabel("RMSE (espace log)")
    plt.title(f"Courbe d'apprentissage - {nom_zone}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier_graphes/f"courbe_apprentissage_{nom_fichier_base}_{type_bien}.png")
    plt.close()
    # Le terrain n'a de sens que pour les maisons
    if type_bien == 'maisons':
        variables_a_tracer.append('surface_terrain')

    for var in variables_a_tracer:
        if var not in X_test.columns:
            continue
        # On saute les variables constantes (ex: terrain a 0 partout)
        if X_test[var].nunique() <= 1:
            continue
        plt.figure()
        shap.dependence_plot(var, shap_values, X_test,
                             interaction_index=None, show=False)
        plt.title(f"Effet de {var} - {nom_zone} ({type_bien})", fontsize=11)
        plt.tight_layout()
        plt.savefig(dossier_graphes / f"shap_dep_{var}_{nom_fichier_base}_{type_bien}.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
    print(f"Dependence plots enregistres pour {type_bien}")
    
    plt.figure(figsize=(8,8))
    plt.scatter(prix_reels_euros, prix_predits_euros, alpha=0.3, s=10)
    lims = [min(prix_reels_euros.min(), prix_predits_euros.min()), max(prix_reels_euros.max(), prix_predits_euros.max())]
    plt.plot(lims, lims, 'r--', linewidth=2, label='Prédiction parfaite')
    plt.axhline(np.mean(prix_predits_euros), color='blue', linestyle=':', linewidth=2, label=f'Moyenne prédite : {np.mean(prix_predits_euros):.0f} EUR/m²')
    plt.xlabel("Prix réel (EUR/m²)")
    plt.ylabel("Prix prédit (EUR/m²)")
    plt.title(f"Prédictions vs réalité - {nom_zone} ({type_bien})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"pred_vs_reels_{nom_fichier_base}_{type_bien}.png", dpi=150)
    plt.close()

    residus = prix_reels_euros.values - prix_predits_euros
    plt.figure(figsize=(9,5))
    plt.hist(residus, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='Erreur nulle')
    plt.axvline(np.mean(residus), color='orange', linestyle='--', label=f'Biais moyen : {np.mean(residus):.0f} EUR/m²' )
    plt.xlabel("Résidu (réel - prédit) en EUR/m²")
    plt.ylabel("Nombre de biens")
    plt.title(f"Distribution des erreurs - {nom_zone} ({type_bien})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"residu_{nom_fichier_base}_{type_bien}.png", dpi=150)
    plt.close()

    print("\n" + "-"*50)
    print(f"RAPPORT XGBOOST - {type_bien.upper()}")
    print("-" * 50)
    print(f"Apprentissage : {len(X_train)} biens | Validation : {len(X_test)} biens")
    print(f"R2 (euros / m2) : {r2_euros * 100:.2f} %")
    print(f"MAE             : {mae:.2f} EUR / m2")
    print(f"MAPE            : {mape:.1f} %")
    print(f"Dans les +/- 10%: {np.mean(erreur_rel <= 0.10)*100:.1f} %")
    print(f"Dans les +/- 20%: {np.mean(erreur_rel <= 0.20)*100:.1f} %")

print("\n" + "="*50)
print(f"Temps de traitement global : {time.time() - temps_total_debut:.2f} secondes.")
