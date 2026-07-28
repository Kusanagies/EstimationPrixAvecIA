"""
MODULE CHARGEMENT - EstimationIA
=================================
Connexions aux bases, extraction des ventes (synthese), chargement des
enrichissements (DPE, revenus, densite, infrastructures, macro), fusion,
calcul des distances, et feature engineering de base.

Fonction principale : charger_et_preparer(FA, filtres, moteurs, gdf_littoral)
  -> renvoie le dataframe 'dp' pret pour l'entrainement + la liste features_base.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sqlalchemy import create_engine

RAYON = 6371000

COLONNES_DPE = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
COLONNES_CHAUFFAGE = ['pct_chauffage_elec', 'pct_chauffage_gaz', 'pct_chauffage_fioul', 'pct_chauffage_urbain']
COLONNES_REVENUS = ['median_revenu_disponible', 'indice_gini', 'pct_minima_sociaux']


def connexions(db_pass):
    """Renvoie (moteur_dvf, moteur_enr)."""
    moteur_dvf = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur_enr = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur_dvf.connect().close()
    moteur_enr.connect().close()
    return moteur_dvf, moteur_enr


def extraire_ventes(moteur_dvf, filtre_dvf):
    """Extrait les ventes depuis synthese, convertit les types, harmonise."""
    maisons = pd.read_sql(f"""
        SELECT communes_code AS code_commune, parcelles_code AS id_parcelle,
               lat AS latitude, lng AS longitude, prix_m2,
               surface AS surface_reelle_bati, typebien AS type_local,
               nb_pieces AS nombre_pieces_principales, surface_terrain,
               YEAR(date) AS annee_vente, MONTH(date) AS mois_vente
        FROM synthese
        WHERE {filtre_dvf}
          AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
          AND typebien IN ('maison', 'appartement');
    """, con=moteur_dvf)
    for col in ['prix_m2', 'surface_reelle_bati', 'surface_terrain',
                'nombre_pieces_principales', 'latitude', 'longitude', 'annee_vente', 'mois_vente']:
        maisons[col] = pd.to_numeric(maisons[col], errors='coerce')
    maisons = maisons.dropna(subset=['prix_m2', 'surface_reelle_bati',
                                     'nombre_pieces_principales', 'latitude', 'longitude'])
    maisons['type_local'] = maisons['type_local'].str.capitalize()
    return maisons


def charger_enrichissements(moteur_enr, FA, filtre_dpe, dep_infra):
    """Charge tous les enrichissements selon les choix FA. Renvoie un dict."""
    enr = {}
    enr['dpe'] = pd.read_sql(f"""
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
        FROM dpe_logements_france
        WHERE etiquette_dpe IN ('A','B','C','D','E','F','G') AND {filtre_dpe}
        GROUP BY code_insee_ban;
    """, con=moteur_enr)

    enr['stations'] = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur_enr)
    if dep_infra == 'FRANCE':
        q_mon = "SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL;"
        q_hop = "SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL;"
        q_uni = "SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL;"
    else:
        q_mon = f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{dep_infra}';"
        q_hop = f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal,2)='{dep_infra}';"
        q_uni = f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee,2)='{dep_infra}';"
    enr['monuments'] = pd.read_sql(q_mon, con=moteur_enr)
    enr['hopitaux'] = pd.read_sql(q_hop, con=moteur_enr)
    enr['universites'] = pd.read_sql(q_uni, con=moteur_enr)

    filtre_rev = "1=1" if dep_infra == 'FRANCE' else f"LEFT(code_commune,2)='{dep_infra}'"
    revenus = pd.read_sql(f"SELECT code_commune, median_revenu_disponible, indice_gini, pct_minima_sociaux FROM demographie_communes WHERE {filtre_rev};", con=moteur_enr)
    for col in COLONNES_REVENUS:
        revenus[col] = pd.to_numeric(revenus[col], errors='coerce')
    enr['revenus'] = revenus

    densite_pop = pd.read_sql("""
        SELECT d.code_commune, d.densite_population
        FROM densite_population d
        INNER JOIN (SELECT code_commune, MAX(annee) AS a FROM densite_population GROUP BY code_commune) m
          ON d.code_commune = m.code_commune AND d.annee = m.a
    """, con=moteur_enr)
    densite_pop['densite_population'] = pd.to_numeric(densite_pop['densite_population'], errors='coerce')
    enr['densite_pop'] = densite_pop

    enr['poles'] = None
    if FA['potentiel_urbain']:
        poles = pd.read_sql("""
            SELECT aav_nom, AVG(latitude) AS latitude, AVG(longitude) AS longitude, COUNT(*) AS poids_aire
            FROM referentiel_communes
            WHERE aav_nom IS NOT NULL AND aav_nom != 'SO' AND latitude IS NOT NULL
            GROUP BY aav_nom HAVING poids_aire >= 10
        """, con=moteur_enr)
        poles_etrangers = pd.DataFrame([
            {'aav_nom': 'Genève', 'latitude': 46.2044, 'longitude': 6.1432, 'poids_aire': 250},
            {'aav_nom': 'Lausanne', 'latitude': 46.5197, 'longitude': 6.6323, 'poids_aire': 90},
            {'aav_nom': 'Bâle', 'latitude': 47.5596, 'longitude': 7.5886, 'poids_aire': 150},
            {'aav_nom': 'Luxembourg', 'latitude': 49.6116, 'longitude': 6.1319, 'poids_aire': 200},
            {'aav_nom': 'Bruxelles', 'latitude': 50.8503, 'longitude': 4.3517, 'poids_aire': 300},
            {'aav_nom': 'Monaco', 'latitude': 43.7384, 'longitude': 7.4246, 'poids_aire': 120},
            {'aav_nom': 'Turin', 'latitude': 45.0703, 'longitude': 7.6869, 'poids_aire': 250},
            {'aav_nom': 'Barcelone', 'latitude': 41.3874, 'longitude': 2.1686, 'poids_aire': 300},
        ])
        poles = pd.concat([poles, poles_etrangers], ignore_index=True)
        for c in ['latitude', 'longitude', 'poids_aire']:
            poles[c] = pd.to_numeric(poles[c], errors='coerce')
        enr['poles'] = poles.dropna(subset=['latitude', 'longitude', 'poids_aire'])

    enr['pib'] = pd.read_sql("SELECT annee, pib_national FROM pib_national", con=moteur_enr) if FA['pib'] else None
    enr['chomage'] = pd.read_sql("SELECT code_departement, annee, trimestre, taux_chomage FROM chomage_departements", con=moteur_enr) if FA['chomage'] else None
    enr['taux'] = pd.read_sql("SELECT annee, mois, taux_credit_immo_fixe, taux_inflation FROM taux_macro", con=moteur_enr) if (FA['taux_credit'] or FA['taux_inflation']) else None
    if FA['ipc']:
        ipc = pd.read_sql("SELECT annee, mois, indice_prix_conso FROM indice_prix_conso", con=moteur_enr)
        ipc['indice_prix_conso'] = pd.to_numeric(ipc['indice_prix_conso'], errors='coerce')
        ipc = ipc.sort_values(['annee', 'mois']).reset_index(drop=True)
        ipc['inflation_mensuelle'] = ipc['indice_prix_conso'].pct_change().fillna(0) * 100
        enr['ipc'] = ipc
    else:
        enr['ipc'] = None
    return enr


def _contour(sg):
    pts = []
    for g in sg.geometry:
        if g.geom_type == 'MultiPolygon':
            for p in g.geoms:
                pts.extend(list(p.exterior.coords))
        else:
            pts.extend(list(g.exterior.coords))
    if not pts:
        return pd.DataFrame(columns=['latitude', 'longitude'])
    a = np.array(pts)
    return pd.DataFrame(a[:, [1, 0]], columns=['latitude', 'longitude'])


def fusion_et_distances(maisons, enr, FA, gdf_littoral):
    """Fusionne enrichissements et calcule les distances. Renvoie 'donnees'."""
    donnees = pd.merge(maisons, enr['dpe'], left_on='code_commune', right_on='code_insee_ban', how='left')
    donnees = pd.merge(donnees, enr['revenus'], on='code_commune', how='left')
    donnees = pd.merge(donnees, enr['densite_pop'], on='code_commune', how='left')
    donnees['code_departement'] = donnees['code_commune'].str[:2]
    donnees['trimestre'] = (donnees['mois_vente'] - 1) // 3 + 1

    if enr['pib'] is not None:
        donnees = pd.merge(donnees, enr['pib'], left_on='annee_vente', right_on='annee', how='left').drop(columns=['annee'], errors='ignore')
    if enr['chomage'] is not None:
        donnees = pd.merge(donnees, enr['chomage'], left_on=['code_departement', 'annee_vente', 'trimestre'],
                           right_on=['code_departement', 'annee', 'trimestre'], how='left').drop(columns=['annee'], errors='ignore')
    if enr['taux'] is not None:
        donnees = pd.merge(donnees, enr['taux'], left_on=['annee_vente', 'mois_vente'],
                           right_on=['annee', 'mois'], how='left').drop(columns=['annee', 'mois'], errors='ignore')
    if enr['ipc'] is not None:
        donnees = pd.merge(donnees, enr['ipc'][['annee', 'mois', 'indice_prix_conso', 'inflation_mensuelle']],
                           left_on=['annee_vente', 'mois_vente'], right_on=['annee', 'mois'], how='left').drop(columns=['annee', 'mois'], errors='ignore')

    points_rad = np.deg2rad(donnees[['latitude', 'longitude']])

    def dist_min(df_points, col):
        if len(df_points) > 0:
            arbre = BallTree(np.deg2rad(df_points.iloc[:, 0:2]), metric='haversine')
            d, _ = arbre.query(points_rad, k=1)
            donnees[col] = d.flatten() * RAYON
        else:
            donnees[col] = 999999

    if FA['dist_transport']: dist_min(enr['stations'], 'dist_transport_m')
    if FA['dist_monument']:  dist_min(enr['monuments'], 'dist_monument_m')
    if FA['dist_hopital']:   dist_min(enr['hopitaux'], 'dist_hopital_m')

    if FA['dist_universite']:
        if len(enr['universites']) > 0:
            au = BallTree(np.deg2rad(enr['universites'][['latitude', 'longitude']]), metric='haversine')
            d, iu = au.query(points_rad, k=1)
            donnees['dist_universite_m'] = d.flatten() * RAYON
            donnees['volume_etudiants_proche'] = enr['universites'].iloc[iu.flatten()]['nombre_etudiants'].values
        else:
            donnees['dist_universite_m'] = 999999
            donnees['volume_etudiants_proche'] = 0
    else:
        donnees['volume_etudiants_proche'] = 0

    if FA['dist_littoral']:
        for cl, col in {'Mer': 'dist_mer_m', 'Lac': 'dist_lac_m', 'Estuaire': 'dist_estuaire_m'}.items():
            dist_min(_contour(gdf_littoral[gdf_littoral['CLASSEMENT'] == cl]), col)

    if FA['potentiel_urbain'] and enr['poles'] is not None and len(enr['poles']) > 0:
        poles = enr['poles']
        ap = BallTree(np.deg2rad(poles[['latitude', 'longitude']].values), metric='haversine')
        pp = poles['poids_aire'].values.astype(float)
        drp, ip = ap.query(points_rad, k=min(20, len(poles)))
        donnees['potentiel_urbain'] = np.sum(pp[ip] / (drp * RAYON + 5000), axis=1)

    return donnees


def nettoyer_et_feature_base(donnees, FA):
    """Remplit les NaN, cree les features de base, renvoie (dp, features_base)."""
    for col in COLONNES_DPE + COLONNES_CHAUFFAGE + COLONNES_REVENUS:
        if col in donnees.columns:
            donnees[col] = donnees[col].fillna(donnees[col].median())
    donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)
    donnees['surface_terrain'] = donnees['surface_terrain'].fillna(0)
    for col in ['pib_national', 'taux_chomage', 'taux_credit_immo_fixe', 'taux_inflation',
                'potentiel_urbain', 'densite_population', 'indice_prix_conso', 'inflation_mensuelle']:
        if col in donnees.columns:
            donnees[col] = donnees[col].fillna(donnees[col].median())

    dp = donnees[(donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)].copy()
    dp.loc[dp['type_local'] == 'Appartement', 'surface_terrain'] = 0
    dp['prix_total'] = dp['prix_m2'] * dp['surface_reelle_bati']
    dp['log_prix_total'] = np.log(dp['prix_total'])
    dp['log_surface'] = np.log(dp['surface_reelle_bati'])
    dp['surface_par_piece'] = dp['surface_reelle_bati'] / dp['nombre_pieces_principales']
    dp['a_terrain'] = (dp['surface_terrain'] > 0).astype(int)
    dp['log_terrain'] = np.log1p(dp['surface_terrain'])
    dp['code_section'] = dp['id_parcelle'].str[:10]

    f = []
    if FA['geo_base']:   f += ['latitude', 'longitude']
    if FA['pieces']:     f += ['nombre_pieces_principales']
    if FA['date']:       f += ['annee_vente', 'mois_vente']
    if FA['terrain']:    f += ['a_terrain', 'surface_terrain', 'log_terrain']
    if FA['surface']:    f += ['surface_reelle_bati', 'log_surface', 'surface_par_piece']
    if FA['dist_universite']: f += ['volume_etudiants_proche']
    if FA['revenus']:    f += COLONNES_REVENUS
    if FA['densite_pop'] and 'densite_population' in dp.columns: f += ['densite_population']
    if FA['potentiel_urbain'] and 'potentiel_urbain' in dp.columns: f += ['potentiel_urbain']
    if FA['dpe']:        f += COLONNES_DPE
    if FA['chauffage']:  f += COLONNES_CHAUFFAGE
    if FA['dist_transport']:  f += ['dist_transport_m']
    if FA['dist_monument']:   f += ['dist_monument_m']
    if FA['dist_hopital']:    f += ['dist_hopital_m']
    if FA['dist_universite']: f += ['dist_universite_m']
    if FA['dist_littoral']:   f += ['dist_mer_m', 'dist_lac_m', 'dist_estuaire_m']
    if FA['pib'] and 'pib_national' in dp.columns: f += ['pib_national']
    if FA['ipc'] and 'indice_prix_conso' in dp.columns: f += ['indice_prix_conso', 'inflation_mensuelle']
    if FA['chomage'] and 'taux_chomage' in dp.columns: f += ['taux_chomage']
    if FA['taux_credit'] and 'taux_credit_immo_fixe' in dp.columns: f += ['taux_credit_immo_fixe']
    if FA['taux_inflation'] and 'taux_inflation' in dp.columns: f += ['taux_inflation']
    features_base = list(dict.fromkeys(f))

    return dp, features_base