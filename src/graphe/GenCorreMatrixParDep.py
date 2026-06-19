import time
import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import geopandas as gpd

# ==========================================
# 0. SAISIE UTILISATEUR
# ==========================================
departement = input("Veuillez saisir le numero du departement a analyser (ex: 34, 75, 13) : ").strip()

CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"
gdf_littoral = gpd.read_file(CHEMIN_GPKG)

if len(departement) < 2:
    print("Erreur : Le format du departement est invalide.")
    sys.exit()

print(f"\nLancement de l'analyse pour le departement : {departement}")
print("-" * 50)
temps_total_debut = time.time()

# ==========================================
# 1. CONNEXION MYSQL ET TÉLÉCHARGEMENT
# ==========================================
print("Etape 1/6 : Telechargement des donnees depuis la base SQL...")
moteur = create_engine("mysql+pymysql://root:1618@localhost:3306/EstimationIA")

# 1. IMMOBILIER (DVF)
maisons = pd.read_sql(f"""
    SELECT code_commune, latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2, surface_reelle_bati, type_local, YEAR(date_mutation) AS annee_vente
    FROM valeurs_foncieres
    WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
      AND LEFT(code_commune, 2) = '{departement}'
      AND type_local IN ('Maison', 'Appartement');
""", con=moteur)

if len(maisons) == 0:
    print(f"Erreur : Aucune donnee immobiliere trouvee pour le departement {departement}.")
    sys.exit()

# 2. DPE
dpe = pd.read_sql(f"""
    SELECT code_insee_ban, 
           (SUM(CASE WHEN etiquette_dpe = 'A' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_A,
           (SUM(CASE WHEN etiquette_dpe = 'B' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_B,
           (SUM(CASE WHEN etiquette_dpe = 'C' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_C,
           (SUM(CASE WHEN etiquette_dpe = 'D' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_D,
           (SUM(CASE WHEN etiquette_dpe = 'E' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_E,
           (SUM(CASE WHEN etiquette_dpe = 'F' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_F,
           (SUM(CASE WHEN etiquette_dpe = 'G' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS pct_dpe_G
    FROM dpe_logements_france
    WHERE etiquette_dpe IN ('A','B','C','D','E','F','G')
      AND LEFT(code_insee_ban, 2) = '{departement}'
    GROUP BY code_insee_ban;
""", con=moteur)

# 3. TRANSPORTS (Toutes les gares de France pour eviter les problemes de frontieres)
stations = pd.read_sql("SELECT latitude, longitude FROM donnees_transport WHERE latitude IS NOT NULL;", con=moteur)

# 4. MONUMENTS HISTORIQUES
monuments = pd.read_sql(f"SELECT latitude, longitude FROM monuments_historiques WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{departement}';", con=moteur)

# 5. HOPITAUX
hopitaux = pd.read_sql(f"SELECT latitude, longitude FROM infrastructures_hopitaux WHERE latitude IS NOT NULL AND LEFT(code_postal, 2) = '{departement}';", con=moteur)

# 6. UNIVERSITES
universites = pd.read_sql(f"SELECT latitude, longitude, nombre_etudiants FROM infrastructures_universites WHERE latitude IS NOT NULL AND LEFT(code_insee, 2) = '{departement}';", con=moteur)

# 7. REVENUS INSEE (Filosofi) par commune
revenus = pd.read_sql(f"""SELECT code_commune,median_revenu_disponible,indice_gini,pct_minima_sociaux 
                      FROM demographie_communes
                      WHERE LEFT(code_commune,2) = '{departement}';
""",con=moteur)

for col in ['median_revenu_disponible','indice_gini','pct_minima_sociaux']: revenus[col] = pd.to_numeric(revenus[col],errors='coerce')

# ==========================================
# 2. FUSION DES DONNEES
# ==========================================
print("Etape 2/6 : Fusion des donnees communales...")
donnees = pd.merge(maisons, dpe, left_on='code_commune', right_on='code_insee_ban', how='left')
donnees = pd.merge(donnees,revenus, on='code_commune',how='left')

donnees = donnees[
    (donnees['latitude'].between(41,51)) &
    (donnees['longitude'].between(-5,10))
]
# ==========================================
# 3. CALCULS SPATIAUX (BallTree)
# ==========================================
print("Etape 3/6 : Calculs geospatiaux des distances...")
RAYON_TERRE_METRES = 6371000
maisons_rad = np.deg2rad(donnees[['latitude', 'longitude']])

def calculer_distance_min(df_points, nom_colonne):
    if len(df_points) > 0:
        points_rad = np.deg2rad(df_points.iloc[:, 0:2])
        arbre = BallTree(points_rad, metric='haversine')
        dist_rad, _ = arbre.query(maisons_rad, k=1)
        donnees[nom_colonne] = dist_rad.flatten() * RAYON_TERRE_METRES
    else :
        donnees[nom_colonne] = 999999

calculer_distance_min(stations, 'dist_transport_m')
calculer_distance_min(monuments, 'dist_monument_m')
calculer_distance_min(hopitaux, 'dist_hopital_m')

if len(universites) > 0:
    univ_rad = np.deg2rad(universites[['latitude', 'longitude']])
    arbre_univ = BallTree(univ_rad, metric='haversine')
    dist_rad, idx_univ = arbre_univ.query(maisons_rad, k=1)
    donnees['dist_universite_m'] = dist_rad.flatten() * RAYON_TERRE_METRES
    donnees['volume_etudiants_proche'] = universites.iloc[idx_univ.flatten()]['nombre_etudiants'].values

def extraire_points_contour(sous_gdf):
    points = []
    for geom in sous_gdf.geometry: 
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                points.extend(list(poly.exterior.coords))
        else :
            points.extend(list(geom.exterior.coords))
    if not points:
        return pd.DataFrame(columns=['latitude','longitude'])
    pts = np.array(points)
    return pd.DataFrame(pts[:,[1,0]],columns=['latitude','longitude'])

classements = {'Mer':'dist_mer_m','Lac':'dist_lac_m','Estuaire':'dist_estuaire_m'}
for classement, nom_colonne in classements.items():
    sous = gdf_littoral[gdf_littoral['CLASSEMENT'] == classement]
    df_points = extraire_points_contour(sous)
    calculer_distance_min(df_points,nom_colonne)

for col in ['dist_transport_m','dist_mer_m','dist_lac_m']:
    print(f"{col} : min={donnees[col].min():.0f}m",
          f"median={donnees[col].median():.0f}m,"
          f"max={donnees[col].max():.0f}m")

# ==========================================
# 4. NETTOYAGE ET NORMALISATION
# ==========================================
print("Etape 4/6 : Nettoyage et normalisation des variables...")

# Securisation des universites si absentes du departement
if 'volume_etudiants_proche' not in donnees.columns:
    donnees['volume_etudiants_proche'] = 0
else:
    donnees['volume_etudiants_proche'] = donnees['volume_etudiants_proche'].fillna(0)

# Remplissage des DPE manquants
colonnes_dpe = ['pct_dpe_A', 'pct_dpe_B', 'pct_dpe_C', 'pct_dpe_D', 'pct_dpe_E', 'pct_dpe_F', 'pct_dpe_G']
for col in colonnes_dpe:
    if col in donnees.columns:
        donnees[col] = donnees[col].fillna(donnees[col].median())

colonnes_revenus = ['median_revenu_disponible','indice_gini','pct_minima_sociaux']
for col in colonnes_revenus: 
    if col in donnees.columns: 
        donnees[col] = donnees[col].fillna(donnees[col].median())

# Filtrage des valeurs aberrantes (Fourchette adaptee a la majorite de la France)

plancher = donnees['prix_m2'].quantile(0.01)
plafond = donnees['prix_m2'].quantile(0.99)

donnees_propres = donnees[
    (donnees['prix_m2'] >= plancher) & (donnees['prix_m2'] <= plafond) & 
    (donnees['surface_reelle_bati'] >= 9) & (donnees['surface_reelle_bati'] <= 300)
].copy()

# Transformations
donnees_propres['log_prix_m2'] = np.log(donnees_propres['prix_m2'])
donnees_propres['est_maison'] = (donnees_propres['type_local'] == 'Maison').astype(int)
donnees_propres['est_appart'] = (donnees_propres['type_local'] == 'Appartement').astype(int)

# Pour une matrice de correlation, ne pas normaliser les distances
# on garde juste la liste des colonnes
colonnes_dist = [col for col in donnees_propres.columns if col.startswith('dist_')]

# Variables quantitatives en Z-Score
colonnes_standard = ['surface_reelle_bati']
if 'volume_etudiants_proche' in donnees_propres.columns and donnees_propres['volume_etudiants_proche'].nunique() > 1:
    colonnes_standard.append('volume_etudiants_proche')

if colonnes_standard:
    donnees_propres[colonnes_standard] = StandardScaler().fit_transform(donnees_propres[colonnes_standard])

# ==========================================
# 5. MATRICE DE CORRELATION
# ==========================================
print("Etape 5/6 : Calcul de la matrice de correlation...")
colonnes_finales = ['log_prix_m2', 'est_maison','prix_m2','est_appart','annee_vente'] + colonnes_dpe + colonnes_standard + colonnes_dist + colonnes_revenus

# Filtrage pour ne garder que les colonnes qui existent et qui ne sont pas constantes
colonnes_valides = [col for col in colonnes_finales if col in donnees_propres.columns and donnees_propres[col].nunique() > 1]
matrice_corr = donnees_propres[colonnes_valides].corr()

# ==========================================
# 6. AFFICHAGE TEXTE DES RESULTATS
# ==========================================
print("Etape 6/6 : Analyse terminee.")
print("\n" + "="*50)
print(f"TOP DES CORRELATIONS AVEC LE PRIX AU M2 (DEP {departement})")
print("="*50)

correlations_prix = matrice_corr['log_prix_m2'].drop('log_prix_m2').sort_values(ascending=False)

print("\nIMPACTS POSITIFS (Font monter le prix) :")
for index, valeur in correlations_prix[correlations_prix > 0].items():
    print(f"  + {index.ljust(25)} : {valeur:+.3f}")

print("\nIMPACTS NEGATIFS (Font baisser le prix) :")
for index, valeur in correlations_prix[correlations_prix < 0].items():
    print(f"  - {index.ljust(25)} : {valeur:+.3f}")

print("\n" + "="*50)
print(f"Temps total d'execution : {time.time() - temps_total_debut:.2f} secondes.")

# Generation du graphe
plt.figure(figsize=(16, 12))

# =========================
# EXPORT CSV
# ========================
# 1. La matrice de correlation complete
nom_fichier_matrice = f"matrice_correlation_dep_{departement}.csv"
matrice_corr.to_csv(nom_fichier_matrice,sep=';',decimal=',',encoding='utf-8-sig')

# 2. Le classement des correlations avec le prix (plus lisible)
nom_fichier_top = f"correlation_prix_dep_{departement}.csv"
correlations_prix.to_csv(nom_fichier_top,sep=';',decimal=',',
                         header=['correlation_avec_log_prix'],encoding='utf-8-sig')
print(f"\nFichiers CSV génerés : ")
print(f" - {nom_fichier_matrice}")
print(f" - {nom_fichier_top}")

masque = np.triu(np.ones_like(matrice_corr, dtype=bool))
sns.heatmap(matrice_corr, mask=masque, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5, annot_kws={"size": 9})
plt.title(f"Master Dataset - Departement {departement}", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

evolution = donnees_propres.groupby('annee_vente')['prix_m2'].median()
print("\nPrix médian au m² par annee :")
print(evolution)