import pandas as pd
from sqlalchemy import create_engine

# ==========================================
# 1. CONNEXION MYSQL
# ==========================================
USER = 'root'
PASSWORD = '1618'
HOTE = 'localhost'
PORT = '3306'
DB = 'EstimationIA'

chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"
moteur = create_engine(chaine_connexion)

# ==========================================
# 2. LECTURE DU NOUVEAU FICHIER CSV
# ==========================================
print("📥 Lecture du fichier des gares ferroviaires...")

# Modifiez le chemin vers votre nouveau fichier de gares
chemin_fichier = '/home/sylvain-huang/Documents/EstimationIA/data/location-gare.csv'

df_gares = pd.read_csv(
    chemin_fichier,
    sep=';',
    dtype=str,
    encoding='latin1'
)

# Renommage des colonnes pour avoir des noms propres en base de données
df_gares = df_gares.rename(columns={
    'CODE_LIGNE': 'code_ligne',
    'NOM': 'nom_gare',
    'NATURE': 'nature',
    'LATITUDE (WGS84)': 'latitude_brute',
    'LONGITUDE (WGS84)': 'longitude_brute'
})

# ==========================================
# 3. NETTOYAGE ET FILTRAGE DES DONNÉES
# ==========================================
print("🧹 Nettoyage des coordonnées et application des filtres...")

# A. Correction du format des coordonnées (virgule -> point)
df_gares['latitude'] = df_gares['latitude_brute'].str.replace(',', '.')
df_gares['longitude'] = df_gares['longitude_brute'].str.replace(',', '.')

# Conversion en nombres décimaux (float)
df_gares['latitude'] = pd.to_numeric(df_gares['latitude'], errors='coerce')
df_gares['longitude'] = pd.to_numeric(df_gares['longitude'], errors='coerce')

# On supprime les anciennes colonnes brutes et les lignes sans coordonnées
df_gares = df_gares.drop(columns=['latitude_brute', 'longitude_brute'])
df_gares = df_gares.dropna(subset=['latitude', 'longitude'])

# B. Le filtre demandé : On supprime les gares qui sont UNIQUEMENT pour les voyageurs
# (donc on ne garde que celles qui ont du Fret ou de l'Infrastructure)
df_gares = df_gares[df_gares['nature'] != 'Desserte Voyageur']

# 💡 NOTE PRO : Si vous vouliez faire l'INVERSE (garder les voyageurs et enlever le fret pur),
# il vous suffirait de remplacer la ligne du dessus par celle-ci :
# df_gares = df_gares[df_gares['nature'].str.contains('Voyageur', na=False)]

print(f"📋 Nombre de gares retenues après filtrage : {len(df_gares)}")

# ==========================================
# 4. EXPORTATION VERS MYSQL
# ==========================================
print("🚀 Remplacement de la table 'donnees_transport' dans MySQL...")

# On utilise 'replace' pour écraser l'ancien fichier de transport par cette table propre
df_gares.to_sql(
    name='donnees_transport',
    con=moteur,
    if_exists='replace',
    index=False
)

print("✅ Données insérées avec succès.")