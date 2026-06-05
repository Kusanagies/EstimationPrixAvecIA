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
# 2. LECTURE ET FILTRAGE DES COLONNES
# ==========================================
print("📥 Lecture des données de DansMaRue (Paris)...")

colonnes_utiles = [
    'ID DECLARATION',
    'TYPE DECLARATION',
    'CODE POSTAL',
    'DATE DECLARATION',
    'geo_point_2d'
]

# Modifiez le chemin vers votre fichier DMR
df_dmr = pd.read_csv(
    '/home/sylvain-huang/Documents/EstimationIA/data/dans-ma-rue.csv',
    sep=';', # Généralement séparé par un point-virgule sur Open Data Paris
    usecols=colonnes_utiles,
    dtype=str
)

# Renommage propre pour SQL
df_dmr = df_dmr.rename(columns={
    'ID DECLARATION': 'id_declaration',
    'TYPE DECLARATION': 'type_anomalie',
    'CODE POSTAL': 'code_postal',
    'DATE DECLARATION': 'date_declaration'
})

# ==========================================
# 3. TRAITEMENT GÉOSPATIAL (Lat/Lon)
# ==========================================
print("🧹 Extraction et conversion des coordonnées GPS...")

# On supprime les lignes sans coordonnées
df_dmr = df_dmr.dropna(subset=['geo_point_2d'])

# La colonne geo_point_2d est au format "Latitude, Longitude"
df_dmr[['latitude', 'longitude']] = df_dmr['geo_point_2d'].str.split(',', expand=True)

# Conversion en formats numériques exploitables par l'IA
df_dmr['latitude'] = pd.to_numeric(df_dmr['latitude'], errors='coerce')
df_dmr['longitude'] = pd.to_numeric(df_dmr['longitude'], errors='coerce')

# Suppression de l'ancienne colonne texte
df_dmr = df_dmr.drop(columns=['geo_point_2d'])
df_dmr = df_dmr.dropna(subset=['latitude', 'longitude'])

# ==========================================
# 4. EXPORTATION VERS MYSQL
# ==========================================
print(f"🚀 Insertion de {len(df_dmr)} anomalies urbaines dans la base de données...")

df_dmr.to_sql(
    name='urbanisme_anomalies_paris',
    con=moteur,
    if_exists='replace',
    index=False
)

print("✅ Table 'urbanisme_anomalies_paris' créée avec succès !")