import pandas as pd
from sqlalchemy import create_engine
import re
from pyproj import Transformer

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
# 2. LECTURE ET FILTRAGE
# ==========================================
print("Lecture du fichier des hôpitaux...")

colonnes_utiles = [
    'osm_id', 'name', 'emergency', 'addr-postcode', 'addr-city', 'the_geom'
]

# Modifiez le chemin vers votre fichier
df_hopitaux = pd.read_csv(
    '/home/sylvain-huang/Documents/EstimationIA/data/hospitals_point.csv',
    usecols=colonnes_utiles,
    dtype=str
)

# Renommage propre
df_hopitaux = df_hopitaux.rename(columns={
    'name': 'nom_hopital',
    'emergency': 'urgences',
    'addr-postcode': 'code_postal',
    'addr-city': 'nom_commune'
})

# ==========================================
# 3. EXTRACTION ET CONVERSION GPS
# ==========================================
print("Extraction et conversion des coordonnées en GPS standard...")

# 1. On extrait les deux chiffres du texte "POINT (X Y)" avec une expression régulière
coords = df_hopitaux['the_geom'].str.extract(r'POINT \(([-.\d]+) ([-.\d]+)\)')

# On crée des colonnes temporaires en Mercator (Mètres)
lon_3857 = pd.to_numeric(coords[0], errors='coerce')
lat_3857 = pd.to_numeric(coords[1], errors='coerce')

# 2. Le convertisseur Magique (De Web Mercator EPSG:3857 vers GPS classique EPSG:4326)
transformateur = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# On applique la conversion sur toutes les lignes d'un coup
df_hopitaux['longitude'], df_hopitaux['latitude'] = transformateur.transform(lon_3857.values, lat_3857.values)

# On supprime la vieille colonne illisible
df_hopitaux = df_hopitaux.drop(columns=['the_geom'])

# On supprime les lignes sans coordonnées
df_hopitaux = df_hopitaux.dropna(subset=['latitude', 'longitude'])

# ==========================================
# 4. INSERTION DANS MYSQL
# ==========================================
# Optionnel : Filtrer uniquement l'IDF si besoin

print(f"{len(df_hopitaux)} hôpitaux trouvés. Insertion en cours...")

df_hopitaux.to_sql(
    name='infrastructures_hopitaux', 
    con=moteur, 
    if_exists='replace', 
    index=False
)

print("✅ Terminé ! La table 'infrastructures_hopitaux' est prête et les coordonnées sont au bon format.")