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
# 2. LECTURE DU FICHIER ET SÉLECTION
# ==========================================
print("🎓 Lecture de la base des universités...")

colonnes_a_lire = [
    "code UAI de l'établissement",
    "sigle de l'établissement",
    "code commune",
    "Commune",
    "gps",
    "nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE"
]

# Attention : modifiez le chemin vers votre fichier
df_univ = pd.read_csv(
    '/home/sylvain-huang/Documents/EstimationIA/data/EtablissementSuperieur.csv',
    sep=';', # Le séparateur est un point-virgule dans ce fichier !
    usecols=colonnes_a_lire,
    dtype=str
)

# Renommage propre pour la base de données SQL
df_univ = df_univ.rename(columns={
    "code UAI de l'établissement": "id_uai",
    "sigle de l'établissement": "nom_universite",
    "code commune": "code_insee",
    "Commune": "nom_commune",
    "gps": "coordonnees_gps",
    "nombre total d’étudiants inscrits hors doubles inscriptions université/CPGE": "nombre_etudiants"
})

# ==========================================
# 3. NETTOYAGE DES DONNÉES
# ==========================================
print("🧹 Nettoyage des coordonnées et conversion...")

# On supprime les lignes sans GPS
df_univ = df_univ.dropna(subset=['coordonnees_gps'])

# L'ASTUCE : On coupe la colonne GPS en deux au niveau de la virgule
df_univ[['latitude', 'longitude']] = df_univ['coordonnees_gps'].str.split(',', expand=True)

# On convertit les textes en vrais chiffres
df_univ['latitude'] = pd.to_numeric(df_univ['latitude'], errors='coerce')
df_univ['longitude'] = pd.to_numeric(df_univ['longitude'], errors='coerce')
df_univ['nombre_etudiants'] = pd.to_numeric(df_univ['nombre_etudiants'], errors='coerce').fillna(0).astype(int)

# On supprime la vieille colonne "gps" fusionnée
df_univ = df_univ.drop(columns=['coordonnees_gps'])

# ==========================================
# 4. FILTRAGE IDF ET AGRÉGATION
# ==========================================
# Dans le fichier brut, une même université apparaît sur plusieurs lignes (une ligne par niveau de diplôme).
# Nous voulons regrouper tout ça pour avoir 1 seule ligne par établissement avec le TOTAL des étudiants.

print("📊 Regroupement des étudiants par établissement...")
df_univ = df_univ.groupby(['id_uai', 'nom_universite', 'code_insee', 'nom_commune', 'latitude', 'longitude'], as_index=False)['nombre_etudiants'].sum()

# On ne garde que l'Île-de-France (Codes INSEE commençant par 75, 77, 78, 91, 92, 93, 94, 95)
idf_prefixes = ('75', '77', '78', '91', '92', '93', '94', '95')
df_univ_idf = df_univ[df_univ['code_insee'].str.startswith(idf_prefixes, na=False)].copy()

print(f"✅ {len(df_univ_idf)} établissements trouvés en IDF. Insertion dans MySQL...")

# ==========================================
# 5. INSERTION DANS LA BASE
# ==========================================
df_univ_idf.to_sql(
    name='infrastructures_universites', 
    con=moteur, 
    if_exists='replace', # Ok ici car c'est la première création
    index=False
)

print("🎉 Terminé ! La table 'infrastructures_universites' est prête.")