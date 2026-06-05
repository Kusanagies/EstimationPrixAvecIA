import pandas as pd
from sqlalchemy import create_engine
import numpy as np

# ==========================================
# 1. PARAMÈTRES ET CONNEXION MYSQL
# ==========================================
USER = 'root'
PASSWORD = '1618'
HOTE = 'localhost'
PORT = '3306'
DB = 'EstimationIA'

chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"
moteur = create_engine(chaine_connexion)

# ==========================================
# 2. SÉLECTION ET LECTURE DU FICHIER
# ==========================================
print("Lecture de la base des Monuments Historiques...")

# Les noms exacts des colonnes de votre fichier
colonnes_utiles = [
    'Reference',
    'Denomination_de_l_edifice',
    'Format_abrege_du_siecle_de_construction',
    'Adresse_forme_editoriale',
    'Commune_forme_index',
    'COG_Insee_lors_de_la_protection',
    'coordonnees_au_format_WGS84'
]

# Lecture du fichier (Attention au séparateur '|')
df_monuments = pd.read_csv(
    '/home/sylvain-huang/Documents/EstimationIA/data/merimee.csv',
    sep='|', # Votre fichier utilise des barres verticales comme séparateur !
    usecols=colonnes_utiles,
    dtype=str # On lit tout en texte pour éviter les erreurs de format
)

# ==========================================
# 3. NETTOYAGE ET SÉPARATION DES COORDONNÉES
# ==========================================
print("Nettoyage des données et séparation des coordonnées GPS...")

# On renomme les colonnes pour que ce soit plus propre dans SQL
df_monuments = df_monuments.rename(columns={
    'Reference': 'id_monument',
    'Denomination_de_l_edifice': 'type_monument',
    'Format_abrege_du_siecle_de_construction': 'siecle',
    'Adresse_forme_editoriale': 'adresse',
    'Commune_forme_index': 'nom_commune',
    'COG_Insee_lors_de_la_protection': 'code_insee'
})

# On supprime les monuments qui n'ont pas de coordonnées GPS
df_monuments = df_monuments.dropna(subset=['coordonnees_au_format_WGS84'])

# L'ASTUCE : On coupe la colonne en deux au niveau de la virgule
df_monuments[['latitude', 'longitude']] = df_monuments['coordonnees_au_format_WGS84'].str.split(',', expand=True)

# On convertit en vrai format numérique pour les calculs de distance futurs
df_monuments['latitude'] = pd.to_numeric(df_monuments['latitude'], errors='coerce')
df_monuments['longitude'] = pd.to_numeric(df_monuments['longitude'], errors='coerce')

# On supprime l'ancienne colonne combinée qui ne sert plus à rien
df_monuments = df_monuments.drop(columns=['coordonnees_au_format_WGS84'])

# ==========================================
# 4. FILTRAGE IDF (Optionnel) ET INSERTION
# ==========================================
# Si vous voulez vous concentrer uniquement sur l'Île-de-France :
idf_prefixes = ('75', '77', '78', '91', '92', '93', '94', '95')
df_monuments_idf = df_monuments[df_monuments['code_insee'].str.startswith(idf_prefixes, na=False)].copy()

print(f"{len(df_monuments_idf)} monuments historiques trouvés. Insertion dans MySQL...")

# Insertion dans la base de données
df_monuments_idf.to_sql(
    name='monuments_historiques', 
    con=moteur, 
    if_exists='replace', # Ici on peut replace car on crée la table pour la 1ère fois
    index=False
)

print("✅ Terminé ! La table 'monuments_historiques' est prête.")