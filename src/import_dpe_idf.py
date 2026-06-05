import pandas as pd
from sqlalchemy import create_engine

# ==========================================
# 1. PARAMÈTRES ET CONNEXION MYSQL
# ==========================================
USER = 'root'
PASSWORD = '1618'  # Votre mot de passe
HOTE = 'localhost'
PORT = '3306'
DB = 'EstimationIA'

chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"
moteur = create_engine(chaine_connexion)

# ==========================================
# 2. FILTRAGE ET LECTURE DU CSV
# ==========================================
print("Lecture et filtrage du fichier DPE...")

# La liste exacte des colonnes stratégiques à conserver
colonnes_a_garder = [
    'numero_dpe', 
    'date_etablissement_dpe', 
    'code_insee_ban', 
    'type_batiment',
    'typologie_logement',
    'annee_construction',
    'surface_habitable_logement',
    'hauteur_sous_plafond',
    'numero_etage_appartement',
    'logement_traversant',
    'etiquette_dpe',
    'etiquette_ges',
    'cout_total_5_usages',
    'type_energie_principale_chauffage'
]

# On lit le CSV en ne chargeant QUE les colonnes intéressantes (gain de RAM massif)
# Si votre fichier utilise un séparateur différent (comme la virgule), changez sep=','
df_dpe = pd.read_csv(
    '/home/sylvain-huang/Documents/EstimationIA/data/dpe03existant.csv', 
    sep=',', 
    usecols=colonnes_a_garder,
    low_memory=False # Évite les avertissements sur les types de données mixtes
)

# On filtre tout de suite pour ne garder que l'Île-de-France (Codes INSEE commençant par 75, 77, 78, 91, 92, 93, 94, 95)
# On s'assure d'abord que le code INSEE est bien une chaîne de caractères
df_dpe['code_insee_ban'] = df_dpe['code_insee_ban'].astype(str)
idf_prefixes = ('75', '77', '78', '91', '92', '93', '94', '95')
df_dpe_idf = df_dpe[df_dpe['code_insee_ban'].str.startswith(idf_prefixes)].copy()

print(f"{len(df_dpe_idf)} DPE trouvés en Île-de-France. Préparation de l'insertion SQL...")

# ==========================================
# 3. INSERTION DANS MYSQL
# ==========================================
# La fonction to_sql crée la table automatiquement et insère les données !
# if_exists='replace' écrasera la table si elle existe déjà, 'append' rajouterait les lignes
df_dpe_idf.to_sql(
    name='dpe_logements_idf', 
    con=moteur, 
    if_exists='replace', 
    index=False,
    chunksize=10000 # Insère par paquets de 10 000 pour ne pas brusquer MySQL
)

print("Succès ! La table 'dpe_logements_idf' a été créée et remplie dans MySQL.")