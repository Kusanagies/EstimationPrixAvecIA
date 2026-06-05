import pandas as pd
from sqlalchemy import create_engine
import time

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
# 2. CONFIGURATION DE L'IMPORTATION
# ==========================================
CHEMIN_FICHIER_DPE = '/home/sylvain-huang/Documents/EstimationIA/data/dpe03existant.csv'
NOM_TABLE_SQL = 'dpe_logements_france'
TAILLE_LOT = 50000  # On lit et on insère par blocs de 50 000 lignes

# Les colonnes stratégiques pour l'immobilier
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

print("Initialisation de l'importation nationale (Mode Chunking)...")

# ==========================================
# 3. LECTURE ET INSERTION EN BOUCLE
# ==========================================
debut_chrono = time.time()
nombre_lignes_inserees = 0
premier_lot = True

try:
    # pd.read_csv avec 'chunksize' renvoie un itérateur (une boucle) au lieu de charger tout le fichier en mémoire
    iterateur_csv = pd.read_csv(
        CHEMIN_FICHIER_DPE,
        sep=',', 
        usecols=colonnes_a_garder,
        chunksize=TAILLE_LOT,
        low_memory=False
    )
    
    for lot in iterateur_csv:
        # Nettoyage léger (Optionnel mais recommandé : s'assurer que le code INSEE est bien lu)
        lot['code_insee_ban'] = lot['code_insee_ban'].astype(str)
        
        # Insertion du lot dans MySQL
        if premier_lot:
            # Pour le 1er lot, on crée (ou remplace) la table
            lot.to_sql(name=NOM_TABLE_SQL, con=moteur, if_exists='replace', index=False)
            premier_lot = False
        else:
            # Pour les lots suivants, on ajoute (append) à la suite
            lot.to_sql(name=NOM_TABLE_SQL, con=moteur, if_exists='append', index=False)
        
        nombre_lignes_inserees += len(lot)
        print(f"Progression : {nombre_lignes_inserees} DPE insérés...")

except Exception as e:
    print(f"❌ Une erreur s'est produite durant l'importation : {e}")

fin_chrono = time.time()
duree_minutes = round((fin_chrono - debut_chrono) / 60, 2)

print("==========================================")
print(f"✅ Terminé ! {nombre_lignes_inserees} logements importés dans la table '{NOM_TABLE_SQL}'.")
print(f"⏱️ Temps total : {duree_minutes} minutes.")
print("==========================================")