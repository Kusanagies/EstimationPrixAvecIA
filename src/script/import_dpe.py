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

# Les colonnes stratégiques pour l'immobilier (donnees INDIVIDUELLES du logement)
colonnes_a_garder = [
    # --- Identifiants et cles de jointure ---
    'numero_dpe',
    'identifiant_ban',
    'code_insee_ban',
    'numero_voie_ban',
    'nom_rue_ban',
    'nom_commune_ban',
    'adresse_ban',
    'adresse_complete_brut',
    # --- Dates ---
    'date_etablissement_dpe',
    # --- Caracteristiques du batiment ---
    'type_batiment',
    'typologie_logement',
    'annee_construction',
    'periode_construction',
    'surface_habitable_logement',
    'hauteur_sous_plafond',
    'nombre_niveau_logement',
    'numero_etage_appartement',
    'logement_traversant',
    # --- Performance energetique (individuelle) ---
    'etiquette_dpe',
    'etiquette_ges',
    'conso_5_usages_par_m2_ep',
    'qualite_isolation_enveloppe',
    'qualite_isolation_menuiseries',
    'cout_total_5_usages',
    'type_energie_principale_chauffage',
    'type_installation_chauffage',
    # --- Contexte geographique fin ---
    'classe_altitude',
    'zone_climatique',
]

print("Initialisation de l'importation nationale (Mode Chunking)...")

# ==========================================
# 3. LECTURE ET INSERTION EN BOUCLE
# ==========================================
debut_chrono = time.time()
nombre_lignes_inserees = 0
premier_lot = True

try:
    # pd.read_csv avec 'chunksize' renvoie un itérateur (une boucle) au lieu de tout charger en mémoire
    iterateur_csv = pd.read_csv(
        CHEMIN_FICHIER_DPE,
        sep=',',
        usecols=colonnes_a_garder,
        chunksize=TAILLE_LOT,
        low_memory=False
    )

    for lot in iterateur_csv:
        # --- Nettoyage des cles textuelles (preserver les zeros initiaux) ---
        lot['code_insee_ban'] = lot['code_insee_ban'].astype(str).str.zfill(5)
        lot['identifiant_ban'] = lot['identifiant_ban'].astype(str)

        # --- Insertion du lot ---
        # if_exists='append' partout : la table doit etre VIDE ou inexistante au depart
        # (tu as drop la table avant, donc le 1er lot la cree automatiquement)
        lot.to_sql(name=NOM_TABLE_SQL, con=moteur, if_exists='append', index=False)

        nombre_lignes_inserees += len(lot)
        print(f"Progression : {nombre_lignes_inserees} DPE insérés...")

except Exception as e:
    print(f"Une erreur s'est produite durant l'importation : {e}")

fin_chrono = time.time()
duree_minutes = round((fin_chrono - debut_chrono) / 60, 2)

print("==========================================")
print(f"Termine. {nombre_lignes_inserees} logements importés dans la table '{NOM_TABLE_SQL}'.")
print(f"Temps total : {duree_minutes} minutes.")
print("==========================================")
