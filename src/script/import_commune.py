"""
IMPORT du référentiel géographique des communes dans MySQL.
- Ne garde que les colonnes utiles au projet.
- Sépare 'geolocalisation' (format "lat, lon") en deux colonnes latitude / longitude.
- Crée une table 'referentiel_communes' indexée.

Lancer :  python3 import_referentiel_communes.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.types import VARCHAR, Float, Integer
from dotenv import load_dotenv

# ==========================================
# 0. CONNEXION
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[2]  # adapte la profondeur si besoin
load_dotenv(RACINE_PROJET / ".env")

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except KeyError:
    print("Erreur : variable DB_PASS introuvable.")
    sys.exit()
except Exception:
    print("Erreur de connexion a la base MySQL.")
    sys.exit()

# ==========================================
# 1. LECTURE DU CSV
# ==========================================
CHEMIN_CSV = "/home/sylvain-huang/Documents/EstimationIA/data/communes_france.csv" 

# On lit tout en texte pour éviter que les codes INSEE perdent leurs zéros
ref = pd.read_csv(CHEMIN_CSV, sep=';', dtype=str, encoding='utf-8')
print(f"Lignes lues : {len(ref)}")
print(f"Colonnes disponibles : {len(ref.columns)}")

# ==========================================
# 2. SÉLECTION DES COLONNES UTILES
# ==========================================
# On ne garde que ce qui sert au projet :
#  - code INSEE (jointure avec valeurs_foncieres / demographie)
#  - nom de commune
#  - aire d'attraction des villes (AAV) : le pôle d'influence
#  - unité urbaine (UU) : agglomération
#  - degré de densité (info sur le caractère urbain/rural)
#  - département
#  - geolocalisation (à séparer en lat/lon)
colonnes_utiles = {
    'COM_CODE': 'code_commune',
    'COM_NOM': 'nom_commune',
    'AAV_CODE': 'aav_code',
    'AAV_NOM': 'aav_nom',
    'UU_LIB': 'uu_nom',
    'DEGRE_DE_DENSITE_NOM': 'degre_densite',
    'DEP_CODE': 'code_departement',
    'geolocalisation': 'geolocalisation',
}

# Sécurité : ne garder que les colonnes réellement présentes
presentes = {k: v for k, v in colonnes_utiles.items() if k in ref.columns}
manquantes = [k for k in colonnes_utiles if k not in ref.columns]
if manquantes:
    print(f"Attention, colonnes absentes du fichier (ignorées) : {manquantes}")

ref = ref[list(presentes.keys())].rename(columns=presentes)

# ==========================================
# 3. SÉPARATION LATITUDE / LONGITUDE
# ==========================================
# 'geolocalisation' au format "45.774537409, 4.379534932"
coords = ref['geolocalisation'].str.split(',', expand=True)
ref['latitude'] = pd.to_numeric(coords[0], errors='coerce')
ref['longitude'] = pd.to_numeric(coords[1].str.strip() if coords.shape[1] > 1 else None, errors='coerce')

# On retire la colonne d'origine, désormais inutile
ref = ref.drop(columns=['geolocalisation'])

# Diagnostic : combien de communes ont des coordonnées valides
n_ok = ref[['latitude', 'longitude']].dropna().shape[0]
print(f"Communes avec coordonnees valides : {n_ok} / {len(ref)}")

# ==========================================
# 4. IMPORT DANS MYSQL
# ==========================================
ref.to_sql(
    name="referentiel_communes",
    con=moteur,
    if_exists="replace",
    index=False,
    chunksize=1000,
    dtype={
        'code_commune': VARCHAR(5),
        'nom_commune': VARCHAR(255),
        'aav_code': VARCHAR(10),
        'aav_nom': VARCHAR(255),
        'uu_nom': VARCHAR(255),
        'degre_densite': VARCHAR(100),
        'code_departement': VARCHAR(5),
        'latitude': Float,
        'longitude': Float,
    },
)
print("Donnees inserees dans 'referentiel_communes'.")

# ==========================================
# 5. INDEX + VÉRIFICATION
# ==========================================
with moteur.connect() as conn:
    # Index sur code_commune (jointures) et aav_nom (agrégats par pôle)
    conn.execute(text("CREATE INDEX idx_ref_commune ON referentiel_communes(code_commune);"))
    conn.execute(text("CREATE INDEX idx_ref_aav ON referentiel_communes(aav_nom(50));"))
    conn.commit()

    total = conn.execute(text("SELECT COUNT(*) FROM referentiel_communes;")).scalar()
    apercu = conn.execute(text("SELECT code_commune, nom_commune, aav_nom, latitude, longitude FROM referentiel_communes LIMIT 5;")).fetchall()

print(f"\nTable creee avec {total} communes.")
print("Apercu :")
for ligne in apercu:
    print(" ", ligne)