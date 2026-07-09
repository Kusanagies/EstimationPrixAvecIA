"""
IMPORT DU PIB NATIONAL DANS LA BASE
====================================
Lit pib_national.csv (annee ; pib_national) et l'importe dans une table
'pib_national', indexee sur l'annee pour une jointure rapide avec les ventes
DVF (via annee_vente).

Types explicites (annee en Integer, pib en Float) pour eviter le bug d'index
sur colonne TEXT rencontre lors d'imports precedents.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, Float
from dotenv import load_dotenv

# --- Connexion ---
RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")

CHEMIN_CSV = Path("/home/sylvain-huang/Documents/EstimationIA/data/pib_national.csv")

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except Exception:
    print("Erreur de connexion a la base.")
    sys.exit()

# --- Lecture du CSV (separateur ; et virgule decimale francaise) ---
# na_values gere d'eventuels tirets ou cases vides (convention INSEE)
table = pd.read_csv(CHEMIN_CSV, sep=';', decimal=',', na_values=['-', '', ' '])

# Verification / typage
table['annee'] = table['annee'].astype(int)
table['pib_national'] = pd.to_numeric(table['pib_national'], errors='coerce')

print(f"Lignes lues : {len(table)}")
print(f"Colonnes    : {list(table.columns)}")
print(f"Periode     : {table['annee'].min()} a {table['annee'].max()}")
print("\nApercu (dernieres annees) :")
print(table.tail(10).to_string(index=False))

# --- Import avec types explicites ---
types_sql = {'annee': Integer(), 'pib_national': Float()}
table.to_sql('pib_national', con=moteur, if_exists='replace', index=False, dtype=types_sql)
print(f"\nTable 'pib_national' creee ({len(table)} lignes).")

# --- Index sur l'annee (pour la jointure avec annee_vente) ---
with moteur.connect() as conn:
    conn.execute(text("CREATE INDEX idx_annee ON pib_national(annee);"))
    conn.commit()
print("Index (annee) cree.")

print("\nPour joindre aux ventes dans un pipeline :")
print("  pib = pd.read_sql('SELECT annee, pib_national FROM pib_national', con=moteur)")
print("  donnees = pd.merge(donnees, pib, left_on='annee_vente', right_on='annee', how='left')")