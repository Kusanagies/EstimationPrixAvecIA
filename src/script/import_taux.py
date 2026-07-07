"""
IMPORT DES TAUX MACROECONOMIQUES DANS LA BASE
==============================================
Lit le fichier concatene (taux_macro_concatenes.csv) et l'importe dans une
table 'taux_macro', indexee sur (annee, mois) pour une jointure rapide avec
les ventes DVF (annee_vente, mois_vente).
"""

import sys
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, Float

# --- Connexion ---
RACINE_PROJET = Path(__file__).resolve().parents[2]
from dotenv import load_dotenv
load_dotenv(RACINE_PROJET / ".env")

CHEMIN_CSV = Path("/home/sylvain-huang/Documents/EstimationIA/data/taux_macro_concatenes.csv")

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except Exception:
    print("Erreur de connexion a la base.")
    sys.exit()

# --- Lecture du CSV concatene (separateur ; et virgule decimale francaise) ---
table = pd.read_csv(CHEMIN_CSV, sep=';', decimal=',', na_values=['-', '', ' '])
# Verification : annee et mois doivent etre entiers
table['annee'] = table['annee'].astype(int)
table['mois'] = table['mois'].astype(int)

print(f"Lignes lues : {len(table)}")
print(f"Colonnes    : {list(table.columns)}")
print(f"Periode     : {table['annee'].min()} a {table['annee'].max()}")
print("\nApercu :")
print(table.head().to_string(index=False))

# --- Types SQL explicites (annee/mois en Integer, taux en Float) ---
colonnes_taux = [c for c in table.columns if c not in ('annee', 'mois')]
types_sql = {'annee': Integer(), 'mois': Integer()}
for c in colonnes_taux:
    types_sql[c] = Float()

# --- Import ---
table.to_sql('taux_macro', con=moteur, if_exists='replace', index=False, dtype=types_sql)
print(f"\nTable 'taux_macro' creee ({len(table)} lignes).")

# --- Index sur (annee, mois) pour la jointure ---
with moteur.connect() as conn:
    from sqlalchemy import text
    conn.execute(text("CREATE INDEX idx_annee_mois ON taux_macro(annee, mois);"))
    conn.commit()
print("Index (annee, mois) cree.")