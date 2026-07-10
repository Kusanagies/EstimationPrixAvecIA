"""
IMPORT DU TAUX DE CHOMAGE PAR DEPARTEMENT ET TRIMESTRE
======================================================
Le fichier source est en format "large" : une colonne par trimestre
(T1_1982, T2_1982, ... T1_2026), avec Code (departement) et Libellé.

Ce script le transforme en format "long" (une ligne par departement-trimestre)
avec des colonnes exploitables : code_departement, annee, trimestre, taux_chomage.

INTERET : le chomage varie PAR DEPARTEMENT (contrairement au PIB national),
donc il discrimine les territoires - potentiellement utile pour le prix.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, Float, VARCHAR
from dotenv import load_dotenv

# --- Connexion ---
RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")

CHEMIN_CSV = Path("/home/sylvain-huang/Documents/EstimationIA/data/TauxDeChomage.csv")

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except Exception:
    print("Erreur de connexion a la base.")
    sys.exit()

# --- Lecture du CSV large ---
# na_values gere les cases vides / tirets (convention INSEE)
# dtype=str sur la 1ere colonne pour garder le zero initial (01, 02... pas 1, 2)
df = pd.read_csv(CHEMIN_CSV, sep=',', na_values=['-', '', ' ', 'N/A'], dtype={0: str})
print(f"Fichier lu : {df.shape[0]} departements x {df.shape[1]} colonnes")
print(f"Premieres colonnes : {list(df.columns[:4])}")

# --- Identification des colonnes ---
# Les 2 premieres sont Code et Libellé, le reste sont les trimestres (T1_1982, ...)
col_code = df.columns[0]      # 'Code'
col_libelle = df.columns[1]   # 'Libellé'
colonnes_trimestres = df.columns[2:]  # toutes les colonnes T*_AAAA

# --- Passage au format long (melt) ---
# Chaque colonne trimestre devient des lignes
table_longue = df.melt(
    id_vars=[col_code, col_libelle],
    value_vars=colonnes_trimestres,
    var_name='periode',        # ex: 'T1_1982'
    value_name='taux_chomage'
)

# --- Extraction annee et trimestre depuis 'periode' (ex: 'T1_1982') ---
# Format : T{trimestre}_{annee}
table_longue['trimestre'] = table_longue['periode'].str[1].astype(int)   # le chiffre apres T
table_longue['annee'] = table_longue['periode'].str[3:].astype(int)      # apres le _

# --- Nettoyage ---
table_longue = table_longue.rename(columns={col_code: 'code_departement', col_libelle: 'nom_departement'})
table_longue['code_departement'] = table_longue['code_departement'].astype(str).str.strip()
table_longue['taux_chomage'] = pd.to_numeric(table_longue['taux_chomage'], errors='coerce')

# On garde les colonnes utiles, ordonnees
table_finale = table_longue[['code_departement', 'nom_departement', 'annee', 'trimestre', 'taux_chomage']]
table_finale = table_finale.dropna(subset=['taux_chomage'])
table_finale = table_finale.sort_values(['code_departement', 'annee', 'trimestre']).reset_index(drop=True)

print(f"\nTable longue : {len(table_finale)} lignes "
      f"({table_finale['code_departement'].nunique()} departements x "
      f"{table_finale['annee'].nunique()} annees x 4 trimestres)")
print(f"Periode : {table_finale['annee'].min()} a {table_finale['annee'].max()}")
print("\nApercu :")
print(table_finale.head(10).to_string(index=False))

# --- Import avec types explicites ---
types_sql = {
    'code_departement': VARCHAR(10),   # VARCHAR (pas TEXT) pour pouvoir indexer
    'nom_departement': VARCHAR(100),
    'annee': Integer(),
    'trimestre': Integer(),
    'taux_chomage': Float(),
}
table_finale.to_sql('chomage_departements', con=moteur, if_exists='replace', index=False, dtype=types_sql)
print(f"\nTable 'chomage_departements' creee ({len(table_finale)} lignes).")

# --- Index sur (departement, annee, trimestre) pour la jointure ---
with moteur.connect() as conn:
    conn.execute(text("CREATE INDEX idx_dep_annee_trim ON chomage_departements(code_departement, annee, trimestre);"))
    conn.commit()
print("Index (code_departement, annee, trimestre) cree.")

print("\nPour joindre aux ventes (le mois de vente -> trimestre) :")
print("  chomage = pd.read_sql('SELECT code_departement, annee, trimestre, taux_chomage FROM chomage_departements', con=moteur)")
print("  donnees['code_departement'] = donnees['code_commune'].str[:2]")
print("  donnees['trimestre'] = (donnees['mois_vente'] - 1) // 3 + 1")
print("  donnees = pd.merge(donnees, chomage,")
print("                     left_on=['code_departement','annee_vente','trimestre'],")
print("                     right_on=['code_departement','annee','trimestre'], how='left')")