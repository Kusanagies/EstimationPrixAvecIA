"""
IMPORT DE LA PYRAMIDE DES AGES PAR DEPARTEMENT (INSEE)
=======================================================
Fichier INSEE multi-feuilles : une feuille par annee (1975-2016).
Structure par feuille :
  - lignes 0-4 : titres et en-tetes
  - donnees a partir de la ligne 5
  - colonnes : [code_dep, nom_dep, puis blocs Ensemble/Hommes/Femmes
    avec 5 tranches d'age + Total chacun]

On extrait le bloc ENSEMBLE et on calcule des features synthetiques :
  - pct_0_19, pct_20_39, pct_40_59, pct_60_74, pct_75_plus (parts par tranche)
  - pct_60_plus (seniors), age proxy
Puis on empile toutes les annees dans UNE table, avec une colonne 'annee'.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --- Connexion ---
RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN = Path("/home/sylvain-huang/Documents/EstimationIA/data/estim-pop-dep-sexe-gca-1975-2025.xlsx")  # adapte

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except Exception:
    print("Erreur de connexion a la base.")
    sys.exit()

# --- Lecture de toutes les feuilles ---
classeur = pd.ExcelFile(CHEMIN)
print(f"{len(classeur.sheet_names)} feuilles (annees) a traiter.")

# Les 5 tranches d'age, dans l'ordre des colonnes du bloc "Ensemble" (col 2 a 6)
# Bloc Ensemble : col 2=0-19, 3=20-39, 4=40-59, 5=60-74, 6=75+, 7=Total
tranches = ['0_19', '20_39', '40_59', '60_74', '75_plus']

lignes_finales = []

for feuille in classeur.sheet_names:
    # L'annee est le nom de la feuille (ex: '2016')
    try:
        annee = int(feuille)
    except ValueError:
        print(f"  Feuille '{feuille}' ignoree (nom non numerique).")
        continue

    # Donnees a partir de la ligne 5 (index 5), sans en-tete
    df = pd.read_excel(CHEMIN, sheet_name=feuille, header=None, skiprows=5)

    # Colonne 0 = code departement, colonne 1 = nom, colonnes 2-6 = tranches Ensemble, 7 = Total
    df = df.rename(columns={0: 'code_departement', 1: 'nom_departement'})

    # On ne garde que les lignes avec un code departement valide (ex: '01', '2A'...)
    df = df[df['code_departement'].notna()].copy()
    df['code_departement'] = df['code_departement'].astype(str).str.strip()
    # On retire d'eventuelles lignes de total / notes (code trop long ou vide)
    df = df[df['code_departement'].str.len() <= 3]
    df = df[df['code_departement'] != '']

    # Extraction du bloc Ensemble (colonnes 2 a 6) + total (colonne 7)
    for i, tr in enumerate(tranches):
        df[f'pop_{tr}'] = pd.to_numeric(df[2 + i], errors='coerce')
    df['pop_total'] = pd.to_numeric(df[7], errors='coerce')

    # Features synthetiques : parts par tranche
    for tr in tranches:
        df[f'pct_{tr}'] = df[f'pop_{tr}'] / df['pop_total'] * 100

    # Indicateur seniors (60 ans et plus)
    df['pct_60_plus'] = (df['pop_60_74'] + df['pop_75_plus']) / df['pop_total'] * 100

    df['annee'] = annee

    colonnes_gardees = ['code_departement', 'nom_departement', 'annee', 'pop_total',
                        'pct_0_19', 'pct_20_39', 'pct_40_59', 'pct_60_74', 'pct_75_plus',
                        'pct_60_plus']
    lignes_finales.append(df[colonnes_gardees])

# --- Concatenation de toutes les annees ---
table = pd.concat(lignes_finales, ignore_index=True)
table = table.dropna(subset=['pop_total'])  # retire lignes vides residuelles
table = table.sort_values(['annee', 'code_departement']).reset_index(drop=True)

print(f"\nTable finale : {len(table)} lignes ({table['annee'].nunique()} annees x ~{table['code_departement'].nunique()} departements)")
print("\nApercu :")
print(table.head(10).to_string(index=False))
print("\nAnnees couvertes :", sorted(table['annee'].unique())[:5], "...", sorted(table['annee'].unique())[-3:])

# --- Import en base ---
table.to_sql('demographie_ages_dep', con=moteur, if_exists='replace', index=False)
print(f"\nTable 'demographie_ages_dep' creee ({len(table)} lignes).")
print("Pense a ajouter un index : CREATE INDEX idx_dep_annee ON demographie_ages_dep(code_departement, annee);")