import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 1. CONNEXION A LA BASE DE DONNEES
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
except KeyError:
    print("Erreur : variable DB_PASS introuvable dans le fichier .env")
    sys.exit()

# ==========================================
# 2. LECTURE ET FILTRAGE DU CSV
# ==========================================
print("Lecture du fichier INSEE Filosofi...")
chemin_csv = "/home/sylvain-huang/Documents/EstimationIA/data/revenu-france.csv"

# Lecture initiale de toutes les colonnes en texte pour securiser le nettoyage
df_brut = pd.read_csv(chemin_csv, sep=";", dtype=str, encoding="utf-8")

# Dictionnaire de selection et de renommage propre
mapping_colonnes = {
    "Code géographique": "code_commune",
    "[DISP] Nbre de ménages fiscaux": "nb_menages_fiscaux",
    "[DISP] Médiane (€)": "median_revenu_disponible",
    "[DISP] Iice de Gini": "indice_gini",
    "[DISP] Part des revenus du patrimoine et autres revenus (%)": "pct_revenu_patrimoine",
    "[DISP] dont part des minima sociaux (%)": "pct_minima_sociaux",
    "[DISP] Part des impôts (%)": "pct_impots"
}

# Filtrage des colonnes
df_filtre = df_brut[list(mapping_colonnes.keys())].rename(columns=mapping_colonnes)

# ==========================================
# 3. NETTOYAGE DES DONNEES NUMERIQUES
# ==========================================
print("Nettoyage et conversion des types de donnees...")

# Nettoyage specifique du code commune (doit faire 5 caracteres, ex: '01001')
df_filtre["code_commune"] = df_filtre["code_commune"].str.strip().str.zfill(5)

# Suppression des lignes sans code commune valide
df_filtre = df_filtre.dropna(subset=["code_commune"])

# Conversion des colonnes numeriques (gestion des espaces de milliers et des virgules)
colonnes_numeriques = [col for col in df_filtre.columns if col != "code_commune"]

for col in colonnes_numeriques:
    df_filtre[col] = df_filtre[col].str.replace(",", ".")
    df_filtre[col] = df_filtre[col].str.replace(r"\s+", "", regex=True)
    df_filtre[col] = pd.to_numeric(df_filtre[col], errors="coerce")

print(f"Dataset pret : {df_filtre.shape[0]} communes filtrees avec {df_filtre.shape[1]} indicateurs clés.")

# ==========================================
# 4. EXPORTATION VERS MYSQL
# ==========================================
print("Insertion de la table 'demographie_communes' dans MySQL...")

df_filtre.to_sql(
    name="demographie_communes",
    con=moteur,
    if_exists="replace",
    index=False
)

print("Table inseree avec succes.")