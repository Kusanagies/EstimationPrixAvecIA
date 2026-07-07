"""
INSPECTION D'UN FICHIER EXCEL MULTI-FEUILLES
=============================================
Ne modifie rien. Affiche juste :
  - la liste des feuilles (onglets)
  - pour chaque feuille : ses dimensions, ses colonnes, et un apercu

A lancer avant d'importer, pour decider de la strategie.
"""

import pandas as pd
from pathlib import Path

# Adapte le chemin vers ton fichier
CHEMIN = Path("/home/sylvain-huang/Documents/EstimationIA/data/estim-pop-dep-sexe-gca-1975-2025.xlsx")

# Ouvre le classeur sans tout charger
classeur = pd.ExcelFile(CHEMIN)

print("=" * 60)
print(f"FICHIER : {CHEMIN.name}")
print("=" * 60)
print(f"\nNombre de feuilles : {len(classeur.sheet_names)}")
print("Noms des feuilles (onglets) :")
for i, nom in enumerate(classeur.sheet_names):
    print(f"  {i}. {nom}")

# Apercu de chaque feuille
for nom in classeur.sheet_names:
    print("\n" + "=" * 60)
    print(f"FEUILLE : {nom}")
    print("=" * 60)
    # Lit les 8 premieres lignes SANS supposer ou est l'en-tete
    apercu = pd.read_excel(CHEMIN, sheet_name=nom, header=None, nrows=8)
    print(f"Dimensions (lignes lues x colonnes) : {apercu.shape}")
    print("Apercu des 8 premieres lignes (brut, sans en-tete) :")
    print(apercu.to_string(max_cols=12))