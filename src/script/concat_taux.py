"""
CONCATENATION DES FICHIERS DE TAUX MACROECONOMIQUES
====================================================
Fusionne les 6 fichiers CSV (taux d'interet, inflation, credits immo) en une
seule table mensuelle, avec une colonne de taux par fichier.

Format d'entree attendu : separateur ';', colonnes 'date' (fin de mois) et 'Pct'
(virgule decimale francaise).
"""

import pandas as pd
from pathlib import Path

# Dossier ou se trouvent les CSV (adapte le chemin)
DOSSIER = Path("/home/sylvain-huang/Documents/EstimationIA/data/taux")

# Association fichier -> nom de colonne voulu pour son Pct
fichiers = {
    'CreditImmoTauxFixe.csv': 'taux_credit_immo_fixe',
    'CreditImmoTauxVar.csv':  'taux_credit_immo_var',
    'TauxInflation.csv':      'taux_inflation',
    'InteretPME.csv':         'taux_interet_pme',
    'InteretETI.csv':         'taux_interet_eti',
    'InteretGE.csv':          'taux_interet_ge',
}

def charger_taux(chemin, nom_colonne):
    """Charge un fichier de taux et renomme Pct selon le fichier."""
    df = pd.read_csv(
        chemin,
        sep=';',              # separateur point-virgule
        decimal=',',          # virgule decimale francaise -> 0,7 devient 0.7
    )
    # Normalise la date (fin de mois) en periode mensuelle
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['annee'] = df['date'].dt.year
    df['mois'] = df['date'].dt.month
    # Renomme la colonne de taux
    df = df.rename(columns={'Pct': nom_colonne})
    return df[['annee', 'mois', nom_colonne]]

# Fusion progressive sur (annee, mois)
table = None
for fichier, nom_col in fichiers.items():
    chemin = DOSSIER / fichier
    if not chemin.exists():
        print(f"  ATTENTION : {fichier} introuvable, ignore.")
        continue
    df = charger_taux(chemin, nom_col)
    print(f"  {fichier:28s} -> {nom_col:24s} ({len(df)} lignes)")
    if table is None:
        table = df
    else:
        table = pd.merge(table, df, on=['annee', 'mois'], how='outer')

# Tri chronologique
table = table.sort_values(['annee', 'mois']).reset_index(drop=True)

print(f"\nTable finale : {len(table)} lignes (mois), {len(table.columns)} colonnes")
print("\nApercu (premieres lignes) :")
print(table.head(10).to_string(index=False))
print("\nApercu (dernieres lignes) :")
print(table.tail(5).to_string(index=False))

# Sauvegarde
sortie = DOSSIER / "taux_macro_concatenes.csv"
table.to_csv(sortie, sep=';', decimal=',', index=False)
print(f"\nSauvegarde dans : {sortie}")