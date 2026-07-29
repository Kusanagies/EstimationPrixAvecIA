"""
IMPORT DE L'INDICE DES PRIX A LA CONSOMMATION (IPC)
====================================================
Source : INSEE - Indice des prix a la consommation (base 100).
Fichier IndicePrixConso2025.csv, format :
    Periode,<horodatage en 1ere ligne>
    1990-01,68.09
    1990-02,68.23
    ...
(colonne 1 = mois 'AAAA-MM', colonne 2 = valeur de l'indice)

Cree la table `indice_prix_conso` dans EstimationIA, avec annee + mois + indice.
Jointure future : par (annee, mois) sur les ventes.

Lancer : python3 import_ipc.py
"""

import os, sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN = Path("/home/sylvain-huang/Documents/EstimationIA/data/IndicePrixConso2025.csv")  # adapte si besoin

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except Exception:
    print("Erreur de connexion a la base."); sys.exit()

if not CHEMIN.exists():
    print(f"Fichier introuvable : {CHEMIN}"); sys.exit()

print(f"Lecture de {CHEMIN.name}...")
# La 1ere ligne a un en-tete particulier ('Periode', horodatage). On lit en donnant
# des noms de colonnes explicites et on ignore les lignes non conformes.
df = pd.read_csv(CHEMIN, header=0, names=['periode', 'ipc'], dtype=str)

# Ne garder que les lignes au format AAAA-MM (les vraies donnees)
df = df[df['periode'].str.match(r'^\d{4}-\d{2}$', na=False)].copy()
df['annee'] = df['periode'].str[:4].astype(int)
df['mois'] = df['periode'].str[5:7].astype(int)
df['indice_prix_conso'] = pd.to_numeric(df['ipc'].str.replace(',', '.'), errors='coerce')
df = df.dropna(subset=['indice_prix_conso'])[['annee', 'mois', 'indice_prix_conso']]
df = df.drop_duplicates(subset=['annee', 'mois']).sort_values(['annee','mois'])

print(f"  {len(df):,} mois d'indice ({df['annee'].min()}-{df['annee'].max()}).")
print(f"  Exemple : {df.iloc[0].to_dict()} ... {df.iloc[-1].to_dict()}")

print("Creation de la table `indice_prix_conso`...")
with moteur.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS indice_prix_conso;"))
    conn.execute(text("""
        CREATE TABLE indice_prix_conso (
            annee INT, mois INT, indice_prix_conso DOUBLE,
            INDEX idx_annee_mois (annee, mois)
        );
    """))
df.to_sql('indice_prix_conso', con=moteur, if_exists='append', index=False)

with moteur.connect() as conn:
    n = conn.execute(text("SELECT COUNT(*) FROM indice_prix_conso;")).scalar()
print(f"\nImport termine : {n:,} lignes dans `indice_prix_conso`.")