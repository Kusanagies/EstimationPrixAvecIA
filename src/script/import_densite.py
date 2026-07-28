"""
IMPORT DE LA DENSITE DE POPULATION PAR COMMUNE
===============================================
Source : data.gouv.fr - "Densite de population" (INSEE).
  https://www.data.gouv.fr/datasets/densite-de-population-1
Fichier CSV avec colonnes :
  annee, code_com, nom_territoire, valeur, numerateur, denominateur
  - valeur       = densite (habitants / km2)  <-- la feature qui nous interesse
  - numerateur   = population de la commune
  - denominateur = superficie (km2)

Cree/remplit la table `densite_population` dans la base EstimationIA.
Cle de jointure : code_commune (5 caracteres, zero initial preserve).

Lancer : python3 import_densite_population.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# --- Connexion ---
RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN = Path("/home/sylvain-huang/Documents/EstimationIA/data/densite-population-com.csv")  # adapte si besoin

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except Exception:
    print("Erreur de connexion a la base.")
    sys.exit()

if not CHEMIN.exists():
    print(f"Fichier introuvable : {CHEMIN}")
    print("Place le CSV a cet emplacement ou adapte la variable CHEMIN.")
    sys.exit()

# --- Lecture du CSV ---
print(f"Lecture de {CHEMIN.name}...")
df = pd.read_csv(CHEMIN, dtype={'code_com': str})  # code_com en str pour garder le zero initial

# Normalisation des noms de colonnes attendus
colonnes_attendues = {'annee', 'code_com', 'valeur', 'numerateur', 'denominateur'}
manquantes = colonnes_attendues - set(df.columns)
if manquantes:
    print(f"Colonnes manquantes dans le CSV : {manquantes}")
    print(f"Colonnes trouvees : {list(df.columns)}")
    sys.exit()

# --- Nettoyage ---
df = df.rename(columns={
    'code_com': 'code_commune',
    'valeur': 'densite_population',      # habitants / km2
    'numerateur': 'population',
    'denominateur': 'superficie_km2',
})

# Code commune sur 5 caracteres (zero initial preserve : ex '01001')
df['code_commune'] = df['code_commune'].str.strip().str.zfill(5)

# Conversions numeriques
for col in ['densite_population', 'population', 'superficie_km2']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['annee'] = pd.to_numeric(df['annee'], errors='coerce').astype('Int64')

# On garde les colonnes utiles + une ligne par (commune, annee)
df = df[['annee', 'code_commune', 'densite_population', 'population', 'superficie_km2']]
df = df.dropna(subset=['code_commune', 'densite_population'])
df = df.drop_duplicates(subset=['annee', 'code_commune'])

print(f"  {len(df):,} lignes (commune x annee) pretes.")
print(f"  Annees presentes : {sorted(df['annee'].dropna().unique().tolist())}")
print(f"  Exemple : {df.iloc[0].to_dict()}")

# --- Creation de la table + insertion ---
print("Creation de la table `densite_population`...")
with moteur.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS densite_population;"))
    conn.execute(text("""
        CREATE TABLE densite_population (
            annee INT,
            code_commune VARCHAR(5),
            densite_population DOUBLE,
            population DOUBLE,
            superficie_km2 DOUBLE,
            INDEX idx_code_commune (code_commune),
            INDEX idx_annee (annee)
        );
    """))

df.to_sql('densite_population', con=moteur, if_exists='append', index=False, chunksize=5000)

# --- Verification ---
with moteur.connect() as conn:
    n = conn.execute(text("SELECT COUNT(*) FROM densite_population;")).scalar()
    apercu = pd.read_sql("SELECT * FROM densite_population LIMIT 5;", con=conn)
print(f"\nImport termine : {n:,} lignes dans `densite_population`.")
print(apercu.to_string(index=False))
print("\nJointure type : ... LEFT JOIN densite_population d ON d.code_commune = <ta_table>.code_commune")
print("(si plusieurs annees, filtrer sur l'annee la plus recente ou joindre par annee)")