import json
import os
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
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
# 1. PARSING DU JSON (extraction des mairies)
# ==========================================
CHEMIN = "/home/sylvain-huang/Documents/EstimationIA/data/annuaire_mairie.json"  # adapte

with open(CHEMIN, encoding="utf-8") as f:
    data = json.load(f)

services = data["service"]
print(f"Total services dans le fichier : {len(services)}")

mairies = []
for s in services:
    pivots = s.get("pivot", [])
    est_mairie = any(p.get("type_service_local") == "mairie" for p in pivots)
    if not est_mairie:
        continue

    code_insee = s.get("code_insee_commune", "")

    lat, lon = None, None
    for adr in s.get("adresse", []):
        la = adr.get("latitude", "").strip()
        lo = adr.get("longitude", "").strip()
        if la and lo:
            try:
                lat = float(la)
                lon = float(lo)
                break
            except ValueError:
                continue

    if lat is not None and lon is not None and code_insee:
        mairies.append({
            "code_insee": code_insee,
            "nom": s.get("nom", ""),
            "latitude": lat,
            "longitude": lon,
        })

df_mairies = pd.DataFrame(mairies)
print(f"Mairies avec coordonnees : {len(df_mairies)}")

# Securite : on retire d'eventuels doublons de code_insee (garder la premiere)
df_mairies = df_mairies.drop_duplicates(subset=["code_insee"], keep="first")
print(f"Mairies apres deduplication par commune : {len(df_mairies)}")

# ==========================================
# 2. IMPORT DANS MYSQL
# ==========================================
# to_sql cree la table automatiquement et insere les donnees.
# if_exists='replace' recree la table a chaque execution (ecrase l'ancienne).
df_mairies.to_sql(
    name="infrastructures_mairies",
    con=moteur,
    if_exists="replace",
    index=False,
    chunksize=1000,   # insertion par lots de 1000 pour la performance
)
print("Donnees inserees dans la table 'infrastructures_mairies'.")

# ==========================================
# 3. INDEX + VERIFICATION
# ==========================================
with moteur.connect() as conn:
    # Index sur code_insee pour les jointures rapides avec valeurs_foncieres
    conn.execute(text(
        "CREATE INDEX idx_mairies_insee ON infrastructures_mairies(code_insee);"
    ))
    conn.commit()

    # Verification
    total = conn.execute(text("SELECT COUNT(*) FROM infrastructures_mairies;")).scalar()
    apercu = conn.execute(text("SELECT * FROM infrastructures_mairies LIMIT 5;")).fetchall()

print(f"\nTable creee avec {total} mairies.")
print("Apercu :")
for ligne in apercu:
    print(" ", ligne)