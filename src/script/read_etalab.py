"""
EXPLORATION DE LA NOUVELLE BASE etalab_dvf
===========================================
Compare la table 'synthese' (pre-agregee) a ta table 'valeurs_foncieres' actuelle,
pour decider si tu peux basculer dessus. Ne modifie rien, affiche seulement.

Repond aux questions cles :
  - Combien de biens dans synthese ? Quelles annees ? Quels types ?
  - synthese ne contient-elle que des ventes ?
  - Les volumes sont-ils comparables a l'ancienne source ?
  - Quelles colonnes sont disponibles (et comment les mapper) ?
"""

import os
import sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")

try:
    db_pass = os.environ["DB_PASS"]
    # Deux connexions : l'ancienne base et la nouvelle
    moteur_ancien = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur_etalab = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur_etalab.connect().close()
except Exception as e:
    print(f"Erreur de connexion : {e}")
    sys.exit()

print("=" * 60)
print("EXPLORATION DE etalab_dvf.synthese")
print("=" * 60)

# --- 1. Structure et volume de synthese ---
print("\n--- 1. Colonnes de synthese ---")
cols = pd.read_sql("SHOW COLUMNS FROM synthese", con=moteur_etalab)
print(cols[['Field', 'Type']].to_string(index=False))

print("\n--- 2. Volume total ---")
n_total = pd.read_sql("SELECT COUNT(*) AS n FROM synthese", con=moteur_etalab)['n'].iloc[0]
print(f"Nombre total de lignes : {n_total:,}")

# --- 3. Repartition par type de bien ---
print("\n--- 3. Repartition par typebien ---")
types = pd.read_sql("SELECT typebien, COUNT(*) AS n FROM synthese GROUP BY typebien", con=moteur_etalab)
print(types.to_string(index=False))

# --- 4. Couverture temporelle ---
print("\n--- 4. Couverture temporelle (par annee) ---")
annees = pd.read_sql("""
    SELECT YEAR(date) AS annee, COUNT(*) AS n
    FROM synthese GROUP BY YEAR(date) ORDER BY annee
""", con=moteur_etalab)
print(annees.to_string(index=False))

# --- 5. Statistiques prix_m2 (pour verifier la coherence) ---
print("\n--- 5. Statistiques du prix_m2 (synthese) ---")
stats = pd.read_sql("""
    SELECT typebien,
           COUNT(*) AS n,
           MIN(prix_m2) AS min_prix,
           AVG(prix_m2) AS moy_prix,
           MAX(prix_m2) AS max_prix,
           SUM(CASE WHEN prix_m2 IS NULL THEN 1 ELSE 0 END) AS nb_null
    FROM synthese GROUP BY typebien
""", con=moteur_etalab)
print(stats.to_string(index=False))

# --- 6. Valeurs manquantes sur les colonnes cles ---
print("\n--- 6. Completude des colonnes cles ---")
completude = pd.read_sql("""
    SELECT
        SUM(CASE WHEN lat IS NULL OR lng IS NULL THEN 1 ELSE 0 END) AS sans_coords,
        SUM(CASE WHEN surface IS NULL OR surface = 0 THEN 1 ELSE 0 END) AS sans_surface,
        SUM(CASE WHEN prix_m2 IS NULL THEN 1 ELSE 0 END) AS sans_prix_m2,
        SUM(CASE WHEN nb_pieces IS NULL OR nb_pieces = 0 THEN 1 ELSE 0 END) AS sans_pieces,
        SUM(CASE WHEN parcelles_code IS NULL THEN 1 ELSE 0 END) AS sans_parcelle
    FROM synthese
""", con=moteur_etalab)
print(completude.to_string(index=False))

# --- 7. Format du parcelles_code (pour extraire la section) ---
print("\n--- 7. Exemples de parcelles_code (pour code_section) ---")
exemples = pd.read_sql("SELECT parcelles_code, communes_code, departements_code FROM synthese LIMIT 5", con=moteur_etalab)
print(exemples.to_string(index=False))
print("  (section cadastrale = 10 premiers caracteres, comme avant ?)")

# --- 8. COMPARAISON avec l'ancienne base (sur un departement) ---
print("\n" + "=" * 60)
print("COMPARAISON avec valeurs_foncieres (ancienne base)")
print("=" * 60)
dep_test = input("\nDepartement pour comparer (ex: 34) : ").strip()

# Ancienne base
try:
    ancien = pd.read_sql(f"""
        SELECT type_local, COUNT(*) AS n
        FROM valeurs_foncieres
        WHERE code_departement = '{dep_test}'
          AND type_local IN ('Maison','Appartement')
          AND nature_mutation = 'Vente' AND nombre_lots <= 3
          AND surface_reelle_bati > 9 AND nombre_pieces_principales > 0
        GROUP BY type_local
    """, con=moteur_ancien)
    print(f"\nAncienne base (valeurs_foncieres) - dep {dep_test}, avec tes filtres :")
    print(ancien.to_string(index=False))
except Exception as e:
    print(f"  Erreur ancienne base : {e}")

# Nouvelle base
try:
    nouveau = pd.read_sql(f"""
        SELECT typebien, COUNT(*) AS n
        FROM synthese
        WHERE departements_code = '{dep_test}'
        GROUP BY typebien
    """, con=moteur_etalab)
    print(f"\nNouvelle base (synthese) - dep {dep_test}, sans filtre :")
    print(nouveau.to_string(index=False))
except Exception as e:
    print(f"  Erreur nouvelle base : {e}")

print("\n" + "=" * 60)
print("CONCLUSIONS A TIRER :")
print("  - synthese couvre-t-elle les memes annees (2021-2025) ?")
print("  - Les volumes sont-ils comparables (a filtres pres) ?")
print("  - prix_m2 est-il deja calcule et coherent (pas de NULL massifs) ?")
print("  - parcelles_code permet-il d'extraire la section (format) ?")
print("  - Manque-t-il des filtres (lots, nature_mutation) a reproduire ?")
print("=" * 60)