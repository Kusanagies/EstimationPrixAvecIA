"""
IMPORT DES FICHIERS .SQL DVF DANS MYSQL
========================================
Execute tous les fichiers data/etalab_dvf/*.sql dans la base EstimationIA,
via le client mysql (le plus rapide et le plus fiable pour de gros dumps).

Les dumps par table contiennent des cles etrangeres : l'ordre alphabetique
ne respecte pas les dependances (une table enfant peut etre importee avant
sa table parente). Le script fonctionne donc en PASSES SUCCESSIVES :
les fichiers en echec sont retentes a la passe suivante, jusqu'a ce que
tout passe ou qu'une passe ne fasse plus aucun progres. L'ordre des
dependances se resout ainsi tout seul.

Prerequis : le client `mysql` doit etre installe (sudo apt install mysql-client).

Lancer : python3 importer_sql.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[2]  # adapte la profondeur si besoin
DOSSIER_SQL = RACINE_PROJET / "data" / "etalab_dvf"
BASE = "EstimationIA"
UTILISATEUR = "root"

# unique_checks=0 accelere les gros imports.
# NE PAS ajouter foreign_key_checks=0 : c'est justement le controle des FK
# qui permet aux passes successives de converger dans le bon ordre.
# NE PAS ajouter autocommit=0 : sans COMMIT final dans les dumps, les
# derniers INSERT seraient annules a la fermeture de la session.
INIT_COMMANDES = "SET unique_checks=0;"

load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
except KeyError:
    print("Erreur : variable DB_PASS introuvable dans le .env")
    sys.exit(1)

# ==========================================
# RECHERCHE DES FICHIERS
# ==========================================
if not DOSSIER_SQL.exists():
    print(f"Erreur : dossier introuvable : {DOSSIER_SQL}")
    sys.exit(1)

fichiers = sorted(DOSSIER_SQL.glob("*.sql"))
if not fichiers:
    print(f"Aucun fichier .sql trouve dans {DOSSIER_SQL}")
    sys.exit(1)

print("-" * 60)
print(f"IMPORT DE {len(fichiers)} FICHIER(S) SQL -> base {BASE}")
print("-" * 60)
for f in fichiers:
    taille_mo = f.stat().st_size / 1024 / 1024
    print(f"  - {f.name} ({taille_mo:.1f} Mo)")

if input("\nLancer l'import ? (o/N) : ").strip().lower() != 'o':
    print("Import annule.")
    sys.exit(0)

# ==========================================
# EXECUTION PAR PASSES SUCCESSIVES
# ==========================================
env = os.environ.copy()
env["MYSQL_PWD"] = db_pass  # evite le mot de passe dans la liste des processus

def importer(fichier):
    """Execute un fichier .sql. Retourne (succes, message_erreur)."""
    with open(fichier, "rb") as f:
        resultat = subprocess.run(
            ["mysql",
             "-u", UTILISATEUR,
             "--default-character-set=utf8mb4",
             f"--init-command={INIT_COMMANDES}",
             BASE],
            stdin=f,
            env=env,
            capture_output=True,
        )
    if resultat.returncode == 0:
        return True, ""
    return False, resultat.stderr.decode(errors='replace').strip()

temps_total = time.time()
restants = list(fichiers)
dernieres_erreurs = {}
passe = 0

while restants and passe < len(fichiers):
    passe += 1
    print(f"\n===== PASSE {passe} : {len(restants)} fichier(s) a importer =====")
    echecs_passe = []

    for fichier in restants:
        print(f"  Import de {fichier.name}...", end=" ", flush=True)
        debut = time.time()
        ok, message = importer(fichier)
        duree = time.time() - debut
        if ok:
            print(f"OK ({duree:.1f} s)")
        else:
            print(f"echec ({duree:.1f} s) -> retente a la passe suivante")
            echecs_passe.append(fichier)
            dernieres_erreurs[fichier.name] = message

    if len(echecs_passe) == len(restants):
        print("\nAucun progres sur cette passe : arret.")
        restants = echecs_passe
        break
    restants = echecs_passe

# ==========================================
# BILAN
# ==========================================
print("\n" + "=" * 60)
nb_ok = len(fichiers) - len(restants)
print(f"TERMINE en {time.time() - temps_total:.1f} s : "
      f"{nb_ok}/{len(fichiers)} importe(s) en {passe} passe(s).")
if restants:
    print("\nFichiers definitivement en echec :")
    for f in restants:
        print(f"  - {f.name}")
        print(f"      {dernieres_erreurs.get(f.name, '?')}")
    sys.exit(1)