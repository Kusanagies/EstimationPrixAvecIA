"""
TEST D'INTEGRATION DE estimer.py
=================================
Tire des biens reels depuis la base (avec leur vrai prix connu), les passe
dans estimer(), et compare l'estimation au prix reel.

Applique le meme nombre de tests aux maisons et aux appartements.
Affiche chaque test (prix reel/estime total + fourchette) et le decompte
des tests dans la marge de 20% vs au-dela.

Lancer : python3 test_estimations.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

# On importe la fonction d'estimation. Adapte le nom si besoin :
#   from estimer_catboost import estimer
from estimer import estimer

# ==========================================
# CONNEXION
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[2]  # adapte la profondeur
load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
    moteur.connect().close()
except Exception:
    print("Erreur de connexion a la base.")
    sys.exit()

# ==========================================
# SAISIE UTILISATEUR
# ==========================================
departement = input("Departement a tester (ex: 34) : ").strip()
nb_test = int(input("Nombre de tests par type de bien (ex: 20) : ").strip())
MARGE = 0.20  # marge d'erreur de reference (20%)

# ==========================================
# TIRAGE DES BIENS REELS
# ==========================================
def tirer_biens(type_local, n):
    """Tire n biens reels aleatoires du departement, avec leur vrai prix."""
    requete = f"""
        SELECT id, code_commune, latitude, longitude,
               valeur_fonciere,
               (valeur_fonciere / surface_reelle_bati) AS prix_m2_reel,
               surface_reelle_bati, nombre_pieces_principales, surface_terrain
        FROM valeurs_foncieres
        WHERE latitude IS NOT NULL AND surface_reelle_bati > 9
          AND nature_mutation = 'Vente' AND nombre_lots <= 3
          AND nombre_pieces_principales > 0
          AND code_departement = '{departement}'
          AND type_local = '{type_local}'
          AND (valeur_fonciere / surface_reelle_bati) BETWEEN 800 AND 15000
        ORDER BY RAND()
        LIMIT {n};
    """
    return pd.read_sql(requete, con=moteur)

# ==========================================
# EXECUTION DES TESTS POUR UN TYPE
# ==========================================
def tester_type(type_local, type_bien_estimer, n):
    biens = tirer_biens(type_local, n)
    if len(biens) == 0:
        print(f"Aucun bien trouve pour {type_local} dans le {departement}.")
        return None

    print("\n" + "=" * 90)
    print(f"TESTS - {type_local.upper()}")
    print("=" * 90)
    # En-tete du tableau
    print(f"{'#':>3} {'idDVF':>9} {'Surf':>5} {'Terr':>6} {'PrixReel':>10} {'PrixEstime':>11} "
          f"{'TotalBas':>10} {'TotalHaut':>10} {'Err%':>6}  Statut")
    print("-" * 108)

    resultats = []
    for i, (_, b) in enumerate(biens.iterrows(), 1):
        geo_resolu = {
            'lat': float(b['latitude']),
            'lon': float(b['longitude']),
            'code_insee': str(b['code_commune']),
            'label': f"bien test {type_local}",
        }
        surface = float(b['surface_reelle_bati'])
        terrain = float(b['surface_terrain']) if pd.notna(b['surface_terrain']) else 0
        nb_pieces = int(b['nombre_pieces_principales'])

        res = estimer(
            adresse="(test)", surface=surface, type_bien=type_bien_estimer,
            nb_pieces=nb_pieces, surface_terrain=terrain,
            geo_resolu=geo_resolu
        )

        if 'erreur' in res or 'suggestions' in res:
            print(f"{i:>3}  -- estimation impossible --")
            continue

        # Prix reels et predits (TOTAL)
        prix_total_reel = float(b['valeur_fonciere'])
        prix_total_estime = res['prix_total_estime']
        total_bas, total_haut = res['prix_total_fourchette']

        # Erreur relative sur le prix total
        erreur_rel = abs(prix_total_estime - prix_total_reel) / prix_total_reel
        dans_marge = erreur_rel <= MARGE
        statut = "OK (<=20%)" if dans_marge else "HORS (>20%)"

        terr_affiche = f"{terrain:.0f}" if terrain > 0 else "-"
        print(f"{i:>3} {str(b['id']):>9} {surface:>5.0f} {terr_affiche:>6} {prix_total_reel:>10.0f} {prix_total_estime:>11.0f} "
              f"{total_bas:>10.0f} {total_haut:>10.0f} {erreur_rel * 100:>5.1f}%  {statut}")

        resultats.append({
            'id': b['id'],
            'prix_total_reel': prix_total_reel,
            'prix_total_estime': prix_total_estime,
            'total_bas': total_bas,
            'total_haut': total_haut,
            'erreur_rel': erreur_rel,
            'dans_marge': dans_marge,
            'dans_fourchette': total_bas <= prix_total_reel <= total_haut,
        })

    if not resultats:
        print(f"Aucune estimation valide pour {type_local}.")
        return None

    df = pd.DataFrame(resultats)
    nb_ok = int(df['dans_marge'].sum())
    nb_hors = len(df) - nb_ok

    print("-" * 108)
    print(f"RESUME {type_local.upper()} ({len(df)} tests) :")
    print(f"  Dans la marge (<= 20%)     : {nb_ok} tests ({nb_ok / len(df) * 100:.1f} %)")
    print(f"  Hors marge (> 20%)         : {nb_hors} tests ({nb_hors / len(df) * 100:.1f} %)")
    print(f"  Erreur % moyenne (MAPE)    : {df['erreur_rel'].mean() * 100:.1f} %")
    print(f"  Couverture (dans fourchette): {df['dans_fourchette'].mean() * 100:.1f} %")

    return df

# ==========================================
# LANCEMENT
# ==========================================
print(f"\nLancement : {nb_test} tests maisons + {nb_test} tests appartements (dep {departement})")

df_maisons = tester_type('Maison', 'maisons', nb_test)
df_apparts = tester_type('Appartement', 'appartements', nb_test)

# ==========================================
# BILAN GLOBAL
# ==========================================
print("\n" + "=" * 90)
print("BILAN GLOBAL")
print("=" * 90)
for nom, df in [('Maisons', df_maisons), ('Appartements', df_apparts)]:
    if df is not None:
        nb_ok = int(df['dans_marge'].sum())
        nb_hors = len(df) - nb_ok
        print(f"{nom:15s} : {nb_ok} OK / {nb_hors} hors marge sur {len(df)} | "
              f"MAPE {df['erreur_rel'].mean() * 100:>4.1f} % | "
              f"couverture {df['dans_fourchette'].mean() * 100:>4.1f} %")