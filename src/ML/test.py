"""
TEST D'ESTIMATION SUR L'ANNEE 2025 (test de generalisation temporelle)
=======================================================================
Tire des biens REELS de l'ANNEE 2025 uniquement et les compare a l'estimation.

Pourquoi 2025 ? Le modele est cense etre entraine sur les annees < 2025
(comme dans correlation_cat.py). Tester sur 2025 = tester sur des biens que
le modele n'a pas vus a l'entrainement -> mesure de generalisation honnete
(et non un test d'integration optimiste sur des biens deja vus).

NB : si le modele de production a ete entraine sur TOUTES les annees (2025
inclus), ce test reste plus realiste qu'un tirage aleatoire, mais n'est pas
100% "hors echantillon". Pour la mesure parfaitement honnete, voir correlation_cat.py.

On passe code_section directement (depuis la base) pour eviter les appels API
IGN lents pendant le test.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Import de la fonction d'estimation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimer import estimer

MARGE = 0.20        # marge de reference (20%)
ANNEE_TEST = 2025   # on ne teste que sur cette annee

# ==========================================
# CONNEXION
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
    # Source des ventes : synthese (adapter en EstimationIA si besoin)
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur.connect().close()
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

departement = input("Departement (ex: 34) : ").strip()
nb_test = int(input("Nombre de tests par type : ").strip())

# ==========================================
# TIRAGE DES BIENS DE TEST (ANNEE 2025 UNIQUEMENT)
# ==========================================
def tirer_biens(type_synthese, n):
    """Tire n biens reels de l'annee 2025 pour un type donne."""
    df = pd.read_sql(f"""
        SELECT id, communes_code AS code_insee, parcelles_code,
               lat, lng, prix_m2, valeur_fonciere,
               surface AS surface_reelle_bati, nb_pieces AS nombre_pieces_principales,
               surface_terrain, adresses_numero, adresses_voie
        FROM synthese
        WHERE departements_code = '{departement}'
          AND typebien = '{type_synthese}'
          AND YEAR(date) = {ANNEE_TEST}          -- <<< FILTRE ANNEE 2025
          AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0
        ORDER BY RAND()
        LIMIT {n};
    """, con=moteur)
    # Conversions numeriques (synthese renvoie du 'object' avec les NULL)
    for c in ['prix_m2','valeur_fonciere','surface_reelle_bati',
              'nombre_pieces_principales','surface_terrain','lat','lng']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['valeur_fonciere','surface_reelle_bati','nombre_pieces_principales','lat','lng'])
    return df

# ==========================================
# TEST D'UN TYPE
# ==========================================
def tester_type(type_synthese, type_modele, n):
    biens = tirer_biens(type_synthese, n)
    if len(biens) == 0:
        print(f"\n{type_modele} : aucun bien 2025 trouve."); return None

    print("\n" + "=" * 108)
    print(f"TESTS - {type_modele.upper()} (annee {ANNEE_TEST}, {len(biens)} biens)")
    print("=" * 108)
    print(f"{'#':>3} {'idDVF':>9} {'Surf':>5} {'Terr':>6} {'PrixReel':>10} {'PrixEstime':>11} "
          f"{'TotalBas':>10} {'TotalHaut':>10} {'Err%':>6}  Statut")
    print("-" * 108)

    resultats = []
    for i, (_, b) in enumerate(biens.iterrows(), 1):
        surface = float(b['surface_reelle_bati'])
        terrain = float(b['surface_terrain']) if pd.notna(b['surface_terrain']) else 0
        if type_modele == 'appartements':
            terrain = 0
        code_section = str(b['parcelles_code'])[:10] if pd.notna(b['parcelles_code']) else None

        # geo_resolu pre-rempli (evite l'appel API de geocodage)
        geo = {'lat': float(b['lat']), 'lon': float(b['lng']),
               'code_insee': str(b['code_insee']), 'label': f"bien {b['id']}"}

        res = estimer(
            adresse=None, surface=surface, type_bien=type_modele,
            nb_pieces=int(b['nombre_pieces_principales']),
            surface_terrain=terrain, annee=ANNEE_TEST, mois=6,
            geo_resolu=geo, code_section=code_section
        )
        if 'erreur' in res:
            continue

        total_reel = float(b['valeur_fonciere'])
        total_est = res['prix_total_estime']
        total_bas, total_haut = res['prix_total_fourchette']
        err = abs(total_est - total_reel) / total_reel
        ok = err <= MARGE
        terr_str = f"{terrain:.0f}" if terrain > 0 else "-"

        print(f"{i:>3} {str(b['id']):>9} {surface:>5.0f} {terr_str:>6} {total_reel:>10.0f} "
              f"{total_est:>11.0f} {total_bas:>10.0f} {total_haut:>10.0f} {err*100:>5.1f}%  {'OK' if ok else 'HORS'}")
        resultats.append({'err': err, 'ok': ok, 'dans': total_bas <= total_reel <= total_haut})

    if not resultats:
        print("Aucune estimation reussie."); return None

    d = pd.DataFrame(resultats)
    nb_ok = int(d['ok'].sum())
    print("-" * 108)
    print(f"RESUME {type_modele.upper()} ({ANNEE_TEST}) : {nb_ok} OK / {len(d)-nb_ok} hors marge sur {len(d)} | "
          f"MAPE {d['err'].mean()*100:.1f} % | couverture {d['dans'].mean()*100:.1f} %")
    return d

# ==========================================
# EXECUTION
# ==========================================
resultats_types = {}
resultats_types['Maison'] = tester_type('maison', 'maisons', nb_test)
resultats_types['Appartement'] = tester_type('appartement', 'appartements', nb_test)

# ==========================================
# BILAN GLOBAL
# ==========================================
print("\n" + "=" * 90)
print(f"BILAN GLOBAL - ANNEE {ANNEE_TEST} (test de generalisation)")
print("=" * 90)
total_ok = 0
total_tests = 0
for tb, d in resultats_types.items():
    if d is None:
        print(f"{tb:12s} : non teste"); continue
    nb_ok = int(d['ok'].sum())
    total_ok += nb_ok; total_tests += len(d)
    print(f"{tb:12s} : {nb_ok} OK / {len(d)-nb_ok} hors marge sur {len(d)} | "
          f"MAPE {d['err'].mean()*100:.1f} % | couverture {d['dans'].mean()*100:.1f} %")
if total_tests > 0:
    print("-" * 90)
    print(f"{'TOTAL':12s} : {total_ok} OK / {total_tests-total_ok} hors marge sur {total_tests} "
          f"({total_ok/total_tests*100:.1f} % dans la marge de {int(MARGE*100)} %)")