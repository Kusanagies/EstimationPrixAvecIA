"""
TEST D'ESTIMATION SUR BIENS REELS (test de generalisation)
===========================================================
Tire des biens REELS depuis synthese et les compare a l'estimation d'estimer.
Au demarrage, on choisit le perimetre temporel :
  - OUI (defaut) : uniquement l'annee 2025 -> test de generalisation temporelle
    (biens que le modele n'a pas vus si entraine sur < 2025).
  - NON : TOUTES les annees -> chaque bien est estime avec SA PROPRE annee/mois
    de vente (indispensable pour ne pas fausser l'estimation).

NB : si le modele de production a ete entraine sur TOUTES les annees (2025
inclus), le test 2025 reste plus realiste qu'un tirage aleatoire, mais n'est pas
100% "hors echantillon". Pour la mesure parfaitement honnete, voir correlation_synthese.py.

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
ANNEE_TEST = 2025   # annee utilisee si on limite le test a une seule annee

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
rep = input(f"Tester uniquement sur l'annee {ANNEE_TEST} ? (O/n) : ").strip().lower()
SEULEMENT_2025 = not rep.startswith('n')   # defaut = oui (comportement historique)
if SEULEMENT_2025:
    print(f"  -> Test restreint a l'annee {ANNEE_TEST} (generalisation temporelle).")
    LIBELLE_PERIODE = f"annee {ANNEE_TEST}"
else:
    print("  -> Test sur TOUTES les annees (chaque bien estime avec sa propre annee).")
    LIBELLE_PERIODE = "toutes annees"

# ==========================================
# TIRAGE DES BIENS DE TEST (ANNEE 2025 UNIQUEMENT)
# ==========================================
def tirer_biens(type_synthese, n):
    """Tire n biens reels de l'annee 2025, APRES filtre de coherence marche.
    Le filtre (identique a l'entrainement/evaluation) compare chaque bien au
    prix median de sa commune et exclut les transactions hors marche."""
    # On charge TOUS les biens 2025 du type (pas seulement n) pour pouvoir
    # calculer des medianes communales fiables avant de filtrer et d'echantillonner.
    filtre_annee = f"AND YEAR(date) = {ANNEE_TEST}" if SEULEMENT_2025 else ""
    df = pd.read_sql(f"""
        SELECT id, communes_code AS code_insee, communes_code AS code_commune,
               parcelles_code, lat, lng, prix_m2, valeur_fonciere,
               surface AS surface_reelle_bati, nb_pieces AS nombre_pieces_principales,
               surface_terrain, adresses_numero, adresses_voie,
               YEAR(date) AS annee_vente, MONTH(date) AS mois_vente
        FROM synthese
        WHERE departements_code = '{departement}'
          AND typebien = '{type_synthese}'
          {filtre_annee}
          AND surface > 9 AND prix_m2 > 0 AND nb_pieces > 0;
    """, con=moteur)
    # Conversions numeriques (synthese renvoie du 'object' avec les NULL)
    for c in ['prix_m2','valeur_fonciere','surface_reelle_bati',
              'nombre_pieces_principales','surface_terrain','lat','lng',
              'annee_vente','mois_vente']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['valeur_fonciere','surface_reelle_bati','nombre_pieces_principales',
                           'lat','lng','annee_vente'])

    if len(df) == 0:
        return df

    # --- Filtres identiques a l'entrainement/evaluation ---
    # 1) Aberrations de prix par quantiles (0.01 / 0.99)
    plancher = df['prix_m2'].quantile(0.01)
    plafond = df['prix_m2'].quantile(0.99)
    df = df[(df['prix_m2'] >= plancher) & (df['prix_m2'] <= plafond)].copy()

    # 2) Coherence marche : ratio prix / mediane communale entre 0.40 et 2.50
    stats_com = df.groupby('code_commune')['prix_m2'].agg(['median', 'size'])
    ref_com = df['code_commune'].map(stats_com['median'])
    n_com = df['code_commune'].map(stats_com['size'])
    ref_com = ref_com.where(n_com >= 10, df['prix_m2'].median())
    ratio = df['prix_m2'] / ref_com
    nb_avant = len(df)
    df = df[ratio.between(0.40, 2.50)].copy()
    print(f"  Filtre coherence marche ({type_synthese}) : {nb_avant - len(df)} biens retires "
          f"({(nb_avant - len(df)) / nb_avant * 100:.1f} %)")

    # Echantillonnage final de n biens parmi les biens valides
    return df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

# ==========================================
# TEST D'UN TYPE
# ==========================================
def tester_type(type_synthese, type_modele, n):
    biens = tirer_biens(type_synthese, n)
    if len(biens) == 0:
        print(f"\n{type_modele} : aucun bien trouve ({LIBELLE_PERIODE})."); return None

    print("\n" + "=" * 108)
    print(f"TESTS - {type_modele.upper()} ({LIBELLE_PERIODE}, {len(biens)} biens)")
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

        # Annee/mois REELS du bien : indispensable en mode "toutes annees"
        # (estimer un bien de 2016 avec l'annee 2025 fausserait le resultat).
        annee_bien = int(b['annee_vente'])
        mois_bien = int(b['mois_vente']) if pd.notna(b['mois_vente']) else 6

        res = estimer(
            adresse=None, surface=surface, type_bien=type_modele,
            nb_pieces=int(b['nombre_pieces_principales']),
            surface_terrain=terrain, annee=annee_bien, mois=mois_bien,
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
    print(f"RESUME {type_modele.upper()} ({LIBELLE_PERIODE}) : {nb_ok} OK / {len(d)-nb_ok} hors marge sur {len(d)} | "
          f"MAPE {d['err'].mean()*100:.1f} % | err. mediane {d['err'].median()*100:.1f} % | "
          f"couverture {d['dans'].mean()*100:.1f} %")
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
print(f"BILAN GLOBAL - {LIBELLE_PERIODE.upper()} (test de generalisation)")
print("=" * 90)
total_ok = 0
total_tests = 0
for tb, d in resultats_types.items():
    if d is None:
        print(f"{tb:12s} : non teste"); continue
    nb_ok = int(d['ok'].sum())
    total_ok += nb_ok; total_tests += len(d)
    print(f"{tb:12s} : {nb_ok} OK / {len(d)-nb_ok} hors marge sur {len(d)} | "
          f"MAPE {d['err'].mean()*100:.1f} % | err. mediane {d['err'].median()*100:.1f} % | "
          f"couverture {d['dans'].mean()*100:.1f} %")
if total_tests > 0:
    print("-" * 90)
    print(f"{'TOTAL':12s} : {total_ok} OK / {total_tests-total_ok} hors marge sur {total_tests} "
          f"({total_ok/total_tests*100:.1f} % dans la marge de {int(MARGE*100)} %)")