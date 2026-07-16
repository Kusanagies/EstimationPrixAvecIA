"""
TEST D'INTEGRATION DE estimer.py
=================================
Tire des biens reels depuis la base (avec leur vrai prix connu), les passe
dans estimer(), et compare l'estimation au prix reel.

Le tirage applique les MEMES filtres que l'entrainement :
  - mutations mono-local (id_mutation avec un seul local bati)
  - surface entre 9 et 300 m2
  - prix/m2 entre les quantiles 1% et 99% du departement (par type)

ATTENTION : les biens tires font partie des donnees d'ENTRAINEMENT du modele
de production (qui apprend sur tout). Les resultats sont donc optimistes.
Pour des metriques honnetes, se referer au script d'evaluation (split temporel).

Lancer : python3 test.py
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
    """Tire n biens reels aleatoires du departement, avec leur vrai prix.
    Applique les memes filtres que l'entrainement :
    mono-local, surface 9-300 m2, puis quantiles 1%/99% du prix/m2."""
    biens = pd.read_sql(f"""
        SELECT id, id_parcelle,code_commune, latitude, longitude,
               valeur_fonciere,
               (valeur_fonciere / surface_reelle_bati) AS prix_m2_reel,
               surface_reelle_bati, nombre_pieces_principales, surface_terrain
        FROM valeurs_foncieres
        WHERE latitude IS NOT NULL
          AND surface_reelle_bati > 9 AND surface_reelle_bati <= 300
          AND nature_mutation = 'Vente' AND nombre_lots <= 3
          AND nombre_pieces_principales > 0
          AND code_departement = '{departement}'
          AND type_local = '{type_local}'
          AND id_mutation IN (
              SELECT id_mutation FROM valeurs_foncieres
              WHERE surface_reelle_bati > 0
              GROUP BY id_mutation HAVING COUNT(*) = 1);
    """, con=moteur)

    if len(biens) == 0:
        return biens

    # Quantiles purs (comme l'entrainement). Calcules sur le departement :
    # si le modele a ete entraine sur la France entiere, les bornes peuvent
    # differer legerement de celles de l'entrainement, sans consequence.
    plancher = biens['prix_m2_reel'].quantile(0.01)
    plafond = biens['prix_m2_reel'].quantile(0.99)
    biens = biens[(biens['prix_m2_reel'] >= plancher) & (biens['prix_m2_reel'] <= plafond)]

    return biens.sample(n=min(n, len(biens))).reset_index(drop=True)


# ==========================================
# VENTILATION DES HORS-MARGE
# ==========================================
def ventiler_hors_marge(df, type_local):
    hors = df[~df['dans_marge']]
    if len(hors) == 0:
        print("  Aucun bien hors marge, rien a ventiler.")
        return

    print("\n" + "-" * 60)
    print(f"VENTILATION DES HORS-MARGE - {type_local.upper()} ({len(hors)}/{len(df)} biens)")
    print("-" * 60)

    # Sens de l'erreur : sous-estimation ou sur-estimation ?
    nb_sous = int(hors['sous_estime'].sum())
    print(f"\n  Sens de l'erreur : {nb_sous} sous-estimes | {len(hors) - nb_sous} sur-estimes")

    # Par tranche de prix reel au m2 (quartiles du jeu de test complet)
    df = df.copy()
    df['tranche_prix'] = pd.qcut(df['prix_m2_reel'], q=4,
                                 labels=['Q1 (bas)', 'Q2', 'Q3', 'Q4 (haut)'],
                                 duplicates='drop')
    print("\n  Par tranche de prix/m2 reel :")
    stats = df.groupby('tranche_prix', observed=True).agg(
        n=('dans_marge', 'size'),
        pct_hors=('dans_marge', lambda s: 100 * (1 - s.mean())),
        mape=('erreur_rel', lambda s: 100 * s.mean()),
    )
    for tranche, r in stats.iterrows():
        print(f"    {str(tranche):10s} : {r['pct_hors']:5.1f} % hors marge | MAPE {r['mape']:5.1f} % | n={int(r['n'])}")

    # Par tranche de surface
    df['tranche_surf'] = pd.cut(df['surface'], bins=[0, 50, 90, 130, 400],
                                labels=['<50 m2', '50-90', '90-130', '>130'])
    print("\n  Par surface :")
    stats = df.groupby('tranche_surf', observed=True).agg(
        n=('dans_marge', 'size'),
        pct_hors=('dans_marge', lambda s: 100 * (1 - s.mean())),
        mape=('erreur_rel', lambda s: 100 * s.mean()),
    )
    for tranche, r in stats.iterrows():
        if r['n'] > 0:
            print(f"    {str(tranche):10s} : {r['pct_hors']:5.1f} % hors marge | MAPE {r['mape']:5.1f} % | n={int(r['n'])}")

    # Avec / sans terrain
    df['cat_terrain'] = np.where(df['terrain'] > 0, 'avec terrain', 'sans terrain')
    print("\n  Terrain :")
    stats = df.groupby('cat_terrain').agg(
        n=('dans_marge', 'size'),
        pct_hors=('dans_marge', lambda s: 100 * (1 - s.mean())),
        mape=('erreur_rel', lambda s: 100 * s.mean()),
    )
    for cat, r in stats.iterrows():
        print(f"    {cat:12s} : {r['pct_hors']:5.1f} % hors marge | MAPE {r['mape']:5.1f} % | n={int(r['n'])}")

    # Communes les plus touchees
    print("\n  Communes avec le plus de hors-marge :")
    top_communes = hors.groupby('code_commune').agg(
        nb_hors=('erreur_rel', 'size'),
        mape=('erreur_rel', lambda s: 100 * s.mean()),
    ).sort_values('nb_hors', ascending=False).head(8)
    for com, r in top_communes.iterrows():
        total_com = (df['code_commune'] == com).sum()
        print(f"    {com} : {int(r['nb_hors'])}/{total_com} hors marge | MAPE {r['mape']:5.1f} %")

    # Les 5 pires erreurs, pour inspection manuelle
    print("\n  Les 5 pires erreurs (a inspecter dans la base) :")
    pires = hors.nlargest(5, 'erreur_rel')
    for _, r in pires.iterrows():
        sens = "sous-estime" if r['sous_estime'] else "sur-estime"
        print(f"    id {r['id']} | {r['code_commune']} | {r['surface']:.0f} m2 | "
              f"reel {r['prix_total_reel']:>9.0f} vs estime {r['prix_total_estime']:>9.0f} "
              f"({r['erreur_rel']*100:.0f} %, {sens})")

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
            geo_resolu=geo_resolu,
            code_section=str(b['id_parcelle'])[:10]
        )

        terrain = float(b['surface_terrain']) if pd.notna(b['surface_terrain']) else 0
        if type_bien_estimer == 'appartements':
            terrain = 0
        
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
            'code_commune': str(b['code_commune']),
            'surface': surface,
            'terrain': terrain,
            'prix_m2_reel': float(b['prix_m2_reel']),
            'prix_total_reel': prix_total_reel,
            'prix_total_estime': prix_total_estime,
            'total_bas': total_bas,
            'total_haut': total_haut,
            'erreur_rel': erreur_rel,
            'sous_estime': prix_total_estime < prix_total_reel,
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
    print(f"  Dans la marge (<= 20%)      : {nb_ok} tests ({nb_ok / len(df) * 100:.1f} %)")
    print(f"  Hors marge (> 20%)          : {nb_hors} tests ({nb_hors / len(df) * 100:.1f} %)")
    print(f"  Erreur % moyenne (MAPE)     : {df['erreur_rel'].mean() * 100:.1f} %")
    print(f"  Couverture (dans fourchette): {df['dans_fourchette'].mean() * 100:.1f} % (attendu ~95 %)")
    
    ventiler_hors_marge(df, type_local)

    return df

# ==========================================
# LANCEMENT
# ==========================================
print(f"\nLancement : {nb_test} tests maisons + {nb_test} tests appartements (dep {departement})")
print("Rappel : biens issus des donnees d'entrainement -> resultats optimistes.")

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