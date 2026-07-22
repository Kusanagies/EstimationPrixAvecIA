"""
INVESTIGATION : COMMENT synthese GERE LES VENTES MULTI-BIENS
=============================================================
Ne modifie rien. Analyse comment la table synthese traite les ventes qui
contiennent plusieurs biens (lots), en remontant la chaine relationnelle :
  disposition (la vente + valeur_fonciere totale)
    -> dp (disposition-parcelle : nb_maisons, nb_appartements, nb_dependances)
       -> synthese (ce qui en ressort au final)

Questions auxquelles ce script repond :
  1. Combien de ventes contiennent plusieurs biens ?
  2. Pour une vente multi-biens, combien de lignes apparaissent dans synthese ?
  3. Le prix (valeur_fonciere) est-il duplique, reparti, ou attribue a un seul ?
  4. Le prix_m2 de synthese est-il coherent pour ces cas ?
"""

import os, sys
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")
try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/etalab_dvf")
    moteur.connect().close()
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

print("=" * 70)
print("INVESTIGATION : gestion des ventes multi-biens dans synthese")
print("=" * 70)

# --- 1. Structure : combien de biens par disposition (via dp) ---
print("\n--- 1. Distribution du nombre de biens par disposition ---")
distrib = pd.read_sql("""
    SELECT nb_biens, COUNT(*) AS nb_dispositions
    FROM (
        SELECT disposition_id,
               SUM(nb_maisons + nb_appartements) AS nb_biens
        FROM dp
        GROUP BY disposition_id
    ) t
    GROUP BY nb_biens
    ORDER BY nb_biens
    LIMIT 15;
""", con=moteur)
print(distrib.to_string(index=False))
print("  -> combien de ventes ont 1 bien, 2 biens, 3 biens... ?")

# --- 2. Trouver des dispositions multi-biens pour les inspecter ---
print("\n--- 2. Recherche de ventes multi-biens (>= 2 logements) ---")
multi = pd.read_sql("""
    SELECT dp.disposition_id,
           SUM(dp.nb_maisons) AS tot_maisons,
           SUM(dp.nb_appartements) AS tot_apparts,
           SUM(dp.nb_dependances) AS tot_dependances,
           COUNT(*) AS nb_lignes_dp
    FROM dp
    GROUP BY dp.disposition_id
    HAVING (SUM(dp.nb_maisons) + SUM(dp.nb_appartements)) >= 2
    LIMIT 5;
""", con=moteur)
print(multi.to_string(index=False))

if len(multi) == 0:
    print("  Aucune vente multi-biens trouvee (structure inattendue).")
    sys.exit()

# --- 3. Pour ces dispositions, voir la valeur_fonciere et ce qui est dans synthese ---
print("\n--- 3. Analyse detaillee de 3 ventes multi-biens ---")
for disp_id in multi['disposition_id'].head(3):
    print("\n" + "-" * 60)
    print(f"DISPOSITION {disp_id}")
    print("-" * 60)

    # La disposition (vente) elle-meme
    dispo = pd.read_sql(f"SELECT id, valeur_fonciere, date FROM disposition WHERE id = {disp_id};", con=moteur)
    if len(dispo) > 0:
        print(f"  Vente : valeur_fonciere = {dispo['valeur_fonciere'].iloc[0]} EUR | date = {dispo['date'].iloc[0]}")

    # Les biens de cette vente (via dp)
    biens_dp = pd.read_sql(f"""
        SELECT parcelle_id, nb_maisons, nb_appartements, nb_dependances, surface_terrain
        FROM dp WHERE disposition_id = {disp_id};
    """, con=moteur)
    print(f"  Composition (dp) : {len(biens_dp)} ligne(s)")
    print(biens_dp.to_string(index=False))

    # Ce qui apparait dans synthese pour cette vente
    # (on relie via geo_objet_id ou la date+valeur - selon le schema)
    synth = pd.read_sql(f"""
        SELECT id, valeur_fonciere, typebien, prix_m2, surface, surface_terrain, nb_dependances
        FROM synthese
        WHERE valeur_fonciere = (SELECT valeur_fonciere FROM disposition WHERE id = {disp_id})
          AND date = (SELECT date FROM disposition WHERE id = {disp_id})
        LIMIT 10;
    """, con=moteur)
    print(f"  Dans synthese : {len(synth)} ligne(s) avec cette valeur_fonciere+date")
    if len(synth) > 0:
        print(synth.to_string(index=False))
        # Le prix est-il duplique ou reparti ?
        vf_unique = synth['valeur_fonciere'].nunique()
        print(f"  -> valeur_fonciere identique sur toutes les lignes ? {'OUI (dupliquee)' if vf_unique == 1 else 'NON (repartie)'}")

# --- 4. Statistique globale : ratio lignes synthese / dispositions ---
print("\n" + "=" * 70)
print("--- 4. Ratio global lignes synthese vs dispositions ---")
n_synth = pd.read_sql("SELECT COUNT(*) AS n FROM synthese;", con=moteur)['n'].iloc[0]
n_dispo = pd.read_sql("SELECT COUNT(*) AS n FROM disposition;", con=moteur)['n'].iloc[0]
print(f"  Lignes dans synthese   : {n_synth:,}")
print(f"  Lignes dans disposition: {n_dispo:,}")
print(f"  Ratio                  : {n_synth/n_dispo:.2f} ligne(s) synthese par disposition")
print("\n  INTERPRETATION :")
print("   - Ratio ~1  -> synthese = 1 ligne par vente (le bien principal, prix total)")
print("   - Ratio >1  -> synthese demultiplexe (1 ligne par bien, prix reparti ou duplique)")
print("=" * 70)