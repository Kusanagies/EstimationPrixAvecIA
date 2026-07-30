"""
INSPECTION DU DECILE D1 (biens les moins chers) - COMPRENDRE LE BIAIS
======================================================================
D1 concentre un biais enorme (-37% maisons, -24% apparts) : le modele
surestime fortement les biens les moins chers. Ce script dissseque D1 pour
savoir si ce sont de VRAIES maisons bon marche ou des ARTEFACTS (micro-surfaces,
prix/m2 aberrants, dependances mal classees, ventes hors-marche).

Trois analyses :
  1. COMPOSITION : D1 vs reste (surface, prix/m2, pieces, terrain, ratio marche)
  2. RATIO AU MARCHE : ou se situe D1 par rapport au filtre 0.40-2.50 actuel
  3. SIMULATION DE FILTRE : combien de biens retirees selon la borne basse

Aucune modification du pipeline : pur diagnostic. On decide APRES.
"""

import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

from eia_chargement import (connexions, extraire_ventes, charger_enrichissements,
                            fusion_et_distances, nettoyer_et_feature_base, RAYON)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)

FA = {k: True for k in ['geo_base','surface','pieces','terrain','date','dpe','chauffage',
                        'revenus','densite_pop','dist_transport','dist_monument','dist_hopital',
                        'dist_universite','dist_littoral','voisins','densite','section','potentiel_urbain']}
FA.update({k: False for k in ['chomage','taux_credit','taux_inflation','pib','ipc','stat_socio_eco']})

RACINE_PROJET = Path(__file__).resolve().parents[3]
load_dotenv(RACINE_PROJET / ".env")
CHEMIN_GPKG = "/home/sylvain-huang/Documents/EstimationIA/data/TableGeo2022.gpkg"

try:
    db_pass = os.environ["DB_PASS"]
    moteur_dvf, moteur_enr = connexions(db_pass)
except Exception as e:
    print(f"Erreur connexion : {e}"); sys.exit()

departement = input("Departement (ex: 34) : ").strip()
type_choisi = input("Type (maison/appartement, defaut maison) : ").strip().lower()
type_local = 'Appartement' if type_choisi.startswith('a') else 'Maison'

filtre_dvf = f"departements_code = '{departement}'"
filtre_dpe = f"LEFT(code_insee_ban, 2) = '{departement}'"

print("\nChargement...")
gdf = gpd.read_file(CHEMIN_GPKG)
if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
    gdf = gdf.to_crs(epsg=4326)
maisons = extraire_ventes(moteur_dvf, filtre_dvf)
enr = charger_enrichissements(moteur_enr, FA, filtre_dpe, departement)
donnees = fusion_et_distances(maisons, enr, FA, gdf)
dp, features_base = nettoyer_et_feature_base(donnees, FA)

df = dp[dp['type_local'] == type_local].copy()

# Memes filtres de base que le pipeline (quantiles 0.01/0.99 par type)
pl, pf = df['prix_m2'].quantile(0.01), df['prix_m2'].quantile(0.99)
df = df[(df['prix_m2']>=pl)&(df['prix_m2']<=pf)].copy()

# Ratio au marche communal (identique au filtre de coherence, AVANT filtrage)
stats_com = df.groupby('code_commune')['prix_m2'].agg(['median','size'])
ref_com = df['code_commune'].map(stats_com['median'])
n_com = df['code_commune'].map(stats_com['size'])
ref_com = ref_com.where(n_com >= 10, df['prix_m2'].median())
df['ratio_marche'] = df['prix_m2'] / ref_com

# On applique le filtre de coherence actuel (0.40-2.50) pour etre dans les memes
# conditions que l'analyse de residus
df_f = df[df['ratio_marche'].between(0.40, 2.50)].copy()
df_f['prix_total'] = df_f['prix_m2'] * df_f['surface_reelle_bati']

# Decoupage en deciles de prix total
df_f['decile'] = pd.qcut(df_f['prix_total'], 10, labels=[f"D{i+1}" for i in range(10)])
d1 = df_f[df_f['decile'] == 'D1']
reste = df_f[df_f['decile'] != 'D1']

print("\n" + "="*72)
print(f"COMPOSITION DE D1 (biens les moins chers) - {type_local} dep {departement}")
print(f"D1 = {len(d1)} biens | reste = {len(reste)} biens")
print("="*72)

def resume(sous, col, unite=''):
    s = sous[col].describe(percentiles=[.1,.25,.5,.75,.9])
    return (f"{s['mean']:>10.1f} {s['10%']:>10.1f} {s['25%']:>10.1f} "
            f"{s['50%']:>10.1f} {s['75%']:>10.1f} {s['90%']:>10.1f}")

cols = [('prix_total','prix total EUR'), ('prix_m2','prix/m2 EUR'),
        ('surface_reelle_bati','surface m2'), ('nombre_pieces_principales','nb pieces'),
        ('surface_terrain','terrain m2'), ('ratio_marche','ratio marche')]

print(f"\n{'Variable':<18} {'moyenne':>10} {'p10':>10} {'p25':>10} {'median':>10} {'p75':>10} {'p90':>10}")
print("-"*82)
for col, lbl in cols:
    print(f"[D1]  {lbl:<12} {resume(d1, col)}")
    print(f"[tot] {lbl:<12} {resume(reste, col)}")
    print()

# Part des biens "suspects" dans D1
print("="*72)
print("SIGNAUX D'ATYPICITE DANS D1")
print("="*72)
micro = (d1['surface_reelle_bati'] < 30).mean()*100
une_piece = (d1['nombre_pieces_principales'] <= 1).mean()*100
sans_terrain = (d1['surface_terrain'] == 0).mean()*100
bas_ratio = (d1['ratio_marche'] < 0.50).mean()*100
print(f"  Surface < 30 m2        : {micro:.1f}% de D1  (vs {(reste['surface_reelle_bati']<30).mean()*100:.1f}% du reste)")
print(f"  <= 1 piece             : {une_piece:.1f}% de D1  (vs {(reste['nombre_pieces_principales']<=1).mean()*100:.1f}% du reste)")
print(f"  Sans terrain           : {sans_terrain:.1f}% de D1  (vs {(reste['surface_terrain']==0).mean()*100:.1f}% du reste)")
print(f"  Ratio marche < 0.50    : {bas_ratio:.1f}% de D1  (vs {(reste['ratio_marche']<0.50).mean()*100:.1f}% du reste)")

# Distribution du ratio marche dans D1 (ou tape le filtre ?)
print("\n" + "="*72)
print("RATIO AU MARCHE DANS D1 (le filtre actuel garde 0.40-2.50)")
print("="*72)
for lo, hi in [(0,0.40),(0.40,0.50),(0.50,0.60),(0.60,0.80),(0.80,1.0),(1.0,2.50),(2.50,99)]:
    part = d1['ratio_marche'].between(lo, hi, inclusive='left').mean()*100
    print(f"  ratio [{lo:.2f} - {hi:.2f}[ : {part:>5.1f}% de D1")

# Simulation : combien de biens (tout le df) retires selon la borne basse
print("\n" + "="*72)
print("SIMULATION : impact d'un durcissement de la BORNE BASSE du filtre")
print("(borne haute fixee a 2.50 ; % calcules sur TOUT le type de bien)")
print("="*72)
total = len(df)
print(f"{'borne basse':>12} {'biens retires':>15} {'% retires':>12} {'dont dans D1 actuel':>22}")
print("-"*64)
for borne in [0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
    retires = df[~df['ratio_marche'].between(borne, 2.50)]
    n_ret = len(retires)
    # combien de ces retires sont dans la tranche de prix D1 actuel
    seuil_d1 = df_f['prix_total'].quantile(0.10)
    dans_d1 = (retires['prix_m2']*retires['surface_reelle_bati'] <= seuil_d1).sum()
    print(f"{borne:>12.2f} {n_ret:>15} {n_ret/total*100:>11.1f}% {dans_d1:>22}")

print("\n" + "="*72)
print("LECTURE")
print("="*72)
print("- Si D1 a bcp de micro-surfaces / <=1 piece / ratio<0.50 -> biens atypiques")
print("  a filtrer (le durcissement de la borne basse est justifie).")
print("- Si D1 ressemble au reste (juste moins cher) -> ce sont de vraies maisons")
print("  bon marche, le biais est structurel (regression vers la moyenne), PAS un")
print("  probleme de filtre : durcir ne ferait que jeter des donnees valides.")
print("- La simulation montre le cout (% de donnees perdues) de chaque borne.")