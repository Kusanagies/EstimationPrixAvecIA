"""
ANALYSE DES ERREURS - script leger
===================================
Recharge resultats.pkl (produit par pipeline.py) et analyse les erreurs
du modele SANS reentrainer. Permet d'identifier quels biens posent probleme.

Usage : python analyse_erreurs.py
        (ou passer le chemin du resultats.pkl en argument)
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# CHARGEMENT DES RESULTATS
# ==========================================
if len(sys.argv) > 1:
    chemin = Path(sys.argv[1])
else:
    # Par defaut, demande la zone pour retrouver le bon dossier
    RACINE = Path(__file__).resolve().parents[2]
    zone = input("Chemin relatif de la zone (ex: 34, 34/34172, FRANCE) : ").strip()
    chemin = RACINE / "out" / zone / "resultats.pkl"

if not chemin.exists():
    print(f"Fichier introuvable : {chemin}")
    print("Lance d'abord pipeline.py pour generer resultats.pkl.")
    sys.exit()

with open(chemin, "rb") as f:
    R = pickle.load(f)

nom_zone = R['nom_zone']
profil = R['profil_test'].copy()
profil['prix_predit'] = R['prix_predits_euros']
profil['prix_reel'] = R['prix_reels_euros']
profil['erreur_abs'] = np.abs(profil['prix_reel'] - profil['prix_predit'])
profil['erreur_rel'] = profil['erreur_abs'] / profil['prix_reel']
profil['surestime'] = profil['prix_predit'] > profil['prix_reel']

print("\n" + "=" * 55)
print(f"ANALYSE DES ERREURS - {nom_zone}")
print("=" * 55)

# ==========================================
# 1. LES 50 PIRES ERREURS
# ==========================================
print("\n--- LES 20 PIRES ERREURS ABSOLUES ---")
pires = profil.sort_values('erreur_abs', ascending=False).head(20)
for _, r in pires.iterrows():
    type_court = str(r['type_local'])[:5]
    print(f"  {r['code_commune']} | {type_court:5} | "
          f"{r['surface_reelle_bati']:5.0f}m2 | terrain {r['surface_terrain']:6.0f} | "
          f"reel {r['prix_reel']:6.0f} vs predit {r['prix_predit']:6.0f} "
          f"(erreur {r['erreur_abs']:5.0f})")

# ==========================================
# 2. PROFIL DES GROSSES ERREURS vs RESTE
# ==========================================
seuil = profil['erreur_abs'].quantile(0.90)  # les 10% pires
grosses = profil[profil['erreur_abs'] >= seuil]
reste = profil[profil['erreur_abs'] < seuil]

print(f"\n--- COMPARAISON : 10% pires erreurs vs reste ---")
print(f"{'Caracteristique':28} | {'Pires 10%':>12} | {'Reste':>12}")
print("-" * 56)
for col, label in [('prix_reel', 'Prix reel moyen'),
                   ('surface_reelle_bati', 'Surface moyenne'),
                   ('surface_terrain', 'Terrain moyen'),
                   ('prix_predit', 'Prix predit moyen')]:
    print(f"{label:28} | {grosses[col].mean():12.0f} | {reste[col].mean():12.0f}")

# Part de maisons dans chaque groupe
pct_maison_grosses = (grosses['type_local'] == 'Maison').mean() * 100
pct_maison_reste = (reste['type_local'] == 'Maison').mean() * 100
print(f"{'% de maisons':28} | {pct_maison_grosses:11.1f}% | {pct_maison_reste:11.1f}%")

# Sens du biais sur les grosses erreurs
pct_sur = grosses['surestime'].mean() * 100
print(f"\nSur les 10% pires : {pct_sur:.0f}% sont des SURESTIMATIONS, "
      f"{100 - pct_sur:.0f}% des sous-estimations.")

# ==========================================
# 3. COMMUNES OU LE MODELE SE TROMPE LE PLUS
# ==========================================
print("\n--- TOP 10 COMMUNES PAR ERREUR MOYENNE (min 20 biens) ---")
par_commune = profil.groupby('code_commune').agg(
    erreur_moyenne=('erreur_abs', 'mean'),
    nb_biens=('erreur_abs', 'size'),
    prix_moyen=('prix_reel', 'mean')
)
par_commune = par_commune[par_commune['nb_biens'] >= 20]
par_commune = par_commune.sort_values('erreur_moyenne', ascending=False).head(10)
for code, r in par_commune.iterrows():
    print(f"  {code} : erreur moyenne {r['erreur_moyenne']:5.0f} EUR/m2 "
          f"({int(r['nb_biens'])} biens, prix moyen {r['prix_moyen']:.0f})")

# ==========================================
# 4. ERREUR PAR TRANCHE DE PRIX
# ==========================================
print("\n--- ERREUR RELATIVE PAR TRANCHE DE PRIX ---")
profil['tranche'] = pd.cut(profil['prix_reel'],
                           bins=[0, 2000, 3500, 5000, 8000, 100000],
                           labels=['<2000', '2000-3500', '3500-5000', '5000-8000', '>8000'])
par_tranche = profil.groupby('tranche', observed=True).agg(
    erreur_rel_moyenne=('erreur_rel', 'mean'),
    nb=('erreur_rel', 'size')
)
for tranche, r in par_tranche.iterrows():
    print(f"  {str(tranche):12} : {r['erreur_rel_moyenne'] * 100:5.1f}% d'erreur "
          f"({int(r['nb'])} biens)")

print("\n" + "=" * 55)
print("Interpretation rapide :")
print("- Si les pires erreurs sont surtout des biens chers -> donnees")
print("  insuffisantes pour le haut de gamme (etat, vue, standing).")
print("- Si une commune ressort -> zone atypique mal capturee.")
print("- Si l'erreur relative explose sur une tranche -> le modele")
print("  maitrise mal ce segment de prix.")
print("=" * 55)