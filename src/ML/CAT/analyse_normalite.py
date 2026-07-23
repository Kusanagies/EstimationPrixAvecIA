"""
ANALYSE DE NORMALITE DES ERREURS LOG (RMSLE)
=============================================
Verifie si les erreurs du modele, mesurees dans l'espace logarithmique
(les residus log = log(predit) - log(reel), ce que le RMSLE quantifie),
suivent une loi normale centree sur 0.

Si oui : le RMSLE s'interprete comme l'ECART-TYPE des erreurs, et la regle
empirique 68-95-99.7 s'applique :
  - ~68 % des biens ont une erreur log dans +/- 1 x RMSLE
  - ~95 % des biens ont une erreur log dans +/- 2 x RMSLE
  - ~99.7 % dans +/- 3 x RMSLE

Produit trois diagnostics :
  1. Histogramme des residus log + courbe normale theorique superposee
  2. QQ-plot (test visuel standard de normalite)
  3. Verification empirique chiffree de la regle 68-95-99.7

Usage (dans correlation_cat.py ou correlation_synthese.py, apres prediction) :
    from analyse_normalite import analyser_normalite_log
    analyser_normalite_log(total_reel, total_pred, dossier_graphes, nom_zone, type_bien)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def analyser_normalite_log(total_reel, total_pred, dossier_graphes, nom_zone, type_bien):
    """
    total_reel : array des vrais prix (total, euros)
    total_pred : array des prix predits (total, euros)
    dossier_graphes : Path ou sauvegarder les figures
    nom_zone, type_bien : pour nommer les fichiers et titres
    """
    total_reel = np.asarray(total_reel, dtype=float)
    total_pred = np.asarray(total_pred, dtype=float)

    # Residus dans l'espace LOG (coeur du RMSLE)
    # log1p pour la securite si une valeur est proche de 0
    residus_log = np.log1p(total_pred) - np.log1p(total_reel)

    # Statistiques de base
    moyenne = np.mean(residus_log)      # biais (idealement ~0)
    ecart_type = np.std(residus_log)    # = RMSLE si moyenne ~ 0
    rmsle = np.sqrt(np.mean(residus_log ** 2))  # RMSLE exact

    base = f"{nom_zone.replace(' ', '_')}_{type_bien}"

    # =========================================================
    # 1. HISTOGRAMME + COURBE NORMALE THEORIQUE
    # =========================================================
    plt.figure(figsize=(10, 6))
    n, bins, _ = plt.hist(residus_log, bins=60, density=True, alpha=0.6,
                          color='steelblue', edgecolor='black', label='Erreurs log observees')
    # Courbe normale theorique (memes moyenne et ecart-type)
    x = np.linspace(residus_log.min(), residus_log.max(), 300)
    courbe = stats.norm.pdf(x, moyenne, ecart_type)
    plt.plot(x, courbe, 'r-', linewidth=2, label=f'Loi normale theorique\n(moy={moyenne:.3f}, sigma={ecart_type:.3f})')
    plt.axvline(0, color='green', linestyle='--', label='Erreur nulle (ideal)')
    plt.axvline(moyenne, color='orange', linestyle=':', label=f'Biais moyen = {moyenne:.3f}')
    plt.xlabel("Erreur dans l'espace log : log(predit) - log(reel)")
    plt.ylabel("Densite")
    plt.title(f"Distribution des erreurs log - {nom_zone} ({type_bien})")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"normalite_hist_{base}.png", dpi=150)
    plt.close()

    # =========================================================
    # 2. QQ-PLOT (quantile-quantile contre la loi normale)
    # =========================================================
    plt.figure(figsize=(8, 8))
    stats.probplot(residus_log, dist="norm", plot=plt)
    plt.title(f"QQ-plot des erreurs log - {nom_zone} ({type_bien})\n"
              "(points sur la droite = erreurs normales)")
    plt.xlabel("Quantiles theoriques (loi normale)")
    plt.ylabel("Quantiles observes (erreurs log)")
    plt.tight_layout()
    plt.savefig(dossier_graphes / f"normalite_qqplot_{base}.png", dpi=150)
    plt.close()

    # =========================================================
    # 3. VERIFICATION EMPIRIQUE DE LA REGLE 68-95-99.7
    # =========================================================
    # On centre les residus (on retire le biais) pour tester la dispersion
    residus_centres = residus_log - moyenne
    dans_1s = np.mean(np.abs(residus_centres) <= 1 * ecart_type) * 100
    dans_2s = np.mean(np.abs(residus_centres) <= 2 * ecart_type) * 100
    dans_3s = np.mean(np.abs(residus_centres) <= 3 * ecart_type) * 100

    # Test de normalite formel (Shapiro sur echantillon si grande taille)
    n_test = min(5000, len(residus_log))  # Shapiro limite a ~5000 points
    echantillon = np.random.choice(residus_log, n_test, replace=False) if len(residus_log) > n_test else residus_log
    try:
        stat_shapiro, p_shapiro = stats.shapiro(echantillon)
    except Exception:
        stat_shapiro, p_shapiro = np.nan, np.nan

    # Asymetrie (skewness) et aplatissement (kurtosis)
    skew = stats.skew(residus_log)
    kurt = stats.kurtosis(residus_log)  # 0 = normale (Fisher)

    print("\n" + "=" * 60)
    print(f"ANALYSE DE NORMALITE DES ERREURS LOG - {type_bien.upper()}")
    print("=" * 60)
    print(f"  RMSLE                    : {rmsle:.4f}")
    print(f"  Biais moyen (log)        : {moyenne:+.4f}  (ideal : 0)")
    print(f"  Ecart-type (log)         : {ecart_type:.4f}")
    print("\n  --- Regle empirique 68-95-99.7 ---")
    print(f"  Dans +/- 1 sigma : {dans_1s:5.1f} %  (theorique normale : 68.3 %)")
    print(f"  Dans +/- 2 sigma : {dans_2s:5.1f} %  (theorique normale : 95.4 %)")
    print(f"  Dans +/- 3 sigma : {dans_3s:5.1f} %  (theorique normale : 99.7 %)")
    print("\n  --- Forme de la distribution ---")
    print(f"  Asymetrie (skewness)     : {skew:+.3f}  (0 = symetrique)")
    print(f"  Aplatissement (kurtosis) : {kurt:+.3f}  (0 = normale ; >0 = queues epaisses)")
    if not np.isnan(p_shapiro):
        print(f"  Test de Shapiro-Wilk     : p = {p_shapiro:.4f}", end="  ")
        print("(p > 0.05 : compatible normale)" if p_shapiro > 0.05 else "(p < 0.05 : s'ecarte de la normale)")
    print("\n  --- Interpretation ---")
    if abs(dans_1s - 68.3) < 5 and abs(dans_2s - 95.4) < 3:
        print("  -> Les erreurs log suivent BIEN la regle 68-95 : distribution")
        print("     quasi-normale. Le RMSLE s'interprete comme l'ecart-type,")
        print("     et les intervalles +/- 2 RMSLE couvrent ~95 % des biens.")
    else:
        print("  -> Les erreurs s'ECARTENT de la normale.")
        if kurt > 1:
            print("     Kurtosis eleve : queues epaisses (erreurs extremes plus")
            print("     frequentes qu'une normale). Probablement du aux ventes")
            print("     aberrantes / hors marche. La regle 68-95 est approximative.")
        if abs(skew) > 0.5:
            sens = "sur-estimations" if skew < 0 else "sous-estimations"
            print(f"     Asymetrie : distribution penchee vers les {sens}.")
    print("=" * 60)

    return {
        'rmsle': rmsle, 'biais': moyenne, 'ecart_type': ecart_type,
        'dans_1s': dans_1s, 'dans_2s': dans_2s, 'dans_3s': dans_3s,
        'skewness': skew, 'kurtosis': kurt, 'p_shapiro': p_shapiro
    }


# =========================================================
# DEMO autonome (donnees simulees)
# =========================================================
if __name__ == "__main__":
    from pathlib import Path
    print("DEMO : simulation de donnees pour illustrer.")

    dossier = Path("/tmp"); np.random.seed(42)
    n = 5000
    # Cas 1 : erreurs bien normales
    reel = np.random.lognormal(12, 0.5, n)
    bruit_normal = np.random.normal(0, 0.20, n)   # erreurs log normales, sigma=0.20
    pred_normal = reel * np.exp(bruit_normal)
    print("\n>>> CAS 1 : erreurs normales (attendu : regle 68-95 respectee)")
    analyser_normalite_log(reel, pred_normal, dossier, "Demo_Normale", "test")

    # Cas 2 : erreurs avec aberrations (queues epaisses)
    bruit_epais = np.random.normal(0, 0.15, n)
    idx_aberrant = np.random.choice(n, n // 50, replace=False)  # 2% d'aberrations
    bruit_epais[idx_aberrant] += np.random.normal(0, 1.5, len(idx_aberrant))
    pred_epais = reel * np.exp(bruit_epais)
    print("\n>>> CAS 2 : erreurs avec aberrations (attendu : queues epaisses)")
    analyser_normalite_log(reel, pred_epais, dossier, "Demo_Aberrante", "test")