"""
MODULE METRIQUES - EstimationIA
================================
Calcul et affichage des metriques d'evaluation (prix total).
Toutes les fonctions prennent les tableaux reel/predit et renvoient / affichent
les metriques, sans dependance a l'etat global du script principal.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def calculer_metriques(total_reel, total_pred, total_bas, total_haut):
    """
    Calcule toutes les metriques sur le PRIX TOTAL.
    Renvoie un dict pret a afficher.
    """
    total_reel = np.asarray(total_reel, dtype=float)
    total_pred = np.asarray(total_pred, dtype=float)

    err_rel = np.abs(total_reel - total_pred) / total_reel
    metriques = {
        'mae': mean_absolute_error(total_reel, total_pred),
        'mape': np.mean(err_rel) * 100,
        'err_med': np.median(np.abs(total_reel - total_pred)),
        'rmse': np.sqrt(np.mean((total_reel - total_pred) ** 2)),
        'rmsle': np.sqrt(np.mean((np.log1p(total_pred) - np.log1p(total_reel)) ** 2)),
        'r2': r2_score(total_reel, total_pred),
        'pe10': np.mean(err_rel <= 0.10) * 100,
        'pe20': np.mean(err_rel <= 0.20) * 100,
        'couv': np.mean((total_reel >= total_bas) & (total_reel <= total_haut)) * 100,
        'largeur': np.mean(total_haut - total_bas),
    }
    # NRMSE : RMSE normalise par la moyenne des prix reels (erreur relative globale)
    metriques['nrmse'] = metriques['rmse'] / np.mean(total_reel) * 100
    return metriques


def afficher_rapport(metriques, nb_train, nb_test, type_bien):
    """Affiche le rapport formate d'un type de bien."""
    m = metriques
    print("\n" + "-" * 50)
    print(f"RAPPORT {type_bien.upper()} (source synthese) - PRIX TOTAL")
    print("-" * 50)
    print(f"Apprentissage : {nb_train} | Test : {nb_test}")
    print(f"R2              : {m['r2']*100:.2f} %")
    print(f"MAE             : {m['mae']:,.0f} EUR")
    print(f"MAPE            : {m['mape']:.1f} %")
    print(f"Erreur mediane  : {m['err_med']:,.0f} EUR")
    print(f"RMSE            : {m['rmse']:,.0f} EUR")
    print(f"NRMSE           : {m['nrmse']:.1f} % (RMSE / prix moyen)")
    print(f"RMSLE           : {m['rmsle']:.4f} (erreur relative, espace log)")
    print(f"PE10 / PE20     : {m['pe10']:.1f} % / {m['pe20']:.1f} %")
    print(f"Couverture 95%  : {m['couv']:.1f} %")
    print(f"Largeur moyenne : {m['largeur']:,.0f} EUR")