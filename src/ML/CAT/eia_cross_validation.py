"""
MODULE CROSS-VALIDATION - EstimationIA
=======================================
Cross-validation K-fold RIGOUREUSE sur le train du split 70/30.
A chaque fold, les features derivees des prix (voisins, section, indice de
marche) sont RECALCULEES sur le sous-train du fold -> pas de fuite entre folds
ni vers le test 30% (qui reste a part pour l'evaluation finale).
"""

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor
from eia_features import construire_features_spatiales


def cross_validation_train(df_bien, features_base, FA, RAYON, index_train, n_folds=5):
    """
    CV sur le train (index_train). Renvoie la liste des RMSLE par fold.
    """
    df_train = df_bien.loc[index_train]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    X_all = df_train[features_base]
    y_all = df_train['log_prix_total']
    rmsle_folds = []

    for i_fold, (pos_tr, pos_te) in enumerate(kf.split(X_all), 1):
        idx_tr = X_all.index[pos_tr]
        idx_te = X_all.index[pos_te]
        X_tr = X_all.loc[idx_tr].copy()
        X_te = X_all.loc[idx_te].copy()
        y_tr = y_all.loc[idx_tr]
        y_te = y_all.loc[idx_te]

        # Features spatiales recalculees sur le sous-train du fold
        X_tr, X_te, ajoutees = construire_features_spatiales(
            X_tr, X_te, df_bien, idx_tr, idx_te, FA, RAYON)
        feats = list(dict.fromkeys(list(features_base) + ajoutees))
        X_tr, X_te = X_tr[feats], X_te[feats]

        # Modele median (le RMSLE se mesure sur la mediane)
        X_t, X_v, y_t, y_v = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
        m = CatBoostRegressor(loss_function='Quantile:alpha=0.5', iterations=1000,
                              learning_rate=0.04, depth=8, random_seed=42,
                              early_stopping_rounds=50, verbose=False)
        m.fit(X_t, y_t, eval_set=(X_v, y_v), use_best_model=True)
        total_reel = np.exp(y_te).values
        total_pred = np.exp(m.predict(X_te))
        rmsle_f = np.sqrt(np.mean((np.log1p(total_pred) - np.log1p(total_reel)) ** 2))
        rmsle_folds.append(rmsle_f)
        print(f"    Fold {i_fold}/{n_folds} : RMSLE = {rmsle_f:.4f}")

    return rmsle_folds


def afficher_resultats_cv(rmsle_cv, rmsle_test):
    """Affiche la synthese CV : stabilite + comparaison au test 30%."""
    moy = np.mean(rmsle_cv)
    ec = np.std(rmsle_cv)
    print(f"  RMSLE CV (train) : {moy:.4f} +/- {ec:.4f} "
          f"(min {min(rmsle_cv):.4f}, max {max(rmsle_cv):.4f})")
    print(f"  -> Stabilite : ecart-type {ec:.4f} "
          f"({'stable' if ec < 0.02 else 'variable, a surveiller'})")
    ecart = abs(moy - rmsle_test)
    print(f"  -> CV(train) {moy:.4f} vs test 30% {rmsle_test:.4f} : ecart {ecart:.4f} "
          f"({'coherent (pas de sur-apprentissage)' if ecart < 0.03 else 'divergence, possible sur-apprentissage'})")
    return moy, ec