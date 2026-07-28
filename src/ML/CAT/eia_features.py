"""
MODULE FEATURE ENGINEERING - EstimationIA
==========================================
Filtre de coherence marche + features spatiales derivees des prix
(voisins, densite, section) avec precautions anti-leakage.

Toutes les features de prix sont en EUR/m2 (comparables entre biens) ;
seule la cible (geree ailleurs) est le prix total.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import KFold


def filtre_coherence_marche(df_bien, borne_basse=0.40, borne_haute=2.50, min_ventes=10):
    """
    Elimine les transactions hors marche : compare chaque bien au prix median
    de SA commune (repli sur la mediane globale si commune < min_ventes ventes)
    et garde ceux dans [borne_basse, borne_haute] x la reference.
    """
    stats_com = df_bien.groupby('code_commune')['prix_m2'].agg(['median', 'size'])
    ref_com = df_bien['code_commune'].map(stats_com['median'])
    n_com = df_bien['code_commune'].map(stats_com['size'])
    ref_com = ref_com.where(n_com >= min_ventes, df_bien['prix_m2'].median())
    ratio = df_bien['prix_m2'] / ref_com
    nb_avant = len(df_bien)
    df_filtre = df_bien[ratio.between(borne_basse, borne_haute)].copy()
    nb_retires = nb_avant - len(df_filtre)
    return df_filtre, nb_retires


def indice_de_marche(df_bien, index_train):
    """
    Calcule le coefficient d'actualisation par annee (sur le train uniquement).
    Ramene les prix passes au niveau de la derniere annee.
    Renvoie le dict coef_marche et les prix actualises du train.
    """
    annees_train = df_bien.loc[index_train, 'annee_vente']
    idx_marche = df_bien.loc[index_train].groupby('annee_vente')['prix_m2'].median()
    ref_marche = idx_marche.loc[idx_marche.index.max()]
    coef_marche = (ref_marche / idx_marche).to_dict()
    prix_train_actu = df_bien.loc[index_train, 'prix_m2'].values * annees_train.map(coef_marche).values
    return coef_marche, prix_train_actu


def ajouter_voisins(X_train, X_test, df_bien, index_train, index_test,
                    prix_train_actu, surface_train, arbre_v, coords_train, RAYON):
    """Ajoute prix_m2_voisins (moyenne ponderee distance+surface, self exclu)."""
    def voisins(dist_rad, idx, sb, self_i=None):
        if self_i is not None:
            keep = idx != self_i
            dist_rad, idx = dist_rad[keep], idx[keep]
        if len(idx) == 0:
            return np.nan
        dm = dist_rad * RAYON
        pv, sv = prix_train_actu[idx], surface_train[idx]
        m = (sv >= sb * 0.6) & (sv <= sb * 1.4)
        if m.sum() >= 3:
            d, p = dm[m], pv[m]
        else:
            d, p = dm, pv
        w = 1.0 / (d + 50.0)
        return np.sum(w * p) / np.sum(w)

    k = min(41, len(coords_train))
    dtr, itr = arbre_v.query(coords_train, k=k)
    sb_tr = df_bien.loc[index_train, 'surface_reelle_bati'].values
    X_train['prix_m2_voisins'] = [voisins(dtr[i], itr[i], sb_tr[i], self_i=i) for i in range(len(itr))]
    coords_test = np.deg2rad(df_bien.loc[index_test, ['latitude', 'longitude']])
    dte, ite = arbre_v.query(coords_test, k=min(40, len(coords_train)))
    sb_te = df_bien.loc[index_test, 'surface_reelle_bati'].values
    X_test['prix_m2_voisins'] = [voisins(dte[i], ite[i], sb_te[i]) for i in range(len(ite))]
    return X_train, X_test


def ajouter_densite(X_train, X_test, df_bien, index_test, arbre_v, coords_train, RAYON):
    """Ajoute densite_ventes_1km (nombre de ventes dans un rayon d'1 km)."""
    rr = 1000 / RAYON
    coords_test_d = np.deg2rad(df_bien.loc[index_test, ['latitude', 'longitude']])
    X_train['densite_ventes_1km'] = arbre_v.query_radius(coords_train, r=rr, count_only=True)
    X_test['densite_ventes_1km'] = arbre_v.query_radius(coords_test_d, r=rr, count_only=True)
    return X_train, X_test


def ajouter_section(X_train, X_test, df_bien, index_train, index_test, coef_marche):
    """
    Ajoute prix_m2_section (mediane de section actualisee) en OUT-OF-FOLD sur
    le train (anti-leakage) + nb_ventes_section.
    """
    df_tr = df_bien.loc[index_train].copy()
    df_tr['prix_actu'] = df_tr['prix_m2'].values * df_tr['annee_vente'].map(coef_marche).values
    med_commune = df_tr.groupby('code_commune')['prix_actu'].median()
    med_globale = df_tr['prix_actu'].median()

    vals_train = pd.Series(np.nan, index=index_train)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for pf, po in kf.split(df_tr):
        ms = df_tr.iloc[pf].groupby('code_section')['prix_actu'].median()
        mc = df_tr.iloc[pf].groupby('code_commune')['prix_actu'].median()
        sous = df_tr.iloc[po]
        v = sous['code_section'].map(ms).fillna(sous['code_commune'].map(mc)).fillna(df_tr.iloc[pf]['prix_actu'].median())
        vals_train.iloc[po] = v.values
    X_train['prix_m2_section'] = vals_train.values

    med_section = df_tr.groupby('code_section')['prix_actu'].median()
    sec_te = df_bien.loc[index_test, 'code_section'].map(med_section)
    com_te = df_bien.loc[index_test, 'code_commune'].map(med_commune)
    X_test['prix_m2_section'] = sec_te.fillna(com_te).fillna(med_globale).values

    nb_vs = df_tr.groupby('code_section').size()
    X_train['nb_ventes_section'] = df_bien.loc[index_train, 'code_section'].map(nb_vs).fillna(0).values
    X_test['nb_ventes_section'] = df_bien.loc[index_test, 'code_section'].map(nb_vs).fillna(0).values
    return X_train, X_test


def construire_features_spatiales(X_train, X_test, df_bien, index_train, index_test, FA, RAYON):
    """
    Orchestre l'ajout des features spatiales selon les choix FA.
    Renvoie X_train, X_test enrichis et la liste des features ajoutees.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    ajoutees = []

    coords_train = np.deg2rad(df_bien.loc[index_train, ['latitude', 'longitude']])
    arbre_v = BallTree(coords_train, metric='haversine')
    coef_marche, prix_train_actu = indice_de_marche(df_bien, index_train)
    surface_train = df_bien.loc[index_train, 'surface_reelle_bati'].values

    if FA['voisins']:
        X_train, X_test = ajouter_voisins(X_train, X_test, df_bien, index_train, index_test,
                                          prix_train_actu, surface_train, arbre_v, coords_train, RAYON)
        ajoutees += ['prix_m2_voisins']
    if FA['densite']:
        X_train, X_test = ajouter_densite(X_train, X_test, df_bien, index_test, arbre_v, coords_train, RAYON)
        ajoutees += ['densite_ventes_1km']
    if FA['section']:
        X_train, X_test = ajouter_section(X_train, X_test, df_bien, index_train, index_test, coef_marche)
        ajoutees += ['prix_m2_section', 'nb_ventes_section']

    return X_train, X_test, ajoutees