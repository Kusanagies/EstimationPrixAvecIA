"""
PHASE 3 - ESTIMATION D'UN BIEN A PARTIR DE SON ADRESSE
=======================================================
Charge les artefacts produits par entrainer_et_sauvegarder.py.
Gère dynamiquement les flux "maisons" et "appartements".

Renvoie une estimation au m2 et totale, avec un intervalle de confiance a 90%
(modeles quantiles bas / median / haut).
"""

import json
import pickle
from pathlib import Path
import sys
import requests
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import xgboost as xgb

# ==========================================
# CHARGEMENT DES ARTEFACTS (Une seule fois au demarrage)
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_MODELE = RACINE_PROJET / "modele_production"

print("Chargement des modeles en memoire...")

# 1. Chargement des 6 modeles (3 pour maisons, 3 pour appartements)
modeles = {'maisons': {}, 'appartements': {}}
types_biens = ['maisons', 'appartements']

for tb in types_biens:
    for nom in ['bas', 'median', 'haut']:
        fichier_modele = DOSSIER_MODELE / f"modele_{tb}_{nom}.json"
        if fichier_modele.exists():
            m = xgb.XGBRegressor()
            m.load_model(str(fichier_modele))
            modeles[tb][nom] = m

# 2. Chargement des features (identiques pour les deux)
with open(DOSSIER_MODELE / "features.json") as f:
    FEATURES = json.load(f)

# 3. Chargement des deux contextes
contextes = {}
for tb in types_biens:
    fichier_ctx = DOSSIER_MODELE / f"contexte_{tb}.pkl"
    if fichier_ctx.exists():
        with open(fichier_ctx, "rb") as f:
            contextes[tb] = pickle.load(f)

# On utilise le contexte d'un des deux pour les donnees globales partagées (infrastructures)
CTX_REF = contextes['maisons'] if 'maisons' in contextes else contextes['appartements']
RAYON_TERRE = CTX_REF['rayon_terre']

# 4. Reconstruction des arbres spatiaux
def _arbre_ou_none(df):
    if df is not None and len(df) > 0:
        return BallTree(np.deg2rad(df[['latitude', 'longitude']]), metric='haversine')
    return None

# Arbres specifiques au type de bien (voisins)
arbres_voisins = {}
for tb, ctx in contextes.items():
    arbres_voisins[tb] = BallTree(ctx['arbre_voisins_data'], metric='haversine')

# Arbres partages (infrastructures)
arbre_stations = _arbre_ou_none(CTX_REF['stations'])
arbre_monuments = _arbre_ou_none(CTX_REF['monuments'])
arbre_hopitaux = _arbre_ou_none(CTX_REF['hopitaux'])
universites = CTX_REF['universites']
arbre_universites = _arbre_ou_none(universites)

print("Moteur d'estimation pret.")


# ==========================================
# GEOCODAGE (adresse -> lat, lon, code INSEE)
# ==========================================
def geocoder(adresse, limite=5, seuil_confiance=0.6):
    """Utilise l'API publique de la Base Adresse Nationale."""
    try:
        r = requests.get("https://api-adresse.data.gouv.fr/search/",
                         params={"q": adresse, "limit": limite}, timeout=10)
        data = r.json()
    except Exception:
        return {'status': 'erreur', 'message': "Service de geocodage indisponible"}
    
    feats = data.get('features', [])
    if not feats: 
        return {'status': 'erreur', 'message': 'Aucune adresse trouvee'}
    
    def extraire(feat):
        lon, lat = feat['geometry']['coordinates']
        p = feat['properties']
        return {
            'lat': lat, 'lon': lon,
            'code_insee': p.get('citycode'),
            'label': p.get('label', adresse),
            'score': p.get('score', 0)
        }
        
    candidats = [extraire(f) for f in feats]
    meilleur = candidats[0]

    if meilleur['score'] >= seuil_confiance:
        return {'status': 'ok', 'resultat': meilleur}
    
    return {'status': 'suggestions', 'suggestions': candidats}

def recuperer_section(lat, lon):
    """Recupere la parcelle cadastrale a partir de coordonnees via l'API carto IGN."""
    try: 
        geom = f'{{"type":"Point","coordinates":[{lon},{lat}]}}'
        r = requests.get("https://apicarto.ign.fr/api/cadastre/parcelle",
                         params={"geom": geom}, timeout=10)
        data = r.json()
    except Exception:
        return None

    feats = data.get('features', [])
    if not feats: 
        return None
    props = feats[0]['properties']

    code_com = props.get('code_dep', '') + props.get('code_com', '')
    prefixe = props.get('com_abs', '000')
    section = props.get('section', '')

    if not code_com or not section: 
        return None
    
    return (code_com + prefixe + section)[:10]
    
# ==========================================
# DISTANCE A L'AMENITE LA PLUS PROCHE
# ==========================================
def _distance_min(arbre, point_rad):
    if arbre is None:
        return 999999.0
    dist_rad, _ = arbre.query(point_rad, k=1)
    return float(dist_rad[0][0] * RAYON_TERRE)


# ==========================================
# CONSTRUCTION DU VECTEUR DE FEATURES
# ==========================================
def construire_features(lat, lon, code_insee, surface, type_bien,
                        nb_pieces, surface_terrain, annee, mois):
    """Reconstruit toutes les features d'un bien en ciblant le bon contexte."""
    point_rad = np.deg2rad([[lat, lon]])
    
    # On charge le contexte specifique au type de bien
    CTX = contextes[type_bien]
    arbre_v = arbres_voisins[type_bien]
    prix_all = CTX['prix_all']

    # Profil communal
    profil = CTX['profils_communes'].get(code_insee, {})
    med_glob = CTX['medianes_globales']

    def val_commune(col):
        v = profil.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return med_glob.get(col, 0.0)
        return v

    # prix_m2_voisins (specifique au type de bien)
    k = min(15, len(prix_all))
    _, idx = arbre_v.query(point_rad, k=k)
    prix_voisins = float(np.median(prix_all[idx[0]]))

    # densite
    rayon_rad = 1000 / RAYON_TERRE
    densite = int(arbre_v.query_radius(point_rad, r=rayon_rad, count_only=True)[0])

    # prix_m2_section
    code_section = recuperer_section(lat, lon)
    if code_section is not None and code_section in CTX['med_section']:
        prix_section = CTX['med_section'][code_section]
    elif code_insee in CTX['med_section']:
        prix_section = CTX['med_commune'][code_insee]
    else: 
        prix_section = CTX['med_globale']

    # nb_ventes_section
    if code_section is not None and code_section in CTX['nb_ventes_section']:
        nb_ventes_sec = CTX['nb_ventes_section'][code_section]
    else: 
        nb_ventes_sec = densite

    # Volume etudiants
    if arbre_universites is not None:
        _, idx_u = arbre_universites.query(point_rad, k=1)
        vol_etudiants = float(universites.iloc[idx_u[0][0]]['nombre_etudiants'])
    else:
        vol_etudiants = 0.0

    surface_terrain = surface_terrain if surface_terrain else 0

    valeurs = {
        'latitude': lat,
        'longitude': lon,
        'nombre_pieces_principales': nb_pieces,
        'annee_vente': annee,
        'mois_vente': mois,
        'a_terrain': int(surface_terrain > 0),
        'surface_reelle_bati': surface,
        'log_surface': np.log(surface),
        'surface_par_piece': surface / nb_pieces if nb_pieces else surface,
        'surface_terrain': surface_terrain,
        'log_terrain': np.log1p(surface_terrain),
        'volume_etudiants_proche': vol_etudiants,
        'dist_transport_m': _distance_min(arbre_stations, point_rad),
        'dist_monument_m': _distance_min(arbre_monuments, point_rad),
        'dist_hopital_m': _distance_min(arbre_hopitaux, point_rad),
        'dist_universite_m': _distance_min(arbre_universites, point_rad),
        'prix_m2_voisins': prix_voisins,
        'densite_ventes_1km': densite,
        'prix_m2_section': prix_section,
        'nb_ventes_section': nb_ventes_sec,
        'pct_dpe_A': val_commune('pct_dpe_A'),
        'pct_dpe_B': val_commune('pct_dpe_B'),
        'pct_dpe_C': val_commune('pct_dpe_C'),
        'pct_dpe_D': val_commune('pct_dpe_D'),
        'pct_dpe_E': val_commune('pct_dpe_E'),
        'pct_dpe_F': val_commune('pct_dpe_F'),
        'pct_dpe_G': val_commune('pct_dpe_G'),
        'pct_chauffage_elec': val_commune('pct_chauffage_elec'),
        'pct_chauffage_gaz': val_commune('pct_chauffage_gaz'),
        'pct_chauffage_fioul': val_commune('pct_chauffage_fioul'),
        'pct_chauffage_urbain': val_commune('pct_chauffage_urbain'),
        'median_revenu_disponible': val_commune('median_revenu_disponible'),
        'indice_gini': val_commune('indice_gini'),
        'pct_minima_sociaux': val_commune('pct_minima_sociaux'),
    }

    manquantes = [f for f in FEATURES if f not in valeurs]
    if manquantes:
        raise ValueError(f"Features manquantes pour le modele XGBoost : {manquantes}")

    return pd.DataFrame([[valeurs[f] for f in FEATURES]], columns=FEATURES)


# ==========================================
# FONCTION D'ESTIMATION PRINCIPALE
# ==========================================
def estimer(adresse, surface, type_bien, nb_pieces,
            surface_terrain=0, annee=2025, mois=6, geo_resolu=None):
    
    if type_bien not in ['maisons', 'appartements']:
        return {'erreur': "Le type de bien doit etre 'maisons' ou 'appartements'."}
        
    if type_bien not in modeles or 'median' not in modeles[type_bien]:
        return {'erreur': f"Le modele pour les {type_bien} n'est pas entraine."}

    # Resolution de l'adresse si non fournie
    if geo_resolu is None:
        geo = geocoder(adresse)
        if geo['status'] == 'erreur': 
            return {'erreur': geo['message']}
        if geo['status'] == 'suggestions':
            return {'suggestions': geo['suggestions']}
        geo_resolu = geo['resultat']

    # Construction des features
    X_bien = construire_features(
        geo_resolu['lat'], geo_resolu['lon'], geo_resolu['code_insee'],
        surface, type_bien, nb_pieces, surface_terrain, annee, mois
    )

    # Prediction sur le bon jeu de modeles
    m2_median = float(np.exp(modeles[type_bien]['median'].predict(X_bien)[0]))
    m2_bas = float(np.exp(modeles[type_bien]['bas'].predict(X_bien)[0]))
    m2_haut = float(np.exp(modeles[type_bien]['haut'].predict(X_bien)[0]))

    return {
        'adresse': geo_resolu['label'],
        'type_retenu': type_bien,
        'prix_m2_estime': round(m2_median),
        'prix_m2_fourchette': (round(m2_bas), round(m2_haut)),
        'prix_total_estime': round(m2_median * surface),
        'prix_total_fourchette': (round(m2_bas * surface), round(m2_haut * surface)),
    }


# ==========================================
# DEMO INTERACTIVE EN LIGNE DE COMMANDE
# ==========================================
if __name__ == "__main__":
    print("\n--- ESTIMATION D'UN BIEN ---")
    adresse = input("Adresse : ").strip()
    surface = float(input("Surface habitable (m2) : "))
    
    choix_type = input("Type (maison/appartement) : ").strip().lower()
    type_bien = 'maisons' if choix_type.startswith('m') else 'appartements'
    
    nb_pieces = int(input("Nombre de pieces : "))
    
    terrain = input("Surface terrain (m2, 0 si aucun) : ").strip()
    surface_terrain = float(terrain) if terrain else 0

    # Lancement de l'estimation
    res = estimer(adresse, surface, type_bien, nb_pieces, surface_terrain)

    # Gestion des adresses incertaines
    if 'suggestions' in res: 
        print("\nAdresse incertaine. Vouliez-vous dire :")
        for i, s in enumerate(res['suggestions'], 1):
            print(f" {i}. {s['label']} (confiance {s['score']:.0%})")
        print(" 0. Aucune de ces adresses")

        choix = input("\nVotre choix (numero) : ").strip()
        if not choix.isdigit() or int(choix) == 0:
            print("Estimation annulee.")
            sys.exit()
            
        idx = int(choix) - 1
        if idx < 0 or idx >= len(res['suggestions']):
            print("Choix invalide.")
            sys.exit()
        
        # On relance avec l'adresse forcee
        res = estimer(adresse, surface, type_bien, nb_pieces,
                      surface_terrain, geo_resolu=res['suggestions'][idx])
        
    # Affichage du resultat
    print("\n" + "=" * 50)
    if 'erreur' in res:
        print(f"ERREUR : {res['erreur']}")
    else:
        print(f"Bien ({res['type_retenu']}) : {res['adresse']}")
        print(f"Prix au m2 estime : {res['prix_m2_estime']} EUR/m2")
        print(f"  Fourchette 90%  : {res['prix_m2_fourchette'][0]} - {res['prix_m2_fourchette'][1]} EUR/m2")
        print("-" * 50)
        print(f"Prix total estime : {res['prix_total_estime']} EUR")
        print(f"  Fourchette 90%  : {res['prix_total_fourchette'][0]} - {res['prix_total_fourchette'][1]} EUR")
    print("=" * 50)