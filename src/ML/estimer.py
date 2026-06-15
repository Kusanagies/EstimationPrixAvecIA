"""
PHASE 3 - ESTIMATION D'UN BIEN A PARTIR DE SON ADRESSE
=======================================================
Charge les artefacts produits par entrainer_et_sauvegarder.py et expose
la fonction estimer(adresse, surface, est_maison, nb_pieces, surface_terrain).

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
# CHARGEMENT DES ARTEFACTS (une seule fois)
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[2]
DOSSIER_MODELE = RACINE_PROJET / "modele_production"

print("Chargement du modele...")

modeles = {}
for nom in ['bas', 'median', 'haut']:
    m = xgb.XGBRegressor()
    m.load_model(str(DOSSIER_MODELE / f"modele_{nom}.json"))
    modeles[nom] = m

with open(DOSSIER_MODELE / "features.json") as f:
    FEATURES = json.load(f)

with open(DOSSIER_MODELE / "contexte.pkl", "rb") as f:
    CTX = pickle.load(f)

RAYON_TERRE = CTX['rayon_terre']

# Reconstruction de l'arbre des voisins a partir des coordonnees sauvegardees
arbre_voisins = BallTree(CTX['arbre_voisins_data'], metric='haversine')
prix_all = CTX['prix_all']

# Arbres des amenites (reconstruits une fois)
def _arbre_ou_none(df):
    if df is not None and len(df) > 0:
        return BallTree(np.deg2rad(df[['latitude', 'longitude']]), metric='haversine')
    return None

arbre_stations = _arbre_ou_none(CTX['stations'])
arbre_monuments = _arbre_ou_none(CTX['monuments'])
arbre_hopitaux = _arbre_ou_none(CTX['hopitaux'])
universites = CTX['universites']
arbre_universites = _arbre_ou_none(universites)

print("Modele pret.")


# ==========================================
# GEOCODAGE (adresse -> lat, lon, code INSEE)
# ==========================================
def geocoder(adresse,limite = 5,seuil_confiance=0.6):
    """Utilise l'API publique de la Base Adresse Nationale (gratuite, sans cle)."""
    try:
        r = requests.get("https://api-adresse.data.gouv.fr/search/",
                         params={"q": adresse, "limit": 1}, timeout=10)
        data = r.json()
    except Exception:
        return {'status':'erreur','message':"Service de geocodage indisponible"}
    feats = data.get('features',[])
    if not feats : 
        return {'status':'erreur','message':'Aucune adresse trouvee'}
    
    def extraire(feat):
        lon,lat = feat['geometry']['coordinates']
        p = feat['properties']
        return {
            'lat': lat, 'lon':lon,
            'code_insee': p.get('citycode'),
            'label': p.get('label',adresse),
            'score': p.get('score',0)
        }
    candidats = [extraire(f) for f in feats]
    meilleur = candidats[0]

    if meilleur['score'] >= seuil_confiance:
        return {'status':'ok', 'resultat':meilleur}
    
    return {'status':'suggestions','suggestions':candidats}

def recuperer_section(lat,lon):
    """
    Recuepre la parcelle cadastrale a partir de coordonnées via l'api carto IGN, et reconstruit
    le code_section au format DVF (code commune + prefixe + section).
    Renvoie None si introuvable
    """
    try : 
        # Geom attend un point geoJSON {"type":"Point","coordinates":[lon,lat]}
        geom = f'{{"type":"Point","coordinates":[{lon},{lat}]}}'
        r = requests.get("https//apicarto.ign.fr/api/cadastre/parcelle",
                         params={"geom":geom},timeout = 10)
        data = r.json()
    except Exception:
        return None

    feats = data.get('features',[])
    if not feats : 
        return None
    props = feats[0]['properties']

    code_com = props.get('code_dep','') + props.get('code_com','')
    prefixe =props.get('com_abs','000')
    section = props.get('section','')

    if not code_com or not section : 
        return None
    
    code_section = (code_com + prefixe + section)[:10]
    return code_section
    
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
def construire_features(lat, lon, code_insee, surface, est_maison,
                        nb_pieces, surface_terrain, annee, mois):
    """Reconstruit toutes les features d'un bien, dans l'ordre de FEATURES."""
    point_rad = np.deg2rad([[lat, lon]])

    # Profil communal (DPE, chauffage, revenus). Repli sur medianes globales.
    profil = CTX['profils_communes'].get(code_insee, {})
    med_glob = CTX['medianes_globales']

    def val_commune(col):
        v = profil.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return med_glob.get(col, 0.0)
        return v

    # prix_m2_voisins : mediane des 15 plus proches ventes
    k = min(15, len(prix_all))
    _, idx = arbre_voisins.query(point_rad, k=k)
    prix_voisins = float(np.median(prix_all[idx[0]]))

    # densite : nb de ventes dans 1 km
    rayon_rad = 1000 / RAYON_TERRE
    densite = int(arbre_voisins.query_radius(point_rad, r=rayon_rad, count_only=True)[0])

    # prix_m2_section : on n'a pas la section de l'adresse, on retombe sur
    # la mediane communale (puis globale). C'est le repli prevu cote train.
    code_section = recuperer_section(lat,lon)
    if code_section is not None and code_section in CTX['med_section']:
        prix_section = CTX['med_section'][code_section]
    elif code_insee in CTX['med_section']:
        prix_section = CTX['med_commune'][code_insee]
    else : 
        prix_section = CTX['med_globale']

    # nb_ventes_section : approxime par les ventes de la commune (pas de section)
    if code_section is not None and code_section in CTX['nb_ventes_section']:
        nb_ventes_sec = CTX['nb_ventes_section'][code_section]
    else : 
        nb_ventes_sec = densite  # proxy raisonnable a l'echelle d'une adresse

    # Volume etudiants de l'universite la plus proche
    if arbre_universites is not None:
        _, idx_u = arbre_universites.query(point_rad, k=1)
        vol_etudiants = float(universites.iloc[idx_u[0][0]]['nombre_etudiants'])
    else:
        vol_etudiants = 0.0

    surface_terrain = surface_terrain if surface_terrain else 0

    valeurs = {
        'est_maison': int(est_maison),
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
        # DPE, chauffage, revenus -> profil communal
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

    # On construit le DataFrame STRICTEMENT dans l'ordre de FEATURES.
    # Si une feature manque, on leve une erreur claire plutot que de predire faux.
    manquantes = [f for f in FEATURES if f not in valeurs]
    if manquantes:
        raise ValueError(f"Features manquantes : {manquantes}")

    return pd.DataFrame([[valeurs[f] for f in FEATURES]], columns=FEATURES)


# ==========================================
# FONCTION D'ESTIMATION
# ==========================================
def estimer(adresse, surface, est_maison, nb_pieces,
            surface_terrain=0, annee=2025, mois=6,geo_resolu=None):
    if geo_resolu is None:
        geo = geocoder(adresse)
        if geo['status'] == 'erreur' : 
            return {'erreur':geo['message']}
        if geo['status'] == 'suggestions':
            return {'suggestions': geo['suggestions']}
        geo_resolu = geo['resultat']

    X_bien = construire_features(
        geo_resolu['lat'], geo_resolu['lon'],geo_resolu['code_insee'],
        surface,est_maison,nb_pieces,surface_terrain,annee,mois
    )
    geo = geocoder(adresse)
    if geo is None:
        return {'erreur': "Adresse introuvable"}

    X_bien = construire_features(
        geo_resolu['lat'], geo_resolu['lon'], geo_resolu['code_insee'],
        surface, est_maison, nb_pieces, surface_terrain, annee, mois
    )

    m2_median = float(np.exp(modeles['median'].predict(X_bien)[0]))
    m2_bas = float(np.exp(modeles['bas'].predict(X_bien)[0]))
    m2_haut = float(np.exp(modeles['haut'].predict(X_bien)[0]))

    return {
        'adresse': geo_resolu['label'],
        'prix_m2_estime': round(m2_median),
        'prix_m2_fourchette': (round(m2_bas), round(m2_haut)),
        'prix_total_estime': round(m2_median * surface),
        'prix_total_fourchette': (round(m2_bas * surface), round(m2_haut * surface)),
    }


# ==========================================
# DEMO INTERACTIVE
# ==========================================
if __name__ == "__main__":
    print("\n--- ESTIMATION D'UN BIEN ---")
    adresse = input("Adresse : ").strip()
    surface = float(input("Surface habitable (m2) : "))
    type_bien = input("Type (maison/appartement) : ").strip().lower()
    est_maison = 1 if type_bien.startswith('m') else 0
    nb_pieces = int(input("Nombre de pieces : "))
    terrain = input("Surface terrain (m2, 0 si aucun) : ").strip()
    surface_terrain = float(terrain) if terrain else 0

    res = estimer(adresse, surface, est_maison, nb_pieces, surface_terrain)

    if 'suggestions' in res: 
        print("\nAdresse incertaine. Vouliez-vous dire :")
        for i, s in enumerate(res['suggestions'],1):
            print(f" {i}. {s['label']} (confiance {s['score']:.0%})")
        print(" 0. Aucune de ces adresses")

        choix = input("\nVotre choix (numero) :").strip()
        if not choix.isdigit() or int(choix) == 0 :
            print("Estimation annulee.")
            sys.exit()
        idx = int(choix) - 1
        if idx < 0 or idx >= len(res['suggestions']):
            print("Choix invalide.")
            sys.exit()
        
        res = estimer(adresse,surface,est_maison,nb_pieces,
                      surface_terrain, geo_resolu=res['suggestions'][idx])
        
    print("\n" + "=" * 50)
    if 'erreur' in res:
        print(res['erreur'])
    else:
        print(f"Bien : {res['adresse']}")
        print(f"Prix au m2 estime : {res['prix_m2_estime']} EUR/m2")
        print(f"  Fourchette 90%  : {res['prix_m2_fourchette'][0]} - {res['prix_m2_fourchette'][1]} EUR/m2")
        print("-" * 50)
        print(f"Prix total estime : {res['prix_total_estime']} EUR")
        print(f"  Fourchette 90%  : {res['prix_total_fourchette'][0]} - {res['prix_total_fourchette'][1]} EUR")
    print("=" * 50)