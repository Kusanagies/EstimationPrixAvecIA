import os
import sys
import requests
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 1. INITIALISATION ET CONNEXION BDD
# ==========================================
RACINE_PROJET = Path(__file__).resolve().parents[2]
load_dotenv(RACINE_PROJET / ".env")

try:
    db_pass = os.environ["DB_PASS"]
    moteur = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
except KeyError:
    print("Erreur : variable DB_PASS introuvable. Verifiez votre fichier .env")
    sys.exit()
except Exception as e:
    print("Erreur de connexion a la base MySQL.")
    sys.exit()

# ==========================================
# 2. FONCTION D'AUTOCOMPLÉTION (API BAN)
# ==========================================
def rechercher_adresse(saisie_utilisateur):
    print("\nRecherche de l'adresse en cours via l'API Gouvernementale...")
    
    # Appel à l'API Base Adresse Nationale
    url = f"https://api-adresse.data.gouv.fr/search/?q={saisie_utilisateur}&limit=5"
    
    try:
        reponse = requests.get(url)
        reponse.raise_for_status()
        donnees = reponse.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur de communication avec l'API : {e}")
        return None

    resultats = donnees.get("features", [])
    
    if not resultats:
        print("Aucune adresse trouvee. Veuillez verifier l'orthographe.")
        return None
        
    return resultats

# ==========================================
# 3. INTERACTION UTILISATEUR
# ==========================================
print("-" * 50)
print("MODULE D'ANALYSE D'ADRESSE ET PROFILAGE ENERGETIQUE")
print("-" * 50)

saisie = input("Entrez l'adresse de votre bien : ").strip()
adresses_trouvees = rechercher_adresse(saisie)

if not adresses_trouvees:
    sys.exit()

# Affichage des propositions (Autocomplétion)
print("\nPlusieurs correspondances trouvées. Veuillez choisir l'adresse exacte :")
for i, feature in enumerate(adresses_trouvees):
    proprietes = feature["properties"]
    label = proprietes.get("label", "Adresse inconnue")
    contexte = proprietes.get("context", "")
    print(f"  [{i + 1}] {label} ({contexte})")

choix = input(f"\nVotre choix (1-{len(adresses_trouvees)}) : ").strip()

try:
    index_choix = int(choix) - 1
    if index_choix < 0 or index_choix >= len(adresses_trouvees):
        raise ValueError
except ValueError:
    print("Choix invalide. Fin du programme.")
    sys.exit()

# Extraction des données géographiques du choix
adresse_selectionnee = adresses_trouvees[index_choix]
props = adresse_selectionnee["properties"]
coords = adresse_selectionnee["geometry"]["coordinates"]

label_complet = props.get("label")
code_insee = props.get("citycode")  # Gère automatiquement les arrondissements
longitude = coords[0]
latitude = coords[1]

print("\n" + "=" * 50)
print(f"ADRESSE VALIDÉE : {label_complet}")
print(f"Coordonnées GPS : Lat {latitude:.5f}, Lon {longitude:.5f}")
print(f"Code INSEE/Secteur : {code_insee}")
print("=" * 50)

# ==========================================
# 4. APPROXIMATION DPE ET CHAUFFAGE (SQL)
# ==========================================
print("\nInterrogation de la base de données pour le profilage du secteur...")

query_profilage = f"""
    SELECT 
        COUNT(*) as total_diagnostics,
        
        -- Calcul du DPE dominant
        SUM(CASE WHEN etiquette_dpe = 'A' THEN 1 ELSE 0 END) AS nb_A,
        SUM(CASE WHEN etiquette_dpe = 'B' THEN 1 ELSE 0 END) AS nb_B,
        SUM(CASE WHEN etiquette_dpe = 'C' THEN 1 ELSE 0 END) AS nb_C,
        SUM(CASE WHEN etiquette_dpe = 'D' THEN 1 ELSE 0 END) AS nb_D,
        SUM(CASE WHEN etiquette_dpe = 'E' THEN 1 ELSE 0 END) AS nb_E,
        SUM(CASE WHEN etiquette_dpe = 'F' THEN 1 ELSE 0 END) AS nb_F,
        SUM(CASE WHEN etiquette_dpe = 'G' THEN 1 ELSE 0 END) AS nb_G,
        
        -- Calcul du Chauffage dominant
        SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Électricité%%' THEN 1 ELSE 0 END) AS ch_elec,
        SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Gaz%%' THEN 1 ELSE 0 END) AS ch_gaz,
        SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Fioul%%' THEN 1 ELSE 0 END) AS ch_fioul,
        SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Réseau de Chaleur%%' THEN 1 ELSE 0 END) AS ch_urbain
    FROM dpe_logements_france
    WHERE code_insee_ban = '{code_insee}';
"""

df_profil = pd.read_sql(query_profilage, con=moteur)

if len(df_profil) == 0 or df_profil['total_diagnostics'].iloc[0] == 0:
    print("Données insuffisantes dans ce secteur pour établir une approximation énergétique.")
else:
    donnees_secteur = df_profil.iloc[0]
    total = donnees_secteur['total_diagnostics']
    
    # Recherche de la lettre DPE avec le plus grand nombre
    dpes = {
        'A': donnees_secteur['nb_A'], 'B': donnees_secteur['nb_B'], 
        'C': donnees_secteur['nb_C'], 'D': donnees_secteur['nb_D'], 
        'E': donnees_secteur['nb_E'], 'F': donnees_secteur['nb_F'], 
        'G': donnees_secteur['nb_G']
    }
    dpe_dominant = max(dpes, key=dpes.get)
    pourcentage_dpe = (dpes[dpe_dominant] / total) * 100
    
    # Recherche du chauffage dominant
    chauffages = {
        'Électrique': donnees_secteur['ch_elec'],
        'Gaz': donnees_secteur['ch_gaz'],
        'Fioul': donnees_secteur['ch_fioul'],
        'Réseau Urbain': donnees_secteur['ch_urbain']
    }
    chauffage_dominant = max(chauffages, key=chauffages.get)
    pourcentage_chauffage = (chauffages[chauffage_dominant] / total) * 100

    print(f"\nPROFIL ÉNERGÉTIQUE APPROXIMATIF (Basé sur {total} diagnostics locaux) :")
    print("-" * 50)
    print(f"-> DPE le plus probable       : {dpe_dominant} (Concerne {pourcentage_dpe:.1f}% des biens du secteur)")
    print(f"-> Type de Chauffage dominant : {chauffage_dominant} (Concerne {pourcentage_chauffage:.1f}% des biens du secteur)")
    print("-" * 50)