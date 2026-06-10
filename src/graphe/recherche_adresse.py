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

# Extraction des donnees de l'API BAN
adresse_selectionnee = adresses_trouvees[index_choix]
props = adresse_selectionnee["properties"]
coords = adresse_selectionnee["geometry"]["coordinates"]

label_complet = props.get("label")
code_insee = props.get("citycode")
longitude = coords[0]
latitude = coords[1]
id_ban = props.get("id") # L'identifiant unique de la porte

# Recuperation du nom de la rue pour la recherche de zone
type_adresse = props.get("type")
if type_adresse == "street":
    rue = props.get("name", "")
else:
    rue = props.get("street", props.get("name", ""))

# Echappement des apostrophes pour eviter les bugs SQL (ex: "d'Abbans" -> "d''Abbans")
rue_sql = rue.replace("'", "''")

print("\n" + "=" * 50)
print(f"ADRESSE VALIDÉE : {label_complet}")
print(f"ID BAN          : {id_ban}")
print(f"Coordonnées GPS : Lat {latitude:.5f}, Lon {longitude:.5f}")
print("=" * 50)

# ==========================================
# 4. RECHERCHE EXACTE (L'ADRESSE)
# ==========================================
print("\nRecherche d'un diagnostic exact pour cette adresse...")

# On cherche si le batiment exact possede un DPE via son ID BAN
query_exacte = f"""
    SELECT etiquette_dpe, type_energie_principale_chauffage, date_etablissement_dpe
    FROM dpe_logements_france
    WHERE identifiant_ban = '{id_ban}'
    ORDER BY date_etablissement_dpe DESC
    LIMIT 1;
"""
df_exact = pd.read_sql(query_exacte, con=moteur)

if len(df_exact) > 0:
    print("\n🎯 MATCH EXACT TROUVÉ DANS LA BASE ADEME !")
    print(f"-> DPE du bien       : {df_exact.iloc[0]['etiquette_dpe']}")
    print(f"-> Chauffage du bien : {df_exact.iloc[0]['type_energie_principale_chauffage']}")
    print(f"-> Date du DPE       : {df_exact.iloc[0]['date_etablissement_dpe']}")
else:
    print("❌ Aucun diagnostic exact trouvé pour ce bâtiment spécifique.")

# ==========================================
# 5. RECHERCHE DE ZONE (LA RUE / LE VOISINAGE)
# ==========================================
print(f"\nCalcul du profil énergétique de la micro-zone (Rue : {rue})...")

query_zone = f"""
    SELECT 
        COUNT(*) as total_diagnostics,
        SUM(CASE WHEN etiquette_dpe = 'A' THEN 1 ELSE 0 END) AS nb_A,
        SUM(CASE WHEN etiquette_dpe = 'B' THEN 1 ELSE 0 END) AS nb_B,
        SUM(CASE WHEN etiquette_dpe = 'C' THEN 1 ELSE 0 END) AS nb_C,
        SUM(CASE WHEN etiquette_dpe = 'D' THEN 1 ELSE 0 END) AS nb_D,
        SUM(CASE WHEN etiquette_dpe = 'E' THEN 1 ELSE 0 END) AS nb_E,
        SUM(CASE WHEN etiquette_dpe = 'F' THEN 1 ELSE 0 END) AS nb_F,
        SUM(CASE WHEN etiquette_dpe = 'G' THEN 1 ELSE 0 END) AS nb_G,
        SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Électricité%%' THEN 1 ELSE 0 END) AS ch_elec,
        SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Gaz%%' THEN 1 ELSE 0 END) AS ch_gaz,
        SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Fioul%%' THEN 1 ELSE 0 END) AS ch_fioul,
        SUM(CASE WHEN type_energie_principale_chauffage LIKE '%%Réseau de Chaleur%%' THEN 1 ELSE 0 END) AS ch_urbain
    FROM dpe_logements_france
    WHERE code_insee_ban = '{code_insee}'
      AND adresse_ban LIKE '%%{rue_sql}%%';
"""

df_zone = pd.read_sql(query_zone, con=moteur)

if len(df_zone) == 0 or df_zone['total_diagnostics'].iloc[0] == 0:
    print(f"Données insuffisantes dans cette rue pour établir une approximation.")
else:
    donnees_zone = df_zone.iloc[0]
    total = donnees_zone['total_diagnostics']
    
    dpes = {
        'A': donnees_zone['nb_A'], 'B': donnees_zone['nb_B'], 
        'C': donnees_zone['nb_C'], 'D': donnees_zone['nb_D'], 
        'E': donnees_zone['nb_E'], 'F': donnees_zone['nb_F'], 
        'G': donnees_zone['nb_G']
    }
    dpe_dominant = max(dpes, key=dpes.get)
    pourcentage_dpe = (dpes[dpe_dominant] / total) * 100
    
    chauffages = {
        'Électrique': donnees_zone['ch_elec'],
        'Gaz': donnees_zone['ch_gaz'],
        'Fioul': donnees_zone['ch_fioul'],
        'Réseau Urbain': donnees_zone['ch_urbain']
    }
    chauffage_dominant = max(chauffages, key=chauffages.get)
    pourcentage_chauffage = (chauffages[chauffage_dominant] / total) * 100

    print(f"\nPROFIL DE LA RUE (Basé sur {total} diagnostics voisins) :")
    print("-" * 50)
    print(f"-> Tendance DPE Quartier       : {dpe_dominant} ({pourcentage_dpe:.1f}% des biens de la rue)")
    print(f"-> Tendance Chauffage Quartier : {chauffage_dominant} ({pourcentage_chauffage:.1f}% des biens de la rue)")
    print("-" * 50)