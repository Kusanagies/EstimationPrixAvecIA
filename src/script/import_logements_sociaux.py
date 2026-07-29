import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

# 1. Chargement des variables d'environnement
RACINE_PROJET = Path(__file__).resolve().parents[2] # Ajuste selon l'emplacement du script
load_dotenv(RACINE_PROJET / ".env")

try:
    db_pass = os.environ["DB_PASS"]
    # Connexion à ta base EstimationIA
    moteur_enr = create_engine(f"mysql+pymysql://root:{db_pass}@localhost:3306/EstimationIA")
except KeyError:
    print("Erreur : DB_PASS introuvable dans .env")
    exit()

# 2. Lecture du fichier CSV
chemin_csv = RACINE_PROJET / "data" / "logementsSOC.csv" # Ajuste le chemin vers ton fichier
print(f"Lecture du fichier {chemin_csv}...")

# On force 'code_departement' et 'code_epci' en string pour garder les zéros (ex: '07')
df = pd.read_csv(
    chemin_csv, 
    sep=';', 
    dtype={'code_departement': str, 'code_epci': str}
)

# 3. Dictionnaire de renommage pour des colonnes SQL propres
renommage = {
    "année_publication": "annee_publication",
    "code_departement": "code_departement",
    "nom_departement": "nom_departement",
    "code_region": "code_region",
    "nom_region": "nom_region",
    "Nombre  d'habitants": "nb_habitants",
    "Densité de population au km²": "densite_population_km2",
    "Variation de la population sur 10 ans (en %)": "var_pop_10ans_pct",
    "Dont contribution du solde naturel (en %)": "solde_naturel_pct",
    "Dont contribution du solde migratoire (en %)": "solde_migratoire_pct",
    "% population de moins de 20 ans": "pop_moins_20_ans_pct",
    "% population de 60 ans et plus": "pop_60_ans_plus_pct",
    "Taux de chômage au T4 (en %)": "taux_chomage_t4_pct",
    "Taux de pauvreté* (en %)": "taux_pauvrete_pct",
    "Nombre de logements": "nb_logements",
    "Nombre de résidences principales": "nb_residences_principales",
    "Taux de logements sociaux* (en %)": "taux_logements_sociaux_pct",
    "Taux de logements vacants* (en %)": "taux_logements_vacants_pct",
    "Taux de logements individuels (en %)": "taux_logements_individuels_pct",
    "Moyenne annuelle de la construction neuve sur 10 ans": "moy_construction_neuve_10ans",
    "Construction": "construction",
    "Parc social - Nombre de logements": "parc_soc_nb_logements",
    "Parc social - Logements mis en location*": "parc_soc_mis_en_location",
    "Parc social - Logements démolis": "parc_soc_demolis",
    "Parc social - Ventes à des personnes physiques": "parc_soc_ventes_physiques",
    "Parc social - Taux de logements vacants* (en %)": "parc_soc_vacants_pct",
    "Parc social - Taux de logements individuels (en %)": "parc_soc_individuels_pct",
    "Parc social - Loyer moyen (en €/m²/mois)*": "parc_soc_loyer_moyen_eur_m2",
    "Parc social - Âge moyen du parc  (en années)": "parc_soc_age_moyen",
    "Parc social - Taux de logements énergivores (E,F,G)* (en %)": "parc_soc_energivores_pct",
    "geom": "geom_geojson",
    "dep_centroid": "dep_centroid",
    "epci": "nom_epci",
    "code_epci": "code_epci"
}

# Appliquer le renommage
df = df.rename(columns=renommage)

# 4. Envoi vers MySQL
nom_table = "statistiques_logements_sociaux"
print(f"Création/Remplacement de la table '{nom_table}' dans EstimationIA...")

# if_exists='replace' écrase la table si elle existe déjà (utile pour tes tests)
# Le paramètre chunksize permet de ne pas saturer la mémoire si le fichier est très lourd
df.to_sql(name=nom_table, con=moteur_enr, if_exists='replace', index=False, chunksize=1000)

print(f"✅ Importation terminée avec succès ! ({len(df)} lignes insérées)")