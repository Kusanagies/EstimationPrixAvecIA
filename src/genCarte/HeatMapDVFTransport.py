import pandas as pd
import folium
from folium.plugins import HeatMap
from sqlalchemy import create_engine

# ==========================================
# 1. CONNEXION MYSQL (Rappel)
# ==========================================
USER = 'root'
PASSWORD = '1618'
HOTE = 'localhost'
PORT = '3306'
DB = 'EstimationIA'

chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"
moteur = create_engine(chaine_connexion)

print("Récupération des données...")

# ==========================================
# 2. RÉCUPÉRATION ET NETTOYAGE DVF (L'Immobilier)
# ==========================================
# On calcule le prix au m² directement en SQL et on exclut les surfaces nulles
requete_dvf = """
SELECT latitude, longitude, (valeur_fonciere / surface_reelle_bati) AS prix_m2
FROM valeurs_foncieres_idf
WHERE latitude IS NOT NULL 
  AND surface_reelle_bati > 9; -- On ignore les placards vendus comme appartements
"""
maisons = pd.read_sql(requete_dvf, con=moteur)

# Filtrage des valeurs extrêmes pour que la carte soit belle
# On ne garde que les biens entre 2 000€ et 15 000€ du m² (Prix classiques IDF)
maisons_filtrees = maisons[(maisons['prix_m2'] >= 2000) & (maisons['prix_m2'] <= 15000)]

print(f"{len(maisons_filtrees)} ventes immobilières retenues pour la heatmap.")

# ==========================================
# 3. RÉCUPÉRATION DES STATIONS (Transports)
# ==========================================
requete_transports = "SELECT stop_name, stop_lat, stop_lon, type_station FROM donnees_transport_idf;"
stations = pd.read_sql(requete_transports, con=moteur)
stations = stations.dropna(subset=['stop_lat', 'stop_lon'])

# ==========================================
# 4. CRÉATION DE LA CARTE 
# ==========================================
# On initialise une carte sombre pour que la chaleur ressorte mieux ('cartodbdark_matter' ou 'cartodbpositron')
carte = folium.Map(location=[48.8566, 2.3522], zoom_start=11, tiles='CartoDB dark_matter')

# A. AJOUT DE LA HEATMAP (En premier, pour qu'elle soit en dessous)
print("Génération de la carte de chaleur...")
# Folium HeatMap attend une liste de listes : [[lat, lon, poids], [lat, lon, poids], ...]
donnees_chaleur = maisons_filtrees[['latitude', 'longitude', 'prix_m2']].values.tolist()

HeatMap(
    donnees_chaleur,
    radius=12,          # Taille du halo de chaleur autour d'une maison
    blur=15,            # Flou pour que les points se fondent entre eux
    max_zoom=1,         # Ajuste l'intensité selon le niveau de zoom
    gradient={0.2: 'blue', 0.5: 'lime', 0.8: 'yellow', 1.0: 'red'} # Code couleur classique
).add_to(carte)

# B. AJOUT DES STATIONS (En second, pour qu'elles soient au-dessus)
print("Ajout des stations de transport...")
for index, ligne in stations.iterrows():
    # On peut même changer la couleur selon le type de station !
    couleur = 'white'
    if ligne['type_station'] == 'tram':
        couleur = 'cyan'
    elif ligne['type_station'] == 'metro':
        couleur = 'magenta'
        
    folium.CircleMarker(
        location=[ligne['stop_lat'], ligne['stop_lon']],
        radius=3,
        color=couleur,
        fill=True,
        fill_opacity=0.9,
        popup=f"{ligne['stop_name']} ({ligne['type_station']})"
    ).add_to(carte)

# ==========================================
# 5. SAUVEGARDE
# ==========================================
nom_fichier = 'heatmap_immo_transports_idf.html'
carte.save(nom_fichier)
print(f"Terminé ! Ouvrez '{nom_fichier}' dans votre navigateur.")