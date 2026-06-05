import pandas as pd
import numpy as np
import folium
from sqlalchemy import create_engine
from sklearn.cluster import DBSCAN 

# ==========================================
# 1. PARAMÈTRES ET CONNEXION
# ==========================================
DISTANCE_X_METRES = 25

USER = 'root'
PASSWORD = '1618'
HOTE = 'localhost'
PORT = '3306'
DB = 'EstimationIA'

chaine_connexion = f"mysql+pymysql://{USER}:{PASSWORD}@{HOTE}:{PORT}/{DB}"
moteur = create_engine(chaine_connexion)

print("Récupération des stations depuis MySQL...")
requete = "SELECT stop_name, stop_lat, stop_lon FROM donnees_transport_idf;"
stations_idf = pd.read_sql(requete, con=moteur)
stations_idf = stations_idf.dropna(subset=['stop_lat', 'stop_lon'])
print(f"Avant filtrage : {len(stations_idf)} stations.")

# ==========================================
# 2. REGROUPEMENT SPATIAL (CLUSTERING)
# ==========================================
print(f"Regroupement des stations distantes de moins de {DISTANCE_X_METRES}m...")

# On convertit les coordonnées en radians
coords_rad = np.deg2rad(stations_idf[['stop_lat', 'stop_lon']])

# On convertit notre distance X en radians (en la divisant par le rayon de la Terre)
RAYON_TERRE_METRES = 6371000
epsilon_rad = DISTANCE_X_METRES / RAYON_TERRE_METRES

# Configuration de l'algorithme DBSCAN
dbscan = DBSCAN(
    eps=epsilon_rad, 
    min_samples=1,            # 1 point suffit pour créer un groupe
    algorithm='ball_tree',    # Optimisé pour la géographie
    metric='haversine'        # Prend en compte la courbure de la Terre
)

# On lance le calcul : chaque station reçoit un "Numéro de groupe"
stations_idf['numero_groupe'] = dbscan.fit_predict(coords_rad)

# LA MAGIE PANDAS : On regroupe par 'numero_groupe' et on ne garde que la 1ère station de chaque groupe
stations_filtrees = stations_idf.groupby('numero_groupe').first().reset_index()

print(f"Après filtrage : {len(stations_filtrees)} stations uniques conservées.")

# ==========================================
# 3. CRÉATION DE LA CARTE ALLÉGÉE
# ==========================================
carte = folium.Map(location=[48.8566, 2.3522], zoom_start=10, tiles='CartoDB positron')

for index, ligne in stations_filtrees.iterrows():
    folium.CircleMarker(
        location=[ligne['stop_lat'], ligne['stop_lon']],
        radius=3,                 # Un peu plus gros puisqu'il y a moins de points !
        color='#e74c3c',          # En rouge pour marquer la différence
        fill=True,
        fill_color='#e74c3c',
        fill_opacity=0.8,
        popup=ligne['stop_name']
    ).add_to(carte)

# Sauvegarde
nom_fichier = 'carte_transports_idf_filtree.html'
carte.save(nom_fichier)

print(f"Carte générée avec succès ! Ouvrez le fichier '{nom_fichier}'.")
# Un peu lourd à envoyer le fichier à git
# Lancer ensuite avec : 
# xdg-open carte_transports_idf.html