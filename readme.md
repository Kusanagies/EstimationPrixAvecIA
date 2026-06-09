Les dépendances nécessaires sont dans le fichier requirements.txt

Il est recommandé d'avoir un environnement virtuel (python env), on le crée avec :
> python3 -m venv env
et
> pip install requirements.txt
et
>source env/bin/activate/
ET VOILA

Pour lancer un script : 
> python3 {nomdufichier}.py

Pour "lancer" les fichiers html pour la visualisation des cartes : 
> xdg-open {nomdufichier}.html

Liens vers les données : 


Pour la France : 

https://www.data.gouv.fr/datasets/referentiel-des-arrets-arrets-transporteur //Pas utilisé dans la db

https://www.data.gouv.fr/datasets/base-adresse-nationale

Nom dans la db : adresses_ban

La première ligne avec une autre ligne de donnée: # id_db, id, id_fantoir, numero, rep, nom_voie, code_postal, code_insee, nom_commune, code_insee_ancienne_commune, nom_ancienne_commune, x, y, lon, lat, type_position, alias, nom_ld, libelle_acheminement, nom_afnor, source_position, source_nom_voie, certification_commune, cad_parcelles

'5', '01001_NGZLQW_00026', '', '26', '', 'Imp des Epis', '01400', '01001', 'L\'Abergement-Clémenciat', '', '', '848670.66', '6563239.01', '4.92651900', '46.15264600', '', '', '', 'L\'ABERGEMENT-CLEMENCIAT', 'IMP DES EPIS', 'arcep', 'arcep', '0', ''

https://www.data.gouv.fr/datasets/dpe-logements-existants-depuis-juillet-2021

Nom dans la db : dpe_logements_france

La première ligne avec une autre ligne de donnée : # numero_dpe, date_etablissement_dpe, etiquette_dpe, etiquette_ges, type_batiment, annee_construction, type_installation_chauffage, hauteur_sous_plafond, typologie_logement, surface_habitable_logement, adresse_ban, numero_voie_ban, nom_rue_ban, nom_commune_ban, code_insee_ban, identifiant_ban, adresse_complete_brut, numero_etage_appartement, logement_traversant, cout_total_5_usages, type_energie_principale_chauffage

'2611E0031228S', '2026-01-07', 'C', 'A', 'immeuble', NULL, 'individuel', '2.5', NULL, NULL, '25 Rue de Belfort 11000 Carcassonne', '25', 'Rue de Belfort', 'Carcassonne', '11069', '11069_0550_00025', '25 Rue de Belfort\nJARDINS DES REMPARTS I 11000 CARCASSONNE', '0', NULL, '23106.4', 'Électricité'

https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres

Nom dans la db : valeurs_foncieres

La première ligne avec une autre ligne de donné : # id, id_mutation, date_mutation, numero_disposition, nature_mutation, valeur_fonciere, adresse_numero, adresse_suffixe, adresse_nom_voie, adresse_code_voie, code_postal, code_commune, nom_commune, code_departement, ancien_code_commune, ancien_nom_commune, id_parcelle, ancien_id_parcelle, numero_volume, lot1_numero, lot1_surface_carrez, lot2_numero, lot2_surface_carrez, lot3_numero, lot3_surface_carrez, lot4_numero, lot4_surface_carrez, lot5_numero, lot5_surface_carrez, nombre_lots, code_type_local, type_local, surface_reelle_bati, nombre_pieces_principales, code_nature_culture, nature_culture, code_nature_culture_speciale, nature_culture_speciale, surface_terrain, longitude, latitude

'1', '2021-1', '2021-01-05', '000001', 'Vente', '185000.00', '5080', '', 'CHE DE VOGELAS', '0471', '01370', '01426', 'Val-Revermont', '01', '', '', '01426312ZC0122', '', '', '', '0.00', '', '0.00', '', '0.00', '', '0.00', '', '0.00', '0', '3', 'Dépendance', '0.00', '0', 'S', 'sols', '', '', '2410.00', '5.38610700', '46.32710100'

https://www.data.gouv.fr/datasets/effectifs-d-etudiants-inscrits-dans-les-etablissements-et-les-formations-de-l-enseignement-superieur

Nom dans la db : infrastructures_universites

La première ligne avec une autre ligne de donnée : # id_uai, nom_universite, code_insee, nom_commune, latitude, longitude, nombre_etudiants

'0292125C', 'ISEN YNCREA OU', '92002', 'Antony', '48.75382927407107', '2.300040078825897', '14'

https://www.data.gouv.fr/datasets/gares-ferroviaires-de-tous-types-exploitees-ou-non

Nom dans la db : donnees_transport 

La première ligne avec une autre ligne de donnée :  # code_ligne, nom_gare, nature, latitude, longitude

'420000', 'Kerhuon', 'Desserte Fret-Desserte Voyageur-Infrastructure', '48.40931', '-4.3882'

https://www.data.gouv.fr/datasets/localisation-des-hopitaux-dans-openstreetmap

Nom dans la db : infrastructures_hopitaux

La première ligne avec une autre ligne de donnée : # osm_id, nom_hopital, urgences, nom_commune, code_postal, longitude, latitude

'38072526', 'Hôpital Bichat-Claude Bernard', 'yes', 'Paris', NULL, '2.3319566083744125', '48.898862750459095'

https://www.data.gouv.fr/datasets/immeubles-proteges-au-titre-des-monuments-historiques-2

Nom dans la db ; merimee.csv 

La première ligne avec une autre ligne de donnée : # id_monument, nom_commune, type_monument, code_insee, siecle, adresse, latitude, longitude

'PA00080241', 'Wy-dit-Joli-Village', 'site archéologique', '95690', 'Antiquité', NULL, '49.1030750412716', '1.83517099965768'

Pour Paris : 
https://www.data.gouv.fr/datasets/dans-ma-rue-anomalies-signalees //On n'utilise pas actuellement

Pour l'IDF : 
https://www.data.gouv.fr/datasets/les-commerces-par-commune-ou-arrondissement-base-permanente-des-equipements-idf //On n'utilise pas actuellement
