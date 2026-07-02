# EstimationIA — Estimation immobilière par Machine Learning

Outil d'estimation du prix au m² d'un bien immobilier à partir de son adresse,
avec intervalle de confiance. Construit sur des données publiques françaises
(DVF, DPE, INSEE) et des modèles à base d'arbres boostés (XGBoost / CatBoost).

---

## Installation

Il est recommandé d'utiliser un environnement virtuel Python.

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Le mot de passe de la base est lu depuis un fichier `.env` à la racine du projet :

```
DB_PASS=votre_mot_de_passe
```

### Lancer un script

```bash
python3 nom_du_fichier.py
```

### Lancer l'API d'estimation (FastAPI)

```bash
uvicorn serveur:app --reload --port 5000
```

Puis ouvrir `http://127.0.0.1:5000` (interface web) ou `http://127.0.0.1:5000/docs`
(documentation interactive de l'API).

### Visualiser une carte HTML

```bash
xdg-open nom_du_fichier.html
```

---

## Sources de données

Toutes les données proviennent de [data.gouv.fr](https://www.data.gouv.fr).

| Table (base de données) | Source | Rôle | Remarque |
|---|---|---|---|
| `valeurs_foncieres` | [Demandes de valeurs foncières (DVF)](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres) | Cible : prix des ventes | Source principale (~20 M lignes) |
| `dpe_logements_france` | [DPE logements existants](https://www.data.gouv.fr/datasets/dpe-logements-existants-depuis-juillet-2021) | Profil énergétique par commune | Depuis juillet 2021 |
| `donnees-revenus-filosofi` | [Revenu des Français à la commune](https://www.data.gouv.fr/datasets/revenu-des-francais-a-la-commune) | Revenu médian, Gini, minima sociaux | **Données 2021** |
| `adresses_ban` | [Base Adresse Nationale](https://www.data.gouv.fr/datasets/base-adresse-nationale) | Géocodage | Non exploitée dans le pipeline actuel |
| `infrastructures_universites` | [Effectifs étudiants enseignement supérieur](https://www.data.gouv.fr/datasets/effectifs-d-etudiants-inscrits-dans-les-etablissements-et-les-formations-de-l-enseignement-superieur) | Distance + volume étudiants | |
| `donnees_transport` | [Gares ferroviaires de tous types](https://www.data.gouv.fr/datasets/gares-ferroviaires-de-tous-types-exploitees-ou-non) | Distance à la gare | |
| `infrastructures_hopitaux` | [Localisation des hôpitaux (OpenStreetMap)](https://www.data.gouv.fr/datasets/localisation-des-hopitaux-dans-openstreetmap) | Distance à l'hôpital | |
| `monuments_historiques` | [Immeubles protégés au titre des monuments historiques](https://www.data.gouv.fr/datasets/immeubles-proteges-au-titre-des-monuments-historiques-2) | Distance au monument | |
| `infrastructures_mairies` | [Annuaire de l'administration (service-public.gouv.fr)](https://www.data.gouv.fr/datasets/lannuaire-de-ladministration-base-de-donnees-locales) | Distance à la mairie (proxy centre-ville) | Filtré sur `type_service_local = mairie` ; ~35 000 communes |
| `TableGeo2022.gpkg` (fichier local) | [Communes de la loi Littoral au COG 2020-2022](https://www.data.gouv.fr/datasets/communes-de-la-loi-littoral-au-code-officiel-geographique-cog-2020-2022) | Distance à la mer / lac / estuaire | Colonne `CLASSEMENT` (Mer/Lac/Estuaire) ; features décisives en zone côtière |

### Sources testées mais écartées

- **BD TOPO de l'IGN (type de bâtiment)** : testée sur le département 34
  (nombre d'étages, nombre de logements, hauteur). Corrélations avec le prix
  quasi nulles (< 0,08 en valeur absolue), attributs souvent mal renseignés.
  Écartée après mesure : gain négligeable pour un coût d'intégration élevé.
- [Référentiel des arrêts — arrêts transporteur](https://www.data.gouv.fr/datasets/referentiel-des-arrets-arrets-transporteur)
- [Dans ma rue — anomalies signalées](https://www.data.gouv.fr/datasets/dans-ma-rue-anomalies-signalees) (Paris)
- [Les commerces par commune ou arrondissement — base permanente des équipements IDF](https://www.data.gouv.fr/datasets/les-commerces-par-commune-ou-arrondissement-base-permanente-des-equipements-idf) (Île-de-France)

---

## Architecture des fichiers

Le projet sépare volontairement plusieurs rôles distincts :

- **Pipeline d'évaluation** (`correlation.py`, `correlation_cat.py`) — mesure la
  performance honnête du modèle (split temporel, métriques, SHAP, graphes).
  Ne sauvegarde pas de modèle.
- **Pipeline rapide** (`pipeline.py`, `pipeline_cat.py`) — même modèle sans les
  graphes, pour itérer vite ; sauvegarde un `resultats.pkl`.
- **Pipeline de production** (`entrainer_prod.py` / version CatBoost) — réentraîne
  sur toutes les données et sauvegarde les artefacts dans `modele_production/`.
- **Estimation** (`estimer.py`) — charge les artefacts et estime un bien depuis
  son adresse, avec une fourchette de prix.
- **API + interface** (`serveur.py`, `index.html`) — expose l'estimation via une
  API REST (FastAPI) et une page web simple.
- **Analyse des erreurs** (`analyse_erreurs.py`) — recharge les résultats
  sauvegardés pour diagnostiquer les pires erreurs sans réentraîner.
- **Matrice de corrélation** — analyse exploratoire des liens entre variables.

### Artefacts de production (`modele_production/`)

- `modele_{type}_{quantile}.cbm` (ou `.json` en XGBoost) — les 6 modèles
  (3 quantiles × 2 types de biens).
- `features.json` — la liste ordonnée des features attendues par les modèles.
- `contexte_{type}.pkl` — les données d'enrichissement (voisinages, médianes de
  section/commune, infrastructures, points de littoral) permettant de
  reconstruire les features d'une nouvelle adresse.

---

## Liste des décisions de construction du modèle

Les décisions sont présentées dans l'ordre logique de construction. Chacune
répond à un choix concret rencontré pendant le développement.

### Phase 0 — Préparation des données

- Rassembler et joindre les sources hétérogènes (ventes, énergie, socio-économique, géographie).
- Choisir les mailles : le **bien individuel** pour DVF, la **commune** pour les données socio-économiques.
- Définir les clés de jointure : `code_commune` et `code_insee`.
- Dédupliquer les ventes DVF sur (`id_parcelle`, `prix_m2`, `surface`).

### Phase 1 — Cadrage du problème

- Prédire le **prix au m²** (et non le prix total).
- Appliquer une transformation **logarithmique** au prix pour gérer son asymétrie.
- **Séparer les modèles Maisons et Appartements** : leurs distributions de prix et
  leurs facteurs de valeur diffèrent trop pour un modèle unique.

### Phase 2 — Nettoyage et filtrage des aberrations

- Ne garder que les ventes **à un ou deux lots** (`nombre_lots <= 2`). Les ventes à 2 lots sont majoritairement un bien + sa dépendance directe (cave, parking) : les inclure récupère ~30 % de données supplémentaires sur les appartements et fait baisser le MAE de ~11 %, sans dégrader la couverture. Les lots plus nombreux (regroupements hétérogènes) restent exclus. *Décision testée et validée par la mesure sur le département 34.*
- Limiter aux types `Maison` et `Appartement`.
- Borner le prix au m² pour couper les artefacts de calcul (faux prix à ~24 000 €/m²).
- Filtrer les aberrations de prix **par type de bien** : maisons et appartements
  ayant des distributions différentes, des bornes calculées sur l'ensemble mélangé
  dégradaient le nettoyage. *Leçon : filtrer chaque population avec ses propres bornes.*
- Corriger un biais DVF : la `surface_terrain` des appartements (parcelle de la
  copropriété entière) est forcée à 0.

### Phase 3 — Analyse exploratoire

- Construire une **matrice de corrélation** pour identifier les variables liées au prix.
- Ne pas se fier qu'aux corrélations linéaires : une variable faiblement corrélée
  (ex. `annee_vente`, masquant un effet de palier réel) peut rester utile au modèle non-linéaire.
- *Inversement, toujours mesurer avant d'intégrer une source : la BD TOPO (type de
  bâtiment), testée ici, s'est révélée sans valeur ajoutée et a été écartée.*

### Phase 4 — Feature engineering

- Features de voisinage : `prix_m2_voisins` (médiane des biens proches), `densite_ventes_1km`.
- Features cadastrales : `prix_m2_section` (médiane par section), `nb_ventes_section`.
- Transformations de surface : `log_surface`, `surface_par_piece`.
- Exploitation du terrain : `surface_terrain`, `log_terrain`, `a_terrain`.
- **Features géographiques décisives** : distances à la mer, au lac et à l'estuaire,
  calculées depuis le fichier littoral. En zone côtière, l'oubli de ces features
  fait chuter le R² des appartements de ~46 % à ~25 %. *Leçon : la qualité des
  features prime sur le réglage du modèle.*
- **Distance à la mairie** (proxy de centralité / centre-ville), la mairie étant
  quasi toujours au cœur historique de la commune. Extraite de l'annuaire de
  l'administration.
- Les voisinages sont calculés **entre biens du même type** (un appartement est
  comparé aux appartements voisins, pas aux maisons).
- **Règle d'or** : toute feature dérivée des prix est calculée **sur le train uniquement**, pour éviter le *data leakage*.

### Phase 5 — Stratégie de validation

- Adopter un **split temporel** (entraîner sur le passé, tester sur l'année récente) plutôt qu'un split aléatoire, car il reflète l'usage réel — prédire des biens futurs. *Décision la plus importante du projet.*
- Ajouter une **cross-validation** pour mesurer la stabilité et détecter l'overfitting via l'écart train/validation.

### Phase 6 — Entraînement du modèle

- Choisir des **arbres boostés** (XGBoost, puis CatBoost) pour leur robustesse sur données tabulaires.
- **XGBoost vs CatBoost** : performances équivalentes sur ce jeu de données, mais
  CatBoost ~5× plus rapide. CatBoost est privilégié pour les réentraînements
  fréquents ou un passage à la France entière.
- Ne pas normaliser les features : inutile pour un modèle à base d'arbres.
- Utiliser l'**early stopping** pour choisir automatiquement le nombre d'arbres.
- Surveiller l'early stopping sur un jeu de **validation issu du train**, jamais sur
  le test, et **en dernière position de l'`eval_set`** : un ordre incorrect fait
  surveiller le train, l'early stopping ne se déclenche jamais, et le modèle
  surapprend (R² appartements divisé par deux). *Bug réel rencontré et corrigé.*

### Phase 7 — Quantification de l'incertitude

- Utiliser la **régression quantile** (modèles 0,025 / 0,50 / 0,975) pour produire
  un intervalle de confiance adaptatif plutôt qu'une prédiction ponctuelle.
- **Pas de correction de Duan** en régression quantile : l'exponentielle préserve
  les quantiles, la correction ne vaut que pour une régression sur la moyenne.
- Valider l'intervalle par la **couverture** : le vrai prix tombe dans l'intervalle
  dans ~90 % des cas (mesuré à ~92-93 % avec les quantiles 0,025 / 0,975).
- *Le R² n'est pas la bonne métrique pour un modèle quantile : il avantage un modèle
  optimisé sur la moyenne (RMSE). Juger un modèle quantile sur le MAE et la couverture.*

### Phase 8 — Évaluation multi-métriques

- Ne jamais juger sur un seul chiffre. Suivre : R² (log et euros), MAE, MAPE, RMSE, erreur médiane, et part des prédictions à ±10 % / ±20 %.
- Comparer les métriques entre elles (RMSE vs MAE, médiane vs moyenne) pour diagnostiquer la nature des erreurs.

### Phase 9 — Interprétation et diagnostic

- Utiliser **SHAP** pour comprendre l'importance et la direction de chaque variable.
- Analyser les **pires erreurs** (par commune, par tranche de prix) pour identifier les segments problématiques et les aberrations résiduelles.

### Phase 10 — Mise en production

- Séparer le fichier d'**évaluation** (qui mesure) du fichier de **production** (qui entraîne sur tout et sauvegarde).
- Créer un fichier d'**estimation** qui géocode une adresse (API BAN) et reconstruit les features, y compris les distances au littoral.
- Exposer l'estimation via une **API REST FastAPI** et une **interface web** simple.
- Veiller à la **cohérence** entre features d'entraînement et features reproductibles
  en production : les trois composants (estimation, serveur, interface) doivent
  partager le même contrat de données.

---

## Optimisation et performance

- **Base de données** : remplacer le filtre `LEFT(code_commune, 2)` (non indexable,
  full scan sur 20 M lignes) par une colonne `code_departement` indexée accélère
  drastiquement les requêtes.
- **Tuning des hyperparamètres** : optimisation bayésienne (Optuna) sur la perte
  pinball du modèle médian ; les meilleurs réglages sont appliqués aux trois
  quantiles (seul `alpha` change).

---

## Limites connues et pistes d'amélioration

- **`prix_m2_section` en production** : l'API d'adresse ne fournit pas la parcelle cadastrale, donc cette feature (parmi les plus importantes) est dégradée en médiane communale. Piste : récupérer la vraie section via l'API Carto de l'IGN.
- **Données individuelles manquantes** : DVF ne contient ni l'état du bien, ni l'étage, ni la vue — facteurs décisifs absents. Le haut de gamme (> 8000 €/m²) reste donc mal prédit, et la fourchette est large sur les appartements.
- **DPE communal et non individuel** : l'appariement DPE↔vente n'atteignait que ~6 %, le profil énergétique est donc agrégé par commune.
- **Revenus INSEE datés de 2021** : à mettre à jour si un millésime plus récent devient disponible.
- **Pistes de features à plus fort potentiel** : distance au centre-ville/mairie
  (centralité), prix des voisins pondéré par distance et tranche de surface,
  tendance de marché locale (pertinente vu la stagnation des prix observée depuis 2023).

---

## Principe transversal

Tout au long du projet, une règle a primé : **préférer une mesure honnête à un
beau chiffre**. La correction du *data leakage* a fait chuter le R² apparent
(de ~83 % à ~50 %), mais le chiffre honnête est celui qui reflète la performance
réelle de l'outil en conditions d'usage. De même, chaque nouvelle source de
données est **testée et mesurée** avant d'être intégrée — ou écartée, comme la
BD TOPO, lorsque les chiffres ne justifient pas le coût.