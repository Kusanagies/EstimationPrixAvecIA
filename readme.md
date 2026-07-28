# EstimationIA — Estimation immobilière par Machine Learning

Outil d'estimation du prix au m² d'un bien immobilier à partir de son adresse,
avec intervalle de confiance. Construit sur des données publiques françaises
(DVF, DPE, INSEE) et des modèles à base d'arbres boostés (XGBoost / CatBoost).

---

## Arborescence du projet

```
EstimationIA/
├── carte/                  Code HTML des cartes (heatmaps) generees
├── data/                   Fichiers de donnees (.csv, .gpkg, .json)
├── env/                    Environnement virtuel (dependances du projet)
├── modele_production/      Artefacts du modele entraine (.cbm/.json, features.json, contexte_*.pkl)
├── out/                    Sorties : graphes (SHAP, correlation...) et resultats (.pkl)
├── src/                    Code source
│   ├── genCarte/           Generation des cartes .html (heatmaps)
│   ├── graphe/             Generation des graphes (correlation, SHAP, etc.)
│   ├── ML/                 Tout le code Machine Learning (pipelines, entrainement, estimation, API)
│   └── script/             Scripts d'import des fichiers CSV vers les tables SQL
├── .env                    Mot de passe de la base (non versionne)
├── .gitignore
├── README.md
└── requirements.txt
```

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
| `synthese` (base `etalab_dvf`) | Base DVF relationnelle pré-traitée (Etalab) | **Source principale actuelle** : ventes pré-agrégées | ~9,4 M lignes, 2014-2025, propre (coords + prix_m2 déjà calculés) |
| `valeurs_foncieres` | [Demandes de valeurs foncières (DVF)](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres) | Cible : prix des ventes | Source historique (~20 M lignes), remplacée par `synthese` |
| `dpe_logements_france` | [DPE logements existants](https://www.data.gouv.fr/datasets/dpe-logements-existants-depuis-juillet-2021) | Profil énergétique par commune | Depuis juillet 2021 |
| `donnees-revenus-filosofi` | [Revenu des Français à la commune](https://www.data.gouv.fr/datasets/revenu-des-francais-a-la-commune) | Revenu médian, Gini, minima sociaux | **Données 2021** |
| `adresses_ban` | [Base Adresse Nationale](https://www.data.gouv.fr/datasets/base-adresse-nationale) | Géocodage | Non exploitée dans le pipeline actuel |
| `infrastructures_universites` | [Effectifs étudiants enseignement supérieur](https://www.data.gouv.fr/datasets/effectifs-d-etudiants-inscrits-dans-les-etablissements-et-les-formations-de-l-enseignement-superieur) | Distance + volume étudiants | |
| `donnees_transport` | [Gares ferroviaires de tous types](https://www.data.gouv.fr/datasets/gares-ferroviaires-de-tous-types-exploitees-ou-non) | Distance à la gare | |
| `infrastructures_hopitaux` | [Localisation des hôpitaux (OpenStreetMap)](https://www.data.gouv.fr/datasets/localisation-des-hopitaux-dans-openstreetmap) | Distance à l'hôpital | |
| `monuments_historiques` | [Immeubles protégés au titre des monuments historiques](https://www.data.gouv.fr/datasets/immeubles-proteges-au-titre-des-monuments-historiques-2) | Distance au monument | |
| `infrastructures_mairies` | [Annuaire de l'administration (service-public.gouv.fr)](https://www.data.gouv.fr/datasets/lannuaire-de-ladministration-base-de-donnees-locales) | Distance à la mairie (proxy centre-ville) | Filtré sur `type_service_local = mairie` ; ~35 000 communes |
| `TableGeo2022.gpkg` (fichier local) | [Communes de la loi Littoral au COG 2020-2022](https://www.data.gouv.fr/datasets/communes-de-la-loi-littoral-au-code-officiel-geographique-cog-2020-2022) | Distance à la mer / lac / estuaire | Colonne `CLASSEMENT` (Mer/Lac/Estuaire) ; features décisives en zone côtière |
| `referentiel_communes` | [Référentiel géographique français (communes, aires urbaines...)](https://www.data.gouv.fr/datasets/referentiel-geographique-francais-communes-unites-urbaines-aires-urbaines-departements-academies-regions-1) | Aires d'attraction des villes → potentiel urbain | Poids d'un pôle = nombre de communes de son aire |
| `chomage_departements` | [Taux de chômage localisé (INSEE)](https://www.insee.fr/fr/statistiques/serie/001515842) | Taux de chômage par département et trimestre | **En cours d'évaluation** — varie géographiquement (voir note ci-dessous) |
| `densite_population` | [Densité de population (INSEE)](https://www.data.gouv.fr/datasets/densite-de-population-1) | Densité communale (hab/km²) | Testée puis écartée sur mono-département (voir sources écartées) |

### Sources testées mais écartées

- **BD TOPO de l'IGN (type de bâtiment)** : testée sur le département 34
  (nombre d'étages, nombre de logements, hauteur). Corrélations avec le prix
  quasi nulles (< 0,08 en valeur absolue), attributs souvent mal renseignés.
  Écartée après mesure : gain négligeable pour un coût d'intégration élevé.
- **Distance à la mairie** (proxy de centralité) : testée sur les départements 34
  et 69. Corrélation quasi nulle en intérieur (−0,065 sur Lyon) et brouillée par
  l'effet littoral sur le 34. Déjà captée par des proxies plus efficaces (distance
  à l'hôpital, densité de ventes). Écartée ; le potentiel urbain (voir Phase 4)
  s'est révélé bien plus pertinent pour l'influence des villes.
- **Taux d'intérêt (crédit immobilier, inflation)** : taux mensuels nationaux
  (crédit immo à taux fixe et variable, inflation, taux entreprises PME/ETI/GE),
  joints aux ventes par année-mois. Testés sur les départements 34 et 01.
  Corrélation avec le prix quasi nulle (< 0,06), MAIS **redondance quasi totale
  avec `annee_vente`** (taux de crédit corrélé à +0,86 avec l'année). L'effet
  économique est réel (taux hauts → capacité d'achat réduite → prix qui stagnent),
  mais il est purement **temporel** : le modèle le capte déjà via `annee_vente`.
  Écartés car ils n'ajoutent aucune information au-delà de l'année. Les taux
  entreprises (PME/ETI/GE) sont de plus redondants avec le taux immobilier.
- **Pyramide des âges (par département)** : parts des tranches d'âge (dont
  `pct_60_plus`, `pct_20_39`) issues des estimations de population INSEE, jointes
  par département et année. Testée sur les départements 34 et 01. **Corrélation
  de +0,98 à +0,997 avec `annee_vente`** : sur un seul département, la structure
  d'âge ne fait qu'évoluer lentement dans le temps, donc elle se confond avec
  l'année. Corrélation avec le prix négligeable (< 0,05). Écartée pour la même
  raison que les taux : redondante avec l'effet temporel déjà capté. *(Pourrait
  avoir plus de sens à l'échelle France entière, où des départements âgés se
  distinguent de départements jeunes — non retenue à ce stade.)*
- **PIB national** : produit intérieur brut annuel de la France (comptes nationaux
  INSEE, 1949-2024). **Corrélation de +0,94 avec `annee_vente`** sur la période DVF,
  et surtout **aucune variation géographique** : le PIB est le même pour tous les
  départements une année donnée. C'est le cas le plus extrême de feature purement
  temporelle — il ne peut pas discriminer les biens entre eux. Écarté. *(Un PIB
  régional ou par habitant, qui distinguerait les territoires, serait en revanche
  potentiellement intéressant — comme l'est déjà le revenu médian communal.)*
- **Inflation, testée sous trois formes** : glissante (12 mois, corr. +0,49 avec
  l'année), instantanée (variation mensuelle, corr. −0,06), et cumulée en indice de
  prix / déflateur (corr. +0,92). **Aucune n'améliore le modèle** (gains de +0,0005
  à +0,0010 de RMSLE, soit une légère dégradation). Enseignement : une feature utile
  doit être à la fois *non redondante avec l'année* ET *corrélée au prix*. L'indice
  cumulé est redondant (0,92) ; l'instantanée est non redondante mais sans lien avec
  le prix immobilier (bruit). L'inflation générale ne prédit pas le prix immobilier,
  quel que soit son encodage. *Écartée définitivement.*
- **Densité de population communale** (INSEE, habitants/km²) : testée sur le
  département 34 (ajoutée à la baseline). **Gain quasi nul (+0,0012 de RMSLE)**.
  Sur un seul département, la localisation (lat/lon) capture déjà l'effet urbain/rural,
  rendant la densité redondante. *(Comme le chômage, son intérêt potentiel ne pourrait
  se révéler qu'à l'échelle France entière, où elle distinguerait métropoles et zones
  rurales sur tout le territoire. Non retenue à ce stade sur mono-département.)*
- [Référentiel des arrêts — arrêts transporteur](https://www.data.gouv.fr/datasets/referentiel-des-arrets-arrets-transporteur)
- [Dans ma rue — anomalies signalées](https://www.data.gouv.fr/datasets/dans-ma-rue-anomalies-signalees) (Paris)
- [Les commerces par commune ou arrondissement — base permanente des équipements IDF](https://www.data.gouv.fr/datasets/les-commerces-par-commune-ou-arrondissement-base-permanente-des-equipements-idf) (Île-de-France)

### Source en cours d'évaluation : taux de chômage départemental

Contrairement aux taux d'intérêt, au PIB et à la pyramide des âges (tous écartés
car purement temporels), le **taux de chômage par département** possède une
**dimension géographique** : il varie fortement d'un département à l'autre (par
exemple ~5-6 % dans l'Ain contre ~12-14 % dans l'Aisne sur les mêmes périodes).
Cette variation entre territoires est précisément ce qui peut apporter du signal,
comme le fait déjà le revenu médian communal.

Point de méthode important : sur un modèle **mono-département**, le chômage ne
varie que dans le temps (une seule série) et redevient donc redondant avec
`annee_vente`. Son intérêt ne peut se révéler qu'à l'échelle **multi-départements /
France entière**, où les écarts entre territoires jouent pleinement. L'évaluation
est donc menée en France entière (matrice de corrélation + SHAP) avant toute
décision d'intégration en production. *Démarche : mesurer dans le bon cadre avant
de conclure.*

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

### Phase 0bis — Migration vers la base `synthese` (Etalab)

- Basculement de `valeurs_foncieres` (DVF brut, ~20 M lignes) vers `etalab_dvf.synthese`,
  base relationnelle pré-traitée par l'organisation : ~9,4 M lignes, 2014-2025, coordonnées
  et prix/m² déjà calculés, une ligne par bien.
- **Investigation avant migration** : exploration comparée (volume, couverture temporelle,
  complétude), et analyse de la gestion des ventes multi-biens. `synthese` démultiplexe déjà
  les ventes complexes (ratio ~1 ligne/vente) → **le filtre `nombre_lots` devient inutile**
  (test A/B/C : configs « brut » et « dépendances ≤ 3 » identiques, config stricte dégrade).
- **Deux connexions** : les ventes viennent de `etalab_dvf`, les enrichissements
  (DPE, revenus, infrastructures) restent dans `EstimationIA`.
- Conversions `pd.to_numeric` obligatoires (colonnes int nullable lues en `object`) ;
  mapping des colonnes (`lat`/`lng`, `typebien`, `communes_code`, `parcelles_code`).

### Phase 0 — Préparation des données

- Rassembler et joindre les sources hétérogènes (ventes, énergie, socio-économique, géographie).
- Choisir les mailles : le **bien individuel** pour DVF, la **commune** pour les données socio-économiques.
- Définir les clés de jointure : `code_commune` et `code_insee`.
- Dédupliquer les ventes DVF sur (`id_parcelle`, `prix_m2`, `surface`).

### Phase 1 — Cadrage du problème

- Prédire le **prix total** (euros) — c'est ce que veut l'utilisateur final. *(Le projet
  prédisait initialement le prix/m² ; un test A/B a montré que les deux cibles sont
  équivalentes en performance avec des arbres boostés — voir Phase 8.)*
- Appliquer une transformation **logarithmique** au prix pour gérer son asymétrie.
- **Séparer les modèles Maisons et Appartements** : leurs distributions de prix et
  leurs facteurs de valeur diffèrent trop pour un modèle unique.

### Phase 2 — Nettoyage et filtrage des aberrations

- *Historique (base DVF brute)* : ne garder que les ventes **à trois lots maximum** (`nombre_lots <= 3`). Les ventes à 2 lots sont majoritairement un bien + sa dépendance directe, au prix/m² cohérent : les inclure récupère ~30 % de données supplémentaires sur les appartements et fait baisser le MAE de ~11 %. *Décision testée sur le 34.* **Devenu inutile avec `synthese`** (déjà démultiplexée — voir Phase 0bis).
- Limiter aux types `Maison` et `Appartement`.
- Borner le prix au m² par **quantiles 0,01 / 0,99 par type de bien** : maisons et
  appartements ayant des distributions différentes, des bornes calculées sur l'ensemble
  mélangé dégradaient le nettoyage. *Leçon : filtrer chaque population avec ses propres bornes.*
- Ajouter un **filtre de cohérence marché** (ratio prix/médiane communale 0,40-2,50) pour
  éliminer les transactions hors marché résiduelles — voir Phase 8bis.
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
- **Potentiel urbain** (modèle de gravité) : influence pondérée des grandes villes,
  calculée comme la somme sur les pôles (aires d'attraction) de `poids / distance`,
  le poids étant la taille de l'aire. Capte l'attractivité des métropoles voisines,
  décisive pour les départements périurbains ou frontaliers. Sur le département 01
  (entre Lyon et Genève), corrélation de +0,20 avec le prix ; inclut des pôles
  étrangers ajoutés manuellement (Genève, Lausanne) pour l'effet transfrontalier.
- Les voisinages sont calculés **entre biens du même type** (un appartement est
  comparé aux appartements voisins, pas aux maisons).
- **Règle d'or** : toute feature dérivée des prix est calculée **sur le train uniquement**, pour éviter le *data leakage*.

### Phase 4bis — Exploration systématique de l'apport des features

Un script d'analyse d'impact (`impact_features.py`) mesure la contribution de chaque
groupe de features ajouté **individuellement à la baseline** (lat/lon + surface + date).
Résultats mesurés sur le département 34 (split aléatoire, cible prix total) :

- **Contributeurs majeurs** : `prix_m2_voisins` (gain ~0,022 de RMSLE), `terrain`
  (~0,041 pour les maisons, 0 pour les appartements — logique, cohérence validée),
  `prix_m2_section` (~0,009).
- **Contributeurs modestes** : distance littoral, distance transport, densité de ventes,
  potentiel urbain, distance hôpital (~0,003 chacun).
- **Apport quasi nul** : revenus, densité de population, pièces, chauffage, DPE,
  dérivées de surface (~0,001 ou moins).
- **Nuisibles (gain négatif)** : PIB, chômage, taux de crédit, taux d'inflation.

**Enseignement clé** : la baseline contenant déjà `lat/lon`, CatBoost reconstruit une
grande partie de la valeur locale à partir des seules coordonnées. Les features élaborées
n'ajoutent donc qu'un apport *marginal* par-dessus (elles restent importantes dans le
modèle complet — SHAP élevé — mais partiellement redondantes avec la localisation brute).
*Plus le modèle est puissant, moins le feature engineering sophistiqué apporte au-delà
des features de base bien choisies.*

### Phase 4ter — Exploration de la dimension temporelle

La dimension temps a été explorée en profondeur pour chercher un meilleur encodage que
`annee_vente` + `mois_vente`. **Toutes les alternatives testées se sont révélées
équivalentes ou inférieures** :

- **Variables économiques en remplacement de la date** (taux de crédit, inflation) :
  testées en split aléatoire ET en généralisation (train < 2025, test = 2025 non vu).
  La date reste meilleure dans les deux cas (écart +0,014 de RMSLE en généralisation
  sur les appartements). L'hypothèse « les variables causales généralisent mieux » est
  *réfutée par la mesure* : ces variables sont trop faiblement corrélées au prix, et
  l'année extrapole bien vers une année adjacente.
- **Encodages alternatifs** (date décimale, mois cyclique sin/cos, mois écoulés, indice
  de marché) : tous équivalents à `annee + mois`. Les arbres captent le temps brut aussi
  bien que les transformations, contrairement aux modèles linéaires.
- **Features temporelles enrichies** (saisonnalité, volume de marché, dynamique locale
  récente calculée sans leakage) : aucune ne bat la référence. La dynamique locale est
  redondante avec la combinaison lat/lon + année.
- **Inflation sous trois formes** (glissante, instantanée, indice cumulé) : aucun apport
  — voir « Sources testées mais écartées ».

*Conclusion : `annee_vente` + `mois_vente` est l'encodage optimal du temps pour ce
modèle. La dimension temporelle apporte un vrai signal (~0,04-0,05 de RMSLE vs sans
temps), mais son encodage brut suffit.*

### Phase 5 — Stratégie de validation

- Le **split est paramétrable** au lancement : aléatoire 70/30 (mélange toutes les
  années) ou temporel (train < dernière année, test = dernière année).
- Le **split temporel** reflète l'usage réel (prédire des biens futurs à partir de
  l'historique) et donne une mesure de généralisation honnête ; le **split aléatoire**
  donne des métriques mécaniquement meilleures car le modèle connaît déjà le niveau de
  prix de chaque année. *Le choix du split est la décision la plus structurante pour
  l'interprétation des chiffres.*
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

- Ne jamais juger sur un seul chiffre. Suivre : R², MAE, MAPE, RMSE, **RMSLE**, erreur médiane, part des prédictions à ±10 % / ±20 % (PE10/PE20), et couverture de l'intervalle.
- **Cible = prix total** (euros), pas prix/m². Un test A/B a montré que les deux cibles
  donnent des résultats quasi identiques (RMSLE 0,2078 vs 0,2079) : les arbres boostés
  reconstruisent la relation quelle que soit la cible, contrairement aux modèles linéaires.
  Le prix total est retenu car c'est ce que veut l'utilisateur final. *Les features de prix
  (voisins, section) restent en €/m² pour rester comparables entre biens ; seule la cible change.*
- **RMSLE** comme métrique de référence : mesure l'erreur relative dans l'espace
  logarithmique (cohérent avec la cible log), robuste aux prix extrêmes, pénalise davantage
  la sous-estimation. Interprétation : `(exp(RMSLE) − 1) × 100` ≈ % d'erreur relative typique.
- Comparer les métriques entre elles (RMSE vs MAE, médiane vs moyenne) pour diagnostiquer la
  nature des erreurs : un écart médiane ≪ moyenne signale des aberrations résiduelles.
- **MAPE trompeuse sur données non filtrées** : quelques ventes hors marché suffisent à la
  faire exploser (moyenne sensible aux extrêmes), alors que le PE20 et l'erreur médiane restent stables.

### Phase 8bis — Filtre de cohérence marché

- Après le filtrage par quantiles départementaux, un **filtre de cohérence marché** compare
  chaque bien au prix médian de **sa commune** (garde-fou : ≥ 10 ventes, sinon repli sur la
  médiane départementale) et exclut les biens hors de la fourchette 40 %-250 % de cette référence.
- Élimine les transactions hors marché (ventes familiales sous-évaluées, parts indivises,
  nue-propriété) que les quantiles laissent passer. Retire ~3-5 % des biens.
- **Effet mesuré** (dép. 34) : kurtosis des erreurs log divisé par ~3 (de ~5 à ~1,7), RMSLE
  en baisse (~−15 %), MAPE assainie, biais de sous-estimation réduit. Appliqué de façon
  identique à l'entraînement, l'évaluation et le test pour la cohérence de la chaîne.

### Phase 8ter — Analyse de normalité des erreurs

- Vérification que les erreurs dans l'espace log suivent (approximativement) une **loi normale** :
  histogramme + courbe théorique, QQ-plot, et vérification empirique de la règle 68-95-99.7.
- Si les erreurs log sont normales, le **RMSLE s'interprète comme l'écart-type** : ~68 % des
  biens dans ±1 RMSLE, ~95 % dans ±2 RMSLE.
- **Résultat mesuré** : après filtre de cohérence marché, les erreurs sont *quasi-normales*
  avec de légères queues épaisses résiduelles (kurtosis ~1,7-2,2 ; ~74 % dans ±1σ, ~94 % dans ±2σ).
  Les queues restantes reflètent les biens atypiques irréductibles (haut de gamme, facteurs
  individuels absents de DVF). *Le kurtosis résiduel explique la MAPE parfois élevée sur données brutes.*

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

- **`prix_m2_section` en production** : l'API d'adresse ne fournit pas la parcelle cadastrale, donc cette feature (parmi les plus importantes) est dégradée en médiane communale. Pistes : récupérer la vraie section via l'API Carto de l'IGN, ou exploiter le **`geo_iris_id`** désormais disponible dans `synthese` (maille IRIS infra-communale, plus fine que la commune).
- **Données individuelles manquantes** : DVF ne contient ni l'état du bien, ni l'étage, ni la vue — facteurs décisifs absents. Le haut de gamme (> 8000 €/m²) reste donc mal prédit, et la fourchette est large sur les appartements.
- **DPE communal et non individuel** : l'appariement DPE↔vente n'atteignait que ~6 %, le profil énergétique est donc agrégé par commune.
- **Revenus INSEE datés de 2021** : à mettre à jour si un millésime plus récent devient disponible.
- **Pistes de features à plus fort potentiel** : vraie section cadastrale via
  l'API Carto de l'IGN (pour ne plus dégrader `prix_m2_section` en médiane
  communale en production), extension du potentiel urbain aux départements
  frontaliers (avec pôles étrangers), et **taux de chômage départemental** (en
  cours d'évaluation — première feature socio-économique temporelle à avoir une
  vraie dimension géographique, contrairement aux taux/PIB écartés). Un **PIB
  régional par habitant** serait dans la même veine prometteuse. *Note : la
  centralité (mairie), les taux d'intérêt, la pyramide des âges et le PIB national
  ont été testés puis écartés — voir « Sources testées mais écartées ».*

---

## Principe transversal

Tout au long du projet, une règle a primé : **préférer une mesure honnête à un
beau chiffre**. La correction du *data leakage* a fait chuter le R² apparent
(de ~83 % à ~50 %), mais le chiffre honnête est celui qui reflète la performance
réelle de l'outil en conditions d'usage. De même, chaque nouvelle source de
données est **testée et mesurée** avant d'être intégrée — ou écartée, comme la
BD TOPO, lorsque les chiffres ne justifient pas le coût.