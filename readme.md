# EstimationIA — Estimation immobilière par Machine Learning

Outil d'estimation du prix au m² d'un bien immobilier à partir de son adresse,
avec intervalle de confiance. Construit sur des données publiques françaises
(DVF, DPE, INSEE) et un modèle XGBoost.

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

### Visualiser une carte HTML

```bash
xdg-open nom_du_fichier.html
```

---


## Sources de données
 
Toutes les données proviennent de [data.gouv.fr](https://www.data.gouv.fr).
 
| Table (base de données) | Source | Rôle | Remarque |
|---|---|---|---|
| `valeurs_foncieres` | [Demandes de valeurs foncières (DVF)](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres) | Cible : prix des ventes | Source principale |
| `dpe_logements_france` | [DPE logements existants](https://www.data.gouv.fr/datasets/dpe-logements-existants-depuis-juillet-2021) | Profil énergétique par commune | Depuis juillet 2021 |
| `donnees-revenus-filosofi` | [Revenu des Français à la commune](https://www.data.gouv.fr/datasets/revenu-des-francais-a-la-commune) | Revenu médian, Gini, minima sociaux | **Données 2021** |
| `adresses_ban` | [Base Adresse Nationale](https://www.data.gouv.fr/datasets/base-adresse-nationale) | Géocodage | Non exploitée dans le pipeline actuel |
| `infrastructures_universites` | [Effectifs étudiants enseignement supérieur](https://www.data.gouv.fr/datasets/effectifs-d-etudiants-inscrits-dans-les-etablissements-et-les-formations-de-l-enseignement-superieur) | Distance + volume étudiants | |
| `donnees_transport` | [Gares ferroviaires de tous types](https://www.data.gouv.fr/datasets/gares-ferroviaires-de-tous-types-exploitees-ou-non) | Distance à la gare | |
| `infrastructures_hopitaux` | [Localisation des hôpitaux (OpenStreetMap)](https://www.data.gouv.fr/datasets/localisation-des-hopitaux-dans-openstreetmap) | Distance à l'hôpital | |
| `monuments_historiques` | [Immeubles protégés au titre des monuments historiques](https://www.data.gouv.fr/datasets/immeubles-proteges-au-titre-des-monuments-historiques-2) | Distance au monument | |
 
### Sources identifiées mais non utilisées
 
- [Référentiel des arrêts — arrêts transporteur](https://www.data.gouv.fr/datasets/referentiel-des-arrets-arrets-transporteur)
- [Dans ma rue — anomalies signalées](https://www.data.gouv.fr/datasets/dans-ma-rue-anomalies-signalees) (Paris)
- [Les commerces par commune ou arrondissement — base permanente des équipements IDF](https://www.data.gouv.fr/datasets/les-commerces-par-commune-ou-arrondissement-base-permanente-des-equipements-idf) (Île-de-France)

---

## Architecture des fichiers

Le projet sépare volontairement trois rôles distincts :

- **Pipeline d'évaluation** — mesure la performance honnête du modèle (split
  temporel, métriques, SHAP, graphes). Ne sauvegarde pas de modèle.
- **Pipeline de production** — réentraîne sur toutes les données et sauvegarde
  les artefacts (`modele_production/`).
- **Estimation** — charge les artefacts et estime un bien depuis son adresse.
- **Analyse des erreurs** — recharge les résultats sauvegardés pour diagnostiquer
  les pires erreurs sans réentraîner.
- **Matrice de corrélation** — analyse exploratoire des liens entre variables.

---

## Liste des décisions de construction du modèle

Les décisions sont présentées dans l'ordre logique de construction. Chacune
répond à un choix concret rencontré pendant le développement.

### Phase 0 — Préparation des données

- Rassembler et joindre les sources hétérogènes (ventes, énergie, socio-économique, géographie).
- Choisir les mailles : le **bien individuel** pour DVF, la **commune** pour les données socio-économiques.
- Définir les clés de jointure : `code_commune` et `code_insee`.

### Phase 1 — Cadrage du problème

- Prédire le **prix au m²** (et non le prix total).
- Appliquer une transformation **logarithmique** au prix pour gérer son asymétrie.
- Entraîner **par département** plutôt que sur la France entière, car les marchés régionaux sont trop différents pour un modèle unique.

### Phase 2 — Nettoyage et filtrage des aberrations

- Ne garder que les ventes **mono-lot** (`nombre_lots <= 1`) pour éviter les prix au m² faussés par des lots multiples.
- Limiter aux types `Maison` et `Appartement`.
- Borner le prix au m² pour couper les artefacts de calcul (faux prix à ~24 000 €/m²).
- Choix retenu : bornes simples après avoir constaté que le filtrage par commune introduisait plus de problèmes qu'il n'en réglait. *Leçon : la solution la plus complexe n'est pas toujours la meilleure.*

### Phase 3 — Analyse exploratoire

- Construire une **matrice de corrélation** pour identifier les variables liées au prix.
- Ne pas se fier qu'aux corrélations linéaires : une variable faiblement corrélée (ex. `annee_vente` à 0,04, masquant un effet de palier réel) peut rester utile au modèle non-linéaire.

### Phase 4 — Feature engineering

- Features de voisinage : `prix_m2_voisins` (médiane des biens proches), `densite_ventes_1km`.
- Features cadastrales : `prix_m2_section` (médiane par section), `nb_ventes_section`.
- Transformations de surface : `log_surface`, `surface_par_piece`.
- Exploitation du terrain : `surface_terrain`, `log_terrain`, `a_terrain`.
- **Règle d'or** : toute feature dérivée des prix est calculée **sur le train uniquement**, pour éviter le *data leakage*.

### Phase 5 — Stratégie de validation

- Adopter un **split temporel** (entraîner sur le passé, tester sur l'année récente) plutôt qu'un split aléatoire, car il reflète l'usage réel — prédire des biens futurs. *Décision la plus importante du projet.*
- Ajouter une **cross-validation** pour mesurer la stabilité et détecter l'overfitting via l'écart train/validation.

### Phase 6 — Entraînement du modèle

- Choisir **XGBoost** pour sa robustesse sur données tabulaires.
- Ne pas normaliser les features : inutile pour un modèle à base d'arbres.
- Utiliser l'**early stopping** pour choisir automatiquement le nombre d'arbres.
- Réguler l'overfitting via `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`.
- Estimer l'early stopping sur un jeu de **validation issu du train**, jamais sur le test.

### Phase 7 — Quantification de l'incertitude

- Utiliser la **régression quantile** (modèles 0,05 / 0,50 / 0,95) pour produire un intervalle de confiance adaptatif plutôt qu'une prédiction ponctuelle.
- Valider l'intervalle par la **couverture** : le vrai prix tombe-t-il dans l'intervalle dans ~90 % des cas ?

### Phase 8 — Évaluation multi-métriques

- Ne jamais juger sur un seul chiffre. Suivre : R² (log et euros), MAE, MAPE, RMSE, erreur médiane, et part des prédictions à ±10 % / ±20 %.
- Comparer les métriques entre elles (RMSE vs MAE, médiane vs moyenne) pour diagnostiquer la nature des erreurs.

### Phase 9 — Interprétation et diagnostic

- Utiliser **SHAP** pour comprendre l'importance et la direction de chaque variable.
- Analyser les **pires erreurs** (par commune, par tranche de prix) pour identifier les segments problématiques et les aberrations résiduelles.

### Phase 10 — Mise en production

- Séparer le fichier d'**évaluation** (qui mesure) du fichier de **production** (qui entraîne sur tout et sauvegarde).
- Créer un fichier d'**estimation** qui géocode une adresse (API BAN) et reconstruit les features.
- Veiller à la **cohérence** entre features d'entraînement et features reproductibles en production.

---

## Limites connues et pistes d'amélioration

- **`prix_m2_section` en production** : l'API d'adresse ne fournit pas la parcelle cadastrale, donc cette feature (la plus importante du modèle) est dégradée en médiane communale. Piste : récupérer la vraie section via l'API Carto de l'IGN.
- **Données individuelles manquantes** : DVF ne contient ni l'état du bien, ni l'étage, ni la vue — facteurs décisifs absents. Le haut de gamme (> 8000 €/m²) reste donc mal prédit.
- **Revenus INSEE datés de 2021** : à mettre à jour si un millésime plus récent devient disponible.
- **Temps d'exécution** : la cross-validation pourrait être déplacée dans un script dédié pour alléger les exécutions courantes.

---

## Principe transversal

Tout au long du projet, une règle a primé : **préférer une mesure honnête à un
beau chiffre**. La correction du *data leakage* a fait chuter le R² apparent
(de ~83 % à ~50 %), mais le chiffre honnête est celui qui reflète la performance
réelle de l'outil en conditions d'usage.
