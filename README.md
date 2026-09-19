# Plateforme Machine Learning automatisée, explicable et planifiée

## 1. Présentation du projet

Ce projet consiste à développer une plateforme **Machine Learning automatisée, générique, modulaire et explicable** pour l’analyse de données tabulaires issues de différentes sources, notamment des fichiers CSV et des bases SQL ou SQLite.

La plateforme permet de :

- importer des données depuis un fichier CSV ou une base SQL / SQLite ;
- détecter automatiquement la structure du dataset ;
- préparer automatiquement les features exploitables ;
- exclure les identifiants et les labels potentiels afin d’éviter le bruit et la fuite d’information ;
- segmenter les observations avec des modèles non supervisés ;
- détecter les anomalies avec Isolation Forest ;
- générer des explications XAI globales et locales ;
- afficher les résultats dans deux dashboards Streamlit ;
- historiser les exécutions dans un Data Warehouse SQLite ;
- sauvegarder les modèles, rapports, artefacts et sessions ;
- déclencher automatiquement le pipeline lorsque le volume de nouvelles données dépasse un seuil défini ;
- synchroniser les nouveaux résultats du scheduler avec le dashboard.

---

## 2. Objectifs techniques

L’objectif principal est de transformer des données tabulaires brutes en résultats analytiques exploitables et compréhensibles.

Le cycle global de traitement est le suivant :

```text
Source CSV / SQL
        ↓
Ingestion des données
        ↓
Détection automatique de structure
        ↓
Préparation automatique des features
        ↓
Machine Learning
        ↓
Explicabilité XAI
        ↓
Dashboards
        ↓
Historisation
        ↓
Planification automatique
```

---

## 3. Fonctionnalités principales

### 3.1 Ingestion des données

- Chargement de fichiers CSV.
- Chargement de bases SQL ou SQLite.
- Exécution de requêtes SQL personnalisées.
- Transformation de la source en DataFrame exploitable.

### 3.2 Préparation automatique des données

- Standardisation des noms de colonnes.
- Suppression des lignes et colonnes entièrement vides.
- Détection automatique des colonnes numériques, catégorielles, temporelles, identifiants et labels potentiels.
- Traitement des valeurs manquantes.
- Traitement des valeurs atypiques.
- Encodage des variables catégorielles.
- Transformation des dates.
- Normalisation et sélection des features.
- Génération du dataset final `X_ready`.

### 3.3 Machine Learning

- Segmentation des observations avec HDBSCAN, K-Means ou MiniBatchKMeans.
- Détection d’anomalies avec Isolation Forest.
- Calcul des métriques de clustering.
- Génération des profils de clusters.
- Attribution de scores et niveaux de sévérité aux anomalies.

### 3.4 Explicabilité XAI

- Explication globale des clusters.
- Modèle surrogate basé sur un arbre de décision.
- Mesure de fidélité du surrogate.
- Explication locale des anomalies avec SHAP.
- Méthode DIFFI-like comme solution de repli.
- Explications contrefactuelles.
- Interprétations automatiques des graphiques.
- Recommandations simples pour l’utilisateur.

### 3.5 Dashboards

La plateforme contient deux espaces Streamlit :

#### Dashboard Technique

Il permet de :

- importer les données ;
- lancer la préparation ;
- consulter la structure détectée ;
- visualiser les colonnes exclues ;
- réintégrer certaines features après validation ;
- consulter les rapports techniques ;
- exporter le dataset préparé.

#### Dashboard Analytique

Il permet de :

- lancer l’analyse Machine Learning ;
- consulter les KPIs ;
- visualiser les clusters ;
- analyser les anomalies ;
- consulter les explications XAI ;
- lire les recommandations ;
- suivre l’historique des exécutions ;
- exporter les résultats.

### 3.6 Data Warehouse

Un Data Warehouse SQLite permet d’historiser :

- les runs de préparation ;
- les runs Machine Learning ;
- les métriques de clustering ;
- les résultats d’anomalies ;
- les profils de clusters ;
- les chemins des modèles, rapports et artefacts.

### 3.7 Scheduler automatique

Le scheduler est un processus indépendant du dashboard Streamlit.

Il permet de :

- surveiller périodiquement la source de données ;
- initialiser un volume de référence ;
- calculer le taux de croissance des données ;
- déclencher automatiquement le pipeline si le seuil configuré est dépassé ;
- sauvegarder l’état dans `scheduler_state.json` ;
- mettre à jour les logs dans `scheduler.log`.

---

## 4. Structure du projet

```text
projet_stage_sqli/
│
├── app/
│   ├── dashboard.py
│   └── pages/
│       ├── 1_Dashboard_Analytique.py
│       └── ...
│
├── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── structure_detector.py
│   ├── ml_controller.py
│   ├── clustering.py
│   ├── anomaly.py
│   ├── supervised.py
│   ├── interpretation.py
│   ├── chart_interpretation.py
│   ├── warehouse.py
│   ├── session_store.py
│   ├── scheduler_service.py
│   └── ui_theme.py
│
├── data/
│   ├── raw/
│   └── warehouse/
│
├── models/
│   └── scheduler_state.json
│
├── reports/
│
├── logs/
│   └── scheduler.log
│
├── diagrams/
│
├── requirements.txt
├── scheduler_config.json
├── scheduler_config.example.json
├── README.md
├── DEPLOYMENT.md
├── Dockerfile
└── docker-compose.yml
```

---

## 5. Technologies utilisées

| Catégorie | Technologies |
|---|---|
| Langage principal | Python |
| Interface utilisateur | Streamlit |
| Manipulation de données | pandas, NumPy |
| Machine Learning | scikit-learn, HDBSCAN |
| Détection d’anomalies | Isolation Forest |
| Explicabilité | SHAP, surrogate model, DIFFI-like |
| Visualisation | Plotly |
| Base de données | SQLite |
| Historisation | Data Warehouse SQLite |
| Planification | APScheduler |
| Sauvegarde d’objets | Joblib, JSON, Parquet |
| Déploiement préparé | Docker, Docker Compose |

---

## 6. Installation locale

### 6.1 Prérequis

Installer les éléments suivants :

- Python 3.10 ou version supérieure ;
- pip ;
- Git ;
- VS Code, recommandé.

### 6.2 Cloner ou récupérer le projet

```powershell
cd "C:\Users\Nour Maghraoui\OneDrive\Desktop"
```

Puis placer le projet dans le dossier souhaité.

### 6.3 Créer un environnement virtuel

```powershell
python -m venv .venv
```

Activer l’environnement :

```powershell
.venv\Scripts\activate
```

### 6.4 Installer les dépendances

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si certaines dépendances sont manquantes, installer les bibliothèques principales :

```powershell
python -m pip install numpy pandas scikit-learn streamlit plotly shap hdbscan apscheduler pyarrow joblib
```

---

## 7. Lancement de l’application

Depuis la racine du projet :

```powershell
python -m streamlit run app/dashboard.py
```

L’application sera accessible dans le navigateur à l’adresse :

```text
http://localhost:8501
```

---

## 8. Utilisation du Dashboard Technique

1. Ouvrir le Dashboard Technique.
2. Choisir le type de source : CSV ou SQL / SQLite.
3. Importer la source de données.
4. Pour une base SQLite, utiliser une requête comme :

```sql
SELECT *
FROM transactions;
```

5. Exécuter la requête.
6. Vérifier l’aperçu des données.
7. Lancer la préparation automatique.
8. Vérifier la génération du dataset `X_ready`.
9. Consulter les colonnes exclues, rapports et artefacts.

---

## 9. Utilisation du Dashboard Analytique

1. Ouvrir le Dashboard Analytique.
2. Vérifier que `X_ready` est disponible.
3. Configurer les paramètres de clustering, anomalies et XAI.
4. Lancer l’analyse Machine Learning.
5. Consulter :
   - le résumé global ;
   - les clusters ;
   - les anomalies ;
   - les visualisations ;
   - les explications XAI ;
   - les recommandations ;
   - les exports ;
   - l’historique des runs.

---

## 10. Configuration du scheduler

Le scheduler utilise le fichier :

```text
scheduler_config.json
```

Exemple de configuration :

```json
{
  "source": {
    "type": "sql",
    "path": "data/raw/credit_card_test.db",
    "query": "SELECT * FROM transactions"
  },

  "integration_config": null,
  "manual_features_to_keep": [],
  "allow_high_risk_features": false,

  "ml_config": {
    "enable_clustering": true,
    "enable_anomaly_detection": true,
    "use_hdbscan": true,
    "use_silhouette_fallback": true,
    "default_k": 3,
    "max_k": 5,
    "min_cluster_size": 10,
    "contamination": "auto",
    "n_estimators": 100,
    "random_state": 42
  },

  "interpretation_config": {
    "surrogate_max_depth": 4,
    "compute_stability": false,
    "anomaly_explanation_top_n": 10
  },

  "schedule": {
    "weekly": {
      "enabled": false,
      "day_of_week": "sun",
      "hour": 2,
      "minute": 0
    },

    "monthly": {
      "enabled": false,
      "day": 1,
      "hour": 3,
      "minute": 0
    }
  },

  "data_change_trigger": {
    "enabled": true,
    "threshold_pct": 10,
    "check_interval_minutes": 1
  },

  "models_base_dir": "models",
  "reports_base_dir": "reports",
  "state_path": "models/scheduler_state.json",
  "log_path": "logs/scheduler.log",
  "timezone": "Africa/Tunis"
}
```

---

## 11. Lancement du scheduler

Le scheduler doit être lancé dans un terminal séparé :

```powershell
python -u -m src.scheduler_service
```

Le service :

- lit `scheduler_config.json` ;
- initialise la référence volumétrique ;
- vérifie périodiquement la source de données ;
- déclenche automatiquement le pipeline si le seuil est dépassé ;
- met à jour `models/scheduler_state.json` ;
- écrit les logs dans `logs/scheduler.log`.

---

## 12. Déclenchement automatique

Le scheduler calcule le taux de croissance des données :

```text
Taux de croissance =
(volume actuel - volume de référence) / volume de référence × 100
```

Exemple de validation :

```text
Volume de référence : 100 lignes
Volume actuel : 150 lignes
Nouvelles lignes : 50
Taux de croissance : 50 %
Seuil configuré : 10 %
Résultat : déclenchement automatique
```

La condition de déclenchement est :

```text
taux de croissance > seuil
```

---

## 13. Fichier d’état du scheduler

Le scheduler sauvegarde son état dans :

```text
models/scheduler_state.json
```

Exemple :

```json
{
  "last_run_id": "auto_20260729_231023",
  "last_run_status": "success",
  "last_row_count": 150,
  "current_row_count": 150,
  "last_growth_pct": 0.0,
  "last_trigger_reason": "nouvelles_donnees_50.0pct"
}
```

---

## 14. Synchronisation scheduler-dashboard

Le scheduler et le dashboard Streamlit sont deux processus indépendants.

Le scheduler ne peut pas modifier directement :

```python
st.session_state
```

La synchronisation repose donc sur :

```text
scheduler_state.json
+ session sauvegardée
+ restauration automatique dans Streamlit
```

Fonctionnement :

```text
Scheduler termine un run
        ↓
Sauvegarde les résultats
        ↓
Met à jour scheduler_state.json
        ↓
Dashboard détecte un nouveau run_id
        ↓
Dashboard restaure la session sauvegardée
        ↓
Dashboard actualise les graphiques
```

---

## 15. Tests principaux

### 15.1 Import CSV

Objectif : vérifier que le dashboard peut charger un fichier CSV.

Résultat attendu : données chargées et affichées.

### 15.2 Import SQLite

Objectif : vérifier que le dashboard peut charger une base SQLite avec une requête SQL.

Requête utilisée :

```sql
SELECT *
FROM transactions;
```

### 15.3 Préparation des données

Objectif : vérifier la génération du dataset `X_ready`.

Résultat attendu : dataset préparé généré avec succès.

### 15.4 Machine Learning

Objectif : vérifier la génération des clusters et des anomalies.

Résultat attendu : clusters, anomalies, métriques et rapports générés.

### 15.5 XAI

Objectif : vérifier la génération des explications globales et locales.

Résultat attendu : règles surrogate, explications SHAP ou DIFFI-like et recommandations disponibles.

### 15.6 Scheduler

Scénario :

```text
100 lignes initiales
+ 50 nouvelles lignes
= 150 lignes
```

Résultat attendu : déclenchement automatique du pipeline.

### 15.7 Synchronisation

Objectif : vérifier que le dashboard détecte le nouveau run automatique.

Résultat attendu : dashboard mis à jour automatiquement après sauvegarde du nouveau run.

---

## 16. Commandes utiles

### Vérifier la syntaxe d’un fichier Python

```powershell
python -m py_compile "src\scheduler_service.py"
```

```powershell
python -m py_compile "app\pages\1_Dashboard_Analytique.py"
```

### Lancer le dashboard

```powershell
python -m streamlit run app/dashboard.py
```

### Lancer le scheduler

```powershell
python -u -m src.scheduler_service
```

### Lire l’état du scheduler

```powershell
Get-Content "models\scheduler_state.json" -Raw
```

### Lire les logs du scheduler

```powershell
Get-Content "logs\scheduler.log" -Tail 100
```

### Vérifier une base SQLite

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/raw/credit_card_test.db'); print(c.execute('SELECT COUNT(*) FROM transactions').fetchone()[0]); c.close()"
```

---

## 17. Préparation au déploiement

Le déploiement complet n’est pas obligatoire dans le cadre de ce livrable, mais le projet est préparé pour être déployé facilement.

L’architecture recommandée repose sur deux services :

```text
Service 1 : Dashboard Streamlit
Service 2 : Scheduler indépendant
```

Les deux services doivent partager les dossiers suivants :

```text
data/
models/
reports/
logs/
```

---

## 18. Déploiement Docker proposé

### 18.1 Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/models /app/reports /app/logs

EXPOSE 8501

CMD [
    "python",
    "-m",
    "streamlit",
    "run",
    "app/dashboard.py",
    "--server.address=0.0.0.0",
    "--server.port=8501"
]
```

### 18.2 docker-compose.yml

```yaml
services:
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile

    container_name: ml_dashboard

    command:
      - python
      - -m
      - streamlit
      - run
      - app/dashboard.py
      - --server.address=0.0.0.0
      - --server.port=8501

    ports:
      - "8501:8501"

    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
      - ./logs:/app/logs
      - ./scheduler_config.json:/app/scheduler_config.json

    restart: unless-stopped

  scheduler:
    build:
      context: .
      dockerfile: Dockerfile

    container_name: ml_scheduler

    command:
      - python
      - -u
      - -m
      - src.scheduler_service

    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
      - ./logs:/app/logs
      - ./scheduler_config.json:/app/scheduler_config.json

    restart: unless-stopped
```

---

## 19. Commandes Docker proposées

Construire les images :

```powershell
docker compose build
```

Lancer les services :

```powershell
docker compose up -d
```

Vérifier les services :

```powershell
docker compose ps
```

Consulter les logs :

```powershell
docker compose logs -f
```

Arrêter les services :

```powershell
docker compose down
```

---

## 20. Limites connues

- SQLite est adapté au prototype et aux démonstrations, mais PostgreSQL serait préférable pour un environnement multi-utilisateur.
- L’authentification utilisateur n’est pas encore intégrée.
- Le déploiement complet sur serveur n’a pas été réalisé.
- Les très grands volumes peuvent nécessiter des optimisations supplémentaires.
- Certaines explications XAI dépendent de la disponibilité de SHAP.
- Les explications contrefactuelles sont fournies comme aide à l’analyse et non comme décision automatique.

---

## 21. Perspectives d’amélioration

- Ajouter une authentification et une gestion des rôles.
- Migrer le Data Warehouse vers PostgreSQL.
- Ajouter des tests automatisés avec `pytest`.
- Ajouter une interface d’administration du scheduler.
- Améliorer la gestion des grands datasets.
- Mettre en place une API REST.
- Déployer la solution avec Docker Compose.
- Ajouter un système d’alertes par email ou notification.
- Intégrer un assistant conversationnel pour interroger les résultats.

---


## 22. Conclusion

Cette plateforme constitue une solution complète pour l’analyse automatique, explicable et historisée de données tabulaires. Elle combine un pipeline de préparation générique, des modèles Machine Learning non supervisés, une couche XAI, deux dashboards, un Data Warehouse, une persistance des sessions et un scheduler indépendant.

Le projet est fonctionnel en environnement local et préparé pour un futur déploiement grâce aux fichiers de configuration, à la documentation technique et à l’architecture Docker proposée.
#   a u t o m a t e d - m l - x a i - p l a t f o r m  
 