# Guide de déploiement

## Plateforme Machine Learning automatisée, explicable et planifiée

Ce document décrit la procédure de préparation et de déploiement de la plateforme Machine Learning. Le déploiement complet n’est pas obligatoire dans le périmètre actuel du stage, mais ce guide permet de rendre le projet prêt à être déployé facilement par la suite.

---

## 1. Objectif du déploiement

Le déploiement a pour objectif de rendre l’application accessible en dehors de l’environnement de développement local.

Il permet notamment de :

- lancer le Dashboard Streamlit sur un serveur ;
- maintenir le scheduler actif indépendamment du dashboard ;
- conserver les données, modèles, rapports et logs dans des volumes persistants ;
- permettre à d’autres utilisateurs d’accéder à l’application via un navigateur ;
- faciliter la reproduction de l’environnement technique ;
- préparer une future mise en production.

---

## 2. Architecture de déploiement proposée

L’architecture recommandée repose sur deux services indépendants :

```text
Service 1 : Dashboard Streamlit
Service 2 : Scheduler automatique
```

Le dashboard et le scheduler partagent les mêmes dossiers persistants :

```text
data/
models/
reports/
logs/
```

### Schéma logique

```text
Utilisateur
    ↓
Dashboard Streamlit
    ↓
Dossiers partagés : data, models, reports, logs
    ↑
Scheduler automatique
    ↑
Source CSV ou SQL / SQLite
```

---

## 3. Services à déployer

### 3.1 Service Dashboard

Le service Dashboard exécute l’application Streamlit :

```powershell
python -m streamlit run app/dashboard.py
```

Ce service permet de :

- importer les données ;
- lancer la préparation ;
- lancer l’analyse Machine Learning ;
- afficher les clusters et anomalies ;
- consulter les explications XAI ;
- consulter l’historique ;
- suivre le statut du scheduler.

### 3.2 Service Scheduler

Le service Scheduler exécute le planificateur automatique :

```powershell
python -u -m src.scheduler_service
```

Ce service permet de :

- surveiller la source de données ;
- initialiser une référence volumétrique ;
- détecter l’arrivée de nouvelles données ;
- déclencher automatiquement le pipeline ;
- mettre à jour `scheduler_state.json` ;
- écrire les journaux dans `scheduler.log`.

---

## 4. Prérequis

Avant de déployer le projet, il faut disposer de :

- Python 3.10 ou version supérieure ;
- pip ;
- Git ;
- les fichiers du projet ;
- un fichier `requirements.txt` propre ;
- un fichier `scheduler_config.json` configuré ;
- les dossiers `data`, `models`, `reports` et `logs` ;
- Docker et Docker Compose si le déploiement Docker est utilisé.

---

## 5. Préparation du projet

### 5.1 Vérifier la structure minimale

La structure minimale attendue est :

```text
projet_stage_sqli/
│
├── app/
├── src/
├── data/
├── models/
├── reports/
├── logs/
├── requirements.txt
├── scheduler_config.json
├── scheduler_config.example.json
├── README.md
├── DEPLOYMENT.md
├── Dockerfile
└── docker-compose.yml
```

### 5.2 Créer les dossiers persistants

Si les dossiers n’existent pas, les créer :

```powershell
New-Item -ItemType Directory -Force "data"
New-Item -ItemType Directory -Force "data\raw"
New-Item -ItemType Directory -Force "data\warehouse"
New-Item -ItemType Directory -Force "models"
New-Item -ItemType Directory -Force "reports"
New-Item -ItemType Directory -Force "logs"
```

---

## 6. Installation locale avant déploiement

### 6.1 Créer un environnement virtuel

```powershell
python -m venv .venv
```

### 6.2 Activer l’environnement virtuel

```powershell
.venv\Scripts\activate
```

### 6.3 Installer les dépendances

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6.4 Vérifier les dépendances

```powershell
python -m pip check
```

---

## 7. Configuration du scheduler

Le scheduler utilise le fichier :

```text
scheduler_config.json
```

Exemple minimal :

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

### Remarques importantes

- Le type de source SQL / SQLite doit être déclaré avec :

```json
"type": "sql"
```

- Les chemins doivent rester relatifs pour faciliter le déploiement :

```text
data/raw/source.db
models/scheduler_state.json
logs/scheduler.log
```

- Il faut éviter les chemins personnels comme :

```text
C:\Users\NomUtilisateur\...
```

---

## 8. Lancement local en mode deux terminaux

Cette méthode est adaptée pour les tests et les démonstrations.

### 8.1 Terminal 1 : lancer le dashboard

```powershell
python -m streamlit run app/dashboard.py
```

Adresse locale :

```text
http://localhost:8501
```

### 8.2 Terminal 2 : lancer le scheduler

```powershell
python -u -m src.scheduler_service
```

Le scheduler doit rester actif dans ce terminal.

---

## 9. Déploiement avec Docker

Le déploiement Docker permet de lancer le dashboard et le scheduler dans deux services séparés, avec des volumes partagés.

---

## 10. Dockerfile proposé

Créer un fichier `Dockerfile` à la racine du projet :

```dockerfile
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/data/raw /app/data/warehouse /app/models /app/reports /app/logs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

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

---

## 11. docker-compose.yml proposé

Créer un fichier `docker-compose.yml` à la racine du projet :

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

    environment:
      PYTHONUNBUFFERED: "1"

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

    environment:
      PYTHONUNBUFFERED: "1"

    restart: unless-stopped
```

---

## 12. Commandes Docker

### 12.1 Construire les images

```powershell
docker compose build
```

### 12.2 Lancer les services

```powershell
docker compose up -d
```

### 12.3 Vérifier les services

```powershell
docker compose ps
```

Résultat attendu :

```text
ml_dashboard   running
ml_scheduler   running
```

### 12.4 Consulter les logs

Tous les logs :

```powershell
docker compose logs -f
```

Logs du dashboard :

```powershell
docker compose logs -f dashboard
```

Logs du scheduler :

```powershell
docker compose logs -f scheduler
```

### 12.5 Arrêter les services

```powershell
docker compose down
```

---

## 13. Accès à l’application déployée

Après lancement avec Docker Compose, l’application est accessible à l’adresse :

```text
http://localhost:8501
```

Sur un serveur distant, elle peut être accessible via :

```text
http://ADRESSE_DU_SERVEUR:8501
```

---

## 14. Volumes persistants

Les volumes suivants doivent être conservés :

```text
data/
models/
reports/
logs/
```

Rôle de chaque dossier :

| Dossier | Rôle |
|---|---|
| `data/` | Contient les données sources et le Data Warehouse. |
| `models/` | Contient les modèles, artefacts et l’état du scheduler. |
| `reports/` | Contient les rapports générés par les runs. |
| `logs/` | Contient les journaux d’exécution. |

---

## 15. Vérification après déploiement

### 15.1 Vérifier le dashboard

Ouvrir :

```text
http://localhost:8501
```

Contrôler que :

- le dashboard se charge correctement ;
- les pages Streamlit sont accessibles ;
- l’import CSV ou SQL fonctionne ;
- la préparation des données fonctionne ;
- les résultats peuvent être affichés.

### 15.2 Vérifier le scheduler

Consulter les logs :

```powershell
docker compose logs -f scheduler
```

Le scheduler doit afficher une initialisation ou une vérification volumétrique.

### 15.3 Vérifier le fichier d’état

Sur l’hôte :

```powershell
Get-Content "models\scheduler_state.json" -Raw
```

Le fichier doit contenir le volume de référence, le statut et éventuellement le dernier run.

---

## 16. Test de déclenchement automatique

Scénario de validation :

```text
Volume initial : 100 lignes
Ajout : 50 lignes
Volume final : 150 lignes
Croissance : 50 %
Seuil : 10 %
```

Résultat attendu :

```text
Le scheduler déclenche automatiquement le pipeline.
```

Le calcul utilisé est :

```text
(150 - 100) / 100 × 100 = 50 %
```

La condition est :

```text
50 % > 10 %
```

---

## 17. Sécurité et recommandations

Pour une production réelle, il est recommandé de :

- ajouter une authentification utilisateur ;
- protéger l’accès au dashboard ;
- placer l’application derrière un reverse proxy comme Nginx ou Traefik ;
- activer HTTPS ;
- ne pas exposer directement le port 8501 sur Internet ;
- stocker les secrets dans un fichier `.env` ou dans un gestionnaire de secrets ;
- ne jamais versionner les tokens, mots de passe ou clés API ;
- sauvegarder régulièrement les dossiers persistants ;
- surveiller les logs du scheduler.

---

## 18. Limites de la configuration actuelle

- SQLite convient pour un prototype, une démonstration ou un faible volume d’écriture.
- Pour une production multi-utilisateur, PostgreSQL est préférable.
- Le scheduler doit rester actif pour que les exécutions automatiques fonctionnent.
- Le dashboard et le scheduler doivent partager les mêmes volumes.
- Les grands volumes de données peuvent nécessiter une optimisation supplémentaire.
- L’authentification et la gestion des rôles ne sont pas encore intégrées.

---

## 19. Dépannage

### 19.1 Le dashboard ne démarre pas

Vérifier :

```powershell
python -m py_compile "app\dashboard.py"
```

Puis vérifier les dépendances :

```powershell
python -m pip check
```

### 19.2 Le scheduler ne démarre pas

Vérifier :

```powershell
python -m py_compile "src\scheduler_service.py"
```

Puis lancer :

```powershell
python -u -m src.scheduler_service
```

### 19.3 `scheduler_state.json` n’existe pas

Vérifier que le scheduler est bien lancé.

Vérifier aussi que la source configurée existe et que `scheduler_config.json` est valide.

### 19.4 Le type de source n’est pas supporté

Pour une base SQLite, utiliser :

```json
"type": "sql"
```

et non :

```json
"type": "sqlite"
```

### 19.5 Le dashboard n’affiche pas le dernier run

Vérifier :

- que `scheduler_state.json` est mis à jour ;
- que les sessions sont sauvegardées ;
- que le dashboard lit le bon chemin ;
- que les dossiers `models/` et `data/` sont partagés entre le dashboard et le scheduler.

---

## 20. Conclusion

Cette procédure permet de préparer le projet pour un déploiement futur sans réaliser une mise en production complète. L’architecture recommandée sépare le dashboard Streamlit et le scheduler automatique en deux services indépendants partageant les mêmes volumes persistants.

Cette organisation facilite la maintenance, la traçabilité, la reprise des résultats, la surveillance automatique des sources de données et la future mise en production de la plateforme.
