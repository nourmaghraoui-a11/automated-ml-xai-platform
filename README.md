# 🚀 Automated Machine Learning & Explainable Analytics Platform

## Overview

This project is a modular and automated Machine Learning platform designed to transform raw tabular data into actionable insights through automated data preparation, clustering, anomaly detection, explainable AI (XAI), and interactive analytics dashboards.

The platform supports multiple data sources including CSV files, SQL databases, and SQLite databases. It provides a complete analytical workflow from data ingestion to model interpretation and monitoring.

---

## ✨ Key Features

### 📥 Data Ingestion
- CSV file import
- SQL and SQLite database connectivity
- Custom SQL query execution
- Automatic DataFrame generation

### 🧹 Automated Data Preparation
- Automatic schema detection
- Missing value handling
- Feature engineering
- Categorical encoding
- Date processing
- Feature normalization
- Feature selection

### 🤖 Machine Learning
- HDBSCAN clustering
- K-Means clustering
- MiniBatchKMeans clustering
- Isolation Forest anomaly detection
- Cluster profiling
- Clustering performance metrics

### 🔍 Explainable AI (XAI)
- SHAP-based explanations
- Surrogate Decision Trees
- Global model explanations
- Local anomaly explanations
- Automated recommendations generation

### 📊 Analytics Dashboards
- Technical Dashboard
- Analytical Dashboard
- KPI visualization
- Cluster analysis
- Anomaly investigation
- Historical run tracking

### ⏱️ Scheduler & Monitoring
- Automated pipeline execution
- Data growth detection
- Configurable triggers
- Execution logging
- State persistence

### 💾 Data Warehouse
- Run history management
- Metrics storage
- Model tracking
- Report archiving
- Artifact management

### 🐳 Deployment Ready
- Docker support
- Docker Compose integration
- Modular architecture

---

## 🏗️ Architecture

```text
CSV / SQL / SQLite
          │
          ▼
   Data Ingestion
          │
          ▼
 Data Preparation
          │
          ▼
 Machine Learning
          │
          ▼
 Explainable AI
          │
          ▼
 Streamlit Dashboards
          │
          ▼
 Data Warehouse
          │
          ▼
 Automated Scheduler
```

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-Learn
- HDBSCAN

### Anomaly Detection
- Isolation Forest

### Explainable AI
- SHAP
- Surrogate Models

### Visualization
- Plotly

### Dashboard Development
- Streamlit

### Storage
- SQLite

### Scheduling
- APScheduler

### Deployment
- Docker
- Docker Compose

---

## 📂 Project Structure

```text
project/
│
├── app/
│   ├── dashboard.py
│   └── pages/
│
├── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── structure_detector.py
│   ├── clustering.py
│   ├── anomaly.py
│   ├── interpretation.py
│   ├── warehouse.py
│   ├── scheduler_service.py
│   └── ...
│
├── data/
├── models/
├── reports/
├── logs/
├── diagrams/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/automated-ml-xai-platform.git

cd automated-ml-xai-platform
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

### Launch Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Access the application at:

```text
http://localhost:8501
```

---

## ⏳ Run the Scheduler

```bash
python -u -m src.scheduler_service
```

The scheduler automatically:

- Monitors data growth
- Detects threshold overruns
- Triggers ML pipelines
- Stores execution history
- Updates dashboard data

---

## 📈 Machine Learning Workflow

```text
Raw Data
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Feature Selection
    ↓
Clustering
    ↓
Anomaly Detection
    ↓
Explainable AI
    ↓
Analytics Dashboard
```

---

## 🎯 Skills Demonstrated

This project demonstrates competencies in:

- Data Engineering
- Data Analytics
- Machine Learning
- Explainable AI (XAI)
- Unsupervised Learning
- Dashboard Development
- Software Architecture
- MLOps Fundamentals
- Data Warehousing
- Docker Containerization

---

## 🚀 Future Improvements

- User authentication and role management
- PostgreSQL integration
- REST API development
- Automated testing with PyTest
- Notification system
- Cloud deployment
- Conversational AI assistant
- Monitoring and observability tools

---

## 👨‍💻 Author

**Nour Maghraoui**

Third-Year Student in Big Data & Data Analytics

---

## ⚠️ Disclaimer

This repository contains a demonstration version of the project developed during an internship. Any confidential company data, proprietary assets, or sensitive information have been removed before publication.
