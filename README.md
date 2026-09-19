# 🚀 Automated Explainable Machine Learning Platform
## Transforming Raw Data into Actionable Insights

This project is a complete end-to-end Machine Learning platform designed to automate the entire analytical lifecycle of tabular data.

Unlike traditional Machine Learning projects that focus only on model training, this platform covers every stage of the data pipeline:

✅ Data ingestion from CSV and SQL sources

✅ Automated data preparation and feature engineering

✅ Unsupervised Machine Learning and anomaly detection

✅ Explainable AI (XAI) for model transparency

✅ Interactive analytical dashboards

✅ Historical tracking and persistence

✅ Automated scheduling and retraining triggers

✅ Docker-ready deployment architecture

The objective is to enable non-technical and technical users to transform raw business data into understandable, explainable, and actionable insights with minimal manual intervention.

---

## Why This Project Matters

In many organizations, data analysis is fragmented across multiple tools:

- Data extraction tools
- Data preparation scripts
- Machine Learning notebooks
- Monitoring dashboards
- Manual reporting processes

This platform unifies all these steps into a single modular architecture capable of:

- Detecting new data automatically
- Launching analytical workflows
- Generating machine learning insights
- Explaining model decisions
- Tracking historical executions
- Presenting results through interactive dashboards

The platform follows a real-world Data Science and MLOps approach by combining Data Engineering, Machine Learning, Explainable AI and application development in a single solution.

---

## Key Capabilities

### Automated Data Engineering

The platform automatically:

- Detects dataset structure
- Identifies numerical and categorical features
- Handles missing values
- Encodes categories
- Processes dates
- Selects usable features
- Produces analysis-ready datasets

### Advanced Machine Learning

The analytical engine supports:

- HDBSCAN clustering
- K-Means clustering
- MiniBatchKMeans clustering
- Isolation Forest anomaly detection

The system automatically generates:

- Cluster profiles
- Segmentation metrics
- Risk levels
- Anomaly scores

### Explainable AI (XAI)

One of the strongest aspects of the platform is its explainability layer.

Rather than producing black-box predictions, the platform provides:

- Global cluster explanations
- SHAP-based local explanations
- Surrogate decision trees
- Feature importance analysis
- Automated interpretations
- Actionable recommendations

### Business Intelligence Dashboards

Two dedicated Streamlit environments provide:

#### Technical Dashboard

Designed for data analysts and engineers:

- Data import
- Schema inspection
- Feature validation
- Dataset preparation
- Technical reports

#### Analytical Dashboard

Designed for decision makers:

- KPI monitoring
- Cluster exploration
- Anomaly investigation
- XAI visualizations
- Historical run analysis

### Automated Monitoring & Scheduling

A dedicated scheduler continuously monitors incoming data.

When a configurable threshold is exceeded, the platform automatically:

1. Detects new records
2. Launches preprocessing
3. Executes Machine Learning pipelines
4. Generates reports
5. Updates dashboards
6. Stores results in the warehouse

This simulates a real production analytics workflow.

---

## High-Level Architecture

```text
CSV / SQL Sources
         │
         ▼
 Data Ingestion
         │
         ▼
 Automated Feature Engineering
         │
         ▼
 Machine Learning Engine
         │
         ├──────── Clustering
         │
         ├──────── Anomaly Detection
         │
         ▼
 Explainable AI Layer
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

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- HDBSCAN
- SHAP
- Streamlit
- Plotly
- SQLite
- APScheduler
- Docker
- Docker Compose

---

## Skills Demonstrated

This project demonstrates competencies in:

- Data Engineering
- Machine Learning
- Explainable AI (XAI)
- MLOps Fundamentals
- Data Visualization
- Dashboard Development
- SQL & Data Warehousing
- Software Architecture
- Workflow Automation
- Containerization with Docker

---

## Internship Context

This project was developed during an internship focused on Data Science, Machine Learning and Analytics Automation.

The objective was to design a reusable analytical framework capable of processing multiple datasets while providing transparency, traceability and automation throughout the Machine Learning lifecycle.

---

## Author

**Nour Maghraoui**
Big Data & Data Analytics Student
