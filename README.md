# 💳 Scoring Model Implementation  

**Projet de bout en bout d’un modèle de scoring de crédit** : de la préparation des données à la mise en production via une API déployée sur Heroku.  
Ce projet illustre l’ensemble du cycle de vie d’un modèle de Machine Learning supervisé appliqué à la prédiction du risque de crédit.

---

## 🎯 Objectif du projet  

L’objectif est de **prédire la probabilité de défaut de paiement d’un client** pour aider la société **“Prêt à Dépenser”** à prendre des décisions d’octroi de crédit plus fiables, transparentes et traçables.

Le projet couvre :
- 🧮 **Modélisation prédictive** : construction et optimisation d’un modèle de scoring.  
- 📈 **Suivi et traçabilité** : intégration complète dans **MLflow** (expérimentations, métriques, modèles).  
- ⚙️ **Industrialisation** : création d’un pipeline CI/CD, tests unitaires, et déploiement via **Heroku**.  
- 🧠 **Interprétabilité** : analyse globale et locale avec **SHAP values**.  
- ☁️ **API REST** : service en ligne de prédiction du score de solvabilité.

---

## 🧩 Contenu du dépôt  

Le dépôt contient plusieurs **branches** :  
- `feature-engineering` → Préparation et agrégation des données (Home Credit Dataset).  
- `modeling` → Construction du pipeline MLflow + GridSearchCV + SMOTE.  
- `optimization` → Ajustement local des hyperparamètres et du seuil optimal de probabilité.  
- `explainability` → Analyse SHAP globale et locale.  
- `deployment` → API Flask + CI/CD + déploiement Heroku.  

### Structure générale

```text
Scoring_Model_Implementation
│
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_modeling_mlflow.ipynb
│   ├── 3_threshold_optimization.ipynb
│   ├── 4_shap_analysis.ipynb
│
├── api/
│   ├── app.py
│   ├── test_app.py
│   ├── Procfile
│   ├── requirements.txt
│   ├── runtime.txt
│   ├── deploy.yml
│
├── mlruns_reduced/
│   ├── model.pkl
│   ├── columns.pkl
│   ├── best_threshold
│
├── Martineau_Alexandre_6_presentation_012025.pdf
└── README.md
```

## ⚙️ Pipeline & Méthodologie

---

### 1️⃣ Préparation et Feature Engineering

- **Données** : issues du jeu *Home Credit Default Risk* (Kaggle).  
- **Taille** : 307 511 clients, 785 variables agrégées (`bureau`, `installments_payments`, `previous_application`, etc.).  
- **Ratios clés créés** :
  - `Payment Rate`  
  - `Annuity / Income`  
  - `Days_Employed_Perc`  
- **Encodage** :
  - One-Hot Encoding pour les variables catégorielles  
  - Factorisation numérique pour les identifiants  

---

### 2️⃣ Modélisation et suivi MLflow

Implémentation d’un **pipeline complet** de Machine Learning avec :

```text
SimpleImputer → StandardScaler → SMOTE → GridSearchCV
```

