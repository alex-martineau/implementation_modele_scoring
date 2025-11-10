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

### 1️⃣ **Préparation et Feature Engineering**
- Données issues du jeu **Home Credit Default Risk (Kaggle)**.  
- 307 511 clients, 785 variables agrégées (données `bureau`, `installments_payments`, `previous_application`, etc.).  
- Création de ratios clés : `Payment Rate`, `Annuity/Income`, `Days_Employed_Perc`.  
- Encodage one-hot et factorisation catégorielle.  

### 2️⃣ **Modélisation et suivi MLflow**
- Implémentation d’un pipeline complet avec :
  - `SimpleImputer`, `StandardScaler`, `SMOTE`,  
  - `GridSearchCV` sur 5 modèles : Dummy, Logistic Regression, Random Forest, Gradient Boosting, LightGBM.  
- **Métrique personnalisée** basée sur le **coût métier** :  
  - Faux négatif = perte du montant du crédit.  
  - Faux positif = perte de ~19 % du crédit.  
- Suivi et versioning via **MLflow Tracking Server**.  

### 3️⃣ **Comparaison des modèles**

| Modèle | Accuracy | ROC AUC | F1 Score | Test Score |
|---------|-----------:|--------:|---------:|-----------:|
| Dummy Classifier | 0.92 | 0.50 | 0.00 | -449.30 |
| Logistic Regression | 0.72 | 0.73 | 0.26 | -452.27 |
| Random Forest | 0.88 | 0.69 | 0.17 | -439.39 |
| Gradient Boosting | 0.91 | 0.75 | 0.06 | -440.21 |
| **LightGBM** | **0.89** | **0.68** | **0.17** | **-432.92** |

🏆 **Meilleur modèle : LightGBM**  
→ Paramètres optimaux : `learning_rate = 0.01`, `max_depth = 3`, `n_estimators = 158`.  
→ **Perte minimale : -435.5 M**.

---

### 4️⃣ **Optimisation du seuil de probabilité**

- Recherche du seuil optimal entre **0.01** et **0.99**.  
- **Seuil optimal : 0.48**  
- **Coût minimal : -426.53 M**

> Permet un équilibre optimal entre détection des clients à risque et minimisation des pertes.

---

### 5️⃣ **Interprétation et transparence**

#### 🔍 Analyse SHAP  
- **Variables les plus influentes** :  
  - `EXT_SOURCE_2` : score externe de solvabilité  
  - `INSTAL_DPD_MAX` : retards de paiement maximum  
  - `CODE_GENDER` : indicateur socio-économique  
  - `BURO_AMT_CREDIT_SUM_DEBT_SUM` : dettes en cours  
  - `NAME_EDUCATION_TYPE_Secondary` : niveau d’éducation  
  - `FLAG_OWN_CAR`, `NAME_INCOME_TYPE_Working`, `FLAG_PHONE`

#### 🌈 Visualisations SHAP
- Diagrammes *beeswarm* → importance globale.  
- *Waterfall plots* → explication locale d’une prédiction client.

---

### 6️⃣ **Déploiement API & CI/CD**

- **API Flask** hébergée sur **Heroku** → prédiction en ligne :  
  🔗 [https://my-scoring-app-546acd78d8fa.herokuapp.com/](https://my-scoring-app-546acd78d8fa.herokuapp.com/)  

#### Endpoints :
| Route | Description |
|-------:|:-------------|
| `/predict` | Retourne la décision de prêt pour un client donné |
| `/best_threshold` | Affiche le seuil de classification optimal |
| `/features` | Liste les variables utilisées |

#### 🔧 Fichiers clés :
- `app.py` → API Flask  
- `Procfile`, `runtime.txt` → configuration Heroku  
- `deploy.yml` → pipeline CI/CD  
- `test_app.py` → tests unitaires avec **pytest**  
- `mlruns_reduced/` → artefacts du modèle (MLflow)  

---

### 7️⃣ **Surveillance & Maintenance**

- Suivi **Data Drift** avec **Evidently AI**.  
- Tests unitaires automatiques à chaque *push*.  
- Versionning du modèle et des métriques dans **MLflow Registry**.  

---

## 🧠 Résumé des performances

| Étape | Meilleur modèle | Seuil | Perte métier | Métriques |
|:------|:----------------|:------:|-------------:|:---------|
| GridSearchCV | LightGBM | 0.5 | -432.92 M | AUC = 0.68 |
| Optimisation fine | LightGBM | 0.48 | **-426.53 M** | AUC = 0.75 |

---

## 🛠️ Technologies utilisées

- **Python** : `scikit-learn`, `lightgbm`, `mlflow`, `shap`, `imblearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`  
- **MLOps** : MLflow, GitHub Actions, CI/CD, Heroku  
- **Tests & Monitoring** : `pytest`, `Evidently AI`  
- **API** : Flask + Gunicorn  

---

## 📦 Dépôt & Liens

🔗 **Dépôt GitHub :** [Scoring_Model_Implementation](https://github.com/alex-martineau/Scoring_Model_Implementation)  
🔗 **API en ligne :** [https://my-scoring-app-546acd78d8fa.herokuapp.com/](https://my-scoring-app-546acd78d8fa.herokuapp.com/)  

---

## ✅ Conclusion  

> Ce projet illustre l’intégration complète du cycle de vie d’un modèle de Machine Learning :  
> de la **préparation des données** à la **mise en production automatisée** avec traçabilité, interprétabilité et suivi du drift.  
> L’approche MLOps adoptée garantit la **robustesse, la transparence et la scalabilité** du modèle de scoring crédit.
