## Research Workflow

This repository contains a complete research workflow for analyzing Self-Repaid SATD (Self-Admitted Technical Debt). The workflow consists of three main stages: **Data Collection**, **Feature Engineering**, and **Model Analysis**.

### Latest Dataset

The latest dataset is available on Google Drive: [Self-Repaid SATD latest dataset](https://drive.google.com/file/d/1_PqKOyhw9-htSniEy5uUsyMV_bLG5TzK/view?usp=drive_link).

### Prerequisites for Research Workflow

**Required Software:**
- Python 3.7+
- Java 1.8+ (for SatdBailiff)
- MySQL 5.4+
- Maven 3 (for building SatdBailiff)
- R (for statistical analysis and visualization)

**Required Python Packages:**
- pandas, numpy, scipy, scikit-learn
- xgboost, matplotlib, seaborn
- shap (for model interpretation)
- mysql-connector-python

### Stage 1: Data Collection

The data collection process involves three steps:

#### Step 1.1: Clone Repositories
Clone the target repositories from GitHub:
```bash
cd Dataset
bash clone_repos.sh
```
This script reads from `repos.csv` and clones repositories into organized directories.

#### Step 1.2: Build and Run SatdBailiff
Build the SatdBailiff tool and mine SATD data:
```bash
cd Dataset
bash build_and_run_satdbailiff.sh
```
This script:
- Builds the SatdBailiff JAR file using Maven
- Runs SatdBailiff to mine SATD instances from the cloned repositories
- Stores results in the MySQL database

**Note:** Before running, ensure:
- MySQL server is running
- Database schema is created (see `sql/satd.sql`)
- `mySQL.properties` is configured with your database credentials
- `repos.csv` contains the list of repositories to analyze

#### Step 1.3: Extract Raw Data
Extract raw SATD data from the database:
```bash
cd Dataset
bash get_raw_data.sh
```
This script:
- Queries the MySQL database using `sql/query_self-fixed_satd.sql`
- Filters data (word count ≥ 2, valid date ranges)
- Outputs `data/raw_data_final_<count>.json`

### Stage 2: Feature Engineering

Extract and merge features from the raw data:

```bash
cd "Feature Engineering"
bash run_scripts.sh
```

This script:
1. **Runs all feature extraction scripts** in `scripts/` directory:
   - SATD features: type, length, quality, path depth, survival days, add date
   - Code features: cyclomatic complexity, file lines, method parameters
   - Developer features: active days, commits, ownership, past commit favor
   - Repository features: active days, commits, developers, file frequency, README score

2. **Merges all features** using `merge_features.py`:
   - Combines features from multiple JSON files
   - Preserves base fields: `satd_id`, `is_self_fixed`, `project_name`
   - Outputs `../Dataset/data/merged_data_<count>.json`

**Manual merge (if needed):**
```bash
cd "Feature Engineering"
python merge_features.py
```

### Stage 3: Model Analysis

The model analysis stage addresses four research questions (RQ1-RQ4):

#### RQ1: Significance Testing and Distribution Analysis

**Location:** `Model/analysis/` and `Model/distribution/`

**Significance Testing:**
```bash
cd Model/analysis
python significance_test.py
```
This script:
- Performs statistical significance tests (Mann-Whitney U for numerical, Chi-square/Fisher for categorical)
- Calculates effect sizes (Cliff's Delta, Cramer's V, Odds Ratio)
- Outputs `significance_test.txt` and `no_significant_features.txt`

**Distribution Analysis:**
```bash
cd Model/distribution
bash run_scripts.sh
```
This generates:
- Distribution plots for self-fixed vs non-self-fixed SATD
- Distribution analysis by SATD type
- Boxplots and statistical tests (Scott-Knott ESD test)

#### RQ2: Survival Analysis

**Location:** `Model/survival/`

```bash
cd Model/survival
bash run_scripts.sh
```
This generates:
- Survival curves with confidence intervals
- Survival analysis by SATD type
- Statistical comparisons between self-fixed and non-self-fixed groups
- Boxplots and Scott-Knott ESD tests

#### RQ3: Predictive Modeling

**Location:** `Model/predictive model/`

Multiple machine learning models are implemented:

**Traditional ML Models:**
- Logistic Regression: `Logistic_regression_logo.py`, `Logistic_regression_10.py`
- Random Forest: `Random_forest_logo.py`, `Random_forest_10.py`
- XGBoost: `XGBoost_logo.py`, `XGBoost_10.py`

**Deep Learning Models:**
- TextCNN: `TextCNN.py`, `TextCNN.ipynb`
- BERT: `BERT.py`, `BERT.ipynb`

**Baseline:**
- Random Guess: `Random_Guess.ipynb`

**Run all models:**
```bash
cd "Model/predictive model"
bash run_scripts.sh
```

Models support:
- Leave-One-Group-Out (LOGO) cross-validation
- 10-fold cross-validation
- Hyperparameter tuning
- ROC curve visualization

#### RQ4: Model Interpretation (SHAP Analysis)

**Location:** `Model/analysis/`

```bash
cd Model/analysis
python xgboost_shap.py
```
This script:
- Trains XGBoost model with 10-fold cross-validation
- Computes SHAP values for model interpretation
- Generates:
  - `xgboost_shap.txt`: Cross-validation results
  - `xgboost_shap.pdf`: SHAP summary plot
  - `xgboost_shap_global_shap_importance.pdf`: Top 15 feature importance
  - `xgboost_shap_mean_abs_shap.csv`: Feature importance rankings

**Note:** The script automatically:
- Removes features with high VIF (Variance Inflation Factor) if `vif_drop.txt` exists
- Removes non-significant features if `no_significant_features.txt` exists

### Output Structure

```
Dataset/
  data/
    raw_data_final_<count>.json          # Raw SATD data
    merged_data_<count>.json             # Merged features

Model/
  analysis/
    significance_test.txt                # Statistical test results
    xgboost_shap.txt                    # Model performance
    xgboost_shap_mean_abs_shap.csv      # Feature importance
    *.pdf                                # Visualizations

  distribution/
    *.pdf                                # Distribution plots
    *.txt                                # Distribution statistics

  survival/
    *.pdf                                # Survival curves
    *.json                               # Survival statistics

  predictive model/
    *_logo.txt                          # LOGO CV results
    *_10fold.txt                        # 10-fold CV results
    *_roc.pdf                           # ROC curves
```

### Quick Start

To run the complete workflow:

```bash
# 1. Data Collection
cd Dataset
bash clone_repos.sh
bash build_and_run_satdbailiff.sh
bash get_raw_data.sh

# 2. Feature Engineering
cd "../Feature Engineering"
bash run_scripts.sh

# 3. Model Analysis
cd ../Model
# RQ1
cd analysis && python significance_test.py && cd ..
cd distribution && bash run_scripts.sh && cd ..
# RQ2
cd survival && bash run_scripts.sh && cd ..
# RQ3
cd "predictive model" && bash run_scripts.sh && cd ..
# RQ4
cd analysis && python xgboost_shap.py && cd ..
```
