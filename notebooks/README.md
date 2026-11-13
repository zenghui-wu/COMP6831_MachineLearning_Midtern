# Home Credit Default Risk: Midterm Project

Predicts whether a loan applicant will default, using the Kaggle Home Credit dataset. Three models are trained and compared: Logistic Regression, a Support Vector Machine, and a Decision Tree.

All code is in one notebook: [`MidtermProject_v4.ipynb`](MidtermProject_v4.ipynb)

Open this README in JupyterLab (right-click the file, then Open With > Markdown Preview) and the section links below jump to the matching heading in the notebook. The notebook also has a Contents cell at the top with the same links, which works without this README.

## Project layout

Keep this README in the same folder as the notebook so the links resolve.

| Path | Contents | Created by |
|---|---|---|
| [`MidtermProject_v4.ipynb`](MidtermProject_v4.ipynb) | All code, tables and figures | |
| [`home-credit-default-risk/`](home-credit-default-risk/) | Raw Kaggle CSV files | [Download the Dataset](<MidtermProject_v4.ipynb#Download-the-Dataset>) |
| [`processed/`](processed/) | `X_train.npy`, `X_val.npy`, `X_test.npy`, labels, `feature_names.csv`, `preprocessor.joblib` | [b.4 to b.8](<MidtermProject_v4.ipynb#b.4--b.8-Train-fitted-preprocessing>) |
| [`lr_process/`](lr_process/) | Logistic Regression tuning results, metrics, coefficients, saved model | [c.5](<MidtermProject_v4.ipynb#c.5-Kaggle-submission>) |
| [`svm_process/`](svm_process/) | `svm_progress.csv`, written after every SVM fit | [(d) setup](<MidtermProject_v4.ipynb#(d)-Model-2:-Support-Vector-Machine-(cuML,-GPU)>) |
| [`dt_processed/`](dt_processed/) | Decision Tree tuning results, metrics, importances, saved model | [e.4](<MidtermProject_v4.ipynb#e.4-Kaggle-submission>) |
| [`submissions/`](submissions/) | One Kaggle submission CSV per model | c.5, d.5, e.4 |

## 1. Data processing

### 1.1 What the data looks like

| Topic | Finding | Code |
|---|---|---|
| Structure | 307,511 labelled applications; numeric, 0/1 flag and categorical columns | [A.1. Structure and data types](<MidtermProject_v4.ipynb#A.1.-Structure-and-data-types>) |
| Class imbalance | 8.07% defaults, about 11 repaid for every default | [A.2.Target Balance](<MidtermProject_v4.ipynb#A.2.Target-Balance>) |
| Missing data | Building descriptors are 50 to 70% missing; missingness itself predicts default | [A.3.Missing values evaluation](<MidtermProject_v4.ipynb#A.3.Missing-values-evaluation>) |
| Outliers | Heavy-tailed money columns; maximum income far above the 99th percentile | [A.4.Statistic summary and anomalies](<MidtermProject_v4.ipynb#A.4.Statistic-summary-and-anomalies>) |
| Anomalies | `DAYS_EMPLOYED = 365243` placeholder in about 18% of rows; `XNA` categories | [A.5.Anomaly Detcection and Analysis](<MidtermProject_v4.ipynb#A.5.Anomaly-Detcection-and-Analysis-after-statistic-summary>) |
| Strongest signals | The three `EXT_SOURCE` scores and age | [A.6.Distribution by Target](<MidtermProject_v4.ipynb#A.6.Distribution-by-Target>), [A.7.Correlation](<MidtermProject_v4.ipynb#A.7.Correlation>) |
| Categorical signal | Top 10 categorical features by mutual information | [A.8.Default rate by Categorical features](<MidtermProject_v4.ipynb#A.8.Default-rate-by-Categorical-features>) |
| Auxiliary tables | Coverage and size of the five side tables; train/test shift | [A.10.Auxiliary Tables from other tables](<MidtermProject_v4.ipynb#A.10.Auxiliary-Tables-from-other-tables>) |
| Summary | Findings and hypotheses that drive the preprocessing | [A.10.Initial findings and Hypothesis of Project](<MidtermProject_v4.ipynb#A.10.Initial-findings-and-Hypothesis-of-Project>) |

### 1.2 Preprocessing pipeline

Every step that learns something from the data is fitted on the training split only, then applied unchanged to validation and test.

| Step | What it does | Code |
|---|---|---|
| b.1 | Stratified 80/20 train/validation split, made first so later steps cannot see validation rows | [b.1 Stratified train/validation split](<MidtermProject_v4.ipynb#b.1-Stratified-train/validation-split>) |
| b.2 | Placeholder `365243` becomes missing plus a flag; `XNA` gender becomes missing; ratio features (credit/income, annuity/income and others); external-score statistics; document count | [b.2 Application table: cleaning and domain features](<MidtermProject_v4.ipynb#b.2-Application-table:-cleaning-and-domain-features>) |
| b.3 | Six auxiliary tables aggregated to one row per applicant (bureau, previous applications, POS/cash, instalments, credit card) and left-joined; rare categories lumped using training clients only | [b.3 Combine the auxiliary tables](<MidtermProject_v4.ipynb#b.3-Combine-the-auxiliary-tables>) |
| b.4 | Drop columns over 80% missing or near-constant; add one missing indicator per distinct missing pattern | [b.4- b.8 Train-fitted preprocessing](<MidtermProject_v4.ipynb#b.4--b.8-Train-fitted-preprocessing>) |
| b.5 | Median imputation for numeric columns; `"Missing"` level for categoricals | same section |
| b.6 | Clip to the 0.1st and 99.9th percentiles; `log1p` on skewed non-negative columns | same section |
| b.7 | Drop one of every pair of columns with \|r\| > 0.98 | same section |
| b.8 | Standardise numeric features; one-hot encode categoricals with rare levels grouped | same section |

Result: 313 raw features become 412 model inputs, saved to [`processed/`](processed/). The training split has 246,008 rows.

Key functions, all in section (b) of the notebook:

- `engineer_application`: step b.2
- `bureau_features`, `previous_features`, `pos_features`, `installments_features`, `credit_card_features`: step b.3
- `build_features`: joins everything into one table
- `HomeCreditPreprocessor`: steps b.4 to b.8, with `fit` and `transform`

## 2. Models

### 2.1 Shared setup

- **Selection metric:** validation ROC-AUC, the competition metric, which is not distorted by the class imbalance.
- **Imbalance handling:** balanced class weights in all three models.
- **Threshold metrics:** precision, recall, F1 and accuracy at probability 0.5 (decision value 0 for the SVM).

Code: [(c) Model 1: Logistic Regression (baseline)](<MidtermProject_v4.ipynb#(c)-Model-1:-Logistic-Regression-(baseline)>), which also defines the shared helpers `classification_metrics`, `report_best` and `write_submission`.

### 2.2 What was tuned

| Model | Hyperparameters tuned | Code |
|---|---|---|
| Logistic Regression | `C` over seven values from 1e-4 to 10 | [c.1](<MidtermProject_v4.ipynb#c.1-Tuning-C-on-the-validation-set>), [c.2](<MidtermProject_v4.ipynb#c.2-Best-configuration:-training-and-validation-metrics>), [c.3](<MidtermProject_v4.ipynb#c.3-Ablation:-why-class-weighting>), [c.4](<MidtermProject_v4.ipynb#c.4-Interpretation:-largest-coefficients>) |
| SVM (cuML, GPU) | Kernel (linear or RBF), `C`, `gamma`, and the loss for the linear SVM | [d.1](<MidtermProject_v4.ipynb#d.1-Stage-1:-linear-SVM-(tuning-C-and-the-loss)>), [d.2](<MidtermProject_v4.ipynb#d.2-Stage-2:-RBF-kernel-SVM-(tuning-C-and-gamma)>), [d.3](<MidtermProject_v4.ipynb#d.3-Intermediate-results>), [d.4](<MidtermProject_v4.ipynb#d.4-Best-configuration:-training-and-validation-metrics>), [d.5](<MidtermProject_v4.ipynb#d.5-Kaggle-submission-(Platt-scaling)>) |
| Decision Tree | `max_depth`, `min_samples_leaf`, `criterion` (36 combinations) | [(e)](<MidtermProject_v4.ipynb#(e)-Model-3:-Decision-Tree>), [e.1](<MidtermProject_v4.ipynb#e.1-Intermediate-results>), [e.2](<MidtermProject_v4.ipynb#e.2-Best-configuration:-training-and-validation-metrics>), [e.3](<MidtermProject_v4.ipynb#e.3-Interpretation>) |

### 2.3 Results

| Model (best configuration) | Train AUC | Val AUC | Gap | Val precision | Val recall | Val F1 |
|---|---|---|---|---|---|---|
| Logistic Regression (C = 1) | 0.7848 | 0.7824 | 0.0024 | 0.1798 | 0.7067 | 0.2866 |
| SVM (RBF, C = 0.1, gamma = 0.5 x scale) | 0.8062 | 0.7850 | 0.0211 | 0.1819 | 0.7118 | 0.2898 |
| Decision Tree (gini, depth 10, leaf 1000) | 0.7566 | 0.7336 | 0.0230 | 0.1544 | 0.6620 | 0.2503 |

Code and Kaggle scores: [f.1 Comparison among three models](<MidtermProject_v4.ipynb#f.1-Comparison-among-three-models>)

### 2.4 Model chosen: Logistic Regression

The RBF SVM has the highest validation ROC-AUC (0.7850), but Logistic Regression is chosen as the most appropriate model for this dataset:

- **Nearly the same ranking quality:** 0.7824 against 0.7850, a difference of 0.0026.
- **Least overfitting:** train/validation gap of 0.0024, against 0.0211 for the SVM and 0.0230 for the tree.
- **Cheapest to train:** about 8 seconds on CPU, against about 9 minutes on a GPU with a separate cuML environment for the SVM.
- **Interpretable:** standardised inputs make coefficient sizes directly comparable.

Code: [f.2 The best model and rational:](<MidtermProject_v4.ipynb#f.2-The-best-model-and-rational:>) and [f.3 Conclusion:](<MidtermProject_v4.ipynb#f.3-Conclusion:>)

## How to run

1. Uncomment and run the download cell once to fetch the data into `home-credit-default-risk/`.
2. Run sections (a), (b) and (c) on CPU.
3. Switch to a GPU runtime before section (d). The notebook installs cuML if it is missing.
4. Run section (e) on CPU.
5. Upload the files in `submissions/` to Kaggle.
