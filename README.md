# Titanic Survival Prediction

This project compares classical machine learning models, boosting algorithms, ensembles, and a PyTorch neural network on the Kaggle Titanic dataset.

CatBoost with one-hot encoded features was selected using local cross-validation. Kaggle scores were used only as an external check.

## Project structure

- `main.py` — command-line entry point
- `config.py` — paths, random seed, and final model parameters
- `features.py` — reusable feature engineering
- `model.py` — preprocessing and model selection
- `dnn.py` — sklearn-compatible PyTorch classifier
- `validation.py` — shared validation procedure
- `titanic_experiments.ipynb` — EDA, tuning, comparisons, and ablation tests
- `requirements.txt` — dependencies for CatBoost
- `requirements_dnn.txt` — CatBoost and PyTorch dependencies
- `requirements_notebook.txt` — complete notebook environment

## Setup

The project was tested with Python 3.12.

On Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

On Linux or macOS, use `python3` instead of `py` and activate the environment with:

```bash
source .venv/bin/activate
```

Download `train.csv` and `test.csv` from the [Titanic competition page](https://www.kaggle.com/competitions/titanic/data) and place them in the `data` directory.

```text
data/
  train.csv
  test.csv
```

Install the additional dependencies when needed:

```powershell
# CatBoost and PyTorch DNN
py -m pip install -r requirements_dnn.txt

# Complete notebook environment
py -m pip install -r requirements_notebook.txt
```

## Running the pipeline

Run the selected CatBoost model:

```powershell
py main.py
```

Run the PyTorch model:

```powershell
py main.py --model dnn
```

On Linux and macOS, replace `py` with `python3`.

The script loads the data, creates features, evaluates the selected model, fits it on all labeled passengers, and saves a Kaggle submission in `submissions/production/`.

Example CatBoost output:

```text
Model: CatBoost One-Hot
Regular CV: 0.8291
Group CV: 0.8259
Selection Score: 0.8275
Holdout Accuracy: 0.8492
Predicted survivors: 162
```

## Validation

Two validation schemes are used:

- Regular CV uses repeated stratified splits.
- Group CV keeps passengers with the same ticket in one fold.

The selection score is the mean of these two estimates. A group-aware holdout is reported as an additional check but is not used for model selection.

All models in the comparison table were evaluated with this procedure after tuning. The results are not nested CV estimates because tuning and final evaluation use the same dataset.

## Local model comparison

| Model | Type | Regular CV | Group CV | Selection score | Holdout |
|---|---|---:|---:|---:|---:|
| **CatBoost One-Hot** | Single model | 0.8291 | 0.8259 | **0.8275** | 0.8492 |
| Equal Soft Voting | Ensemble | 0.8319 | 0.8216 | 0.8268 | 0.8436 |
| Equal Hard Voting | Ensemble | 0.8286 | 0.8212 | 0.8249 | 0.8436 |
| CatBoost Native | Single model | 0.8226 | 0.8226 | 0.8226 | 0.8380 |
| Stacking Logistic | Ensemble | 0.8263 | 0.8188 | 0.8226 | 0.8492 |
| Gradient Boosting | Single model | 0.8258 | 0.8160 | 0.8209 | 0.8492 |
| Stacking Ridge | Ensemble | 0.8230 | 0.8174 | 0.8202 | 0.8492 |
| SVC | Single model | 0.8202 | 0.8188 | 0.8195 | 0.8324 |
| Logistic Regression | Single model | 0.8221 | 0.8141 | 0.8181 | 0.8212 |
| PyTorch DNN | Single model | 0.8226 | 0.8104 | 0.8165 | 0.8492 |
| XGBoost | Single model | 0.8235 | 0.8062 | 0.8149 | 0.8268 |
| Random Forest | Single model | 0.8202 | 0.8081 | 0.8142 | 0.8492 |
| LightGBM | Single model | 0.8174 | 0.8104 | 0.8139 | 0.8380 |
| Bagging Decision Tree | Single model | 0.8230 | 0.8047 | 0.8139 | 0.8715 |
| KNN | Single model | 0.8226 | 0.8005 | 0.8116 | 0.8324 |
| Decision Tree | Single model | 0.8085 | 0.7870 | 0.7977 | 0.8045 |
| Sex Rule | Baseline | 0.7851 | 0.7852 | 0.7851 | 0.7933 |
| Sex + Pclass Tree | Baseline | 0.7734 | 0.7735 | 0.7734 | 0.7933 |
| Dummy Classifier | Baseline | 0.6166 | 0.6166 | 0.6166 | 0.6145 |

CatBoost and Equal Soft Voting differ by only `0.0007` in selection score, which is within expected CV noise. CatBoost was kept as the production model because it is a simpler single-model solution.

## Kaggle results

Only models that were submitted to Kaggle are listed here.

| Submission | Public score |
|---|---:|
| **Gradient Boosting** | **0.78708** |
| Stacking Ridge | 0.78229 |
| CatBoost Native | 0.77990 |
| Bagging Decision Tree | 0.77990 |
| Equal Hard Voting | 0.77751 |
| CatBoost One-Hot, full fit | 0.77511 |
| CatBoost One-Hot, 5-fold | 0.77511 |
| Equal Soft Voting | 0.77511 |
| SVC | 0.77033 |
| Random Forest | 0.76794 |
| LightGBM | 0.76794 |
| PyTorch DNN | 0.76555 |
| Sex Rule | 0.76555 |

Gradient Boosting achieved the highest public score, while CatBoost ranked first under the local selection procedure. The leaderboard was treated as an external check rather than a model-selection tool.

The CatBoost full-fit and five-fold submissions differed for only two passengers and received the same public score. Small leaderboard differences should not be interpreted as reliable differences between models.

## EDA and feature engineering

The training data showed large survival differences by sex, passenger class, age, and family structure:

- women: 74%, men: 19%;
- first, second, and third class: 63%, 47%, and 24%;
- children under 16: 59%, adults: 38%;
- solo passengers: 30%;
- families of 2–4 people: 58%, families of 5 or more: 16%.

The pipeline creates title, family, ticket, surname, cabin, fare-per-passenger, group-composition, and group-age features. The implementation is in `features.py`, while the analysis and ablation tests are documented in the notebook.

Ticket and family statistics use the combined passenger manifest without using the `Survived` target.

## PyTorch DNN

The DNN uses two hidden layers, dropout, optional batch normalization, and `BCEWithLogitsLoss`. Its sklearn-compatible wrapper allows it to use the same features and validation procedure as CatBoost.

## Limitations

- Titanic is small, so a few predictions can noticeably change validation and leaderboard scores.
- The holdout was inspected during development and is not a fully independent test set.
- Hyperparameters were tuned on the same dataset used for the final comparison.
- Some models were originally tuned with different CV schemes, although all final scores use the same validation function.
- Group features use the unlabeled test manifest, making them a target-free but transductive choice.
- Categorical embeddings were not tested. Most categorical features have few levels, so one-hot encoding was kept for simplicity and consistent comparison across models.