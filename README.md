# Titanic Survival Prediction

This project compares classical machine learning models, boosting algorithms, ensembles, and a PyTorch neural network on the Kaggle Titanic dataset.

CatBoost with one-hot encoded features was selected as the production model using local cross-validation rather than the public Kaggle leaderboard.

## Project structure

- `titanic_experiments.ipynb` — EDA, hyperparameter tuning, model comparison, and ablation tests
- `config.py` — paths, random seed, and final model parameters
- `features.py` — reusable feature engineering
- `model.py` — preprocessing and model pipelines
- `dnn.py` — PyTorch model and sklearn-compatible classifier
- `validation.py` — shared validation procedure
- `main.py` — command-line entry point
- `requirements.txt` — dependencies required by `main.py`
- `requirements_notebook.txt` — additional dependencies for the notebook

## Setup

Clone the repository and open its directory.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

To run the notebook on Windows, install the additional dependencies with:

```powershell
py -m pip install -r requirements_notebook.txt
```

On Linux and macOS, use:

```bash
python3 -m pip install -r requirements_notebook.txt
```

Download `train.csv` and `test.csv` from the [Titanic competition page](https://www.kaggle.com/competitions/titanic/data) and place them in the `data` directory:

```text
data/
├── train.csv
└── test.csv
```

## Running the pipeline

The commands below use the Windows Python launcher. On Linux and macOS, replace `py` with `python3`.

Run the selected CatBoost model:

```powershell
py main.py
```

The explicit equivalent is:

```powershell
py main.py --model catboost
```

Run the PyTorch DNN:

```powershell
py main.py --model dnn
```

The script loads the data, creates features, evaluates the selected model, fits it on all labeled passengers, and writes a Kaggle submission to `submissions/production/`.

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

The selection score is the mean of the Regular CV and Group CV estimates. A group-aware holdout is reported as an additional development check but is not used for model selection.

All model families are evaluated with the same final validation procedure. Hyperparameter tuning and final evaluation reuse the available data, so these results are not nested CV estimates.

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

CatBoost One-Hot ranked first locally. Equal Soft Voting was close, but its larger difference between Regular CV and Group CV made CatBoost the more stable choice.

## Kaggle results

Only models that were actually submitted are included below. Kaggle scores were used as an external check, not for hyperparameter tuning or final model selection.

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

Gradient Boosting achieved the highest public score. CatBoost remained the production model because it ranked first under the local selection procedure.

The CatBoost 5-fold submission differed from the full-fit submission for only two passengers, and both received the same public score.

## EDA findings

The notebook includes the following observations from the training data:

- Women survived in approximately 74% of cases, compared with 19% for men.
- Survival rates by passenger class were approximately 63%, 47%, and 24%.
- Children younger than 16 survived in approximately 59% of cases, compared with 38% for adults.
- Passengers traveling alone had a survival rate of approximately 30%.
- Families of two to four people had a survival rate of approximately 58%, while larger families had approximately 16%.
- Age is missing for 177 passengers and cabin information for 687 of 891 passengers.

These observations motivated the use of sex, passenger class, age, family structure, title, cabin, and group features.

## Feature engineering

The pipeline creates features from:

- passenger titles;
- family size and solo travel;
- ticket and surname groups;
- cabin deck and cabin count;
- fare per passenger;
- group composition and average age.

Missing ages are estimated using passenger title and class, with broader fallbacks when a group median is unavailable. Remaining numeric and categorical missing values are handled inside the preprocessing pipeline.

Ticket and family statistics are calculated from the available `train` and `test` passenger manifest without using the `Survived` target. The ablation test showed that these features improved both validation estimates.

## PyTorch DNN

The DNN uses two hidden layers, dropout, optional batch normalization, and `BCEWithLogitsLoss`. It is wrapped in an sklearn-compatible estimator, so it uses the same feature pipeline and validation function as CatBoost.

Its final comparable results were:

- Regular CV: `0.8226`
- Group CV: `0.8104`
- Selection score: `0.8165`
- Holdout accuracy: `0.8492`
- Kaggle public score: `0.76555`

## Limitations

- Titanic is a small dataset, so a few changed predictions can noticeably affect the results.
- The holdout was inspected during development and is not a fully independent test set.
- Hyperparameters were tuned using the same dataset later used for comparison; nested CV was not performed.
- Some models were tuned with Regular CV and others with Group CV, although the final comparison uses the same validation procedure for every model.
- Group CV based on ticket numbers may be strict for passengers from previously unseen groups.
- Group statistics use the unlabeled test manifest. This is target-free but still a transductive feature-engineering choice.
- Only the group-composition feature set was tested separately.
- Small differences on the public leaderboard are not enough to establish that one model is consistently better.