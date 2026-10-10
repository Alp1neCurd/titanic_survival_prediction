# Titanic Survival Prediction

This project predicts passenger survival in the Titanic Kaggle competition. I used it to compare classic machine learning models, boosting algorithms, ensembles, and a PyTorch neural network.

The final model is CatBoost with one-hot encoded features. It was selected using local cross-validation rather than the public Kaggle leaderboard.

## Project structure

- `titanic_experiments.ipynb` — data analysis and model experiments
- `main.py` — runs the final pipeline
- `config.py` — paths, random seed, and model parameters
- `features.py` — feature engineering
- `model.py` — preprocessing and model setup
- `requirements.txt` — project dependencies

## Setup

Create a virtual environment on Windows:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download `train.csv` and `test.csv` from the [Titanic competition page](https://www.kaggle.com/competitions/titanic/data) and place them in the `data` directory:

```text
data/
├── train.csv
└── test.csv
```

Run the pipeline:

```powershell
python main.py
```

The script evaluates the model, fits it on all labeled data, and saves the submission to:

```text
submissions/production/titanic_catboost_one_hot.csv
```

## Validation

I used two validation schemes:

- Regular CV uses repeated stratified splits.
- Group CV keeps passengers with the same ticket in one fold.

The final selection score is the mean of the Regular CV and Group CV scores. The number of folds, repeats, and the random seed are defined in `config.py`.

## Models and results

The notebook contains 19 models, including baselines, tuned models, ensembles, and a PyTorch neural network.

The "Best CV" column shows the best result obtained during tuning. These values are not always directly comparable because different validation schemes were used.

The "Selection score" is the mean of Regular CV and Group CV. It was calculated only for the final candidates and used to select the production model.

Simple research baselines, such as Dummy Classifier and Sex + Pclass Tree, are available in the notebook but omitted from this summary.

| Model | Best CV | Tuning setup | Selection score | Kaggle |
|---|---:|---|---:|---:|
| XGBoost | 0.8342 | Regular CV | — | — |
| Equal Soft Voting | 0.8328 | Mixed | 0.8268 | 0.77511 |
| **CatBoost One-Hot** | 0.8315 | Group CV | **0.8275** | 0.77511 |
| Equal Hard Voting | 0.8314 | Mixed | 0.8249 | 0.77751 |
| PyTorch DNN | 0.8300 | Regular CV | — | 0.76555 |
| KNN | 0.8272 | Regular CV | — | — |
| Stacking Logistic | 0.8272 | Mixed | — | — |
| Logistic Regression | 0.8258 | Regular CV | — | — |
| CatBoost Native | 0.8231 | Group CV | 0.8226 | 0.77990 |
| Stacking Ridge | 0.8230 | Mixed | 0.8202 | 0.78229 |
| **Gradient Boosting** | 0.8189 | Group CV | 0.8209 | **0.78708** |
| SVC | 0.8174 | Group CV | 0.8195 | 0.77033 |
| Decision Tree | 0.8174 | Regular CV | — | — |
| Bagging Decision Tree | 0.8174 | Group CV | 0.8139 | 0.77990 |
| LightGBM | 0.8146 | Group CV | 0.8139 | 0.76794 |
| Random Forest | 0.8105 | Group CV | 0.8142 | 0.76794 |
| Sex Rule | 0.7850 | Not tuned | 0.7851 | 0.76555 |

CatBoost One-Hot ranked first in the final local comparison and scored `0.8492` on the group holdout. The Sex Rule baseline scored `0.7933` on the same holdout.

Gradient Boosting had the best public leaderboard score. I kept CatBoost One-Hot as the final model because it was selected by local validation. Kaggle scores were used only as an external check.

The five-fold CatBoost submission averaged predicted probabilities. It differed from the full-fit submission for only two passengers, and both submissions scored `0.77511`.

## PyTorch DNN

The DNN scored `0.8300` on Regular CV and `0.76555` on Kaggle. It did not outperform the classic models, so it was kept as a separate experiment.

The network has two hidden layers and uses `BCEWithLogitsLoss`. The number of layers and the loss function were not tuned.

## Features

The pipeline creates features from passenger titles, family size, tickets, cabins, fares, and group composition.

Group statistics are calculated from `train` and `test` together without using the `Survived` target. They improved both local validation scores, so they were kept.

Feature engineering, missing-value imputation, scaling, and encoding are performed inside the pipeline.

## Limitations

- The dataset is small, so a few changed predictions can noticeably affect the score.
- The holdout was checked during development and is not a fully independent test set.
- Hyperparameters were tuned on the same folds used for CV. Nested CV was not used.
- DNN, XGBoost, Logistic Regression, and KNN were tuned on Regular CV and were not included in the final model selection.
- Group CV uses ticket numbers as group identifiers and may be too strict for passengers from unseen groups.
- Preprocessing was evaluated as one pipeline. Only the group features were tested separately.
- Public leaderboard scores are close, so they are not enough to prove that one model is better.