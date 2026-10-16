# Titanic Survival Prediction

This project compares classical ML models, boosting, ensembles, and a PyTorch neural network on the Kaggle Titanic dataset.

CatBoost with one-hot encoded features was selected using local cross-validation.

## Project files

- `main.py` - runs the final CatBoost or DNN pipeline
- `config.py` - paths, seed, and model parameters
- `features.py` - feature engineering
- `model.py` - preprocessing and model selection
- `dnn.py` - PyTorch classifier
- `validation.py` - shared validation
- `titanic_experiments.ipynb` - EDA, tuning, and model comparison
- `requirements*.txt` - dependencies for CatBoost, DNN, and the notebook

## Setup

Tested with Python 3.12.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Use `requirements_dnn.txt` for the DNN or `requirements_notebook.txt` for the complete notebook environment:

```powershell
py -m pip install -r requirements_dnn.txt
py -m pip install -r requirements_notebook.txt
```

On Linux and macOS, replace `py` with `python3` and activate the environment with:

```bash
source .venv/bin/activate
```

Download `train.csv` and `test.csv` from the [Titanic competition page](https://www.kaggle.com/competitions/titanic/data) and place them in `data/`.

## Run

CatBoost:

```powershell
py main.py
```

PyTorch DNN:

```powershell
py main.py --model dnn
```
The embedding DNN is evaluated in the notebook and is not available through the `--model` option.

The script evaluates the model, trains it on all labeled passengers, and saves a submission in `submissions/production/`.

A full DNN run on CPU may take several minutes. Its result may differ slightly across hardware and PyTorch versions.

## Validation

Regular CV uses repeated stratified splits. Group CV keeps passengers with the same ticket in one fold.

The selection score is the mean of Regular CV and Group CV. The holdout is a development check and is not used for model selection.

## Local results

| Model | Regular CV | Group CV | Selection | Holdout |
|---|---:|---:|---:|---:|
| **CatBoost One-Hot** | 0.8291 | 0.8259 | **0.8275** | 0.8492 |
| Equal Soft Voting | 0.8319 | 0.8216 | 0.8268 | 0.8436 |
| Equal Hard Voting | 0.8286 | 0.8212 | 0.8249 | 0.8436 |
| CatBoost Native | 0.8226 | 0.8226 | 0.8226 | 0.8380 |
| Stacking Logistic | 0.8263 | 0.8188 | 0.8226 | 0.8492 |
| Gradient Boosting | 0.8258 | 0.8160 | 0.8209 | 0.8492 |
| Stacking Ridge | 0.8230 | 0.8174 | 0.8202 | 0.8492 |
| SVC | 0.8202 | 0.8188 | 0.8195 | 0.8324 |
| Logistic Regression | 0.8221 | 0.8141 | 0.8181 | 0.8212 |
| PyTorch DNN | 0.8226 | 0.8104 | 0.8165 | 0.8492 |
| XGBoost | 0.8235 | 0.8062 | 0.8149 | 0.8268 |
| Random Forest | 0.8202 | 0.8081 | 0.8142 | 0.8492 |
| LightGBM | 0.8174 | 0.8104 | 0.8139 | 0.8380 |
| Bagging Decision Tree | 0.8230 | 0.8047 | 0.8139 | 0.8715 |
| PyTorch DNN Embeddings | 0.8165 | 0.8085 | 0.8125 | 0.8324 |
| KNN | 0.8226 | 0.8005 | 0.8116 | 0.8324 |
| Decision Tree | 0.8085 | 0.7870 | 0.7977 | 0.8045 |
| Sex Rule | 0.7851 | 0.7852 | 0.7851 | 0.7933 |
| Sex + Pclass Tree | 0.7734 | 0.7735 | 0.7734 | 0.7933 |
| Dummy Classifier | 0.6166 | 0.6166 | 0.6166 | 0.6145 |

CatBoost and Equal Soft Voting differ by only `0.0007`, which is within expected CV noise. CatBoost was kept because it is a simpler single model.

## Kaggle results

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
| PyTorch DNN Embeddings | 0.77511 |
| SVC | 0.77033 |
| Random Forest | 0.76794 |
| LightGBM | 0.76794 |
| PyTorch DNN | 0.76555 |
| Sex Rule | 0.76555 |

Gradient Boosting achieved the highest public score (`0.78708`), while CatBoost had the best local selection score (`0.8275`). This difference shows that local cross-validation and public leaderboard rankings do not always agree. Kaggle was used only as an external check, not for model selection, and small differences may be leaderboard noise.

The notebook currently exports the selected CatBoost submissions and the embedding DNN; other scores came from earlier full-fit experiments.

Embeddings lowered the local DNN selection score from 0.8165 to 0.8125, but raised its public score from 0.76555 to 0.77511, so they did not show a consistent improvement.

## EDA and features

The main observations were:

- women survived more often than men: 74% versus 19%;
- survival decreased from 63% in first class to 24% in third class;
- children under 16 survived more often than adults: 59% versus 38%;
- families of 2-4 people had 58% survival, while families of 5 or more had 16%.

The pipeline creates title, family, ticket, surname, cabin, fare, and group features. Group statistics use train and test features without using `Survived`.

## Limitations

- Titanic is small, so a few predictions can change the results.
- Tuning and final comparison use the same dataset.
- Some models were originally tuned with different CV schemes.