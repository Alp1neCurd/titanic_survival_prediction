# Titanic Survival Prediction

Classification models for Kaggle Titanic

CatBoost One-Hot was selected by local cross-validation
Gradient Boosting achieved the best submitted public score

## Files

- `main.py` - evaluation, training and submission export
- `config.py` - paths, model selection and parameters
- `features.py` - feature engineering
- `model.py` - preprocessing and model construction
- `validation.py` - regular CV, group CV and holdout evaluation
- `dnn.py` - PyTorch classifier
- `titanic_experiments.ipynb` - EDA, tuning and model comparison

## Setup

Tested with Python 3.14.6

Download `train.csv` and `test.csv` from the [competition](https://www.kaggle.com/competitions/titanic/data) and place them in `data/`

From the project directory:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

For DNN:

```powershell
py -m pip install -r requirements_dnn.txt
```

For the complete notebook environment:

```powershell
py -m pip install -r requirements_notebook.txt
```

On Linux and macOS:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use `python` instead of `py` for subsequent commands on Linux and macOS

## Run

Run the model selected by `MODEL_NAME` in `config.py`, currently CatBoost:

```powershell
py main.py
```

Override the model:

```powershell
py main.py --model catboost
py main.py --model dnn
```

The script prints validation metrics, fits the model on all labelled passengers and saves a CSV in `submissions/production/`

The DNN runs on CPU
Categorical embeddings and other experimental models are available in the notebook

## Method

The pipeline creates title, family, ticket, surname, cabin and fare features
Group statistics use train and test features without using `Survived`
Age imputation and other learned preprocessing are fitted inside each training fold

A group holdout separates development and holdout passengers by ticket
On the development subset, Regular CV uses repeated stratified splits and Group CV keeps shared tickets together
The selection score is the mean of Regular CV and Group CV
The holdout is a development check and is not used for model selection

## Local results

Higher scores are better

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

CatBoost and Equal Soft Voting differ by only `0.0007` on these splits
CatBoost was kept as a simpler single model; this small difference does not establish a reliable ranking

## Kaggle results

| Submission | Public score |
|---|---:|
| **Gradient Boosting** | **0.78708** |
| Stacking Ridge | 0.78229 |
| CatBoost Native | 0.77990 |
| Bagging Decision Tree | 0.77990 |
| Equal Hard Voting | 0.77751 |
| CatBoost One-Hot, full fit | 0.77511 |
| CatBoost One-Hot, five-fold | 0.77511 |
| Equal Soft Voting | 0.77511 |
| PyTorch DNN Embeddings | 0.77511 |
| SVC | 0.77033 |
| Random Forest | 0.76794 |
| LightGBM | 0.76794 |
| PyTorch DNN | 0.76555 |
| Sex Rule | 0.76555 |

Gradient Boosting had the highest public score, while CatBoost had the highest local selection score
The public leaderboard was used as an external check rather than the model selection criterion

The notebook exports full-fit and five-fold submissions for the selected model, plus the embedding DNN
Other public scores came from earlier full-fit experiments

Embeddings lowered the local DNN selection score but raised its public score, so they did not show a consistent improvement

## Limitations

- Titanic is small, so a few predictions can change the results
- Tuning and final comparison reuse the same development data
- Some models were tuned with different CV schemes
- DNN results may vary across hardware and PyTorch versions