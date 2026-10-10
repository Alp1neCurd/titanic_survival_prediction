from catboost import CatBoostClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATBOOST_PARAMS, DNN_PARAMS, SEED
from dnn import TorchDNNClassifier
from features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_feature_engineer)


def make_preprocessor(numeric_features=NUMERIC_FEATURES, categorical_features=CATEGORICAL_FEATURES):
    """Build numeric and categorical preprocessing"""
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())])

    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer([("numeric", numeric_pipeline, numeric_features),
                              ("categorical", categorical_pipeline, categorical_features)])


def make_model_pipeline(train_features, test_features, model_name="catboost",):
    """Build a preprocessing and modeling pipeline."""
    feature_engineer = make_feature_engineer(train_features, test_features)

    if model_name == "catboost":
        model = CatBoostClassifier(**CATBOOST_PARAMS)
    elif model_name == "dnn":
        model = TorchDNNClassifier(parameters=DNN_PARAMS, random_state=SEED,)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("features", feature_engineer),
                     ("preprocessor", make_preprocessor()),
                     ("model", model)])