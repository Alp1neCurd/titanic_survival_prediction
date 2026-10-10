from catboost import CatBoostClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATBOOST_PARAMS
from features import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_feature_engineer)


def make_preprocessor():
    """Build numeric and categorical preprocessing"""
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())])

    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer([("numeric", numeric_pipeline, NUMERIC_FEATURES),
                              ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)])


def make_model_pipeline(train_features, test_features):
    """Build the selected CatBoost One-Hot pipeline"""
    feature_engineer = make_feature_engineer(train_features, test_features)

    model = CatBoostClassifier(**CATBOOST_PARAMS)

    return Pipeline([("features", feature_engineer),
                     ("preprocessor", make_preprocessor()),
                     ("model", model)])