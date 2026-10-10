import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, cross_val_score

from config import (N_REPEATS, N_SPLITS, SEED, SUBMISSION_DIR, SUBMISSION_PATH, TARGET, TEST_PATH, TRAIN_PATH)

from model import make_model_pipeline


def load_data():
    """Load Titanic train and test datasets"""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def evaluate_model(model, X, y, groups):
    """Evaluate the selected model with the notebook validation scheme"""
    group_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    train_index, holdout_index = next(group_cv.split(X, y, groups))

    train_X = X.iloc[train_index]
    holdout_X = X.iloc[holdout_index]

    train_y = y.iloc[train_index]
    holdout_y = y.iloc[holdout_index]
    train_groups = groups.iloc[train_index]

    regular_cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    regular_scores = cross_val_score(model, train_X, train_y, cv=regular_cv, scoring="accuracy", n_jobs=-1)

    group_score_parts = []

    for repeat in range(N_REPEATS):
        repeated_group_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)

        scores = cross_val_score(model, train_X, train_y, cv=repeated_group_cv, groups=train_groups, scoring="accuracy", n_jobs=-1)
        group_score_parts.append(scores)

    group_scores = np.concatenate(group_score_parts)

    holdout_model = clone(model)
    holdout_model.fit(train_X, train_y)
    holdout_predictions = holdout_model.predict(holdout_X)

    regular_mean = regular_scores.mean()
    group_mean = group_scores.mean()
    selection_score = (regular_mean + group_mean) / 2
    holdout_accuracy = accuracy_score(holdout_y, holdout_predictions)

    return {"Regular CV": regular_mean,
            "Group CV": group_mean,
            "Selection Score": selection_score,
            "Holdout Accuracy": holdout_accuracy}


def save_submission(model, X, y, test_df):
    """Train on all labelled passengers and save the submission"""
    final_model = clone(model)
    final_model.fit(X, y)

    predictions = final_model.predict(test_df).astype(int)
    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], TARGET: predictions})

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)

    return predictions


def main():
    train_df, test_df = load_data()

    X = train_df.drop(columns=TARGET)
    y = train_df[TARGET]
    groups = X["Ticket"]

    model = make_model_pipeline(X, test_df)
    metrics = evaluate_model(model, X, y, groups)
    predictions = save_submission(model, X, y, test_df)

    print("Model: CatBoost One-Hot")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("Predicted survivors:", predictions.sum())
    print("Submission saved to:", SUBMISSION_PATH)


if __name__ == "__main__":
    main()