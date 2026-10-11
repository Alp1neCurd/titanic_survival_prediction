import numpy as np

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, cross_val_score

from config import N_REPEATS, N_SPLITS, SEED


def evaluate_model(model, X, y, groups, n_jobs=-1):
    """Evaluate a model with regular CV, group CV, and a group holdout"""
    group_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    train_index, holdout_index = next(group_cv.split(X, y, groups))

    train_X = X.iloc[train_index]
    holdout_X = X.iloc[holdout_index]

    train_y = y.iloc[train_index]
    holdout_y = y.iloc[holdout_index]
    train_groups = groups.iloc[train_index]

    regular_cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    regular_scores = cross_val_score(model, train_X, train_y, cv=regular_cv, scoring="accuracy", n_jobs=n_jobs)

    group_score_parts = []

    for repeat in range(N_REPEATS):
        repeated_group_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)

        scores = cross_val_score(model, train_X, train_y, cv=repeated_group_cv, groups=train_groups, scoring="accuracy", n_jobs=n_jobs)
        group_score_parts.append(scores)

    group_scores = np.concatenate(group_score_parts)

    holdout_model = clone(model)
    holdout_model.fit(train_X, train_y)
    holdout_predictions = holdout_model.predict(holdout_X)

    regular_mean = regular_scores.mean()
    group_mean = group_scores.mean()

    return {"Regular CV": regular_mean,
            "Group CV": group_mean,
            "Selection Score": (regular_mean + group_mean) / 2,
            "Holdout Accuracy": accuracy_score(holdout_y, holdout_predictions)}