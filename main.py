import argparse

import pandas as pd
from sklearn.base import clone

from config import MODEL_NAMES, SUBMISSION_PATHS, TARGET, TEST_PATH, TRAIN_PATH

from model import make_model_pipeline
from validation import evaluate_model


def parse_args():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a Titanic survival model.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="catboost", help="Model to run.",)

    return parser.parse_args()


def load_data():
    """Load the Titanic train and test datasets."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def save_submission(model, X, y, test_df, submission_path):
    """Fit a model on all labeled data and save its predictions."""
    final_model = clone(model)
    final_model.fit(X, y)

    predictions = final_model.predict(test_df).astype(int)
    submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], TARGET: predictions,})

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    return predictions


def main():
    args = parse_args()
    train_df, test_df = load_data()

    X = train_df.drop(columns=TARGET)
    y = train_df[TARGET]
    groups = X["Ticket"]

    model = make_model_pipeline(X, test_df, model_name=args.model,)

    n_jobs = 1 if args.model == "dnn" else -1
    metrics = evaluate_model(model, X, y, groups, n_jobs=n_jobs,)

    submission_path = SUBMISSION_PATHS[args.model]
    predictions = save_submission(model, X, y, test_df, submission_path,)

    print("Model:", MODEL_NAMES[args.model])

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("Predicted survivors:", predictions.sum())
    print("Submission saved to:", submission_path)


if __name__ == "__main__":
    main()