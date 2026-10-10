from pathlib import Path

SEED = 42
N_SPLITS = 5
N_REPEATS = 3

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
SUBMISSION_DIR = PROJECT_DIR / "submissions" / "production"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

TARGET = "Survived"

MODEL_NAMES = {"catboost": "CatBoost One-Hot",
               "dnn": "PyTorch DNN"}

SUBMISSION_PATHS = {"catboost": SUBMISSION_DIR / "titanic_catboost_one_hot.csv",
                    "dnn": SUBMISSION_DIR / "titanic_pytorch_dnn.csv"}

# Best CatBoost One-Hot parameters from group-aware Optuna tuning
CATBOOST_PARAMS = {"iterations": 200,
                   "learning_rate": 0.04299542921255021,
                   "depth": 3,
                   "l2_leaf_reg": 14.407964235716207,
                   "random_strength": 0.894679420410602,
                   "border_count": 32,
                   "loss_function": "Logloss",
                   "random_seed": SEED,
                   "verbose": 0,
                   "allow_writing_files": False,
                   "thread_count": 1}

# Best DNN parameters from regular stratified Optuna tuning
DNN_PARAMS = {"hidden_size_1": 16,
              "hidden_size_2": 4,
              "dropout_1": 0.2,
              "dropout_2": 0.0,
              "learning_rate": 0.00019986730447841636,
              "weight_decay": 7.527339029445822e-05,
              "batch_size": 16,
              "epochs": 90,
              "activation": "ELU",
              "batch_norm": False,
              "optimizer": "Adam",
              "scheduler": "None"}