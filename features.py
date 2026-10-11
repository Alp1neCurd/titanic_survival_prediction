import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


TITLE_MAPPING = {"Mlle": "Miss",
                 "Ms": "Miss",
                 "Mme": "Mrs",
                 "Dr": "Other",
                 "Major": "Other",
                 "Col": "Other",
                 "Capt": "Other",
                 "Rev": "Other",
                 "Don": "Other",
                 "Sir": "Other",
                 "Jonkheer": "Other",
                 "Lady": "Other",
                 "Countess": "Other",
                 "Dona": "Other"}

NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Family_Size", "Alone", "Cabin_Count", "Name_Length", "Fare_Log", "Ticket_Group_Size", "Surname_Group_Size", "Fare_Per_Person",
                    "Ticket_Female_Count","Ticket_Child_Count", "Ticket_Mean_Age", "Surname_Female_Count", "Surname_Child_Count", "Surname_Mean_Age"]

CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck"]

def prepare_passenger_data(data):
    """Create features shared by group statistics and the transformer"""
    data = data.copy()

    data["Title"] = data["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
    data["Title"] = data["Title"].replace(TITLE_MAPPING).fillna("Other")

    data["Family_Size"] = data["SibSp"] + data["Parch"] + 1
    data["Alone"] = (data["Family_Size"] == 1).astype(int)
    data["Surname"] = data["Name"].str.split(",").str[0].str.strip()
    data["Family_Key"] = data["Surname"] + "_" + data["Family_Size"].astype(str)
    data["Is_Female"] = (data["Sex"] == "female").astype(int)
    data["Is_Child"] = (data["Age"] < 16).astype(int)

    return data


def make_group_stats(data, group, prefix):
    """Calculate size and composition statistics for passenger groups"""
    stats = data.groupby(group).agg(Group_Size=(group, "size"),
                                    Female_Count=("Is_Female", "sum"),
                                    Child_Count=("Is_Child", "sum"),
                                    Mean_Age=("Age", "mean"))

    return stats.add_prefix(f"{prefix}_")


class TitanicFeatureEngineer(TransformerMixin, BaseEstimator):
    """Create Titanic features and add stored group statistics"""

    def __init__(self, ticket_stats=None, family_stats=None):
        self.ticket_stats = ticket_stats
        self.family_stats = family_stats

    def prepare_data(self, X):
        """Create features shared by fit and transform"""
        return prepare_passenger_data(X)

    def fit(self, X, y=None):
        data = self.prepare_data(X)

        self.age_by_title_class_ = (
            data.groupby(["Title", "Pclass"])["Age"].median().to_dict())
        
        self.age_by_title_ = data.groupby("Title")["Age"].median().to_dict()
        self.global_age_median_ = data["Age"].median()
        
        if self.ticket_stats is None:
            self.ticket_stats_ = make_group_stats(data, "Ticket", "Ticket")
        else:
            self.ticket_stats_ = self.ticket_stats

        if self.family_stats is None:
            self.family_stats_ = make_group_stats(data, "Family_Key", "Surname")
        else:
            self.family_stats_ = self.family_stats

        return self

    def transform(self, X):
        data = self.prepare_data(X)

        group_age = pd.Series([self.age_by_title_class_.get((title, pclass), np.nan) for title, pclass in zip(data["Title"], data["Pclass"])], index=data.index)

        data["Age"] = (data["Age"].fillna(group_age).fillna(data["Title"].map(self.age_by_title_)).fillna(self.global_age_median_))

        data = data.join(self.ticket_stats_, on="Ticket")
        data = data.join(self.family_stats_, on="Family_Key")

        fallbacks = {"Female_Count": "Is_Female", "Child_Count": "Is_Child", "Mean_Age": "Age"}

        for prefix in ["Ticket", "Surname"]:
            data[f"{prefix}_Group_Size"] = (data[f"{prefix}_Group_Size"].fillna(1).astype(int))

            for feature, fallback in fallbacks.items():
                column = f"{prefix}_{feature}"
                data[column] = data[column].fillna(data[fallback])

        single_passenger = data["Family_Size"] == 1
        data.loc[single_passenger, "Surname_Group_Size"] = 1
        data.loc[single_passenger, "Surname_Female_Count"] = data.loc[single_passenger, "Is_Female"]
        data.loc[single_passenger, "Surname_Child_Count"] = data.loc[single_passenger, "Is_Child"]
        data.loc[single_passenger, "Surname_Mean_Age"] = data.loc[single_passenger, "Age"]

        data["Deck"] = data["Cabin"].str[0].fillna("U").replace({"T": "Other"})
        data["Cabin_Count"] = data["Cabin"].fillna("").str.split().str.len()
        data["Fare_Per_Person"] = (data["Fare"] / data["Ticket_Group_Size"].clip(lower=1))
        data["Name_Length"] = data["Name"].str.len()
        data["Fare_Log"] = np.log1p(data["Fare"])
        data["Pclass"] = data["Pclass"].astype(str)

        return data.drop(columns=["PassengerId",
                                  "Name",
                                  "Ticket",
                                  "Cabin",
                                  "Surname",
                                  "Family_Key",
                                  "Is_Female",
                                  "Is_Child"], errors="ignore")


def make_feature_engineer(train_features, test_features):
    """Build the feature transformer from the available passenger manifest"""
    manifest = pd.concat([train_features, test_features], ignore_index=True)
    manifest = prepare_passenger_data(manifest)

    ticket_stats = make_group_stats(manifest, "Ticket", "Ticket")
    family_stats = make_group_stats(manifest, "Family_Key", "Surname")

    return TitanicFeatureEngineer(ticket_stats=ticket_stats, family_stats=family_stats)