"""
Titanic survival prediction - Python / scikit-learn logistic regression.

Every step prints what it does, so the whole analysis can be followed from
the terminal output alone:

  1. Load train.csv and explore it
  2. Clean the data and engineer features
  3. Fit a logistic regression and measure training accuracy
  4. Load test.csv, apply the same changes and predict
  5. Measure test-set accuracy and save the predictions
"""

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Inside Docker DATA_DIR=/app/data (set in the Dockerfile); locally it defaults to src/data
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
OUTPUT_FILE = DATA_DIR / "predictions_python.csv"
RANDOM_STATE = 42

COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
# Fixed category lists so train and test always get identical dummy columns.
# The first category of each is the reference level dropped by one-hot encoding.
CATEGORIES = {
    "Pclass": [1, 2, 3],
    "Embarked": ["C", "Q", "S"],
    "Title": ["Mr/Mrs", "Miss", "Master", "Rare"],
}
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def banner(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        sys.exit(
            f"\nERROR: could not find {path}\n"
            "Download the data from https://www.kaggle.com/competitions/titanic/data,\n"
            "put train.csv and test.csv in src/data/ and mount that folder into the\n"
            "container with -v (see README.md)."
        )
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def pct(series):
    return series.map("{:.1%}".format).to_string()


def explore(train):
    banner("STEP 1 - Explore train.csv")
    print("\nFirst 5 rows:")
    print(train.head().to_string())
    print("\nColumn types:")
    print(train.dtypes.to_string())
    missing = train.isna().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0].to_string())
    print(f"\nOverall survival rate: {train['Survived'].mean():.1%}")
    print("\nSurvival rate by sex:")
    print(pct(train.groupby("Sex")["Survived"].mean()))
    print("\nSurvival rate by passenger class:")
    print(pct(train.groupby("Pclass")["Survived"].mean()))


def extract_title(names):
    """'Braund, Mr. Owen Harris' -> 'Mr'. Uncommon titles are grouped as 'Rare'."""
    titles = names.str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    titles = titles.replace(TITLE_ALIASES)
    return titles.where(titles.isin(COMMON_TITLES), "Rare")


def learn_fill_values(train):
    """Fill values come from the training data only, then are reused for test."""
    titles = extract_title(train["Name"])
    return {
        "age_by_title": train.groupby(titles)["Age"].median(),
        "age_overall": train["Age"].median(),
        "embarked_mode": train["Embarked"].mode()[0],
        "fare_median": train["Fare"].median(),
    }


def prepare(df, fill, name):
    """Apply the same cleaning and feature steps to any split, printing every change."""
    df = df.copy()
    tag = f"[{name}]"
    print(f"\n{tag} Starting shape: {df.shape[0]} rows x {df.shape[1]} columns")

    df["Title"] = extract_title(df["Name"])
    counts = ", ".join(f"{k}={v}" for k, v in df["Title"].value_counts().items())
    print(f"{tag} Added 'Title' parsed from Name (uncommon titles grouped as 'Rare'): {counts}")

    n = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Title"].map(fill["age_by_title"])).fillna(fill["age_overall"])
    print(f"{tag} Filled {n} missing 'Age' values with the training median age for the passenger's title")

    n = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(fill["embarked_mode"])
    print(f"{tag} Filled {n} missing 'Embarked' values with the most common port ('{fill['embarked_mode']}')")

    n = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fill["fare_median"])
    print(f"{tag} Filled {n} missing 'Fare' values with the training median fare ({fill['fare_median']:.2f})")

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    print(f"{tag} Added 'HasCabin' (1 if a cabin number is recorded): {df['HasCabin'].sum()} of {len(df)} passengers")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"{tag} Added 'FamilySize' (SibSp + Parch + 1) and 'IsAlone': {df['IsAlone'].sum()} passengers travelled alone")

    df["Sex"] = (df["Sex"] == "male").astype(int)
    print(f"{tag} Encoded 'Sex' as 0 = female, 1 = male")

    df["Title"] = df["Title"].replace({"Mr": "Mr/Mrs", "Mrs": "Mr/Mrs"})
    print(f"{tag} Merged titles 'Mr' and 'Mrs' into 'Mr/Mrs': with 'Sex' in the model they add no extra "
          "information, and keeping both would make 'Sex' and 'Title' redundant")

    df = df.drop(columns=DROP_COLUMNS)
    print(f"{tag} Dropped columns not used as features: {', '.join(DROP_COLUMNS)}")

    for col, cats in CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    df = pd.get_dummies(df, columns=list(CATEGORIES), drop_first=True, dtype=int)
    print(f"{tag} One-hot encoded {', '.join(CATEGORIES)} -> final shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def main():
    print(f"Reading data from: {DATA_DIR}")

    # ---------------------------------------------------------------- step 1
    train_raw = load_csv("train.csv")
    explore(train_raw)

    # ---------------------------------------------------------------- step 2
    banner("STEP 2 - Clean the data and engineer features")
    fill = learn_fill_values(train_raw)
    print("Median age by title (learned from train, used to fill missing ages):")
    print(fill["age_by_title"].round(1).to_string())
    train = prepare(train_raw, fill, "train")
    print("\nSurvival rate by title:")
    print(pct(train_raw.groupby(extract_title(train_raw["Name"]))["Survived"].mean()))
    print("\nFirst 5 rows after preparation:")
    print(train.head().to_string())

    # ---------------------------------------------------------------- step 3
    banner("STEP 3 - Train a logistic regression model")
    X_train = train.drop(columns="Survived")
    y_train = train["Survived"]
    print(f"Features ({X_train.shape[1]}): {', '.join(X_train.columns)}")
    print("Model: StandardScaler -> LogisticRegression (L2 penalty, C=1.0)")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)

    coefs = pd.Series(model[-1].coef_[0], index=X_train.columns).sort_values()
    print("\nCoefficients on standardized features (positive = more likely to survive):")
    print(coefs.round(3).to_string())

    train_acc = accuracy_score(y_train, model.predict(X_train))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"\nTraining accuracy:               {train_acc:.4f}")
    print(f"5-fold cross-validated accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ---------------------------------------------------------------- step 4
    banner("STEP 4 - Predict on test.csv")
    test_raw = load_csv("test.csv")
    test = prepare(test_raw, fill, "test")
    X_test = test.reindex(columns=X_train.columns, fill_value=0)
    predictions = model.predict(X_test)
    print(f"\nPredicted {predictions.sum()} survivors out of {len(predictions)} test passengers "
          f"({predictions.mean():.1%})")

    submission = pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Survived": predictions})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved predictions to {OUTPUT_FILE}")

    # ---------------------------------------------------------------- step 5
    banner("STEP 5 - Test-set accuracy")
    test_acc = None
    if "Survived" in test_raw.columns:
        test_acc = accuracy_score(test_raw["Survived"], predictions)
        label = "true labels"
    else:
        print("test.csv has no 'Survived' column: Kaggle keeps the true test labels hidden.")
        reference_path = DATA_DIR / "gender_submission.csv"
        if reference_path.exists():
            reference = pd.read_csv(reference_path)
            merged = submission.merge(reference, on="PassengerId", suffixes=("_pred", "_ref"))
            test_acc = (merged["Survived_pred"] == merged["Survived_ref"]).mean()
            label = "gender_submission.csv"
            print("Using gender_submission.csv (Kaggle's sample answer: all women survive, all men die)")
            print("as the reference labels for the test set.")
        else:
            print("gender_submission.csv not found either, so test accuracy cannot be computed.")
    if test_acc is not None:
        print(f"\nTest accuracy (vs {label}): {test_acc:.4f}")
    print(f"For the official score, upload {OUTPUT_FILE.name} to the Kaggle competition.")

    # --------------------------------------------------------------- summary
    banner("SUMMARY")
    print(f"Training accuracy:               {train_acc:.4f}")
    print(f"5-fold cross-validated accuracy: {cv_scores.mean():.4f}")
    if test_acc is not None:
        print(f"Test accuracy (vs {label}): {test_acc:.4f}")
    print(f"Predictions file:                {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
