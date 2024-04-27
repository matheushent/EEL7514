"""Train a Decision Tree model, a Gradient Boosting model and a Random Forest using Grid Search with hold-out validation."""  # noqa: E501

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

balanced_accuracy = make_scorer(balanced_accuracy_score)

df = pd.read_csv("datasets/Tweets.csv")

df["target"] = df["sentiment"].replace({"neutral": 0, "positive": 0, "negative": 1})
df = df.dropna()
df = df.drop(columns=["textID", "selected_text", "sentiment"])

X_train_val, X_test, y_train_val, y_test = train_test_split(
    df["text"], df["target"], test_size=0.2, stratify=df["target"], random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,  # 25% of 80% is 20% of the total data
    stratify=y_train_val,
    random_state=42,
)

split_index = [-1] * len(X_train) + [0] * len(X_val)
X_train_val_combined = np.concatenate((X_train, X_val))
y_train_val_combined = np.concatenate((y_train, y_val))
ps = PredefinedSplit(test_fold=split_index)

cpus_for_training = os.cpu_count() - 1

print("Using {} CPUs for training".format(cpus_for_training))

pipeline_dt = Pipeline(
    [
        ("preprocessing", CountVectorizer()),
        ("classifier", DecisionTreeClassifier()),
    ]
)
param_grid_dt = {
    "classifier__max_depth": list(range(1, 21)),
    "classifier__min_samples_split": list(range(2, 6)),
    "classifier__min_samples_leaf": list(range(2, 21)),
    "classifier__max_leaf_nodes": list(range(2, 51)),
    "classifier__min_impurity_decrease": np.geomspace(1e-4, 1e-2, 20),
    "classifier__ccp_alpha": np.geomspace(1e-4, 1.5e-2, 20),
}
param_grid_gb = {
    "classifier__n_estimators": np.linspace(10, 1000, 50, dtype=int),
    "classifier__learning_rate": np.geomspace(3e-3, 1, 20),
    "classifier__min_samples_split": list(range(2, 6)),
    "classifier__min_samples_leaf": list(range(2, 21)),
    "classifier__max_depth": list(range(1, 21)),
}
param_grid_rf = {
    "classifier__n_estimators": 10 ** np.arange(1, 5),
    "classifier__min_samples_split": list(range(2, 6)),
    "classifier__min_samples_leaf": list(range(2, 21)),
    "classifier__max_leaf_nodes": list(range(2, 51)),
    "classifier__min_impurity_decrease": np.geomspace(1e-4, 1e-2, 20),
    "classifier__ccp_alpha": np.geomspace(1e-4, 1.5e-2, 20),
}

grid_search_dt = GridSearchCV(
    pipeline_dt, param_grid_dt, scoring=balanced_accuracy, n_jobs=cpus_for_training, verbose=2, cv=ps
)

pipeline_rf = Pipeline(
    [
        ("preprocessing", CountVectorizer()),
        ("classifier", RandomForestClassifier()),
    ]
)
grid_search_rf = GridSearchCV(
    pipeline_rf, param_grid_rf, scoring=balanced_accuracy, n_jobs=cpus_for_training, verbose=2, cv=ps
)
pipeline_gb = Pipeline(
    [
        ("preprocessing", CountVectorizer()),
        ("classifier", GradientBoostingClassifier()),
    ]
)
grid_search_gb = GridSearchCV(
    pipeline_gb, param_grid_gb, scoring=balanced_accuracy, n_jobs=cpus_for_training, verbose=2, cv=ps
)

grid_search_dict = {
    "Random Forest": grid_search_rf,
    "Gradient Boosting": grid_search_gb,
    "Decision Tree": grid_search_dt,
}

for model_name, grid_search in grid_search_dict.items():
    print(f"Start training the {model_name} model.")
    grid_search.fit(X_train_val_combined, y_train_val_combined)

for model_name, grid_search in grid_search_dict.items():
    print(f"Best params ({model_name}):", grid_search.best_params_)
    print(f"Best score ({model_name}):", grid_search.best_score_)
