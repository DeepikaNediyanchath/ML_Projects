import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

DATA_PATH = "data/iris.csv"
MODEL_PATH = "model/iris_pipeline.pkl"

df = pd.read_csv(DATA_PATH)
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Iris-Optuna")

def objective(trial):
    C = trial.suggest_float("C", 0.001, 10, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 500)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=max_iter))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    with mlflow.start_run():
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "model")

    return acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("Best params:", best_params)

# Train final model
best_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(**best_params))
])

best_model.fit(X_train, y_train)
joblib.dump(best_model, MODEL_PATH)
print("Saved best model:", MODEL_PATH)
