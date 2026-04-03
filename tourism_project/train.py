"""Pipeline Step 3 — Train XGBoost, log to MLflow, register on HF Hub."""
import os, warnings, joblib
import mlflow, mlflow.sklearn
import pandas as pd
from datasets import load_dataset
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError
warnings.filterwarnings("ignore")

HF_TOKEN     = os.environ["HF_TOKEN"]
DATASET_REPO = "rknv1984/tourism-dataset"
MODEL_REPO   = "rknv1984/tourism-project-model"
login(token=HF_TOKEN, add_to_git_credential=False)

train_df = load_dataset(DATASET_REPO, data_files="data/train.csv", split="train", token=HF_TOKEN).to_pandas()
test_df  = load_dataset(DATASET_REPO, data_files="data/test.csv",  split="train", token=HF_TOKEN).to_pandas()
X_train, y_train = train_df.drop("ProdTaken", axis=1), train_df["ProdTaken"]
X_test,  y_test  = test_df.drop("ProdTaken",  axis=1), test_df["ProdTaken"]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(random_state=42, eval_metric="logloss",
                          use_label_encoder=False, verbosity=0))
])
param_grid = {
    "xgb__n_estimators":  [100, 200],
    "xgb__max_depth":     [3, 5],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__subsample":     [0.8],
}

mlflow.set_experiment("Tourism-XGBoost-CI")
with mlflow.start_run(run_name="ci_gridsearch"):
    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    mlflow.log_params(gs.best_params_)
    mlflow.log_metric("best_cv_f1", gs.best_score_)
    best = gs.best_estimator_
    y_pred = (best.predict_proba(X_test)[:, 1] >= 0.45).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metrics({"test_accuracy": report["accuracy"], "test_f1": report["1"]["f1-score"]})

os.makedirs("tourism_project/model_building", exist_ok=True)
joblib.dump(best, "tourism_project/model_building/best-tourism-model-v1.joblib")

api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(MODEL_REPO, repo_type="model")
except RepositoryNotFoundError:
    create_repo(MODEL_REPO, token=HF_TOKEN, repo_type="model")

api.upload_file(
    path_or_fileobj="tourism_project/model_building/best-tourism-model-v1.joblib",
    path_in_repo="best-tourism-model-v1.joblib",
    repo_id=MODEL_REPO, repo_type="model",
    token=HF_TOKEN, commit_message="CI: register best model",
)
print("✅ Training pipeline complete.")
