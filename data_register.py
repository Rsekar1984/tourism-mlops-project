"""Step 1 of the MLOps pipeline — register raw data on HF Hub."""
import os, shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN     = os.environ["HF_TOKEN"]
DATASET_REPO = "rknv1984/tourism-dataset"
RAW_CSV      = "tourism_project/data/tourism.csv"

Path("tourism_project/data").mkdir(parents=True, exist_ok=True)

api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(DATASET_REPO, repo_type="dataset")
except RepositoryNotFoundError:
    create_repo(DATASET_REPO, token=HF_TOKEN, repo_type="dataset")

if os.path.exists(RAW_CSV):
    api.upload_file(
        path_or_fileobj=RAW_CSV,
        path_in_repo="data/tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message="CI: register raw dataset",
    )
    print("Data registration complete.")
else:
    print(f"[SKIP] {RAW_CSV} not found — skipping upload.")