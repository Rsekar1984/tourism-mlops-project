"""Pipeline Step 1 — Register raw data on HF Hub."""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN     = os.environ["HF_TOKEN"]
DATASET_REPO = "rknv1984/tourism-dataset"
login(token=HF_TOKEN, add_to_git_credential=False)

Path("tourism_project/data").mkdir(parents=True, exist_ok=True)
api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(DATASET_REPO, repo_type="dataset")
    print(f"Dataset repo exists: {DATASET_REPO}")
except RepositoryNotFoundError:
    create_repo(DATASET_REPO, token=HF_TOKEN, repo_type="dataset")
    print(f"Created: {DATASET_REPO}")

csv_path = "tourism_project/data/tourism.csv"
if os.path.exists(csv_path):
    api.upload_file(
        path_or_fileobj=csv_path,
        path_in_repo="data/tourism.csv",
        repo_id=DATASET_REPO, repo_type="dataset",
        token=HF_TOKEN, commit_message="CI: register raw dataset",
    )
    print("✅ Data registration complete.")
else:
    print("[SKIP] tourism.csv not in repo — using existing HF data.")
