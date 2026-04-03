import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN   = os.environ.get("HF_TOKEN")
SPACE_REPO = "rknv1984/tourism-predictor"

api = HfApi(token=HF_TOKEN)

# Create Space if it does not exist yet
try:
    api.repo_info(SPACE_REPO, repo_type="space")
    print(f"Space already exists: {SPACE_REPO}")
except RepositoryNotFoundError:
    create_repo(SPACE_REPO, token=HF_TOKEN, repo_type="space", space_sdk="docker")
    print(f"Created Space: {SPACE_REPO}")

# Upload all deployment files
FILES = ["app.py", "requirements.txt", "Dockerfile", "README.md"]
for fname in FILES:
    local = f"tourism_project/deployment/{fname}"
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=fname,
        repo_id=SPACE_REPO,
        repo_type="space",
        token=HF_TOKEN,
        commit_message=f"Deploy {fname}",
    )
    print(f"Uploaded {fname} to Space")

print(f"Space URL: https://huggingface.co/spaces/{SPACE_REPO}")