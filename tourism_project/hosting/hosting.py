"""Hosting script — push deployment files to HF Docker Space."""
import os, sys
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN   = os.environ["HF_TOKEN"]
SPACE_REPO = "rknv1984/tourism-predictor"
login(token=HF_TOKEN, add_to_git_credential=False)

api = HfApi(token=HF_TOKEN)
try:
    api.repo_info(SPACE_REPO, repo_type="space")
    print(f"Space exists: {SPACE_REPO}")
except RepositoryNotFoundError:
    create_repo(SPACE_REPO, token=HF_TOKEN, repo_type="space", space_sdk="docker", private=False)
    print(f"Created Space: {SPACE_REPO}")

# Resolve deployment folder relative to this script — works on both
# GitHub Actions runner and any local/Colab path
script_dir  = os.path.dirname(os.path.abspath(__file__))
deploy_dir  = os.path.normpath(os.path.join(script_dir, "..", "deployment"))
print(f"Deployment folder: {deploy_dir}")

for fname in ["app.py", "requirements.txt", "Dockerfile", "README.md"]:
    fpath = os.path.join(deploy_dir, fname)
    if not os.path.exists(fpath):
        print(f"⚠️  Missing: {fpath} — skipping")
        continue
    api.upload_file(
        path_or_fileobj=fpath,
        path_in_repo=fname,
        repo_id=SPACE_REPO, repo_type="space",
        token=HF_TOKEN, commit_message=f"Deploy {fname}",
    )
    print(f"✅ Uploaded {fname}")

print(f"\n🚀 App live: https://huggingface.co/spaces/{SPACE_REPO}")
