"""Pipeline Step 2 — Load from HF, clean, split, re-upload."""
import os, warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from huggingface_hub import HfApi, login
warnings.filterwarnings("ignore")

HF_TOKEN     = os.environ["HF_TOKEN"]
DATASET_REPO = "rknv1984/tourism-dataset"
login(token=HF_TOKEN, add_to_git_credential=False)

ds = load_dataset(DATASET_REPO, data_files="data/tourism.csv", split="train", token=HF_TOKEN)
df = ds.to_pandas()
df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

for col in [c for c in df.select_dtypes(include=["float64","int64"]).columns if c != "ProdTaken"]:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col].astype(str))

X, y = df.drop("ProdTaken", axis=1), df["ProdTaken"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_df = Xtr.copy(); train_df["ProdTaken"] = ytr.values
test_df  = Xte.copy(); test_df["ProdTaken"]  = yte.values

os.makedirs("tourism_project/data", exist_ok=True)
train_df.to_csv("tourism_project/data/train.csv", index=False)
test_df.to_csv("tourism_project/data/test.csv",   index=False)

api = HfApi(token=HF_TOKEN)
for fname in ["train.csv", "test.csv"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/data/{fname}",
        path_in_repo=f"data/{fname}",
        repo_id=DATASET_REPO, repo_type="dataset",
        token=HF_TOKEN, commit_message=f"CI: upload {fname}",
    )
print("✅ Data preparation complete.")
