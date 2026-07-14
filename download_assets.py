"""
Downloads model and vectorstore from Hugging Face Hub if not already present.
Called at Streamlit app startup.
"""
import os

HF_REPO = os.getenv("HF_REPO", "ranjithkumar111/skin-lesion-ai-assets")

FILES = {
    "models/efficientnet_skin_best.pth": "models/efficientnet_skin_best.pth",
    "vectorstore/faiss.index":           "vectorstore/faiss.index",
    "vectorstore/chunks.pkl":            "vectorstore/chunks.pkl",
}


def download_assets():
    if not HF_REPO:
        return  # running locally — assets already present

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub not installed. Add it to requirements.txt.")

    for local_path, repo_filename in FILES.items():
        if os.path.exists(local_path):
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {repo_filename} from {HF_REPO}...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=repo_filename,
            repo_type="model",
            local_dir=".",
        )
        print(f"  Saved to {local_path}")
