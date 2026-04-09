from huggingface_hub import snapshot_download
from pathlib import Path
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"
repo_id = "Qwen/Qwen2.5-3B"
out_dir = "/root/autodl-tmp/models/Qwen/Qwen2.5-3B"
Path(out_dir).mkdir(parents=True, exist_ok=True)
print(f"START repo={repo_id} out_dir={out_dir}", flush=True)
path = snapshot_download(
    repo_id=repo_id,
    local_dir=out_dir,
    resume_download=True,
    force_download=False,
)
print(f"DONE path={path}", flush=True)
