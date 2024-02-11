import os
from huggingface_hub import snapshot_download


def get_script_dir():
    return os.path.dirname(
        os.path.abspath(__file__))


def build_download_path(repo_name):
    return f"{get_script_dir()}/{repo_name}"


def make_download_folders(download_path):
    os.makedirs(download_path)


repo_name = "MBZUAI/LaMini-Flan-T5-248M"
print(f"Downloading {repo_name}")
download_path = build_download_path(repo_name)
make_download_folders(download_path)
snapshot_download(repo_id=repo_name,
                  local_dir=download_path,
                  local_dir_use_symlinks=False)
