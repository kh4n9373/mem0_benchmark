from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="KhangPTT373/locomo",
    local_dir="locomo_dataset",
    repo_type="dataset"
)
