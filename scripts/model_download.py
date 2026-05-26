from modelscope import snapshot_download

snapshot_download(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    cache_dir="models",
)
