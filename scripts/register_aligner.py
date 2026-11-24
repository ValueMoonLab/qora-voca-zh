"""Register downloaded aligner weights into the local HF cache layout.

huggingface_hub expects models--<org>--<name>/snapshots/<sha>/<file> with
refs/main pointing at the commit. Files are hardlinked (same E: volume).
Verifies safetensors integrity and loads Wav2Vec2ForCTC with local_files_only.
"""
import json
import os
from pathlib import Path

REPO = "lyonlu13/wav2vec2-large-zh-singing-phoneme-ctc"
SHA = "aa19db9ca380d6aac0f8b5dd5e7c6fa6359f0518"
SRC = Path(str(Path(__file__).resolve().parents[1] / "models\wav2vec2-zh-phoneme-ctc"))
CACHE = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
FILES = ["config.json", "processor_config.json", "tokenizer_config.json", "vocab.json", "model.safetensors"]


def main():
    model_dir = CACHE / ("models--" + REPO.replace("/", "--"))
    snap = model_dir / "snapshots" / SHA
    snap.mkdir(parents=True, exist_ok=True)
    (model_dir / "refs").mkdir(exist_ok=True)
    (model_dir / "refs" / "main").write_text(SHA, encoding="utf-8")
    for name in FILES:
        src, dst = SRC / name, snap / name
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                continue
            dst.unlink()
        if src.stat().st_size == 0:
            raise SystemExit(f"source {name} is empty — relay not finished")
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
        print("linked", name, dst.stat().st_size, flush=True)
    # integrity: safetensors header must parse and tensor count > 0
    from safetensors import safe_open
    with safe_open(str(snap / "model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    print("safetensors tensors:", len(keys), flush=True)
    # end-to-end: the exact loader the annotator uses
    from transformers import Wav2Vec2ForCTC
    from huggingface_hub import hf_hub_download
    m = Wav2Vec2ForCTC.from_pretrained(REPO, local_files_only=True)
    v = json.loads(Path(hf_hub_download(REPO, "vocab.json", local_files_only=True)).read_text(encoding="utf-8"))
    print("loaded:", type(m).__name__, "vocab size:", len(v), "pad:", m.config.pad_token_id, flush=True)
    print("REGISTER_OK", flush=True)


if __name__ == "__main__":
    main()
