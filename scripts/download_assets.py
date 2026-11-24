"""Download pipeline assets: the Mandarin CTC aligner (ModelScope / hf-mirror)."""
import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
ROOT = str(Path(__file__).resolve().parents[1])
os.makedirs(ROOT, exist_ok=True)

def ms(model_id, local):
    from modelscope import snapshot_download
    p = snapshot_download(model_id, local_dir=os.path.join(ROOT, local))
    print("MS_OK", model_id, p, flush=True)

def hf(repo_id, local):
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id, local_dir=os.path.join(ROOT, local), max_workers=4)
    print("HF_OK", repo_id, p, flush=True)

jobs = [
    ("ms", "Qwen/Qwen3-TTS-Tokenizer-12Hz", "models/Qwen3-TTS-Tokenizer-12Hz"),
    ("ms", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "models/Qwen3-TTS-12Hz-1.7B-Base"),
    ("ms", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    ("hf", "lyonlu13/wav2vec2-large-zh-singing-phoneme-ctc", "models/wav2vec2-zh-phoneme-ctc"),
]
for kind, rid, local in jobs:
    for attempt in range(3):
        try:
            (ms if kind == "ms" else hf)(rid, local)
            break
        except Exception as e:
            print("RETRY", rid, attempt, repr(e)[:300], flush=True)
            time.sleep(10)
    else:
        print("FAILED", rid, flush=True)
print("ALL_DONE", flush=True)
