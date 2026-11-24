"""Split the generated manifest into train/validation/test feature inputs.

The annotation gate marks a run complete only when all three splits exist
and rejects any text_hash shared across splits, so split by SENTENCE
(each sentence has F2+M2 clips; both voices stay in the same split).
"""
import json
import random
from pathlib import Path

root = Path("data/generated")
rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rng = random.Random(20261119)
by_text = {}
for r in rows:
    by_text.setdefault(r["text_id"], []).append(r)
texts = list(by_text)
rng.shuffle(texts)
val_texts, test_texts = set(texts[:32]), set(texts[32:64])
subsets = {
    "train": [r for r in rows if r["text_id"] not in val_texts and r["text_id"] not in test_texts],
    "validation": [r for t in val_texts for r in by_text[t]],
    "test": [r for t in test_texts for r in by_text[t]],
}
out = Path("data/split-inputs")
out.mkdir(exist_ok=True)
for name, subset in subsets.items():
    for r in subset:
        r["split"] = name
        # audio paths are resolved relative to the manifest's directory
        r["audio"] = "../generated/" + r["audio"]
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
    print(name, len(subset))
ids = [set(r["id"] for r in s) for s in subsets.values()]
print("overlap:", len(ids[0] & ids[1]), len(ids[0] & ids[2]), len(ids[1] & ids[2]))
