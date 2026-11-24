"""Stratified train/validation split for an annotation run."""
import json
import random
from pathlib import Path

rows = [json.loads(l) for l in Path("data/annotated-m035/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rng = random.Random(20261119)
by_voice = {}
for r in rows:
    by_voice.setdefault(r["speaker"], []).append(r)
val = []
for v, items in by_voice.items():
    rng.shuffle(items)
    val.extend(items[:32])
val_ids = {r["id"] for r in val}
train = [r for r in rows if r["id"] not in val_ids]
out = Path("data/annotated-m035")
(out / "validation.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in val), encoding="utf-8")
(out / "train-final.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train), encoding="utf-8")
print("train:", len(train), "validation:", len(val), "overlap:", len({r["id"] for r in train} & val_ids))
