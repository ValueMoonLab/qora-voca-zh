"""Consolidate three-speaker features into one physical root for training.

train.py forbids feature paths escaping the acoustic root, so ../features/...
relative paths fail. Hardlink every referenced .pt into data/features-3v/<split>/
and rewrite manifest paths to <split>/<hash>.pt. Hardlinks cost no extra disk.
Also rewrite annotations.json acoustic_root.
"""
import json
import os
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
FEAT = PROJ / "data/features-3v"
ANN = PROJ / "data/annotated-v3"

for split in ("train", "validation", "test"):
    dst_dir = FEAT / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest = FEAT / f"{split}.jsonl"
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    linked = 0
    for row in rows:
        old = row["path"]  # ../features/<split>/<hash>.pt or ../features-m3/...
        src = (FEAT / old).resolve()
        name = Path(old).name
        dst = dst_dir / name
        if not dst.exists():
            os.link(src, dst)
            linked += 1
        row["path"] = f"{split}/{name}"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(split, "rows", len(rows), "new links", linked, flush=True)

# rewrite annotated manifests
for split in ("train", "validation", "test"):
    p = ANN / f"{split}.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    for row in rows:
        row["path"] = f"{split}/{Path(row['path']).name}"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print("annotated", split, len(rows), flush=True)

meta_p = ANN / "annotations.json"
meta = json.loads(meta_p.read_text(encoding="utf-8"))
meta["acoustic_root"] = str(FEAT)
meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("acoustic_root ->", FEAT, flush=True)
print("CONSOLIDATE_DONE", flush=True)
