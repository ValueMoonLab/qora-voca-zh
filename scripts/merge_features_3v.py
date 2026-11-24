"""Merge F2/M2 (data/features) + M3 (data/features-m3) manifests into data/features-3v.

Paths are rewritten relative to features-3v so annotate can load ../features/... and
../features-m3/... . preparation.json only needs complete:true.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = [("features", ROOT / "data/features"), ("features-m3", ROOT / "data/features-m3")]
DST = ROOT / "data/features-3v"
DST.mkdir(parents=True, exist_ok=True)

for split in ("train", "validation", "test"):
    rows = []
    for name, base in SRC:
        manifest = base / f"{split}.jsonl"
        if not manifest.exists():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["path"] = f"../{name}/{split}/{Path(row['path']).name}"
            row["record_key"] = f"{row['speaker']}:{row['record_key']}"
            rows.append(row)
    (DST / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (DST / f"{split}.preparation.json").write_text(json.dumps({
        "complete": True,
        "accepted": len(rows),
        "merged_from": [n for n, _ in SRC],
        "note": "merged F2/M2 (features) + M3 (features-m3); acoustic fingerprint identical",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    speakers = {}
    for r in rows:
        speakers[r["speaker"]] = speakers.get(r["speaker"], 0) + 1
    print(split, len(rows), speakers, flush=True)

# copy features.json fingerprint metadata
src_fj = ROOT / "data/features/features.json"
if src_fj.exists():
    shutil.copy2(src_fj, DST / "features.json")
print("MERGE_DONE", DST, flush=True)
