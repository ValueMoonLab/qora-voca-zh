"""Parallel range download of the aligner weights from hf-mirror.

Single-stream hf-mirror can be throttled; ranged parallel download is faster.
8 concurrent ranged streams against the CDN should aggregate much higher.
Chunks are verified by size; final file verified by safetensors parse.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

URL = "https://hf-mirror.com/lyonlu13/wav2vec2-large-zh-singing-phoneme-ctc/resolve/main/model.safetensors"
DST = Path(str(Path(__file__).resolve().parents[1] / "models\wav2vec2-zh-phoneme-ctc\model.safetensors"))
TMP = DST.with_suffix(".parts")
SIZE = 1261967332
WORKERS = 8
CHUNK = 16 * 1024 * 1024


def ranges():
    out = []
    start = 0
    while start < SIZE:
        end = min(start + CHUNK, SIZE) - 1
        out.append((start, end))
        start = end + 1
    return out


def fetch(i, lo, hi):
    part = TMP / f"{i:04d}.part"
    want = hi - lo + 1
    for attempt in range(6):
        if part.exists() and part.stat().st_size == want:
            return i, True
        try:
            with requests.get(URL, headers={"Range": f"bytes={lo}-{hi}"}, stream=True, timeout=(15, 120)) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"status {r.status_code}")
                with open(part, "wb") as f:
                    for block in r.iter_content(1 << 20):
                        f.write(block)
            if part.stat().st_size != want:
                raise RuntimeError(f"size {part.stat().st_size} != {want}")
            return i, True
        except Exception as e:
            print(f"chunk {i} attempt {attempt}: {e}", flush=True)
            time.sleep(3 * (attempt + 1))
    return i, False


def main():
    TMP.mkdir(exist_ok=True)
    chunks = ranges()
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, i, lo, hi) for i, (lo, hi) in enumerate(chunks)]
        for fut in futures:
            i, ok = fut.result()
            if not ok:
                print(f"FAILED chunk {i}", flush=True)
                sys.exit(1)
            done += 1
            if done % 10 == 0:
                mb = done * CHUNK / 1e6
                print(f"{done}/{len(chunks)} chunks ~{mb:.0f}MB {time.time()-t0:.0f}s", flush=True)
    # assemble
    with open(DST, "wb") as out:
        for i in range(len(chunks)):
            part = TMP / f"{i:04d}.part"
            with open(part, "rb") as f:
                while True:
                    block = f.read(1 << 22)
                    if not block:
                        break
                    out.write(block)
    assert DST.stat().st_size == SIZE, f"final size {DST.stat().st_size}"
    from safetensors import safe_open
    with safe_open(str(DST), framework="pt") as f:
        n = len(list(f.keys()))
    print(f"ASSEMBLED {SIZE} bytes, {n} tensors, {time.time()-t0:.0f}s", flush=True)
    # cleanup parts
    for p in TMP.iterdir():
        p.unlink()
    TMP.rmdir()
    print("DOWNLOAD_OK", flush=True)


if __name__ == "__main__":
    main()
