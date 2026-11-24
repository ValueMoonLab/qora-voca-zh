"""Chinese fork of the upstream teacher-audio generator.

Qwen3-TTS 1.7B Base with the VoiceDesign references, batches of four
(12 GB card), faster-whisper Chinese CER screening at 15%, one retry, peak
normalisation to 0.95. Resumable via SQLite; STOP file halts cleanly.
Output: <root>/audio/<id>.flac, manifest.jsonl (prepare-ready), review.jsonl.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import numpy as np
import soundfile as sf
import torch

MODEL = str(Path(__file__).resolve().parents[1] / "models" / "Qwen3-TTS-12Hz-1.7B-Base")
BATCH = 4


def norm_zh(s):
    return re.sub(r"[^\u4e00-\u9fff]", "", s)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True, help="Directory with F2.wav, M2.wav")
    parser.add_argument("--voices", type=Path, required=True, help="voices.json from design_voices.py")
    parser.add_argument("--limit", type=int, help="Only the first N scripts (smoke test)")
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--voice-list", default="F2,M2", help="Comma-separated voice ids to generate")
    args = parser.parse_args()
    voice_list = tuple(v.strip() for v in args.voice_list.split(",") if v.strip())
    from faster_whisper import WhisperModel
    from jiwer import wer
    from qwen_tts import Qwen3TTSModel
    from scipy.signal import resample_poly
    audio = args.root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.root / "state.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS clips(id TEXT PRIMARY KEY, status TEXT, attempts INTEGER, payload TEXT)")
    db.commit()
    rows = [json.loads(line) for line in args.scripts.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        row["character_count"] = len(norm_zh(row["text"]))
    rows = sorted(rows, key=lambda r: (r["character_count"], r["id"]))[:args.limit]
    jobs = [{"id": f"{row['id']}_{voice}", "voice": voice, "row": row} for row in rows for voice in voice_list]
    references = json.loads(args.voices.read_text(encoding="utf-8"))
    started = time.time()

    def out_of_time():
        return (args.root / "STOP").exists()

    def report(stage):
        counts = dict(db.execute("SELECT status, count(*) FROM clips GROUP BY status").fetchall())
        hours = sum(json.loads(p)["duration_seconds"] for (p,) in db.execute("SELECT payload FROM clips WHERE status='accepted'")) / 3600
        value = {"stage": stage, "target_clips": len(jobs), "counts": counts, "accepted_audio_hours": round(hours, 3),
                 "elapsed_seconds_this_run": round(time.time() - started), "updated": time.time()}
        temporary = args.root / "progress.json.tmp"
        temporary.write_text(json.dumps(value, indent=2))
        os.replace(temporary, args.root / "progress.json")
        print(json.dumps(value), flush=True)

    torch.set_num_threads(6)
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompts = {}
    for v in voice_list:
        meta = references[v]
        # Every published voice is a synthetic VoiceDesign reference with a known
        # transcript; cloning always conditions on that transcript.
        prompts[v] = model.create_voice_clone_prompt(ref_audio=str(args.references / f"{v}.wav"), ref_text=meta["text"],
                                                     x_vector_only_mode=False)[0]
    asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=6)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    futures = []

    def analyze(batch, waves, rate, attempt, seed, batch_id, seconds):
        records = []
        for job, wave in zip(batch, waves):
            row, w = job["row"], np.asarray(wave)
            duration, peak, reasons = len(w) / rate, float(np.max(np.abs(w))) if len(w) else 0.0, []
            if not np.isfinite(w).all() or peak < 1e-5:
                reasons.append("invalid_or_silent_audio")
            if duration < 0.5 or duration > 35:
                reasons.append("duration")
            if duration > 0 and not 1.0 <= row["character_count"] / duration <= 10:
                reasons.append("speaking_rate")
            if len(w) and float(np.mean(np.abs(w) < 1e-4)) > 0.8:
                reasons.append("excessive_silence")
            transcript, error = "", 1.0
            if not reasons:
                segments, _ = asr.transcribe(resample_poly(w.astype("float32"), 2, 3), language="zh", beam_size=3,
                                             condition_on_previous_text=False, initial_prompt="以下是普通话的句子。")
                transcript = "".join(s.text.strip() for s in segments)
                ref, hyp = list(norm_zh(row["text"])), list(norm_zh(transcript))
                error = wer(" ".join(ref), " ".join(hyp) or "<empty>") if ref else 1.0
                if error > 0.15:
                    reasons.append("asr_mismatch")
            if np.isfinite(w).all() and peak > 0:
                sf.write(audio / (job["id"] + ".flac"), w * (0.95 / peak), rate, subtype="PCM_16")
            payload = {"id": job["id"], "text_id": row["id"], "voice_id": job["voice"], "text": row["text"],
                       "split": row.get("split", "train"), "category": row.get("category", "zh"),
                       "origin": row.get("source", "template"), "source_id": row["id"],
                       "source_license": "generated", "source_url": "", "teacher_model": MODEL,
                       "reference_audio": job["voice"] + ".wav", "audio": f"audio/{job['id']}.flac",
                       "duration_seconds": round(duration, 3), "generation_seed": seed, "generation_batch": batch_id,
                       "attempt": attempt, "generation_seconds": seconds,
                       "asr_transcript": transcript, "normalized_cer": round(float(error), 4), "review_reasons": reasons}
            records.append((job["id"], "review" if reasons else "accepted", attempt, json.dumps(payload, ensure_ascii=False)))
        return records

    def collect(future):
        records = future.result()
        db.executemany("INSERT INTO clips(id, status, attempts, payload) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                       "status=excluded.status, attempts=excluded.attempts, payload=excluded.payload", records)
        db.commit()

    def generate(batch, attempt, batch_id):
        seed = args.seed + batch_id + attempt * 100000
        torch.manual_seed(seed)
        start = time.time()
        try:
            waves, rate = model.generate_voice_clone(text=[j["row"]["text"] for j in batch], language=["Chinese"] * len(batch),
                                                     voice_clone_prompt=[prompts[j["voice"]] for j in batch], max_new_tokens=600)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            generate(batch[:middle], attempt, batch_id)
            generate(batch[middle:], attempt, batch_id + 50000)
            return
        futures.append(pool.submit(analyze, batch, waves, rate, attempt, seed, batch_id, round(time.time() - start, 3)))
        if len(futures) >= 2:
            collect(futures.pop(0))

    stopped = False
    for attempt in (1, 2):
        existing = {uid: (status, n) for uid, status, n in db.execute("SELECT id, status, attempts FROM clips")}
        pending = [j for j in jobs if (attempt == 1 and j["id"] not in existing)
                   or (attempt == 2 and j["id"] in existing and existing[j["id"]][0] == "review" and existing[j["id"]][1] < 2)]
        size = BATCH if attempt == 1 else 1
        for start in range(0, len(pending), size):
            if out_of_time():
                stopped = True
                break
            generate(pending[start:start + size], attempt, start // BATCH if attempt == 1 else start + 20000)
            if (start // size) % 16 == 15:
                report("generating")
        while futures:
            collect(futures.pop(0))
        if stopped:
            break
    pool.shutdown(wait=True)
    accepted = [json.loads(p) for (p,) in db.execute("SELECT payload FROM clips WHERE status='accepted' ORDER BY id")]
    (args.root / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in accepted), encoding="utf-8")
    (args.root / "review.jsonl").write_text("".join(p + "\n" for (p,) in db.execute("SELECT payload FROM clips WHERE status!='accepted' ORDER BY id")), encoding="utf-8")
    report("paused" if stopped else "complete")


if __name__ == "__main__":
    main()
