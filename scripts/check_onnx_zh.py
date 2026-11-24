"""ONNX vs PyTorch parity check with Chinese sentences.

Usage:
    python scripts/check_onnx_zh.py \
        --checkpoint release/qora-voca-zh.pt \
        --onnx release/onnx/model.onnx

Exports a fresh parity graph from the checkpoint, compares it against both
PyTorch and the shipped ONNX file, then times the shipped file on CPU.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_voca.export import Exportable, export
from tiny_voca.infer import load_model
from tiny_voca.text import phonemize

ROOT = Path(__file__).resolve().parents[1]

SENTENCES = (
    "今天天气真不错，我们一起去公园散步吧。",
    "人工智能正在改变世界。",
    "你昨天说的那本书，我已经买回来了。",
    "请问火车站怎么走？前面路口左转就到了。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "release/qora-voca-zh.pt"))
    parser.add_argument("--onnx", default=str(ROOT / "release/onnx/model.onnx"))
    parser.add_argument("--output", default=str(ROOT / "onnx-check-zh"))
    args = parser.parse_args()

    OUT = Path(args.output)
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)

    model = load_model(args.checkpoint, "cpu")
    wrapper = Exportable(model, external_noise=True).eval()
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 4
    with tempfile.TemporaryDirectory() as tmp:
        parity_path = Path(tmp) / "parity.onnx"
        export(model, parity_path, external_noise=True)
        exact = ort.InferenceSession(str(parity_path), opts, providers=["CPUExecutionProvider"])
        release = ort.InferenceSession(args.onnx, opts, providers=["CPUExecutionProvider"])
        rate, hop = model.config.sample_rate, model.config.hop

        def feeds(phones, voice, speed=1.0):
            one = lambda v: np.array([v], dtype=np.float32)
            return {"tokens": np.array([phones.ids], dtype=np.int64),
                    "stress": np.array([phones.stress], dtype=np.int64),
                    "voice": np.array([voice], dtype=np.int64),
                    "speed": one(speed), "pitch_shift": one(0.0), "pitch_range": one(1.0)}

        rows = []
        for n, text in enumerate(SENTENCES):
            phones = phonemize(text)
            for voice, name in enumerate(model.config.speakers):
                tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
                ref, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]))
                ref = ref[0, :int(lengths[0])].numpy()
                noise = torch.randn(len(ref), generator=torch.Generator().manual_seed(0))[None]
                one = lambda v: torch.tensor([v])
                with torch.no_grad():
                    wrapper(tokens, stress, torch.tensor([voice]), one(1.0), one(0.0), one(1.0), noise)
                padded = np.zeros((1, hop * model.config.max_frames), dtype=np.float32)
                padded[0, :noise.shape[1]] = noise[0].numpy()
                exported, exp_dur = exact.run(None, {**feeds(phones, voice), "noise": padded})
                peak = float(np.abs(ref).max())
                same_len = exported.shape[1] == len(ref)
                rows.append({"sentence": n, "voice": name, "tokens": len(phones),
                             "seconds": round(len(ref) / rate, 2),
                             "same_durations": bool((exp_dur == details["durations"].numpy()).all()),
                             "same_length": same_len,
                             "onnx_error": float(np.abs(exported[0, :len(ref)] - ref).max() / peak) if same_len else None})
                sf.write(OUT / f"s{n}-{name}-torch.wav", ref, rate)
                sf.write(OUT / f"s{n}-{name}-onnx-exact.wav", exported[0], rate)

        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        bad = [r for r in rows if not r["same_durations"] or not r["same_length"] or (r["onnx_error"] or 1) > 0.02]
        print("PARITY_FAIL" if bad else "PARITY_OK", "max_error=", max(r["onnx_error"] or 0 for r in rows))

        # release export: CPU timing + listening samples
        import statistics, time
        text = SENTENCES[0]
        phones = phonemize(text)
        for voice, name in enumerate(model.config.speakers):
            feed = feeds(phones, voice)
            for _ in range(3):
                release.run(None, feed)
            times = []
            for _ in range(10):
                t0 = time.perf_counter()
                audio = release.run(None, feed)[0]
                times.append(time.perf_counter() - t0)
            seconds = audio.shape[1] / rate
            print(json.dumps({"voice": name, "audio_seconds": round(seconds, 2),
                              "onnx_seconds": round(statistics.median(times), 3),
                              "rtfx": round(seconds / statistics.median(times), 1)}))
            sf.write(OUT / f"release-{name}.wav", audio[0], rate)


if __name__ == "__main__":
    main()
