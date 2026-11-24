"""Design brighter, more articulate male teacher voice candidates (M3).

Screen with faster-whisper CER + spectral clarity metrics; save top candidates.
"""
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
OUT = ROOT / "data" / "voices-m3"
OUT.mkdir(parents=True, exist_ok=True)
REF_TEXT = "今天天气很好，我们出去走走吧。公园里有很多人在散步，孩子们在草地上跑来跑去。"
DESIGNS = [
    "一位二十多岁的年轻男性，音色清亮干净、朝气蓬勃，普通话标准，吐字清晰有力，语速适中，语气阳光自信，像年轻的电台主播，声音明亮突出、毫不含糊。",
    "一位二十五岁左右的年轻男播音员，嗓音清朗透亮、字正腔圆，普通话非常标准，每个字都咬得清楚干脆，语气轻快有活力，声音干净没有沙哑和含糊。",
    "一位二十出头的年轻男性，声音清亮通透、共鸣集中在中高频，普通话标准，吐字干净利落，语速适中，语气专业而年轻，音质清晰锐利、突出靠前。",
]


def norm(s):
    return re.sub(r"[^\u4e00-\u9fff]", "", s)


def cer(ref, hyp):
    import jiwer
    r, h = list(norm(ref)), list(norm(hyp))
    if not r:
        return 1.0
    return jiwer.wer(" ".join(r), " ".join(h) or "<empty>")


def clarity(w, rate):
    """RMS + high-frequency energy ratio (2-8kHz vs total) as brightness proxy."""
    rms = float(np.sqrt((w ** 2).mean()))
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), 1 / rate)
    hi = spec[(freqs >= 2000) & (freqs <= 8000)].sum()
    tot = spec[freqs <= 8000].sum() + 1e-9
    return rms, float(hi / tot)


def main():
    from faster_whisper import WhisperModel
    from qwen_tts import Qwen3TTSModel
    from scipy.signal import resample_poly

    torch.set_num_threads(6)
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=6)
    results = []
    for di, instruct in enumerate(DESIGNS):
        for attempt in range(2):
            seed = 20261121 + di * 10 + attempt
            torch.manual_seed(seed)
            waves, rate = model.generate_voice_design(
                text=REF_TEXT, instruct=instruct, language="Chinese",
                max_new_tokens=600, do_sample=True, temperature=0.9, top_p=0.9)
            w = np.asarray(waves[0], dtype=np.float32)
            dur = len(w) / rate
            if not np.isfinite(w).all() or float(np.abs(w).max()) < 1e-5 or not 2.0 <= dur <= 20.0:
                print(di, attempt, "invalid", flush=True)
                continue
            segs, _ = asr.transcribe(resample_poly(w, 2, 3), language="zh", beam_size=3,
                                     condition_on_previous_text=False, initial_prompt="以下是普通话的句子。")
            hyp = "".join(s.text for s in segs)
            e = cer(REF_TEXT, hyp)
            rms, hi = clarity(w, rate)
            name = f"M3-d{di}-a{attempt}"
            w = w * (0.95 / float(np.abs(w).max()))
            sf.write(OUT / f"{name}.wav", w, rate, subtype="PCM_16")
            results.append({"name": name, "design": di, "seed": seed, "cer": round(e, 4),
                            "rms": round(rms, 4), "hf_ratio": round(hi, 4),
                            "seconds": round(dur, 2), "transcript": hyp})
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    results.sort(key=lambda r: (r["cer"], -r["hf_ratio"]))
    (OUT / "candidates.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST:", json.dumps(results[0], ensure_ascii=False), flush=True)
    del model
    torch.cuda.empty_cache()
    print("M3_DESIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
