"""Design the two Chinese teacher voices (F2 female / M2 male) with VoiceDesign.

Produces data/voices/F2.wav, M2.wav + voices.json (reference text per voice).
Each candidate is screened with faster-whisper (Chinese CER); best kept.
"""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = str(ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign")
OUT = ROOT / "data" / "voices"
OUT.mkdir(parents=True, exist_ok=True)
REF_TEXT = "今天天气很好，我们出去走走吧。公园里有很多人在散步，孩子们在草地上跑来跑去。"
DESIGNS = {
    "F2": "一位二十多岁的年轻女性，音色清亮温柔，普通话标准，语速适中，吐字清晰，语气自然亲切，像电台播音员。",
    "M2": "一位三十多岁的成熟男性，音色低沉浑厚，普通话标准，语速平稳，吐字清晰，语气沉稳自然，像新闻主播。",
}


def main():
    from faster_whisper import WhisperModel
    from qwen_tts import Qwen3TTSModel
    from scipy.signal import resample_poly
    import re

    def norm(s):
        return re.sub(r"[^\u4e00-\u9fff]", "", s)

    def cer(ref, hyp):
        import jiwer
        r, h = list(norm(ref)), list(norm(hyp))
        if not r:
            return 1.0
        return jiwer.wer(" ".join(r), " ".join(h) or "<empty>")

    torch.set_num_threads(6)
    model = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=6)
    voices = {}
    for voice, instruct in DESIGNS.items():
        best = None
        for attempt in range(3):
            torch.manual_seed(20261116 + attempt)
            waves, rate = model.generate_voice_design(
                text=REF_TEXT, instruct=instruct, language="Chinese",
                max_new_tokens=600, do_sample=True, temperature=0.9, top_p=0.9)
            w = np.asarray(waves[0], dtype=np.float32)
            dur = len(w) / rate
            peak = float(np.max(np.abs(w))) if len(w) else 0.0
            if not np.isfinite(w).all() or peak < 1e-5 or not 2.0 <= dur <= 20.0:
                print(voice, attempt, "invalid audio", flush=True)
                continue
            segs, _ = asr.transcribe(resample_poly(w, 2, 3), language="zh", beam_size=3,
                                     condition_on_previous_text=False, initial_prompt="以下是普通话的句子。")
            hyp = "".join(s.text for s in segs)
            e = cer(REF_TEXT, hyp)
            print(voice, attempt, f"dur={dur:.1f}s cer={e:.3f} hyp={hyp[:40]}", flush=True)
            if best is None or e < best[0]:
                best = (e, w, rate, hyp)
            if e <= 0.10:
                break
        if best is None:
            raise SystemExit(f"{voice}: no usable candidate")
        e, w, rate, hyp = best
        w = w * (0.95 / float(np.max(np.abs(w))))
        sf.write(OUT / f"{voice}.wav", w, rate, subtype="PCM_16")
        voices[voice] = {"text": REF_TEXT, "instruct": instruct, "cer": round(e, 4),
                         "asr_transcript": hyp, "sample_rate": rate}
        print(voice, "SAVED", f"cer={e:.3f}", flush=True)
    (OUT / "voices.json").write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
    # free the teacher before the clone model loads
    del model
    torch.cuda.empty_cache()
    print("VOICES_DONE", flush=True)


if __name__ == "__main__":
    main()
