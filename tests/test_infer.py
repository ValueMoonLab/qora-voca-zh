"""Reading long Chinese text and keeping the audio below full scale.

The tests put a small front end in its place: every CJK character becomes
one syllable ("ma1" -> m a), every CJK mark becomes a token, anything else
says nothing. No pypinyin or jieba is needed for the splitting logic.
"""

import contextlib
import dataclasses
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tiny_voca import infer
from tiny_voca.infer import CHINESE_ONLY, PEAK, finished, limit_peak, sentences, shorter, speak, speak_text
from tiny_voca.model import TinyVoca, UtteranceTooLong
from tiny_voca.text import NothingToPronounce, tokenize
from tiny_voca.train import export_model
from helpers import small_config

BOM = chr(0xFEFF)                 # the mark some editors put at the start of a file; typed out it cannot be seen
_MARKS = "，。！？；：、…—"


def phonemize(text):
    """One syllable per CJK character, a token per mark; latin says nothing."""
    pieces = []
    for char in text:
        if char in _MARKS:
            pieces.append((True, char))
        elif "\u4e00" <= char <= "\u9fff":
            pieces.append((False, [["ma1"]]))
    return tokenize(pieces)


def model(frames=3, **settings):
    """A small random model that gives every token ``frames`` frames."""
    torch.manual_seed(5)
    network = TinyVoca(dataclasses.replace(small_config(), **settings)).eval()
    with torch.no_grad():
        for head in (network.vocoder.harmonic, network.vocoder.noise):
            head.weight.normal_(std=0.05)
        network.duration.out.weight.zero_()
        network.duration.out.bias.fill_(math.log1p(frames))
    return network


class SplittingTests(unittest.TestCase):
    def test_sentences(self):
        for text, expected in (
                ("你好。今天天气真不错！我们一起去公园散步吧？", ["你好。", "今天天气真不错！", "我们一起去公园散步吧？"]),
                (BOM + "第一句。\n\n  第二句…\t第三句。 第四句", ["第一句。", " 第二句…", " 第三句。", " 第四句"]),
                ("你好世界", ["你好世界"]),
                ("!!! ...", []),
                ("   ", [])):
            self.assertEqual(sentences(text), expected, text)

    ENDINGS = (
        ("你好世界", "你好世界。"),
        ("你好世界 \n", "你好世界。"),
        ("第一句。第二句", "第一句。第二句。"),
        ("他说“你好”", "他说“你好”。"),
        ("等等…", "等等…"),
        ("你好，", "你好，"),
        ("列表：", "列表："),
        ("“停！”", "“停！”"),
        ("你好。", "你好。"),
        ("真的吗？", "真的吗？"),
        ("", ""), ("  \n", "  \n"))

    def test_a_text_that_ends_without_a_mark_gets_a_full_stop(self):
        for text, expected in self.ENDINGS:
            self.assertEqual(finished(text), expected, text)

    def test_shorter_pieces(self):
        self.assertEqual(shorter("第一，第二；第三：第四—第五"), ["第一，", "第二；", "第三：", "第四—", "第五"])
        self.assertEqual(shorter("你好世界"), ["你好", "世界"])
        self.assertEqual(shorter("你"), [])


class PeakTests(unittest.TestCase):
    def test_loud_audio_is_scaled_down_as_a_whole(self):
        wave = torch.tensor([0.0, 0.5, -1.077, 0.25])
        limited = limit_peak(wave)
        self.assertAlmostEqual(float(limited.abs().max()), PEAK, places=6)
        torch.testing.assert_close(limited, wave * (PEAK / 1.077))        # every sample by the same factor: no clipping
        self.assertAlmostEqual(float(limit_peak(wave, 0.5).abs().max()), 0.5, places=6)
        samples = (limited.numpy() * 32767).round()
        self.assertLess(abs(samples).max(), 32767)

    def test_quiet_audio_is_left_alone(self):
        wave = torch.tensor([0.0, 0.5, -0.98])
        self.assertIs(limit_peak(wave), wave)
        self.assertEqual(limit_peak(torch.zeros(0)).numel(), 0)


@patch.object(infer, "phonemize", phonemize)
class LongTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sentences_are_read_one_by_one_and_joined(self):
        network = model()
        text = "第一句。第二句，长一点的那种！第三句？"
        wave, spoken = speak_text(network, text, "F2", peak=None)
        parts = [speak(network, sentence, "F2", seed=seed)[0] for seed, sentence in enumerate(sentences(text))]
        self.assertEqual(len(spoken), 3)
        torch.testing.assert_close(wave, torch.cat(parts), rtol=0, atol=0)
        self.assertEqual([len(p) for p in spoken], [len(phonemize(sentence)) for sentence in sentences(text)])
        # One sentence is what speak() gives.
        alone, phones, _ = speak(network, "就这一句。", "M2", speed=1.2)
        wave, spoken = speak_text(network, "就这一句。", "M2", speed=1.2, peak=None)
        torch.testing.assert_close(wave, alone, rtol=0, atol=0)
        self.assertEqual(spoken, [phones])

    def test_a_text_without_a_final_mark_is_spoken_with_a_full_stop(self):
        network = model()
        wave, spoken = speak_text(network, "第一句。第二句", "F2", peak=None)
        self.assertEqual([phones.symbols[-2] for phones in spoken], ["。", "。"])
        whole, _, _ = speak(network, "第二句。", "F2", seed=1)
        torch.testing.assert_close(wave[-len(whole):], whole, rtol=0, atol=0)           # as if the mark had been typed
        self.assertNotIn("。", speak(network, "第二句", "F2")[1].symbols)                  # speak() adds nothing

    def test_a_sentence_with_too_many_phonemes_is_cut_at_commas_then_in_half(self):
        network = model(max_tokens=52)                       # 40 tokens for one call
        text = "一二三四五六，七八九十甲乙丙丁戊己庚辛壬癸子丑。"
        with self.assertRaisesRegex(ValueError, "too long"):
            speak(network, text, "F2")
        wave, spoken = speak_text(network, text, "F2")
        self.assertEqual(len(spoken), 3)                     # the comma, then the long half in two
        self.assertTrue(all(len(phones) <= 40 for phones in spoken))
        self.assertEqual(sum(phones.symbols.count("a") for phones in spoken), 22)   # every character is said
        self.assertEqual(len(wave), sum(3 * len(phones) for phones in spoken) * network.config.hop)

    def test_a_sentence_with_too_many_frames_is_cut_too(self):
        network = model(frames=20)                           # 400 frames for one call: 20 tokens
        text = "一二三四五六七八九十。"
        with self.assertRaises(UtteranceTooLong):
            speak(network, text, "F2")
        wave, spoken = speak_text(network, text, "F2")
        self.assertGreaterEqual(len(spoken), 2)
        self.assertEqual(sum(phones.symbols.count("a") for phones in spoken), 10)
        self.assertLessEqual(max(len(phones) for phones in spoken) * 20, network.config.max_frames)
        with self.assertRaisesRegex(ValueError, "speed"):    # other errors are not hidden by the splitting
            speak_text(network, text, "F2", speed=0)

    def test_nothing_to_pronounce(self):
        network = model()
        wave, spoken = speak_text(network, "你好。XD", "F2")
        self.assertEqual(len(spoken), 1)                     # "XD" is skipped, the rest is read
        wave, spoken = speak_text(network, "こんにちは。你好。XD。再见。", "F2")
        self.assertEqual(len(spoken), 2)                     # also first and in the middle, and another script
        alone, _ = speak_text(network, "你好。再见。", "F2")
        torch.testing.assert_close(wave, alone, rtol=0, atol=0)                # as if the skipped ones were not there
        for text in ("XD", "!!! ...", "", "こんにちは。", "XD。こんにちは。XD"):
            with self.assertRaisesRegex(NothingToPronounce, "没有可朗读的中文内容"):
                speak_text(network, text, "F2")
        with self.assertRaises(ValueError):                  # as before: one exception type for callers
            speak_text(network, "XD", "F2")
        with self.assertRaisesRegex(ValueError, "one word is too long"):
            speak_text(model(max_tokens=14), "你好世界", "F2")  # nowhere to cut

    def test_the_audio_is_kept_below_full_scale(self):
        network = model()
        loud = lambda *arguments: (torch.tensor([0.2, -1.077, 0.6]), {})
        with patch.object(infer, "_voice", loud):
            wave, _ = speak_text(network, "你好。再见。", "F2")
            self.assertAlmostEqual(float(wave.abs().max()), PEAK, places=6)
            self.assertEqual(len(wave), 6)
            torch.testing.assert_close(wave[:3], wave[3:])                  # one factor for the whole text
            raw, _ = speak_text(network, "你好。再见。", "F2", peak=None)
            self.assertAlmostEqual(float(raw.abs().max()), 1.077, places=6)

    def test_the_command_reads_long_text_from_a_file(self):
        import soundfile as sf
        network = model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_model(network, root / "model.pt", 1, "test")
            (root / "story.txt").write_text(BOM + "第一句。\n第二句！\n\n第三句？\n", encoding="utf-8")
            output = io.StringIO()
            arguments = ["qora-voca-speak", "--checkpoint", str(root / "model.pt"), "--text-file", str(root / "story.txt"),
                         "--output", str(root / "out/story.wav")]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(output):
                infer.main()
            lines = output.getvalue().strip().split("\n")
            self.assertEqual(len(lines), 4)                  # three utterances and the summary
            self.assertIn("in 3 utterances", lines[-1])
            audio, rate = sf.read(root / "out/story.wav")
            expected, _ = speak_text(network, "第一句。 第二句！ 第三句？", "F2")
            self.assertEqual((rate, len(audio)), (network.config.sample_rate, len(expected)))
            self.assertLessEqual(float(abs(audio).max()), PEAK + 1e-6)

    def test_the_command_says_so_in_plain_words_when_nothing_could_be_read(self):
        network = model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export_model(network, root / "model.pt", 1, "test")
            for text in ("こんにちは。", "XD"):
                said = io.StringIO()
                arguments = ["qora-voca-speak", "--checkpoint", str(root / "model.pt"), "--text", text, "--output", str(root / "out.wav")]
                with patch("sys.argv", arguments), contextlib.redirect_stderr(said), self.assertRaises(SystemExit) as stopped:
                    infer.main()
                self.assertEqual(stopped.exception.code, 1)
                self.assertEqual(said.getvalue(), CHINESE_ONLY + "\n")
                self.assertFalse((root / "out.wav").exists())


if __name__ == "__main__":
    unittest.main()
