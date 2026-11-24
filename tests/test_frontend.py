"""The Chinese front end: pypinyin + jieba -> POLYGLOT phones with tones.

No network is needed; pypinyin's phrase dictionary is installed with it.
"""

import unittest

from tiny_voca.text import (PHONES, PUNCTUATION, SPECIALS, SYMBOL_TO_ID, SYMBOLS, VOWEL_SET,
                            NothingToPronounce, frontend_fingerprint, frontend_identity,
                            normalize_text, phonemize, tokenize)
from tiny_voca.zh_text import chunks, syllable_phones


class InventoryTests(unittest.TestCase):
    def test_symbols_are_specials_punctuation_then_phones(self):
        self.assertEqual(SYMBOLS, SPECIALS + PUNCTUATION + PHONES)
        self.assertEqual(len(SYMBOLS), 45)
        self.assertEqual(len(set(SYMBOLS)), len(SYMBOLS))
        self.assertEqual(SYMBOL_TO_ID["<pad>"], 0)
        self.assertEqual(SYMBOL_TO_ID["<bos>"], 1)
        self.assertEqual(SYMBOL_TO_ID["<eos>"], 2)

    def test_vowels_are_the_tone_carriers(self):
        self.assertEqual(VOWEL_SET, frozenset(("a", "o", "e", "i", "u", "ü", "ir")))
        self.assertTrue(VOWEL_SET <= set(PHONES))

    def test_identity_is_pinned_and_fingerprinted(self):
        identity = frontend_identity()
        self.assertEqual(identity["frontend_version"], 2)
        self.assertEqual(identity["phonemizer"], "pypinyin")
        self.assertEqual(identity["symbols"], list(SYMBOLS))
        fp = frontend_fingerprint()
        self.assertEqual(len(fp), 64)
        self.assertEqual(fp, frontend_fingerprint())   # deterministic

    def test_maps_emit_only_inventory_phones(self):
        identity = frontend_identity()
        for table in ("initial_map", "final_map", "zero_initial"):
            for key, phones in identity[table].items():
                # y/w are dead entries: syllable_phones routes them through
                # ZERO_INITIAL before INITIAL_MAP is consulted.
                if table == "initial_map" and key in ("y", "w"):
                    continue
                for phone in phones:
                    self.assertIn(phone, PHONES, f"{table}[{key}] -> {phone}")


class NormalizationTests(unittest.TestCase):
    def test_ascii_punctuation_becomes_cjk_and_runs_collapse(self):
        self.assertEqual(normalize_text("你好, 世界...  再见!"), "你好， 世界… 再见！")

    def test_quotes_and_ellipsis(self):
        self.assertEqual(normalize_text("‘你好’……"), "“你好”…")

    def test_nfkc_fullwidth_forms(self):
        self.assertEqual(normalize_text("ＡＢＣ１２３"), "ABC123")

    def test_chunks_separate_cjk_latin_and_marks(self):
        self.assertEqual(list(chunks("你好ABC，世界")),
                         [(False, "你好"), (False, "ABC"), (True, "，"), (False, "世界")])


class SyllableTests(unittest.TestCase):
    def test_initial_and_final_split(self):
        self.assertEqual(syllable_phones("zhong1"), (("zh", "o", "ng"), 1))
        self.assertEqual(syllable_phones("ma5"), (("m", "a"), 0))   # neutral tone -> 0

    def test_zero_initial_syllables(self):
        self.assertEqual(syllable_phones("yi1"), (("i",), 1))
        self.assertEqual(syllable_phones("wu3"), (("u",), 3))
        self.assertEqual(syllable_phones("yu2"), (("ü",), 2))
        self.assertEqual(syllable_phones("er4"), (("ir",), 4))

    def test_jqx_u_is_really_v(self):
        self.assertEqual(syllable_phones("ju1"), (("jh", "ü"), 1))
        self.assertEqual(syllable_phones("qu2"), (("ch", "ü"), 2))
        self.assertEqual(syllable_phones("xu3"), (("sh", "ü"), 3))

    def test_observed_checkpoint_mappings(self):
        # z -> dz, c -> ts, x -> sh; ai -> a+i, eng -> a+ng, ian -> i+a+n
        self.assertEqual(syllable_phones("zi1"), (("dz", "i"), 1))
        self.assertEqual(syllable_phones("ci4"), (("ts", "i"), 4))
        self.assertEqual(syllable_phones("xie3"), (("sh", "i", "e"), 3))
        self.assertEqual(syllable_phones("bai2"), (("b", "a", "i"), 2))
        self.assertEqual(syllable_phones("feng1"), (("f", "a", "ng"), 1))
        self.assertEqual(syllable_phones("tian1"), (("t", "i", "a", "n"), 1))

    def test_unmapped_syllable_raises(self):
        with self.assertRaises(KeyError):
            syllable_phones("zzz9")


class PhonemizeTests(unittest.TestCase):
    def test_hello_world(self):
        result = phonemize("你好，世界。")
        self.assertEqual(result.symbols,
                         ("<bos>", "n", "i", "h", "a", "o", "，", "sh", "i", "jh", "i", "e", "。", "<eos>"))
        self.assertEqual(result.stress, (0, 0, 3, 0, 3, 3, 0, 0, 4, 0, 4, 4, 0, 0))
        self.assertEqual(result.unknown, ())
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))

    def test_tones_ride_on_vowels_only(self):
        result = phonemize("妈妈骑马。")
        for symbol, tone in zip(result.symbols, result.stress):
            if symbol in VOWEL_SET:
                self.assertIn(tone, (0, 1, 2, 3, 4))
            else:
                self.assertEqual(tone, 0, symbol)

    def test_word_boundaries_get_sp(self):
        result = phonemize("妈妈骑马。")
        self.assertIn("<sp>", result.symbols)

    def test_punctuation_tokens_are_kept(self):
        result = phonemize("你好？再见！")
        self.assertIn("？", result.symbols)
        self.assertIn("！", result.symbols)

    def test_unknown_syllables_become_unk_and_are_reported(self):
        result = tokenize([(False, [["ni3", "zzz9"]])])
        self.assertIn("<unk>", result.symbols)
        self.assertEqual(result.unknown, ("zzz9",))

    def test_latin_only_text_has_nothing_to_pronounce(self):
        for text in ("hello world", "12345", "!!! ..."):
            with self.assertRaises(NothingToPronounce):
                phonemize(text)

    def test_empty_text_is_rejected(self):
        for text in ("", "   ", None):
            with self.assertRaises(ValueError):
                phonemize(text)

    def test_punctuation_only_raises(self):
        with self.assertRaises(NothingToPronounce):
            tokenize([(True, "。"), (True, "，")])

    def test_adjacent_duplicate_marks_collapse(self):
        result = tokenize([(False, [["ni3"]]), (True, "。"), (True, "。")])
        self.assertEqual(result.symbols.count("。"), 1)

    def test_every_phone_is_reachable_from_the_maps(self):
        produced = set()
        identity = frontend_identity()
        for table in ("initial_map", "final_map", "zero_initial"):
            for key, phones in identity[table].items():
                if table == "initial_map" and key in ("y", "w"):
                    continue
                produced.update(phones)
        self.assertEqual(produced, set(PHONES))


if __name__ == "__main__":
    unittest.main()
