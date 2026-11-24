# Release checklist — qora-voca-zh v1.0.0

Run through this before every public push. All items must be verified, not assumed.

## Provenance / legal

- [x] Only synthetic voices in the release (F2/M2/M3). No human-recording-derived weights, embeddings, prompts, or audio anywhere in the tree.
- [x] `scripts/strip_voice.py` used to drop non-release voices; remaining voices verified bit-identical (same-seed waveform max abs diff = 0.0).
- [x] Corpus is original templates + public-domain classical poetry; no third-party copyrighted text.
- [x] LICENSE = Apache-2.0 (upstream tiny-voca, Muhammed Nazeem); NOTICE records fork attribution and third-party offline-only models.
- [x] Generator script has no voice-cloning branch (`x_vector_only_mode=False` everywhere).

## Sensitive-data scan (must be zero hits)

Patterns: real names, local usernames, `C:\Users\...`, absolute drive paths, private IPs (10.x / 192.168.x / 100.x tailnet), public server IPs, internal ports, NAS paths, email addresses, `sk-`/`ghp_` tokens, internal run names tied to private infrastructure.

- [x] Full-tree text scan clean (only false positive: corpus sentence id `zh08091`).
- [x] No `.env`, keys, cookies, or deploy scripts in tree.
- [x] `.gitignore` excludes runs/, features/, annotated/, generated/, models/, logs.

## Stranger test (clone → run without edits)

- [x] `pip install -e .` resolves (CPU torch acceptable).
- [x] `qora-voca-speak --checkpoint release/qora-voca-zh.pt --text ... --voice F2 --output out.wav` works on CPU.
- [x] `PYTHONPATH=src python -m pytest tests -q` → 69 passed.
- [x] `python scripts/check_onnx_zh.py` → PARITY_OK (max_error 0.0033), ~50× realtime CPU.
- [x] Every file mentioned in README exists.

## Repo hygiene

- [x] `.gitattributes` (LF normalization, binary markers for .pt/.onnx/.wav).
- [x] CI workflow: ubuntu + windows, tests + CPU smoke synthesis.
- [x] Fresh `git init` (no old .git / remotes / author history).
- [x] Commit author/committer = GitHub noreply email (GH007 privacy).
- [ ] `gh repo create ValueMoonLab/qora-voca-zh --public --source . --push`
- [ ] Post-push: anonymous README fetch, `gh repo view --json visibility`, CI green, fresh clone re-scan.

## Numbers frozen for docs

- Parameters 4,532,299; checkpoint 17.3 MB; ONNX 21.4 MB; 24 kHz mono.
- Data: 28,231 accepted utterances / 25.25 h (F2 10,101 / M2 8,339 / M3 9,791).
- Step 160,000 validation: mel 0.539 / STFT 1.211 / pause 97.3% / voicing 96.0% / duration ratio 0.975 / pitch error 1.26 st.
- CPU ONNX: 50–53× realtime (4 threads).
