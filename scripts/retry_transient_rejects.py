import sys
"""Retry transient prepare rejections via --resume.

All 3095 train rejections were transient I/O failures during the E:-full
incident (3070 LibsndfileError, 17 ENOSPC, 8 truncated torch .pt writes).
The source flacs now read fine through the F: junction. prepare --resume
skips every record_key present in accepted OR rejected journals, so we park
the old rejected journal in data/bad-splits/ and let prepare retry those
rows; accepted rows and their .pt features are untouched. Identity is
unchanged (inputs/acoustic/admission), so --resume is legal.
"""
import json, os, shutil, subprocess, sys, time

ROOT = str(Path(__file__).resolve().parents[1])
PY = sys.executable
os.chdir(ROOT)

rej_path = 'data/features/train.rejected.jsonl'
meta_path = 'data/features/train.preparation.json'
assert json.load(open(meta_path, encoding='utf-8'))['complete'] is True

n = sum(1 for _ in open(rej_path, encoding='utf-8'))
print(f'parking {n} transient rejections', flush=True)
os.makedirs('data/bad-splits', exist_ok=True)
dst = f'data/bad-splits/train.rejected.transient-{int(time.time())}.jsonl'
shutil.move(rej_path, dst)
print('parked ->', dst, flush=True)

env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
cmd = [PY, '-m', 'tiny_voca.prepare',
       '--inputs', 'data/split-inputs/train.jsonl',
       '--output', 'data/features',
       '--text-column', 'text', '--speaker-column', 'voice_id',
       '--split', 'train', '--resume']
with open('prepare-train4.log', 'w', encoding='utf-8') as log:
    p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
print('PREPARE_EXIT=', p.returncode, flush=True)
meta = json.load(open(meta_path, encoding='utf-8'))
print('prepare meta:', json.dumps({k: meta[k] for k in ('complete', 'accepted', 'rejected')}), flush=True)
if p.returncode != 0 or not meta.get('complete'):
    sys.exit(1)