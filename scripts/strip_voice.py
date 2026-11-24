"""Remove a voice from a qora-voca-zh inference checkpoint.

Drops the speaker's embedding rows from both voice tables and removes the
id from the config inventory. All other weights are copied bit-identical,
so the remaining voices sound exactly as before. Useful for publishing a
subset of the voices a checkpoint was trained with.

Usage:
    python scripts/strip_voice.py --input runs/my-run/inference.pt --output release/model.pt --drop VOICE_ID
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop", required=True, help="Speaker id to remove")
    args = parser.parse_args()
    state = torch.load(args.input, map_location="cpu", weights_only=True)
    speakers = list(state["config"]["speakers"])
    if args.drop not in speakers:
        raise SystemExit(f"{args.drop} not in {speakers}; nothing to do")
    idx = speakers.index(args.drop)
    kept = [s for s in speakers if s != args.drop]
    model = dict(state["model"])
    for key in ("voices.weight", "frame_voices.weight"):
        table = model[key]
        if table.shape[0] != len(speakers):
            raise SystemExit(f"{key} rows {table.shape[0]} != speakers {len(speakers)}")
        model[key] = torch.cat([table[:idx], table[idx + 1:]], dim=0).clone()
    config = dict(state["config"])
    config["speakers"] = tuple(kept)
    out = {**state, "config": config, "model": model}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output),
                      "dropped": args.drop, "speakers": kept, "step": state["step"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
