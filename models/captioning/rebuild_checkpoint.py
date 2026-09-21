"""
rebuild_checkpoint.py

Rebuilds the merged captioner checkpoint (the `caption_model/` folder caption_tool.py loads) from the two
small files the Kaggle notebook also saves, instead of downloading its 1GB merged output:

    caption_ckpt/best_trainable.pt   ~117MB  the trained LoRA + connector tensors
    caption_model/caption_meta.json  <1KB    prompt, base model id, evaluation metrics

The base model (SmolVLM-500M-Instruct) comes from the Hugging Face cache/Hub. The Kaggle Output-tab
download of the 1GB `model.safetensors` stalled at ~1 KB/s here (the CLI reads the whole file in one
non-resumable request), while the 117MB state came down fine -- and the merge is deterministic, so this
reproduces the notebook's export (same file list, same 1019MB size).

It refuses to write anything unless the trained state lines up exactly with the rebuilt model: every
trainable tensor in the file must exist in the model, the model must have no trainable tensor the file
does not cover, and every tensor must read back equal after loading.

Usage:

    python models/captioning/rebuild_checkpoint.py \\
        --state path/to/best_trainable.pt --meta path/to/caption_meta.json \\
        --out models/captioning/checkpoints/caption_model

The LoRA settings below must stay identical to `add_lora()` in
notebooks/kaggle_finetune_caption_vrsbench.ipynb -- a different rank or target regex makes the keys
mismatch, which this script reports rather than papering over.
"""

import argparse
import shutil
import sys
from pathlib import Path

LORA_R, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.05
LORA_TARGETS = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def rebuild(state_path: Path, meta_path: Path, out_dir: Path, base_model: str) -> None:
    import json

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    meta = json.loads(meta_path.read_text())
    base_model = base_model or meta["base_model"]

    base = AutoModelForImageTextToText.from_pretrained(base_model, dtype=torch.float32)
    model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS))
    for name, param in model.named_parameters():
        if "connector" in name:  # the vision-to-text connector was trained in full alongside the LoRA
            param.requires_grad = True

    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
    if missing or unexpected:
        sys.exit(
            f"The trained state does not line up with the rebuilt model: {len(missing)} tensors missing from the file "
            f"(e.g. {missing[:2]}), {len(unexpected)} in the file the model lacks (e.g. {unexpected[:2]}). "
            "Check that LORA_* here matches the notebook and that peft/transformers are recent enough."
        )

    model.load_state_dict(state, strict=False)  # strict=False only because the frozen base weights are not in the file
    params = dict(model.named_parameters())
    for name, tensor in state.items():
        if not torch.equal(params[name].detach().cpu(), tensor.to(params[name].dtype)):
            sys.exit(f"{name} did not load correctly")
    print(f"all {len(state)} trained tensors loaded and verified")

    merged = model.merge_and_unload().half()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(base_model)
    processor.image_processor.do_image_splitting = bool(meta.get("do_image_splitting", False))
    processor.save_pretrained(out_dir)
    shutil.copy(meta_path, out_dir / "caption_meta.json")
    size_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"wrote {out_dir} ({size_mb:.0f} MB): {sorted(f.name for f in out_dir.iterdir())}")


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", required=True, type=Path, help="best_trainable.pt from the notebook's caption_ckpt/")
    parser.add_argument("--meta", required=True, type=Path, help="caption_meta.json from the notebook's caption_model/")
    parser.add_argument("--out", required=True, type=Path, help="Directory to write the merged checkpoint to")
    parser.add_argument("--base-model", default="", help="Override the base model id (default: the one in caption_meta.json)")
    args = parser.parse_args()
    for path in (args.state, args.meta):
        if not path.exists():
            sys.exit(f"Not found: {path}")
    rebuild(args.state, args.meta, args.out, args.base_model)


if __name__ == "__main__":
    _cli()
