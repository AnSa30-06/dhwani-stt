"""Convert an HF-format whisper fine-tune to mlx_whisper's on-disk format.

Exists because mlx-whisper's PyPI wheel (0.4.3) ships NO converter at all —
`python -m mlx_whisper.convert` fails with ModuleNotFoundError, which is why the
mix model silently stayed on the 2x-slower transformers runtime on the first
Apple-hardware run. The mapping below is written against the loader that IS in
the wheel (load_models.py + whisper.py): config.json carrying ModelDimensions
fields, weights.safetensors keyed by attribute path, Conv1d weights as
(out, kernel, in), encoder positional embedding COMPUTED (never stored), decoder
positional embedding LEARNED (must be stored).

Torch-free on purpose: safetensors + numpy only, so it runs at warm time in a
subprocess without loading a second copy of torch.

    python -m solution.hf_to_mlx shunyalabs/zero-stt-hinglish /path/out
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _snapshot(repo_or_path: str) -> Path:
    p = Path(repo_or_path)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_or_path,
                                  allow_patterns=["*.json", "*.safetensors", "*.bin"]))


def _load_state(snap: Path) -> dict:
    st = sorted(snap.glob("*.safetensors"))
    if st:
        from safetensors import safe_open
        out: dict = {}
        for f in st:
            with safe_open(str(f), framework="numpy") as fh:
                for k in fh.keys():
                    out[k] = fh.get_tensor(k)
        return out
    bins = sorted(snap.glob("pytorch_model*.bin"))
    if bins:
        import torch
        out = {}
        for f in bins:
            for k, v in torch.load(str(f), map_location="cpu",
                                   weights_only=True).items():
                out[k] = v.to(torch.float32).numpy()
        return out
    raise FileNotFoundError(f"no safetensors or bin weights under {snap}")


# HF module path -> MLX attribute path, per repeated decoder/encoder layer.
_LAYER_MAP = [
    ("self_attn.q_proj", "attn.query"),
    ("self_attn.k_proj", "attn.key"),          # no bias on either side
    ("self_attn.v_proj", "attn.value"),
    ("self_attn.out_proj", "attn.out"),
    ("self_attn_layer_norm", "attn_ln"),
    ("encoder_attn.q_proj", "cross_attn.query"),
    ("encoder_attn.k_proj", "cross_attn.key"),
    ("encoder_attn.v_proj", "cross_attn.value"),
    ("encoder_attn.out_proj", "cross_attn.out"),
    ("encoder_attn_layer_norm", "cross_attn_ln"),
    ("fc1", "mlp1"),
    ("fc2", "mlp2"),
    ("final_layer_norm", "mlp_ln"),
]


def convert(repo_or_path: str, out_dir: str, dtype: str = "float16") -> Path:
    snap = _snapshot(repo_or_path)
    cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "whisper":
        raise ValueError(f"not a whisper checkpoint: model_type={cfg.get('model_type')}")

    state = _load_state(snap)
    if any(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        # proj_out is tied to embed_tokens; token_embedding.as_linear covers it.

    np_dtype = np.float16 if dtype == "float16" else np.float32
    mlx: dict[str, np.ndarray] = {}

    def put(dst: str, src: str, transpose_conv: bool = False) -> None:
        v = state.pop(src)
        if transpose_conv:
            v = np.transpose(v, (0, 2, 1))     # (out,in,k) -> (out,k,in)
        mlx[dst] = np.ascontiguousarray(v.astype(np_dtype))

    for i in (1, 2):
        put(f"encoder.conv{i}.weight", f"encoder.conv{i}.weight", transpose_conv=True)
        put(f"encoder.conv{i}.bias", f"encoder.conv{i}.bias")
    put("encoder.ln_post.weight", "encoder.layer_norm.weight")
    put("encoder.ln_post.bias", "encoder.layer_norm.bias")
    put("decoder.token_embedding.weight", "decoder.embed_tokens.weight")
    put("decoder.positional_embedding", "decoder.embed_positions.weight")
    put("decoder.ln.weight", "decoder.layer_norm.weight")
    put("decoder.ln.bias", "decoder.layer_norm.bias")
    state.pop("encoder.embed_positions.weight", None)   # fixed sinusoids; MLX recomputes

    for side, n_layers in (("encoder", cfg["encoder_layers"]),
                           ("decoder", cfg["decoder_layers"])):
        for n in range(n_layers):
            for hf_leaf, mlx_leaf in _LAYER_MAP:
                if side == "encoder" and hf_leaf.startswith("encoder_attn"):
                    continue                    # encoder blocks have no cross-attn
                base = f"{side}.layers.{n}.{hf_leaf}"
                dst = f"{side}.blocks.{n}.{mlx_leaf}"
                put(f"{dst}.weight", f"{base}.weight")
                if f"{base}.bias" in state:
                    put(f"{dst}.bias", f"{base}.bias")

    leftovers = [k for k in state if not k.startswith("proj_out")]
    if leftovers:
        raise ValueError(f"unmapped weights (layout drift?): {leftovers[:8]}")

    dims = {
        "n_mels": cfg["num_mel_bins"],
        "n_audio_ctx": cfg["max_source_positions"],
        "n_audio_state": cfg["d_model"],
        "n_audio_head": cfg["encoder_attention_heads"],
        "n_audio_layer": cfg["encoder_layers"],
        "n_vocab": cfg["vocab_size"],
        "n_text_ctx": cfg["max_target_positions"],
        "n_text_state": cfg["d_model"],
        "n_text_head": cfg["decoder_attention_heads"],
        "n_text_layer": cfg["decoder_layers"],
    }
    emb = mlx["decoder.positional_embedding"]
    if emb.shape != (dims["n_text_ctx"], dims["n_text_state"]):
        raise ValueError(f"decoder positional embedding {emb.shape} != "
                         f"({dims['n_text_ctx']}, {dims['n_text_state']})")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(mlx, str(out / "weights.safetensors"))
    (out / "config.json").write_text(json.dumps(dims, indent=1), encoding="utf-8")

    # The weights are only half the model. The first conversion shipped without
    # this and DEGRADED: the fine-tune's generation_config carries a suppress
    # list transformers applies on every generate() and mlx_whisper knows
    # nothing about — decoded without it the model rambled ("... तो तो ..."),
    # slower and worse at once. Store the recipe next to the weights so the
    # decode path can hand it to mlx's DecodingOptions.suppress_tokens.
    recipe: dict = {}
    for src in (snap / "generation_config.json", snap / "config.json"):
        if not src.exists():
            continue
        g = json.loads(src.read_text(encoding="utf-8"))
        tokens = g.get("suppress_tokens")
        if tokens and not recipe.get("suppress_tokens"):
            recipe["suppress_tokens"] = sorted({int(t) for t in tokens if int(t) >= 0})
    if recipe:
        (out / "recipe.json").write_text(json.dumps(recipe, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    path = convert(sys.argv[1], sys.argv[2])
    print(f"converted -> {path}")
