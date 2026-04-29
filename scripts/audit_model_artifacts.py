#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_hf_cache(model: str) -> Path | None:
    p = Path(model).expanduser()
    if p.exists():
        return p
    if "/" not in model:
        return None
    root = Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--"))
    if not root.exists():
        return None
    ref = root / "refs/main"
    if ref.exists():
        snap = root / "snapshots" / ref.read_text().strip()
        if snap.exists():
            return snap
    snap = root / "snapshots/main"
    return snap if snap.exists() else None


def resolve_hipfire(model: str) -> Path | None:
    p = Path(model).expanduser()
    if p.exists():
        return p
    aliases = {
        "qwen3.5:0.8b": "qwen3.5-0.8b.mq4",
        "qwen3.5:4b": "qwen3.5-4b.mq4",
        "qwen3.5:9b": "qwen3.5-9b.mq4",
        "qwen3.5:27b": "qwen3.5-27b.mq4",
    }
    name = aliases.get(model, model)
    p = Path.home() / ".hipfire/models" / name
    return p if p.exists() else None


def hf_artifact(label: str, model: str) -> dict:
    root = resolve_hf_cache(model)
    out = {
        "label": label,
        "model": model,
        "kind": "hf_or_mlx",
        "resolved": str(root) if root else "",
        "format": "missing",
        "primary_file": "",
        "primary_md5": "",
        "primary_bytes": 0,
        "source_model_type": "",
        "quant": "",
        "tie_word_embeddings": "",
    }
    if not root:
        return out
    cfg_path = root / "config.json" if root.is_dir() else root.parent / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            out["source_model_type"] = str(cfg.get("model_type", ""))
            q = cfg.get("quantization") or cfg.get("quantization_config") or {}
            out["quant"] = json.dumps(q, sort_keys=True) if q else "none"
            out["tie_word_embeddings"] = str(cfg.get("tie_word_embeddings", ""))
        except Exception as exc:
            out["quant"] = f"config_error:{exc}"
    if root.is_file():
        primary = root
    else:
        safetensors = sorted(root.glob("*.safetensors"))
        primary = safetensors[0] if safetensors else None
    if primary:
        out["primary_file"] = str(primary)
        out["primary_md5"] = file_md5(primary)
        out["primary_bytes"] = primary.stat().st_size
        out["format"] = "safetensors"
    return out


def hipfire_artifact(label: str, model: str) -> dict:
    p = resolve_hipfire(model)
    out = {
        "label": label,
        "model": model,
        "kind": "hipfire",
        "resolved": str(p) if p else "",
        "format": "missing",
        "primary_file": str(p) if p else "",
        "primary_md5": "",
        "primary_bytes": 0,
        "source_model_type": "qwen3_5",
        "quant": "",
        "tie_word_embeddings": "",
    }
    if not p:
        return out
    out["primary_md5"] = file_md5(p)
    out["primary_bytes"] = p.stat().st_size
    suffix = p.suffix.lower()
    out["format"] = suffix[1:] if suffix else "file"
    if suffix == ".mq4":
        out["quant"] = json.dumps(
            {
                "bits": 4,
                "group_size": 256,
                "mode": "magnumquant_fwht_rotated",
                "runtime_dtype": "MQ4G256",
            },
            sort_keys=True,
        )
    return out


def print_tsv(rows: list[dict]) -> None:
    cols = [
        "label",
        "kind",
        "model",
        "resolved",
        "format",
        "primary_file",
        "primary_md5",
        "primary_bytes",
        "source_model_type",
        "quant",
        "tie_word_embeddings",
    ]
    print("\t".join(cols))
    for row in rows:
        print("\t".join(str(row.get(c, "")).replace("\t", " ") for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemon", required=True)
    ap.add_argument("--hipfire", required=True)
    ap.add_argument("--source", default="")
    args = ap.parse_args()

    rows = []
    if args.source:
        rows.append(hf_artifact("source", args.source))
    rows.append(hf_artifact("lemon", args.lemon))
    rows.append(hipfire_artifact("hipfire", args.hipfire))
    print_tsv(rows)

    lemon = rows[-2]
    hipfire = rows[-1]
    exact = (
        lemon["primary_md5"]
        and lemon["primary_md5"] == hipfire["primary_md5"]
        and lemon["format"] == hipfire["format"]
    )
    print(f"exact_artifact_match\t{1 if exact else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
