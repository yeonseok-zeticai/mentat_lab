"""Generate PATHS.md — absolute paths to every per-variant artifact.

Run after build_variant.py / run_all.sh produces the artifacts; the
table is what teammates copy-paste to fetch a single variant's files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path("/mnt/disks/zeticai_database/models/sapiens2")

# Files we surface, in deploy order.
ARTIFACTS = [
    ("safetensors", "{stem}.safetensors", "HuggingFace weights (the canonical source)"),
    ("sample_input", "sample_input.npy", "(1, 3, 1024, 768) float32 NCHW, RNG-seeded"),
    ("sample_output", "sample_output.npy", "PyTorch reference output"),
    ("model_pt2", "model.pt2", "torch.export ExportedProgram (PyTorch ≥ 2.9)"),
    ("model_onnx", "model.onnx", "ONNX graph header"),
    ("model_onnx_data", "model.onnx.data", "ONNX external-data weights"),
    ("model_dlc", "model.dlc", "QNN/QAIRT FP32 DLC (reference)"),
    ("model_fp16_dlc", "model_fp16.dlc", "QNN/QAIRT FP16 DLC (HTP-deployable)"),
    ("model_mlpackage", "model.mlpackage", "Apple CoreML MLProgram"),
    ("metadata", "metadata.json", "per-stage build status + I/O contract"),
]


def stem_for(task: str, size: str) -> str:
    if task == "pretrain":
        return f"sapiens2_{size}_pretrain"
    return f"sapiens2_{size}_{task}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "PATHS.md"

    catalog = json.loads((root / "catalog.json").read_text())
    catalog = [v for v in catalog if "error" not in v]

    lines: list[str] = []
    lines.append("# sapiens2 — absolute artifact paths")
    lines.append("")
    lines.append("Every variant lives under "
                 f"`{root}/<task>_<size>/`. Files referenced here are the canonical")
    lines.append("artifacts a teammate would pull from this machine into their own runtime.")
    lines.append("")
    lines.append("## Quick fetch (rsync from this host)")
    lines.append("")
    lines.append("```bash")
    lines.append("# Single variant")
    lines.append("rsync -a --info=progress2 \\")
    lines.append(f"    \"$(hostname):{root}/pose_0_4b/\" \\")
    lines.append("    ./local-cache/sapiens2/pose_0_4b/")
    lines.append("")
    lines.append("# Whole family (~250 GB once everything is built)")
    lines.append(f"rsync -a --info=progress2 \"$(hostname):{root}/\" ./local-cache/sapiens2/")
    lines.append("```")
    lines.append("")

    # Per-variant blocks.
    for v in sorted(catalog, key=lambda r: (r["task"], r["size"])):
        task, size = v["task"], v["size"]
        d = root / f"{task}_{size.replace('.', '_')}"
        stem = stem_for(task, size)

        lines.append(f"## `{task}-{size}`  ←  `{v['repo_id']}`")
        lines.append("")
        if not d.exists():
            lines.append("_Not built yet._")
            lines.append("")
            continue

        # Stage status from metadata.
        meta_path = d / "metadata.json"
        stages = {}
        if meta_path.exists():
            try:
                stages = json.loads(meta_path.read_text()).get("stages", {})
            except Exception:
                stages = {}

        def st(name: str) -> str:
            r = stages.get(name)
            if not r:
                return "·"
            return {"ok": "✅", "fail": "❌", "skipped": "—", "running": "⏳"}.get(
                r.get("status", "?"), "?"
            )

        lines.append(
            "Stages: "
            f"build {st('build_and_load')}  pt2 {st('torch_export')}  "
            f"onnx {st('onnx_export')}  qnn {st('qnn_convert')}  coreml {st('coreml_convert')}  "
            f"verify-pt2 {st('verify_pt2')}  verify-ort {st('verify_onnx')}"
        )
        lines.append("")
        lines.append("| artifact | bytes | absolute path | description |")
        lines.append("|---|---|---|---|")
        for key, fname, desc in ARTIFACTS:
            actual = fname.format(stem=stem)
            p = d / actual
            if p.exists():
                if p.is_dir():
                    bytes_total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    sz = f"{bytes_total / 1e6:.1f} MB"
                else:
                    sz = f"{p.stat().st_size / 1e6:.1f} MB"
                lines.append(f"| `{key}` | {sz} | `{p}` | {desc} |")
            else:
                lines.append(f"| `{key}` | _missing_ | `{p}` | {desc} |")
        lines.append("")

    out.write_text("\n".join(lines) + "\n")
    print(f"[paths] wrote {out}  ({len(catalog)} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
