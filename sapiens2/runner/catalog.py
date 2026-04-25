"""Index parser for facebook/sapiens2.

facebook/sapiens2 is an *index* repo — only README.md, no weights — that
points to a fan-out of sub-repos:

  * `facebook/sapiens2-pretrain-{0.1b,0.4b,0.8b,1b,5b}`   — ViT backbones
  * `facebook/sapiens2-{pose,seg,normal,pointmap}-{0.4b,0.8b,1b,5b}`  — task heads

This script:
  1. Resolves every sub-repo via the HF API.
  2. Records its files (path, size, sha) and pulls the README.md text.
  3. Cross-references each task variant with the matching sapiens config under
     ``sapiens2/sapiens/{pose,dense}/configs/...`` so the build runner can
     re-instantiate the architecture from python.
  4. Writes a per-variant ``manifest.json`` to
     ``/mnt/disks/zeticai_database/models/sapiens2/<variant>/manifest.json``.
  5. Writes a top-level ``INDEX.md`` summarising every variant.

No safetensors are downloaded — this stage produces only metadata so the
runner can plan disk + RAM use up front.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, hf_hub_download

REPO_ROOT = Path("/home/yeonseok/workspace/exp/mentat_lab/sapiens2/sapiens2")
OUT_ROOT = Path("/mnt/disks/zeticai_database/models/sapiens2")

# (task, size_label) — the canonical inventory the README index advertises.
PRETRAIN_SIZES = ["0.1b", "0.4b", "0.8b", "1b", "5b"]
TASK_SIZES = ["0.4b", "0.8b", "1b", "5b"]
TASKS = ["pose", "seg", "normal", "pointmap"]

# Map (task, size) → sapiens config path under REPO_ROOT.
CONFIG_MAP = {
    ("pose", s): REPO_ROOT
    / f"sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_{s}_keypoints308_shutterstock_goliath_3po-1024x768.py"
    for s in TASK_SIZES
}
CONFIG_MAP.update(
    {
        ("seg", s): REPO_ROOT
        / f"sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_{s}_seg_shutterstock_goliath-1024x768.py"
        for s in TASK_SIZES
    }
)
CONFIG_MAP.update(
    {
        ("normal", s): REPO_ROOT
        / f"sapiens/dense/configs/normal/metasim_render_people/sapiens2_{s}_normal_metasim_render_people-1024x768.py"
        for s in TASK_SIZES
    }
)
CONFIG_MAP.update(
    {
        ("pointmap", s): REPO_ROOT
        / f"sapiens/dense/configs/pointmap/render_people/sapiens2_{s}_pointmap_render_people-1024x768.py"
        for s in TASK_SIZES
    }
)


def variant_repo_id(task: str, size: str) -> str:
    if task == "pretrain":
        return f"facebook/sapiens2-pretrain-{size}"
    return f"facebook/sapiens2-{task}-{size}"


def safe_dirname(task: str, size: str) -> str:
    return f"{task}_{size.replace('.', '_')}"


def resolve_variant(api: HfApi, task: str, size: str) -> Optional[dict]:
    repo_id = variant_repo_id(task, size)
    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        return {"task": task, "size": size, "repo_id": repo_id, "error": str(e)}

    file_records = []
    for fname in files:
        info = api.get_paths_info(repo_id, [fname])[0]
        file_records.append(
            {
                "path": fname,
                "size": getattr(info, "size", None),
                "lfs_sha256": getattr(getattr(info, "lfs", None), "sha256", None),
                "blob_id": getattr(info, "blob_id", None),
            }
        )

    safetensors = [f for f in file_records if f["path"].endswith(".safetensors")]
    weights_size = sum((f["size"] or 0) for f in safetensors)

    config = CONFIG_MAP.get((task, size))
    return {
        "task": task,
        "size": size,
        "repo_id": repo_id,
        "files": file_records,
        "weights_safetensors": [f["path"] for f in safetensors],
        "weights_bytes": weights_size,
        "config_path": str(config) if config else None,
        "config_exists": (config is not None and config.exists()),
    }


def fetch_readme(api: HfApi, repo_id: str, dest: Path) -> Optional[Path]:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="README.md")
    except Exception:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(path).read_bytes())
    return dest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default=str(OUT_ROOT))
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    catalog: list[dict] = []

    for size in PRETRAIN_SIZES:
        catalog.append(resolve_variant(api, "pretrain", size))
    for task in TASKS:
        for size in TASK_SIZES:
            catalog.append(resolve_variant(api, task, size))

    # Per-variant directory + manifest.
    for v in catalog:
        d = out_root / safe_dirname(v["task"], v["size"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(v, indent=2))
        if "repo_id" in v and "error" not in v:
            fetch_readme(api, v["repo_id"], d / "source_README.md")

    # Top-level INDEX.md.
    rows = ["| variant | repo | params | weight size | config |", "|---|---|---|---|---|"]
    for v in sorted(catalog, key=lambda r: (r["task"], r["size"])):
        gb = (v.get("weights_bytes") or 0) / 1e9
        cfg = "✓" if v.get("config_exists") else ("—" if v["task"] == "pretrain" else "MISSING")
        err = v.get("error", "")
        repo = f"`{v.get('repo_id', '?')}`"
        rows.append(
            f"| {v['task']}-{v['size']} | {repo} | — | {gb:.2f} GB | {cfg}{(' **err**: ' + err[:60]) if err else ''} |"
        )

    total_gb = sum((v.get("weights_bytes") or 0) for v in catalog) / 1e9
    out_index = out_root / "INDEX.md"
    out_index.write_text(
        "# sapiens2 model catalog\n\n"
        f"Generated by `runner/catalog.py`. Total weight footprint: **{total_gb:.1f} GB**.\n\n"
        + "\n".join(rows)
        + "\n"
    )
    print(f"[catalog] wrote {out_index}; total weights = {total_gb:.1f} GB")
    print(f"[catalog] {sum(1 for v in catalog if 'error' not in v)}/{len(catalog)} variants resolved")
    (out_root / "catalog.json").write_text(json.dumps(catalog, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
