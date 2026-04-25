"""End-to-end build pipeline for a single sapiens2 HuggingFace variant.

Usage:
    python build_variant.py --task pretrain --size 0.1b
    python build_variant.py --task pose --size 0.4b
    python build_variant.py --task pose --size 0.4b --skip qnn,coreml

For each variant the pipeline produces (under
``/mnt/disks/zeticai_database/models/sapiens2/<task>_<size>/``):

    sapiens2_<size>_<task>.safetensors   downloaded weights
    sample_input.npy                     (1, 3, H, W) float32
    sample_output.npy                    PyTorch reference output
    model.pt2                            torch.export ExportedProgram
    model.onnx (+ external data)         ONNX export of the same graph
    model.dlc                            QNN/QAIRT FP32 DLC
    model_int8.dlc                       INT8 per-channel quantised DLC
    model_int8_htp.dlc                   INT8 + offline HTP cache
    model_int8_encoding.json             per-tensor scale/offset
    model.mlpackage/                     CoreML MLProgram
    metadata.json                        I/O contract + pipeline status
    build.log                            stdout/stderr from the run

Failures are non-fatal: the runner records the error in metadata.json under
``stages[<stage>] = {"status": "fail", "error": ...}`` and continues with
the next stage that does not depend on the failed one.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Make sapiens registry imports succeed (force-import the side-effect modules).
import sapiens.backbones  # noqa: F401  registers Sapiens2

OUT_ROOT = Path("/mnt/disks/zeticai_database/models/sapiens2")
REPO_ROOT = Path("/home/yeonseok/workspace/exp/mentat_lab/sapiens2/sapiens2")

# Per-config full path (re-derived to avoid importing the whole catalog).
TASK_CONFIGS = {
    ("pose", "0.4b"): "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    ("pose", "0.8b"): "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.8b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    ("pose", "1b"):   "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_1b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    ("pose", "5b"):   "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py",
    ("seg", "0.4b"):  "sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
    ("seg", "0.8b"):  "sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_0.8b_seg_shutterstock_goliath-1024x768.py",
    ("seg", "1b"):    "sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_1b_seg_shutterstock_goliath-1024x768.py",
    ("seg", "5b"):    "sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_5b_seg_shutterstock_goliath-1024x768.py",
    ("normal", "0.4b"): "sapiens/dense/configs/normal/metasim_render_people/sapiens2_0.4b_normal_metasim_render_people-1024x768.py",
    ("normal", "0.8b"): "sapiens/dense/configs/normal/metasim_render_people/sapiens2_0.8b_normal_metasim_render_people-1024x768.py",
    ("normal", "1b"):   "sapiens/dense/configs/normal/metasim_render_people/sapiens2_1b_normal_metasim_render_people-1024x768.py",
    ("normal", "5b"):   "sapiens/dense/configs/normal/metasim_render_people/sapiens2_5b_normal_metasim_render_people-1024x768.py",
    ("pointmap", "0.4b"): "sapiens/dense/configs/pointmap/render_people/sapiens2_0.4b_pointmap_render_people-1024x768.py",
    ("pointmap", "0.8b"): "sapiens/dense/configs/pointmap/render_people/sapiens2_0.8b_pointmap_render_people-1024x768.py",
    ("pointmap", "1b"):   "sapiens/dense/configs/pointmap/render_people/sapiens2_1b_pointmap_render_people-1024x768.py",
    ("pointmap", "5b"):   "sapiens/dense/configs/pointmap/render_people/sapiens2_5b_pointmap_render_people-1024x768.py",
}

# All configs use 1024x768 input.
INPUT_HW = (1024, 768)


def variant_dir(task: str, size: str) -> Path:
    return OUT_ROOT / f"{task}_{size.replace('.', '_')}"


def repo_id_for(task: str, size: str) -> str:
    if task == "pretrain":
        return f"facebook/sapiens2-pretrain-{size}"
    return f"facebook/sapiens2-{task}-{size}"


def safetensors_filename(task: str, size: str) -> str:
    if task == "pretrain":
        return f"sapiens2_{size}_pretrain.safetensors"
    return f"sapiens2_{size}_{task}.safetensors"


# ----------------------------------------------------------------------
# Model construction


def build_pretrain_backbone(size: str) -> torch.nn.Module:
    """Construct the Sapiens2 ViT backbone alone for pretrain variants.

    ``pos_embed_rope_dtype="fp32"`` is forced — coremltools' torch frontend
    has no bfloat16 entry in TORCH_DTYPE_TO_NUM, and QNN's qairt-converter
    drops bfloat16 ops to CPU. fp32 RoPE costs <1 % of the backbone's
    activation memory, so the precision tradeoff is irrelevant.
    """
    from sapiens.backbones import Sapiens2

    return Sapiens2(
        arch=f"sapiens2_{size}",
        img_size=INPUT_HW,
        patch_size=16,
        with_cls_token=True,
        out_type="featmap",
        final_norm=True,
        use_tokenizer=False,
        pos_embed_rope_dtype="fp32",
    )


def build_task_model(task: str, size: str) -> torch.nn.Module:
    """Construct the full task-head estimator from the sapiens config."""
    from sapiens.engine.config import Config
    from sapiens.registry import MODELS

    cfg_rel = TASK_CONFIGS[(task, size)]
    cfg_path = REPO_ROOT / cfg_rel
    cfg = Config.fromfile(str(cfg_path))

    # Drop the pretrained-init lookup so MODELS.build doesn't try to fetch
    # the backbone weights — we'll load the merged checkpoint ourselves.
    if "init_cfg" in cfg.model.get("backbone", {}):
        cfg.model["backbone"].pop("init_cfg")
    # Force fp32 RoPE for coremltools/QNN compatibility (see backbone fn).
    cfg.model["backbone"]["pos_embed_rope_dtype"] = "fp32"

    model = MODELS.build(cfg.model)
    return model


def load_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    sd = load_file(str(ckpt_path), device="cpu")
    incompat = model.load_state_dict(sd, strict=False)
    if incompat.missing_keys:
        print(f"[load] missing_keys: {len(incompat.missing_keys)} (first 3: {incompat.missing_keys[:3]})")
    if incompat.unexpected_keys:
        print(f"[load] unexpected_keys: {len(incompat.unexpected_keys)} (first 3: {incompat.unexpected_keys[:3]})")


class WrapBackboneOutput(torch.nn.Module):
    """Reduce Sapiens2 backbone's tuple output to its first feature map.

    The estimators use ``backbone(x)[0]`` already; we mimic that for pretrain
    variants so the exported program returns a single tensor.
    """

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


# ----------------------------------------------------------------------
# Stage helpers


@contextmanager
def stage(metadata: dict, name: str):
    print(f"\n[stage] {name} ...")
    t0 = time.time()
    record = {"status": "running"}
    metadata.setdefault("stages", {})[name] = record
    try:
        yield record
        record["status"] = "ok"
    except Exception as e:
        record["status"] = "fail"
        record["error"] = repr(e)
        record["traceback"] = traceback.format_exc()
        print(f"[stage] {name} FAILED: {e}")
    finally:
        record["seconds"] = round(time.time() - t0, 2)
        # Save metadata after every stage so partial state is durable.
        meta_path = metadata["__path__"]
        Path(meta_path).write_text(json.dumps({k: v for k, v in metadata.items() if k != "__path__"}, indent=2))


def maybe_skip(metadata: dict, name: str, skip_set: set[str]) -> bool:
    if name in skip_set:
        metadata.setdefault("stages", {})[name] = {"status": "skipped"}
        print(f"[stage] {name} skipped")
        return True
    return False


# ----------------------------------------------------------------------
# Main pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["pretrain", "pose", "seg", "normal", "pointmap"])
    parser.add_argument("--size", required=True)
    parser.add_argument("--skip", default="", help="Comma-separated stage names to skip.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-h", type=int, default=INPUT_HW[0])
    parser.add_argument("--input-w", type=int, default=INPUT_HW[1])
    args = parser.parse_args()

    out_dir = variant_dir(args.task, args.size)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume previous metadata (so individual stage re-runs don't blow away
    # successful runs from earlier invocations).
    meta_path = out_dir / "metadata.json"
    metadata: dict = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception:
            metadata = {}
    metadata.update(
        {
            "task": args.task,
            "size": args.size,
            "repo_id": repo_id_for(args.task, args.size),
            "input_shape": [1, 3, args.input_h, args.input_w],
            "input_layout": "NCHW",
            "input_dtype": "float32",
            "value_range": "[0, 1]",
            "torch_version": torch.__version__,
        }
    )
    metadata.setdefault("stages", {})
    metadata["__path__"] = str(meta_path)
    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}

    # ------------------------------------------------------------------
    # 1. Download
    ckpt_path: Optional[Path] = None
    with stage(metadata, "download") as rec:
        repo_id = repo_id_for(args.task, args.size)
        fname = safetensors_filename(args.task, args.size)
        downloaded = Path(hf_hub_download(repo_id=repo_id, filename=fname))
        ckpt_path = out_dir / fname
        if not ckpt_path.exists() or ckpt_path.stat().st_size != downloaded.stat().st_size:
            shutil.copy(downloaded, ckpt_path)
        rec["path"] = str(ckpt_path)
        rec["bytes"] = ckpt_path.stat().st_size
    if metadata["stages"]["download"]["status"] != "ok":
        return 1

    # ------------------------------------------------------------------
    # 2. Build + load + run reference forward
    model: Optional[torch.nn.Module] = None
    sample_input: Optional[torch.Tensor] = None
    sample_out: Optional[torch.Tensor] = None
    with stage(metadata, "build_and_load") as rec:
        if args.task == "pretrain":
            backbone = build_pretrain_backbone(args.size)
            # Pretrain checkpoint keys are flat (no "backbone." prefix), so
            # load *before* wrapping; the wrapper just selects feature[0].
            load_weights(backbone, ckpt_path)
            model = WrapBackboneOutput(backbone)
        else:
            model = build_task_model(args.task, args.size)
            load_weights(model, ckpt_path)
        model.eval()

        rng = np.random.default_rng(args.seed)
        sample_np = rng.uniform(0.0, 1.0, size=(1, 3, args.input_h, args.input_w)).astype(np.float32)
        np.save(out_dir / "sample_input.npy", sample_np)
        sample_input = torch.from_numpy(sample_np)
        with torch.inference_mode():
            sample_out = model(sample_input)
        if isinstance(sample_out, (list, tuple)):
            for i, t in enumerate(sample_out):
                np.save(out_dir / f"sample_output_{i}.npy", t.detach().cpu().numpy())
            rec["output_shapes"] = [list(t.shape) for t in sample_out]
        else:
            np.save(out_dir / "sample_output.npy", sample_out.detach().cpu().numpy())
            rec["output_shapes"] = [list(sample_out.shape)]
        rec["param_count"] = sum(p.numel() for p in model.parameters())
        rec["param_bytes"] = rec["param_count"] * 4
    if metadata["stages"]["build_and_load"]["status"] != "ok":
        return 1

    # ------------------------------------------------------------------
    # 3. torch.export
    if not maybe_skip(metadata, "torch_export", skip_set):
        with stage(metadata, "torch_export") as rec:
            ep = torch.export.export(model, (sample_input,), strict=False)
            pt2 = out_dir / "model.pt2"
            torch.export.save(ep, str(pt2))
            rec["path"] = str(pt2)
            # Round-trip sanity: load and re-run with the same input.
            loaded = torch.export.load(str(pt2))
            with torch.inference_mode():
                rt = loaded.module()(sample_input)
            if isinstance(rt, (tuple, list)):
                rt = rt[0]
            ref = sample_out[0] if isinstance(sample_out, (list, tuple)) else sample_out
            rec["roundtrip_max_abs_diff"] = float((rt - ref).abs().max().item())

    # ------------------------------------------------------------------
    # 4. ONNX export (needed for QNN qairt-converter)
    onnx_path: Optional[Path] = None
    if not maybe_skip(metadata, "onnx_export", skip_set):
        with stage(metadata, "onnx_export") as rec:
            onnx_path = out_dir / "model.onnx"
            # Use dynamo_export for torch >= 2.6; falls back to legacy if missing.
            try:
                # Modern path: TorchScript ONNX export with external data so
                # weights >2 GB still serialize cleanly.
                torch.onnx.export(
                    model,
                    (sample_input,),
                    str(onnx_path),
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=17,
                    do_constant_folding=True,
                    dynamic_axes=None,
                )
            except Exception as e:
                rec["legacy_export_error"] = repr(e)
                # Try dynamo-based exporter as fallback.
                onnx_program = torch.onnx.export(
                    model, (sample_input,), str(onnx_path), dynamo=True
                )
                onnx_program.save(str(onnx_path))
            rec["path"] = str(onnx_path)
            rec["bytes"] = onnx_path.stat().st_size if onnx_path.exists() else 0

    # ------------------------------------------------------------------
    # 5. QNN  (delegated to a sibling shell script)
    if onnx_path is None:
        candidate = out_dir / "model.onnx"
        if candidate.exists():
            onnx_path = candidate
    if not maybe_skip(metadata, "qnn_convert", skip_set) and onnx_path is not None:
        with stage(metadata, "qnn_convert") as rec:
            script = Path(__file__).parent / "run_qnn.sh"
            res = subprocess.run(
                ["bash", str(script), str(out_dir), str(onnx_path), "input"],
                capture_output=True, text=True,
            )
            (out_dir / "qnn.log").write_text(res.stdout + "\n" + res.stderr)
            if res.returncode != 0:
                raise RuntimeError(f"run_qnn.sh exit={res.returncode} (see qnn.log)")
            rec["dlc_fp32"] = str(out_dir / "model.dlc")
            rec["dlc_fp16"] = str(out_dir / "model_fp16.dlc")

    # ------------------------------------------------------------------
    # 6. CoreML
    if not maybe_skip(metadata, "coreml_convert", skip_set):
        with stage(metadata, "coreml_convert") as rec:
            import coremltools as ct
            ep = torch.export.load(str(out_dir / "model.pt2"))
            ep = ep.run_decompositions({})
            inputs = [ct.TensorType(name="input", shape=tuple(metadata["input_shape"]), dtype=np.float32)]
            mlmodel = ct.convert(
                ep,
                source="pytorch",
                inputs=inputs,
                convert_to="mlprogram",
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.iOS17,
            )
            ml_path = out_dir / "model.mlpackage"
            if ml_path.exists():
                shutil.rmtree(ml_path)
            mlmodel.save(str(ml_path))
            rec["path"] = str(ml_path)

    # Final write
    Path(metadata["__path__"]).write_text(
        json.dumps({k: v for k, v in metadata.items() if k != "__path__"}, indent=2)
    )

    failed = [n for n, s in metadata["stages"].items() if s.get("status") == "fail"]
    print(f"\n[done] {args.task}-{args.size} → {out_dir}")
    print(f"       failed stages: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
