"""Extract the pooled `g` tensor from the affinity module for saved inputs.

Loads pre-saved affinity module inputs from
``<out_dir>/processed/affinity_module_inputs/<record_id>/affinity_input_<record_id>.npz``,
runs the pretrained affinity module(s) from a Boltz-2 affinity checkpoint, and
saves the pooled pair-masked representation `g` (before the out MLP; see
`AffinityModule` / `AffinityHeadsTransformer` in `src/boltz/model/modules/affinity.py`)
to ``<out_dir>/processed/g/<record_id>/g_<record_id>.npz``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from boltz.model.modules.affinity import AffinityModule


AFFINITY_MODULE_PREFIXES = ("affinity_module", "affinity_module1", "affinity_module2")


def _load_affinity_state_dicts(ckpt_path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    hparams = ckpt["hyper_parameters"]

    ensemble = hparams.get("affinity_ensemble", False)
    modules: dict[str, dict] = {}
    if ensemble:
        args_map = {
            "affinity_module1": hparams["affinity_model_args1"],
            "affinity_module2": hparams["affinity_model_args2"],
        }
    else:
        args_map = {"affinity_module": hparams["affinity_model_args"]}

    token_s = hparams["token_s"]
    token_z = hparams["token_z"]

    for prefix, args in args_map.items():
        sub_sd = {
            k[len(prefix) + 1 :]: v
            for k, v in state_dict.items()
            if k.startswith(prefix + ".")
        }
        if not sub_sd:
            continue
        args = {k: v for k, v in args.items() if k != "use_cross_transformer"}
        module = AffinityModule(
            token_s=token_s,
            token_z=token_z,
            **args,
        )
        module.load_state_dict(sub_sd, strict=True)
        module.eval()
        modules[prefix] = module

    return modules


def _to_tensor(npz: np.lib.npyio.NpzFile, device: torch.device) -> dict:
    s_inputs = torch.from_numpy(npz["s_inputs"]).to(device)
    z_affinity = torch.from_numpy(npz["z_affinity"]).to(device)
    coords_affinity = torch.from_numpy(npz["coords_affinity"]).to(device)
    feats = {
        "token_to_rep_atom": torch.from_numpy(npz["token_to_rep_atom"]).to(device),
        "token_pad_mask": torch.from_numpy(npz["token_pad_mask"]).to(device),
        "mol_type": torch.from_numpy(npz["mol_type"]).to(device),
        "affinity_token_mask": torch.from_numpy(npz["affinity_token_mask"]).to(device),
    }
    return {
        "s_inputs": s_inputs,
        "z": z_affinity,
        "x_pred": coords_affinity,
        "feats": feats,
    }


def _run_and_capture_g(module: AffinityModule, inputs: dict) -> np.ndarray:
    captured: list[torch.Tensor] = []

    def pre_hook(_mod, args):
        # Input to affinity_out_mlp is the pooled `g` from lines 208-210.
        (g,) = args
        captured.append(g.detach().cpu())

    handle = module.affinity_heads.affinity_out_mlp.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            module(
                s_inputs=inputs["s_inputs"],
                z=inputs["z"],
                x_pred=inputs["x_pred"],
                feats=inputs["feats"],
                multiplicity=1,
                use_kernels=False,
            )
    finally:
        handle.remove()

    assert captured, "forward pre-hook did not fire"
    return captured[0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="The boltz_results_<stem> directory containing processed/affinity_module_inputs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("boltz2_aff.ckpt"),
        help="Path to the Boltz-2 affinity checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite existing g_<record_id>.npz files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N input files (for quick sampling).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    inputs_root = args.out_dir / "processed" / "affinity_module_inputs"
    g_root = args.out_dir / "processed" / "g"
    g_root.mkdir(parents=True, exist_ok=True)

    if not inputs_root.exists():
        raise FileNotFoundError(f"Affinity inputs directory not found: {inputs_root}")

    print(f"Loading affinity modules from {args.checkpoint}")
    modules = _load_affinity_state_dicts(args.checkpoint)
    for mod in modules.values():
        mod.to(device)
    print(f"Loaded modules: {list(modules.keys())}")

    npz_files = sorted(inputs_root.glob("*/affinity_input_*.npz"))
    if not npz_files:
        print(f"No affinity input npz files found under {inputs_root}")
        return
    if args.limit is not None:
        npz_files = npz_files[: args.limit]
        print(f"Limiting to first {len(npz_files)} inputs.")

    for npz_path in npz_files:
        record_id = npz_path.parent.name
        # Use the npz file stem (affinity_input_<name>) so multiple input variants
        # in the same record dir don't collide.
        input_name = npz_path.stem.removeprefix("affinity_input_")
        out_dir = g_root / record_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"g_{input_name}.npz"
        if out_path.exists() and not args.override:
            print(f"[skip] {out_path} exists (use --override to overwrite)")
            continue

        print(f"[run ] {record_id}/{input_name}")
        with np.load(npz_path) as npz:
            inputs = _to_tensor(npz, device)

        g_by_module = {
            name: _run_and_capture_g(mod, inputs) for name, mod in modules.items()
        }
        np.savez_compressed(out_path, **g_by_module)
        print(f"[save] {out_path}  keys={list(g_by_module.keys())}  "
              f"shape={next(iter(g_by_module.values())).shape}")


if __name__ == "__main__":
    main()
