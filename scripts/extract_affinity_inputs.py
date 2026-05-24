"""Extract affinity module inputs from pre_affinity npz files and predicted structures.

This script runs the Boltz2 affinity model on already-predicted structures
to extract and save the intermediate representations (s_inputs, z_affinity,
coords_affinity, feats) needed for training the standalone affinity module.

The pre_affinity_{id}.npz files provide atom coordinates for cropping/featurization,
but the model still runs its full forward pass (trunk + diffusion) to produce
the learned embeddings (s_inputs, z) and to re-predict coordinates from the
cropped input. That is why sampling_steps and diffusion_samples are still needed.

Usage:
    python scripts/extract_affinity_inputs.py \
        /path/to/boltz_results_dir \
        --cache ~/.boltz \
        --devices 1

The script expects the standard boltz output directory layout:
    out_dir/
        processed/
            manifest.json
            structures/
            msa/
            constraints/
            templates/
            mols/
        predictions/
            {record_id}/
                pre_affinity_{record_id}.npz
"""

import gc
import multiprocessing as mp
import os
import warnings
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.strategies import DDPStrategy
from torch import Tensor

from boltz.data.crop.affinity import AffinityCropper
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule, load_input
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Manifest, Record
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    get_cache_path,
)
from boltz.model.models.boltz2 import Boltz2


def _prescan_record(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path],
    template_dir: Optional[Path],
    extra_mols_dir: Optional[Path],
) -> tuple[str, bool, str]:
    """Run tokenize + AffinityCropper on a single record (CPU only).

    Returns (record_id, ok, error_message). ok=False means the record
    cannot be processed and its pre_affinity file should be removed.
    """
    try:
        input_data = load_input(
            record=record,
            target_dir=target_dir,
            msa_dir=msa_dir,
            constraints_dir=constraints_dir,
            template_dir=template_dir,
            extra_mols_dir=extra_mols_dir,
            affinity=True,
        )
        tokenized = Boltz2Tokenizer().tokenize(input_data)
        AffinityCropper().crop(tokenized, max_tokens=256, max_atoms=2048)
    except Exception as e:  # noqa: BLE001
        return record.id, False, f"{type(e).__name__}: {e}"
    return record.id, True, ""


class AffinityInputExtractor(BasePredictionWriter):
    """Writer that saves only affinity module inputs to disk."""

    def __init__(self, output_dir: str) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed = 0

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module,
        prediction: dict[str, Tensor],
        batch_indices: list[int],
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if prediction["exception"]:
            self.failed += 1
            records: list[Record] = batch["record"]
            for record in records:
                click.echo(f"| OOM failure for record: {record.id}")
            return

        record_id = batch["record"][0].id
        s_inputs = prediction["s_inputs"].cpu().numpy()
        z_affinity = prediction["z_affinity"].cpu().numpy()
        coords_affinity = prediction["coords_affinity"].cpu().numpy()

        feats_raw = {
            k: v
            for k, v in prediction["feats"].items()
            if k in ("token_to_rep_atom", "token_pad_mask", "mol_type", "affinity_token_mask")
        }
        feats_dict = {
            k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in feats_raw.items()
        }

        struct_dir = self.output_dir / record_id
        struct_dir.mkdir(exist_ok=True)
        path = struct_dir / f"affinity_input_{record_id}.npz"
        np.savez_compressed(
            path,
            s_inputs=s_inputs,
            z_affinity=z_affinity,
            coords_affinity=coords_affinity,
            **feats_dict,
        )
        click.echo(f"  Saved {path}")

    def on_predict_epoch_end(self, trainer, pl_module) -> None:
        click.echo(f"Done. Failed: {self.failed}")


@click.command()
@click.argument("out_dir", type=click.Path(exists=True))
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="Boltz cache directory (default: ~/.boltz or $BOLTZ_CACHE).",
    default=get_cache_path,
)
@click.option(
    "--affinity_checkpoint",
    type=click.Path(exists=True),
    help="Path to the affinity model checkpoint. Defaults to cache/boltz2_aff.ckpt.",
    default=None,
)
@click.option("--devices", type=int, default=1, help="Number of GPU devices.")
@click.option("--accelerator", type=click.Choice(["gpu", "cpu"]), default="gpu")
@click.option("--num_workers", type=int, default=1, help="Dataloader workers.")
@click.option(
    "--prescan_workers",
    type=int,
    default=8,
    help="Parallel CPU workers for the pre-scan that drops records that crop empty.",
)
@click.option(
    "--no_prescan",
    is_flag=True,
    help="Skip the pre-scan step that removes pre_affinity files for records that crop empty.",
)
@click.option(
    "--delete_bad",
    is_flag=True,
    default=True,
    help="Delete pre_affinity files for records that fail the pre-scan (default: on).",
)
@click.option("--recycling_steps", type=int, default=5)
@click.option("--sampling_steps", type=int, default=200)
@click.option("--diffusion_samples", type=int, default=5)
@click.option("--seed", type=int, default=None)
@click.option("--override", is_flag=True, help="Re-extract even if outputs exist.")
@click.option("--no_kernels", is_flag=True, help="Disable custom CUDA kernels.")
@click.option(
    "--save_dir",
    type=click.Path(),
    default=None,
    help="Output directory for extracted inputs. Defaults to out_dir/processed/affinity_module_inputs.",
)
def main(
    out_dir: str,
    cache: str,
    affinity_checkpoint: Optional[str],
    devices: int,
    accelerator: str,
    num_workers: int,
    prescan_workers: int,
    no_prescan: bool,
    delete_bad: bool,
    recycling_steps: int,
    sampling_steps: int,
    diffusion_samples: int,
    seed: Optional[int],
    override: bool,
    no_kernels: bool,
    save_dir: Optional[str],
) -> None:
    """Extract affinity module inputs from an existing Boltz prediction run.

    OUT_DIR is the boltz results directory (e.g. ./boltz_results_mydata).
    """
    warnings.filterwarnings(
        "ignore", ".*that has Tensor Cores. To properly utilize them.*"
    )
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    if seed is not None:
        seed_everything(seed, workers=True)

    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")

    out_dir = Path(out_dir).expanduser()
    cache = Path(cache).expanduser()
    processed_dir = out_dir / "processed"
    mol_dir = cache / "mols"

    # Load manifest
    manifest = Manifest.load(processed_dir / "manifest.json")

    # Filter to records that have pre_affinity files and need extraction
    if save_dir is not None:
        output_path = Path(save_dir)
    else:
        output_path = processed_dir / "affinity_module_inputs"

    records_to_run = []
    for r in manifest.records:
        pre_aff = out_dir / "predictions" / r.id / f"pre_affinity_{r.id}.npz"
        if not pre_aff.exists():
            continue
        if not override:
            existing = output_path / r.id / f"affinity_input_{r.id}.npz"
            if existing.exists():
                continue
        records_to_run.append(r)

    if not records_to_run:
        click.echo("No records to process (all either missing pre_affinity or already extracted).")
        return

    if not no_prescan:
        predictions_dir = out_dir / "predictions"
        msa_dir = processed_dir / "msa"
        constraints_dir = (
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        )
        template_dir = (
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        )
        extra_mols_dir = (
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        )

        worker = partial(
            _prescan_record,
            target_dir=predictions_dir,
            msa_dir=msa_dir,
            constraints_dir=constraints_dir,
            template_dir=template_dir,
            extra_mols_dir=extra_mols_dir,
        )

        click.echo(
            f"Pre-scanning {len(records_to_run)} records with "
            f"{prescan_workers} workers (tokenize + crop)..."
        )

        bad_ids: dict[str, str] = {}
        n = len(records_to_run)
        ctx = mp.get_context("spawn")
        with ctx.Pool(prescan_workers) as pool:
            for i, (rid, ok, err) in enumerate(
                pool.imap_unordered(worker, records_to_run, chunksize=8), start=1
            ):
                if not ok:
                    bad_ids[rid] = err
                if i % 500 == 0 or i == n:
                    click.echo(
                        f"  pre-scan {i}/{n} ({len(bad_ids)} bad so far)"
                    )

        if bad_ids:
            click.echo(f"Pre-scan dropped {len(bad_ids)} records:")
            for rid, err in list(bad_ids.items())[:20]:
                click.echo(f"  {rid}: {err}")
            if len(bad_ids) > 20:
                click.echo(f"  ... and {len(bad_ids) - 20} more")

            if delete_bad:
                removed = 0
                for rid in bad_ids:
                    p = predictions_dir / rid / f"pre_affinity_{rid}.npz"
                    if p.exists():
                        p.unlink()
                        removed += 1
                click.echo(f"Deleted {removed} pre_affinity files.")

            records_to_run = [r for r in records_to_run if r.id not in bad_ids]

        if not records_to_run:
            click.echo("No records remain after pre-scan.")
            return

    click.echo(f"Extracting affinity inputs for {len(records_to_run)} records.")

    filtered_manifest = Manifest(records_to_run)

    # Data module — affinity mode loads pre_affinity_{id}.npz as the structure
    data_module = Boltz2InferenceDataModule(
        manifest=filtered_manifest,
        target_dir=out_dir / "predictions",
        msa_dir=processed_dir / "msa",
        mol_dir=mol_dir,
        num_workers=num_workers,
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
        override_method="other",
        affinity=True,
    )

    # Model params
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.5
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True)

    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.guidance_update = False
    steering_args.physical_guidance_update = False
    steering_args.contact_guidance_update = False

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    # Load affinity checkpoint
    if affinity_checkpoint is None:
        affinity_checkpoint = cache / "boltz2_aff.ckpt"
    click.echo(f"Loading checkpoint: {affinity_checkpoint}")

    model_module = Boltz2.load_from_checkpoint(
        affinity_checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    # Writer callback
    extractor = AffinityInputExtractor(output_dir=str(output_path))

    strategy = "auto"
    if isinstance(devices, int) and devices > 1:
        strategy = DDPStrategy(start_method="fork")
        devices = min(devices, len(records_to_run))

    trainer = Trainer(
        default_root_dir=str(out_dir),
        strategy=strategy,
        callbacks=[extractor],
        accelerator=accelerator,
        devices=devices,
        precision="bf16-mixed",
    )

    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )


if __name__ == "__main__":
    main()
