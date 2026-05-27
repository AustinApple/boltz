"""Retrain the affinity head on Qwen3 embeddings with user-selected fields masked out.

Sibling to ``train_AffinityModule_assaycontext_ablation_study.py``: same head, same cached
``g`` backbone, same data module. The difference is that this script **retrains** the head
on the masked embeddings (per seed) instead of evaluating a pre-trained baseline at test
time. Useful when you want a model whose head actually learned without the masked fields.

The user picks the fields to mask on the CLI via ``--mask_fields`` (comma-separated). The
fields must be leaf paths of ``structured_description`` (see ``<assay>_schema.json``); the
special token ``assay_type`` drops the appended assay-type tag instead of a leaf. Pass
multiple fields and they are all removed from the JSON before Qwen3 encoding.

Example:
    python scripts/train/train_AffinityModule_assaycontext_masked.py \
        --assay_type itc \
        --mask_fields protein.name,reactant.identity \
        --seeds 0,1,2 --max_epochs 100
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ablation_embed import (  # noqa: E402
    ASSAY_LABELS,
    ASSAY_TYPE_VARIANT,
    assay_paths,
    enumerate_leaf_fields,
    variant_dirname,
)
from train_AffinityGHead import LightningAffinityGHead  # noqa: E402

_EMBED_PYTHON_CANDIDATES = (
    "/data/mwu11/miniconda3/envs/llm_affinity/bin/python",
    "/data/mwu11/anaconda3/envs/llm_affinity/bin/python",
)
EMBED_PYTHON = next(
    (p for p in _EMBED_PYTHON_CANDIDATES if Path(p).exists()),
    _EMBED_PYTHON_CANDIDATES[0],
)

from boltz.data.module.training_AffinityGHead import (  # noqa: E402
    ASSAY_TYPES,
    AffinityGHeadDataModule,
    AssayConfig,
    DataConfig,
)


def _validate_mask_fields(mask_fields: list[str], schema_path: Path) -> None:
    """Verify every requested mask field is a leaf of the schema (or the assay_type tag)."""
    valid_leaves = set(enumerate_leaf_fields(schema_path))
    unknown = [f for f in mask_fields if f != ASSAY_TYPE_VARIANT and f not in valid_leaves]
    if unknown:
        raise click.UsageError(
            f"--mask_fields contains unknown fields: {unknown}.\n"
            f"Valid leaves: {sorted(valid_leaves)} (plus the special token '{ASSAY_TYPE_VARIANT}')."
        )


def _datacfg(
    *,
    assay_type: str,
    df_path: str,
    target_dir: str,
    seed: int,
    batch_size: int,
    assay_embedding_model: str,
    g_module_key: str,
    split_method: str,
    assay_context_dim: int,
    null_ids_set: frozenset[str],
) -> DataConfig:
    return DataConfig(
        assays=[AssayConfig(assay_type=assay_type, df_path=df_path, target_dir=target_dir)],
        seed=seed,
        batch_size=batch_size,
        assay_embedding_model=assay_embedding_model,
        g_module_key=g_module_key,
        split_method=split_method,
        load_assay_context=True,
        assay_context_dim=assay_context_dim,
        null_context_assays=null_ids_set,
    )


def train_and_eval_seed(
    *,
    seed: int,
    variant: str,
    ckpt_root: Path,
    assay_type: str,
    df_path: str,
    target_dir: str,
    token_z: int,
    input_token_s: int,
    assay_context_dim: int,
    g_module_key: str,
    fusion: str,
    null_ids_set: frozenset[str],
    null_ids: list[int],
    split_method: str,
    max_epochs: int,
    device: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    run_name: str,
    group: str,
) -> dict[str, Optional[float]]:
    """Train the head from scratch on the masked embedding, then test the best checkpoint."""
    assay_embedding_model = f"ablation_{variant_dirname(variant)}"
    ckpt_path = ckpt_root / f"seed_{seed}.ckpt"

    datacfg = _datacfg(
        assay_type=assay_type,
        df_path=df_path,
        target_dir=target_dir,
        seed=seed,
        batch_size=batch_size,
        assay_embedding_model=assay_embedding_model,
        g_module_key=g_module_key,
        split_method=split_method,
        assay_context_dim=assay_context_dim,
        null_ids_set=null_ids_set,
    )
    data_module = AffinityGHeadDataModule(datacfg)

    if ckpt_path.exists():
        print(f"[masked] seed {seed}: checkpoint exists at {ckpt_path}, skipping training")
        model_module = LightningAffinityGHead.load_from_checkpoint(
            str(ckpt_path),
            map_location=torch.device("cpu"),
            assay_context_random_generation=False,
        )
    else:
        model_module = LightningAffinityGHead(
            token_z=token_z,
            input_token_s=input_token_s,
            assay_context=True,
            assay_context_random_generation=False,
            assay_context_dim=assay_context_dim,
            use_g=True,
            fusion=fusion,
            null_context_ids=null_ids,
            lr=lr,
            weight_decay=weight_decay,
        )

        early_stopping = EarlyStopping(
            monitor="val/loss", mode="min", patience=patience, min_delta=0.0, verbose=True
        )
        checkpoint_cb = ModelCheckpoint(
            dirpath=str(ckpt_root),
            filename=f"seed_{seed}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        )
        if use_wandb:
            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                group=group,
                save_dir=str(ckpt_root),
                log_model=False,
            )
        else:
            wandb_logger = None

        trainer = Trainer(
            max_epochs=max_epochs,
            devices=[int(device)],
            accelerator="gpu",
            log_every_n_steps=1,
            callbacks=[early_stopping, checkpoint_cb],
            logger=wandb_logger if wandb_logger else True,
        )
        trainer.fit(model_module, datamodule=data_module)
        if wandb_logger is not None:
            wandb_logger.experiment.finish()

        model_module = LightningAffinityGHead.load_from_checkpoint(
            str(ckpt_path),
            map_location=torch.device("cpu"),
            assay_context_random_generation=False,
        )

    model_module.eval()
    test_trainer = Trainer(devices=[int(device)], accelerator="gpu", log_every_n_steps=1)
    result = test_trainer.test(model_module, datamodule=data_module)[0]
    return {
        "seed": seed,
        "test/mse": result.get("test/mse"),
        "test/rp": result.get("test/rp"),
        "test/cindex": result.get("test/cindex"),
    }


def _stat(values: list, fn) -> float:
    clean = [float(v) for v in values if v is not None]
    return float(fn(clean)) if clean else float("nan")


def write_summary(rows: list[dict], assay_type: str, variant: str, out_dir: Path) -> None:
    """Persist per-seed results plus mean/std as CSV + Markdown."""
    import pandas as pd

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"masked_results_{assay_type}_{variant_dirname(variant)}.csv"
    out_md = out_dir / f"masked_results_{assay_type}_{variant_dirname(variant)}.md"
    df.to_csv(out_csv, index=False)

    mses = [r["test/mse"] for r in rows]
    rps = [r["test/rp"] for r in rows]
    cs = [r["test/cindex"] for r in rows]

    lines = [
        f"# Masked-embedding retraining — {assay_type.upper()}",
        "",
        f"Masked fields: `{variant}`",
        f"Embedding dir: `ablation_{variant_dirname(variant)}/`",
        "",
        "| seed | test/mse | test/rp | test/cindex |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "—"
        lines.append(f"| {r['seed']} | {fmt(r['test/mse'])} | {fmt(r['test/rp'])} | {fmt(r['test/cindex'])} |")
    lines += [
        "",
        f"**mse**: mean={_stat(mses, np.mean):.4f}, std={_stat(mses, np.std):.4f}",
        f"**rp**:  mean={_stat(rps, np.mean):.4f}, std={_stat(rps, np.std):.4f}",
        f"**cindex**: mean={_stat(cs, np.mean):.4f}, std={_stat(cs, np.std):.4f}",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[masked] summary written to:\n  {out_csv}\n  {out_md}")


@click.command()
@click.option("--assay_type", required=True, type=click.Choice(list(ASSAY_LABELS.keys())),
              help="Assay type; df_path and target_dir are derived from it.")
@click.option("--mask_fields", required=True, type=str,
              help="Comma-separated leaf fields of structured_description to mask "
                   "(e.g. 'protein.name,reactant.identity'). Use the special token "
                   "'assay_type' to also drop the appended assay-type tag.")
@click.option("--token_z", type=int, default=128, show_default=True,
              help="Pairwise token dim of the cached g.")
@click.option("--input_token_s", type=int, default=384, show_default=True,
              help="Single-token dim of the cached g.")
@click.option("--g_module_key", type=click.Choice(["affinity_module1", "affinity_module2"]),
              default="affinity_module1", show_default=True,
              help="Which g vector to use from the cached g_*.npz.")
@click.option("--assay_context_dim", type=int, default=4096, show_default=True,
              help="Dimensionality of the assay-context vector fed to the head.")
@click.option("--fusion", type=click.Choice(["concat", "film"]), default="concat", show_default=True,
              help="How to combine g with assay_context in the head.")
@click.option("--null_context_assays", type=str, default="", show_default=True,
              help="Comma-separated assays whose context slot is replaced by the learned null "
                   "vector instead of a real embedding.")
@click.option("--seeds", type=str, default="0", show_default=True,
              help="Comma-separated list of seeds (e.g. '0,1,2').")
@click.option("--split_method", type=click.Choice(["pair-level-random", "pmid-level-random"]),
              default="pmid-level-random", show_default=True)
@click.option("--max_epochs", type=int, default=100, show_default=True)
@click.option("--device", type=str, default="0", show_default=True, help="CUDA device index.")
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight_decay", type=float, default=0.0, show_default=True)
@click.option("--patience", type=int, default=20, show_default=True)
@click.option("--embed_batch_size", type=int, default=16, show_default=True,
              help="Batch size for the Qwen3 embedding pass.")
@click.option("--skip_embedding", is_flag=True, default=False,
              help="Assume the masked embedding is already cached; skip the embedding step.")
@click.option("--use_wandb", is_flag=True, default=False, show_default=True)
@click.option("--wandb_project", type=str, default="affinity-assaycontext-masked", show_default=True)
@click.option("--wandb_entity", type=str, default="news012147stw", show_default=True)
def main(
    assay_type: str,
    mask_fields: str,
    token_z: int,
    input_token_s: int,
    g_module_key: str,
    assay_context_dim: int,
    fusion: str,
    null_context_assays: str,
    seeds: str,
    split_method: str,
    max_epochs: int,
    device: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    embed_batch_size: int,
    skip_embedding: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
) -> None:
    """Retrain the affinity head with the user-selected fields masked from the assay context."""
    paths = assay_paths(assay_type)
    df_path = str(paths["df_path"])
    target_dir = str(paths["target_dir"])
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    fields = [f.strip() for f in mask_fields.split(",") if f.strip()]
    if not fields:
        raise click.UsageError("--mask_fields must list at least one field")
    _validate_mask_fields(fields, paths["schema"])
    # Canonical order so the same set always yields the same dir / variant string.
    fields_sorted = sorted(set(fields))
    variant = "+".join(fields_sorted)

    null_set = frozenset(x.strip() for x in null_context_assays.split(",") if x.strip())
    bad = null_set - set(ASSAY_TYPES)
    if bad:
        raise click.UsageError(f"--null_context_assays has unknown names: {sorted(bad)}")
    null_ids = [ASSAY_TYPES.index(n) for n in null_set]

    print(f"[masked] {assay_type}: masking {fields_sorted} -> variant='{variant}'")

    # ---- Step 1: cache the masked embeddings (one variant) ----
    if not skip_embedding:
        cmd = [
            EMBED_PYTHON, str(_THIS_DIR / "ablation_embed.py"),
            "--assay_type", assay_type, "--batch_size", str(embed_batch_size),
            "--variants", variant,
            "--no_verify",
        ]
        print(f"[masked] caching embeddings: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    # ---- Step 2: retrain per seed and collect test metrics ----
    ckpt_root = Path(target_dir) / "checkpoints" / f"masked_{assay_type}" / variant_dirname(variant)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    group = f"masked_{assay_type}_{variant_dirname(variant)}"

    rows: list[dict] = []
    for seed in seed_list:
        print(f"\n{'=' * 70}\n[masked] seed {seed} | variant '{variant}'\n{'=' * 70}")
        rows.append(train_and_eval_seed(
            seed=seed,
            variant=variant,
            ckpt_root=ckpt_root,
            assay_type=assay_type,
            df_path=df_path,
            target_dir=target_dir,
            token_z=token_z,
            input_token_s=input_token_s,
            assay_context_dim=assay_context_dim,
            g_module_key=g_module_key,
            fusion=fusion,
            null_ids_set=null_set,
            null_ids=null_ids,
            split_method=split_method,
            max_epochs=max_epochs,
            device=device,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            run_name=f"{assay_type}_{variant_dirname(variant)}_seed{seed}",
            group=group,
        ))

    # ---- Step 3: summary ----
    write_summary(rows, assay_type, variant, _THIS_DIR)


if __name__ == "__main__":
    main()
