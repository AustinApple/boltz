"""Leaf-field ablation study for the assay-context embedding (test-time ablation).

Trains a small head on the pre-computed `g` + Qwen3 assay-context embedding (baseline,
``qwen3``) once per seed, then for every ablation variant (each leaf field of
``structured_description`` + the ``assay_type`` tag) feeds the masked test-set embeddings
into that pre-trained baseline head and records how the test metrics change. No
re-training per variant — only the test-time embeddings differ. The pairformer backbone
is **never trained** here; ``g`` comes from the cached
``<target_dir>/processed/g/<rid>/g_<rid>.npz`` files written by ``scripts/extract_affinity_g.py``.

Works for SPR / ITC / RBA / FPA; the assay type is a CLI argument and ``df_path`` / ``target_dir``
are derived from it.

Embeddings for every variant are cached up front by ``ablation_embed.build_ablation_embeddings``;
the baseline head reads from ``qwen3/`` at training time, and at evaluation each masked
variant is consumed by ``AffinityGHeadDataModule`` via its ``assay_embedding_model`` subdir
(``ablation_<field>/``).

Example:
    python scripts/train/train_AffinityModule_assaycontext_ablation_study.py \
        --assay_type itc --seeds 0,1,2 --max_epochs 50
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

# Reuse the Lightning wrapper / metrics from the g-head training script.
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

# Qwen3-Embedding-8B / sentence-transformers live in the llm_affinity env, not the boltz env,
# so the embedding pass is run as a subprocess with that interpreter.
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


def embedding_model_for(variant: str) -> str:
    """The ``assay_embedding_model`` subdir a variant reads from.

    The baseline reuses the existing ``qwen3/`` embeddings directly; every masked variant
    uses its own ``ablation_<field>/`` dir.
    """
    if variant == "baseline":
        return "qwen3"
    return f"ablation_{variant_dirname(variant)}"


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


def train_baseline(
    *,
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
    seed_list: list[int],
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
) -> Path:
    """Train the baseline (qwen3 embeddings) head once per seed; return the checkpoint root dir."""
    ckpt_root = Path(target_dir) / "checkpoints" / f"ablation_{assay_type}" / "baseline"
    assay_embedding_model = embedding_model_for("baseline")

    for seed in seed_list:
        ckpt_path = ckpt_root / f"seed_{seed}.ckpt"
        if ckpt_path.exists():
            print(f"[baseline] seed {seed}: checkpoint exists, skipping training")
            continue

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
        data_module = AffinityGHeadDataModule(datacfg)

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
                name=f"{assay_type}_baseline_seed{seed}",
                group=f"ablation_{assay_type}",
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

    return ckpt_root


def run_variant(
    variant: str,
    *,
    assay_type: str,
    df_path: str,
    target_dir: str,
    g_module_key: str,
    split_method: str,
    assay_context_dim: int,
    null_ids_set: frozenset[str],
    seed_list: list[int],
    baseline_ckpt_root: Path,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate one ablation variant on the test set using the pre-trained baseline checkpoint.

    The head weights come from the baseline checkpoint (trained on ``qwen3`` embeddings); only
    the test-time ``assay_embedding_model`` differs across variants. No re-training.
    """
    assay_embedding_model = embedding_model_for(variant)

    mses, rps, cindexes = [], [], []
    for seed in seed_list:
        ckpt_path = baseline_ckpt_root / f"seed_{seed}.ckpt"
        if not ckpt_path.exists():
            print(f"[{variant}] seed {seed}: baseline checkpoint missing at {ckpt_path}, skipping")
            continue
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
        model_module = LightningAffinityGHead.load_from_checkpoint(
            str(ckpt_path),
            map_location=torch.device("cpu"),
            assay_context_random_generation=False,
        )
        model_module.eval()
        data_module = AffinityGHeadDataModule(datacfg)
        trainer = Trainer(devices=[int(device)], accelerator="gpu", log_every_n_steps=1)
        result = trainer.test(model_module, datamodule=data_module)[0]
        mses.append(result.get("test/mse"))
        rps.append(result.get("test/rp"))
        cindexes.append(result.get("test/cindex"))

    def _stat(values: list, fn) -> float:
        clean = [float(v) for v in values if v is not None]
        return float(fn(clean)) if clean else float("nan")

    return {
        "variant": variant,
        "n_seeds": len([v for v in mses if v is not None]),
        "mse_mean": _stat(mses, np.mean),
        "mse_std": _stat(mses, np.std),
        "rp_mean": _stat(rps, np.mean),
        "rp_std": _stat(rps, np.std),
        "cindex_mean": _stat(cindexes, np.mean),
        "cindex_std": _stat(cindexes, np.std),
    }


def write_summary(rows: list[dict], assay_type: str) -> None:
    """Write the ablation summary as CSV + Markdown, sorted by Δrp vs baseline."""
    import pandas as pd

    df = pd.DataFrame(rows)
    baseline = df[df["variant"] == "baseline"]
    base_rp = float(baseline["rp_mean"].iloc[0]) if len(baseline) else float("nan")
    df["delta_rp_vs_baseline"] = df["rp_mean"] - base_rp

    # Most important field = largest drop in rp when removed (most negative delta), baseline first.
    df["_order"] = (df["variant"] != "baseline").astype(int)
    df = df.sort_values(["_order", "delta_rp_vs_baseline"]).drop(columns="_order").reset_index(drop=True)

    out_csv = _THIS_DIR / f"ablation_results_{assay_type}.csv"
    out_md = _THIS_DIR / f"ablation_results_{assay_type}.md"
    df.to_csv(out_csv, index=False)

    cols = ["variant", "n_seeds", "mse_mean", "mse_std", "rp_mean", "rp_std",
            "cindex_mean", "cindex_std", "delta_rp_vs_baseline"]
    lines = [
        f"# Ablation results — {assay_type.upper()}",
        "",
        f"Baseline test rp = {base_rp:.4f}. A more negative `delta_rp_vs_baseline` means removing "
        "that field hurt the model more, i.e. the field is more important.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, r in df.iterrows():
        cells = [str(r["variant"]), str(int(r["n_seeds"]))]
        cells += [f"{r[c]:.4f}" for c in cols[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[ablation] summary written to:\n  {out_csv}\n  {out_md}")


@click.command()
@click.option("--assay_type", required=True, type=click.Choice(list(ASSAY_LABELS.keys())),
              help="Assay type; df_path and target_dir are derived from it.")
@click.option("--token_z", type=int, default=128, show_default=True,
              help="Pairwise token dim of the cached g (must match the backbone that produced g_*.npz).")
@click.option("--input_token_s", type=int, default=384, show_default=True,
              help="Single-token dim of the cached g.")
@click.option("--g_module_key", type=click.Choice(["affinity_module1", "affinity_module2"]),
              default="affinity_module1", show_default=True,
              help="Which g vector to use from the cached g_*.npz.")
@click.option("--assay_context_dim", type=int, default=4096, show_default=True,
              help="Dimensionality of the assay-context vector fed to the head "
                   "(truncates the 4096-dim Qwen3 embedding; raise for higher ablation sensitivity).")
@click.option("--fusion", type=click.Choice(["concat", "film"]), default="concat", show_default=True,
              help="How to combine g with assay_context in the head: 'concat' or 'film'.")
@click.option("--null_context_assays", type=str, default="", show_default=True,
              help="Comma-separated assays whose context slot is replaced by the learned null "
                   "vector instead of a real embedding. Default '' = real context for every assay.")
@click.option("--seeds", type=str, default="0", show_default=True,
              help="Comma-separated list of seeds (e.g. '0,1,2').")
@click.option("--split_method", type=click.Choice(["pair-level-random", "pmid-level-random"]),
              default="pmid-level-random", show_default=True)
@click.option("--ablation_fields", type=str, default=None,
              help="Comma-separated subset of variants to run (default: baseline + all leaf "
                   "fields + assay_type). 'baseline' is always included.")
@click.option("--max_epochs", type=int, default=100, show_default=True)
@click.option("--device", type=str, default="0", show_default=True, help="CUDA device index.")
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight_decay", type=float, default=0.0, show_default=True)
@click.option("--patience", type=int, default=20, show_default=True)
@click.option("--embed_batch_size", type=int, default=16, show_default=True,
              help="Batch size for the Qwen3 embedding pass.")
@click.option("--skip_embedding", is_flag=True, default=False,
              help="Assume embeddings are already cached; skip the embedding step.")
@click.option("--use_wandb", is_flag=True, default=False, show_default=True)
@click.option("--wandb_project", type=str, default="affinity-assaycontext-ablation", show_default=True)
@click.option("--wandb_entity", type=str, default="news012147stw", show_default=True)
def main(
    assay_type: str,
    token_z: int,
    input_token_s: int,
    g_module_key: str,
    assay_context_dim: int,
    fusion: str,
    null_context_assays: str,
    seeds: str,
    split_method: str,
    ablation_fields: Optional[str],
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
    """Run the leaf-field ablation study for one assay type (head-only training)."""
    paths = assay_paths(assay_type)
    df_path = str(paths["df_path"])
    target_dir = str(paths["target_dir"])
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    null_set = frozenset(x.strip() for x in null_context_assays.split(",") if x.strip())
    bad = null_set - set(ASSAY_TYPES)
    if bad:
        raise click.UsageError(f"--null_context_assays has unknown names: {sorted(bad)}")
    null_ids = [ASSAY_TYPES.index(n) for n in null_set]

    # ---- Step 1: cache embeddings for every variant ----
    requested = None
    if ablation_fields:
        requested = [f.strip() for f in ablation_fields.split(",") if f.strip()]

    full_variants = ["baseline", *enumerate_leaf_fields(paths["schema"]), ASSAY_TYPE_VARIANT]
    variants = full_variants if requested is None else (
        ["baseline"] + [v for v in full_variants if v in requested and v != "baseline"]
    )

    if not skip_embedding:
        # Only non-baseline variants need fresh embeddings; baseline reads qwen3/ directly.
        to_embed = [v for v in variants if v != "baseline"]
        if to_embed:
            cmd = [
                EMBED_PYTHON, str(_THIS_DIR / "ablation_embed.py"),
                "--assay_type", assay_type, "--batch_size", str(embed_batch_size),
                "--variants", ",".join(to_embed),
                "--no_verify",
            ]
            print(f"[ablation] caching embeddings: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

    print(f"[ablation] {assay_type}: {len(variants)} variants to evaluate: {variants}")

    # ---- Step 2: train the baseline head once per seed ----
    baseline_ckpt_root = train_baseline(
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
        seed_list=seed_list,
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
    )

    # ---- Step 3: evaluate every variant against the baseline checkpoint ----
    rows = []
    for variant in variants:
        print(f"\n{'=' * 70}\n[ablation] variant: {variant}\n{'=' * 70}")
        rows.append(run_variant(
            variant,
            assay_type=assay_type,
            df_path=df_path,
            target_dir=target_dir,
            g_module_key=g_module_key,
            split_method=split_method,
            assay_context_dim=assay_context_dim,
            null_ids_set=null_set,
            seed_list=seed_list,
            baseline_ckpt_root=baseline_ckpt_root,
            device=device,
            batch_size=batch_size,
        ))

    # ---- Step 4: summary ----
    write_summary(rows, assay_type)


if __name__ == "__main__":
    main()
