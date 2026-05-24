"""Leaf-field ablation study for the assay-context embedding.

Trains the affinity model once per ablation variant (baseline + every leaf field of
``structured_description`` + the ``assay_type`` tag) and compares test metrics, so we can see
which field carries predictive signal. Works for SPR / ITC / RBA / FPA; the assay type is a CLI
argument and ``df_path`` / ``target_dir`` are derived from it.

Embeddings for every variant are cached up front by ``ablation_embed.build_ablation_embeddings``;
each variant is then consumed by ``AffinityModuleDataModule`` via its ``assay_embedding_model``
subdir (``qwen3`` for the baseline, ``ablation_<field>`` for each masked variant).

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
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Reuse the Lightning wrapper / metrics from the single-run training script.
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
from train_AffinityModule_assaycontext import LightningAffinityModule  # noqa: E402

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

from boltz.data.module.training_AffinityModule_assaycontext import (  # noqa: E402
    AffinityModuleDataModule,
    DataConfig,
)


def embedding_model_for(variant: str) -> str:
    """The ``assay_embedding_model`` subdir a variant trains from.

    The baseline reuses the existing ``qwen3/`` embeddings directly (verified equal to a
    freshly generated baseline); every masked variant uses its own ``ablation_<field>/`` dir.
    """
    if variant == "baseline":
        return "qwen3"
    return f"ablation_{variant_dirname(variant)}"


def run_variant(
    variant: str,
    *,
    assay_type: str,
    cfg,
    df_path: str,
    target_dir: str,
    assay_context_dim: int,
    assay_embedding_injection_method: str,
    seed_list: list[int],
    split_method: str,
    max_epochs: int,
    device: str,
    batch_size: int,
    patience: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
) -> dict[str, float]:
    """Train + evaluate one ablation variant across all seeds; return mean/std metrics."""
    assay_embedding_model = embedding_model_for(variant)
    ckpt_root = Path(target_dir) / "checkpoints" / f"ablation_{assay_type}" / variant_dirname(variant)

    # ---- Training ----
    for seed in seed_list:
        ckpt_path = ckpt_root / f"seed_{seed}.ckpt"
        if ckpt_path.exists():
            print(f"[{variant}] seed {seed}: checkpoint exists, skipping training")
            continue

        datacfg = DataConfig(
            df_path=df_path,
            target_dir=target_dir,
            seed=seed,
            batch_size=batch_size,
            assay_embedding_model=assay_embedding_model,
            split_method=split_method,
        )
        model_module = LightningAffinityModule(
            token_s=cfg.token_s,
            token_z=cfg.token_z,
            assay_context=True,
            assay_context_random_generation=False,
            assay_context_dim=assay_context_dim,
            assay_embedding_injection_method=assay_embedding_injection_method,
            affinity_model_args=cfg.affinity_model_args2,
        )
        data_module = AffinityModuleDataModule(datacfg)

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
                name=f"{assay_type}_{variant_dirname(variant)}_seed{seed}",
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

    # ---- Evaluation ----
    mses, rps, cindexes = [], [], []
    for seed in seed_list:
        ckpt_path = ckpt_root / f"seed_{seed}.ckpt"
        if not ckpt_path.exists():
            print(f"[{variant}] seed {seed}: no checkpoint, skipping eval")
            continue
        datacfg = DataConfig(
            df_path=df_path,
            target_dir=target_dir,
            seed=seed,
            batch_size=batch_size,
            assay_embedding_model=assay_embedding_model,
            split_method=split_method,
        )
        model_module = LightningAffinityModule.load_from_checkpoint(
            str(ckpt_path),
            map_location=torch.device("cpu"),
            assay_context_random_generation=False,
            assay_embedding_injection_method=assay_embedding_injection_method,
        )
        model_module.eval()
        data_module = AffinityModuleDataModule(datacfg)
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
@click.option("--hparams_yaml", type=click.Path(exists=True, dir_okay=False, path_type=str),
              default="/data/mwu11/boltz/BindingDB/itc/boltz_results_yaml_affinity_input/lightning_logs/version_1/hparams.yaml",
              show_default=True, help="Lightning hparams.yaml providing model architecture args.")
@click.option("--assay_context_dim", type=int, default=4096, show_default=True,
              help="Dimensionality of the assay-context vector fed to the model "
                   "(truncates the 4096-dim Qwen3 embedding; raise for higher ablation sensitivity).")
@click.option("--assay_embedding_injection_method",
              type=click.Choice(["dual_film", "concatenate_g_after_meanpooling"]),
              default="concatenate_g_after_meanpooling", show_default=True)
@click.option("--seeds", type=str, default="0", show_default=True,
              help="Comma-separated list of seeds (e.g. '0,1,2').")
@click.option("--split_method", type=click.Choice(["pair-level-random", "pmid-level-random"]),
              default="pmid-level-random", show_default=True)
@click.option("--ablation_fields", type=str, default=None,
              help="Comma-separated subset of variants to run (default: baseline + all leaf "
                   "fields + assay_type). 'baseline' is always included.")
@click.option("--max_epochs", type=int, default=50, show_default=True)
@click.option("--device", type=str, default="0", show_default=True, help="CUDA device index.")
@click.option("--batch_size", type=int, default=16, show_default=True)
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
    hparams_yaml: str,
    assay_context_dim: int,
    assay_embedding_injection_method: str,
    seeds: str,
    split_method: str,
    ablation_fields: Optional[str],
    max_epochs: int,
    device: str,
    batch_size: int,
    patience: int,
    embed_batch_size: int,
    skip_embedding: bool,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
) -> None:
    """Run the leaf-field ablation study for one assay type."""
    paths = assay_paths(assay_type)
    df_path = str(paths["df_path"])
    target_dir = str(paths["target_dir"])
    cfg = OmegaConf.load(hparams_yaml)
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    # ---- Step 1: cache embeddings for every variant ----
    requested = None
    if ablation_fields:
        requested = [f.strip() for f in ablation_fields.split(",") if f.strip()]

    if not skip_embedding:
        # Run the Qwen3 embedding pass via the llm_affinity env (has sentence-transformers).
        cmd = [
            EMBED_PYTHON, str(_THIS_DIR / "ablation_embed.py"),
            "--assay_type", assay_type, "--batch_size", str(embed_batch_size),
        ]
        if requested is not None:
            fields = [f for f in requested if f not in ("baseline", ASSAY_TYPE_VARIANT)]
            if fields:
                cmd += ["--fields", ",".join(fields)]
        print(f"[ablation] caching embeddings: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    variants = ["baseline", *enumerate_leaf_fields(paths["schema"]), ASSAY_TYPE_VARIANT]
    if requested is not None:
        variants = ["baseline"] + [v for v in variants if v in requested and v != "baseline"]

    print(f"[ablation] {assay_type}: {len(variants)} variants to train: {variants}")

    # ---- Step 2: train + evaluate every variant ----
    rows = []
    for variant in variants:
        print(f"\n{'=' * 70}\n[ablation] variant: {variant}\n{'=' * 70}")
        rows.append(run_variant(
            variant,
            assay_type=assay_type,
            cfg=cfg,
            df_path=df_path,
            target_dir=target_dir,
            assay_context_dim=assay_context_dim,
            assay_embedding_injection_method=assay_embedding_injection_method,
            seed_list=seed_list,
            split_method=split_method,
            max_epochs=max_epochs,
            device=device,
            batch_size=batch_size,
            patience=patience,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
        ))

    # ---- Step 3: summary ----
    write_summary(rows, assay_type)


if __name__ == "__main__":
    main()
