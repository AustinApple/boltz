"""Train a small head on pre-computed g + assay-context to predict binding affinity.

The backbone (AffinityModule pairformer + pooling producing `g`) is frozen and
already applied by scripts/extract_affinity_g.py, whose outputs live at
``<target_dir>/processed/g/<rid>/g_<rid>.npz``. Only the head MLPs are trained.

Two training modes:

Mode A — single assay (legacy):
    --df_path ... --target_dir ... --assay_type {itc|spr|rba|fpa}
    Optionally --assay_context / --assay_context_dim ...

Mode B — joint multi-assay:
    --itc_df ... --itc_dir ...
    --spr_df ... --spr_dir ...
    --rba_df ... --rba_dir ...
    --fpa_df ... --fpa_dir ...
    --assay_context --assay_context_dim D
    (any subset of the four assay pairs may be supplied)

In both modes, ITC rows are routed through a learned null context vector rather
than a real per-sample embedding.
"""

from typing import Any, Dict, List, Optional, Tuple
import os

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from boltz.model.modules.affinity_g_head import AffinityGHead
from boltz.data.module.training_AffinityGHead import (
    ASSAY_TYPES,
    DEFAULT_NULL_CONTEXT_ASSAYS,
    ITC_ID,
    AffinityGHeadDataModule,
    AssayConfig,
    DataConfig,
)

try:
    from torchmetrics.functional import mean_squared_error, pearson_corrcoef
except Exception:
    mean_squared_error = None
    pearson_corrcoef = None


def get_cindex(pred: Tensor, gt: Tensor) -> Tensor:
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = diff > 0
    h_half = diff == 0
    return torch.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / torch.sum(gt_mask)


class LightningAffinityGHead(LightningModule):
    def __init__(
        self,
        token_z: int,
        input_token_s: int,
        assay_context: bool,
        assay_context_random_generation: bool,
        assay_context_dim: Optional[int],
        use_g: bool = True,
        fusion: str = "concat",
        null_context_ids: Optional[List[int]] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        scheduler: Optional[str] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.assay_context = assay_context
        self.assay_context_random_generation = assay_context_random_generation
        self.assay_context_dim = assay_context_dim
        self.use_g = use_g
        self.fusion = fusion
        self.null_context_ids = list(null_context_ids or [])

        self.model = AffinityGHead(
            token_z=token_z,
            input_token_s=input_token_s,
            assay_context_dim=assay_context_dim if assay_context else 0,
            use_g=use_g,
            fusion=fusion,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args or {}
        self.loss_fn = loss_fn or nn.MSELoss()

    def extract_inputs_targets(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Dict[str, Any], Tensor, Tensor]:
        g, assay_context_embedding, assay_type_ids, y = batch

        if self.assay_context and self.assay_context_dim is not None:
            if assay_context_embedding.shape[-1] > self.assay_context_dim:
                assay_context_embedding = assay_context_embedding[
                    ..., : self.assay_context_dim
                ]
            if self.assay_context_random_generation:
                assay_context_embedding = torch.randn_like(assay_context_embedding)
            assay_context_embedding = F.normalize(assay_context_embedding, p=2, dim=-1)

        if y.ndim > 1:
            y = y.squeeze(-1)

        model_inputs: Dict[str, Any] = {}
        if self.use_g:
            model_inputs["g"] = g
        if self.assay_context:
            model_inputs["assay_context"] = assay_context_embedding
            if self.null_context_ids:
                ids = torch.tensor(self.null_context_ids, device=assay_type_ids.device)
                no_context_mask = torch.isin(assay_type_ids, ids)
            else:
                no_context_mask = torch.zeros_like(assay_type_ids, dtype=torch.bool)
            model_inputs["no_context_mask"] = no_context_mask
        return model_inputs, y, assay_type_ids

    def step(self, batch, stage: str) -> Tensor:
        model_inputs, y, _ = self.extract_inputs_targets(batch)
        out = self.model(**model_inputs)
        y_hat = out["affinity_pred_value"]
        if y_hat.ndim > 1:
            y_hat = y_hat.squeeze(-1)

        loss = self.loss_fn(y_hat, y)

        if stage == "train":
            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            mse = mean_squared_error(y_hat, y)
            self.log(f"{stage}/mse", mse, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            r = pearson_corrcoef(y_hat, y)
            self.log(f"{stage}/rp", r, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            ci = get_cindex(y_hat, y)
            self.log(f"{stage}/cindex", ci, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
        elif stage == "val":
            if not hasattr(self, "_val_preds"):
                self._val_preds = []
                self._val_targets = []
            self._val_preds.append(y_hat.detach().float().cpu())
            self._val_targets.append(y.detach().float().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_targets = []

    def on_validation_epoch_end(self) -> None:
        if getattr(self, "_val_preds", None):
            preds = torch.cat(self._val_preds, dim=0)
            targets = torch.cat(self._val_targets, dim=0)
            mse = mean_squared_error(preds, targets)
            rp = pearson_corrcoef(preds, targets)
            ci = get_cindex(preds, targets)
            self.log("val/loss", mse, prog_bar=True, on_epoch=True)
            self.log("val/mse", mse, prog_bar=True, on_epoch=True)
            self.log("val/rp", rp, prog_bar=True, on_epoch=True)
            self.log("val/cindex", ci, prog_bar=True, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self._test_preds = []
        self._test_targets = []
        self._test_assay_ids = []

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        model_inputs, y, assay_type_ids = self.extract_inputs_targets(batch)
        out = self.model(**model_inputs)
        y_hat = out["affinity_pred_value"]
        if y_hat.ndim > 1:
            y_hat = y_hat.squeeze(-1)
        loss = self.loss_fn(y_hat, y)
        self._test_preds.append(y_hat.detach().float().cpu())
        self._test_targets.append(y.detach().float().cpu())
        self._test_assay_ids.append(assay_type_ids.detach().long().cpu())
        return loss

    def on_test_epoch_end(self) -> None:
        if self._test_preds and self._test_targets:
            preds = torch.cat(self._test_preds, dim=0)
            targets = torch.cat(self._test_targets, dim=0)
            assay_ids = torch.cat(self._test_assay_ids, dim=0)
            mse = mean_squared_error(preds, targets)
            rp = pearson_corrcoef(preds, targets)
            ci = get_cindex(preds, targets)
            self.log("test/loss", mse, prog_bar=True, on_epoch=True)
            self.log("test/mse", mse, prog_bar=True, on_epoch=True)
            self.log("test/rp", rp, prog_bar=True, on_epoch=True)
            self.log("test/cindex", ci, prog_bar=True, on_epoch=True)

            # Per-assay breakdown — only log for assays actually present in the test set
            # and with at least 2 samples (pearson_corrcoef needs >=2).
            for aid, name in enumerate(ASSAY_TYPES):
                sel = assay_ids == aid
                if sel.sum().item() < 2:
                    continue
                p = preds[sel]
                t = targets[sel]
                self.log(f"test/mse_{name}", mean_squared_error(p, t), on_epoch=True)
                self.log(f"test/rp_{name}", pearson_corrcoef(p, t), on_epoch=True)
                self.log(f"test/cindex_{name}", get_cindex(p, t), on_epoch=True)
                self.log(f"test/n_{name}", float(sel.sum().item()), on_epoch=True)

    def configure_optimizers(self):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "norm" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        optim = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.999),
        )
        if self.scheduler is None:
            return optim
        if self.scheduler == "cosine":
            t_max = self.trainer.max_epochs if self.trainer is not None else 100
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=t_max, **self.scheduler_args)
        elif self.scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                optim,
                step_size=self.scheduler_args.get("step_size", 10),
                gamma=self.scheduler_args.get("gamma", 0.1),
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val/loss"},
        }


def _resolve_assays(
    df_path: Optional[str],
    target_dir: Optional[str],
    assay_type: Optional[str],
    itc_df: Optional[str], itc_dir: Optional[str],
    spr_df: Optional[str], spr_dir: Optional[str],
    rba_df: Optional[str], rba_dir: Optional[str],
    fpa_df: Optional[str], fpa_dir: Optional[str],
    assay_context: bool,
) -> Tuple[List[AssayConfig], str]:
    """Build the list of AssayConfig from CLI args and return (assays, mode_tag)."""
    joint_pairs = [
        ("itc", itc_df, itc_dir),
        ("spr", spr_df, spr_dir),
        ("rba", rba_df, rba_dir),
        ("fpa", fpa_df, fpa_dir),
    ]
    any_joint = any((d is not None) or (t is not None) for _, d, t in joint_pairs)

    if any_joint:
        if df_path is not None or target_dir is not None:
            raise click.UsageError(
                "Cannot mix single-assay flags (--df_path/--target_dir) with joint-mode flags."
            )
        assays: List[AssayConfig] = []
        for name, d, t in joint_pairs:
            if d is None and t is None:
                continue
            if d is None or t is None:
                raise click.UsageError(f"For assay '{name}' both --{name}_df and --{name}_dir must be provided.")
            assays.append(AssayConfig(assay_type=name, df_path=d, target_dir=t))
        mode_tag = "assays_" + "-".join(a.assay_type for a in assays)
        return assays, mode_tag

    # Single-assay mode
    if df_path is None or target_dir is None or assay_type is None:
        raise click.UsageError(
            "Single-assay mode requires --df_path, --target_dir, and --assay_type."
        )
    if assay_type not in ASSAY_TYPES:
        raise click.UsageError(f"--assay_type must be one of {ASSAY_TYPES}")
    assays = [AssayConfig(assay_type=assay_type, df_path=df_path, target_dir=target_dir)]
    mode_tag = f"assays_{assay_type}"
    return assays, mode_tag


@click.command()
# Single-assay (legacy) mode flags
@click.option("--df_path", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--target_dir", type=click.Path(file_okay=False, path_type=str), default=None)
@click.option("--assay_type", type=click.Choice(ASSAY_TYPES), default=None,
              help="Required in single-assay mode.")
# Joint-mode per-assay flags
@click.option("--itc_df", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--itc_dir", type=click.Path(file_okay=False, path_type=str), default=None)
@click.option("--spr_df", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--spr_dir", type=click.Path(file_okay=False, path_type=str), default=None)
@click.option("--rba_df", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--rba_dir", type=click.Path(file_okay=False, path_type=str), default=None)
@click.option("--fpa_df", type=click.Path(exists=True, dir_okay=False, path_type=str), default=None)
@click.option("--fpa_dir", type=click.Path(file_okay=False, path_type=str), default=None)
# Shared flags
@click.option("--g_module_key", type=click.Choice(["affinity_module1", "affinity_module2"]), default="affinity_module1", show_default=True)
@click.option("--token_z", type=int, default=128, show_default=True)
@click.option("--input_token_s", type=int, default=384, show_default=True)
@click.option("--assay_context", is_flag=True)
@click.option("--no_g", is_flag=True,
              help="Drop g from the head (context-only ablation). Requires --assay_context.")
@click.option("--assay_context_random_generation", is_flag=True)
@click.option("--assay_context_dim", type=int, default=128, show_default=True)
@click.option("--fusion", type=click.Choice(["concat", "film"]), default="concat", show_default=True,
              help="How to combine g with assay_context: 'concat' (original) or 'film' "
                   "(Feature-wise Linear Modulation; requires both g and assay_context).")
@click.option("--null_context_assays", type=str, default="itc", show_default=True,
              help="Comma-separated assays whose context slot is replaced by the learned null "
                   "vector instead of a real embedding. Pass '' (empty) to use real context "
                   "for every assay.")
@click.option("--assay_embedding_model", default="qwen3", show_default=True)
@click.option("--seeds", type=str, default="0", show_default=True)
@click.option("--split_method", type=click.Choice(["pair-level-random", "pmid-level-random"]), default="pmid-level-random", show_default=True)
@click.option("--max_epochs", type=int, default=50, show_default=True)
@click.option("--device", type=str, default="0", show_default=True)
@click.option("--batch_size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight_decay", type=float, default=0.0, show_default=True)
@click.option("--patience", type=int, default=20, show_default=True)
@click.option("--ckpt_root", type=click.Path(file_okay=False, path_type=str), default=None,
              help="Where to write checkpoints. Defaults to the (single) target_dir/checkpoints in Mode A, or ./checkpoints in Mode B.")
@click.option("--use_wandb", is_flag=True)
@click.option("--wandb_project", type=str, default="affinity-g-head", show_default=True)
@click.option("--wandb_entity", type=str, default="news012147stw", show_default=True)
@click.option("--wandb_name", type=str, default=None)
def main(
    df_path: Optional[str],
    target_dir: Optional[str],
    assay_type: Optional[str],
    itc_df: Optional[str], itc_dir: Optional[str],
    spr_df: Optional[str], spr_dir: Optional[str],
    rba_df: Optional[str], rba_dir: Optional[str],
    fpa_df: Optional[str], fpa_dir: Optional[str],
    g_module_key: str,
    token_z: int,
    input_token_s: int,
    assay_context: bool,
    no_g: bool,
    assay_context_random_generation: bool,
    assay_context_dim: int,
    fusion: str,
    null_context_assays: str,
    assay_embedding_model: str,
    seeds: str,
    split_method: str,
    max_epochs: int,
    device: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    ckpt_root: Optional[str],
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_name: Optional[str],
) -> None:
    assays, mode_tag = _resolve_assays(
        df_path, target_dir, assay_type,
        itc_df, itc_dir, spr_df, spr_dir, rba_df, rba_dir, fpa_df, fpa_dir,
        assay_context,
    )

    if no_g and not assay_context:
        raise click.UsageError("--no_g requires --assay_context (context-only needs a context input).")
    use_g = not no_g

    if fusion == "film" and (not assay_context or not use_g):
        raise click.UsageError("--fusion film requires both g and --assay_context (no --no_g).")

    null_set = frozenset(x.strip() for x in null_context_assays.split(",") if x.strip())
    bad = null_set - set(ASSAY_TYPES)
    if bad:
        raise click.UsageError(f"--null_context_assays has unknown names: {sorted(bad)}")
    null_ids = [ASSAY_TYPES.index(n) for n in null_set]

    # Ckpt root: legacy behavior in single-assay mode keeps writing under target_dir/checkpoints.
    if ckpt_root is None:
        if len(assays) == 1:
            ckpt_root = os.path.join(assays[0].target_dir, "checkpoints")
        else:
            ckpt_root = os.path.join(os.getcwd(), "checkpoints")

    monitor_metric = "val/loss"
    monitor_mode = "min"

    all_test_mse: list = []
    all_test_rp: list = []
    all_test_cindex: list = []

    seed_list = [int(s) for s in seeds.split(",") if s.strip()]

    head_tag = (
        "c_only" if (assay_context and not use_g)
        else ("g_c" if assay_context else "g_only")
    )
    null_tag = ("null_" + "-".join(sorted(null_set))) if (assay_context and null_set) else "null_none"
    ckpt_dir = os.path.join(
        ckpt_root,
        f"head_{head_tag}_{g_module_key}_{mode_tag}_dim_{assay_context_dim if assay_context else 'NA'}_{null_tag}_split_{split_method}_fusion_{fusion}",
    )

    for seed in seed_list:
        Datacfg = DataConfig(
            assays=assays,
            seed=seed,
            batch_size=batch_size,
            assay_embedding_model=assay_embedding_model,
            g_module_key=g_module_key,
            split_method=split_method,
            load_assay_context=assay_context,
            assay_context_dim=assay_context_dim,
            null_context_assays=null_set,
        )

        ckpt_path = os.path.join(ckpt_dir, f"seed_{Datacfg.seed}.ckpt")
        if os.path.exists(ckpt_path):
            print(f"Checkpoint for seed {Datacfg.seed} already exists at {ckpt_path}, skipping training.")
        else:
            early_stopping = EarlyStopping(
                monitor=monitor_metric, mode=monitor_mode, patience=patience, min_delta=0.0, verbose=True
            )
            checkpoint_cb = ModelCheckpoint(
                dirpath=ckpt_dir,
                filename=f"seed_{Datacfg.seed}",
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=1,
                save_last=False,
                auto_insert_metric_name=False,
            )

            model_module = LightningAffinityGHead(
                token_z=token_z,
                input_token_s=input_token_s,
                assay_context=assay_context,
                assay_context_random_generation=assay_context_random_generation,
                assay_context_dim=assay_context_dim,
                use_g=use_g,
                fusion=fusion,
                null_context_ids=null_ids,
                lr=lr,
                weight_decay=weight_decay,
            )
            data_module = AffinityGHeadDataModule(Datacfg)

            if use_wandb:
                wandb_logger = WandbLogger(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=f'{wandb_name}_seed{seed}',
                    group=f"g_head_{g_module_key}_{mode_tag}",
                    save_dir=ckpt_dir,
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

    for seed in seed_list:
        Datacfg = DataConfig(
            assays=assays,
            seed=seed,
            batch_size=batch_size,
            assay_embedding_model=assay_embedding_model,
            g_module_key=g_module_key,
            split_method=split_method,
            load_assay_context=assay_context,
            assay_context_dim=assay_context_dim,
            null_context_assays=null_set,
        )
        ckpt_path = os.path.join(ckpt_dir, f"seed_{Datacfg.seed}.ckpt")

        model_module = LightningAffinityGHead.load_from_checkpoint(
            ckpt_path,
            map_location=torch.device("cpu"),
            assay_context_random_generation=assay_context_random_generation,
        )
        model_module.eval()
        data_module = AffinityGHeadDataModule(Datacfg)
        trainer = Trainer(devices=[int(device)], accelerator="gpu", log_every_n_steps=1)
        test_result = trainer.test(model_module, datamodule=data_module)

        all_test_mse.append(test_result[0].get("test/mse", None))
        all_test_rp.append(test_result[0].get("test/rp", None))
        all_test_cindex.append(test_result[0].get("test/cindex", None))

    print(f"mean test mse: {np.mean(all_test_mse):.2f}  std: {np.std(all_test_mse):.2f}")
    print(f"mean test rp : {np.mean(all_test_rp):.2f}  std: {np.std(all_test_rp):.2f}")
    print(f"mean test ci : {np.mean(all_test_cindex):.2f}  std: {np.std(all_test_cindex):.2f}")


if __name__ == "__main__":
    main()
