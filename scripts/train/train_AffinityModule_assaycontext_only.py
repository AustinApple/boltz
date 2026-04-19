
#%%
# lightning_affinity_assaycontext_only.py
from typing import Any, Dict, Optional, Tuple
import os
import click

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from boltz.model.modules.affinity_assaycontext_only import AffinityAssayContextOnlyModule
from boltz.data.module.training_AffinityModule_assaycontext_only import AssayContextOnlyDataModule, DataConfig

import numpy as np

try:
    from torchmetrics.functional import mean_squared_error, pearson_corrcoef
except Exception:
    mean_squared_error = None
    pearson_corrcoef = None


def get_cindex(pred: Tensor, gt: Tensor) -> Tensor:
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = torch.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / torch.sum(gt_mask)
    return CI


class LightningAffinityAssayContextOnlyModule(LightningModule):
    """
    Lightning wrapper for assay-context-only affinity prediction.

    Expected batch format:
        batch = (assay_context, y)
        assay_context: FloatTensor[B, assay_context_dim]
        y:             FloatTensor[B, 1]   (pKd)
    """

    def __init__(
        self,
        assay_context_dim: int = 128,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        scheduler: Optional[str] = None,
        scheduler_args: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.assay_context_dim = assay_context_dim

        self.model = AffinityAssayContextOnlyModule(
            assay_context_dim=assay_context_dim,
            hidden_dim=hidden_dim,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args or {}
        self.loss_fn = loss_fn or nn.MSELoss()

    def extract_inputs_targets(self, batch: tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Unpack batch and prepare inputs."""
        assay_context: Tensor = batch[0]
        y: Tensor = batch[1]

        # Truncate to assay_context_dim if needed
        if assay_context.shape[1] > self.assay_context_dim:
            assay_context = assay_context[:, :self.assay_context_dim]

        # L2 normalize
        assay_context = F.normalize(assay_context, p=2, dim=-1)

        if y.ndim > 1:
            y = y.squeeze(-1)

        return assay_context, y

    def step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        assay_context, y = self.extract_inputs_targets(batch)
        out = self.model(assay_context)
        y_hat = out['affinity_pred_value']

        if y_hat.ndim > 1:
            y_hat = y_hat.squeeze(-1)

        loss = self.loss_fn(y_hat, y)

        if stage == "train":
            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            mse = mean_squared_error(y_hat, y)
            self.log(f"{stage}/mse", mse, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            r = pearson_corrcoef(y_hat, y)
            self.log(f"{stage}/rp", r, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            cindex = get_cindex(y_hat, y)
            self.log(f"{stage}/cindex", cindex, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
        else:
            if stage == "val":
                if not hasattr(self, "_val_preds"):
                    self._val_preds = []
                    self._val_targets = []
                self._val_preds.append(y_hat.detach().float().cpu())
                self._val_targets.append(y.detach().float().cpu())

        return loss

    def training_step(self, batch, batch_idx: int) -> Tensor:
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
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
            cindex = get_cindex(preds, targets)
            self.log("val/loss", mse, prog_bar=True, on_epoch=True)
            self.log("val/mse", mse, prog_bar=True, on_epoch=True)
            self.log("val/rp", rp, prog_bar=True, on_epoch=True)
            self.log("val/cindex", cindex, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        assay_context, y = self.extract_inputs_targets(batch)
        out = self.model(assay_context)
        y_hat = out['affinity_pred_value']

        if y_hat.ndim > 1:
            y_hat = y_hat.squeeze(-1)

        loss = self.loss_fn(y_hat, y)

        if not hasattr(self, "_test_preds"):
            self._test_preds = []
            self._test_targets = []
        self._test_preds.append(y_hat.detach().float().cpu())
        self._test_targets.append(y.detach().float().cpu())

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:
        assay_context, _ = self.extract_inputs_targets(batch)
        out = self.model(assay_context)
        return {"pred": out['affinity_pred_value']}

    def on_test_epoch_start(self) -> None:
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self) -> None:
        preds_list = self._test_preds
        targets_list = self._test_targets
        if preds_list and targets_list:
            preds = torch.cat(preds_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
            mse = mean_squared_error(preds, targets)
            rp = pearson_corrcoef(preds, targets)
            cindex = get_cindex(preds, targets)
            self.log("test/loss", mse, prog_bar=True, on_epoch=True)
            self.log("test/mse", mse, prog_bar=True, on_epoch=True)
            self.log("test/rp", rp, prog_bar=True, on_epoch=True)
            self.log("test/cindex", cindex, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
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
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }


@click.command()
@click.option("--df_path", type=click.Path(exists=True, dir_okay=False, path_type=str), default="/data/mwu11/LLM_affinity/BindingDB/non_null_spr_itc_structured_description_subset.csv", show_default=True, help="Path to CSV with binding data.")
@click.option("--target_dir", type=click.Path(file_okay=False, path_type=str), default="/data/mwu11/boltz/BindingDB/boltz_results_yaml_affinity_input", show_default=True, help="Directory with processed assay context embeddings.")
@click.option("--assay_context_dim", type=int, default=128, show_default=True, help="Dimensionality of assay context input to model.")
@click.option("--hidden_dim", type=int, default=256, show_default=True, help="Hidden dimension of the MLP.")
@click.option("--assay_embedding_model", type=click.Choice(["qwen3", "pubmedbert"]), default="qwen3", show_default=True, help="Embedding model name for assay context.")
@click.option("--num_seeds", type=int, default=5, show_default=True, help="Number of seeds to iterate over.")
@click.option("--split_method", type=click.Choice(["pair-level-random", "pmid-level-random"]), default="pair-level-random", show_default=True, help="Data split strategy: 'pair-level-random' splits rows randomly; 'pmid-level-random' groups all rows with the same PMID into the same partition.")
@click.option("--max_epochs", type=int, default=100, show_default=True, help="Training epochs per seed.")
@click.option("--device", type=str, default="0", show_default=True, help="CUDA device index to use.")
@click.option("--batch_size", type=int, default=16, show_default=True, help="Batch size.")
@click.option("--patience", type=int, default=20, show_default=True, help="Early stopping patience (epochs).")
@click.option("--use_wandb", is_flag=True, default=False, show_default=True, help="Enable Weights & Biases logging.")
@click.option("--wandb_project", type=str, default="affinity-assaycontext", show_default=True, help="Weights & Biases project name.")
@click.option("--wandb_entity", type=str, default='news012147stw', show_default=True, help="Weights & Biases entity.")
@click.option("--wandb_name", type=str, default="assay_context_only", show_default=True, help="Run name for Weights & Biases.")
def main(
    df_path: str,
    target_dir: str,
    assay_context_dim: int,
    hidden_dim: int,
    assay_embedding_model: str,
    num_seeds: int,
    split_method: str,
    max_epochs: int,
    device: str,
    batch_size: int,
    patience: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_name: Optional[str],
) -> None:
    """Ablation: train affinity prediction using only assay context embeddings."""

    monitor_metric = "val/loss"
    monitor_mode = "min"

    all_test_mse: list[float] = []
    all_test_rp: list[float] = []
    all_test_cindex: list[float] = []

    # Training phase
    for seed in range(num_seeds):
        Datacfg = DataConfig(
            df_path=df_path,
            target_dir=target_dir,
            seed=seed,
            batch_size=batch_size,
            assay_embedding_model=assay_embedding_model,
            split_method=split_method,
        )

        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=patience,
            min_delta=0.0,
            verbose=True,
        )

        ckpt_dir = os.path.join(Datacfg.target_dir, "checkpoints", f"assay_context_only_dim_{assay_context_dim}_split_{split_method}")

        if os.path.exists(os.path.join(ckpt_dir, f"seed_{Datacfg.seed}.ckpt")):
            print(f"Checkpoint for seed {Datacfg.seed} already exists at {os.path.join(ckpt_dir, f'seed_{Datacfg.seed}.ckpt')}, skipping training.")
            continue

        checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"seed_{Datacfg.seed}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        )

        model_module = LightningAffinityAssayContextOnlyModule(
            assay_context_dim=assay_context_dim,
            hidden_dim=hidden_dim,
        )

        data_module = AssayContextOnlyDataModule(Datacfg)

        if use_wandb:
            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                group="ablation_study",
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

    # Evaluation phase
    for seed in range(num_seeds):
        Datacfg = DataConfig(
            df_path=df_path,
            target_dir=target_dir,
            seed=seed,
            batch_size=batch_size,
            assay_embedding_model=assay_embedding_model,
            split_method=split_method,
        )

        ckpt_path = os.path.join(Datacfg.target_dir, "checkpoints",
                                 f"assay_context_only_dim_{assay_context_dim}_split_{split_method}",
                                 f"seed_{Datacfg.seed}.ckpt")

        model_module = LightningAffinityAssayContextOnlyModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
        model_module.eval()
        data_module = AssayContextOnlyDataModule(Datacfg)
        trainer = Trainer(devices=[int(device)], accelerator="gpu", log_every_n_steps=1)
        test_result = trainer.test(model_module, datamodule=data_module)

        all_test_mse.append(test_result[0].get('test/mse', None))
        all_test_rp.append(test_result[0].get('test/rp', None))
        all_test_cindex.append(test_result[0].get('test/cindex', None))

    print(f"mean test mse: {np.mean(all_test_mse):.2f}")
    print(f"std test mse: {np.std(all_test_mse):.2f}")

    print(f"mean test rp: {np.mean(all_test_rp):.2f}")
    print(f"std test rp: {np.std(all_test_rp):.2f}")

    print(f"mean test cindex: {np.mean(all_test_cindex):.2f}")
    print(f"std test cindex: {np.std(all_test_cindex):.2f}")


if __name__ == "__main__":
    main()
