
#%%
# lightning_affinity.py
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

from boltz.model.modules.affinity_assaycontext import AffinityAssayContextModule_concatenate_g_after_meanpooling, AffinityAssayContextModule_dual_film
from boltz.model.modules.affinity import AffinityModule
from boltz.data.module.training_AffinityModule_assaycontext import AffinityModuleDataModule, DataConfig
from omegaconf import OmegaConf, ListConfig

import numpy as np

# Optional metrics (install torchmetrics if you want these)
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


class LightningAffinityModule(LightningModule):
    """
    Lightning wrapper for AffinityModule.

    Expected batch format (customize in `extract_inputs_targets` if yours differs):
        batch = {
            "s_inputs": FloatTensor[B, N, token_s],
            "z":        FloatTensor[B, N, N, token_z],
            "x_pred":   FloatTensor[B, N, 3] or [B, mult, N, 3],
            "feats":    Dict[str, Tensor],  # contains keys used by AffinityModule.forward
            "assay_context": FloatTensor[B, assay_context_dim],
            "y":        FloatTensor[B] or [B, 1]   # regression target (e.g., pKd)
        }

    The wrapped model's forward returns a dict. We try a few common keys to grab the
    prediction (customize via `prediction_key` or override `select_prediction`).
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        assay_context: bool,
        assay_context_random_generation: bool,
        assay_context_dim: Optional[int],
        assay_embedding_injection_method: str,
        affinity_model_args: Optional[dict[str, Any]] = None,        
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        scheduler: Optional[str] = None,         # "cosine", "step", or None
        scheduler_args: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[nn.Module] = None,     # defaults to MSELoss
    ):
        super().__init__()
        self.save_hyperparameters()
        self.assay_context = assay_context
        self.assay_context_random_generation = assay_context_random_generation
        self.assay_context_dim = assay_context_dim
        self.assay_embedding_injection_method = assay_embedding_injection_method

        if assay_context and assay_embedding_injection_method == "concatenate_g_after_meanpooling":
            self.model = AffinityAssayContextModule_concatenate_g_after_meanpooling(token_s, token_z, assay_context_dim, **affinity_model_args)
        elif assay_context and assay_embedding_injection_method == "dual_film":
            self.model = AffinityAssayContextModule_dual_film(token_s, token_z, assay_context_dim, **affinity_model_args)
        else:
            self.model = AffinityModule(token_s, token_z, **affinity_model_args)
            
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args or {}
        self.loss_fn = loss_fn or nn.MSELoss()

    # --------- Utilities ---------
    def extract_inputs_targets(self, batch: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]) -> Tuple[Dict[str, Any], Tensor]:
        """
        Map your dataloader batch dict into (model_inputs, targets).
        Modify here if your keys differ.
        """
        s_inputs: Tensor = batch[0]
        z: Tensor = batch[1]
        x_pred: Tensor = batch[2]
        feats: Dict[str, Tensor] = batch[3]
        assay_context_embedding: Tensor = batch[4]
            
        # truncate assay-context based on assay_context_dim
        if (self.assay_context_dim is not None) and (assay_context_embedding.shape[1] > self.assay_context_dim) and self.assay_context:
            assay_context_embedding = assay_context_embedding[:, :self.assay_context_dim]
            
            if self.assay_context_random_generation:
                # replace with random vectors if random generation is enabled
                assay_context_embedding = torch.randn_like(assay_context_embedding)
            
            assay_context_embedding = F.normalize(assay_context_embedding, p=2, dim=-1)
            
        y: Tensor = batch[5]
        if y.ndim > 1:
            y = y.squeeze(-1)
            
        if self.assay_context:    
            model_inputs = dict(s_inputs=s_inputs, z=z, x_pred=x_pred, feats=feats, assay_context=assay_context_embedding)
        else:
            model_inputs = dict(s_inputs=s_inputs, z=z, x_pred=x_pred, feats=feats)
        return model_inputs, y


    def step(self, batch: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor, Tensor], stage: str) -> Tensor:
        model_inputs, y = self.extract_inputs_targets(batch)
        out = self.model(**model_inputs)         # AffinityModule.forward returns dict
        y_hat = out['affinity_pred_value']

        # If y_hat is [B,1], squeeze
        if y_hat.ndim > 1:
            y_hat = y_hat.squeeze(-1)

        loss = self.loss_fn(y_hat, y)

        # Log training metrics per-step; for val/test, compute full-dataset in epoch_end
        if stage == "train":
            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            mse = mean_squared_error(y_hat, y)
            self.log(f"{stage}/mse", mse, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            r = pearson_corrcoef(y_hat, y)
            self.log(f"{stage}/rp", r, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
            cindex = get_cindex(y_hat, y)
            self.log(f"{stage}/cindex", cindex, prog_bar=True, on_step=True, on_epoch=True, batch_size=y.shape[0])
        else:
            # Accumulate preds/targets for epoch-level metrics
            if stage == "val":
                if not hasattr(self, "_val_preds"):
                    self._val_preds = []
                    self._val_targets = []
                self._val_preds.append(y_hat.detach().float().cpu())
                self._val_targets.append(y.detach().float().cpu())
            elif stage == "test":
                # default test (no dataloader idx here), handled in test_step below where we know the idx/name
                pass

        return loss

    # --------- Lightning hooks ---------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self.step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        self.step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_targets = []

    def on_validation_epoch_end(self) -> None:
        if getattr(self, "_val_preds", None):
            preds = torch.cat(self._val_preds, dim=0)
            targets = torch.cat(self._val_targets, dim=0)
            # Compute full-dataset metrics
            mse = mean_squared_error(preds, targets)
            rp = pearson_corrcoef(preds, targets)
            cindex = get_cindex(preds, targets)
            # Log epoch-level metrics with canonical keys so trainer.validate() returns them
            self.log("val/loss", mse, prog_bar=True, on_epoch=True)
            self.log("val/mse", mse, prog_bar=True, on_epoch=True)
            self.log("val/rp", rp, prog_bar=True, on_epoch=True)
            self.log("val/cindex", cindex, prog_bar=True, on_epoch=True)

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        model_inputs, y = self.extract_inputs_targets(batch)
        out = self.model(**model_inputs)         # AffinityModule.forward returns dict
        y_hat = out['affinity_pred_value']

        # If y_hat is [B,1], squeeze
        if y_hat.ndim > 1:
            y_hat = y_hat.squeeze(-1)

        loss = self.loss_fn(y_hat, y)
        # Accumulate per-dataloader predictions/targets for full-dataset metrics
        if not hasattr(self, "_test_preds"):
            self._test_preds = []
            self._test_targets = []
        self._test_preds.append(y_hat.detach().float().cpu())
        self._test_targets.append(y.detach().float().cpu())

        return loss

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:
        model_inputs, _ = self.extract_inputs_targets(batch)
        out = self.model(**model_inputs)
        pred = out['affinity_pred_value']
        return {"pred": pred}

    def on_test_epoch_start(self) -> None:
        self._test_preds = []
        self._test_targets = []

    def on_test_epoch_end(self) -> None:
        # Compute and log full-dataset metrics per test dataloader
        
        preds_list = self._test_preds
        targets_list = self._test_targets
        if preds_list and targets_list:
            preds = torch.cat(preds_list, dim=0)
            targets = torch.cat(targets_list, dim=0)
            mse = mean_squared_error(preds, targets)
            rp = pearson_corrcoef(preds, targets)
            cindex = get_cindex(preds, targets)
            # Align keys with how the caller expects in results
            self.log(f"test/loss", mse, prog_bar=True, on_epoch=True)
            self.log(f"test/mse", mse, prog_bar=True, on_epoch=True)
            self.log(f"test/rp", rp, prog_bar=True, on_epoch=True)
            self.log(f"test/cindex", cindex, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        # Split out weight decay for LayerNorm/Embeddings if desired
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in n for nd in ["norm", 
                                      "rel_pose", 
                                      ".s_init", 
                                      ".z_init_", 
                                      "token_bonds", 
                                      "embed_atom_features",
                                      "dist_bin_pairwise_embed"]):
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
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }

@click.command()
@click.option("--hparams_yaml", type=click.Path(exists=True, dir_okay=False, path_type=str), default="/data/mwu11/boltz/BindingDB/itc/boltz_results_yaml_affinity_input/lightning_logs/version_1/hparams.yaml", show_default=True, help="Path to Lightning hparams.yaml used to construct model.")
@click.option("--df_path", type=click.Path(exists=True, dir_okay=False, path_type=str), default="/data/mwu11/LLM_affinity/BindingDB/non_null_spr_itc_structured_description_subset.csv", show_default=True, help="Path to CSV with columns for constructing AffinityModuleDataModule (e.g., s/z/x/feats/assay_context/y).")
@click.option("--target_dir", type=click.Path(file_okay=False, path_type=str), default="/data/mwu11/boltz/BindingDB/boltz_results_yaml_affinity_input", show_default=True, help="Directory with processed affinity inputs + where checkpoints are saved.")
@click.option("--assay_context", type=bool, is_flag=True, help="Whether to use assay context as input to model.")
@click.option("--assay_context_random_generation", type=bool, is_flag=True, help="Whether to use randomly generated assay context instead of real embeddings. Only used if --assay_context is set.")
@click.option("--assay_context_dim", type=int, default=128, show_default=True, help="Dimensionality of assay context vector input to model. Note: only set 768 for pubmedbert")
@click.option("--assay_embedding_model", type=click.Choice(["qwen3", "pubmedbert"]), default="qwen3", show_default=True, help="Embedding model name for embedding assay context. Only used if --assay_context is set.")
@click.option("--assay_embedding_injection_method", type=click.Choice(["dual_film", "concatenate_g_after_meanpooling"]), default="concatenate_g_after_meanpooling", show_default=True, help="Method for injecting assay context into model")
@click.option("--num_seeds", type=int, default=1, show_default=True, help="Number of seeds to iterate over.")
@click.option("--split_method", type=click.Choice(["pair-level-random", "pmid-level-random"]), default="pair-level-random", show_default=True, help="Data split strategy: 'pair-level-random' splits rows randomly; 'pmid-level-random' groups all rows with the same PMID into the same partition.")
@click.option("--max_epochs", type=int, default=50, show_default=True, help="Training epochs per seed.")
@click.option("--device", type=str, default="0", show_default=True, help="CUDA device index to use.")
@click.option("--batch_size", type=int, default=16, show_default=True, help="Batch size.")
@click.option("--patience", type=int, default=20, show_default=True, help="Early stopping patience (epochs).")
@click.option("--use_wandb", is_flag=True, default=False, show_default=True, help="Enable Weights & Biases logging.")
@click.option("--wandb_project", type=str, default="affinity-assaycontext", show_default=True, help="Weights & Biases project name.")
@click.option("--wandb_entity", type=str, default='news012147stw', show_default=True, help="Weights & Biases entity (team or username). Uses default if None.")
@click.option("--wandb_name", type=str, default=None, show_default=True, help="Run name for Weights & Biases.")
def main(
    hparams_yaml: str,
    df_path: str,
    target_dir: str,
    assay_context: bool,
    assay_context_random_generation: bool,
    assay_context_dim: int,
    assay_embedding_model: str,
    assay_embedding_injection_method: str,
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
    """Run training & evaluation for LightningAffinityModule with configurable data split method."""

    cfg = OmegaConf.load(hparams_yaml)

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

        ckpt_dir = os.path.join(Datacfg.target_dir, "checkpoints", f"assay_context_{assay_context}_assay_context_dim_{assay_context_dim if assay_context else 'NA'}{f'_{assay_embedding_injection_method}' if assay_context else ''}_split_{split_method}")
        
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
        
        model_module = LightningAffinityModule(
            token_s=cfg.token_s,
            token_z=cfg.token_z,
            assay_context=assay_context,
            assay_context_random_generation=assay_context_random_generation,
            assay_context_dim=assay_context_dim,
            assay_embedding_injection_method=assay_embedding_injection_method,
            affinity_model_args=cfg.affinity_model_args2,
        )

        data_module = AffinityModuleDataModule(Datacfg)

        if use_wandb:
            wandb_logger = WandbLogger(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                group=assay_embedding_injection_method,
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
                                 f"assay_context_{assay_context}_assay_context_dim_{assay_context_dim if assay_context else 'NA'}{f'_{assay_embedding_injection_method}' if assay_context else ''}_split_{split_method}",
                                 f"seed_{Datacfg.seed}.ckpt")
        
        model_module = LightningAffinityModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"), assay_context_random_generation=assay_context_random_generation, assay_embedding_injection_method=assay_embedding_injection_method)
        model_module.eval()
        data_module = AffinityModuleDataModule(Datacfg)
        trainer = Trainer(devices=[int(device)], accelerator="gpu", log_every_n_steps=1)
        test_result = trainer.test(model_module, datamodule=data_module)

        # Expect epoch-level metrics without dataloader_idx suffix as we log in on_test_epoch_end
        all_test_mse.append(test_result[0].get('test/mse', None))
        all_test_rp.append(test_result[0].get('test/rp', None))
        all_test_cindex.append(test_result[0].get('test/cindex', None))
        

    print(f"mean test mse: {np.mean(all_test_mse):.2f}")
    print(f"std test mse: {np.std(all_test_mse):.2f}")

    print(f"mean test rp: {np.mean(all_test_rp):.2f}")
    print(f"std test rp: {np.std(all_test_rp):.2f}")
    
    print(f"mean test cindex: {np.mean(all_test_cindex):.2f}")
    print(f"std test cindex: {np.std(all_test_cindex):.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()

