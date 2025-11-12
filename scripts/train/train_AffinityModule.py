
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

from boltz.model.modules.affinity import AffinityModule
from boltz.data.module.training_AffinityModule import AffinityModuleDataModule, DataConfig
from omegaconf import OmegaConf, listconfig 

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
            "y":        FloatTensor[B] or [B, 1]   # regression target (e.g., pKd)
        }

    The wrapped model's forward returns a dict. We try a few common keys to grab the
    prediction (customize via `prediction_key` or override `select_prediction`).
    """

    def __init__(
        self,
        token_s: int,
        token_z: int,
        affinity_model_args: Optional[dict[str, Any]] = None,        
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        scheduler: Optional[str] = None,         # "cosine", "step", or None
        scheduler_args: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[nn.Module] = None,     # defaults to MSELoss
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AffinityModule(token_s, token_z, **affinity_model_args)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args or {}
        self.loss_fn = loss_fn or nn.MSELoss()

    # --------- Utilities ---------
    def extract_inputs_targets(self, batch: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]) -> Tuple[Dict[str, Any], Tensor]:
        """
        Map your dataloader batch dict into (model_inputs, targets).
        Modify here if your keys differ.
        """
        s_inputs: Tensor = batch[0]
        z: Tensor = batch[1]
        x_pred: Tensor = batch[2]
        feats: Dict[str, Tensor] = batch[3]
        y: Tensor = batch[4]
        if y.ndim > 1:
            y = y.squeeze(-1)
        model_inputs = dict(s_inputs=s_inputs, z=z, x_pred=x_pred, feats=feats)
        return model_inputs, y


    def step(self, batch: tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor], stage: str) -> Tensor:
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
        _dict = {0: "test", 1: "test_wt", 2: "test_mutation"}
        model_inputs, y = self.extract_inputs_targets(batch)
        out = self.model(**model_inputs)         # AffinityModule.forward returns dict
        y_hat = out['affinity_pred_value']

        # If y_hat is [B,1], squeeze
        if y_hat.ndim > 1:
            y_hat = y_hat.squeeze(-1)

        loss = self.loss_fn(y_hat, y)
        # Accumulate per-dataloader predictions/targets for full-dataset metrics
        name = _dict[dataloader_idx]
        if not hasattr(self, "_test_preds"):
            self._test_preds = {"test": [], "test_wt": [], "test_mutation": []}
            self._test_targets = {"test": [], "test_wt": [], "test_mutation": []}
        self._test_preds[name].append(y_hat.detach().float().cpu())
        self._test_targets[name].append(y.detach().float().cpu())

        return loss

    def predict_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, Tensor]:
        model_inputs, _ = self.extract_inputs_targets(batch)
        out = self.model(**model_inputs)
        pred = out['affinity_pred_value']
        return {"pred": pred}

    def on_test_epoch_start(self) -> None:
        self._test_preds = {"test": [], "test_wt": [], "test_mutation": []}
        self._test_targets = {"test": [], "test_wt": [], "test_mutation": []}

    def on_test_epoch_end(self) -> None:
        # Compute and log full-dataset metrics per test dataloader
        for name in ["test", "test_wt", "test_mutation"]:
            preds_list = self._test_preds.get(name, [])
            targets_list = self._test_targets.get(name, [])
            if preds_list and targets_list:
                preds = torch.cat(preds_list, dim=0)
                targets = torch.cat(targets_list, dim=0)
                mse = mean_squared_error(preds, targets)
                rp = pearson_corrcoef(preds, targets)
                cindex = get_cindex(preds, targets)
                # Align keys with how the caller expects in results
                self.log(f"{name}/loss", mse, prog_bar=True, on_epoch=True)
                self.log(f"{name}/mse", mse, prog_bar=True, on_epoch=True)
                self.log(f"{name}/rp", rp, prog_bar=True, on_epoch=True)
                self.log(f"{name}/cindex", cindex, prog_bar=True, on_epoch=True)

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
@click.option("--split_method", type=click.Choice([
    'random',
    'drug_name',
    'drug_structure',
    'protein_modification',
    'protein_name',
    'protein_modification_drug_name',
    'protein_seqid_drug_structure',
    'protein_seqid',
    'wt_mutation'
]), default='random', show_default=True, help='Data splitting strategy for train/val/test.')
@click.option("--hparams_yaml", type=click.Path(exists=True, dir_okay=False, path_type=str), default="/data/mwu11/boltz/DAVIS/boltz_results_affinity_input/boltz_results_yaml_affinity_input/lightning_logs/version_1/hparams.yaml", show_default=True, help="Path to Lightning hparams.yaml used to construct model.")
@click.option("--df_path", type=click.Path(exists=True, dir_okay=False, path_type=str), default="/data/mwu11/DAVIS-complete/data/davis_complete/davis_complete_with_smiles.tsv", show_default=True, help="Path to DAVIS complete TSV with smiles.")
@click.option("--mmseqs_cluster_path", type=click.Path(exists=True, dir_okay=False, path_type=str), default="/data/mwu11/DAVIS-complete/data/davis_complete/davis_complete_id50_cluster.tsv", show_default=True, help="Path to mmseqs seq cluster file.")
@click.option("--target_dir", type=click.Path(file_okay=False, path_type=str), default="/data/mwu11/boltz/DAVIS/boltz_results_affinity_input/boltz_results_yaml_affinity_input", show_default=True, help="Directory with processed affinity inputs + where checkpoints are saved.")
@click.option("--num_seeds", type=int, default=5, show_default=True, help="Number of seeds to iterate over.")
@click.option("--max_epochs", type=int, default=100, show_default=True, help="Training epochs per seed.")
@click.option("--device", type=str, default="0", show_default=True, help="CUDA device index to use.")
@click.option("--batch_size", type=int, default=64, show_default=True, help="Batch size.")
@click.option("--patience", type=int, default=5, show_default=True, help="Early stopping patience (epochs).")
def main(
    split_method: str, 
    hparams_yaml: str, 
    df_path: str, 
    mmseqs_cluster_path: str, 
    target_dir: str, 
    num_seeds: int,
    max_epochs: int, 
    device: str, 
    batch_size: int, 
    patience: int
    ) -> None:
    """Run training & evaluation for LightningAffinityModule with configurable data split method."""

    cfg = OmegaConf.load(hparams_yaml)

    monitor_metric = "val/loss"
    monitor_mode = "min"

    all_test_mse: list[float] = []
    all_test_rp: list[float] = []
    all_test_cindex: list[float] = []
    all_test_wt_mse: list[float] = []
    all_test_wt_rp: list[float] = []
    all_test_wt_cindex: list[float] = []
    all_test_mutation_mse: list[float] = []
    all_test_mutation_rp: list[float] = []
    all_test_mutation_cindex: list[float] = []

    # Training phase
    for seed in range(num_seeds):
        Datacfg = DataConfig(
            df_path=df_path,
            split_method=split_method,
            target_dir=target_dir,
            mmseqs_seq_clus_df_path=mmseqs_cluster_path,
            seed=seed,
            batch_size=batch_size,
        )

        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=patience,
            min_delta=0.0,
            verbose=True,
        )

        ckpt_dir = os.path.join(Datacfg.target_dir, "checkpoints", Datacfg.split_method)
        
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
            affinity_model_args=cfg.affinity_model_args2,
        )
        data_module = AffinityModuleDataModule(Datacfg)
        
        trainer = Trainer(
            max_epochs=max_epochs,
            devices=[int(device)],
            accelerator="gpu",
            log_every_n_steps=1,
            callbacks=[early_stopping, checkpoint_cb]
        )
        
        
        trainer.fit(model_module, datamodule=data_module)

    # Evaluation phase
    for seed in range(num_seeds):
        Datacfg = DataConfig(
            df_path=df_path,
            split_method=split_method,
            target_dir=target_dir,
            mmseqs_seq_clus_df_path=mmseqs_cluster_path,
            seed=seed,
            batch_size=batch_size,
        )

        ckpt_path = os.path.join(
            Datacfg.target_dir,
            "checkpoints",
            Datacfg.split_method,
            f"seed_{Datacfg.seed}.ckpt",
        )
        model_module = LightningAffinityModule.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))
        model_module.eval()
        data_module = AffinityModuleDataModule(Datacfg)
        trainer = Trainer(devices=[int(device)], accelerator="gpu", log_every_n_steps=1)
        test_result = trainer.test(model_module, datamodule=data_module)

        # Expect epoch-level metrics without dataloader_idx suffix as we log in on_test_epoch_end
        all_test_mse.append(test_result[0].get('test/mse', None))
        all_test_rp.append(test_result[0].get('test/rp', None))
        all_test_cindex.append(test_result[0].get('test/cindex', None))
        
        if split_method != 'wt_mutation':
            all_test_wt_mse.append(test_result[1].get('test_wt/mse', None))
            all_test_wt_rp.append(test_result[1].get('test_wt/rp', None))
            all_test_wt_cindex.append(test_result[1].get('test_wt/cindex', None))

            all_test_mutation_mse.append(test_result[2].get('test_mutation/mse', None))
            all_test_mutation_rp.append(test_result[2].get('test_mutation/rp', None))
            all_test_mutation_cindex.append(test_result[2].get('test_mutation/cindex', None))

    print(f"mean test mse: {np.mean(all_test_mse):.2f}")
    print(f"std test mse: {np.std(all_test_mse):.2f}")
    
    if split_method != 'wt_mutation':
        print(f"mean test_wt mse: {np.mean(all_test_wt_mse):.2f}")
        print(f"std test_wt mse: {np.std(all_test_wt_mse):.2f}")
        print(f"mean test_mutation mse: {np.mean(all_test_mutation_mse):.2f}")
        print(f"std test_mutation mse: {np.std(all_test_mutation_mse):.2f}")

    print(f"mean test rp: {np.mean(all_test_rp):.2f}")
    print(f"std test rp: {np.std(all_test_rp):.2f}")
    
    if split_method != 'wt_mutation':
        print(f"mean test_wt rp: {np.mean(all_test_wt_rp):.2f}")
        print(f"std test_wt rp: {np.std(all_test_wt_rp):.2f}")
        print(f"mean test_mutation rp: {np.mean(all_test_mutation_rp):.2f}")
        print(f"std test_mutation rp: {np.std(all_test_mutation_rp):.2f}")

    print(f"mean test cindex: {np.mean(all_test_cindex):.2f}")
    print(f"std test cindex: {np.std(all_test_cindex):.2f}")
    
    if split_method != 'wt_mutation':
        print(f"mean test_wt cindex: {np.mean(all_test_wt_cindex):.2f}")
        print(f"std test_wt cindex: {np.std(all_test_wt_cindex):.2f}")
        print(f"mean test_mutation cindex: {np.mean(all_test_mutation_cindex):.2f}")
        print(f"std test_mutation cindex: {np.std(all_test_mutation_cindex):.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()

# %%
