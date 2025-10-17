
# %%
from pyexpat import model
from typing import Any, Dict, Optional, Tuple
import os
import click

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from boltz.model.modules.affinity import AffinityModule
from boltz.data.module.finetuning_AffinityModule import AffinityModuleDataModule, DataConfig, AffinityModuleDataset
from omegaconf import OmegaConf, listconfig 

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

from kdbnet.dta_davis_complete import create_fine_tuning_different_mutation_same_drug_split, create_fine_tuning_different_mutation_different_drug_split, create_fine_tuning_same_mutation_different_drug_split
from torchmetrics.functional import mean_squared_error, pearson_corrcoef



def get_cindex(pred: Tensor, gt: Tensor) -> Tensor:
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = torch.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / torch.sum(gt_mask)
    return CI


# Utility to convert lists of tensor scalars to plain Python floats
def _as_float_list_if_tensors(x):
    """
    If x is a list containing torch.Tensor scalars, convert each to Python float.
    Otherwise, return x unchanged. This is useful before building a pandas DataFrame
    or writing to CSV to avoid 'tensor(…)' strings.
    """
    if isinstance(x, list) and any(isinstance(e, torch.Tensor) for e in x):
        out = []
        for e in x:
            if isinstance(e, torch.Tensor):
                # detach and move to cpu just in case
                if e.numel() == 1:
                    out.append(e.detach().cpu().item())
                else:
                    # if a non-scalar sneaks in, reduce to mean
                    out.append(e.detach().cpu().float().mean().item())
            else:
                # best-effort numeric cast; if it's non-numeric it will raise upstream
                try:
                    out.append(float(e))
                except Exception:
                    out.append(e)
        return out
    return x


def val(model, dataloader, trainer):
    model.eval()

    pred_out = trainer.predict(model, dataloaders=dataloader)   # list of dicts per batch
    preds   = torch.cat([o["preds"]   for o in pred_out], dim=0)
    targets = torch.cat([o["targets"] for o in pred_out], dim=0)
    
    
    coff = pearson_corrcoef(preds, targets)
    cindex = get_cindex(preds, targets)
    mse = mean_squared_error(preds, targets)
    rmse = mean_squared_error(preds, targets, squared=False)

    return mse, rmse, coff, cindex, preds, targets


def val_wt_groundtruth_baseline(wt_affinity, dataloader):
    label_list = []
    for data in dataloader:
        label = data[4]
        label_list.append(label.detach().cpu())

    label = torch.cat(label_list, axis=0)

    if len(label) != len(wt_affinity):
        wt_affinity = torch.ones_like(label) * wt_affinity
    else:
        assert len(label) == len(wt_affinity)

    mse = mean_squared_error(wt_affinity, label)
    rmse = mean_squared_error(wt_affinity, label, squared=False)
    coff = pearson_corrcoef(wt_affinity, label)
    cindex = get_cindex(wt_affinity, label)

    return mse, rmse, coff, cindex

def get_mutation_name(data_df, protein_name):
    return list(data_df[(data_df['protein'].str.contains(f"{protein_name}_[a-z][0-9]") | data_df['protein'].str.contains(f"{protein_name}_itd") | data_df['protein'].str.contains(f"{protein_name}_p"))]['protein'].unique())

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
            pass
        return loss

    # --------- Lightning hooks ---------
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self.step(batch, "train")

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Tensor]:
        model_inputs, y = self.extract_inputs_targets(batch)
        out = self.model(**model_inputs)
        y_hat = out['affinity_pred_value']
        
        if y_hat.ndim > 1:
            y_hat = y_hat.squeeze(-1)
            
        return {'preds': y_hat, 'targets': y}
            
        
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




# %%

@click.command()
@click.option("--split_method", type=click.Choice([
    'different_mutation_same_drug',
    'same_mutation_different_drug'
]), default='random', show_default=True, help='Data splitting strategy for train/test.')
@click.option("--df_path", type=click.Path(exists=True, dir_okay=False, path_type=str), default="/data/mwu11/DAVIS-complete/data/davis_complete/davis_complete_with_smiles.tsv", show_default=True, help="Path to DAVIS complete TSV with smiles.")
@click.option("--target_dir", type=click.Path(file_okay=False, path_type=str), default="/data/mwu11/boltz/DAVIS/boltz_results_affinity_input/boltz_results_yaml_affinity_input", show_default=True, help="Directory with processed affinity inputs + where checkpoints are saved.")
@click.option("--num_seeds", type=int, default=5, show_default=True, help="Number of seeds to iterate over.")
@click.option("--max_epochs", type=int, default=100, show_default=True, help="Training epochs per seed.")
@click.option("--device", type=str, default="0", show_default=True, help="CUDA device index to use.")
@click.option("--nontruncated_affinity", is_flag=True, default=True, show_default=True, help="Whether to use non-truncated affinity values.")
def main(
    split_method: str, 
    df_path: str, 
    target_dir: str, 
    num_seeds: int,
    max_epochs: int, 
    device: str, 
    nontruncated_affinity: bool
    ) -> None:  
    
    
    all_train_num = []
    all_test_num = []
    all_protein = []
    
    all_mutation = []
    all_drug_type = []
    all_drug_name = []
    

    all_mean_test_mse_wt_groundtruth_baseline = []
    all_std_test_mse_wt_groundtruth_baseline = []
    all_mean_test_mse_wt_prediction_baseline = []
    all_std_test_mse_wt_prediction_baseline = []

    all_mean_test_rp_wt_groundtruth_baseline = []
    all_std_test_rp_wt_groundtruth_baseline = []
    all_mean_test_rp_wt_prediction_baseline = []
    all_std_test_rp_wt_prediction_baseline = []

    all_mean_test_cindex_wt_groundtruth_baseline = []
    all_std_test_cindex_wt_groundtruth_baseline = []
    all_mean_test_cindex_wt_prediction_baseline = []
    all_std_test_cindex_wt_prediction_baseline = []

    all_mean_all_mse_wt_groundtruth_baseline = []
    all_std_all_mse_wt_groundtruth_baseline = []
    all_mean_all_mse_wt_prediction_baseline = []
    all_std_all_mse_wt_prediction_baseline = []

    all_mean_all_rp_wt_groundtruth_baseline = []
    all_std_all_rp_wt_groundtruth_baseline = [] 
    all_mean_all_rp_wt_prediction_baseline = []
    all_std_all_rp_wt_prediction_baseline = []
    
    all_mean_all_cindex_wt_groundtruth_baseline = []
    all_std_all_cindex_wt_groundtruth_baseline = []
    all_mean_all_cindex_wt_prediction_baseline = []
    all_std_all_cindex_wt_prediction_baseline = []

    all_mean_test_mse_original = []
    all_std_test_mse_original = []
    all_mean_all_mse_original = []
    all_std_all_mse_original = []

    all_mean_test_rp_original = []
    all_std_test_rp_original = []
    all_mean_all_rp_original = []
    all_std_all_rp_original = []

    all_mean_test_cindex_original = []
    all_std_test_cindex_original = []
    all_mean_all_cindex_original = []
    all_std_all_cindex_original = []

    all_mean_test_mse_finetuning = []
    all_std_test_mse_finetuning = []

    all_mean_test_rp_finetuning = []
    all_std_test_rp_finetuning = []

    all_mean_test_cindex_finetuning = []
    all_std_test_cindex_finetuning = []


    # protein = ['abl1', 'egfr', 'flt3', 'kit', 'met', 'pik3ca', 'ret']
    ## remove gcn2, as it does not have corresponding wild-type
    data_df = pd.read_csv(df_path, sep='\t')
    protein = ['abl1', 'braf', 'egfr', 'fgfr3', 'flt3', 'kit', 'lrrk2', 'met', 'pik3ca', 'ret'] 
    ligand = list(pd.read_csv('/data/mwu11/DAVIS-complete/data/davis_complete/davis_inhibitor_binding_mode.csv')['Compound'])


    if split_method == 'different_mutation_same_drug':
        combinations = list(product(protein, ligand))

    elif split_method == 'same_mutation_different_drug':
        combinations = []
        for protein_name in protein:
            for mutation_name in get_mutation_name(data_df, protein_name):
                combinations.append((protein_name, mutation_name))
    else:
        raise ValueError('split_method is not supported')
    
    
    ckpt_dir = os.path.join(target_dir, "checkpoints")

    for combination in tqdm(combinations):
        print(f'Now we are doing {combination}')
        protein = combination[0]
        
        if split_method == 'different_mutation_same_drug':
            drug_type = None
            mutation_name = None
            drug_name = combination[1]
            split_df, drug_name, train_num, test_num = create_fine_tuning_different_mutation_same_drug_split(
                protein=protein,
                drug=drug_name,
                df=data_df,
                nontruncated_affinity=nontruncated_affinity
            )
            if not split_df:
                continue
            job_name = f'fine_tuning_{split_method}_{protein}_{drug_name}'
            # wt_affinity = data_df[(data_df['protein'] == protein) & (data_df['drug_name'] == drug_name)]['y'].values[0]
            wt_all_affinity = split_df['wt_all']['y'].values
            wt_test_affinity = split_df['wt_test']['y'].values
            print(f'Now we are doing {job_name}')

        elif split_method == 'same_mutation_different_drug':
            drug_name = None
            mutation_name = combination[1]
            split_df, train_num, test_num = create_fine_tuning_same_mutation_different_drug_split(protein=protein, mutation=mutation_name, df=data_df, nontruncated_affinity=nontruncated_affinity)
            if not split_df:
                continue
            job_name = f'fine_tuning_{split_method}_{protein}_{mutation_name}'
            wt_all_affinity = split_df['wt_all']['y'].values
            wt_test_affinity = split_df['wt_test']['y'].values
            print(f'Now we are doing {job_name}')
        else:
            raise ValueError('split_method and combination_seed are not matched')
        
        
        Datacfg = DataConfig(
            df_path=df_path,
            target_dir=target_dir,
            split_method=split_method,
            split='train',
            protein=protein,
            mutation=mutation_name,
            drug=drug_name,
            nontruncated_affinity=nontruncated_affinity
        )

        for model_seed in range(num_seeds):
            if os.path.exists(os.path.join(ckpt_dir, split_method, job_name, f"seed_{model_seed}.ckpt")):
                print(f"Checkpoint for seed {model_seed} already exists at {os.path.join(ckpt_dir, split_method, job_name, f"seed_{model_seed}.ckpt")}, skipping training.")
                continue
            pretrain_ckpt_path = os.path.join(ckpt_dir, "wt_mutation", f"seed_{model_seed}.ckpt")
            
            model_module = LightningAffinityModule.load_from_checkpoint(pretrain_ckpt_path, map_location=torch.device("cpu"))
            model_module.train()
            data_module = AffinityModuleDataModule(Datacfg)
            
            trainer = Trainer(
                max_epochs=max_epochs,
                devices=[int(device)],
                accelerator="gpu",
                log_every_n_steps=1
            )
        
            trainer.fit(model_module, datamodule=data_module)
        
        all_test_mse_wt_groundtruth_baseline = []
        all_test_rp_wt_groundtruth_baseline = []
        all_test_cindex_wt_groundtruth_baseline = []

        all_all_mse_wt_groundtruth_baseline = []
        all_all_rp_wt_groundtruth_baseline = []
        all_all_cindex_wt_groundtruth_baseline = []

        all_test_mse_wt_prediction_baseline = []
        all_test_rp_wt_prediction_baseline = []
        all_test_cindex_wt_prediction_baseline = []

        all_all_mse_wt_prediction_baseline = []
        all_all_rp_wt_prediction_baseline = []
        all_all_cindex_wt_prediction_baseline = []

        all_test_mse_original = []
        all_test_rp_original = []
        all_test_cindex_original = []
        all_all_mse_original = []
        all_all_rp_original = []
        all_all_cindex_original = []

        all_test_mse_finetuning = []
        all_test_rp_finetuning = []
        all_test_cindex_finetuning = []


        for model_seed in range(num_seeds):

            all_set = AffinityModuleDataset(
                split='all', 
                df_path=Datacfg.df_path, 
                target_dir=Datacfg.target_dir, 
                split_method=Datacfg.split_method, 
                protein=Datacfg.protein,
                mutation=Datacfg.mutation,
                drug=Datacfg.drug,
                nontruncated_affinity=Datacfg.nontruncated_affinity
            )

            test_set = AffinityModuleDataset(
                split='test',
                df_path=Datacfg.df_path,
                target_dir=Datacfg.target_dir,
                split_method=Datacfg.split_method,
                protein=Datacfg.protein,
                mutation=Datacfg.mutation,
                drug=Datacfg.drug,
                nontruncated_affinity=Datacfg.nontruncated_affinity
            )

            wt_all_set = AffinityModuleDataset(
                split='wt_all',
                df_path=Datacfg.df_path,
                target_dir=Datacfg.target_dir, 
                split_method=Datacfg.split_method, 
                protein=Datacfg.protein,
                mutation=Datacfg.mutation,
                drug=Datacfg.drug,
                nontruncated_affinity=Datacfg.nontruncated_affinity
            )
            wt_test_set = AffinityModuleDataset(
                split='wt_test',
                df_path=Datacfg.df_path,
                target_dir=Datacfg.target_dir, 
                split_method=Datacfg.split_method, 
                protein=Datacfg.protein,
                mutation=Datacfg.mutation,
                drug=Datacfg.drug,
                nontruncated_affinity=Datacfg.nontruncated_affinity
            )

            all_loader = DataLoader(all_set, batch_size=len(all_set), shuffle=False)
            test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
            wt_all_loader = DataLoader(wt_all_set, batch_size=len(wt_all_set), shuffle=False)
            wt_test_loader = DataLoader(wt_test_set, batch_size=len(wt_test_set), shuffle=False)
            
            pretrain_ckpt_path = os.path.join(ckpt_dir, "wt_mutation", f"seed_{model_seed}.ckpt")
            pretrain_model_module = LightningAffinityModule.load_from_checkpoint(pretrain_ckpt_path, map_location=torch.device("cpu"))
              
            finetuning_ckpt_path = os.path.join(ckpt_dir, split_method, job_name, f"seed_{model_seed}.ckpt")
            finetuning_model_module = LightningAffinityModule.load_from_checkpoint(finetuning_ckpt_path, map_location=torch.device("cpu"))

            test_mse_wt_groundtruth_baseline, test_rmse_wt_groundtruth_baseline, test_rp_wt_groundtruth_baseline, test_cindex_wt_groundtruth_baseline = val_wt_groundtruth_baseline(wt_test_affinity, test_loader)
            _, _, _, _, wt_test_prediction, _ = val(pretrain_model_module, wt_test_loader, trainer)
            test_mse_wt_prediction_baseline, test_rmse_wt_prediction_baseline, test_rp_wt_prediction_baseline, test_cindex_wt_prediction_baseline = val_wt_groundtruth_baseline(wt_test_prediction, test_loader)

            all_mse_wt_groundtruth_baseline, all_rmse_wt_groundtruth_baseline, all_rp_wt_groundtruth_baseline, all_cindex_wt_groundtruth_baseline = val_wt_groundtruth_baseline(wt_all_affinity, all_loader)
            _, _, _, _, wt_all_prediction, _ = val(pretrain_model_module, wt_all_loader, trainer)
            all_mse_wt_prediction_baseline, all_rmse_wt_prediction_baseline, all_rp_wt_prediction_baseline, all_cindex_wt_prediction_baseline = val_wt_groundtruth_baseline(wt_all_prediction, all_loader)

            test_mse_original, test_rmse_original, test_rp_original, test_cindex_original, prediction_original, label = val(pretrain_model_module, test_loader, trainer)
            test_mse_finetuning, test_rmse_finetuning, test_rp_finetuning, test_cindex_finetuning, prediction_finetuning, label = val(finetuning_model_module, test_loader, trainer)
            all_mse_original, all_rmse_original, all_rp_original, all_cindex_original, prediction_original_all, label_all = val(finetuning_model_module, all_loader, trainer)


            if split_method == 'different_mutation_same_drug':
                print(f'label: {label}')
                print(f'label_all: {label_all}')
                print(f'gt_wt: {wt_test_affinity}')
                print(f'gt_wt_all: {wt_all_affinity}')
                print(f'prediction_wt: {wt_test_prediction}')
                print(f'prediction_original: {prediction_original}')
                print(f'prediction_finetuning: {prediction_finetuning}')
                print(f'prediction_original_all: {prediction_original_all}')

            elif split_method == 'same_mutation_different_drug':
                print(f'label: {label}')
                print(f'label_all: {label_all}')
                print(f'gt_wt: {wt_test_affinity}')
                print(f'gt_wt_all: {wt_all_affinity}')
                print(f'prediction_wt: {wt_test_prediction}')
                print(f'prediction_original: {prediction_original}')
                print(f'prediction_finetuning: {prediction_finetuning}')
                print(f'prediction_original_all: {prediction_original_all}')
            
            else:
                raise ValueError('split_method and combination_seed are not matched')

            msg = f"model_seed: {model_seed}, test_mse_original: {test_mse_original:.4f}, test_rmse_original: {test_rmse_original:.4f}, test_rp_original: {test_rp_original:.4f}, test_cindex_original: {test_cindex_original:.4f},\
                    test_mse_finetuning: {test_mse_finetuning:.4f}, test_rmse_finetuning: {test_rmse_finetuning:.4f}, test_rp_finetuning: {test_rp_finetuning:.4f}, test_cindex_finetuning: {test_cindex_finetuning:.4f}"
            print(msg)
            
            all_test_mse_wt_groundtruth_baseline.append(test_mse_wt_groundtruth_baseline)
            all_test_rp_wt_groundtruth_baseline.append(test_rp_wt_groundtruth_baseline)
            all_test_cindex_wt_groundtruth_baseline.append(test_cindex_wt_groundtruth_baseline)

            all_test_mse_wt_prediction_baseline.append(test_mse_wt_prediction_baseline)
            all_test_rp_wt_prediction_baseline.append(test_rp_wt_prediction_baseline)
            all_test_cindex_wt_prediction_baseline.append(test_cindex_wt_prediction_baseline)
            
            all_all_mse_wt_groundtruth_baseline.append(all_mse_wt_groundtruth_baseline)
            all_all_rp_wt_groundtruth_baseline.append(all_rp_wt_groundtruth_baseline)
            all_all_cindex_wt_groundtruth_baseline.append(all_cindex_wt_groundtruth_baseline)
            
            all_all_mse_wt_prediction_baseline.append(all_mse_wt_prediction_baseline)
            all_all_rp_wt_prediction_baseline.append(all_rp_wt_prediction_baseline)
            all_all_cindex_wt_prediction_baseline.append(all_cindex_wt_prediction_baseline)
                
            all_test_mse_original.append(test_mse_original)
            all_test_rp_original.append(test_rp_original)
            all_test_cindex_original.append(test_cindex_original)
            all_all_mse_original.append(all_mse_original)
            all_all_rp_original.append(all_rp_original)
            all_all_cindex_original.append(all_cindex_original)

            all_test_mse_finetuning.append(test_mse_finetuning)
            all_test_rp_finetuning.append(test_rp_finetuning)
            all_test_cindex_finetuning.append(test_cindex_finetuning)
        
        
        
        all_protein.append(protein)

        if split_method == 'different_mutation_same_drug':
            all_drug_type.append(drug_type)
            all_drug_name.append(drug_name)
            all_train_num.append(train_num)
            all_test_num.append(test_num)
        elif split_method == 'same_mutation_different_drug':
            all_mutation.append(mutation_name)
            all_train_num.append(train_num)
            all_test_num.append(test_num)
        else:
            raise ValueError('split_method is not supported')

        
        def _mean(lst):
            return torch.mean(torch.stack(lst)) if len(lst) > 0 else torch.tensor(float('nan'))
        def _std(lst):
            return torch.std(torch.stack(lst)) if len(lst) > 0 else torch.tensor(float('nan'))

        all_mean_test_mse_wt_groundtruth_baseline.append(_mean(all_test_mse_wt_groundtruth_baseline))
        all_std_test_mse_wt_groundtruth_baseline.append(_std(all_test_mse_wt_groundtruth_baseline))
        all_mean_test_mse_wt_prediction_baseline.append(_mean(all_test_mse_wt_prediction_baseline))
        all_std_test_mse_wt_prediction_baseline.append(_std(all_test_mse_wt_prediction_baseline))

        all_mean_test_rp_wt_groundtruth_baseline.append(_mean(all_test_rp_wt_groundtruth_baseline))
        all_std_test_rp_wt_groundtruth_baseline.append(_std(all_test_rp_wt_groundtruth_baseline))
        all_mean_test_rp_wt_prediction_baseline.append(_mean(all_test_rp_wt_prediction_baseline))
        all_std_test_rp_wt_prediction_baseline.append(_std(all_test_rp_wt_prediction_baseline))

        all_mean_test_cindex_wt_groundtruth_baseline.append(_mean(all_test_cindex_wt_groundtruth_baseline))
        all_std_test_cindex_wt_groundtruth_baseline.append(_std(all_test_cindex_wt_groundtruth_baseline))
        all_mean_test_cindex_wt_prediction_baseline.append(_mean(all_test_cindex_wt_prediction_baseline))
        all_std_test_cindex_wt_prediction_baseline.append(_std(all_test_cindex_wt_prediction_baseline))
        
        all_mean_all_mse_wt_groundtruth_baseline.append(_mean(all_all_mse_wt_groundtruth_baseline))
        all_std_all_mse_wt_groundtruth_baseline.append(_std(all_all_mse_wt_groundtruth_baseline))
        all_mean_all_mse_wt_prediction_baseline.append(_mean(all_all_mse_wt_prediction_baseline))
        all_std_all_mse_wt_prediction_baseline.append(_std(all_all_mse_wt_prediction_baseline))

        all_mean_all_rp_wt_groundtruth_baseline.append(_mean(all_all_rp_wt_groundtruth_baseline))
        all_std_all_rp_wt_groundtruth_baseline.append(_std(all_all_rp_wt_groundtruth_baseline))
        all_mean_all_rp_wt_prediction_baseline.append(_mean(all_all_rp_wt_prediction_baseline))
        all_std_all_rp_wt_prediction_baseline.append(_std(all_all_rp_wt_prediction_baseline))

        all_mean_all_cindex_wt_groundtruth_baseline.append(_mean(all_all_cindex_wt_groundtruth_baseline))
        all_std_all_cindex_wt_groundtruth_baseline.append(_std(all_all_cindex_wt_groundtruth_baseline))
        all_mean_all_cindex_wt_prediction_baseline.append(_mean(all_all_cindex_wt_prediction_baseline))
        all_std_all_cindex_wt_prediction_baseline.append(_std(all_all_cindex_wt_prediction_baseline))
        
        all_mean_test_mse_original.append(_mean(all_test_mse_original))
        all_std_test_mse_original.append(_std(all_test_mse_original))
        all_mean_test_rp_original.append(_mean(all_test_rp_original))
        all_std_test_rp_original.append(_std(all_test_rp_original))
        all_mean_test_cindex_original.append(_mean(all_test_cindex_original))
        all_std_test_cindex_original.append(_std(all_test_cindex_original))

        all_mean_all_mse_original.append(_mean(all_all_mse_original))
        all_std_all_mse_original.append(_std(all_all_mse_original))
        all_mean_all_rp_original.append(_mean(all_all_rp_original))
        all_std_all_rp_original.append(_std(all_all_rp_original))
        all_mean_all_cindex_original.append(_mean(all_all_cindex_original))
        all_std_all_cindex_original.append(_std(all_all_cindex_original))

        all_mean_test_mse_finetuning.append(_mean(all_test_mse_finetuning))
        all_std_test_mse_finetuning.append(_std(all_test_mse_finetuning))
        all_mean_test_rp_finetuning.append(_mean(all_test_rp_finetuning))
        all_std_test_rp_finetuning.append(_std(all_test_rp_finetuning))
        all_mean_test_cindex_finetuning.append(_mean(all_test_cindex_finetuning))
        all_std_test_cindex_finetuning.append(_std(all_test_cindex_finetuning))

    
    if split_method == 'different_mutation_same_drug':
        data = {
            'protein': all_protein,
            'drug_type': all_drug_type,
            'drug_name': all_drug_name,
            'train_num': all_train_num,
            'test_num': all_test_num,
            'mean_test_mse_wt_groundtruth_baseline': all_mean_test_mse_wt_groundtruth_baseline,
            'std_test_mse_wt_groundtruth_baseline': all_std_test_mse_wt_groundtruth_baseline,
            'mean_test_mse_wt_prediction_baseline': all_mean_test_mse_wt_prediction_baseline,
            'std_test_mse_wt_prediction_baseline': all_std_test_mse_wt_prediction_baseline,
            'mean_test_mse_original': all_mean_test_mse_original,
            'std_test_mse_original': all_std_test_mse_original,
            'mean_test_mse_finetuning': all_mean_test_mse_finetuning,
            'std_test_mse_finetuning': all_std_test_mse_finetuning,
            'mean_test_rp_original': all_mean_test_rp_original,
            'std_test_rp_original': all_std_test_rp_original,
            'mean_test_rp_finetuning': all_mean_test_rp_finetuning,
            'std_test_rp_finetuning': all_std_test_rp_finetuning,
            'mean_test_cindex_original': all_mean_test_cindex_original,
            'std_test_cindex_original': all_std_test_cindex_original,
            'mean_test_cindex_finetuning': all_mean_test_cindex_finetuning,
            'std_test_cindex_finetuning': all_std_test_cindex_finetuning,
            'mean_all_mse_wt_groundtruth_baseline': all_mean_all_mse_wt_groundtruth_baseline,
            'std_all_mse_wt_groundtruth_baseline': all_std_all_mse_wt_groundtruth_baseline,
            'mean_all_mse_wt_prediction_baseline': all_mean_all_mse_wt_prediction_baseline,
            'std_all_mse_wt_prediction_baseline': all_std_all_mse_wt_prediction_baseline,
            'mean_all_mse_original': all_mean_all_mse_original,
            'std_all_mse_original': all_std_all_mse_original,
            'mean_all_rp_original': all_mean_all_rp_original,
            'std_all_rp_original': all_std_all_rp_original,
            'mean_all_cindex_original': all_mean_all_cindex_original,
            'std_all_cindex_original': all_std_all_cindex_original,
        }
        # convert tensor lists to float lists for safe CSV writing
        for k, v in list(data.items()):
            data[k] = _as_float_list_if_tensors(v)
        df = pd.DataFrame(data)
    elif split_method == 'same_mutation_different_drug':
        data = {
            'protein': all_protein,
            'mutation': all_mutation,
            'train_num': all_train_num,
            'test_num': all_test_num,
            'mean_test_mse_wt_groundtruth_baseline': all_mean_test_mse_wt_groundtruth_baseline,
            'std_test_mse_wt_groundtruth_baseline': all_std_test_mse_wt_groundtruth_baseline,
            'mean_test_mse_wt_prediction_baseline': all_mean_test_mse_wt_prediction_baseline,
            'std_test_mse_wt_prediction_baseline': all_std_test_mse_wt_prediction_baseline,
            'mean_test_rp_wt_groundtruth_baseline': all_mean_test_rp_wt_groundtruth_baseline,
            'std_test_rp_wt_groundtruth_baseline': all_std_test_rp_wt_groundtruth_baseline,
            'mean_test_rp_wt_prediction_baseline': all_mean_test_rp_wt_prediction_baseline,
            'std_test_rp_wt_prediction_baseline': all_std_test_rp_wt_prediction_baseline,
            'mean_test_cindex_wt_groundtruth_baseline': all_mean_test_cindex_wt_groundtruth_baseline,
            'std_test_cindex_wt_groundtruth_baseline': all_std_test_cindex_wt_groundtruth_baseline,
            'mean_test_cindex_wt_prediction_baseline': all_mean_test_cindex_wt_prediction_baseline,
            'std_test_cindex_wt_prediction_baseline': all_std_test_cindex_wt_prediction_baseline,
            'mean_test_mse_original': all_mean_test_mse_original,
            'std_test_mse_original': all_std_test_mse_original,
            'mean_test_mse_finetuning': all_mean_test_mse_finetuning,
            'std_test_mse_finetuning': all_std_test_mse_finetuning,
            'mean_test_rp_original': all_mean_test_rp_original,
            'std_test_rp_original': all_std_test_rp_original,
            'mean_test_rp_finetuning': all_mean_test_rp_finetuning,
            'std_test_rp_finetuning': all_std_test_rp_finetuning,
            'mean_test_cindex_original': all_mean_test_cindex_original,
            'std_test_cindex_original': all_std_test_cindex_original,
            'mean_test_cindex_finetuning': all_mean_test_cindex_finetuning,
            'std_test_cindex_finetuning': all_std_test_cindex_finetuning,
            'mean_all_mse_wt_groundtruth_baseline': all_mean_all_mse_wt_groundtruth_baseline,
            'std_all_mse_wt_groundtruth_baseline': all_std_all_mse_wt_groundtruth_baseline,
            'mean_all_mse_wt_prediction_baseline': all_mean_all_mse_wt_prediction_baseline,
            'std_all_mse_wt_prediction_baseline': all_std_all_mse_wt_prediction_baseline,
            'mean_all_rp_wt_groundtruth_baseline': all_mean_all_rp_wt_groundtruth_baseline,
            'std_all_rp_wt_groundtruth_baseline': all_std_all_rp_wt_groundtruth_baseline,
            'mean_all_rp_wt_prediction_baseline': all_mean_all_rp_wt_prediction_baseline,
            'std_all_rp_wt_prediction_baseline': all_std_all_rp_wt_prediction_baseline,
            'mean_all_cindex_wt_groundtruth_baseline': all_mean_all_cindex_wt_groundtruth_baseline,
            'std_all_cindex_wt_groundtruth_baseline': all_std_all_cindex_wt_groundtruth_baseline,
            'mean_all_cindex_wt_prediction_baseline': all_mean_all_cindex_wt_prediction_baseline,
            'std_all_cindex_wt_prediction_baseline': all_std_all_cindex_wt_prediction_baseline,
            'mean_all_mse_original': all_mean_all_mse_original,
            'std_all_mse_original': all_std_all_mse_original,
            'mean_all_rp_original': all_mean_all_rp_original,
            'std_all_rp_original': all_std_all_rp_original,
            'mean_all_cindex_original': all_mean_all_cindex_original,
            'std_all_cindex_original': all_std_all_cindex_original,
        }
        for k, v in list(data.items()):
            data[k] = _as_float_list_if_tensors(v)
        df = pd.DataFrame(data)
    else:
        raise ValueError('split_method is not supported')

    df.to_csv(f'result_finetuning_{split_method}_epoch{max_epochs}_lr{model_module.lr}.csv', index=False)
    
if __name__ == "__main__":
    main()