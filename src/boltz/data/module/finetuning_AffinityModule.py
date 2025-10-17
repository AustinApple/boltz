#%%
from dataclasses import dataclass
from pathlib import Path
from re import I
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import pad

from boltz.data.crop.cropper import Cropper
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.symmetry import get_symmetries
from boltz.data.filter.dynamic.filter import DynamicFilter
from boltz.data.pad import pad_to_max
from boltz.data.sample.sampler import Sample, Sampler
from boltz.data.tokenize.tokenizer import Tokenizer
from boltz.data.types import MSA, Connection, Input, Manifest, Record, Structure

import pandas as pd
from kdbnet.dta_davis_complete import create_fine_tuning_different_mutation_same_drug_split, create_fine_tuning_same_mutation_different_drug_split


def pad_to_max(data: list[Tensor], value: float = 0) -> tuple[Tensor, Tensor]:
    """Pad the data in all dimensions to the maximum found.

    Parameters
    ----------
    data : list[Tensor]
        list of tensors to pad.
    value : float
        The value to use for padding.

    Returns
    -------
    Tensor
        The padded tensor.
    Tensor
        The padding mask.

    """
    if isinstance(data[0], str):
        return data, 0

    # Check if all have the same shape
    if all(d.shape == data[0].shape for d in data):
        return torch.cat(data, dim=0), 0

    # Get the maximum in each dimension
    num_dims = len(data[0].shape)
    max_dims = [max(d.shape[i] for d in data) for i in range(num_dims)]

    # Get the padding lengths
    pad_lengths = []
    for d in data:
        dims = []
        for i in range(num_dims):
            dims.append(0)
            dims.append(max_dims[num_dims - i - 1] - d.shape[num_dims - i - 1])
        pad_lengths.append(dims)

    # Pad the data
    padding = [
        pad(torch.ones_like(d), pad_len, value=0)
        for d, pad_len in zip(data, pad_lengths)
    ]
    data = [pad(d, pad_len, value=value) for d, pad_len in zip(data, pad_lengths)]

    # concatenate the data
    padding = torch.cat(padding, dim=0)
    data = torch.cat(data, dim=0)

    return data, padding




@dataclass
class DataConfig:
    """Data configuration."""

    df_path: Path
    target_dir: Path
    split_method: str
    split: str
    protein: str
    mutation: str
    drug: str
    nontruncated_affinity: bool



def load_input(protein: str, ligand: str, target_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load the given input data.

    Parameters
    ----------
    protein : str
        The protein identifier.
    ligand : str
        The ligand identifier.
    target_dir : Path
        The path to the data directory.

    Returns

    -------
    Input
        The loaded input.
    """
    # Load the precomputed inputs
    inputs = np.load(target_dir / "processed" / "affinity_module_inputs_shrink" / f"{protein}_{ligand}" / f"affinity_input_{protein}_{ligand}.npz", allow_pickle=True)
    feats = {k: v for k, v in inputs.items() if k not in ['s_inputs', 'z_affinity', 'coords_affinity']}

    return inputs["s_inputs"], inputs["z_affinity"], inputs["coords_affinity"], feats
    


def collate(data: list[tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]]) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]:
    """Collate the data.

    Parameters
    ----------
    data: list[tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]]
        The data to collate.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]
        The collated data.

    """
    s_inputs = [d[0] for d in data]
    if not all(s.shape == s_inputs[0].shape for s in s_inputs):
        s_inputs, _ = pad_to_max(s_inputs, 0)
    else:
        s_inputs = torch.cat(s_inputs, dim=0)

    z_affinity = [d[1] for d in data]
    if not all(z.shape == z_affinity[0].shape for z in z_affinity):
        z_affinity, _ = pad_to_max(z_affinity, 0)
    else:
        z_affinity = torch.cat(z_affinity, dim=0)

    coords_affinity = [d[2] for d in data]
    if not all(c.shape == coords_affinity[0].shape for c in coords_affinity):
        coords_affinity, _ = pad_to_max(coords_affinity, 0)
    else:
        coords_affinity = torch.cat(coords_affinity, dim=0)

    # get the feature keys
    keys = data[0][3].keys()

    # collate the features
    feats = {}
    for key in keys:
        values = [d[3][key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
        ]:
            if not all(v.shape == values[0].shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.cat(values, dim=0)

        feats[key] = values

    y = torch.stack([d[4] for d in data], dim=0)

    return s_inputs, z_affinity, coords_affinity, feats, y



class AffinityModuleDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df_path: Path, 
                 target_dir: Path, 
                 split_method: str, 
                 split: str,
                 protein: str,
                 mutation: str,
                 drug: str,
                 nontruncated_affinity: bool) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.df = pd.read_csv(df_path, sep='\t')
        self.target_dir = target_dir
        self.split = split
        self.split_method = split_method
        self.protein = protein
        self.mutation = mutation
        self.drug = drug
        self.nontruncated_affinity = nontruncated_affinity
        

        if self.split_method == 'different_mutation_same_drug':
            self.split_df, _, _, _ = create_fine_tuning_different_mutation_same_drug_split(protein=self.protein, 
                                                                                      drug=self.drug, 
                                                                                      df=self.data_df, 
                                                                                      nontruncated_affinity=self.nontruncated_affinity)
        elif self.split_method == 'same_mutation_different_drug' and not self.seed:
            self.split_df, _, _ = create_fine_tuning_same_mutation_different_drug_split(protein=self.protein, 
                                                                                   mutation=self.mutation, 
                                                                                   df=self.data_df, 
                                                                                   nontruncated_affinity=self.nontruncated_affinity)
        else:
            raise ValueError("Unknown split method: {}".format(self.split_method))    
        
        if self.split == 'all':
            self.split_df = self.split_df['all']
        elif self.split == 'train':
            self.split_df = self.split_df['train']
        elif self.split == 'test':
            self.split_df = self.split_df['test']
        elif self.split == 'wt_all':
            self.split_df = self.split_df['wt_all']
        elif self.split == 'wt_test':
            self.split_df = self.split_df['wt_test']
        else:
            raise ValueError("Unknown split: {}".format(self.split))


    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """

        try:
            idx = np.random.choice(len(self.split_df))
            protein = self.split_df.iloc[idx]['protein']
            ligand = self.split_df.iloc[idx]['drug']
            y = self.split_df.iloc[idx]['y']
            s_inputs, z_affinity, coords_affinity, feats = load_input(protein, ligand, self.target_dir)

            s_inputs = torch.tensor(s_inputs).float()
            z_affinity = torch.tensor(z_affinity).float()
            coords_affinity = torch.tensor(coords_affinity).float()
            y = torch.tensor(y).float().unsqueeze(0)
            feats = {k: torch.tensor(v).float() if isinstance(v, np.ndarray) else v for k, v in feats.items()}

        except Exception as e:
            #print(f"Error loading data for index {idx}: {e}")
            return self.__getitem__(idx)

        return s_inputs, z_affinity, coords_affinity, feats, y

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.split_df)
    

class AffinityModuleDataModule(pl.LightningDataModule):
    """DataModule for AffinityModule."""
    def __init__(self, cfg: DataConfig) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        return

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        train_set = AffinityModuleDataset(
            df_path=Path(self.cfg.df_path),
            target_dir=Path(self.cfg.target_dir),
            split_method=self.cfg.split_method,
            split='train',
            protein=self.cfg.protein,
            mutation=self.cfg.mutation,
            drug=self.cfg.drug,
            nontruncated_affinity=self.cfg.nontruncated_affinity
        )
        return DataLoader(
            train_set,
            batch_size=len(train_set),
            num_workers=5,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate,
        )

def dm_sanity_check(dm: pl.LightningDataModule, n_batches: int = 1):

    loaders = {
        "train": dm.train_dataloader(),
        "val":   dm.val_dataloader(),
        "test":  dm.test_dataloader(),
    }

    for name, loader in loaders.items():
        print(f"\n[{name.upper()}] checking {n_batches} batch(es)…")
        it = iter(loader)
        for b in range(n_batches):
            batch = next(it)
            s_inputs, z_affinity, coords_affinity, feats, y = batch

            # Basic type/shape checks
            assert isinstance(s_inputs, torch.Tensor), "s_inputs must be a Tensor"
            assert isinstance(z_affinity, torch.Tensor), "z_affinity must be a Tensor"
            assert isinstance(coords_affinity, torch.Tensor), "coords_affinity must be a Tensor"
            assert isinstance(feats, dict), "feats must be a dict"
            assert isinstance(y, torch.Tensor), "y must be a Tensor"

            bs = s_inputs.shape[0]
            assert z_affinity.shape[0] == bs, "Batch dim mismatch: z_affinity"
            assert coords_affinity.shape[0] == bs, "Batch dim mismatch: coords_affinity"

            # Check that every feat is batched consistently (or intentionally left as list)
            for k, v in feats.items():
                if isinstance(v, torch.Tensor):
                    assert v.shape[0] == bs, f"Feature '{k}' batch dim mismatch"
                else:
                    # Some keys are intentionally NOT stacked (lists, etc.)
                    # Verify length == batch size when list-like
                    try:
                        if hasattr(v, "__len__"):
                            assert len(v) == bs, f"Feature '{k}' length != batch size"
                    except TypeError:
                        pass

            print(f"  batch {b+1}:")
            print(f"    s_inputs: {tuple(s_inputs.shape)}  {s_inputs.dtype}")
            print(f"    z_affinity: {tuple(z_affinity.shape)}  {z_affinity.dtype}")
            print(f"    coords_affinity: {tuple(coords_affinity.shape)}  {coords_affinity.dtype}")
            print(f"    feats keys: {list(feats.keys())[:8]}{' …' if len(feats)>8 else ''}")
            print(f"    (e.g. 'token_pad_mask': {tuple(feats['token_pad_mask'].shape)}  {feats['token_pad_mask'].dtype})")
            print(f"    (e.g. 'mol_type': {tuple(feats['mol_type'].shape)}  {feats['mol_type'].dtype})")
            print(f"    (e.g. 'y': {tuple(y.shape)}  {y.dtype})")

    print("\nSanity check passed")

#%%
if __name__ == "__main__":
    cfg = DataConfig(
        df_path='/data/mwu11/DAVIS-complete/data/davis_complete/davis_complete_with_smiles.tsv',
        split_method='random',
        target_dir='/data/mwu11/boltz/DAVIS/boltz_results_affinity_input/boltz_results_yaml_affinity',
        mmseqs_seq_clus_df_path='/data/mwu11/DAVIS-complete/data/davis_complete/davis_complete_id50_cluster.tsv',
        seed=42,
        batch_size=2
    )

    data_module = AffinityModuleDataModule(cfg)
    data_module.setup()

    dm_sanity_check(data_module, n_batches=1)
# %%
