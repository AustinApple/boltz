#%%
from dataclasses import dataclass
from pathlib import Path

from typing import Dict, Optional, List, Tuple, Set, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.functional import pad
import pandas as pd


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

    df_path: str
    target_dir: str
    seed: int
    batch_size: int
    assay_embedding_model: str
    split_method: str = "pair-level-random"  # "pair-level-random" or "pmid-level-random"



def load_input(reactant_set_id: str, target_dir: Path, assay_embedding_model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load the given input data.

    Parameters
    ----------
    reactant_set_id : str
        The reactant set identifier.
    target_dir : Path
        The path to the data directory.
    assay_embedding_model : str
        The name of the assay embedding model.

    Returns

    -------
    Input
        The loaded input.
    """
    # Load the precomputed inputs
    inputs = np.load(target_dir / "processed" / "affinity_module_inputs" / f"{reactant_set_id}" / f"affinity_input_{reactant_set_id}.npz", allow_pickle=True)
    assay_context = torch.load(target_dir / "processed" / "assay_context_embedding" / assay_embedding_model / f"{reactant_set_id}.pt")
    feats = {k: v for k, v in inputs.items() if k not in ['s_inputs', 'z_affinity', 'coords_affinity']}

    return inputs["s_inputs"], inputs["z_affinity"], inputs["coords_affinity"], feats, assay_context
    


def collate(data: list[tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]]) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data: list[tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]]
        The data to collate.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]
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
    
    assay_contexts = [d[4] for d in data]
    assay_contexts = torch.stack(assay_contexts, dim=0)
    
    y = torch.stack([d[5] for d in data], dim=0)

    return s_inputs, z_affinity, coords_affinity, feats, assay_contexts, y


class AffinityModuleDataset(torch.utils.data.Dataset):
    def __init__(self, df_path: Path, split: str, seed: int, target_dir: Path, assay_embedding_model: str, split_method: str = "pair-level-random") -> None:
        """Initialize the training dataset."""
        super().__init__()

        sep = '\t' if str(df_path).endswith('.tsv') else ','
        self.df = pd.read_csv(df_path, sep=sep)
        self.target_dir = Path(target_dir)
        self.assay_embedding_model = assay_embedding_model
        self.split_method = split_method

        # Keep rows whose record id has both affinity module inputs and assay context embedding available
        affinity_dir = self.target_dir / "processed" / "affinity_module_inputs"
        embedding_dir = self.target_dir / "processed" / "assay_context_embedding" / self.assay_embedding_model
        ids = self.df['BindingDB Reactant_set_id'].astype(str)
        has_inputs = ids.map(lambda rid: (affinity_dir / rid / f"affinity_input_{rid}.npz").exists())
        has_embedding = ids.map(lambda rid: (embedding_dir / f"{rid}.pt").exists())
        self.df = self.df[has_inputs & has_embedding].reset_index(drop=True)

        # exclude inequality pairs across Kd / Ki / IC50
        affinity_cols = ['Kd (nM)', 'Ki (nM)', 'IC50 (nM)']
        for col in affinity_cols:
            self.df = self.df[~self.df[col].astype(str).str.contains('[<>]', regex=True, na=False)]

        values = self.df[affinity_cols].apply(pd.to_numeric, errors='coerce')
        n_present = values.notna().sum(axis=1)
        if (n_present > 1).any():
            offending = self.df.loc[n_present > 1, 'BindingDB Reactant_set_id'].tolist()
            raise ValueError(
                f"Expected at most one of {affinity_cols} per row, "
                f"found {(n_present > 1).sum()} rows with multiple: {offending[:10]}"
            )

        self.df = self.df[n_present == 1].reset_index(drop=True)
        affinity_nm = self.df[affinity_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
        self.df['y'] = -np.log10(affinity_nm * 1e-9)


        self.split_frac = [0.7, 0.1, 0.2]  # train, val, test
        self.seed = seed
        self.split = split

        if self.split_method == "pmid-level-random":
            split_df = self.create_fold_by_pmid(self.df, self.seed, self.split_frac)
        else:
            split_df = self.create_fold(self.df, self.seed, self.split_frac)
          
        
        if self.split == 'train':
            self.split_df = split_df['train']
        elif self.split == 'valid':
            self.split_df = split_df['valid']
        elif self.split == 'test':
            self.split_df = split_df['test']
        else:
            raise ValueError("Unknown split: {}".format(self.split))
        
        
    def create_fold(self, df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float]) -> Dict[str, pd.DataFrame]:
        """
        Random split into train/valid/test folds.
        """
        
        train_frac, val_frac, test_frac = frac
        test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)

        train_val = df[~df.index.isin(test.index)]
        val = train_val.sample(frac=val_frac/(1-test_frac), replace=False, random_state=1)
        train = train_val[~train_val.index.isin(val.index)]

        return {
            'train': train.reset_index(drop=True),
            'valid': val.reset_index(drop=True),
            'test': test.reset_index(drop=True)
        }

    def create_fold_by_pmid(self, df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float]) -> Dict[str, pd.DataFrame]:
        """Group-level random split by PMID. All rows with the same PMID go into the same partition."""
        train_frac, val_frac, test_frac = frac
        pmids = df['PMID'].unique()
        rng = np.random.RandomState(fold_seed)
        rng.shuffle(pmids)
        n = len(pmids)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test_pmids = set(pmids[:n_test])
        val_pmids = set(pmids[n_test:n_test + n_val])
        train_pmids = set(pmids[n_test + n_val:])
        return {
            'train': df[df['PMID'].isin(train_pmids)].reset_index(drop=True),
            'valid': df[df['PMID'].isin(val_pmids)].reset_index(drop=True),
            'test': df[df['PMID'].isin(test_pmids)].reset_index(drop=True),
        }

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor, Tensor]:
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
            reactant_set_id = self.split_df.iloc[idx]['BindingDB Reactant_set_id']
            y = self.split_df.iloc[idx]['y']
            s_inputs, z_affinity, coords_affinity, feats, assay_context = load_input(reactant_set_id=reactant_set_id, target_dir=self.target_dir, assay_embedding_model=self.assay_embedding_model)
            s_inputs = torch.tensor(s_inputs).float()
            z_affinity = torch.tensor(z_affinity).float()
            coords_affinity = torch.tensor(coords_affinity).float()
            y = torch.tensor(y).float().unsqueeze(0)
            feats = {k: torch.tensor(v).float() if isinstance(v, np.ndarray) else v for k, v in feats.items()}
            assay_context = torch.tensor(assay_context).float()
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return self.__getitem__(0)

        return s_inputs, z_affinity, coords_affinity, feats, assay_context, y

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
            split='train',
            seed=self.cfg.seed,
            target_dir=Path(self.cfg.target_dir),
            assay_embedding_model=self.cfg.assay_embedding_model,
            split_method=self.cfg.split_method,
        )
        return DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.

        """
        val_set = AffinityModuleDataset(
            df_path=Path(self.cfg.df_path),
            split='valid',
            seed=self.cfg.seed,
            target_dir=Path(self.cfg.target_dir),
            assay_embedding_model=self.cfg.assay_embedding_model,
            split_method=self.cfg.split_method,
        )
        return DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader.

        Returns
        -------
        DataLoader
            The test dataloader.

        """
        test_set = AffinityModuleDataset(
            df_path=Path(self.cfg.df_path),
            split='test',
            seed=self.cfg.seed,
            target_dir=Path(self.cfg.target_dir),
            assay_embedding_model=self.cfg.assay_embedding_model,
            split_method=self.cfg.split_method,
        )
        
        return DataLoader(
            test_set,
            batch_size=self.cfg.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
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
            s_inputs, z_affinity, coords_affinity, feats, assay_context, y = batch

            # Basic type/shape checks
            assert isinstance(s_inputs, torch.Tensor), "s_inputs must be a Tensor"
            assert isinstance(z_affinity, torch.Tensor), "z_affinity must be a Tensor"
            assert isinstance(coords_affinity, torch.Tensor), "coords_affinity must be a Tensor"
            assert isinstance(feats, dict), "feats must be a dict"
            assert isinstance(assay_context, torch.Tensor), "assay_context must be a Tensor"
            assert isinstance(y, torch.Tensor), "y must be a Tensor"

            bs = s_inputs.shape[0]
            assert z_affinity.shape[0] == bs, "Batch dim mismatch: z_affinity"
            assert coords_affinity.shape[0] == bs, "Batch dim mismatch: coords_affinity"
            assert assay_context.shape[0] == bs, "Batch dim mismatch: assay_context"

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
            print(f"    assay_context: {tuple(assay_context.shape)}  {assay_context.dtype}")
            print(f"    feats keys: {list(feats.keys())[:8]}{' …' if len(feats)>8 else ''}")
            print(f"    (e.g. 'token_pad_mask': {tuple(feats['token_pad_mask'].shape)}  {feats['token_pad_mask'].dtype})")
            print(f"    (e.g. 'mol_type': {tuple(feats['mol_type'].shape)}  {feats['mol_type'].dtype})")
            print(f"    (e.g. 'y': {tuple(y.shape)}  {y.dtype})")

    print("\nSanity check passed")

#%%
if __name__ == "__main__":
    cfg = DataConfig(
        df_path='/data/mwu11/LLM_affinity/BindingDB/non_null_spr_itc_structured_description_subset.csv',
        target_dir='/data/mwu11/boltz/BindingDB/boltz_results_yaml_affinity_input',
        seed=42,
        batch_size=2
    )

    data_module = AffinityModuleDataModule(cfg)
    data_module.setup()

    dm_sanity_check(data_module, n_batches=1)
# %%
