#%%
from dataclasses import dataclass
from pathlib import Path

from typing import Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import pandas as pd


@dataclass
class DataConfig:
    """Data configuration."""

    df_path: str
    target_dir: str
    seed: int
    batch_size: int
    assay_embedding_model: str
    split_method: str = "pair-level-random"  # "pair-level-random" or "pmid-level-random"


def load_input(reactant_set_id: str, target_dir: Path, assay_embedding_model: str) -> Tensor:
    """Load only the assay context embedding for a given reactant set.

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
    Tensor
        The assay context embedding.
    """
    assay_context = torch.load(target_dir / "processed" / "assay_context_embedding" / assay_embedding_model / f"{reactant_set_id}.pt")
    return assay_context


def collate(data: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : list[tuple[Tensor, Tensor]]
        The data to collate. Each element is (assay_context, y).

    Returns
    -------
    tuple[Tensor, Tensor]
        The collated (assay_context, y).
    """
    assay_contexts = torch.stack([d[0] for d in data], dim=0)
    y = torch.stack([d[1] for d in data], dim=0)
    return assay_contexts, y


class AssayContextOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, df_path: Path, split: str, seed: int, target_dir: Path, assay_embedding_model: str, split_method: str = "pair-level-random") -> None:
        """Initialize the dataset."""
        super().__init__()

        self.df = pd.read_csv(df_path)
        self.target_dir = target_dir
        self.assay_embedding_model = assay_embedding_model
        self.split_method = split_method

        # Read successful runs and include only them
        success_path = Path('/data/mwu11/boltz/BindingDB/boltz_results_yaml_affinity_input/success.txt')
        if success_path.exists():
            with open(success_path, 'r') as f:
                success_ids = set(line.strip() for line in f if line.strip())
            self.df = self.df[self.df['BindingDB Reactant_set_id'].astype(str).isin(success_ids)].reset_index(drop=True)

        # exclude Kd involved inequality pairs
        self.df = self.df[~self.df['Kd (nM)'].astype(str).str.contains('[<>]', regex=True)]
        # convert Kd to pKd
        self.df['y'] = -np.log10(self.df['Kd (nM)'].astype(float) * 1e-9)

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
        """Random split into train/valid/test folds."""
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

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        tuple[Tensor, Tensor]
            (assay_context, y)
        """
        try:
            reactant_set_id = self.split_df.iloc[idx]['BindingDB Reactant_set_id']
            y = self.split_df.iloc[idx]['y']
            assay_context = load_input(
                reactant_set_id=reactant_set_id,
                target_dir=self.target_dir,
                assay_embedding_model=self.assay_embedding_model,
            )
            assay_context = torch.tensor(assay_context).float()
            y = torch.tensor(y).float().unsqueeze(0)
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return self.__getitem__(0)

        return assay_context, y

    def __len__(self) -> int:
        return len(self.split_df)


class AssayContextOnlyDataModule(pl.LightningDataModule):
    """DataModule for assay-context-only ablation."""
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        return

    def train_dataloader(self) -> DataLoader:
        train_set = AssayContextOnlyDataset(
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
        val_set = AssayContextOnlyDataset(
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
        test_set = AssayContextOnlyDataset(
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


#%%
if __name__ == "__main__":
    cfg = DataConfig(
        df_path='/data/mwu11/LLM_affinity/BindingDB/non_null_spr_itc_structured_description_subset.csv',
        target_dir='/data/mwu11/boltz/BindingDB/boltz_results_yaml_affinity_input',
        seed=42,
        batch_size=2,
        assay_embedding_model='qwen3',
    )

    data_module = AssayContextOnlyDataModule(cfg)
    data_module.setup()

    # Sanity check
    for name, loader in [("train", data_module.train_dataloader()), ("val", data_module.val_dataloader()), ("test", data_module.test_dataloader())]:
        batch = next(iter(loader))
        assay_context, y = batch
        print(f"[{name}] assay_context: {tuple(assay_context.shape)} {assay_context.dtype}, y: {tuple(y.shape)} {y.dtype}")
    print("Sanity check passed")
# %%
