from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader


# Fixed assay id mapping. index = assay_type_id.
ASSAY_TYPES: List[str] = ["itc", "spr", "rba", "fpa"]
ITC_ID: int = ASSAY_TYPES.index("itc")
# Default set of assays whose context slot is filled by the model's learned
# null vector instead of a real per-sample embedding. Override per-run via
# DataConfig.null_context_assays / CLI --null_context_assays.
DEFAULT_NULL_CONTEXT_ASSAYS: FrozenSet[str] = frozenset({"itc"})


def assay_type_id(assay_type: str) -> int:
    return ASSAY_TYPES.index(assay_type)


@dataclass
class AssayConfig:
    assay_type: str
    df_path: str
    target_dir: str


@dataclass
class DataConfig:
    assays: List[AssayConfig]
    seed: int
    batch_size: int
    assay_embedding_model: str
    # which saved g to use from extract_affinity_g.py output
    g_module_key: str = "affinity_module1"
    split_method: str = "pair-level-random"
    # When False, skip assay-context loading/filtering for a clean g-only baseline.
    load_assay_context: bool = True
    # Target dim for the context slot (used for zero placeholders and truncation).
    # Only meaningful when load_assay_context=True.
    assay_context_dim: int = 128
    # Assay types whose rows should use the model's learned null context instead
    # of a real embedding. Default: {"itc"}.
    null_context_assays: FrozenSet[str] = DEFAULT_NULL_CONTEXT_ASSAYS


def load_g(reactant_set_id: str, target_dir: Path, g_module_key: str) -> np.ndarray:
    path = target_dir / "processed" / "g" / f"{reactant_set_id}" / f"g_{reactant_set_id}.npz"
    with np.load(path) as d:
        g = d[g_module_key]
    return g  # shape (1, token_z)


def collate(data):
    g = torch.stack([d[0] for d in data], dim=0)
    assay_context = torch.stack([d[1] for d in data], dim=0)
    assay_type_ids = torch.stack([d[2] for d in data], dim=0)
    y = torch.stack([d[3] for d in data], dim=0)
    return g, assay_context, assay_type_ids, y


def _filter_affinity_df(df: pd.DataFrame) -> pd.DataFrame:
    affinity_cols = ["Kd (nM)", "Ki (nM)", "IC50 (nM)"]
    for col in affinity_cols:
        df = df[~df[col].astype(str).str.contains("[<>]", regex=True, na=False)]
    values = df[affinity_cols].apply(pd.to_numeric, errors="coerce")
    n_present = values.notna().sum(axis=1)
    df = df[n_present == 1].reset_index(drop=True)
    affinity_nm = (
        df[affinity_cols]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1, min_count=1)
    )
    positive = affinity_nm > 0
    df = df[positive].reset_index(drop=True)
    affinity_nm = affinity_nm[positive].reset_index(drop=True)
    df["y"] = -np.log10(affinity_nm * 1e-9)
    return df


class AffinityGHeadDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        assays: List[AssayConfig],
        split: str,
        seed: int,
        assay_embedding_model: str,
        g_module_key: str,
        split_method: str = "pair-level-random",
        load_assay_context: bool = True,
        assay_context_dim: int = 128,
        null_context_assays: FrozenSet[str] = DEFAULT_NULL_CONTEXT_ASSAYS,
    ) -> None:
        super().__init__()

        self.assay_embedding_model = assay_embedding_model
        self.g_module_key = g_module_key
        self.split_method = split_method
        self.load_assay_context = load_assay_context
        self.assay_context_dim = assay_context_dim
        self.null_context_assays = frozenset(null_context_assays)

        per_assay_dfs: List[pd.DataFrame] = []
        for ac in assays:
            df_path = Path(ac.df_path)
            target_dir = Path(ac.target_dir)
            sep = "\t" if str(df_path).endswith(".tsv") else ","
            df = pd.read_csv(df_path, sep=sep)

            g_dir = target_dir / "processed" / "g"
            ids = df["BindingDB Reactant_set_id"].astype(str)
            mask = ids.map(lambda rid: (g_dir / rid / f"g_{rid}.npz").exists())

            # Always require both g and assay-context embedding to exist for every
            # row, so baseline / g+c / context-only scenarios train on the same
            # row set and results are directly comparable.
            emb_dir = (
                target_dir
                / "processed"
                / "assay_context_embedding"
                / assay_embedding_model
            )
            has_emb = ids.map(lambda rid: (emb_dir / f"{rid}.pt").exists())
            mask = mask & has_emb

            df = df[mask].reset_index(drop=True)
            df = _filter_affinity_df(df)

            df["assay_type"] = ac.assay_type
            df["_target_dir"] = str(target_dir)
            per_assay_dfs.append(df)

        if not per_assay_dfs:
            raise ValueError("No assays configured")
        self.df = pd.concat(per_assay_dfs, ignore_index=True)

        self.split_frac = [0.7, 0.1, 0.2]
        self.seed = seed
        self.split = split

        if self.split_method == "pmid-level-random":
            split_df = self.create_fold_by_pmid(self.df, self.seed, self.split_frac)
        else:
            split_df = self.create_fold(self.df, self.seed, self.split_frac)

        if self.split == "train":
            self.split_df = split_df["train"]
        elif self.split == "valid":
            self.split_df = split_df["valid"]
        elif self.split == "test":
            self.split_df = split_df["test"]
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def create_fold(
        self, df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float]
    ) -> Dict[str, pd.DataFrame]:
        train_frac, val_frac, test_frac = frac
        test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
        train_val = df[~df.index.isin(test.index)]
        val = train_val.sample(
            frac=val_frac / (1 - test_frac), replace=False, random_state=1
        )
        train = train_val[~train_val.index.isin(val.index)]
        return {
            "train": train.reset_index(drop=True),
            "valid": val.reset_index(drop=True),
            "test": test.reset_index(drop=True),
        }

    def create_fold_by_pmid(
        self, df: pd.DataFrame, fold_seed: int, frac: Tuple[float, float, float]
    ) -> Dict[str, pd.DataFrame]:
        train_frac, val_frac, test_frac = frac
        pmids = df["PMID"].unique()
        rng = np.random.RandomState(fold_seed)
        rng.shuffle(pmids)
        n = len(pmids)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test_pmids = set(pmids[:n_test])
        val_pmids = set(pmids[n_test : n_test + n_val])
        train_pmids = set(pmids[n_test + n_val :])
        return {
            "train": df[df["PMID"].isin(train_pmids)].reset_index(drop=True),
            "valid": df[df["PMID"].isin(val_pmids)].reset_index(drop=True),
            "test": df[df["PMID"].isin(test_pmids)].reset_index(drop=True),
        }

    def __getitem__(self, idx: int):
        try:
            row = self.split_df.iloc[idx]
            rid = row["BindingDB Reactant_set_id"]
            y = row["y"]
            assay_type = row["assay_type"]
            target_dir = Path(row["_target_dir"])

            g = load_g(
                reactant_set_id=rid,
                target_dir=target_dir,
                g_module_key=self.g_module_key,
            )
            g = torch.from_numpy(g).float().squeeze(0)  # (token_z,)

            if self.load_assay_context:
                if assay_type in self.null_context_assays:
                    # Placeholder; the model overwrites this with its learned null.
                    assay_context = torch.zeros(self.assay_context_dim)
                else:
                    assay_context = torch.load(
                        target_dir
                        / "processed"
                        / "assay_context_embedding"
                        / self.assay_embedding_model
                        / f"{rid}.pt"
                    )
                    assay_context = torch.as_tensor(assay_context).float()
                    if assay_context.shape[-1] > self.assay_context_dim:
                        assay_context = assay_context[..., : self.assay_context_dim]
            else:
                assay_context = torch.zeros(1)

            aid = torch.tensor(assay_type_id(assay_type), dtype=torch.long)
            y = torch.tensor(y).float().unsqueeze(0)
        except Exception as e:
            print(f"Error loading data for index {idx}: {e}")
            return self.__getitem__(0)

        return g, assay_context, aid, y

    def __len__(self) -> int:
        return len(self.split_df)


class AffinityGHeadDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        return

    def _make_loader(self, split: str, shuffle: bool) -> DataLoader:
        ds = AffinityGHeadDataset(
            assays=self.cfg.assays,
            split=split,
            seed=self.cfg.seed,
            assay_embedding_model=self.cfg.assay_embedding_model,
            g_module_key=self.cfg.g_module_key,
            split_method=self.cfg.split_method,
            load_assay_context=self.cfg.load_assay_context,
            assay_context_dim=self.cfg.assay_context_dim,
            null_context_assays=self.cfg.null_context_assays,
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=shuffle,
            collate_fn=collate,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader("valid", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader("test", shuffle=False)
