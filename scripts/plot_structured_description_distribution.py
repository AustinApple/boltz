"""Plot distribution of non-null fields in structured_description across
train/valid/test splits for seeds 0-4 on BindingDB_fpa_kd_ki_ic50.tsv.

The split logic mirrors AffinityModuleDataset (pair-level-random). The
has_embedding filter is skipped because fpa has no assay_context_embedding
directory; the has_inputs and single-affinity filters are applied.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NULL_STRINGS = {"not specified", "not reported"}


def is_null_leaf(v):
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in NULL_STRINGS:
        return True
    return False


def count_non_null_leaves(obj):
    if isinstance(obj, dict):
        return sum(count_non_null_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_non_null_leaves(v) for v in obj)
    return 0 if is_null_leaf(obj) else 1


def count_total_leaves(obj):
    if isinstance(obj, dict):
        return sum(count_total_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_total_leaves(v) for v in obj)
    return 1


def load_structured_descriptions(sd_dir):
    lookup = {}
    for p in glob.glob(str(Path(sd_dir) / "*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}")
            continue
        if not isinstance(d, dict):
            continue
        for rid, v in d.items():
            if isinstance(v, dict):
                lookup[str(rid)] = v.get("structured_description")
    return lookup


def filter_df(df, target_dir):
    affinity_dir = Path(target_dir) / "processed" / "affinity_module_inputs"
    ids = df["BindingDB Reactant_set_id"].astype(str)
    has_inputs = ids.map(
        lambda rid: (affinity_dir / rid / f"affinity_input_{rid}.npz").exists()
    )
    df = df[has_inputs].reset_index(drop=True)

    affinity_cols = ["Kd (nM)", "Ki (nM)", "IC50 (nM)"]
    for col in affinity_cols:
        df = df[~df[col].astype(str).str.contains("[<>]", regex=True, na=False)]

    values = df[affinity_cols].apply(pd.to_numeric, errors="coerce")
    n_present = values.notna().sum(axis=1)
    df = df[n_present == 1].reset_index(drop=True)
    return df


def create_fold(df, fold_seed, frac=(0.7, 0.1, 0.2)):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df_path",
        default="/data/mwu11/boltz/BindingDB/fpa/BindingDB_fpa_kd_ki_ic50.tsv",
    )
    parser.add_argument(
        "--target_dir",
        default="/data/mwu11/boltz/BindingDB/fpa/boltz_results_yaml_affinity_input",
    )
    parser.add_argument(
        "--sd_dir",
        default="/data/mwu11/LLM_affinity/qwen_test/fpa_extraction_results_two_step_bindingdb_standardized",
    )
    parser.add_argument(
        "--out_path",
        default="/data/mwu11/boltz/structured_description_distribution.png",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    df = pd.read_csv(args.df_path, sep="\t")
    print(f"raw rows: {len(df)}")
    df = filter_df(df, args.target_dir)
    print(f"after has_inputs + affinity filter: {len(df)}")

    sd_lookup = load_structured_descriptions(args.sd_dir)
    populated = sum(1 for v in sd_lookup.values() if v is not None)
    print(
        f"structured_description records: {len(sd_lookup)} "
        f"(populated / non-null dict: {populated})"
    )

    # drop rows whose structured_description has zero non-null leaf fields
    # (treating None and "Not specified" / "Not reported" as null)
    populated_ids = {
        rid for rid, v in sd_lookup.items()
        if v is not None and count_non_null_leaves(v) > 0
    }
    before = len(df)
    df = df[df["BindingDB Reactant_set_id"].astype(str).isin(populated_ids)].reset_index(drop=True)
    print(f"after dropping rows with zero non-null structured_description fields: {len(df)} (removed {before - len(df)})")

    # infer a total-leaf count from a populated example for axis labeling
    total_leaves = None
    for v in sd_lookup.values():
        if isinstance(v, dict):
            total_leaves = count_total_leaves(v)
            break
    print(f"inferred total leaf fields per structured_description: {total_leaves}")

    seeds = args.seeds
    splits = ["train", "valid", "test"]

    fig, axes = plt.subplots(
        len(seeds), len(splits), figsize=(15, 3 * len(seeds)), sharex=True
    )
    if len(seeds) == 1:
        axes = np.array([axes])

    summary = []
    all_counts = {split: [] for split in splits}

    for si, seed in enumerate(seeds):
        folds = create_fold(df, seed)
        for sj, split in enumerate(splits):
            sub = folds[split]
            ids = sub["BindingDB Reactant_set_id"].astype(str).tolist()
            counts = [count_non_null_leaves(sd_lookup[rid]) for rid in ids]

            # Kd is grouped with Ki per user preference
            affinity_cols = ["Kd (nM)", "Ki (nM)", "IC50 (nM)"]
            vals = sub[affinity_cols].apply(pd.to_numeric, errors="coerce")
            n_ki = int((vals["Ki (nM)"].notna() | vals["Kd (nM)"].notna()).sum())
            n_ic50 = int(vals["IC50 (nM)"].notna().sum())

            all_counts[split].append(counts)
            summary.append(
                dict(
                    seed=seed,
                    split=split,
                    n=len(ids),
                    n_Ki_Kd=n_ki,
                    n_IC50=n_ic50,
                    mean=float(np.mean(counts)) if counts else 0.0,
                    median=float(np.median(counts)) if counts else 0.0,
                    min=int(np.min(counts)) if counts else 0,
                    max=int(np.max(counts)) if counts else 0,
                )
            )

            ax = axes[si, sj]
            ax.hist(counts, bins=30, edgecolor="black")
            ax.set_title(f"seed={seed} | {split} (n={len(ids)})")
            if si == len(seeds) - 1:
                ax.set_xlabel("# non-null leaf fields")
            if sj == 0:
                ax.set_ylabel("count")

    suptitle = "Non-null fields in structured_description per split"
    if total_leaves:
        suptitle += f"  (total leaf fields ≈ {total_leaves})"
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=120)
    print(f"saved plot to {args.out_path}")

    print("\nsummary:")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
