"""Build leaf-field ablation embeddings for the assay-context study.

For a given assay type, this module:
  1. Enumerates the leaf fields of ``structured_description`` from the assay schema.
  2. For every reactant, removes one field at a time (``baseline`` keeps all fields),
     re-serialises the JSON, and embeds it with Qwen3-Embedding-8B.
  3. Caches the masked JSON and the per-reactant embedding so that
     ``AffinityModuleDataModule`` can consume each variant via its ``assay_embedding_model``.

The text recipe mirrors ``/data/mwu11/LLM_affinity/embedding/embed_descriptions.py`` exactly so
that the ``baseline`` variant reproduces the existing ``qwen3/`` embeddings (verification anchor).

CLI usage (standalone caching):
    python scripts/train/ablation_embed.py --assay_type itc --batch_size 128

Library usage (called by the training script):
    from ablation_embed import build_ablation_embeddings
    variants = build_ablation_embeddings("itc", target_dir, df_path)
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch

# Human-readable assay labels, identical to embed_descriptions.py.
ASSAY_LABELS = {
    "spr": "Surface Plasmon Resonance (SPR)",
    "itc": "Isothermal Titration Calorimetry (ITC)",
    "fpa": "Fluorescence Polarization Assay (FPA)",
    "rba": "Radioligand Binding Assay (RBA)",
}

QWEN3_NAME = "Qwen/Qwen3-Embedding-8B"
QWEN3_MODEL_KWARGS = {"device_map": "auto"}
QWEN3_TOKENIZER_KWARGS = {"padding_side": "left"}

# Augmentation key used to ablate the assay-type tag itself.
ASSAY_TYPE_VARIANT = "assay_type"


def assay_paths(assay_type: str) -> dict[str, Path]:
    """Return the canonical per-assay paths."""
    base = Path("/data/mwu11/boltz/BindingDB") / assay_type
    return {
        "extraction_dir": base / f"{assay_type}_extraction_results_two_step_bindingdb_standardized",
        "schema": base / f"{assay_type}_schema.json",
        "df_path": base / f"BindingDB_{assay_type}.tsv",
        "target_dir": base / "boltz_results_yaml_affinity_input",
    }


def enumerate_leaf_fields(schema_path: Path) -> list[str]:
    """Recurse the JSON schema and return dotted paths to every leaf field.

    Descends into ``type: "object"`` nodes. Excludes the top-level ``pmid`` key, which is
    part of the schema but not of the extracted ``structured_description`` payload.
    """
    schema = json.loads(Path(schema_path).read_text())
    leaves: list[str] = []

    def recurse(props: dict, prefix: str) -> None:
        for key, node in props.items():
            if isinstance(node, dict) and node.get("type") == "object" and "properties" in node:
                recurse(node["properties"], f"{prefix}{key}.")
            else:
                leaves.append(f"{prefix}{key}")

    recurse(schema["properties"], "")
    return [f for f in leaves if f != "pmid"]


def _delete_dotted(obj: dict, dotted: str) -> None:
    """Delete a (possibly nested) dotted key in-place. Absent key -> no-op."""
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)


def variant_dirname(variant: str) -> str:
    """Filesystem-safe name for a variant (dotted field path -> double underscore)."""
    if variant == "baseline":
        return "baseline"
    return variant.replace(".", "__")


def build_augmented_json(
    structured_description: dict,
    entry: dict,
    assay_label: str,
    variant: str,
) -> dict:
    """Build the augmented JSON object that gets serialised and embedded.

    Matches embed_descriptions.py: deep-copy the structured_description, append the
    ``assay_type`` label and (when available) ``binding_affinity_type``. ``variant`` selects
    what to ablate:
      - ``baseline``                -> nothing removed
      - ``assay_type``              -> the assay_type augmentation key omitted
      - any dotted path             -> that structured_description leaf removed
      - ``a.b+c.d`` (``+``-joined)  -> every listed leaf removed; include ``assay_type``
                                       in the list to also drop the assay-type tag
    """
    masked = copy.deepcopy(structured_description)
    include_assay_type = True
    if variant != "baseline":
        parts = variant.split("+")
        if ASSAY_TYPE_VARIANT in parts:
            include_assay_type = False
        for part in parts:
            if part and part != ASSAY_TYPE_VARIANT:
                _delete_dotted(masked, part)

    if include_assay_type:
        masked["assay_type"] = assay_label

    affinity = entry.get("affinity_data")
    if affinity and affinity.get("type"):
        masked["binding_affinity_type"] = affinity["type"]
    return masked


def _needed_reactant_ids(df_path: Path, target_dir: Path, extraction_dir: Path) -> list[str]:
    """Reactant ids that the training DataModule will actually use.

    Mirrors AffinityModuleDataset's row filter so we don't waste embedding compute on rows
    that get dropped at training time:
      - precomputed affinity input present
      - extraction JSON present
      - no inequality ('<' or '>') in Kd / Ki / IC50
      - exactly one of Kd / Ki / IC50 populated
    """
    sep = "\t" if str(df_path).endswith(".tsv") else ","
    df = pd.read_csv(df_path, sep=sep, low_memory=False)

    affinity_cols = ["Kd (nM)", "Ki (nM)", "IC50 (nM)"]
    for col in affinity_cols:
        df = df[~df[col].astype(str).str.contains("[<>]", regex=True, na=False)]
    values = df[affinity_cols].apply(pd.to_numeric, errors="coerce")
    df = df[values.notna().sum(axis=1) == 1]

    affinity_dir = target_dir / "processed" / "affinity_module_inputs"
    ids = df["BindingDB Reactant_set_id"].astype(str).unique().tolist()
    return [
        rid for rid in ids
        if (affinity_dir / rid / f"affinity_input_{rid}.npz").exists()
        and (extraction_dir / f"{rid}.json").exists()
    ]


def _load_entries(reactant_ids: list[str], extraction_dir: Path) -> dict[str, dict]:
    """Load each reactant's full extraction entry; skip entries lacking structured_description."""
    entries: dict[str, dict] = {}
    for rid in reactant_ids:
        try:
            entry = json.loads((extraction_dir / f"{rid}.json").read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[ablation_embed] skip {rid}: cannot read JSON ({exc})")
            continue
        if entry.get("structured_description") is None:
            continue
        entries[rid] = entry
    return entries


def _variant_complete(emb_dir: Path, reactant_ids: list[str]) -> bool:
    """True if every reactant already has a cached embedding for this variant."""
    if not emb_dir.is_dir():
        return False
    return all((emb_dir / f"{rid}.pt").exists() for rid in reactant_ids)


def build_ablation_embeddings(
    assay_type: str,
    target_dir: Optional[Path] = None,
    df_path: Optional[Path] = None,
    extraction_dir: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    variants: Optional[list[str]] = None,
    batch_size: int = 128,
    verify_baseline: bool = True,
) -> list[str]:
    """Generate and cache masked-JSON embeddings for the requested ablation variants.

    ``variants`` is the **exact** list to produce. Each item is either ``"baseline"``,
    ``"assay_type"``, or a dotted ``structured_description`` leaf path. When ``None``, the
    default is the full study set: baseline + every leaf field + assay_type.

    Embeddings land in ``target_dir/processed/assay_context_embedding/ablation_<variant>/``;
    masked JSON in ``.../assay_context_embedding/masked_json/<variant>/``.
    """
    paths = assay_paths(assay_type)
    target_dir = Path(target_dir or paths["target_dir"])
    df_path = Path(df_path or paths["df_path"])
    extraction_dir = Path(extraction_dir or paths["extraction_dir"])
    schema_path = Path(schema_path or paths["schema"])
    assay_label = ASSAY_LABELS[assay_type]

    if variants is None:
        variants = ["baseline", *enumerate_leaf_fields(schema_path), ASSAY_TYPE_VARIANT]

    reactant_ids = _needed_reactant_ids(df_path, target_dir, extraction_dir)
    print(f"[ablation_embed] {assay_type}: {len(reactant_ids)} reactants with inputs + extraction JSON")
    entries = _load_entries(reactant_ids, extraction_dir)
    reactant_ids = [rid for rid in reactant_ids if rid in entries]
    print(f"[ablation_embed] {len(reactant_ids)} reactants with a structured_description")
    if not reactant_ids:
        raise RuntimeError(f"No usable reactants for assay {assay_type}")

    emb_root = target_dir / "processed" / "assay_context_embedding"
    json_root = emb_root / "masked_json"

    # Decide which variants still need work before paying the cost of loading the 8B model.
    todo = [v for v in variants if not _variant_complete(emb_root / f"ablation_{variant_dirname(v)}", reactant_ids)]
    if not todo:
        print("[ablation_embed] all variants already cached; nothing to do")
        if verify_baseline:
            _verify_baseline(emb_root, reactant_ids)
        return variants

    print(f"[ablation_embed] {len(todo)}/{len(variants)} variants need embedding")
    model = _load_qwen3()

    for variant in todo:
        vdir = variant_dirname(variant)
        emb_dir = emb_root / f"ablation_{vdir}"
        jdir = json_root / vdir
        emb_dir.mkdir(parents=True, exist_ok=True)
        jdir.mkdir(parents=True, exist_ok=True)

        texts: list[str] = []
        for rid in reactant_ids:
            entry = entries[rid]
            masked = build_augmented_json(
                entry["structured_description"], entry, assay_label, variant
            )
            (jdir / f"{rid}.json").write_text(json.dumps(masked, indent=1))
            texts.append(json.dumps(masked))

        print(f"[ablation_embed] embedding variant '{variant}' ({len(texts)} texts)")
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        embeddings = np.asarray(embeddings)
        for rid, vec in zip(reactant_ids, embeddings):
            torch.save(torch.from_numpy(vec).float(), emb_dir / f"{rid}.pt")
        print(f"[ablation_embed] saved {len(reactant_ids)} embeddings to {emb_dir}")

    del model
    torch.cuda.empty_cache()

    if verify_baseline and "baseline" in variants:
        _verify_baseline(emb_root, reactant_ids)
    return variants


def _load_qwen3():
    """Load the Qwen3-Embedding-8B SentenceTransformer with the canonical config."""
    from sentence_transformers import SentenceTransformer

    print(f"[ablation_embed] loading {QWEN3_NAME}")
    return SentenceTransformer(
        QWEN3_NAME,
        model_kwargs=QWEN3_MODEL_KWARGS,
        tokenizer_kwargs=QWEN3_TOKENIZER_KWARGS,
    )


def _verify_baseline(emb_root: Path, reactant_ids: list[str], min_cosine: float = 0.999) -> None:
    """Check that the freshly generated baseline matches the existing qwen3/ embeddings.

    Two independent runs of an 8B transformer differ by fp noise (kernel / batching / library
    version), so an exact match is not expected. We require high cosine similarity instead:
    a low value would indicate the text recipe genuinely differs from how qwen3/ was produced.
    """
    ref_dir = emb_root / "qwen3"
    base_dir = emb_root / "ablation_baseline"
    if not ref_dir.is_dir():
        print(f"[ablation_embed] WARNING: no reference dir {ref_dir}; skipping baseline check")
        return

    sims: list[float] = []
    for rid in reactant_ids:
        ref_p, base_p = ref_dir / f"{rid}.pt", base_dir / f"{rid}.pt"
        if not (ref_p.exists() and base_p.exists()):
            continue
        ref = torch.load(ref_p).float()
        base = torch.load(base_p).float()
        sims.append(float(torch.nn.functional.cosine_similarity(ref, base, dim=0)))

    if not sims:
        print("[ablation_embed] WARNING: no overlapping ids to verify baseline against qwen3/")
        return

    sims_t = torch.tensor(sims)
    lo, mean = sims_t.min().item(), sims_t.mean().item()
    if lo >= min_cosine:
        print(
            f"[ablation_embed] baseline verified vs qwen3/: cosine sim min={lo:.6f} "
            f"mean={mean:.6f} over {len(sims)} reactants (fp-noise level differences)"
        )
    else:
        n_bad = int((sims_t < min_cosine).sum())
        print(
            f"[ablation_embed] WARNING: baseline vs qwen3/ cosine sim min={lo:.6f} mean={mean:.6f}; "
            f"{n_bad}/{len(sims)} below {min_cosine} — the text recipe may not match how "
            "qwen3/ was produced"
        )


@click.command()
@click.option("--assay_type", required=True, type=click.Choice(list(ASSAY_LABELS.keys())))
@click.option("--batch_size", default=128, show_default=True, type=int)
@click.option("--variants", type=str, default=None,
              help="Comma-separated exact list of variants to produce (e.g. 'baseline', "
                   "'baseline,assay_type', 'protein.name'). Default: baseline + every leaf "
                   "field + assay_type.")
@click.option("--no_verify", is_flag=True, default=False, help="Skip baseline-vs-qwen3 verification.")
def main(assay_type: str, batch_size: int, variants: Optional[str], no_verify: bool) -> None:
    """Standalone caching entry point (run with the llm_affinity conda env)."""
    variant_list = None
    if variants:
        variant_list = [v.strip() for v in variants.split(",") if v.strip()]
    produced = build_ablation_embeddings(
        assay_type, variants=variant_list, batch_size=batch_size, verify_baseline=not no_verify
    )
    print(f"[ablation_embed] done: {len(produced)} variants for {assay_type}")


if __name__ == "__main__":
    main()
