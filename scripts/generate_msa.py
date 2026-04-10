"""Generate MSA CSV files from a JSON file of {uniprot_id: protein_sequence}.

Usage:
    python scripts/generate_msa.py input.json --out_dir ./msa_output

The input JSON should look like:
    {
        "P12345": "MKTAYIAKQRQISFVKSH...",
        "Q67890": "MVLSPADKTNVKAAWGKV..."
    }

Each protein gets its own CSV file at <out_dir>/<uniprot_id>.csv with columns:
    key,sequence
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to path so we can import boltz modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from boltz.data import const
from boltz.data.msa.mmseqs2 import run_mmseqs2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _process_single(
    name: str,
    sequence: str,
    out_dir: Path,
    msa_server_url: str,
    pairing_strategy: str,
    auth_kwargs: dict,
) -> str:
    """Process a single protein following the same logic as compute_msa in main.py."""
    logger.info(f"Generating MSA for {name}...")

    # Paired MSA (only meaningful for multi-chain, but follow original logic)
    if len(sequence) > 1 and isinstance(sequence, list):
        paired_msas = run_mmseqs2(
            sequence,
            prefix=str(out_dir / f"{name}_paired_tmp"),
            use_env=True,
            use_pairing=True,
            pairing_strategy=pairing_strategy,
            host_url=msa_server_url,
            **auth_kwargs,
        )
    else:
        paired_msas = [""]

    # Unpaired MSA
    unpaired_msa = run_mmseqs2(
        sequence if isinstance(sequence, list) else [sequence],
        prefix=str(out_dir / f"{name}_unpaired_tmp"),
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        **auth_kwargs,
    )

    # Write one CSV per chain (follows compute_msa logic from main.py)
    names_list = [name] if isinstance(sequence, str) else [f"{name}_{i}" for i in range(len(sequence))]

    for idx, chain_name in enumerate(names_list):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [i for i, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # query already present in paired

        # Combine
        seqs = paired + unpaired
        all_keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_lines = ["key,sequence"] + [f"{k},{seq}" for k, seq in zip(all_keys, seqs)]
        csv_path = out_dir / f"{chain_name}.csv"
        csv_path.write_text("\n".join(csv_lines))
        logger.info(f"Wrote {len(seqs)} MSA sequences to {csv_path}")

    return name


def generate_msa(
    data: dict[str, str],
    out_dir: Path,
    msa_server_url: str = "https://api.colabfold.com",
    pairing_strategy: str = "greedy",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    auth_headers: dict[str, str] | None = None,
    workers: int = 4,
) -> None:
    """Generate MSA CSV files for all sequences in data.

    Parameters
    ----------
    data : dict[str, str]
        Mapping of uniprot_id -> protein_sequence.
    out_dir : Path
        Directory to write CSV files into.
    msa_server_url : str
        MMSeqs2 server URL.
    pairing_strategy : str
        Pairing strategy ("greedy" or "complete").
    msa_server_username : str, optional
        Username for basic auth.
    msa_server_password : str, optional
        Password for basic auth.
    auth_headers : dict, optional
        Custom auth headers (e.g. {"X-API-Key": "..."}).
    workers : int
        Number of parallel workers.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    auth_kwargs = dict(
        msa_server_username=msa_server_username,
        msa_server_password=msa_server_password,
        auth_headers=auth_headers,
    )

    # Process each protein independently in parallel
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_single, name, seq, out_dir, msa_server_url,
                pairing_strategy, auth_kwargs,
            ): name
            for name, seq in data.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception:
                logger.exception(f"Failed to generate MSA for {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MSA CSV files from a JSON of {uniprot_id: sequence}."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to input JSON file ({uniprot_id: protein_sequence}).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("msa_output"),
        help="Output directory for CSV files (default: ./msa_output).",
    )
    parser.add_argument(
        "--msa_server_url",
        type=str,
        default="https://api.colabfold.com",
        help="MMSeqs2 server URL.",
    )
    parser.add_argument(
        "--pairing_strategy",
        type=str,
        default="greedy",
        choices=["greedy", "complete"],
        help="MSA pairing strategy (default: greedy).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4).",
    )
    parser.add_argument(
        "--msa_server_username",
        type=str,
        default=None,
        help="Username for basic auth (or set BOLTZ_MSA_USERNAME env var).",
    )
    parser.add_argument(
        "--msa_server_password",
        type=str,
        default=None,
        help="Password for basic auth (or set BOLTZ_MSA_PASSWORD env var).",
    )
    parser.add_argument(
        "--api_key_header",
        type=str,
        default=None,
        help="Custom header name for API key auth (e.g. X-API-Key).",
    )
    parser.add_argument(
        "--api_key_value",
        type=str,
        default=None,
        help="API key value (or set MSA_API_KEY_VALUE env var).",
    )

    args = parser.parse_args()

    # Load input JSON
    with open(args.input_json) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} sequences from {args.input_json}")

    # Resolve credentials from env if not provided
    username = args.msa_server_username or os.environ.get("BOLTZ_MSA_USERNAME")
    password = args.msa_server_password or os.environ.get("BOLTZ_MSA_PASSWORD")
    api_key_value = args.api_key_value or os.environ.get("MSA_API_KEY_VALUE")

    # Build auth headers if API key is provided
    auth_headers = None
    if api_key_value:
        header = args.api_key_header or "X-API-Key"
        auth_headers = {"Content-Type": "application/json", header: api_key_value}

    generate_msa(
        data=data,
        out_dir=args.out_dir,
        msa_server_url=args.msa_server_url,
        pairing_strategy=args.pairing_strategy,
        msa_server_username=username,
        msa_server_password=password,
        auth_headers=auth_headers,
        workers=args.workers,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
