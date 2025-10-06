import os
import pandas as pd
import numpy as np
import csv
from typing import List, Optional, Dict
from collections import Counter, defaultdict
from anarci import anarci

ACID_EMBEDDINGS = "acid_embeddings"
ACID_EMBEDDINGS_SCOPE = "acid_emb_scope"
REAL_PROTEINS = "real_proteins"
FAKE_PROTEINS = "fake_proteins"
CLASS_MAPPING = "class_mapping"
NUM_AMINO_ACIDS = 21
SEQ_LENGTH = "seq_length"


from pathlib import Path

_HMMER_READY = False

def _ensure_hmmer_on_path():
    global _HMMER_READY
    if _HMMER_READY:
        return
    bin_dir = str(Path("~/.conda/envs/tf2_16/bin").expanduser())
    path = os.environ.get("PATH", "")
    parts = path.split(":")
    if bin_dir not in parts:
        os.environ["PATH"] = ":".join([*parts, bin_dir] if path else [bin_dir])
    # Optional: de-dupe in case previous runs appended it repeatedly
    os.environ["PATH"] = ":".join(dict.fromkeys(os.environ["PATH"].split(":")))
    _HMMER_READY = True


def get_file(filename, flags):
    
    embedding_path = os.path.join(flags.data_dir, flags.dataset, filename)

    return np.load(embedding_path)


def extract_regions(seqs: List[str],
                    save_txt: Optional[str] = None,
                    save_csv: Optional[str] = None,
                    save_metadata: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Extracts FR1–FR4 and CDR1–CDR3 regions from VH sequences using ANARCI (IMGT).
    
    Optionally saves:
        - Full sequences to `save_txt`
        - All regions to `save_csv`
        - Metadata (germline, bitscore, etc.) to `save_metadata`
    
    Returns:
        List of dicts: {'FR1': ..., 'CDR1': ..., ..., 'FR4': ..., 'full': ...}
    """
    _ensure_hmmer_on_path()
    seq_tuples = [(f"seq{i}", seq.replace("-", "").strip()) for i, seq in enumerate(seqs)]
    results = anarci(seq_tuples, scheme="imgt", assign_germline=True)
    numbered_hits, metadata_list = results[0], results[1]

    # Region boundaries
    region_ranges = {
        "FR1":   range(1, 27),
        "CDR1":  range(27, 39),
        "FR2":   range(39, 56),
        "CDR2":  range(56, 66),
        "FR3":   range(66, 105),
        "CDR3":  range(105, 118),
        "FR4":   range(118, 129),
    }

    parsed = []
    metadata_out = []

    for i, hit in enumerate(numbered_hits):
        if not hit:
            continue
        alignment, _, _ = hit[0]
        region_map = defaultdict(list)

        for (pos, _), res in alignment:
            for region, valid_range in region_ranges.items():
                if region.startswith("CDR") and res == "-":
                    continue  # Skip gaps only in CDRs
                if pos in valid_range:
                    region_map[region].append(res)
                    break

        if not any(region_map.values()):
            continue

        region_entry = {
            "FR1": "".join(region_map.get("FR1", [])),
            "CDR1": "".join(region_map.get("CDR1", [])),
            "FR2": "".join(region_map.get("FR2", [])),
            "CDR2": "".join(region_map.get("CDR2", [])),
            "FR3": "".join(region_map.get("FR3", [])),
            "CDR3": "".join(region_map.get("CDR3", [])),
            "FR4": "".join(region_map.get("FR4", [])),
        }
        region_entry["full"] = (
            region_entry["FR1"] + region_entry["CDR1"] +
            region_entry["FR2"] + region_entry["CDR2"] +
            region_entry["FR3"] + region_entry["CDR3"] +
            region_entry["FR4"]
        )

        region_entry["missing_regions"] = [
            k for k, v in region_entry.items()
            if (k.startswith("FR") or k.startswith("CDR")) and not v
        ]

        parsed.append(region_entry)
        
        # Metadata
        if save_metadata:
            meta = metadata_list[i]
            meta_row = {
                "query_name": meta.get("query_name"),
                "species": meta.get("species"),
                "chain_type": meta.get("chain_type"),
                "evalue": meta.get("evalue"),
                "bitscore": meta.get("bitscore"),
                "v_gene": meta.get("germlines", {}).get("v_gene", [("", "")])[0][1],
                "j_gene": meta.get("germlines", {}).get("j_gene", [("", "")])[0][1],
            }
            metadata_out.append(meta_row)

    # Deduplicate based on framework only (FR1 + FR2 + FR3 + FR4)
    unique_by_framework = {}
    for entry in parsed:
        key = entry["FR1"] + entry["FR2"] + entry["FR3"] + entry["FR4"]
        if key not in unique_by_framework:
            unique_by_framework[key] = entry
    parsed = list(unique_by_framework.values())
    
    # === Save full sequences (txt) ===
    if save_txt:
        os.makedirs(os.path.dirname(save_txt), exist_ok=True)
        with open(save_txt, "w") as f:
            for entry in parsed:
                f.write(entry["full"] + "\n")

    # === Save regions (csv) ===
    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(parsed[0].keys()))
            writer.writeheader()
            writer.writerows(parsed)

    # === Save metadata (csv) ===
    if save_metadata and metadata_out:
        os.makedirs(os.path.dirname(save_metadata), exist_ok=True)
        with open(save_metadata, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata_out[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_out)

    return parsed
