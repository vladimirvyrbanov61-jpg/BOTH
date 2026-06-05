"""
SPECK 32/64 utilities for anomaly detection and cryptanalysis experiments.
"""

from speck3264.cipher import ALPHA, BETA, ROUNDS, Speck3264
from speck3264.dataset import (
    DatasetConfig,
    generate_labeled_dataset,
    labeled_batch,
    sample_keys,
    sample_plaintexts,
    stratified_split,
)
from speck3264.encoding import (
    block_to_bits,
    block_to_bytes,
    blocks_to_bits,
    bytes_to_block,
    bytes_to_blocks,
    validate_block,
    validate_blocks,
)
from speck3264.faults import corrupt_block, identity_or_swap, random_blocks

__all__ = [
    "ALPHA",
    "BETA",
    "ROUNDS",
    "Speck3264",
    "DatasetConfig",
    "generate_labeled_dataset",
    "labeled_batch",
    "sample_keys",
    "sample_plaintexts",
    "stratified_split",
    "block_to_bits",
    "block_to_bytes",
    "blocks_to_bits",
    "bytes_to_block",
    "bytes_to_blocks",
    "validate_block",
    "validate_blocks",
    "corrupt_block",
    "identity_or_swap",
    "random_blocks",
]
