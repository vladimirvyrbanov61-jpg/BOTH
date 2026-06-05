"""
SIMON 32/64 utilities for anomaly detection (synthetic oracle + real data ingest).

Built on the primitive core in simon.py.
"""

from simon3264.cipher import Simon3264
from simon3264.encoding import (
    block_to_bits,
    block_to_bytes,
    blocks_to_bits,
    bytes_to_block,
    bytes_to_blocks,
    validate_block,
    validate_blocks,
)
from simon3264.dataset import (
    DatasetConfig,
    generate_labeled_dataset,
    labeled_batch,
    sample_keys,
    sample_plaintexts,
    stratified_split,
)
from simon3264.faults import (
    corrupt_block,
    identity_or_swap,
    random_blocks,
)
from simon3264.features import (
    batch_hw_stats,
    block_stats,
    blocks_to_feature_matrix,
    chi_square_hw_vs_reference,
    sliding_window_xor_features,
)
from simon3264.io import (
    blocks_from_file,
    blocks_to_feature_file,
    decrypt_check,
    load_blocks_npz,
    save_blocks_npz,
)
from simon3264.trace import (
    encrypt_stop_at_round,
    encrypt_trace,
    subkey_bits,
    subkey_summary_stats,
)

__all__ = [
    "Simon3264",
    "block_to_bits",
    "block_to_bytes",
    "blocks_to_bits",
    "bytes_to_block",
    "bytes_to_blocks",
    "validate_block",
    "validate_blocks",
    "DatasetConfig",
    "generate_labeled_dataset",
    "labeled_batch",
    "sample_keys",
    "sample_plaintexts",
    "stratified_split",
    "corrupt_block",
    "identity_or_swap",
    "random_blocks",
    "batch_hw_stats",
    "block_stats",
    "blocks_to_feature_matrix",
    "chi_square_hw_vs_reference",
    "sliding_window_xor_features",
    "blocks_from_file",
    "blocks_to_feature_file",
    "decrypt_check",
    "load_blocks_npz",
    "save_blocks_npz",
    "encrypt_stop_at_round",
    "encrypt_trace",
    "subkey_bits",
    "subkey_summary_stats",
]
