#!/usr/bin/env python3
"""Print SIMON 32/64 profile facts and run quick cipher self-tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simon import Simon, _SIMON_PARAMS
from simon3264.cipher import (
    BLOCK_BITS,
    KEY_BITS,
    M_WORDS,
    N_BITS,
    ROUNDS,
    Z_INDEX,
    Simon3264,
)


def main() -> int:
    c = Simon3264()
    print("=== SIMON 32/64 profile (simon3264) ===")
    print(f"  Block size     : {BLOCK_BITS} bits  (two {N_BITS}-bit words)")
    print(f"  Key size       : {KEY_BITS} bits  ({M_WORDS} key words)")
    print(f"  Full rounds T  : {ROUNDS}  (z_index={Z_INDEX})")
    print(f"  Paper (n,m)    : ({N_BITS}, {M_WORDS}) -> {_SIMON_PARAMS[(N_BITS, M_WORDS)]}")
    print(f"  Instance       : {c!r}")

    # Official KAT (Beaulieu et al. Appendix B)
    s = Simon(N_BITS, M_WORDS)
    pt = np.array([[0x6565, 0x6877]], dtype=np.uint16)
    key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
    ct = s.encrypt(pt, key)[0]
    ok_kat = int(ct[0]) == 0xC69B and int(ct[1]) == 0xE9BB
    print(f"\n  Appendix B KAT : {'PASS' if ok_kat else 'FAIL'}  ct={[hex(int(x)) for x in ct]}")

    # Round range
    pt2 = np.array([[0x1234, 0x5678]], dtype=np.uint16)
    key2 = np.array([0x0100, 0x0200, 0x0300, 0x0400], dtype=np.uint16)
    ct32 = c.encrypt(pt2, key2)[0]
    ct8 = c.encrypt_rounds(pt2, key2, 8)[0]
    sk = c.get_subkeys(key2)
    print(f"\n  Subkeys shape  : {sk.shape}  (expect (1, {ROUNDS}))")
    print(f"  8-round != 32  : {not np.array_equal(ct8, ct32)}")

    # encrypt_rounds capped at ROUNDS; core Simon() can exceed for experiments
    s40 = Simon(N_BITS, M_WORDS, rounds=40)
    print(f"  Simon(n,m,rounds=40) allowed rounds: {s40.rounds}  (research override)")

    # Algebraic inverse at full rounds
    pt_r = np.random.default_rng(0).integers(0, 0x10000, size=(100, 2), dtype=np.uint16)
    key_r = np.random.default_rng(1).integers(0, 0x10000, size=(100, 4), dtype=np.uint16)
    ct_r = c.encrypt(pt_r, key_r)
    pt_back = c.decrypt(ct_r, key_r)
    ok_rt = np.array_equal(pt_r, pt_back)
    print(f"  Dec(Enc(P))==P  : {'PASS' if ok_rt else 'FAIL'}  (100 random blocks)")

    print("\n=== Round policy ===")
    print(f"  simon3264.encrypt uses exactly {ROUNDS} rounds (spec default).")
    print(f"  encrypt_rounds(r) accepts r in 1..{ROUNDS}.")
    print("  Wrong-round faults in ML data use encrypt_rounds with r < T.")

    return 0 if ok_kat and ok_rt else 1


if __name__ == "__main__":
    raise SystemExit(main())
