import sys
sys.path.insert(0, r"D:\BOTH\Simon")
import numpy as np
from simon import Simon

# inmcm-style test for 32/64
pt = np.array([0x6565, 0x6877], dtype=np.uint16)
key = np.array([0x1918, 0x1110, 0x0908, 0x0100], dtype=np.uint16)
ct = Simon(16,4).encrypt(pt, key)
print("32/64", ct[0].tolist(), ct[0].tolist() == [0xc69b, 0xe9bb])

# 64/96
pt = np.array([0x6e696c63, 0x6f722067], dtype=np.uint32)
key = np.array([0x13121110, 0x0b0a0908, 0x03020100], dtype=np.uint32)
ct = Simon(32,3).encrypt(pt, key)
exp = np.array([0x111a8fc8, 0x5ca2e27f], dtype=np.uint32)
print("64/96", ct[0].tolist(), ct[0].tolist() == exp.tolist())
