import sys
sys.path.insert(0, r"D:\BOTH\Simon")
import numpy as np
from simon import Simon
from test_kat_vectors import kat_arrays, SIMON_OFFICIAL_KATS

for n,m,p,c,k in SIMON_OFFICIAL_KATS:
    if (n,m) != (32,3):
        continue
    pt, ct_exp, key = kat_arrays(n,p,c,k)
    c = Simon(n=n,m=m)
    ct = c.encrypt(pt, key)
    print("pt", [hex(x) for x in pt])
    print("key", [hex(x) for x in key])
    print("got", [hex(x) for x in ct[0]])
    print("exp", [hex(x) for x in ct_exp])
    print("match", np.array_equal(ct[0], ct_exp))
    print("swap match", np.array_equal(ct[0], ct_exp[::-1]))

key2 = np.array([0x13121110, 0x0b0a0908, 0x03020100], dtype=np.uint32)
pt2 = np.array([0x6e696c63, 0x6f722067], dtype=np.uint32)
ct2 = Simon(32,3).encrypt(pt2, key2)
print("paper style key match", np.array_equal(ct2[0], np.array([0x111a8fc8, 0x5ca2e27f], dtype=np.uint32)))
