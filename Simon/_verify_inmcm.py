import urllib.request
import sys
url = "https://raw.githubusercontent.com/inmcm/Simon_Speck_Ciphers/master/Python/simonspeckciphers/simon/simon.py"
code = urllib.request.urlopen(url).read().decode()
ns = {}
exec(code, ns)
SimonCipher = ns['SimonCipher']

# 32/64 block_size=32 key_size=64
key = 0x1918111009080100
w = SimonCipher(key, key_size=64, block_size=32)
t = w.encrypt(0x65656877)
print("inmcm 32/64", hex(t))

# 64/96 block_size=64 key_size=96  
key2 = 0x131211100b0a090803020100
w2 = SimonCipher(key2, key_size=96, block_size=64)
t2 = w2.encrypt(0x6f7220676e696c63)
print("inmcm 64/96", hex(t2), "exp", hex(0x111a8fc85ca2e27f))

sys.path.insert(0, r"D:\BOTH\Simon")
import numpy as np
from simon import Simon
pt = np.array([0x6e696c63, 0x6f722067], dtype=np.uint32)
key = np.array([0x13121110, 0x0b0a0908, 0x03020100], dtype=np.uint32)
ct = Simon(32,3).encrypt(pt, key)
print("ours 64/96", [hex(x) for x in ct[0]])
