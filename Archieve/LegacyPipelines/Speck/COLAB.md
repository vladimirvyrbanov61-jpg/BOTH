# Google Colab (Speck 32/64)

**Full step-by-step guide (Simon + Speck):** [`../COLAB_GUIDE.md`](../COLAB_GUIDE.md)

1. Upload or clone this **Speck** folder to Colab (not the Simon project).
2. Install dependencies:

```python
%cd /content/Speck  # adjust path
!pip install -q -r requirements-colab.txt
```

3. Train the autoencoder:

```python
!python ml/train.py --torch-only
```

4. Run cryptanalysis experiments:

```python
!python experiments/run_all.py --quick
```
