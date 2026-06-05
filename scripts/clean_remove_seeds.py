"""Clean multi-seed CSVs by removing rows for specific seeds and back up originals."""
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime

SEEDS_TO_REMOVE = {42, 100, 101}
BASE = Path('results') / 'thesis'
TS = datetime.now().strftime('%Y%m%d%H%M%S')
BACKUP_DIR = BASE / f'backup_{TS}'
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

print('Backup dir:', BACKUP_DIR)
for fp in BASE.glob('*_multi_seed_raw.csv'):
    print('Processing', fp)
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print('  ERROR reading', fp, e)
        continue
    if 'seed' not in df.columns:
        print('  No seed column, skipping', fp)
        continue
    before = len(df)
    df2 = df[~df['seed'].isin(list(SEEDS_TO_REMOVE))]
    after = len(df2)
    if before == after:
        print(f'  No rows to remove for {fp.name}')
    else:
        bak = BACKUP_DIR / fp.name
        shutil.copy(fp, bak)
        df2.to_csv(fp, index=False)
        print(f'  Cleaned {fp.name}: {before}->{after}, backup={bak}')
print('Done')
