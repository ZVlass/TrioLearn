from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "backend" / "data" / "topics" / "all_topic_vectors.parquet"  # your path in the log
META_CANDIDATES = [
    ROOT / "data" / "interim" / "courses_metadata.csv",
    ROOT / "data" / "interim" / "books_metadata.csv",
    ROOT / "data" / "interim" / "ml_videos_metadata.csv",
    ROOT / "backend" / "data" / "interim" / "courses_metadata.csv",
    ROOT / "backend" / "data" / "interim" / "books_metadata.csv",
    ROOT / "backend" / "data" / "interim" / "ml_videos_metadata.csv",
]

df = pd.read_parquet(PARQUET)
print("[parquet] columns:", list(df.columns))

for p in META_CANDIDATES:
    if not p.exists(): 
        continue
    meta = pd.read_csv(p, low_memory=False, nrows=200)  # sample header
    commons = set(df.columns) & set(meta.columns)
    print(f"[meta] {p.name} columns:", list(meta.columns)[:20])
    print(f"   ∩ common keys with parquet:", commons & {"global_id","external_id","id","item_id"})
