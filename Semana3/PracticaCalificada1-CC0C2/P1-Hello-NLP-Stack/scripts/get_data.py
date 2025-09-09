# get_data.py — tiny HF loader to CSV (comments in English)
import argparse
from datasets import load_dataset
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True, help="Hugging Face dataset id")
    ap.add_argument("--subset", default=None, help="Subset/config (optional)")
    ap.add_argument("--split", default="train", help="Split name")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--n", type=int, default=0, help="Sample size (0=all)")
    args = ap.parse_args()

    if args.subset:
        ds = load_dataset(args.hf_id, args.subset, split=args.split)
    else:
        ds = load_dataset(args.hf_id, split=args.split)

    if args.n and len(ds) > args.n:
        ds = ds.select(range(args.n))

    pd.DataFrame(ds).to_csv(args.out, index=False)
    print(f"[OK] Saved {len(ds):,} rows to {args.out}")

if __name__ == "__main__":
    main()
