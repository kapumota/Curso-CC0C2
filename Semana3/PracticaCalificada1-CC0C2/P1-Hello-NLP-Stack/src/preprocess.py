# preprocess.py — data cleaning/featurization (comments in English)
import argparse, os, pandas as pd, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Input directory (data/raw)")
    ap.add_argument("--out", dest="outdir", required=True, help="Output directory (data/processed)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.indir, "*.csv")))
    if not files:
        raise SystemExit("No CSV files found in data/raw/. Did you run `make data`?")
    df = pd.read_csv(files[0]).dropna().reset_index(drop=True)
    out_path = os.path.join(args.outdir, "clean.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote cleaned CSV → {out_path} (#rows={len(df)})")

if __name__ == "__main__":
    main()
