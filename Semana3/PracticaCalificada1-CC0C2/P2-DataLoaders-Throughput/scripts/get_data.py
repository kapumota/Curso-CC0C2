# get_data.py — cargador mínimo de datasets de Hugging Face a CSV (comentarios en español)
import argparse
from datasets import load_dataset
import pandas as pd
def main():
    ap = argparse.ArgumentParser(description="Descarga un dataset de HF y lo guarda como CSV.")
    ap.add_argument("--hf-id", required=True, help="ID del dataset en Hugging Face (ej.: mteb/tweet_sentiment_multilingual)")
    ap.add_argument("--subset", default=None, help="Subconjunto/config del dataset (opcional)")
    ap.add_argument("--split", default="train", help="Split a cargar (train/validation/test)")
    ap.add_argument("--out", required=True, help="Ruta del CSV de salida")
    ap.add_argument("--n", type=int, default=0, help="Tamaño de muestra (0 = todo)")
    args = ap.parse_args()
    if args.subset:
        ds = load_dataset(args.hf_id, args.subset, split=args.split)
    else:
        ds = load_dataset(args.hf_id, split=args.split)
    if args.n and len(ds) > args.n:
        ds = ds.select(range(args.n))
    pd.DataFrame(ds).to_csv(args.out, index=False)
    print(f"[OK] Guardadas {len(ds):,} filas en {args.out}")
if __name__ == "__main__":
    main()
