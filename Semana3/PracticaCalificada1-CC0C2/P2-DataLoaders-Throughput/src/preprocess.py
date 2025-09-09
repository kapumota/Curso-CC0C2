# preprocess.py — limpieza y featurización (comentarios en español)
import argparse, os, glob
import pandas as pd
def main():
    ap = argparse.ArgumentParser(description="Preprocesa CSV(s) de data/raw y genera un CSV limpio.")
    ap.add_argument("--in", dest="indir", required=True, help="Directorio de entrada (data/raw)")
    ap.add_argument("--out", dest="outdir", required=True, help="Directorio de salida (data/processed)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    archivos = sorted(glob.glob(os.path.join(args.indir, "*.csv")))
    if not archivos:
        raise SystemExit("No se encontraron CSV en data/raw/. ¿Ejecutaste `make data`?")
    df = pd.read_csv(archivos[0]).dropna().reset_index(drop=True)
    out_path = os.path.join(args.outdir, "clean.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] CSV limpio → {out_path} (filas={len(df)})")
if __name__ == "__main__":
    main()
