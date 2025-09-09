# preprocess.py - limpieza/preparación de datos 

import argparse, os, pandas as pd, glob  # Importación de módulos necesarios

def main():
    ap = argparse.ArgumentParser()  # Inicializa el analizador de argumentos
    ap.add_argument("--in", dest="indir", required=True, help="Directorio de entrada (data/raw)")
    ap.add_argument("--out", dest="outdir", required=True, help="Directorio de salida (data/processed)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)  # Crea el directorio de salida si no existe

    files = sorted(glob.glob(os.path.join(args.indir, "*.csv")))  # Busca archivos CSV en el directorio de entrada
    if not files:
        raise SystemExit("No se encontraron archivos CSV en data/raw/. ¿Ejecutaste `make data`?")  # Termina si no hay CSVs

    df = pd.read_csv(files[0]).dropna().reset_index(drop=True)  # Lee el primer CSV, elimina valores nulos y reinicia índices
    out_path = os.path.join(args.outdir, "clean.csv")  # Define la ruta del archivo limpio
    df.to_csv(out_path, index=False)  # Guarda el DataFrame limpio sin índices
    print(f"[OK] Escribio el CSV limpio -> {out_path} (#rows={len(df)})")  # Imprime mensaje con confirmación

if __name__ == "__main__":
    main()  
