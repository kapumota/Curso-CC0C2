# get_data.py -pequeño cargador de datasets de Hugging Face a CSV

import argparse  # Módulo para parsear argumentos desde la línea de comandos
from datasets import load_dataset  # Función para cargar datasets desde Hugging Face
import pandas as pd  # Librería para manipulación de datos en tablas

def main():
    ap = argparse.ArgumentParser()  # Inicializa el parser de argumentos
    ap.add_argument("--hf-id", required=True, help="ID del dataset en Hugging Face")
    ap.add_argument("--subset", default=None, help="Subconjunto o configuración (opcional)")
    ap.add_argument("--split", default="train", help="Nombre del split (por defecto 'train')")
    ap.add_argument("--out", required=True, help="Ruta de salida para el archivo CSV")
    ap.add_argument("--n", type=int, default=0, help="Tamaño de muestra (0=todo)")
    args = ap.parse_args()  # Parsea los argumentos

    if args.subset:
        ds = load_dataset(args.hf_id, args.subset, split=args.split)  # Carga dataset con subconjunto
    else:
        ds = load_dataset(args.hf_id, split=args.split)  # Carga dataset sin subconjunto

    if args.n and len(ds) > args.n:
        ds = ds.select(range(args.n))  # Selecciona una muestra si se especifica un tamaño

    pd.DataFrame(ds).to_csv(args.out, index=False)  # Convierte a DataFrame y guarda como CSV
    print(f"[OK] Guardado {len(ds):,} filas en {args.out}")  # Mensaje de confirmación

if __name__ == "__main__":
    main() 
