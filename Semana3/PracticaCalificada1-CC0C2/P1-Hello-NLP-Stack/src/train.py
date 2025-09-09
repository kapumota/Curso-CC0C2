# train.py - entrenamiento base

import argparse, os, json, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def main():
    ap = argparse.ArgumentParser()  # Parser para argumentos de línea de comandos
    ap.add_argument("--data", required=True, help="Directorio con los datos procesados")
    ap.add_argument("--out", required=True, help="Directorio de salida")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)  # Crea el directorio de salida si no existe

    df = pd.read_csv(os.path.join(args.data, "clean.csv"))  # Carga el archivo clean.csv
    text_col = next((c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()), df.columns[0])  # Detecta la columna de texto
    label_col = next((c for c in df.columns if "label" in c.lower() or "sentiment" in c.lower()), None)  # Detecta la columna de etiqueta
    if label_col is None:
        df["label"] = 0  # Si no existe columna de etiquetas, se crea una por defecto
        label_col = "label"

    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000)),
                     ("clf", LogisticRegression(max_iter=200))])  # Pipeline: vectorizador TF-IDF + regresión logística
    pipe.fit(df[text_col].astype(str), df[label_col])  # Entrenamiento del modelo
    joblib.dump(pipe, os.path.join(args.out, "model.joblib"))  # Guarda el modelo entrenado
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump({"rows": int(len(df))}, f)  # Guarda metadatos del entrenamiento
    print("[OK] Modelo base entrenado -> out/model.joblib")  # Mensaje de confirmación

if __name__ == "__main__":
    main()  
