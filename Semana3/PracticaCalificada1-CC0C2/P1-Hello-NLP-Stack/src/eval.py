# eval.py

# Importamos las librerías necesarias
import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

def main():
    # Definimos los argumentos de línea de comandos
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)  # Ruta al directorio con el archivo clean.csv
    ap.add_argument("--model", required=True)  # Ruta al directorio con el modelo entrenado
    ap.add_argument("--out", required=True)  # Ruta donde se guardarán las métricas
    args = ap.parse_args()

    # Creamos el directorio de salida si no existe
    os.makedirs(args.out, exist_ok=True)

    # Leemos el dataset limpio
    df = pd.read_csv(os.path.join(args.data, "clean.csv"))

    # Detectamos automáticamente la columna de texto (buscando 'text' o 'sentence' en el nombre)
    text_col = next((c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()), df.columns[0])

    # Detectamos la columna de etiquetas (label o sentiment)
    label_col = next((c for c in df.columns if "label" in c.lower() or "sentiment" in c.lower()), None)

    # Ruta al modelo entrenado en formato joblib
    model_path = os.path.join(args.model, "model.joblib")

    # Si el modelo existe, realizamos la predicción y cálculo de métricas
    if os.path.exists(model_path):
        clf = joblib.load(model_path)  # Cargamos el modelo
        y_true = df[label_col] if label_col else [0]*len(df)  # Etiquetas verdaderas si existen, si no se asume dummy
        y_pred = clf.predict(df[text_col].astype(str))  # Predicciones del modelo

        # Calculamos métricas si hay etiquetas reales
        acc = accuracy_score(y_true, y_pred) if label_col is not None else None
        f1 = f1_score(y_true, y_pred, average="macro") if label_col is not None else None

        # Guardamos las métricas en un archivo JSON
        metrics = {"accuracy": acc, "f1_macro": f1}
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Si hay etiquetas, graficamos la matriz de confusión
        if label_col is not None:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "cm.png"))  # Guardamos la figura
            plt.close()

        # Mensaje de éxito
        print("[OK] Escribir métricas -> out/metrics.json")

    else:
        # Si no hay modelo entrenado, escribimos un placeholder en el archivo de métricas
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump({"nota": "no hay modelo; provide task-specific metrics"}, f, indent=2)
        print("[WARN] No se encontró el modelo; se escribieron métricas de relleno.")

# Punto de entrada principal del script
if __name__ == "__main__":
    main()
