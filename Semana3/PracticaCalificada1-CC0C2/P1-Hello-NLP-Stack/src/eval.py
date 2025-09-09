# eval.py — evaluation and figures (comments in English)
import argparse, os, json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(os.path.join(args.data, "clean.csv"))
    text_col = next((c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "label" in c.lower() or "sentiment" in c.lower()), None)

    model_path = os.path.join(args.model, "model.joblib")
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
        y_true = df[label_col] if label_col else [0]*len(df)
        y_pred = clf.predict(df[text_col].astype(str))
        acc = accuracy_score(y_true, y_pred) if label_col is not None else None
        f1 = f1_score(y_true, y_pred, average="macro") if label_col is not None else None
        metrics = {"accuracy": acc, "f1_macro": f1}
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        if label_col is not None:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, "cm.png"))
            plt.close()
        print("[OK] Wrote metrics → out/metrics.json")
    else:
        with open(os.path.join(args.out, "metrics.json"), "w") as f:
            json.dump({"note": "no model; provide task-specific metrics"}, f, indent=2)
        print("[WARN] No model found; wrote placeholder metrics.")

if __name__ == "__main__":
    main()
