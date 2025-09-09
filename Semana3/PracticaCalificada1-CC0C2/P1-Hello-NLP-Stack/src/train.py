# train.py — baseline training (comments in English)
import argparse, os, json, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Processed data directory")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(os.path.join(args.data, "clean.csv"))
    text_col = next((c for c in df.columns if "text" in c.lower() or "sentence" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "label" in c.lower() or "sentiment" in c.lower()), None)
    if label_col is None:
        df["label"] = 0
        label_col = "label"

    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000)),
                     ("clf", LogisticRegression(max_iter=200))])
    pipe.fit(df[text_col].astype(str), df[label_col])
    joblib.dump(pipe, os.path.join(args.out, "model.joblib"))
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump({"rows": int(len(df))}, f)
    print("[OK] Trained baseline model → out/model.joblib")

if __name__ == "__main__":
    main()
