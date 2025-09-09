# prepare_ud_conllu.py — converts CoNLL-U to token-level CSV (comments in English)
import argparse
import pandas as pd
from conllu import parse_incr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .conllu file")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            sent_id = sent.metadata.get("sent_id", "")
            sent_text = sent.metadata.get("text", "")
            for tok in sent:
                if isinstance(tok["id"], tuple):  # skip multi-word tokens
                    continue
                rows.append({
                    "sent_id": sent_id,
                    "text": sent_text,
                    "id": tok.get("id"),
                    "form": tok.get("form"),
                    "lemma": tok.get("lemma"),
                    "upos": tok.get("upos"),
                    "xpos": tok.get("xpos"),
                    "feats": tok.get("feats"),
                    "head": tok.get("head"),
                    "deprel": tok.get("deprel"),
                    "deps": tok.get("deps"),
                    "misc": tok.get("misc"),
                })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[OK] Wrote {len(rows):,} tokens → {args.out}")

if __name__ == "__main__":
    main()
