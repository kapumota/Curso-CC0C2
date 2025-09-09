# prepare_ud_conllu.py -convierte archivos CoNLL-U a CSV a nivel de token (comentarios en español)
import argparse
import pandas as pd
from conllu import parse_incr
def main():
    ap = argparse.ArgumentParser(description="Convierte un .conllu de UD a un CSV tokenizado.")
    ap.add_argument("--input", required=True, help="Ruta al archivo .conllu de entrada")
    ap.add_argument("--out", required=True, help="Ruta del CSV de salida")
    args = ap.parse_args()
    filas = []
    with open(args.input, "r", encoding="utf-8") as f:
        for sentencia in parse_incr(f):
            sent_id = sentencia.metadata.get("sent_id", "")
            sent_text = sentencia.metadata.get("text", "")
            for tok in sentencia:
                if isinstance(tok["id"], tuple):  # tokens multi-palabra
                    continue
                filas.append({
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
    pd.DataFrame(filas).to_csv(args.out, index=False)
    print(f"[OK] Escribí {len(filas):,} filas de tokens -> {args.out}")
if __name__ == "__main__":
    main()
