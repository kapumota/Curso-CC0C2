# prepare_ud_conllu.py - convierte archivos CoNLL-U a CSV a nivel de tokens

import argparse  # Módulo para argumentos desde la línea de comandos
import pandas as pd  # Librería para manipulación tabular
from conllu import parse_incr  # Función para parsear archivos CoNLL-U incrementalmente

def main():
    ap = argparse.ArgumentParser()  # Inicializa el parser de argumentos
    ap.add_argument("--input", required=True, help="Ruta al archivo .conllu de entrada")
    ap.add_argument("--out", required=True, help="Ruta del archivo CSV de salida")
    args = ap.parse_args()

    rows = []  # Lista que almacenará los datos de cada token
    with open(args.input, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):  # Itera sobre cada oración del archivo CoNLL-U
            sent_id = sent.metadata.get("sent_id", "")  # Obtiene el ID de la oración (si está disponible)
            sent_text = sent.metadata.get("text", "")   # Obtiene el texto original de la oración
            for tok in sent:
                if isinstance(tok["id"], tuple):  # Omite tokens de múltiples palabras
                    continue
                # Extrae la información del token y la agrega a la lista
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
    pd.DataFrame(rows).to_csv(args.out, index=False)  # Convierte a DataFrame y guarda como CSV
    print(f"[OK] Escribe {len(rows):,} tokens -> {args.out}")  # Imprime confirmación con cantidad de tokens procesados

if __name__ == "__main__":
    main() 
