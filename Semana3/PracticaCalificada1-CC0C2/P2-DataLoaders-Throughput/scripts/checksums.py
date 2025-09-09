# checksums.py - manifiesto SHA256 
import argparse, hashlib, os
def sha256_de(path, chunk=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()
def main():
    ap = argparse.ArgumentParser(description="Genera hashes SHA256 de los datos.")
    ap.add_argument("--root", default="data", help="Directorio raíz a recorrer")
    ap.add_argument("--out", default="evidencias/data_sha256.txt", help="Archivo de salida del manifiesto")
    args = ap.parse_args()
    registros = []
    for r, _, files in os.walk(args.root):
        for fn in files:
            if fn.endswith((".csv", ".conllu", ".jsonl", ".txt")):
                p = os.path.join(r, fn)
                registros.append((p, sha256_de(p)))
    registros.sort()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p, h in registros:
            f.write(f"{h}  {p}\n")
    print(f"[OK] {len(registros)} archivos hasheados -> {args.out}")
if __name__ == "__main__":
    main()
